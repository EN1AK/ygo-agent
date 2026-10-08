"""Run complete normal-opening Laya duels, optional oracle search and direct RL updates."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from scripts.train_jev_direct_rl import LayaPolicy, sha, train_group
from ygoai.rl.jev_duel import Worker, play


def verify_replay(config, path):
    audit=json.loads((path/'audit.json').read_text())
    records=[json.loads(line) for line in (path/'decisions.jsonl').read_text().splitlines()]
    commits=[r for r in records if not r.get('terminal') and r['operation'][0]=='commit']
    worker=Worker(config,audit['initial'],path/'replay.log')
    try:
        for row in commits:
            if worker.current['fingerprint']!=row['root_fingerprint']:
                raise ValueError('replay pre-action divergence')
            worker.step(row['operation'][1])
            if worker.current['fingerprint']!=row['next_fingerprint']:
                raise ValueError('replay post-action divergence')
        actual=worker.call('audit')
        for key in ('responses','frames','fields','core_seed'):
            if actual[key]!=audit[key]: raise ValueError('replay mismatch '+key)
        result=dict(passed=True,model_commits=len(commits),core_responses=len(actual['responses']),
                    events=len(actual['frames']),natural_terminal=worker.current['done'] and
                    not worker.current['invalid'] and worker.current['termination_reason']==1)
        if not result['natural_terminal']: raise ValueError('replay not natural terminal')
        (path/'replay-verification.json').write_text(json.dumps(result,indent=2))
        return result
    finally: worker.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','native','database','scripts','code-list','semantics','deck1','deck2','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--init-weights',type=Path)
    p.add_argument('--seed',type=int,default=8102026)
    p.add_argument('--duels',type=int,default=1)
    p.add_argument('--device',default='cuda')
    p.add_argument('--greedy',action='store_true')
    p.add_argument('--max-decisions',type=int,default=1500)
    p.add_argument('--max-len',type=int,default=8192)
    p.add_argument('--search-budget',type=int,default=0)
    p.add_argument('--search-mode',choices=('off','oracle'),default='off')
    p.add_argument('--horizon',type=int,default=4)
    p.add_argument('--updates',type=int,default=0)
    p.add_argument('--group-size',type=int,default=2)
    p.add_argument('--learning-rate',type=float,default=1e-6)
    a=p.parse_args()
    if a.duels<1 or a.updates<0 or a.group_size<2 or a.max_decisions<1 or not 0<a.max_len<=8192 or a.horizon<1 or a.search_budget<0:
        p.error('invalid budget')
    if a.search_budget and a.search_mode!='oracle': p.error('exact replay search must explicitly use --search-mode oracle')
    if a.updates and a.greedy: p.error('on-policy RL requires sampled decisions')
    if a.updates and a.search_budget: p.error('full-duel RL with search continuation credit assignment is not enabled yet; evaluate search with --updates 0')
    for k,v in vars(a).items():
        if isinstance(v,Path): setattr(a,k,v.resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    config={k:str(getattr(a,k)) for k in ('native','database','scripts','code_list','semantics','deck1','deck2')}
    config['seed']=a.seed
    metadata={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    metadata['asset_hashes']={k:sha(Path(v)) for k,v in config.items() if k not in ('scripts','semantics','seed')}
    metadata['semantics_metadata_sha256']=sha(a.semantics/'metadata.json')
    metadata['card_scripts_sha256']=hashlib.sha256(''.join(str(f.relative_to(a.scripts))+sha(f) for f in sorted(a.scripts.rglob('*.lua'))).encode()).hexdigest()
    metadata['input_model_sha256']=sha(a.init_weights or a.model_dir/'model.safetensors')
    metadata['source_sha256']={str(f):sha(f) for f in (Path(__file__),Path('scripts/jev_duel_worker.py'),Path('ygoai/rl/jev_duel.py'),Path('ygoai/rl/jev_duel_observation.py'))}
    (a.output/'config.json').write_text(json.dumps(metadata,indent=2))
    import torch
    torch.set_num_threads(2); torch.manual_seed(a.seed)
    policy=LayaPolicy(a.model_dir,a.device,a.max_len,a.init_weights)
    start=time.monotonic(); results=[]
    try:
        for n in range(a.duels):
            cfg=dict(config,seed=a.seed+n)
            path=a.output/f'duel-{n}'
            _,summary=play(policy,cfg,path,greedy=a.greedy,max_decisions=a.max_decisions,
                           search_budget=a.search_budget,horizon=a.horizon)
            summary['replay']=verify_replay(cfg,path)
            results.append(summary); print(json.dumps(summary),flush=True)
        if a.updates:
            optimizer=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=a.learning_rate,weight_decay=0.)
            metrics=[]
            for update in range(a.updates):
                episodes=[]
                for n in range(a.group_size):
                    path=a.output/f'train-{update}-{n}'
                    e,_=play(policy,config,path,max_decisions=a.max_decisions)
                    verify_replay(config,path); episodes.append(e)
                metrics.append(train_group(policy,optimizer,episodes))
                print(json.dumps(dict(update=update,**metrics[-1])),flush=True)
            from safetensors.torch import save_file
            save_file({n:v.detach().cpu().contiguous() for n,v in policy.model.state_dict().items()},str(a.output/'model.safetensors'))
            torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),a.output/'optimizer.pt')
        else: metrics=[]
        final=dict(success=True,duels=results,updates=metrics,elapsed_seconds=time.monotonic()-start,
                   full_duel_policy=True,strength_claim=False)
        if a.updates: final['output_weight_sha256']=sha(a.output/'model.safetensors')
        (a.output/'summary.json').write_text(json.dumps(final,indent=2))
    except Exception as exc:
        (a.output/'failure.json').write_text(json.dumps(dict(error=repr(exc),elapsed_seconds=time.monotonic()-start)))
        raise


if __name__=='__main__':
    main()
