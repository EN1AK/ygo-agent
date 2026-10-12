"""Profile and train the existing duel policy with parallel actors and exact-batch replay."""
import argparse
import concurrent.futures
import json
import time
from collections import Counter
from pathlib import Path

from scripts.run_jev_duel import verify_replay
from scripts.train_jev_direct_rl import sha
from ygoai.rl.jev_parallel import collect,train_tape


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True,help='Existing run config with fixed model and simulator assets')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--actors',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--episodes',type=int,default=4)
    p.add_argument('--updates',type=int,default=1)
    p.add_argument('--batch-wait-ms',type=float,default=3.)
    p.add_argument('--baseline',choices=('zero','leave-one-out'),default='zero')
    p.add_argument('--no-checkpoint',action='store_true',help='Disable activation recomputation; uses more VRAM')
    p.add_argument('--vary-seeds',action='store_true')
    p.add_argument('--init-weights',type=Path)
    a=p.parse_args()
    if min(a.actors,a.batch_size,a.updates)<1 or a.episodes<2 or a.batch_wait_ms<0: p.error('invalid parallel budget')
    a.output=a.output.resolve(); a.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads(a.config.read_text())
    cfg={k:meta[k] for k in ('native','database','scripts','code_list','semantics','deck1','deck2','seed','recent_events')}
    metadata=dict(source_config=meta,source_config_sha256=sha(a.config),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  source_sha256={f:sha(f) for f in ('scripts/run_jev_parallel.py','ygoai/rl/jev_parallel.py','ygoai/rl/jev_qwen_policy.py','ygoai/rl/jev_duel.py')})
    if a.init_weights: metadata['adapter_sha256']=sha(a.init_weights)
    (a.output/'config.json').write_text(json.dumps(metadata,indent=2))
    import torch
    from ygoai.rl.jev_qwen_policy import OpenJevQwenPolicy
    torch.set_num_threads(2); torch.manual_seed(meta['seed'])
    policy=OpenJevQwenPolicy(meta['model_dir'],'cuda',meta['max_len'],a.init_weights)
    if a.no_checkpoint: policy.model.gradient_checkpointing_disable()
    optimizer=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=meta['learning_rate'],weight_decay=0.)
    start=time.perf_counter(); updates=[]
    try:
        for update in range(a.updates):
            configs=[dict(cfg,seed=cfg['seed']+update*a.episodes+i if a.vary_seeds else cfg['seed']) for i in range(a.episodes)]
            paths=[a.output/f'group-{update}-duel-{i}' for i in range(a.episodes)]
            torch.cuda.reset_peak_memory_stats()
            results,tape,timing=collect(policy,configs,paths,actors=a.actors,batch_size=a.batch_size,batch_wait_ms=a.batch_wait_ms,
                seed=meta['seed']+10000+update*a.episodes,search_budget=meta['search_budget'],horizon=meta['horizon'],max_decisions=meta['max_decisions'])
            timing['batch_size_histogram']=dict(Counter(timing.pop('batch_sizes')))
            (a.output/f'batches-{update}.json').write_text(json.dumps([[r['sample_id'] for r in rows] for rows in tape]))
            print(json.dumps(dict(stage='collected',update=update,**timing)),flush=True)
            replay_start=time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.actors) as pool:
                replays=list(pool.map(lambda pair:verify_replay(*pair),zip(configs,paths)))
            replay_seconds=time.perf_counter()-replay_start
            episodes=[r[0] for r in results]
            training=train_tape(policy,optimizer,episodes,tape,microbatch=a.batch_size,baseline_mode=a.baseline)
            record=dict(update=update,collection=timing,replay_seconds=replay_seconds,training=training,
                        episode_returns=[e['total'] for e in episodes],duels=[r[1] for r in results],replays=replays,
                        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
            updates.append(record); print(json.dumps(dict(stage='updated',**record)),flush=True)
            (a.output/'updates.json').write_text(json.dumps(updates,indent=2))
        policy.save(a.output/'adapter.safetensors')
        torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),a.output/'optimizer.pt')
        summary=dict(success=True,updates=updates,elapsed_seconds=time.perf_counter()-start,
                     parameter_update_verified=all(r['training']['parameter_max_abs_delta']>0 for r in updates),
                     output_weight_sha256=sha(a.output/'adapter.safetensors'),device=torch.cuda.get_device_name(),strength_claim=False)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(dict(stage='complete',elapsed_seconds=summary['elapsed_seconds'],parameter_update_verified=summary['parameter_update_verified'])),flush=True)
    except BaseException as exc:
        (a.output/'failure.json').write_text(json.dumps(dict(error=repr(exc),elapsed_seconds=time.perf_counter()-start)))
        raise


if __name__=='__main__': main()
