"""Capture live-engine oracle fixtures and repeat candidate rollouts exactly.

This is a smoke/differential gate, not a fair-play strength test. Chain-budget
fallbacks are valid smoke outcomes and are reported separately from completed
candidate comparisons. It does not replace historical tactical fixtures.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

# Import the sibling before its bootstrap removes the scripts directory.
from run_counterfactual_search import SnapshotModel
from ygoai.rl.candidate_search import SearchBudget, candidate_search
from ygoai.rl.replay_search import ReplaySearchBackend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--native-sha256',required=True)
    parser.add_argument('--deck',type=Path,required=True)
    parser.add_argument('--code-list-file',type=Path,required=True)
    parser.add_argument('--semantic-assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=71900845)
    parser.add_argument('--capture-steps',type=int,default=120)
    parser.add_argument('--depth',type=int,default=8)
    parser.add_argument('--wall-seconds',type=float,default=120.)
    cli=parser.parse_args()
    import ygoenv.ygopro.ygopro_ygoenv as native_module
    if hashlib.sha256(cli.checkpoint.read_bytes()).hexdigest()!=cli.checkpoint_sha256:
        raise ValueError('Checkpoint hash mismatch')
    if hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest()!=cli.native_sha256:
        raise ValueError('Native hash mismatch')
    if cli.capture_steps<1: parser.error('Positive capture limit required')
    cli.output.mkdir(parents=True,exist_ok=False)
    args=SimpleNamespace(deck=str(cli.deck),code_list_file=str(cli.code_list_file),seed=1,
        checkpoint_a_schema='structured-lite-v1',checkpoint_b_schema='structured-lite-v1',
        checkpoint_a_variant='full',checkpoint_b_variant='full',checkpoint_a=str(cli.checkpoint),
        checkpoint_b=str(cli.checkpoint),player_a=0,deck1=cli.deck.stem,deck2=cli.deck.stem,
        max_options=128,n_history_actions=32,verbose=False,observation_schema='structured-lite-v1',
        semantic_asset_dir=str(cli.semantic_assets),n_public_events=32,max_group_references=8)
    base={'schema':'ygo-decision-fixture-v1','player':0,'decision_id':'initial',
          'snapshot':{'seed':cli.seed,'actions':[],'play_mode':'self',
            'deck1':cli.deck.stem,'deck2':cli.deck.stem,'max_options':128,'n_history_actions':32,
            'observation_schema':'structured-lite-v1'}}
    model=SnapshotModel(base,args)
    backend=ReplaySearchBackend(model)
    fixtures={}
    prefix=[]
    try:
        with backend.branch(0,1) as branch:
            for step in range(cli.capture_steps):
                root=branch.evaluate()
                if root.terminal_root_return is not None: break
                action=root.legal_actions[max(range(len(root.logits)),key=lambda i:root.logits[i])]
                if len(root.legal_actions)>=2:
                    name=None
                    if not fixtures: name='opening'
                    elif root.prompt=='select_chain' and 'chain' not in fixtures: name='chain'
                    elif root.prompt in ('prompt-3','prompt-9','prompt-12') and 'staged' not in fixtures: name='staged'
                    elif len(prefix)>=40 and 'long-prefix' not in fixtures: name='long-prefix'
                    if name:
                        row=json.loads(json.dumps(base))
                        row.update(player=root.player,decision_id=f'{name}:{step}',
                            observation_digest=root.observation_digest,legal_actions=list(root.legal_actions),
                            policy_logits=list(root.logits),state_value=root.value,selected_action=action)
                        row['snapshot']['actions']=list(prefix)
                        fixtures[name]=row
                branch.step(action)
                prefix.append(action)
                if len(fixtures)==4: break
    finally: backend.close()
    reports=[]
    for name,row in fixtures.items():
        (cli.output/f'{name}.jsonl').write_text(json.dumps(row)+'\n')
        model.row=row
        model.root_player=row['player']
        backend=ReplaySearchBackend(model)
        try:
            with backend.branch(0,1) as branch: root=branch.evaluate()
            if (root.observation_digest!=row['observation_digest'] or list(root.logits)!=row['policy_logits']
                    or root.player!=row['player'] or list(root.legal_actions)!=row['legal_actions']):
                raise RuntimeError('Root replay identity mismatch')
            budget=SearchBudget(candidates=2,rollouts_per_action=2,depth=cli.depth,wall_seconds=cli.wall_seconds)
            result=candidate_search(root,backend,budget,seed=42)
            again=candidate_search(root,backend,budget,seed=42)
            for key in ('completed','fallback_reason','rollouts','updated_policy','selected_action'):
                if result[key]!=again[key]: raise RuntimeError(f'Nondeterministic search field: {key}')
            if backend.active: raise RuntimeError('Leaked replay branch')
        finally: backend.close()
        result.update(checkpoint_sha256=cli.checkpoint_sha256,native_sha256=cli.native_sha256)
        (cli.output/f'{name}-candidate.json').write_text(json.dumps(result,indent=2)+'\n')
        reports.append({'fixture':name,'comparison_complete':result['completed'],
                        'fallback':result['fallback_reason'],'seconds':result['elapsed_seconds']})
    report={'schema':'ygo-candidate-replay-smoke-v1','fixtures':reports,
            'missing_fixtures':sorted({'opening','chain','staged','long-prefix'}-fixtures.keys()),
            'checkpoint_sha256':cli.checkpoint_sha256,'native_sha256':cli.native_sha256,
            'exact_repetition_passed':bool(fixtures),'fair_play_promoted':False}
    (cli.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not fixtures: raise RuntimeError('No live decisions captured')
    (cli.output/'completed.txt').write_text('oracle smoke only; inspect missing fixtures and fallbacks\n')


if __name__=='__main__': main()
