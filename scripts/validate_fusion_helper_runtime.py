"""Compare actual model/engine traces with an isolated fusion script variant."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

from run_counterfactual_search import SnapshotModel
import ygoai.utils as runtime_utils
from ygoai.rl.replay_search import ReplaySearchBackend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--deck-dir',type=Path,required=True)
    p.add_argument('--code-list',type=Path,required=True)
    p.add_argument('--semantics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--games',type=int,default=12)
    args=p.parse_args()
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=args.checkpoint_sha256:
        raise ValueError('Checkpoint hash mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    # Test-process-only override: production utils and script assets are untouched.
    runtime_utils.get_root_directory=lambda: str(args.runtime_root.resolve())
    model_args=SimpleNamespace(deck=str(args.deck_dir),code_list_file=str(args.code_list),seed=42,
        checkpoint_a=str(args.checkpoint),checkpoint_b=str(args.checkpoint),
        checkpoint_a_schema='structured-lite-v1',checkpoint_b_schema='structured-lite-v1',
        checkpoint_a_variant='full',checkpoint_b_variant='full',player_a=0,
        deck1=None,deck2=None,max_options=128,n_history_actions=32,verbose=False,
        observation_schema='structured-lite-v1',semantic_asset_dir=str(args.semantics),
        n_public_events=32,max_group_references=8)
    row={'player':0,'snapshot':{'seed':71900845,'actions':[],'play_mode':'self',
         'deck1':'elfnote','deck2':'deck-0db5f537194f9bfc'}}
    model=SnapshotModel(row,model_args)
    reports=[]
    for game in range(args.games):
        seed=71900845+game//2
        row['snapshot'].update(seed=seed,
            deck1='elfnote' if game%2==0 else 'deck-0db5f537194f9bfc',
            deck2='deck-0db5f537194f9bfc' if game%2==0 else 'elfnote')
        backend=ReplaySearchBackend(model)
        start=time.monotonic(); digest=hashlib.sha256(); rng=random.Random(seed)
        try:
            with (args.output/f'game-{game}.jsonl').open('x') as stream, backend.branch(0,seed) as branch:
                for step in range(1001):
                    state=branch.evaluate()
                    entry={'step':step,'player':state.player,'digest':state.observation_digest,
                           'menu':state.legal_actions,'logits':state.logits,'value':state.value,
                           'terminal':state.terminal_root_return}
                    if state.terminal_root_return is None:
                        if step==1000: raise RuntimeError('Runtime fixture reached step cap')
                        best=max(state.logits)
                        if game<6:
                            action=max(state.legal_actions,key=lambda a:state.logits[a])
                        else:
                            import math
                            action=rng.choices(state.legal_actions,
                                weights=[math.exp(x-best) for x in state.logits])[0]
                        entry['action']=action
                    encoded=json.dumps(entry,sort_keys=True,allow_nan=False)
                    digest.update(encoded.encode()); stream.write(encoded+'\n'); stream.flush()
                    if state.terminal_root_return is not None: break
                    branch.step(action)
                reports.append({'game':game,'seed':seed,'seat':game%2,'steps':step,
                                'trace_sha256':digest.hexdigest(),
                                'terminal_root_return':state.terminal_root_return,
                                'seconds':time.monotonic()-start})
        finally:
            backend.close()
        print(json.dumps(reports[-1]),flush=True)
        (args.output/'report.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__': main()
