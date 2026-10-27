"""Reconstruct full-menu model inputs for terminal or partial human replays.

No FTK/winner gate. A valid partial recording is not given a terminal label.
This produces legal observational data, not a claim of optimal human choices.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

PROJECT=Path(os.environ.get('YGO_PROJECT_ROOT',Path(__file__).resolve().parents[3]))
sys.path.insert(0,str(PROJECT))
from scripts.collect_teaching_exercise import load_native
from ygoai.rl.exercise_actor import array_digest,extract_trace,hidden_identity_audit
from ygoai.rl.human_replay import response_action,canonical_response,comparable_game_events
from ygoai.rl.observation_schema import tensor_contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--proof',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for k in ('source','catalog','proof','output'):setattr(a,k,getattr(a,k).resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.source))
    import exercise_core
    import ygoai.rl.exercise_actor as trace_module
    trace_module.messages=exercise_core.messages
    proof=json.loads(a.proof.read_text('utf-8'));assert proof['status'] in ('complete','prefix_only')
    sha=proof['source']['source_sha256'];seat=proof['source']['seat']
    replay=json.loads((a.catalog/'parsed.json').read_text('utf-8'))[sha]
    assert (replay['initial_lp'],replay['initial_hand'],replay['draw_count'])==(8000,5,1),'unsupported opening configuration'
    runtime=a.source/'actor-runtime-v3'
    args=SimpleNamespace(native=runtime/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so',database=runtime/'assets/cards.cdb',scripts=runtime/'assets/script',code_list=runtime/'code_list.txt',semantics=runtime/'semantics')
    native=load_native(args,runtime/'registration.ydk')
    initial=[[c,s,loc,0,8] for s in (0,1) for loc,deck in ((1,replay['main_decks'][s]),(64,replay['extra_decks'][s])) for c in deck]
    env=native.ExerciseActor('',initial,str(args.semantics),1,'structured-lite-v1',replay['seed_sequence'],replay['duel_flags'],2048,True)
    record=dict(schema='crystron-general-human-demonstration-v1',source_sha256=sha,
                definition=dict(initial=initial,controlled=seat,seed_sequence=replay['seed_sequence'],duel_flags=replay['duel_flags']),
                training_ready=False,strategic_quality='unreviewed_human_choices',
                terminal_outcome=proof['final'].get('winner') if proof['status']=='complete' else None,
                source_status=proof['status'],proof_sha256=hashlib.sha256(a.proof.read_bytes()).hexdigest(),decisions=[])
    arrays=[]
    try:
        env.reset()
        for step in range(2048):
            snap=env.snapshot();events,state,responses=extract_trace(env,snap)
            assert not snap['invalid']
            for i,r in enumerate(responses):
                assert i<len(replay['responses']),'adapter consumed beyond recording'
                prompt=proof['decisions'][i]['prompt']
                assert canonical_response(prompt,r['data'])==canonical_response(prompt,replay['responses'][i]),f'response drift at {i}'
            if len(responses)==len(replay['responses']):
                assert comparable_game_events(events)==comparable_game_events(proof['events']),'event drift'
                assert state==proof['final'],'full state drift'
                assert bool(snap['done'])==(proof['status']=='complete'),'terminal boundary mismatch'
                record.update(status='converted',input_verified=True,observation_schema='structured-lite-v1',full_legal_menu=True)
                break
            assert not snap['done'],'early termination'
            action=response_action(snap,proof['decisions'][len(responses)]['prompt'],replay['responses'][len(responses)])
            obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
            if step==0:hidden_identity_audit(obs)
            record['decisions'].append(dict(step=step,player=snap['player'],message=snap['message'],phase=snap['phase'],turn_player=snap['turn_player'],turn=sum(e['op']==40 for e in events),menu=snap['menu'],action=action,teaching=snap['player']==seat and len(snap['menu'])>1,core_response_count=len(responses),observation_sha256=array_digest(obs)))
            arrays.append({k:v[0].copy() for k,v in obs.items()});env.step(action)
        else:raise TimeoutError('2048-decision capacity; not truncated for training')
    except Exception as exc:record.update(status='rejected',error=str(exc))
    finally:
        snap=env.snapshot();events,state,responses=extract_trace(env,snap)
        record.update(events=events,final=state,core_responses=responses,native_sha256=hashlib.sha256(args.native.read_bytes()).hexdigest())
        env.close()
    (a.output/'record.json').write_text(json.dumps(record,ensure_ascii=False),'utf-8')
    if arrays:np.savez_compressed(a.output/'record.npz',**{k:np.stack([v[k] for v in arrays]) for k in arrays[0]})
    print(json.dumps(dict(sha=sha,status=record['status'],decisions=len(record['decisions']),controlled=seat,source_status=proof['status'],error=record.get('error'))),flush=True)


if __name__=='__main__':main()
