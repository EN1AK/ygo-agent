"""Pilot exact-response conversion through the existing policy adapter."""
import argparse,json,pathlib,sys,os,hashlib
from types import SimpleNamespace
import numpy as np
root=pathlib.Path(__file__).resolve().parent
project=root.parents[1]/'ygo-agent';sys.path[:0]=[str(project),str(project/'scripts')]
import exercise_core
import ygoai.rl.exercise_actor as trace_module
trace_module.messages=exercise_core.messages
from scripts.collect_teaching_exercise import load_native
from ygoai.rl.exercise_actor import array_digest,extract_trace,hidden_identity_audit
from ygoai.rl.human_replay import response_action,canonical_response,comparable_game_events
from ygoai.rl.observation_schema import tensor_contract
p=argparse.ArgumentParser();p.add_argument('--sha',required=True);p.add_argument('--audit',default='audit-repaired-v7');a=p.parse_args()
out=root/'actor-converted-v4';out.mkdir(exist_ok=True)
runtime=root/'actor-runtime-v3'
args=SimpleNamespace(native=runtime/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so',database=runtime/'assets/cards.cdb',scripts=runtime/'assets/script',code_list=runtime/'code_list.txt',semantics=runtime/'semantics')
native=load_native(args,runtime/'registration.ydk')
replay=json.loads((root/'parsed.json').read_text())[a.sha]
proof=json.loads((root/a.audit/(a.sha+'.json')).read_text())
assert proof['status']=='complete'
initial=[[code,s,loc,0,8] for s in (0,1) for loc,deck in ((1,replay['main_decks'][s]),(64,replay['extra_decks'][s])) for code in deck]
env=native.ExerciseActor('',initial,str(args.semantics),1,'structured-lite-v1',replay['seed_sequence'],replay['duel_flags'],2048,True)
record=dict(schema='crystron-actor-conversion-v1',source_sha256=a.sha,definition=dict(initial=initial,controlled=proof['source']['seat'],seed_sequence=replay['seed_sequence'],duel_flags=replay['duel_flags']),training_ready=False,full_legal_menu=True,decisions=[])
arrays=[]
try:
    env.reset()
    for step in range(2048):
        snap=env.snapshot();assert snap['full_legal_menu'];events,state,responses=extract_trace(env,snap)
        for i,r in enumerate(responses):
            if i>=len(replay['responses']) or canonical_response(proof['decisions'][i]['prompt'],r['data'])!=canonical_response(proof['decisions'][i]['prompt'],replay['responses'][i]):raise ValueError(f'semantic response drift at {i}: {r}')
        if snap['invalid']:raise ValueError('native invalid')
        if snap['done']:
            assert len(responses)==len(replay['responses']) and state['winner']==proof['final']['winner'] and state['lp']==proof['final']['lp']
            expected=[e['raw'] for e in proof['events'] if e['op'] not in (16,31)]
            assert comparable_game_events(events)==comparable_game_events(proof['events']),'game event drift'
            assert state==proof['final'],'full final state drift'
            record['wire_events_equal_excluding_layout']=[e['raw'] for e in events if e['op'] not in (16,31)]==expected
            record['zero_padding_differences']=[dict(index=i,source=replay['responses'][i],actual=r['data']) for i,r in enumerate(responses) if replay['responses'][i]!=r['data']]
            record.update(status='converted',training_ready=True);break
        if len(responses)>=len(replay['responses']):raise ValueError('actor needs extra responses')
        prompt=proof['decisions'][len(responses)]['prompt']
        action=response_action(snap,prompt,replay['responses'][len(responses)])
        obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
        # The starter audit assumes no prior reveals; later known opponent
        # cards can legitimately remain visible in the adapter's observation.
        if step==0:hidden_identity_audit(obs)
        record['decisions'].append(dict(step=step,player=snap['player'],message=snap['message'],menu=snap['menu'],action=action,core_response_count=len(responses),observation_sha256=array_digest(obs),teaching=snap['player']==record['definition']['controlled']))
        arrays.append({k:v[0].copy() for k,v in obs.items()});env.step(action)
    else:raise TimeoutError('decision budget')
except Exception as e:
    record.update(status='rejected',error=str(e))
finally:
    snap=env.snapshot();events,state,responses=extract_trace(env,snap)
    record.update(events=events,final=state,core_responses=responses,native_sha256=hashlib.sha256(args.native.read_bytes()).hexdigest(),mapper_sha256=hashlib.sha256((project/'ygoai/rl/human_replay.py').read_bytes()).hexdigest());env.close()
(out/(a.sha+'.json')).write_text(json.dumps(record,ensure_ascii=False),'utf-8')
if arrays:np.savez_compressed(out/(a.sha+'.npz'),**{k:np.stack([v[k] for v in arrays]) for k in arrays[0]})
print(json.dumps(dict(sha=a.sha,status=record['status'],decisions=len(record['decisions']),error=record.get('error'))),flush=True)
