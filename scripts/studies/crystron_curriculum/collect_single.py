"""Counterfactual one-card resource proof guided by an archived human route.

The guide proposes actions; only the actual legal engine menu can execute them.
Stops at the first unmatched decision. Does not label a prefix optimal or FTK.
"""
import argparse
import ctypes
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import numpy as np

PROJECT = Path(os.environ.get('YGO_PROJECT_ROOT', Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(PROJECT))
from scripts.collect_teaching_exercise import load_native
from ygoai.rl.exercise_actor import array_digest, extract_trace, hidden_identity_audit
from ygoai.rl.observation_schema import tensor_contract
from ygoai.rl.human_replay import comparable_game_events


def signature(action):
    # Locations are kept; only positional aliases may change with the new hand.
    zone = re.sub(r'\d+', '', action['spec'])
    return tuple(action[k] for k in ('code','act','phase','finish','position','effect','unselect','place'))+(zone,)


def lua_for(initial, duel_flags):
    lines = [f'Debug.ReloadFieldBegin({duel_flags}|DUEL_ATTACK_FIRST_TURN|DUEL_SIMPLE_AI|DUEL_PSEUDO_SHUFFLE)',
             'Debug.SetPlayerInfo(0,8000,0,0)', 'Debug.SetPlayerInfo(1,8000,0,0)']
    lines += [f'Debug.AddCard({c},{p},{p},{loc},{seq},{pos})' for c,p,loc,seq,pos in initial]
    return '\n'.join(lines+['Debug.ReloadFieldEnd()'])


def independent_start(core, lua, flags):
    """Use the recorded rule version; the older fixture helper fixed MR4."""
    core.lp = [8000,8000]
    content = lua.encode()
    name = b'__crystron_resource_fixture.lua'
    core.buffers[name.decode()] = (ctypes.create_string_buffer(content),len(content))
    core.assets[name.decode()] = hashlib.sha256(content).hexdigest()
    if not core.preload_script(core.d,name,len(name)):
        core._check()
        raise ValueError('independent fixture preload rejected')
    core._check()
    core.start_duel(core.d,flags|2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sha',required=True)
    p.add_argument('--starter',type=int,required=True)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--runtime',type=Path)
    p.add_argument('--max-decisions',type=int,default=192)
    a=p.parse_args(); a.source=a.source.resolve(); a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.source))
    import exercise_core
    import ygoai.rl.exercise_actor as trace_module
    trace_module.messages=exercise_core.messages
    from modern_core import Core
    guide=json.loads((a.source/'actor-converted-v4'/(a.sha+'.json')).read_text('utf-8'))
    seat=guide['definition']['controlled']
    initial=[list(x) for x in guide['definition']['initial'] if x[1]==seat]
    for x in initial:x[1]=0
    removed=next(i for i,x in enumerate(initial) if x[0]==a.starter and x[2]==1)
    initial.pop(removed)
    initial.append([a.starter,0,2,0,1])
    # Opponent has no interactive resource. Deck only prevents empty-deck loss.
    initial += [[46986414,1,1,0,8] for _ in range(10)]
    assert [x[0] for x in initial if x[1:3]==[0,2]]==[a.starter]
    assert all(x[2] in (1,2,64) for x in initial)
    flags=guide['definition']['duel_flags']
    lua=lua_for(initial, flags)
    runtime=a.runtime.resolve() if a.runtime else a.source/'actor-runtime-v3'
    args=SimpleNamespace(native=runtime/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so',database=runtime/'assets/cards.cdb',scripts=runtime/'assets/script',code_list=runtime/'code_list.txt',semantics=runtime/'semantics')
    native=load_native(args,runtime/'registration.ydk')
    env=native.ExerciseActor(lua,initial,str(args.semantics),a.seed,'structured-lite-v1',[],flags,2048,True)
    decisions=[];arrays=[];boundaries=[];cursor=0
    record=dict(schema='crystron-single-resource-discovery-v1',training_ready=False,
                definition=dict(initial=initial,lua=lua,seed=a.seed,controlled=0,seed_sequence=[],duel_flags=flags),
                source_sha256=a.sha,starter=a.starter,
                scope='synthetic one-card resource proof; no extra starting hand/field/grave; not a legal five-card opening',
                guide_is_teacher_proposal=True,opponent='no hand/field; passive')
    source=[d for d in guide['decisions'] if d['player']==seat]
    try:
        env.reset()
        for step in range(a.max_decisions):
            snap=env.snapshot();events,state,responses=extract_trace(env,snap)
            if snap['invalid']:raise ValueError('invalid native state')
            if snap['message']==11 and snap['player']==0:
                boundaries.append(dict(step=step,response_count=len(responses),state=state,
                                       event_count=len(events)))
            if snap['done']:
                record['stop']='terminal';break
            obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
            if step==0:hidden_identity_audit(obs)
            if snap['player']!=0:
                matches=[i for i,x in enumerate(snap['menu']) if x['act']=='Cancel']
                if not matches:raise ValueError('unexpected active opponent')
                action=matches[0]
            else:
                if cursor==len(source):record['stop']='guide_exhausted';break
                expected=source[cursor];chosen=expected['menu'][expected['action']]
                matches=[i for i,x in enumerate(snap['menu']) if signature(x)==signature(chosen)]
                if snap['message']!=expected['message'] or not matches:
                    record.update(stop='guide_diverged',divergence=dict(guide_step=expected['step'],expected=chosen,message=snap['message'],menu=snap['menu']));break
                action=matches[0];cursor+=1
            decisions.append(dict(step=step,player=snap['player'],message=snap['message'],menu=snap['menu'],action=action,teaching=snap['player']==0 and len(snap['menu'])>1,observation_sha256=array_digest(obs),core_response_count=len(responses)))
            arrays.append({k:v[0].copy() for k,v in obs.items()});env.step(action)
        else:record['stop']='budget_exhausted'
    except Exception as exc:
        record.update(stop='error',error=str(exc))
    finally:
        snap=env.snapshot();events,state,responses=extract_trace(env,snap)
        record.update(decisions=decisions,events=events,final=state,core_responses=responses,
                      boundaries=boundaries,core_seed=env.trace()['core_seed'],
                      native_sha256=hashlib.sha256(args.native.read_bytes()).hexdigest())
        env.close()
    # A second engine replays the changed starting state, never the old replay.
    c=Core(a.source/'resources/libocgcore-current-v2.so',args.database,args.scripts,seed=record['core_seed'])
    try:
        independent_start(c,lua,flags)
        for response in responses:
            prompt=c.next_prompt()
            if prompt is None:raise ValueError('independent core ended early')
            c.respond(prompt,bytes.fromhex(response['data']))
        c.next_prompt()
        assert comparable_game_events(c.events)==comparable_game_events(events),'independent event mismatch'
        assert c.state()==state,'independent state mismatch'
        record['independent_audit']=dict(passed=True,events=len(c.events),responses=len(responses))
    except Exception as exc:
        record['independent_audit']=dict(passed=False,error=str(exc))
    finally:c.close()
    (a.output/'discovery.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),'utf-8')
    if arrays:np.savez_compressed(a.output/'discovery.npz',**{k:np.stack([x[k] for x in arrays]) for k in arrays[0]})
    print(json.dumps(dict(starter=a.starter,stop=record['stop'],decisions=len(decisions),stable_boundaries=len(boundaries),audit=record['independent_audit'],final_resources=[x for x in state['cards'] if x['player']==0 and x['location'] not in (1,64)])))


if __name__=='__main__':main()
