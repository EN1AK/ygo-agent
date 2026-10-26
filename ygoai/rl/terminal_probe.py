"""Rebuild an observed opening and test its board with a public effect."""
import copy
from collections import Counter
import numpy as np
from ygoai.rl.exercise_actor import array_digest,extract_trace,hidden_identity_audit
from ygoai.rl.observation_schema import tensor_contract
from ygoai.rl.elfnote_recovery import challenge_evidence


def definition_for(source,challenge):
    d=copy.deepcopy(source['definition'])
    deck=[c for c in d['initial'] if c[1:3]==[1,1]]
    assert len(deck)==40
    before=deck[-1][0];deck[-1][0]=challenge['code']
    assert max(Counter(c[0] for c in deck).values())<=3
    return d,dict(kind='opponent_hidden_starting_card',old_code=before,new_code=challenge['code'])


def run(native,semantics,actor,source,challenge,plan,mode):
    assert mode in ['scripted','pass','policy']
    d,change=definition_for(source,challenge);prefix=source['decisions']
    env=native.ExerciseActor('',d['initial'],str(semantics),1,'structured-lite-v1',d['seed_sequence'],d['duel_flags'],plan['max_decisions']+2)
    actor.reset();rows=[];arrays=[];start=None;checked=0;error=None;evidence=dict(completed=False,success=False)
    record=dict(schema='ygo-terminal-functional-probe-v1',definition=d,case=source['case'],mode=mode,challenge=challenge,
        transformation=change,training_ready=False,status='unknown',source_prefix_steps=len(prefix))
    try:
        env.reset()
        for step in range(plan['max_decisions']):
            snap=env.snapshot();events,state,responses=extract_trace(env,snap)
            if snap['invalid']:raise ValueError('invalid core')
            if step==len(prefix):start=len(events)
            if start is not None:
                evidence=challenge_evidence(events,start,challenge=challenge['code'],negators=tuple(plan['negators']))
                if evidence['completed']:
                    record['status']='success' if evidence['success'] else 'challenge_resolved';break
                if snap['done'] or sum(e['op']==40 for e in events)>=3:
                    record['status']='challenge_not_observed';break
            obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
            if snap['player']==0:hidden_identity_audit(obs)
            digest=array_digest(obs);logits,value,rnn=actor.predict(obs,snap['player']);menu=snap['menu']
            if not menu:raise ValueError('empty menu')
            def find(**fields):return next((i for i,m in enumerate(menu) if all(m[k]==v for k,v in fields.items())),None)
            cancel=find(act='Cancel');finish=find(finish=True)
            passive=cancel if cancel is not None else finish if finish is not None else 0
            if step<len(prefix):
                previous=prefix[step]
                assert (snap['player'],snap['message'],menu)==(previous['player'],previous['message'],previous['menu']),'changed prefix menu'
                if snap['player']==0:
                    assert digest==previous['observation_sha256'],'hidden challenge changed controlled observation'
                    assert rnn==previous['rnn_sha256'],'controlled memory changed'
                    checked+=1
                action=previous['action']
            elif snap['player']==1:
                summon=find(code=challenge['code'],act='Summon') if challenge['summon'] else None
                activate=find(code=challenge['code'],act='Activate')
                end=find(phase='End')
                action=summon if summon is not None else activate if activate is not None else end if end is not None else passive
            elif snap['turn_player']==0:
                end=find(phase='End');action=end if end is not None else passive
            elif mode=='policy':action=int(np.argmax(logits[:len(menu)]))
            else:
                tail=events[start:];last=max((i for i,e in enumerate(tail) if e['op']==74),default=-1)
                active=any(e['op']==70 and int.from_bytes(bytes.fromhex(e['raw'])[1:5],'little')==challenge['code'] for e in tail[last+1:])
                negate=next((i for code in plan['negators'] for i,m in enumerate(menu) if m['code']==code and m['act']=='Activate'),None) if mode=='scripted' and active and snap['message']==16 else None
                action=negate if negate is not None else passive
            assert 0<=action<len(menu)
            rows.append(dict(step=step,player=snap['player'],message=snap['message'],menu=menu,action=action,
                observation_sha256=digest,rnn_sha256=rnn,logits=logits[:len(menu)].tolist(),value=value,
                core_response_count=len(responses),teaching=False,searchable=False))
            arrays.append({k:v[0].copy() for k,v in obs.items()});env.step(action)
        else:record['status']='budget_exhausted'
    except (ValueError,AssertionError) as exc:
        error=str(exc);record['status']='reconstruction_failed'
    finally:
        snap=env.snapshot();events,state,responses=extract_trace(env,snap)
        record.update(decisions=rows,events=events,final=state,core_responses=responses,core_seed=env.trace()['core_seed'],
            prefix_own_observations_verified=checked,probe_event_start=start,functional_evidence=evidence,error=error)
        env.close()
    return record,{k:np.stack([x[k] for x in arrays]) for k in arrays[0]} if arrays else {}
