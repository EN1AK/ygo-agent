"""Functional recovery: legal opening plus a real next-turn negation challenge."""
import numpy as np
from ygoai.rl.exercise_actor import array_digest,extract_trace,hidden_identity_audit
from ygoai.rl.observation_schema import tensor_contract

CHALLENGE=45812361
NEGATORS=(50954680,84815190)


def board_resources(state):
    own=[c for c in state['cards'] if c['player']==0]
    return dict(negators=[c['code'] for c in own if c['location']==4 and c['position']&5 and c['code'] in NEGATORS],
                hand=sum(c['location']==2 for c in own))


def challenge_evidence(events,start):
    """Bind chain-local source IDs, controller, resolution and negation index."""
    sources={};solved=set();negated=set();disabled=set()
    for e in events[start:]:
        raw=bytes.fromhex(e['raw']);op=e['op']
        if op==70:
            if len(raw)!=17:raise ValueError('unexpected chaining layout')
            link=raw[16]
            if link==1:sources={};solved=set();negated=set();disabled=set()
            sources[link]=dict(code=int.from_bytes(raw[1:5],'little'),player=raw[9])
        elif op==73:solved.add(raw[1])
        elif op==75:negated.add(raw[1])
        elif op==76:disabled.add(raw[1])
        elif op==74:
            targets=[link for link,s in sources.items() if s==dict(code=CHALLENGE,player=1)]
            if targets:
                target=targets[0]
                protectors=[dict(link=link,**s) for link,s in sources.items() if link==target+1 and s['player']==0 and s['code'] in NEGATORS
                            and link in solved and link not in negated|disabled]
                return dict(completed=True,challenge_link=target,challenge_negated=target in negated,
                            resolved_negators=protectors,success=target in negated and bool(protectors))
            sources={};solved=set();negated=set();disabled=set()
    return dict(completed=False,success=False)


def run(native,semantics,actor,case,plan,*,prefix=(),sample_seed=None,baseline=None,detailed=True,probe_mode='scripted'):
    if sample_seed is not None:raise ValueError('functional intake uses deterministic branching only')
    if probe_mode not in ['scripted','policy','pass']:raise ValueError('unknown probe mode')
    d=case['definition'];env=native.ExerciseActor('',d['initial'],str(semantics),1,'structured-lite-v1',d['seed_sequence'],d['duel_flags'],plan['max_decisions']+2)
    actor.reset();decisions=[];arrays=[];attempted=False;root=None;probe=None;probe_start=None;own=0
    record=dict(schema='ygo-elfnote-functional-recovery-v1',definition=d,case=case['name'],training_ready=False,probe_mode=probe_mode)
    try:
        env.reset()
        for step in range(plan['max_decisions']):
            snap=env.snapshot();events,state,responses=extract_trace(env,snap);resources=board_resources(state)
            if root is None and attempted and snap['player']==0 and snap['message']==11:root=step
            if probe is None and root is not None and snap['player']==0 and snap['message']==11 and resources['negators'] and resources['hand']>=plan['minimum_hand']:
                probe=len(events);probe_start=step;record['board_at_probe']=dict(resources=resources,state=state)
            evidence=challenge_evidence(events,probe) if probe is not None else dict(completed=False,success=False)
            if evidence['completed']:
                success=evidence['success'] and resources['hand']>=plan['minimum_hand']
                record.update(status='success' if success else 'goal_not_reached',functional_evidence=evidence);break
            if snap['invalid']:raise ValueError('invalid functional fixture')
            turns=sum(e['op']==40 for e in events)
            if snap['done'] or turns>=3 or (probe is None and turns>=2):
                record.update(status='goal_not_reached',functional_evidence=evidence);break
            if own>=plan['max_own_decisions']:
                record.update(status='budget_exhausted',functional_evidence=evidence);break
            obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')};obs_sha=array_digest(obs)
            if step==0:hidden_identity_audit(obs)
            logits,value,rnn=actor.predict(obs,snap['player']);menu=snap['menu'];n=len(menu)
            if not n:raise ValueError('empty functional menu')
            def find(**fields):return next((i for i,m in enumerate(menu) if all(m[k]==v for k,v in fields.items())),None)
            cancel=find(act='Cancel');finish=find(finish=True)
            passive=cancel if cancel is not None else finish if finish is not None else 0
            if step<len(prefix):
                expected=prefix[step]
                if (obs_sha,menu,snap['player'],snap['message'])!=(expected['observation_sha256'],expected['menu'],expected['player'],expected['message']):raise ValueError('prefix restore mismatch')
                action=expected['action']
            elif baseline is not None and step<baseline['root_step']:
                expected=baseline['decisions'][step]
                assert (obs_sha,menu,snap['player'],snap['message'])==(expected['observation_sha256'],expected['menu'],expected['player'],expected['message'])
                action=expected['action']
            elif probe is not None:
                if snap['player']==1:
                    summon=find(code=CHALLENGE,act='Summon') if snap['message']==11 else None
                    activate=find(code=CHALLENGE,act='Activate') if snap['message']==11 else None
                    end=find(phase='End') if snap['message'] in [10,11] else None
                    action=summon if summon is not None else activate if activate is not None else end if end is not None else passive
                elif probe_mode=='policy' and snap['turn_player']==1:
                    action=int(np.argmax(logits[:n]))
                else:
                    negate=None
                    if probe_mode=='scripted' and snap['turn_player']==1 and snap['message']==16:
                        # Only respond to the currently open challenge chain.
                        tail=events[probe:]
                        last_end=max((i for i,e in enumerate(tail) if e['op']==74),default=-1)
                        active=any(e['op']==70 and int.from_bytes(bytes.fromhex(e['raw'])[1:5],'little')==CHALLENGE for e in tail[last_end+1:])
                        if active:negate=next((i for code in NEGATORS for i,m in enumerate(menu) if m['code']==code and m['act']=='Activate'),None)
                    end=find(phase='End') if snap['turn_player']==0 and snap['message'] in [10,11] else None
                    action=negate if negate is not None else end if end is not None else passive
            elif snap['player']==0:action=int(np.argmax(logits[:n]))
            else:
                interrupt=find(code=case['intervention'],act='Activate') if not attempted and snap['message']==16 else None
                action=interrupt if interrupt is not None else passive
            assert 0<=action<n
            selected=menu[action]
            if probe is None and snap['player']==1 and snap['message']==16 and selected['code']==case['intervention'] and selected['act']=='Activate':attempted=True
            searchable=snap['player']==0 and root is not None and probe is None
            decisions.append(dict(step=step,player=snap['player'],message=snap['message'],menu=menu,action=action,observation_sha256=obs_sha,
                                  logits=logits[:n].tolist(),value=value,rnn_sha256=rnn,core_response_count=len(responses),teaching=searchable,searchable=searchable))
            if detailed:arrays.append({k:v[0].copy() for k,v in obs.items()})
            own+=snap['player']==0;env.step(action)
        else:record.update(status='budget_exhausted',functional_evidence=evidence)
    finally:
        snap=env.snapshot();events,state,responses=extract_trace(env,snap);resources=board_resources(state)
        record.update(decisions=decisions,events=events,final=state,core_responses=responses,core_seed=env.trace()['core_seed'],
                      root_step=root,intervention_attempted=attempted,probe_start_step=probe_start,
                      probe_event_start=probe,
                      search_progress=(bool(resources['negators'])+min(resources['hand']/plan['minimum_hand'],1))/2)
        env.close()
    if root is None:record['status']='intervention_not_observed'
    return record,{k:np.stack([x[k] for x in arrays]) for k in arrays[0]} if arrays else {}
