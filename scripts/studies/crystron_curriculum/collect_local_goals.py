"""Extract settled local skills and natural-hand transfer from legal histories.

Every admitted endpoint is replayed through the policy adapter and a second
engine. Goal discovery uses human evidence, never a held-out model score.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import numpy as np
from supervision_runtime import Runtime,read,write,sha,extract_trace,array_digest,tensor_contract
from freeze_supervision import components
from goals import achieved,own_attacks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('frozen','study','practice','output','parent'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    for k in vars(a):setattr(a,k,getattr(a,k).resolve())
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'rows').mkdir()
    rt=Runtime(a.frozen);plan=copy.deepcopy(rt.plan);rows=[];attempts=[]
    groups=components(read(a.study/'source/catalog.json'));held=set(plan['split']['holdout_components'])
    def admit(name,record,arrays,goal,group,split,root,origin):
        record=copy.deepcopy(record)
        for i,d in enumerate(record['decisions']):d['teaching']=i>=root and d['player']==record['definition']['controlled'] and len(d['menu'])>1
        if not any(d['teaching'] for d in record['decisions']):return
        proof=rt.independent(record)
        record.update(training_ready=True,local_goal=goal,independent_endpoint_audit=proof,admission=dict(label_meaning='verified local human outcome; not a global optimality claim',source_record_sha256=sha(origin)))
        target=a.output/'rows'/(name+'.json');write(target,record)
        np.savez_compressed(target.with_suffix('.npz'),**arrays)
        rows.append(dict(id=name,path=str(target.relative_to(a.output)),record_sha256=sha(target),observations_sha256=sha(target.with_suffix('.npz')),group=group,split=split,root_step=root,source_sha256=record['source_sha256'],family=groups[record['source_sha256']],goal=goal,own_steps=sum(d['player']==record['definition']['controlled'] for d in record['decisions'])))
    # Keep the six cost-declared courses as rehearsal, with immutable provenance.
    for item in plan['rows']:
        if item['group'] not in ('foundation','derivation'):continue
        r=rt.records[item['id']]
        admit(item['id'],r,rt.observations[item['id']],item['goal'],item['group'],'fit',0,a.frozen/item['path'])
    cases=[]
    for path in sorted(a.practice.glob('*/record.json')):
        r=read(path)
        if r.get('status')=='converted' and r.get('input_verified'):cases.append((path,r,None))
    for item in plan['rows']:
        if item['group']=='foundation':
            path=rt.source/'actor-converted-v4'/(item['source_sha256']+'.json')
            cases.append((path,read(path),item))
    for path,reference,natural in cases:
        seat=reference['definition']['controlled'];env=rt.env(reference)
        decisions=[];tensors=[];pending=[];first_root=None;root_state=None;target_turn=None;found=set();response_seen=False;recovery_root=None;last_events=0
        try:
            env.reset()
            for step in range(len(reference['decisions'])+1):
                snap=env.snapshot();events,state,responses=extract_trace(env,snap)
                if snap['invalid']:raise ValueError('invalid replay')
                if first_root is None and snap['message']==11 and snap['player']==seat and state['turn_player']==seat:
                    first_root=step;root_state=copy.deepcopy(state);target_turn=sum(e['op']==40 for e in events)
                if first_root is not None:
                    for event in events[last_events:]:
                        raw=bytes.fromhex(event['raw'])
                        if event['op']==70 and raw[5]==1-seat:response_seen=True
                    turn=sum(e['op']==40 for e in events)
                    stable=snap['message'] in (10,11) and snap['player']==seat
                    if response_seen and recovery_root is None and stable:recovery_root=step
                    proposals=[]
                    if natural:
                        goal={**natural['goal'],'max_turn':target_turn}
                        if achieved(goal,state,snap,events,seat):proposals.append(('natural',goal,first_root))
                    elif turn==target_turn:
                        if 'board_break' not in found and stable:
                            old=Counter(c['code'] for c in root_state['cards'] if c['player']==1-seat and c['location'] in (4,8) and c['position']&5)
                            now=Counter(c['code'] for c in state['cards'] if c['player']==1-seat and c['location'] in (4,8))
                            removed=sorted(k for k in old if now[k]<old[k])
                            if removed:proposals.append(('board_break',dict(kind='remove_public_threat',code=removed[0],max_remaining=old[removed[0]]-1,max_turn=target_turn),first_root))
                        if 'battle' not in found and own_attacks(events,seat)>0 and state['lp'][1-seat]<root_state['lp'][1-seat]:
                            proposals.append(('battle',dict(kind='battle_damage',min_completed_attacks=1,opponent_lp_at_most=state['lp'][1-seat],max_turn=target_turn),first_root))
                        if 'postbattle' not in found and stable and state['phase']==0x100 and state['lp'][1-seat]>0 and own_attacks(events,seat)>0:
                            count=sum(c['player']==seat and (c['location']==2 or c['location']==4 and bool(c['position']&5)) for c in state['cards'])
                            if count:proposals.append(('postbattle',dict(kind='postbattle_resources',min_resources=count,max_turn=target_turn),first_root))
                        if 'recovery' not in found and recovery_root is not None and step>recovery_root and stable:
                            monsters=Counter(c['code'] for c in state['cards'] if c['player']==seat and c['location']==4 and c['position']&5)
                            if sum(monsters.values())>=2:proposals.append(('recovery',dict(kind='recovery_board',monsters=dict(monsters),max_turn=target_turn,scope='after recorded opponent effect; no claim that it was necessarily disruptive'),recovery_root))
                    for group,goal,root in proposals:
                        if group in found or not achieved(goal,state,snap,events,seat):continue
                        recorded=dict(schema='crystron-local-outcome-v1',source_sha256=reference['source_sha256'],definition=reference['definition'],decisions=copy.deepcopy(decisions),events=copy.deepcopy(events),final=copy.deepcopy(state),core_responses=copy.deepcopy(responses),core_seed=env.trace()['core_seed'])
                        arrays={k:np.stack([x[k] for x in tensors]) for k in tensors[0]}
                        split='holdout' if groups[reference['source_sha256']] in held else 'fit'
                        # Close before opening another core (callbacks are global).
                        pending.append((reference['source_sha256'][:12]+'-'+group,recorded,arrays,goal,group,split,root,path))
                        found.add(group)
                    if natural and found or turn>target_turn:break
                last_events=len(events)
                if step==len(reference['decisions']) or snap['done']:break
                d=reference['decisions'][step]
                obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                assert (snap['message'],snap['player'],snap['menu'],array_digest(obs))==(d['message'],d['player'],d['menu'],d['observation_sha256']),('reference drift',step)
                decisions.append(dict(step=step,player=snap['player'],message=snap['message'],phase=snap['phase'],turn_player=snap['turn_player'],menu=snap['menu'],action=d['action'],core_response_count=len(responses),observation_sha256=array_digest(obs)))
                tensors.append({k:v[0].copy() for k,v in obs.items()});env.step(d['action'])
            attempts.append(dict(source=reference['source_sha256'],natural=bool(natural),found=sorted(found)))
        except Exception as exc:
            attempts.append(dict(source=reference['source_sha256'],natural=bool(natural),found=[],error=str(exc)))
            pending.clear()
        finally:env.close()
        for candidate in pending:
            try:admit(*candidate)
            except Exception as exc:
                attempts[-1].setdefault('rejected_endpoints',[]).append(dict(id=candidate[0],error=str(exc)))
        print(json.dumps(attempts[-1]),flush=True)
    plan.update(id='crystron-local-practical-supervision-v1',source_root=str(rt.source),rows=rows,parent=str(a.parent),parent_sha256=sha(a.parent))
    counts=Counter(x['group']+'/'+x['split'] for x in rows)
    mix={'foundation':.25,'derivation':.15,'natural':.1,'board_break':.2,'battle':.15,'recovery':.1,'postbattle':.05}
    mix={g:w for g,w in mix.items() if counts[g+'/fit']}
    plan['stages']=[dict(name='local_skills',updates=1024,mix=mix)]
    plan['evaluation'].update(local_fit_count=4,local_holdout_count=3,fit_practice_count=0,holdout_practice_count=0)
    plan['split']['holdout_previously_evaluated']=True
    write(a.output/'plan.json',plan);write(a.output/'discovery.json',attempts)
    write(a.output/'summary.json',dict(rows=len(rows),counts=dict(counts),claim='local functional outcomes; full-game strength remains unproven'))
    print(json.dumps(dict(rows=len(rows),counts=dict(counts))),flush=True)


if __name__=='__main__':main()
