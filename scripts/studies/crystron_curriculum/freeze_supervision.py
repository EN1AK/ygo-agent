"""Freeze supervised curricula with connected player/deck/match splits.

Known player/deck families stay in fit. Curriculum derivatives never enter
independent practice holdout. Holdout is fixed before any new model inference.
"""
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf-8')


def components(rows):
    parents=list(range(len(rows)));seen={}
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,row in enumerate(rows):
        for key in [('player',row['player']),('deck',row['deck_sha256']),('match',row['match_id'])]:
            if key in seen:parents[find(i)]=find(seen[key])
            seen[key]=i
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[find(i)].append(row['source_sha256'])
    return {s:hashlib.sha256('|'.join(sorted(g)).encode()).hexdigest() for g in groups.values() for s in g}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','study','practice','output','runtime'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    for k in vars(a):setattr(a,k,getattr(a,k).resolve())
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'rows').mkdir()
    catalog=read(a.study/'source/catalog.json');groups=components(catalog)
    known={x['sha'] for x in read(a.source/'learning-plan-v1.json')['selected']}
    known|={x['source_sha256'] for p in (a.study/'resource-courses-v1').glob('*/discovery.json') for x in [read(p)]}
    known_groups={groups[s] for s in known}
    candidates=[];quarantine=[]
    for path in sorted(a.practice.glob('*/record.json')):
        r=read(path);s=r['source_sha256']
        if r['status']!='converted' or not r.get('input_verified'):
            quarantine.append(dict(sha=s,reason=r.get('error','input not verified')));continue
        # Initial practice curriculum teaches completed winning human games.
        # Other legal observations remain available, with no fabricated labels.
        if r['source_status']!='complete' or r['terminal_outcome']!=r['definition']['controlled']:
            quarantine.append(dict(sha=s,reason='partial or nonwinning game; needs local-goal review'));continue
        candidates.append((path,r))
    fresh=sorted({groups[r['source_sha256']] for _,r in candidates}-known_groups)
    # Deterministic complete components; no random frame-level split.
    holdout=set(fresh[::3])
    if not holdout:raise ValueError('no unexposed player/deck component for independent holdout')
    rows=[]
    def add(name,r,obs,path,group,split,goal):
        target=a.output/'rows'/(name+'.json')
        r=copy.deepcopy(r);r['training_ready']=True
        r['admission']=dict(group=group,split=split,label_meaning='verified resource witness' if group!='practice' else 'observed winning human actions; not claimed optimal',source_record_sha256=sha(path))
        write(target,r);np.savez_compressed(target.with_suffix('.npz'),**obs)
        rows.append(dict(id=name,path=str(target.relative_to(a.output)),record_sha256=sha(target),observations_sha256=sha(target.with_suffix('.npz')),group=group,split=split,source_sha256=r['source_sha256'],family=groups[r['source_sha256']],goal=goal,own_steps=sum(d['player']==r['definition']['controlled'] for d in r['decisions'])))
    for path in sorted((a.study/'resource-courses-v1').glob('*/discovery.json')):
        r=read(path)
        assert r['independent_audit']['passed']
        boundary=r['boundaries'][-1];end=boundary['step']
        if end<3:continue
        # Only a settled main-phase boundary may be used as a resource endpoint.
        r['decisions']=r['decisions'][:end];r['core_responses']=r['core_responses'][:boundary['response_count']]
        r['events']=r['events'][:boundary['event_count']];r['final']=boundary['state']
        r['admission_requires_fresh_endpoint_replay']=True
        hand=[x[0] for x in r['definition']['initial'] if x[1:3]==[0,2]]
        assert hand==r['starting_hand']
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as f:obs={k:v[:end] for k,v in f.items()}
        targets=Counter((x['location'],x['code']) for x in r['final']['cards'] if x['player']==0 and x['location'] in (2,4,8,16,32))
        goal=dict(kind='settled_resource_inventory',cards=[dict(location=k[0],code=k[1],count=v) for k,v in targets.items()],functional_disruption_claim=False,grading_scope='specified resource inventory; alternative functional boards scored separately in future')
        add(path.parent.name,r,obs,path,'foundation' if len(hand)==1 else 'derivation','fit',goal)
    for path,r in candidates:
        s=r['source_sha256'];split='holdout' if groups[s] in holdout else 'fit'
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as f:obs=dict(f)
        add(s[:12],r,obs,path,'practice',split,dict(kind='win_after_first_own_main',opponent_after_root='cancel optional chain, otherwise first legal action',not_original_opponent_policy=True))
    parent=a.source/'learning-v1/recovery-augmented/model.flax_model'
    plan=dict(id='crystron-theory-practice-supervision-v1',seed=27102026,rows=rows,
              runtime=str(a.runtime),parent=str(parent),parent_sha256=sha(parent),
              code_list_sha256=sha(a.runtime/'code_list.txt'),semantics_sha256=sha(a.runtime/'semantics/metadata.json'),native_sha256=sha(a.runtime/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'),
              batch_size=2,learning_rate=2e-5,max_sequence_steps=2048,gradient_clip=1.0,
              stages=[dict(name='foundation',updates=512,mix={'foundation':1}),dict(name='derivation',updates=512,mix={'foundation':.5,'derivation':.5}),dict(name='practice',updates=1024,mix={'foundation':.4,'derivation':.2,'practice':.4})],
              evaluation=dict(resource_decision_budget=256,practice_decision_budget=512,fit_practice_count=8,holdout_practice_count=8,resource_seeds=[1,271,907],no_checkpoint_selection_on_holdout=True),
              split=dict(unit='connected components of player, exact deck multiset and match',known_components=sorted(known_groups),holdout_components=sorted(holdout)),
              optimizer='fresh Adam; moments carried across curriculum stages',promotion_eligible=False)
    write(a.output/'plan.json',plan);write(a.output/'quarantine.json',quarantine)
    summary=dict(rows=len(rows),counts=dict(Counter(x['group']+'/'+x['split'] for x in rows)),max_own_steps=max(x['own_steps'] for x in rows),quarantined=len(quarantine),endpoint_replay_required_before_training=True)
    write(a.output/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
