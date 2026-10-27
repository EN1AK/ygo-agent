"""Freeze curriculum and evaluation roots before new parameter updates."""
import pathlib,json,hashlib
root=pathlib.Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8'))
original=read(root/'trial-plan-v1.json');selected=original['selected']
plan=dict(id='crystron-segmented-v1-20261027',selected=selected,
          parent='trial-v1/model.flax_model',parent_sha256='d278429ebcdaff25d13b2fbd2e4efc0813cdce5a9a0c79ea3d0faf4d12fd8742',
          runtime='actor-runtime-v3',trajectories='actor-converted-v4',full_legal_menu=True,
          seed=20261027,learning_rate=2e-5,optimizer='fresh Adam, global norm clip 1; carried across curriculum stages',
          sequence_steps=256,batch_histories=4,
          stages=[dict(name='suffix32',own_suffix=32,updates=256),dict(name='suffix96',own_suffix=96,updates=512),dict(name='full',own_suffix=None,updates=1024)],
          roots={},promotion_eligible=False,
          scope='same four fit and four player/deck/source-disjoint holdouts as trial-v1; no holdout gradient or search-teaching admission; all recurrent prefixes preserved',
          evaluation_policy='greedy model after root; opponent cancels optional actions or chooses first legal action; first-turn FTK or 384-decision continuation budget',
          search_policy='only fit-source off-reference states can supply new teaching data; holdout search results, if any, stay evaluation-only')
for item in selected:
    record=read(root/plan['trajectories']/(item['sha']+'.json'));assert record['training_ready'] and record['full_legal_menu']
    proof=read(root/'audit-repaired-v7'/(item['sha']+'.json'))
    decisions=record['decisions'];own=[i for i,d in enumerate(decisions) if d['player']==0]
    if item['category']=='uninterrupted':start=own[0];mode='opening'
    else:
        eligible=[]
        for chain in item['opponent_chains']:
            end=next((i for i,e in enumerate(proof['events']) if i>chain['event'] and e['op']==74),None)
            if end is None:continue
            raw=next((i for i,d in enumerate(proof['decisions']) if d['event_index']>end and d['prompt']['player']==0),None)
            if raw is not None:eligible.append(raw)
        boundary=max(eligible)
        start=next(i for i,d in enumerate(decisions) if d['player']==0 and d['core_response_count']>=boundary);mode='recovery'
    finish=max(i for i,d in enumerate(decisions) if d['player']==0 and d['message']==11 and d['menu'][d['action']]['act']=='Activate' and d['menu'][d['action']]['code']==56910167)
    idle=[i for i in own if start<=i<=finish and decisions[i]['message']==11]
    roots=[dict(mode=mode,step=start)]
    for label,fraction in [('middle',1/3),('late',2/3)]:
        target=start+(finish-start)*fraction
        step=min(idle,key=lambda i:abs(i-target));roots.append(dict(mode=label,step=step))
    roots.append(dict(mode='finish',step=finish));plan['roots'][item['sha']]=roots
    item['record_sha256']=hashlib.sha256((root/plan['trajectories']/(item['sha']+'.json')).read_bytes()).hexdigest()
    item['observations_sha256']=hashlib.sha256((root/plan['trajectories']/(item['sha']+'.npz')).read_bytes()).hexdigest()
    assert len(own)<=plan['sequence_steps']
out=root/'learning-plan-v1.json'
assert not out.exists();out.write_text(json.dumps(plan,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(dict(plan=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),roots=plan['roots']),ensure_ascii=False))
