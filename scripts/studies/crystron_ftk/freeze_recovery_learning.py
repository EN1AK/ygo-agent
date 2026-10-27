"""Freeze independently certified recovery teaching before optimization."""
import hashlib,json,pathlib
root=pathlib.Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=read(root/'learning-plan-v1.json');proof=read(root/'recovery-search-v2/independent-certification.json')
assert proof['completed'] and proof['admitted']>0
fit={x['sha'] for x in base['selected'] if x['split']=='fit'}
held={x['sha'] for x in base['selected'] if x['split']=='holdout'}
rows=[]
for item in proof['results']:
    if not item['passed']:continue
    path=root/'recovery-search-v2/admitted'/(item['id']+'.json');r=read(path)
    assert r['training_ready'] and r['source_sha256'] in fit-held
    assert r['independent_proof']==item
    boundary=item['recovery_root']
    assert not any(d['teaching'] for d in r['decisions'][:boundary])
    assert sum(d['teaching'] for d in r['decisions'])==item['teacher_decisions']>0
    assert sum(d['player']==0 for d in r['decisions'])<=base['sequence_steps']
    rows.append(dict(id=item['id'],sha=r['source_sha256'],path=str(path.relative_to(root)),record_sha256=sha(path),observations_sha256=sha(path.with_suffix('.npz')),root_step=boundary))
plan=dict(id='crystron-recovery-teaching-v1-20261027',base_plan_sha256=sha(root/'learning-plan-v1.json'),certification_sha256=sha(root/'recovery-search-v2/independent-certification.json'),parent='learning-v1/full/model.flax_model',parent_sha256=None,rows=rows,updates=512,learning_rate=2e-5,sequence_steps=base['sequence_steps'],optimizer='fresh Adam with global norm clip 1',loss='equal total mass for four original full fit histories and certified recovery suffixes; wrong-action prefix supplies recurrent state only',holdout_gradient=False,evaluation='five identical off-reference roots before and after; original 32 frozen roots retained',promotion_eligible=False)
out=root/'recovery-learning-plan-v1.json';assert not out.exists()
out.write_text(json.dumps(plan,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(dict(rows=len(rows),plan_sha256=sha(out))))
