"""Audit every curriculum/recovery gate against hashes and executed evidence."""
import collections,hashlib,json,math,pathlib
root=pathlib.Path(__file__).resolve().parent;run=root/'learning-v1'
read=lambda p:json.loads(p.read_text('utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(root/'learning-plan-v1.json');ap=read(root/'recovery-learning-plan-v1.json')
assert sha(root/'learning-plan-v1.json')==ap['base_plan_sha256']
for key in ['sha','player','deck_sha256']:assert len({x[key] for x in plan['selected']})==8
fit={x['sha'] for x in plan['selected'] if x['split']=='fit'};held={x['sha'] for x in plan['selected'] if x['split']=='holdout'}
assert len(fit)==len(held)==4 and not fit&held
interface=read(root/'interface-independent-audit-v1.json');assert interface['passed'] and len(interface['results'])==110 and all(x['passed'] for x in interface['results'])
assert read(root/'conversion-summary-v4.json')['status']=={'converted':110}
assert sum(x['model_decisions'] for x in interface['results'])==22058
assert {x['sha'] for x in interface['results']}==set(read(root/'conversion-summary-v4.json')['converted'])
runtime=root/plan['runtime'];assert sha(runtime/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so')==read(runtime/'manifest.json')['native_sha256']
for item in plan['selected']:
    path=root/plan['trajectories']/(item['sha']+'.json')
    assert sha(path)==item['record_sha256'] and sha(path.with_suffix('.npz'))==item['observations_sha256']
training=[];parent=plan['parent_sha256']
for stage in plan['stages']+[dict(name='recovery-augmented',updates=ap['updates'])]:
    folder=run/stage['name'];s=read(folder/'summary.json');m=[json.loads(v) for v in (folder/'metrics.jsonl').read_text().splitlines()]
    assert s['completed'] and len(m)==s['updates']==stage['updates'] and s['parameter_delta_l2']>0
    assert [v['update'] for v in m]==list(range(1,stage['updates']+1))
    assert all(v['finite'] and math.isfinite(v['loss']) and math.isfinite(v['grad_norm']) for v in m)
    assert s['parent_sha256']==parent;parent=s['checkpoint_sha256']
    assert sha(folder/'model.flax_model')==parent
    sidecar=read(folder/'model.flax_model.metadata.json')
    assert sidecar['checkpoint_sha256']==parent and sidecar['code_list_hash']==sha(runtime/'code_list.txt')
    assert sidecar['semantic_table_hash']==sha(runtime/'semantics/metadata.json')
    assert sidecar['training_context']['full_legal_menu'] and not sidecar['training_context']['promotion_eligible']
    training.append(dict(stage=stage['name'],updates=len(m),checkpoint_sha256=parent,parameter_delta_l2=s['parameter_delta_l2'],summary_sha256=sha(folder/'summary.json'),metrics_sha256=sha(folder/'metrics.jsonl')))
assert (root/'learning-launch.exit').read_text().strip()=='0' and (root/'recovery-learning-launch.exit').read_text().strip()=='0'
evaluations=[]
for kind,labels,count in [('evaluation',['before','suffix32','suffix96','full','recovery-augmented'],32),('recovery-evaluation',['before','after'],5)]:
    for label in labels:
        folder=run/kind/label;s=read(folder/'summary.json');audit=read(folder/'independent-audit.json')
        assert s['completed'] and len(s['results'])==count and audit['passed'] and audit['count']==count
        assert audit['summary_sha256']==sha(folder/'summary.json')
        proof_by_name={pathlib.Path(v['path']).name:v for v in audit['results']}
        assert len(proof_by_name)==count
        counts=collections.Counter()
        for item in s['results']:
            path=folder/pathlib.Path(item['path']).name;r=read(path);proof=proof_by_name[path.name]
            assert sha(path)==item['sha256']==proof['record_sha256'] and proof['passed'] and proof['yrp_parse_passed']
            assert r['root_reached'] and r['status']!='invalid' and r['checkpoint_sha256']==s['checkpoint_sha256']
            for name,h in proof['artifacts'].items():assert sha(folder/name)==h
            counts[(item['split'],item['mode'],item['status'])]+=1
        evaluations.append(dict(kind=kind,label=label,count=count,checkpoint_sha256=s['checkpoint_sha256'],summary_sha256=sha(folder/'summary.json'),audit_sha256=sha(folder/'independent-audit.json'),counts=[dict(split=k[0],mode=k[1],status=k[2],count=v) for k,v in sorted(counts.items())]))
full=read(run/'evaluation/full/summary.json');retained=read(run/'evaluation/recovery-augmented/summary.json')
assert sum(x['status']=='success' for x in full['results'] if x['split']=='fit')==16
assert sum(x['status']=='success' for x in retained['results'] if x['split']=='fit')==16
post=read(run/'recovery-evaluation/after/summary.json');assert len(post['results'])==5 and all(x['status']=='success' for x in post['results'])
cert=read(root/'recovery-search-v2/independent-certification.json');assert cert['completed'] and cert['admitted']==5 and all(x['passed'] for x in cert['results'])
assert sha(root/'recovery-search-v2/independent-certification.json')==ap['certification_sha256']
for item in ap['rows']:
    path=root/item['path'];r=read(path);proof=r['independent_proof']
    assert sha(path)==item['record_sha256'] and sha(path.with_suffix('.npz'))==item['observations_sha256']
    assert r['training_ready'] and r['source_sha256'] in fit and r['source_sha256'] not in held
    assert proof in cert['results'] and proof['source_result_sha256']==sha(root/'recovery-search-v2'/(item['id']+'.json'))
    assert not any(d['teaching'] for d in r['decisions'][:item['root_step']])
    assert sum(d['teaching'] for d in r['decisions'])==proof['teacher_decisions']>0
teachers=read(root/'recovery-search-v2/teacher-replays/manifest.json');assert teachers['passed'] and teachers['count']==5
for item in teachers['results']:
    for name,h in item['artifacts'].items():assert sha(root/'recovery-search-v2/teacher-replays'/name)==h
result=dict(passed=True,scope='110 certified interface traces; four fit and four player/deck/source-disjoint holdouts; fixed 32-root autonomous suite; five actual off-reference recovery roots',interface_replays=110,interface_decisions=22058,updates=sum(x['updates'] for x in training),training=training,evaluations=evaluations,independently_verified_policy_replays=sum(x['count'] for x in evaluations),certified_teacher_replays=5,recovery_teacher_decisions=sum(x['teacher_decisions'] for x in cert['results']),holdout_used_for_gradients=False,promotion_eligible=False,final_checkpoint_sha256=parent,limits=['holdout results are reported separately; no general mastery claim','opponent passes optional actions after takeover; no arbitrary adversarial continuation guarantee','seven unsuccessful bounded searches remain unknown, not impossible'])
(run/'completion-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(dict(passed=True,updates=result['updates'],policy_replays=result['independently_verified_policy_replays'],final_checkpoint_sha256=parent)))
