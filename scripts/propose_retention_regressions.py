"""Collect audited post-PPO goal-loss candidates without altering any dataset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import _repo_bootstrap
from ygoai.rl.retention_candidates import normal_regression
from ygoai.rl.defense_candidates import from_files


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['before','assessment','audit','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for name,value in vars(a).items():setattr(a,name,value.resolve())
    before=read(a.before/'final-verification.json');assessment=read(a.assessment/'summary.json');audit=read(a.audit)
    assert before['passed'] and assessment['passed'] and audit['passed']
    assert audit['attempts']==2048 and audit['new_core_audits']==157 and audit['old_core_audits']==5
    assert read(a.audit.with_name(a.audit.stem+'.exit.json'))['returncode']==0
    assert Path(str(a.before)+'.train.exit').read_text().strip()==Path(str(a.assessment)+'.exit').read_text().strip()=='0'
    prior=a.before/'teach';current=a.assessment/'retention';after=read(current/'final-verification.json')
    assert after['passed'] and before['checkpoints']['teach']==audit['models']['defense_initial'] and after['checkpoint_sha256']==audit['models']['after_defense']
    assert after==assessment['retention']
    assert after['normal']==audit['normal'] and after['functional']==audit['functional']
    assert sha(current/'final-verification.json')==audit['protected'][str(current/'final-verification.json')]
    old_manifest=read(a.before/'manifest.json');new_manifest=read(current/'manifest.json');assert old_manifest['normal_cases']==new_manifest['normal_cases']
    assert before['checkpoints']['teach']==sha(prior/'model.flax_model')
    protected={str(path):sha(path) for path in [a.before/'final-verification.json',a.before/'manifest.json',prior/'model.flax_model',a.assessment/'summary.json',current/'final-verification.json',current/'manifest.json',a.audit]}
    def pair(path):
        proofpath=path.with_name(path.stem+'.audit.json');proof=read(proofpath);assert proof['passed'] and proof['record_sha256']==sha(path) and proof['observations_sha256']==sha(path.with_suffix('.npz'))
        for item in [path,path.with_suffix('.npz'),proofpath]:protected[str(item)]=sha(item)
        return read(path),dict(path=str(path),record_sha256=sha(path),observations_sha256=sha(path.with_suffix('.npz')),core_audit_sha256=sha(proofpath),native_sha256=proof['native_sha256'])
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    def save(candidate):
        name=candidate['case'];assert Path(name).name==name and name not in ['.','..']
        candidate.update(parent_sha256=after['checkpoint_sha256'],prior_checkpoint_sha256=before['checkpoints']['teach'],source_audit_sha256=sha(a.audit),exposure='Previously exposed development condition',training_ready=False)
        path=a.output/(candidate['family']+'-'+name+'.json')
        with path.open('x') as f:json.dump(candidate,f,indent=2)
        rows.append(dict(case=name,family=candidate['family'],status=candidate['status'],path=str(path),candidate_sha256=sha(path),training_ready=False))
    prior_summary=read(prior/'normal-endpoint/summary.json');current_summary=read(current/'normal/summary.json');unknown=[]
    for path in [prior/'normal-endpoint/summary.json',current/'normal/summary.json',prior/'endpoint/summary.json',current/'functional/summary.json']:protected[str(path)]=sha(path)
    assert len(prior_summary['cases'])==len(current_summary['cases'])==len(new_manifest['normal_cases'])==53
    for i,(case,b,n) in enumerate(zip(new_manifest['normal_cases'],prior_summary['cases'],current_summary['cases'])):
        assert b['name']==n['name']==case['name']
        if b['status']!='success' or n['status']=='success':continue
        if n['status']!='goal_not_reached':unknown.append(dict(case=case['name'],family='normal',status=n['status'],training_ready=False));continue
        old,prior_source=pair(prior/'normal-endpoint'/f'case-{i:02}.json');new,current_source=pair(current/'normal'/f'case-{i:02}.json')
        assert prior_source['native_sha256']==current_source['native_sha256'] and old['status']==b['status'] and new['status']==n['status']
        candidate=normal_regression(old,new,case);candidate['sources']=[prior_source,current_source];save(candidate)
    old_functional=read(prior/'endpoint/summary.json');new_functional=read(current/'functional/summary.json')
    for old in old_functional['cases']:
        if old['mode']!='policy' or old['status']!='success':continue
        name=old['name'];new=next(r for r in new_functional['cases'] if r['name']==name and r['mode']=='policy')
        if new['status']=='success':continue
        if new['status']!='goal_not_reached':unknown.append(dict(case=name,family='functional',status=new['status'],training_ready=False));continue
        teacher=next(r for r in new_functional['cases'] if r['name']==name and r['mode']=='scripted')
        prior_record,prior_source=pair(Path(old['path']));current_record,current_source=pair(Path(new['path']))
        assert prior_record['case']==current_record['case']==name and prior_record['definition']==current_record['definition']
        assert prior_source['native_sha256']==current_source['native_sha256'] and prior_record['status']==old['status'] and current_record['status']==new['status']
        candidate=dict(schema='ygo-retention-candidate-v1',case=name,family='functional_goal',status='observed_functional_goal_regression',sources=[prior_source,current_source],label_semantics='Observed loss of the declared functional goal. Different-policy histories are not a same-root causal pair; current-model search or reconstruction remains required.',required_admission=['New current-parent protocol and valid solution discovery','Full current history/RNN reconstruction','Double replay and native audit before supervision'])
        if teacher['status']=='success':
            pair(Path(teacher['path']))
            try:response=from_files(Path(teacher['path']),Path(new['path']))
            except (AssertionError,ValueError,KeyError):response=None
            if response is not None:candidate['current_response_candidate']=response
        save(candidate)
    assert len(prior_summary['old'])==len(current_summary['old'])==5
    for i,(b,n) in enumerate(zip(prior_summary['old'],current_summary['old'])):
        if not b['success'] or n['success']:continue
        paths=[prior/'normal-endpoint/old'/f'{i}.json',current/'normal/old'/f'{i}.json'];old_records=[read(path) for path in paths];assert old_records[0]['case']==old_records[1]['case']
        sources=[]
        for path in paths:
            for item in [path,path.with_suffix('.npz')]:protected[str(item)]=sha(item)
            sources.append(dict(path=str(path),record_sha256=sha(path),observations_sha256=sha(path.with_suffix('.npz'))))
        save(dict(schema='ygo-retention-candidate-v1',case=f'legacy-{i}',family='legacy_fixture',status='observed_legacy_fixture_regression',sources=sources,label_semantics='Observed old-fixture score loss, not a strategic action label. The separately audited legacy native/synthetic fixture needs its dedicated intake before teaching.',required_admission=['Dedicated current-model legacy-fixture reconstruction','Verify an actual valid alternative using this fixture family audit','New explicit teaching protocol; synthetic KL-only examples cannot silently become CE']))
    assert all(sha(Path(path))==value for path,value in protected.items())
    result=dict(schema='ygo-retention-intake-v1',parent_sha256=after['checkpoint_sha256'],prior_checkpoint_sha256=before['checkpoints']['teach'],candidates=rows,counts=dict(Counter(r['family'] for r in rows)),unknown=unknown,protected=protected,training_ready=False,training_started=False,producer_sha256=sha(Path(__file__)),scope='Posthoc candidate capture only after completed formal assessment and independent raw audit. Strict goal loss does not prove strategic error. No source masks, checkpoints, current evaluation, or supervision changed.')
    with (a.output/'manifest.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['protected','candidates']}),flush=True)


if __name__=='__main__':main()
