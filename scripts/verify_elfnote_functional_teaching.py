"""Independent artifact, objective, complete-engine, and curriculum-boundary audit."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from scripts.study_elfnote_coverage import read,write
from scripts.study_elfnote_root_repair import score
from ygoai.rl.elfnote_curriculum import make_cases,correction_row,followup_cases
from ygoai.rl.elfnote_recovery import challenge_evidence
from ygoai.rl.exercise_core import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','study']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();s=a.study;m=read(s/'manifest.json');plan=m['plan'];summary=read(s/'summary.json');collection=read(s/'collection.json')
    assert summary['completed'] and collection['admitted'] and not summary['production_default_replaced']
    assert all(sha(Path(path))==value for path,value in m['protected'].items())
    assert all(sha(a.root/path)==value for path,value in m['runtime'].items())
    assert all(hashlib.sha256((a.source/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for path,value in m['source']['all_files'].items())
    cases=followup_cases(plan,read(a.root/plan['reuse_study']/'manifest.json')['cases']) if plan.get('reuse_study') else make_cases(plan,read(a.root/plan['intake']/'manifest.json')['cases'],read(a.root/plan['definition_source'])['cases'])
    assert cases==m['cases'] and len(cases)==(28 if plan.get('reuse_study') else 20)
    accepted=[r for r in collection['results'] if 'path' in r];assert set(collection['required'])<={r['case'] for r in accepted}
    for r in accepted:
        path=Path(r['path']);record=read(path);_,steps=correction_row(record,{})
        assert sha(path)==r['record_sha256'] and steps==r['correction_steps']
        control=read(Path(r['negative_control']['path']));assert control['status']=='goal_not_reached' and not control['functional_evidence']['success']
        stop=record['probe_start_step'];assert record['decisions'][:stop]==control['decisions'][:stop]
    rows=read(s/'training-rows.json');assert rows['histories']==len(rows['rows'])==(70 if plan.get('context_rehearsal') else 64) and rows['accepted_functional']==len(accepted)
    for row in rows['rows']:
        path=Path(row['path']);assert sha(path)==row['record_sha256'] and sha(path.with_suffix('.npz'))==row['observations_sha256']
    proofs=list(s.rglob('*.audit.json'));assert len(proofs)==3*len(m['normal_cases'])+6*len(cases)+4*len(accepted),len(proofs)
    for path in proofs:
        proof=read(path);record_path=path.with_name(path.name.replace('.audit.json','.json'));record=read(record_path)
        assert proof['passed'] and proof['native_sha256']==plan['native_sha256'] and proof['record_sha256']==sha(record_path) and proof['observations_sha256']==sha(record_path.with_suffix('.npz'))
        if record.get('schema')=='ygo-elfnote-functional-recovery-v1':
            root,probe=record['root_step'],record['probe_start_step']
            if root is not None:assert record['intervention_evidence']['success']
            assert all(not d['searchable'] and not d['teaching'] for d in record['decisions'][:root or 0])
            if probe is not None:assert all(not d['searchable'] and not d['teaching'] for d in record['decisions'][probe:])
            if record['status']=='success':assert challenge_evidence(record['events'],record['probe_event_start'])==record['functional_evidence'] and record['functional_evidence']['success']
    endpoints={};normal={};errors={}
    for arm in summary['arms']:
        mode=arm['arm']['name'];folder=s/mode;admission=read(folder/'admission.json');n=len(accepted)
        cc=np.array([1/24]*24+[.5]*2+[0.]*34+[arm['arm']['correction_coef']/n]*n+[0.]*(4-n));kc=np.array([0.]*26+[.1]*10+[1/24]*24+[0.]*4)
        if plan.get('context_rehearsal'):cc=np.concatenate([cc,[.5,.5]+[0.]*4]);kc=np.concatenate([kc,[.5,.5]+[.25]*4])
        assert admission['passed'] and admission['histories']==rows['histories'] and admission['gradient_relative_error']<.002 and admission['max_logit_error']<.002
        if plan.get('context_rehearsal'):
            assert admission['functional_scope']==arm['arm']['functional_scope'] and len(admission['supervised_steps'])==70
            for i in range(60,64):
                record=read(Path(rows['rows'][i]['path']));steps=rows['rows'][i]['steps']
                expected=[t for t,d in enumerate(record['decisions']) if d['player']==record['definition']['controlled'] and record['root_step']<=t<record['probe_start_step']] if arm['arm']['functional_scope']=='complete_recovery' else steps
                assert admission['supervised_steps'][i]==expected
            for i in range(64,70):
                entry=rows['rows'][i];record=read(Path(entry['path']));expected=[t for t,d in enumerate(record['decisions']) if d['player']==record['definition']['controlled'] and (i<66 or (t<record['probe_start_step'] and t not in entry['excluded_steps']))]
                assert admission['supervised_steps'][i]==expected
        np.testing.assert_allclose(admission['ce_coeff'],cc);np.testing.assert_allclose(admission['kl_coeff'],kc)
        metrics=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()];assert [r['update'] for r in metrics]==list(range(1,513))
        residual=[]
        for r in metrics:
            assert np.isfinite([r['loss'],r['grad_norm']]+r['ce']+r['kl']).all()
            expected=float(np.dot(cc,r['ce'])+np.dot(kc,r['kl']));residual.append(abs(expected-r['loss'])/max(1.,abs(expected),abs(r['loss'])))
        assert max(residual)<=8*np.finfo(np.float32).eps;errors[mode]=max(residual)
        assert sha(folder/'model.flax_model')==arm['checkpoint_sha256'] and read(folder/'endpoint/summary.json')==arm['result'] and read(folder/'normal-endpoint/summary.json')==arm['normal']
        endpoints[mode]=arm['result']['scores'];normal[mode]=score(arm['normal'])
        for step in plan['probe_updates']:
            assert all(c['split']=='fit' for c in read(folder/f'probe-{step}/summary.json')['cases'])
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008';native=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014/exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    old_paths=sorted(s.glob('**/old/[0-4].json'));assert len(old_paths)==55,len(old_paths)
    for path in old_paths:
        command=[sys.executable,'-m','scripts.verify_elfnote_study','--study',str(s),'--source',str(a.source),'--native',str(native),'--database',str(release/'repo/assets/locale/zh/cards.cdb'),'--scripts',str(release/'repo/scripts/script'),'--output',str(s/'unused.json'),'--old-record',str(path)]
        result=subprocess.run(command,check=True,capture_output=True,text=True,cwd=a.source);assert json.loads(result.stdout.strip().splitlines()[-1])['passed']
    candidate=plan.get('formal_candidate_arm','teach');fit=endpoints[candidate]['fit'];retained=normal[candidate]
    gate=fit['scripted']>=3 and retained['rehearsal']==24 and retained['repair']==2 and retained['original']==retained['old']==5
    result=dict(passed=True,source_commit=m['source']['commit'],new_audits=len(proofs),old_audits=len(old_paths),accepted_lessons=len(accepted),endpoints=endpoints,normal=normal,objective_errors=errors,
                formal_training_exercise_gate=gate,defensive_policy_gate=fit['policy']>=3,production_default_replaced=False,scope='Exposed elfnote family; curriculum and formal runtime admission remain separate; no inferred live strength')
    write(s/'final-verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
