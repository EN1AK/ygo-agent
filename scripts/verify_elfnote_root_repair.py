"""Independent evidence, loss and retention checks for root repair arms."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from scripts.study_elfnote_coverage import read,write
from scripts.study_elfnote_root_repair import cases_for,correction_indices,score
from ygoai.rl.exercise_core import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','study']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();s=a.study;m=read(s/'manifest.json');plan=m['plan'];c=read(s/'collection.json');summary=read(s/'summary.json')
    assert c['admitted'] and summary['completed'] and not summary['production_eligible']
    assert all(sha(Path(path))==value for path,value in m['protected'].items())
    assert all(sha(a.root/path)==value for path,value in m['runtime'].items())
    execution=m['source'];collection_source=a.source
    if (s/'training-source.json').exists():
        resumed=read(s/'training-source.json');execution=resumed['source'];collection_source=Path(resumed['collection_source'])
        assert read(s/'numerical-gate-diagnostic-01.json')['updates_executed']==0
        assert all(sha(Path(path))==value for path,value in resumed['protected'].items())
    assert all(hashlib.sha256((collection_source/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for path,value in m['source']['all_files'].items())
    assert all(hashlib.sha256((a.source/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for path,value in execution['all_files'].items())
    assert cases_for(plan,read(a.root/plan['previous_study']/'manifest.json')['cases'])==m['cases']
    assert len(c['results'])==2
    for r in c['results']:
        path=Path(r['path']);record=read(path)
        assert sha(path)==r['record_sha256'] and record['status']=='success'
        assert correction_indices(record)==r['correction_steps']
        assert r['paired_root']['branches'][0]['status']=='goal_not_reached'
    training=read(s/'training-rows.json');assert len(training['rows'])==60
    for row in training['rows']:
        assert sha(Path(row['path']))==row['record_sha256'] and sha(Path(row['path']).with_suffix('.npz'))==row['observations_sha256']
    proofs=list(s.rglob('*.audit.json'));assert len(proofs)==224,len(proofs)
    for path in proofs:
        proof=read(path);record=path.with_name(path.name.replace('.audit.json','.json'))
        assert proof['passed'] and proof['native_sha256']==plan['native_sha256']
        assert proof['record_sha256']==sha(record) and proof['observations_sha256']==sha(record.with_suffix('.npz'))
    endpoints={};curves=[];errors={}
    for arm in summary['arms']:
        mode=arm['arm']['name'];folder=s/mode;admission=read(folder/'admission.json')
        assert admission['passed'] and admission['histories']==60 and admission['micro_batch_histories']==2
        assert admission['max_logit_error']<.002 and admission['gradient_relative_error']<.002 and admission['gradient_objective_error']<1e-5
        assert admission['gradient_check_pairs']==[12,13] and admission['gradient_check_ce_coeff']==[.5,.5,0,0] and admission['gradient_check_kl_coeff']==[0,0,.5,.5]
        cc=np.asarray([1/24]*24+[arm['arm']['correction_coef']/2]*2+[0.]*34)
        kc=np.asarray([0.]*26+[1/10]*10+[arm['arm']['route_retention_coef']/24]*24)
        np.testing.assert_allclose(admission['ce_coeff'],cc);np.testing.assert_allclose(admission['kl_coeff'],kc)
        metrics=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
        assert [x['update'] for x in metrics]==list(range(1,plan['updates_per_arm']+1))
        assert all(np.isfinite([x['loss'],x['grad_norm']]+x['ce']+x['kl']).all() for x in metrics)
        error=max(abs(x['loss']-np.dot(cc,x['ce'])-np.dot(kc,x['kl'])) for x in metrics);assert error<1e-5,error;errors[mode]=error
        assert sha(folder/'model.flax_model')==arm['checkpoint_sha256'] and read(folder/'endpoint/summary.json')==arm['result']
        assert score(arm['result'])==arm['score'];endpoints[mode]=arm['score']
        assert len(arm['restored_roots'])==2 and read(folder/'restored-roots/summary.json')==arm['restored_roots']
        for step in plan['probe_updates']:
            probe=read(folder/f'probe-{step}/summary.json')
            assert all(x['split'] not in ['fresh_repair','fresh_triple','exposed_transfer'] for x in probe['cases'])
            curves.append(dict(mode=mode,update=step,score=score(probe)))
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008'
    native=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014/exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    old_paths=sorted(s.glob('**/old/[0-4].json'));assert len(old_paths)==80,len(old_paths);old=[]
    for path in old_paths:
        command=[sys.executable,'-m','scripts.verify_elfnote_study','--study',str(s),'--source',str(a.source),'--native',str(native),
                 '--database',str(release/'repo/assets/locale/zh/cards.cdb'),'--scripts',str(release/'repo/scripts/script'),'--output',str(s/'unused.json'),'--old-record',str(path)]
        r=subprocess.run(command,check=True,capture_output=True,text=True,cwd=a.source);proof=json.loads(r.stdout.strip().splitlines()[-1]);assert proof['passed']
        old.append(dict(path=str(path.relative_to(s)),proof=proof))
    protected=endpoints['repair_retain']
    gate=protected['repair']==2 and protected['rehearsal']==24 and protected['exposed_transfer']>=8 and protected['original']==protected['old']==5 and protected['fresh_repair']>=4
    result=dict(passed=True,source_commit=execution['commit'],collection_source_commit=m['source']['commit'],source_files=len(execution['all_files']),runtime_files=len(m['runtime']),
                new_audits=len(proofs),old_audits=old,objective_errors=errors,learning_curve=curves,baseline=score(c['before']),endpoints=endpoints,
                development_gate_passed=gate,production_eligible=False,summary_sha256=sha(s/'summary.json'))
    write(s/'final-verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['old_audits','learning_curve']}),flush=True)


if __name__=='__main__':main()
