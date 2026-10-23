"""Independent source, runtime, split, objective and core checks for coverage study."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from scripts.study_elfnote_coverage import make_cases, read, write
from ygoai.rl.exercise_core import sha


def scores(result, bases):
    return {**{split:sum(c['status']=='success' for c in result['cases'] if c['split']==split)
               for split in ['train','fresh','diagnostic','original']},
            'bases':sum(c['status']=='success' for c in result['cases'] if c['name'] in bases),
            'old':sum(x['success'] for x in result['old'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','study']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); s=a.study
    m=read(s/'manifest.json'); plan=m['plan']; collection=read(s/'collection.json')
    assert m['frozen_before_inference'] and collection['admitted']
    assert all(sha(Path(path))==value for path,value in m['protected'].items())
    assert all(sha(a.root/name)==value for name,value in m['runtime'].items())
    execution=m['source']; collection_source=a.source
    if (s/'training-source.json').exists():
        resumed=read(s/'training-source.json');execution=resumed['source'];collection_source=Path(resumed['collection_source'])
        assert read(collection_source/'source-manifest.json')==m['source']
        assert read(s/'compile-attempt-01.json')['updates_executed']==0
        assert all(sha(Path(path))==value for path,value in resumed['protected'].items())
    assert all(hashlib.sha256((collection_source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value
               for name,value in m['source']['all_files'].items())
    assert all(hashlib.sha256((a.source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value
               for name,value in execution['all_files'].items())
    cases,bases,_=make_cases(plan,read(a.root/plan['previous_study']/'manifest.json')['cases'])
    assert cases==m['cases'] and bases==m['bases']
    assert {x['case'] for x in collection['lessons']}=={c['name'] for c in cases if c['split']=='train'}
    for x in collection['lessons']:
        assert sha(Path(x['path']))==x['record_sha256']
        assert read(Path(x['path']))['status']=='success'
    proofs=list(s.rglob('*.audit.json'))
    expected=3*(len(cases)+5)+24+2
    for result in collection['diagnosis']:
        if 'path' in result:
            expected+=1
        if 'paired_root' in result:
            expected+=2
    assert len(proofs)==expected,(len(proofs),expected)
    for path in proofs:
        proof=read(path);record=path.with_name(path.name.replace('.audit.json','.json'))
        assert proof['passed'] and proof['native_sha256']==plan['native_sha256']
        assert proof['record_sha256']==sha(record) and proof['observations_sha256']==sha(record.with_suffix('.npz'))
    summary=read(s/'summary.json');assert summary['completed'] and not summary['production_eligible']
    results={};curves=[];objective_errors={}
    for arm in summary['arms']:
        mode=arm['mode'];folder=s/mode;admission=read(folder/'admission.json')
        assert admission['passed'] and admission['max_logit_error']<.002 and admission['histories']==34
        assert admission['micro_batch_histories']==2 and admission['gradient_relative_error']<.002 and admission['gradient_objective_error']<1e-5
        names=admission['lesson_names'];assert len(names)==24
        if mode=='narrow':
            assert names==[name for name in bases for _ in range(6)]
        else:
            assert names==[c['name'] for c in cases if c['split']=='train']
        rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert [x['update'] for x in rows]==list(range(1,plan['updates_per_arm']+1))
        assert all(np.isfinite([x['loss'],x['grad_norm']]+x['ce']+x['kl']).all() for x in rows)
        error=max(abs(x['loss']-(np.mean(x['ce'][:24])+plan['retention_coef']*np.mean(x['kl'][24:]))) for x in rows)
        assert error<1e-5,error;objective_errors[mode]=error
        assert sha(folder/'model.flax_model')==arm['checkpoint_sha256']
        assert read(folder/'endpoint/summary.json')==arm['result']
        results[mode]=scores(arm['result'],bases)
        for update in plan['probe_updates']:
            probe=read(folder/f'probe-{update}/summary.json')
            assert all(c['split'] in ['train','original'] for c in probe['cases'])
            curves.append(dict(mode=mode,update=update,**scores(probe,bases)))
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008'
    native=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014/exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    old_paths=sorted(s.glob('**/old/[0-4].json'));assert len(old_paths)==45
    old=[]
    for path in old_paths:
        cmd=[sys.executable,'-m','scripts.verify_elfnote_study','--study',str(s),'--source',str(a.source),
             '--native',str(native),'--database',str(release/'repo/assets/locale/zh/cards.cdb'),
             '--scripts',str(release/'repo/scripts/script'),'--output',str(s/'unused.json'),'--old-record',str(path)]
        response=subprocess.run(cmd,check=True,capture_output=True,text=True,cwd=a.source)
        proof=json.loads(response.stdout.strip().splitlines()[-1]);assert proof['passed']
        old.append(dict(path=str(path.relative_to(s)),proof=proof))
    narrow,varied=results['narrow'],results['varied']
    gain=varied['fresh']-narrow['fresh']
    gate=gain>=2 and varied['bases']>=narrow['bases'] and varied['original']==narrow['original']==5 and varied['old']==narrow['old']==5
    result=dict(passed=True,source_commit=execution['commit'],collection_source_commit=m['source']['commit'],source_files=len(execution['all_files']),runtime_files=len(m['runtime']),
                new_audits=len(proofs),old_audits=old,objective_errors=objective_errors,learning_curve=curves,
                baseline=scores(collection['before'],bases),endpoints=results,fresh_gain=gain,development_gate_passed=gate,
                production_eligible=False,summary_sha256=sha(s/'summary.json'))
    write(s/'final-verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['old_audits','learning_curve']}),flush=True)


if __name__=='__main__':
    main()
