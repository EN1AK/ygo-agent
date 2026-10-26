"""Finish post-PPO retention audit using each fixture family's correct core path."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','training','study']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.root=a.root.resolve();a.study=a.study.resolve();source=Path(__file__).resolve().parents[1]
    manifest=read(a.study/'manifest.json');training_role=manifest.get('training_role','plain')
    assert training_role in ['plain','control','rehearse','after_defense']
    training=read(a.training/training_role/'verification.json')
    if training_role in ['control','rehearse']:assert read(a.training/training_role/'rehearsal-verification.json')['passed']
    assert training['passed'] and training['process_exit_zero']
    course=a.root/'training-runs/elfnote-recovery-context-20261023'
    assert manifest['course_manifest_sha256']==sha(course/'manifest.json')
    if manifest.get('role')=='initial_course':
        expected=next(x['checkpoint_sha256'] for x in read(course/'summary.json')['arms'] if x['arm']['name']=='full_context')
        assert manifest['checkpoint_sha256']==expected==sha(course/'full_context/model.flax_model')
    else:assert manifest['checkpoint_sha256']==training['checkpoint_sha256']==sha(Path(training['checkpoint']))
    assert all(sha(a.root/name)==value for name,value in manifest['runtime'].items())
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008'
    native=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014/exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    functional=read(a.study/'functional/summary.json');normal=read(a.study/'normal/summary.json')
    assert len(normal['cases'])==len(manifest['normal_cases']) and len(functional['cases'])==2*len(manifest['functional_cases'])
    assert [(r['name'],r['split']) for r in normal['cases']]==[(c['name'],c['split']) for c in manifest['normal_cases']]
    assert [(r['name'],r['mode']) for r in functional['cases']]==[(c['name'],mode) for c in manifest['functional_cases'] for mode in ['scripted','policy']]
    proofs=list((a.study/'functional').glob('*.audit.json'))+list((a.study/'normal').glob('*.audit.json'))
    assert len(proofs)==len(manifest['normal_cases'])+2*len(manifest['functional_cases'])
    for path in proofs:
        proof=read(path);record=path.with_name(path.name.replace('.audit.json','.json'))
        assert proof['passed'] and proof['record_sha256']==sha(record) and proof['observations_sha256']==sha(record.with_suffix('.npz')) and proof['native_sha256']==sha(native)
    old=[]
    env=dict(os.environ,PYTHONPATH=str(source),JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for i in range(5):
        path=a.study/'normal/old'/f'{i}.json'
        command=[sys.executable,'-m','scripts.verify_elfnote_study','--study',str(a.study),'--source',str(source),'--native',str(native),'--database',str(release/'repo/assets/locale/zh/cards.cdb'),'--scripts',str(release/'repo/scripts/script'),'--output',str(a.study/'unused.json'),'--old-record',str(path)]
        response=subprocess.run(command,env=env,cwd=source,capture_output=True,text=True,check=True)
        proof=json.loads(response.stdout.strip().splitlines()[-1]);assert proof['passed'];old.append(dict(path=str(path),proof=proof))
    counts={s:sum(r['status']=='success' for r in normal['cases'] if r['split']==s) for s in ['repair','rehearsal','exposed_transfer','fresh_repair','fresh_triple','original']}
    counts['old']=sum(r['success'] for r in normal['old'])
    before=read(a.root/'training-runs/elfnote-recovery-context-20261023/final-verification.json')
    gate=counts['repair']==2 and counts['rehearsal']==24 and counts['original']==counts['old']==5 and functional['scores']['fit']['scripted']>=3
    result=dict(passed=True,retention_gate=gate,role=manifest.get('role','plain'),checkpoint_sha256=manifest['checkpoint_sha256'],normal=counts,
        before=before['normal']['full_context'],functional=functional['scores'],functional_before=before['endpoints']['full_context'],
        new_audits=len(proofs),old_audits=old,evaluator_sha256=manifest['evaluator_sha256'],verifier_sha256=sha(Path(__file__)),
        scope='Independent artifact/core audit passed; retention gate reports actual skills. Legacy old fixtures use their existing dedicated audit paths; original failed generic-audit evidence preserved.')
    with (a.study/'final-verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='old_audits'}),flush=True)


if __name__=='__main__':main()
