"""Admit a frozen runtime, then run matched ordinary/alternating rehearsal PPO."""
import argparse,json,os,sys,shutil,hashlib,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=['prepare','smoke','train'],required=True);p.add_argument('--role',choices=['control','rehearse'],default='control');a=p.parse_args()
    a.root=a.root.resolve();a.output=a.output.resolve();source=Path(__file__).resolve().parents[1];sys.path[:0]=[str(source),str(source/'scripts')]
    from scripts.run_elfnote_formal import read,write,sha,child,command,verify_runtime,verify_training
    from scripts.prepare_elfnote_rehearsal import prepare
    plan=read(source/'assets/exercises/teaching-v1/elfnote-ppo-rehearsal.json');release=a.root/plan['runtime_release']
    env=dict(os.environ,JAX_PLATFORMS='cuda',JAX_DEFAULT_MATMUL_PRECISION='float32',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
        LD_LIBRARY_PATH='/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:'+str(a.root/'.venv/lib/python3.10/site-packages/nvidia/cudnn/lib'))
    cpu=dict(env,JAX_PLATFORMS='cpu',PYTHONPATH=str(release/'repo/ygoenv')+':'+str(release/'repo'))
    if a.phase=='prepare':
        a.output.mkdir(parents=True,exist_ok=False);base=a.root/plan['base_runtime'];verify_runtime(base)
        execution=read(source/'source-manifest.json')
        assert all(hashlib.sha256((source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in execution['all_files'].items())
        assert not release.exists();shutil.copytree(base,release,ignore=shutil.ignore_patterns('__pycache__'))
        (release/'frozen-manifest.json').rename(release/'previous-frozen-manifest.json')
        for name in execution['all_files']:
            assert not name.endswith('.so');target=release/'repo'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
            if name.endswith('.sh'):target.write_bytes(target.read_bytes().replace(b'\r\n',b'\n'))
        assert all(hashlib.sha256((release/'repo'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in execution['all_files'].items())
        old=read(release/'preparation.json');old.update(source=execution['commit'],source_files=len(execution['all_files']),base_runtime=str(base))
        (release/'preparation.json').write_text(json.dumps(old,indent=2));(release/'source-manifest.json').write_text(json.dumps(execution,indent=2))
        assert sha(release/'repo/ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so')==plan['native_sha256']
        child([sys.executable,str(release/'repo/scripts/validate_formal_runtime.py'),'--release',str(release),'--attempt','cpu-rehearsal'],a.output,'cpu-runtime',cpu,900)
        assert read(release/'validation/cpu-rehearsal/report.json')['passed']
        protocol=release/'validation/protocol-rehearsal';protocol.mkdir()
        child([sys.executable,'-m','unittest','discover','-s',str(release/'repo/tests'),'-p','test_ygocore_protocol_boundaries.py'],a.output,'boundary-tests',cpu)
        child([sys.executable,str(release/'repo/scripts/build_protocol_coverage.py'),'--contract',str(release/'validation/protocol-v1/contract-a.json'),
            '--adapter',str(release/'build-provenance/compiled-adapter.h'),'--boundary-test',str(release/'repo/tests/test_ygocore_protocol_boundaries.py'),
            '--runtime-smoke',str(release/'validation/protocol-v1/historical-runtime-smoke-50-v4.json'),'--structured-environment',str(release/'validation/cpu-rehearsal/report.json'),
            '--output',str(protocol/'coverage.json')],a.output,'protocol-coverage',cpu)
        assert read(protocol/'coverage.json')['summary']['uncovered_branches']==0
        for role in plan['roles']:prepare(a.root,role,a.output/('rehearsal-'+role+'.json'))
        c=read(a.output/'rehearsal-control.json');r=read(a.output/'rehearsal-rehearse.json');c.pop('role');r.pop('role');assert c==r
        artifacts=[dict(path=str(path.relative_to(release)),sha256=sha(path),bytes=path.stat().st_size) for path in sorted(release.rglob('*')) if path.is_file() and '__pycache__' not in path.parts]
        write(release/'frozen-manifest.json',dict(format_version=1,source_commit=execution['commit'],native_sha256=plan['native_sha256'],artifacts=artifacts))
        write(a.output/'preparation.json',dict(passed=True,plan=plan,source=execution,runtime_manifest_sha256=sha(release/'frozen-manifest.json')))
        print(json.dumps(dict(phase='prepare',passed=True,source=execution['commit'])),flush=True);return
    prep=read(a.output/'preparation.json');assert prep['passed'];verify_runtime(release)
    assert sha(release/'frozen-manifest.json')==prep['runtime_manifest_sha256']
    if a.phase=='train':
        for role in plan['roles']:
            assert read(a.output/('smoke-'+role)/'verification.json')['passed']
            assert read(a.output/('smoke-'+role)/'rehearsal-verification.json')['passed']
    folder=a.output/(('smoke-' if a.phase=='smoke' else '')+a.role);folder.mkdir(exist_ok=False)
    checkpoint=release/'checkpoints/full-context.flax_model';assert sha(checkpoint)==plan['course_checkpoint_sha256']
    steps=plan['smoke_timesteps'] if a.phase=='smoke' else plan['total_timesteps_per_arm'];config=a.output/('rehearsal-'+a.role+'.json')
    argv=command(plan,release,checkpoint,folder,steps)+['--scene-rehearsal-config',str(config)]
    write(folder/'launch-plan.json',dict(plan=plan,role=a.role,phase=a.phase,command=argv,rehearsal_config_sha256=sha(config),runtime_manifest_sha256=sha(release/'frozen-manifest.json'),started=time.time()))
    if a.phase=='smoke':
        config_log=child(argv+['--config-only'],folder,'config-only',env);assert 'cuda' in config_log.lower()
        assert not (folder/'scene-rehearsal-admission.json').exists(),'config-only initialized an optimizer'
    child(argv,folder,'train',env,plan['max_seconds_smoke'] if a.phase=='smoke' else plan['max_seconds_per_arm'])
    verify_runtime(release);batch=plan['local_num_envs']*plan['num_steps']*plan['num_actor_threads']
    result=verify_training(folder,checkpoint,steps,batch)
    logs=[json.loads(x) for x in (folder/'scene-rehearsal.jsonl').read_text().splitlines()];assert len(logs)==steps//batch
    admission=read(folder/'scene-rehearsal-admission.json');assert admission['admission']['passed']
    import math
    cfg=read(config);cc=admission['admission']['ce_coeff'];kc=admission['admission']['kl_coeff']
    for i,row in enumerate(logs,1):
        assert row['ppo_update']==row['rehearsal_updates']==i and row['role']==a.role and row['finite']
        assert all(math.isfinite(x) for x in [row['loss'],row['grad_norm'],row['parameter_delta_l2']]+row['ce']+row['kl'])
        expected=sum(x*y for x,y in zip(cc,row['ce']))+sum(x*y for x,y in zip(kc,row['kl']))
        assert abs(expected-row['loss'])/max(1.,abs(expected),abs(row['loss']))<1e-6
        if a.role=='control':assert row['loss']==row['grad_norm']==row['parameter_delta_l2']==0.
    if a.role=='rehearse':assert any(r['parameter_delta_l2']>0 for r in logs)
    assert sha(config)==read(folder/'launch-plan.json')['rehearsal_config_sha256']
    assert all(sha(Path(path))==digest for path,digest in cfg['protected'].items())
    assert all(sha(Path(row['path']))==row['record_sha256'] and sha(Path(row['path']).with_suffix('.npz'))==row['observations_sha256'] for row in cfg['rows'])
    write(folder/'rehearsal-verification.json',dict(passed=True,role=a.role,updates=len(logs),model=result['checkpoint_sha256'],
        extra_scene_seconds=sum(x['seconds'] for x in logs),config_sha256=sha(config)))
    print(json.dumps(dict(phase=a.phase,role=a.role,passed=True,checkpoint_sha256=result['checkpoint_sha256'])),flush=True)


if __name__=='__main__':main()
