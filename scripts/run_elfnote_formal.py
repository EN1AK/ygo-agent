"""Frozen full-runtime admission, then bounded ordinary mixed PPO with audit logs."""
import argparse,hashlib,json,math,os,re,shutil,subprocess,sys,time
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)


def child(command,folder,label,env,timeout=300):
    with (folder/(label+'.log')).open('x') as log:
        result=subprocess.run(command,env=env,cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    write(folder/(label+'.exit.json'),dict(returncode=result.returncode,command=command));result.check_returncode()
    return (folder/(label+'.log')).read_text()


def freeze_runtime(a,plan,release):
    repo=release/'repo';folder=release/'validation/protocol-v1';folder.mkdir(parents=True,exist_ok=False)
    cpu=read(release/'validation/cpu-v3/report.json');assert cpu['passed'] and cpu['native_sha256']==plan['native_sha256']
    core=a.root/'training-runs/protocol-source-ygopro-core-f969296';adapter=release/'build-provenance/compiled-adapter.h'
    env=dict(os.environ,PYTHONPATH=str(repo/'ygoenv')+':'+str(repo),JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for suffix in ['a','b']:
        child([sys.executable,str(repo/'scripts/audit_ygocore_protocol.py'),'--core-source',str(core),'--adapter',str(adapter),'--expected-revision','f96929650ff8685b82fd48670126eae406366734','--output',str(folder/('contract-'+suffix+'.json'))],folder,'audit-'+suffix,env)
    assert read(folder/'contract-a.json')==read(folder/'contract-b.json')
    historic=next((p for p in [a.root/'training-runs/protocol-gate-v1-20260923/runtime-smoke-50-v4.json',a.root/'ygo-agent/training-runs/protocol-gate-v1-20260923/runtime-smoke-50-v4.json'] if p.exists()),None)
    assert historic is not None,'Original protocol evidence missing; do not substitute a fabricated fifty-game result'
    shutil.copy2(historic,folder/'historical-runtime-smoke-50-v4.json')
    child([sys.executable,'-m','unittest','discover','-s',str(repo/'tests'),'-p','test_ygocore_protocol_boundaries.py'],folder,'boundary-tests',env)
    child([sys.executable,str(repo/'scripts/build_protocol_coverage.py'),'--contract',str(folder/'contract-a.json'),'--adapter',str(adapter),'--boundary-test',str(repo/'tests/test_ygocore_protocol_boundaries.py'),'--runtime-smoke',str(folder/'historical-runtime-smoke-50-v4.json'),'--structured-environment',str(release/'validation/cpu-v3/report.json'),'--output',str(folder/'coverage-current.json')],folder,'coverage',env)
    assert read(folder/'coverage-current.json')['summary']['uncovered_branches']==0
    candidate=a.root/plan['course_study']/plan['course_arm']/'model.flax_model';assert sha(candidate)==plan['course_checkpoint_sha256']
    target=release/'checkpoints/full-context.flax_model';assert not target.exists();shutil.copy2(candidate,target);shutil.copy2(candidate.with_name(candidate.name+'.metadata.json'),target.with_name(target.name+'.metadata.json'))
    assert sha(release/'checkpoints/parent.flax_model')==plan['history_checkpoint_sha256'] and sha(release/'decks/replay-elfnote.ydk')==plan['deck_sha256']
    artifacts=[dict(path=str(p.relative_to(release)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(release.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    write(release/'frozen-manifest.json',dict(format_version=1,source_commit=plan['runtime_source_commit'],native_sha256=plan['native_sha256'],artifacts=artifacts,
        evidence_scope='Current native56 boundary tests and CPU schema admission; retained original50 smoke is historical, not asserted as current-native fifty games'))


def verify_runtime(release):
    m=read(release/'frozen-manifest.json');assert all(sha(release/e['path'])==e['sha256'] for e in m['artifacts']);return m


def command(plan,release,checkpoint,folder,steps):
    return [sys.executable,'-u',str(release/'repo/scripts/cleanba.py'),'--checkpoint',str(checkpoint),'--ckpt-dir',str(folder/'checkpoints'),'--run-name',plan['id']+'__'+str(plan['seed']),'--seed',str(plan['seed']),
        '--deck',str(release/'decks/replay-elfnote.ydk'),'--code-list-file',str(release/'code_list.txt'),'--semantic-asset-dir',str(release/'semantics'),'--observation-schema','structured-lite-v1','--max-options','128',
        '--train-opponent','mixed','--historical-checkpoints',str(release/'checkpoints/parent.flax_model'),'--mixed-self-actors','5','--mixed-history-actors','3','--mixed-bot-actors','2',
        '--local-num-envs',str(plan['local_num_envs']),'--local-env-threads',str(plan['local_env_threads']),'--num-actor-threads',str(plan['num_actor_threads']),'--num-steps',str(plan['num_steps']),
        '--num-minibatches',str(plan['num_minibatches']),'--learning-rate',str(plan['learning_rate']),'--update-epochs',str(plan['update_epochs']),'--actor-device-ids','0','--learner-device-ids','0',
        '--total-timesteps',str(steps),'--save-interval','10','--max-checkpoints','4','--eval-interval','0','--local-eval-episodes','4','--tb-dir','None','--log-frequency','1','--no-concurrency']


def finite_training_metrics(text,updates):
    lines=[line for line in text.splitlines() if line.startswith('losses total=')];finite=[line for line in text.splitlines() if line.startswith('optimizer_finite=')]
    assert len(lines)==len(finite)==updates,(len(lines),len(finite),updates)
    for line in lines:
        values=re.findall(r'=([^,\s]+)',line);assert len(values)==5 and all(math.isfinite(float(x)) for x in values)
    for line in finite:
        assert 'True' in line and 'False' not in line,line
        for field in ['notfinite_count','total_notfinite']:
            match=re.search(r'\b'+field+r'=([^,]+)',line);assert match,line
            values=re.findall(r'[-+]?\d+',match[1]);assert values and all(int(v)==0 for v in values),line
    return dict(updates=updates,zero_nonfinite_updates=True)


def verify_training(folder,start,steps,batch):
    import numpy as np
    import flax.serialization
    text=(folder/'train.log').read_text();updates=steps//batch
    finite_training_metrics(text,updates)
    checkpoints=sorted((folder/'checkpoints').glob('*.flax_model'));assert checkpoints
    latest=checkpoints[-1];assert f'_step_{steps:012d}.' in latest.name,latest
    before=flax.serialization.msgpack_restore(start.read_bytes())['params'];after=flax.serialization.msgpack_restore(latest.read_bytes())['params']
    def leaves(tree):
        if isinstance(tree,dict):return [v for k in sorted(tree) for v in leaves(tree[k])]
        return [np.asarray(tree)]
    x,y=leaves(before),leaves(after);assert len(x)==len(y) and all(a.shape==b.shape and np.isfinite(b).all() for a,b in zip(x,y))
    delta=sum(float(np.sum((a.astype(np.float64)-b.astype(np.float64))**2)) for a,b in zip(x,y));assert delta>0
    metadata=read(latest.with_name(latest.name+'.metadata.json'));assert metadata['checkpoint_sha256']==sha(latest)
    result=dict(passed=True,updates=updates,fitted_env_steps=steps,expected_collected_env_steps=steps+batch,checkpoint=str(latest),checkpoint_sha256=sha(latest),parameter_delta_l2=math.sqrt(delta),zero_nonfinite_updates=True,process_exit_zero=read(folder/'train.exit.json')['returncode']==0)
    assert result['process_exit_zero'];write(folder/'verification.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','plan','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=['preflight','train'],required=True);p.add_argument('--arm',choices=['plain','after_defense'],default='plain');a=p.parse_args();a.root=a.root.resolve();a.output=a.output.resolve()
    plan=read(a.plan);release=a.root/plan['runtime_release'];course=read(a.root/plan['course_study']/'final-verification.json');assert course['passed'] and course['formal_training_exercise_gate']
    env=dict(os.environ,JAX_PLATFORMS='cuda',JAX_DEFAULT_MATMUL_PRECISION='float32',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
        LD_LIBRARY_PATH='/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:'+str(a.root/'.venv/lib/python3.10/site-packages/nvidia/cudnn/lib'))
    if a.phase=='preflight':
        if not (release/'frozen-manifest.json').exists():freeze_runtime(a,plan,release)
        folder=a.output/'preflight';steps=plan['smoke_timesteps'];checkpoint=release/'checkpoints/full-context.flax_model'
    else:
        assert read(a.output/'preflight/verification.json')['passed']
        folder=a.output/a.arm;steps=plan['total_timesteps_per_arm'];checkpoint=release/'checkpoints/full-context.flax_model'
        if a.arm=='after_defense':
            source=a.root/plan['arms'][a.arm]['study'];proof=read(source/'final-verification.json');assert proof['passed'] and proof['defensive_policy_gate']
            assert read(source/'manifest.json')['plan']['parent_sha256']==plan['course_checkpoint_sha256']
            checkpoint=source/plan['arms'][a.arm]['arm']/'model.flax_model'
    runtime=verify_runtime(release);folder.mkdir(parents=True,exist_ok=False)
    argv=command(plan,release,checkpoint,folder,steps);batch=plan['local_num_envs']*plan['num_steps']*plan['num_actor_threads']
    write(folder/'launch-plan.json',dict(plan=plan,plan_sha256=sha(a.plan),runtime_manifest_sha256=sha(release/'frozen-manifest.json'),checkpoint_sha256=sha(checkpoint),command=argv,phase=a.phase,arm=a.arm,started=time.time(),environment={k:env[k] for k in ['JAX_PLATFORMS','JAX_DEFAULT_MATMUL_PRECISION','XLA_PYTHON_CLIENT_PREALLOCATE']}))
    if a.phase=='preflight':
        child([sys.executable,str(Path(__file__).with_name('probe_elfnote_formal_gpu.py')),'--release',str(release),'--checkpoint',str(checkpoint),'--output',str(folder/'model-reload.json')],folder,'model-reload',env)
        config=child(argv+['--config-only'],folder,'config-only',env);assert 'cuda' in config.lower()
    child(argv,folder,'train',env,plan['max_seconds_smoke'] if a.phase=='preflight' else plan['max_seconds_per_arm'])
    verify_runtime(release);result=verify_training(folder,checkpoint,steps,batch)
    print(json.dumps(dict(phase=a.phase,arm=a.arm,**result)),flush=True)


if __name__=='__main__':main()
