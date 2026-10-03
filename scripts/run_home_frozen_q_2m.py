"""User-approved home Q continuation; isolated bounded throughput sweep first."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--repo',required=True,type=Path)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();args.root.mkdir(exist_ok=False,parents=True)
    base=Path('/home/ygo/ygo-agent')
    previous=base/'training-runs/skystriker-warm-q-262k-e66a427-20261004'
    release=base/'dist/runtime-releases/skystriker-place-forward-505b855-20261003'
    cp=previous/'checkpoints/41032032_step_000000262144.flax_model.candidate_q'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(cp)=='92c45d476fa8c1b4593c3d9382dffea51b2e53fc7082cdd4cf957d334ed81895'
    metadata=json.loads(cp.with_name(cp.name+'.candidate_q.json').read_text())
    assert metadata['checkpoint_sha256']==sha(cp)
    for item in json.loads((release/'manifest.json').read_text())['artifacts']:
        assert sha(release/item['path'])==item['sha256'],item['path']
    original=json.loads((previous/'run-manifest.json').read_text())['command']
    gate=json.loads((previous/'baseline-manifest.json').read_text())
    env=dict(os.environ,PYTHONPATH=f'{args.repo}/scripts:{args.repo}/ygoenv:{args.repo}',
        LD_PRELOAD=str(release/'libcompat_glibc.so'),JAX_PLATFORMS='cuda',CUDA_VISIBLE_DEVICES='0',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONFAULTHANDLER='1')
    import flax.serialization
    import numpy as np
    # Keep verification imports on CPU in the supervisor; children select CUDA.
    os.environ['JAX_PLATFORMS']='cpu'
    import jax
    parent=flax.serialization.msgpack_restore(cp.read_bytes())
    actor_hash=hashlib.sha256(flax.serialization.to_bytes(parent['actor_variables'])).hexdigest()
    def run(name,actors,envs,steps,seed,formal=False):
        batch=actors*envs*64;out=args.root/name;out.mkdir()
        cmd=original.copy()
        values={'--seed':str(seed),'--ckpt-dir':str(out/'checkpoints'),'--run-name':name,
            '--tb-offset':'262144','--num-actor-threads':str(actors),'--local-num-envs':str(envs),
            '--local-env-threads':'4','--num-minibatches':str(batch//128),'--total-timesteps':str(steps),
            '--save-interval':str(max(1,250000//batch) if formal else 1000000),
            '--q-source-commit':args.source,'--q-baseline-manifest':str(out/'baseline-manifest.json')}
        for flag,val in values.items():cmd[cmd.index(flag)+1]=val
        cmd+=['--q-resume-checkpoint',str(cp),'--q-resume-source-commit',metadata['context']['source_commit']]
        manifest=dict(gate,max_new_steps=steps)
        if formal:
            manifest['frozen_critic_continuation']=dict(authorization='user-2m-20261004',
                start_step=262144,q_checkpoint_sha256=sha(cp))
        (out/'baseline-manifest.json').write_text(json.dumps(manifest,indent=2))
        record=dict(command=cmd,source=args.source,status='running',formal=formal,actors=actors,
            envs_per_actor=envs,batch=batch,minibatch=128,start_step=262144,
            target_step=262144+steps,parent_q_sha256=sha(cp),seed=seed,started=time.time(),
            actor_frozen=True,optimizer_restored=True,pilot_weights_promoted=False)
        def save(): (out/'run-manifest.json').write_text(json.dumps(record,indent=2))
        save();events=[]
        try:
            with (out/'train.log').open('x') as log:
                process=subprocess.Popen(cmd,cwd=args.repo,env=env,stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,text=True,bufsize=1)
                record['pid']=process.pid;save()
                for line in process.stdout:
                    log.write(line);log.flush()
                    if line.startswith('shadow_q='):
                        metric=json.loads(line.split('=',1)[1]);assert metric['finite']
                        assert metric['menu_mismatches']==0
                        events.append(dict(time=time.monotonic(),**metric))
                        (out/'timing.json').write_text(json.dumps(events,indent=2))
                rc=process.wait()
            assert rc==0,f'trainer exit {rc}'
            assert events[-1]['global_step']==262144+steps
            proof=json.loads((out/'frozen-actor-proof.json').read_text())
            assert proof['exact'] and proof['actor_update_count']==0
            end=sorted((out/'checkpoints').glob('*.candidate_q'))[-1]
            side=json.loads(end.with_name(end.name+'.candidate_q.json').read_text())
            assert sha(end)==side['checkpoint_sha256']
            state=flax.serialization.msgpack_restore(end.read_bytes())
            assert all(np.isfinite(x).all() for x in jax.tree.leaves(state))
            assert hashlib.sha256(flax.serialization.to_bytes(state['actor_variables'])).hexdigest()==actor_hash
            # Exclude four startup updates; interval throughput includes rollout,
            # checks, copies and learner work, not just device compute.
            steady=events[4:]
            sps=(steady[-1]['global_step']-steady[0]['global_step'])/(steady[-1]['time']-steady[0]['time'])
            record.update(status='completed',sustained_sps=sps,
                median_sps=statistics.median([(b['global_step']-a['global_step'])/(b['time']-a['time']) for a,b in zip(steady,steady[1:])]),
                checkpoint=str(end),checkpoint_sha256=sha(end),finite_arrays=len(jax.tree.leaves(state)),
                steady_seconds=steady[-1]['time']-steady[0]['time'],actor_exact=True)
            (out/'completed.txt').write_text('completed')
        except BaseException as exc:
            if 'process' in locals() and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            record.update(status='failed',error=repr(exc));(out/'failed.txt').write_text(repr(exc))
            if formal:raise
        finally:
            record['seconds']=time.time()-record['started'];save()
        return record
    schedule=dict(source=args.source,status='sweep',pilots=[],criterion='best healthy sustained end-to-end SPS; minibatch128 unchanged')
    def save_schedule(): (args.root/'schedule.json').write_text(json.dumps(schedule,indent=2))
    save_schedule()
    for actors,envs in [(2,16),(2,32),(4,16),(4,32),(6,16),(6,32)]:
        batch=actors*envs*64
        result=run(f'pilot-{actors}x{envs}',actors,envs,16*batch,81042001)
        schedule['pilots'].append(result);save_schedule()
        print(json.dumps({'pilot':f'{actors}x{envs}','status':result['status'],'sps':result.get('sustained_sps')}),flush=True)
        if result['status']!='completed':
            schedule['status']='pilot_failed_needs_review';save_schedule();return
    best=max(schedule['pilots'],key=lambda r:r['sustained_sps'])
    schedule.update(status='training',selected=best);save_schedule()
    steps=math.ceil((2000000-262144)/best['batch'])*best['batch']
    schedule['formal']=run('formal-2m',best['actors'],best['envs_per_actor'],steps,81042002,True)
    schedule['status']='completed';save_schedule()


if __name__=='__main__':main()
