"""Run the frozen post-training matrix serially, retaining every invalid attempt."""
import argparse,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)


def paired_summary(before,after):
    def indexed(rows):
        result={}
        for result_row in rows:
            for row in result_row['episodes']:
                key=(result_row['seed'],row['candidate_seat'],row['environment_index'])
                assert key not in result;result[key]=row
        return result
    x,y=indexed(before),indexed(after);assert x.keys()==y.keys()
    blocks={seed:[] for seed,_,_ in x}
    for key in x:blocks[key[0]].append(float(y[key]['outcome']=='win')-float(x[key]['outcome']=='win'))
    block_means=[sum(v)/len(v) for v in blocks.values()];rng=random.Random(2026102405)
    draws=sorted(sum(rng.choice(block_means) for _ in block_means)/len(block_means) for _ in range(4000))
    valid=[key for key in x if x[key]['outcome']!='invalid' and y[key]['outcome']!='invalid']
    return dict(paired_attempts=len(x),paired_valid_games=len(valid),
        win_per_attempt_delta=sum(block_means)/len(block_means),
        paired_seed_block_bootstrap95=[draws[99],draws[3899]],seed_blocks=len(block_means),
        valid_pair_win_delta=(sum(float(y[k]['outcome']=='win')-float(x[k]['outcome']=='win') for k in valid)/len(valid)) if valid else None,
        scope='Four seed blocks only; invalid attempts retained as nonwins. Valid-pair result is secondary; no selection or invalid replacement.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','plan','training','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume-from',type=Path,help='Reuse only exit-zero complete batches after a collector-only repair; preserve original evidence.')
    a=p.parse_args();a.root=a.root.resolve();a.training=a.training.resolve();a.output=a.output.resolve()
    plan=read(a.plan);release=a.root/plan['runtime_release'];proof=read(a.training/'plain/verification.json');assert proof['passed'] and proof['process_exit_zero']
    assert not (a.training/'plain/train.exit.json').is_file() or read(a.training/'plain/train.exit.json')['returncode']==0
    models=dict(initial=release/'checkpoints/full-context.flax_model',plain=Path(proof['checkpoint']))
    assert sha(models['initial'])==plan['course_checkpoint_sha256'] and sha(models['plain'])==proof['checkpoint_sha256']
    history=release/'checkpoints/parent.flax_model';assert sha(history)==plan['history_checkpoint_sha256']
    frozen=read(release/'frozen-manifest.json');assert all(sha(release/e['path'])==e['sha256'] for e in frozen['artifacts'])
    a.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,JAX_PLATFORMS='cuda',JAX_DEFAULT_MATMUL_PRECISION='float32',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
        LD_LIBRARY_PATH='/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:'+str(a.root/'.venv/lib/python3.10/site-packages/nvidia/cudnn/lib'))
    jobs=[]
    for name,checkpoint in models.items():
        for opponent in ['history','greedy']:
            for seed in plan['evaluation']['seeds']:
                for seat in plan['evaluation']['seats']:
                    jobs.append(dict(model=name,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),opponent=opponent,seed=seed,seat=seat))
    previous=read(a.resume_from/'manifest.json') if a.resume_from else None
    if previous is not None:
        assert previous['plan']==plan and previous['jobs']==jobs and previous['training_proof']==proof
    write(a.output/'manifest.json',dict(plan=plan,plan_sha256=sha(a.plan),evaluator_sha256=sha(Path(__file__).with_name('eval_elfnote_formal.py')),jobs=jobs,training_proof=proof,started=time.time(),
        resumed_from=str(a.resume_from) if a.resume_from else None,previous_evaluator_sha256=previous['evaluator_sha256'] if previous else None,
        repair_scope='Terminal bot seat assertion only. Completed batches reused after validation; failed collector batch reruns same frozen conditions. No invalid episode replacement.' if previous else None))
    rows={name:{opponent:[] for opponent in ['history','greedy']} for name in models}
    for index,job in enumerate(jobs):
        folder=a.output/f'{index:02d}-{job["model"]}-{job["opponent"]}-{job["seed"]}-seat{job["seat"]}';folder.mkdir()
        old=a.resume_from/folder.name if a.resume_from else None
        if old and (old/'exit.json').exists() and read(old/'exit.json')['returncode']==0:
            import shutil
            row=read(old/'result.json')
            assert row['checkpoint_sha256']==job['checkpoint_sha256'] and row['seed']==job['seed'] and row['candidate_seat']==job['seat'] and row['opponent']==job['opponent']
            assert row['attempts']==plan['evaluation']['episodes_per_seed_seat'] and len(row['episodes'])==row['attempts']
            assert row['runtime_manifest_sha256']==sha(release/'frozen-manifest.json')
            assert len({x['environment_index'] for x in row['episodes']})==row['attempts'] and all(x['candidate_seat']==job['seat'] for x in row['episodes'])
            shutil.copy2(old/'result.json',folder/'result.json');write(folder/'reuse.json',dict(source=str(old),result_sha256=sha(old/'result.json'),exit_sha256=sha(old/'exit.json')))
            rows[job['model']][job['opponent']].append(row)
            print(json.dumps(dict(job=index,reused_verified_batch=True)),flush=True);continue
        argv=[sys.executable,'-u',str(Path(__file__).with_name('eval_elfnote_formal.py')),'--release',str(release),'--checkpoint',job['checkpoint'],'--opponent',job['opponent'],'--opponent-checkpoint',str(history),'--seed',str(job['seed']),'--seat',str(job['seat']),'--episodes',str(plan['evaluation']['episodes_per_seed_seat']),'--output',str(folder/'result.json')]
        with (folder/'run.log').open('x') as log:result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        write(folder/'exit.json',dict(returncode=result.returncode,command=argv));result.check_returncode()
        row=read(folder/'result.json');assert row['attempts']==plan['evaluation']['episodes_per_seed_seat'] and row['checkpoint_sha256']==job['checkpoint_sha256']
        assert len({x['environment_index'] for x in row['episodes']})==row['attempts'] and all(x['candidate_seat']==job['seat'] for x in row['episodes'])
        rows[job['model']][job['opponent']].append(row)
        print(json.dumps(dict(job=index,**job,attempts=row['attempts'],wins=row['wins'],invalid=row['invalid_games'])),flush=True)
    # Reuse the frozen validity and Wilson definitions, after all GPU children exit.
    sys.path.insert(0,str(release/'repo'))
    from ygoai.rl.match_report import summarize_matches
    summary={name:{opponent:summarize_matches([x for row in runs for x in row['episodes']]) for opponent,runs in by_opponent.items()} for name,by_opponent in rows.items()}
    differences={opponent:paired_summary(rows['initial'][opponent],rows['plain'][opponent]) for opponent in ['history','greedy']}
    assert all(sha(release/e['path'])==e['sha256'] for e in frozen['artifacts'])
    write(a.output/'summary.json',dict(passed=True,summary=summary,paired_differences=differences,
        scope='Fixed initial and plain PPO endpoints, unassisted full games; defense-teaching comparison and exercise retention remain separate.'))
    print(json.dumps(dict(passed=True,summary=summary,paired_differences=differences)),flush=True)


if __name__=='__main__':main()
