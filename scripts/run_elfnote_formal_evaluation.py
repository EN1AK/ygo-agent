"""Run the frozen post-training matrix serially, retaining every invalid attempt."""
import argparse,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)


def verified_endpoint(training,role,plan,runtime_hash):
    proof=read(training/role/'verification.json');launch=read(training/role/'launch-plan.json')
    assert proof['passed'] and proof['process_exit_zero'] and proof['zero_nonfinite_updates']
    assert proof['fitted_env_steps']==plan['total_timesteps_per_arm']
    assert read(training/role/'train.exit.json')['returncode']==0
    assert launch['phase']=='train' and launch['arm']==role and launch['runtime_manifest_sha256']==runtime_hash
    for key in ['seed','total_timesteps_per_arm','local_num_envs','num_actor_threads','local_env_threads','num_steps','num_minibatches','learning_rate','update_epochs']:
        assert launch['plan'][key]==plan[key],key
    if role=='after_defense':assert launch['checkpoint_sha256']==plan['arms'][role]['checkpoint_sha256']
    checkpoint=Path(proof['checkpoint']);assert sha(checkpoint)==proof['checkpoint_sha256']
    return checkpoint,proof


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
    p.add_argument('--rehearsal',action='store_true',help='Evaluate both frozen alternating-rehearsal roles; neither may be dropped.')
    p.add_argument('--arm',choices=['plain','after_defense'],default='plain')
    a=p.parse_args();a.root=a.root.resolve();a.training=a.training.resolve();a.output=a.output.resolve()
    plan=read(a.plan);release=a.root/plan['runtime_release']
    models=dict(initial=release/'checkpoints/full-context.flax_model')
    assert not a.rehearsal or a.arm=='plain'
    if a.rehearsal:
        assert plan['roles']==['control','rehearse'];proof={}
        for role in plan['roles']:
            pp=read(a.training/role/'verification.json');rp=read(a.training/role/'rehearsal-verification.json')
            assert pp['passed'] and pp['process_exit_zero'] and pp['fitted_env_steps']==plan['total_timesteps_per_arm']
            assert rp['passed'] and rp['role']==role and rp['model']==pp['checkpoint_sha256']
            assert sha(a.training/('rehearsal-'+role+'.json'))==rp['config_sha256']
            assert read(a.training/role/'train.exit.json')['returncode']==0
            models[role]=Path(pp['checkpoint']);assert sha(models[role])==pp['checkpoint_sha256']
            proof[role]=dict(ppo=pp,rehearsal=rp)
    elif a.arm=='plain':
        models['plain'],proof=verified_endpoint(a.training,'plain',plan,sha(release/'frozen-manifest.json'))
    else:
        models['plain'],plain=verified_endpoint(a.root/plan['plain_comparison_study'],'plain',plan,sha(release/'frozen-manifest.json'))
        assert sha(models['plain'])==plan['plain_comparison_checkpoint_sha256']
        models['after_defense'],trained=verified_endpoint(a.training,'after_defense',plan,sha(release/'frozen-manifest.json'))
        spec=plan['arms']['after_defense'];study=a.root/spec['study'];teacher=read(study/'final-verification.json')
        assert teacher['passed'] and teacher['defensive_policy_gate'] and Path(str(study)+'.train.exit').read_text().strip()=='0'
        models['defense_initial']=study/spec['arm']/'model.flax_model'
        assert sha(models['defense_initial'])==spec['checkpoint_sha256']==teacher['checkpoints'][spec['arm']]
        proof=dict(plain=plain,after_defense=trained,defense_initial=teacher)
    assert sha(models['initial'])==plan['course_checkpoint_sha256']
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
    cross=None
    if a.rehearsal:
        cross_rows=[]
        for seed in plan['evaluation']['seeds']:
            for seat in plan['evaluation']['seats']:
                folder=a.output/f'cross-rehearse-control-{seed}-seat{seat}';folder.mkdir()
                argv=[sys.executable,'-u',str(release/'repo/scripts/eval_mixed_match.py'),'--checkpoint',str(models['rehearse']),
                    '--opponent-checkpoint',str(models['control']),'--output',str(folder/'result.json'),'--seed',str(seed),
                    '--num-episodes',str(plan['evaluation']['episodes_per_seed_seat']),'--candidate-seat',str(seat),'--env-threads','4',
                    '--deck',str(release/'decks/replay-elfnote.ydk'),'--code-list-file',str(release/'code_list.txt'),
                    '--semantic-asset-dir',str(release/'semantics'),'--max-options','128','--max-steps','1000',
                    '--observation-schema','structured-lite-v1','--checkpoint-schema','structured-lite-v1','--opponent-schema','structured-lite-v1']
                with (folder/'run.log').open('x') as log:result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
                write(folder/'exit.json',dict(returncode=result.returncode,command=argv));result.check_returncode()
                row=read(folder/'result.json')
                assert row['checkpoint_sha256']==sha(models['rehearse']) and row['opponent_checkpoint_sha256']==sha(models['control'])
                assert row['attempts']==plan['evaluation']['episodes_per_seed_seat'] and len(row['episodes'])==row['attempts']
                assert len({x['environment_index'] for x in row['episodes']})==row['attempts'] and all(x['candidate_seat']==seat for x in row['episodes'])
                cross_rows.extend(row['episodes']);print(json.dumps(dict(cross_arm=True,seed=seed,seat=seat,wins=row['wins'],invalid=row['invalid_games'])),flush=True)
        cross=summarize_matches(cross_rows)
    if a.rehearsal:
        differences={name:{opponent:paired_summary(rows[left][opponent],rows[right][opponent]) for opponent in ['history','greedy']}
            for name,left,right in [('control_vs_initial','initial','control'),('rehearse_vs_initial','initial','rehearse'),('rehearse_vs_control','control','rehearse')]}
    elif a.arm=='after_defense':
        pairs=[('plain_vs_initial','initial','plain'),('defense_initial_vs_initial','initial','defense_initial'),
            ('after_defense_vs_defense_initial','defense_initial','after_defense'),('after_defense_vs_plain','plain','after_defense'),('after_defense_vs_initial','initial','after_defense')]
        differences={name:{opponent:paired_summary(rows[left][opponent],rows[right][opponent]) for opponent in ['history','greedy']} for name,left,right in pairs}
    else:differences={opponent:paired_summary(rows['initial'][opponent],rows['plain'][opponent]) for opponent in ['history','greedy']}
    assert all(sha(release/e['path'])==e['sha256'] for e in frozen['artifacts'])
    write(a.output/'summary.json',dict(passed=True,summary=summary,paired_differences=differences,cross_arm=cross,
        scope='Fixed initial and both alternating-rehearsal roles, unassisted full games; exercise retention remains separate.' if a.rehearsal else ('Fixed course parent, ordinary PPO, defense-teaching start and after-defense PPO endpoints; teaching and PPO gains reported separately, same-runtime paired games, exercise retention separate.' if a.arm=='after_defense' else 'Fixed initial and plain PPO endpoints, unassisted full games; exercise retention separate.')))
    print(json.dumps(dict(passed=True,summary=summary,paired_differences=differences,cross_arm=cross)),flush=True)


if __name__=='__main__':main()
