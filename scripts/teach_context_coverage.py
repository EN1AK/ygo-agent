"""Bounded audited comparison of narrow and expanded legal-history coverage."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scripts.teach_three_capabilities import BRANCHES, DEPTHS, audit_folder, save
from scripts.collect_teaching_exercise import execute,load_native
from scripts.eval_capability_exercises import FrozenActor,run as synthetic_run
from ygoai.rl.exercise_contexts import PROFILES,context_opening
from ygoai.rl.exercise_starters import CASES
from ygoai.rl.exercise_core import sha
from ygoai.rl.scene_batch import pack_scenes
from ygoai.rl.exercise_retention import parent_kl_rows


def collect(a,native,plan):
    start=time.monotonic(); a.data.mkdir(parents=True,exist_ok=False); proofs=[]
    assert len(PROFILES)*len(BRANCHES)==plan['max_candidates']
    for profile in PROFILES:
        for kind,branches in BRANCHES.items():
            definition=context_opening(a.database,a.code_list,kind,profile); roots=[]
            for branch in branches:
                records=[]
                for repeat in range(2):
                    assert time.monotonic()-start<plan['collection_max_seconds']
                    record,arrays=execute(native,definition,a.semantics,branch)
                    path=save(a.data,f'{profile["name"]}-{kind}-{branch}-{repeat}',record,arrays)
                    records.append(record)
                    if repeat==0: proof=dict(path=path.name,sha256=sha(path),observations_sha256=sha(path.with_suffix('.npz')))
                assert records[0]==records[1], 'reference nondeterminism'
                assert record['result']['success']==(branch==branches[0]), (profile,kind,branch,record['result'])
                roots.append(record['decisions'][record['roots'][DEPTHS[kind][-1]]]['observation_sha256'])
                proofs.append(dict(profile=profile['name'],split=profile['split'],kind=kind,branch=branch,positive=branch==branches[0],**proof))
            assert len(set(roots))==1, ('contrast root mismatch',profile,kind)
            print(json.dumps(dict(collected=profile['name'],kind=kind)),flush=True)
    audits={r['path']:r for r in audit_folder(a,a.data)['results']}
    assert all(audits[r['path']]['passed'] and audits[r['path']]['sha256']==r['sha256'] for r in proofs)
    with (a.data/'dataset.json').open('x') as f:
        json.dump(dict(passed=True,profiles=PROFILES,results=proofs,seconds=time.monotonic()-start,
                       native_sha256=sha(a.native),unseen_families=[],source_sha256=sha(__file__)),f,indent=2)


def dataset(a):
    manifest=json.loads((a.data/'dataset.json').read_text())
    assert manifest['passed'] and manifest['native_sha256']==sha(a.native) and manifest['profiles']==PROFILES
    rows={}
    for r in manifest['results']:
        p=a.data/r['path']
        assert sha(p)==r['sha256'] and sha(p.with_suffix('.npz'))==r['observations_sha256']
        if r['positive']:
            with np.load(p.with_suffix('.npz'),allow_pickle=False) as data: obs=dict(data)
            record=json.loads(p.read_text()); assert record['normal_opening'] and record['result']['success']
            rows[r['profile'],r['kind']]=(record,obs)
    assert len(rows)==36
    return rows


def evaluate(a,native,actor,folder,rows,full):
    folder.mkdir(); results=[]
    for profile in PROFILES:
        if not full and profile['name'] not in ('base','seat-low'): continue
        for kind,branches in BRANCHES.items():
            definition=rows[profile['name'],kind][0]['definition']
            depths=DEPTHS[kind] if full and profile['name']=='base' else ('opening',)
            for depth in depths:
                records=[]
                for repeat in range(2 if full else 1):
                    r,arrays=execute(native,definition,a.semantics,branches[0],actor,depth)
                    save(folder,f'{profile["name"]}-{kind}-{depth}-{repeat}',r,arrays); records.append(r)
                if full: assert records[0]==records[1]
                results.append(dict(profile=profile['name'],split=profile['split'],kind=kind,depth=depth,**r['result']))
    if full:
        for case,d in CASES.items():
            records=[]
            for repeat in range(2):
                r,arrays=synthetic_run(a,native,actor,case,next(iter(d['branches'])),'policy')
                assert r['status'] in ('success','verified_failure'),r
                save(folder,f'synthetic-{case}-{repeat}',r,arrays); records.append(r)
            assert records[0]==records[1]
            results.append(dict(kind='synthetic',case=case,success=r['status']=='success',status=r['status']))
        audit_folder(a,folder)
    with (folder/'summary.json').open('x') as f: json.dump(results,f,indent=2)
    print(json.dumps(dict(stage=str(folder),passed=sum(r['success'] for r in results),total=len(results),
                          failures=[r for r in results if not r['success']])),flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['collect','preflight','fit'],default='preflight')
    for name in ('release','checkpoint','plan','data','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path): setattr(a,k,v.resolve())
    a.native=a.release/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    a.database=a.release/'repo/assets/locale/zh/cards.cdb'; a.scripts=a.release/'repo/scripts/script'
    a.code_list=a.release/'code_list.txt'; a.semantics=a.release/'semantics'
    plan=json.loads(a.plan.read_text()); start=time.monotonic()
    assert sha(a.checkpoint)==plan['parent_sha256'] and sha(a.native)==plan['native_sha256']
    protected={str(a.release/r['path']):r['sha256'] for r in json.loads((a.release/'manifest.json').read_text())['artifacts']}
    protected.update({str(p):sha(p) for p in (a.plan,a.checkpoint,Path(str(a.checkpoint)+'.metadata.json'))})
    assert all(sha(p)==v for p,v in protected.items())
    native=load_native(a,a.release/'opening-02/registration.ydk')
    if a.mode=='collect':
        collect(a,native,plan)
        assert all(sha(p)==v for p,v in protected.items()); return
    rows=dataset(a); a.output.mkdir(parents=True,exist_ok=False)
    protected[str(a.data/'dataset.json')]=sha(a.data/'dataset.json')
    with (a.output/'manifest.json').open('x') as f:
        json.dump(dict(plan=plan,protected=protected,optimizer_enabled=a.mode=='fit',source_sha256=sha(__file__)),f,indent=2)
    import jax
    import jax.numpy as jnp
    import optax
    import flax.serialization
    from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
    from ygoai.rl.jax.agent import ModelArgs
    jax.config.update('jax_default_matmul_precision','highest')
    actor=FrozenActor(a.checkpoint,a.code_list,a.semantics,{k:v[:1] for k,v in rows['base','combo'][1].items()})
    parent=actor.params
    infer=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3])
    actor.forward=lambda obs,state:infer(actor.params,obs,state)
    @jax.jit
    def sequence(params,batch):
        return actor.agent.apply(params,batch,actor.agent.init_rnn_state(4),jnp.zeros(plan['sequence_steps']*4,bool),None)[1]
    batches={}; errors=[]
    # Reserved development combinations never enter the learner or parent targets.
    for profile in [p for p in PROFILES if p['split']=='fit']:
        chosen=[rows[profile['name'],kind] for kind in BRANCHES]
        tensors,targets,weights,legal=pack_scenes(chosen,plan['sequence_steps'])
        batch=jax.tree.map(jnp.asarray,tensors)
        anchor=jax.lax.stop_gradient(sequence(parent,batch)); assert np.isfinite(np.asarray(anchor)).all()
        batches[profile['name']]=(batch,jnp.asarray(targets),jnp.asarray(weights),jnp.asarray(legal),anchor)
        for b,(record,obs) in enumerate(chosen):
            actor.reset(); own=0
            for i,d in enumerate(record['decisions']):
                logits,_,_=actor.predict({k:v[i:i+1] for k,v in obs.items()},d['player'])
                if d['player']==record['definition']['controlled']:
                    n=len(d['menu']); errors.append(float(np.abs(logits[:n]-np.asarray(anchor)[4*own+b,:n]).max())); own+=1
    assert max(errors)<.002
    with (a.output/'sequence-parity.json').open('x') as f: json.dump(dict(passed=True,max_error=max(errors),scenes=24),f)
    if a.mode=='preflight': print('Preflight passed, optimizer disabled',flush=True); return
    before=evaluate(a,native,actor,a.output/'before',rows,True); summaries=[]
    for arm in plan['arms']:
        folder=a.output/arm['name']; folder.mkdir(); arm_start=time.monotonic(); actor.params=parent
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(plan['learning_rate'])); state=tx.init(parent)
        def objective(params,batch,targets,weights,legal,anchor):
            logits=sequence(params,batch)
            ce=optax.softmax_cross_entropy_with_integer_labels(logits,targets).reshape(-1,4)
            kl=parent_kl_rows(logits,anchor,legal).reshape(-1,4)
            losses=(weights*ce).sum(0); kls=(weights*kl).sum(0)
            return losses.mean()+arm['parent_kl_coef']*kls.mean(),(losses,kls)
        @jax.jit
        def update(params,state,*batch):
            (value,terms),grad=jax.value_and_grad(objective,has_aux=True)(params,*batch)
            changes,next_state=tx.update(grad,state,params); candidate=optax.apply_updates(params,changes)
            finite=jnp.all(jnp.stack([jnp.isfinite(v).all() for v in jax.tree.leaves(candidate)]))
            return candidate,next_state,value,terms,optax.global_norm(grad),finite
        with (folder/'metrics.jsonl').open('x') as f:
            for step in range(1,plan['updates_per_arm']+1):
                assert time.monotonic()-start<plan['max_total_seconds'] and time.monotonic()-arm_start<plan['max_seconds_per_arm']
                profile=arm['profiles'][(step-1)%len(arm['profiles'])]
                candidate,next_state,value,terms,norm,finite=update(actor.params,state,*batches[profile])
                row=dict(update=step,profile=profile,loss=float(value),teacher_ce=np.asarray(terms[0]).tolist(),
                         parent_kl=np.asarray(terms[1]).tolist(),grad_norm=float(norm))
                assert bool(finite) and np.isfinite([row['loss'],row['grad_norm']]+row['teacher_ce']+row['parent_kl']).all()
                actor.params,state=candidate,next_state; f.write(json.dumps(row)+'\n'); f.flush()
                if step in plan['probe_updates']:
                    evaluate(a,native,actor,folder/f'probe-{step}',rows,False)
                    print(json.dumps(dict(arm=arm['name'],**row)),flush=True)
        checkpoint=folder/'model.flax_model'; checkpoint.write_bytes(flax.serialization.to_bytes(actor.params))
        write_checkpoint_metadata(checkpoint,observation_schema='structured-lite-v1',model_args=ModelArgs(observation_schema='structured-lite-v1'),
            code_list_hash=sha(a.code_list),semantic_table_hash=sha(a.semantics/'metadata.json'),
            capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),
            training_context=dict(experiment=plan['id'],parent_sha256=sha(a.checkpoint),arm=arm,updates=plan['updates_per_arm'],promotion_eligible=False))
        actor.params=flax.serialization.from_bytes(parent,checkpoint.read_bytes())
        after=evaluate(a,native,actor,folder/'endpoint',rows,True)
        summary=dict(arm=arm,results=after,all_scenes_passed=all(r['success'] for r in after),
                     checkpoint_sha256=sha(checkpoint),seconds=time.monotonic()-arm_start)
        with (folder/'summary.json').open('x') as f: json.dump(summary,f,indent=2)
        summaries.append(summary)
    assert all(sha(p)==v for p,v in protected.items()); dataset(a)
    result=dict(before=before,arms=summaries,protected_unchanged=True,seconds=time.monotonic()-start,promotion_eligible=False)
    with (a.output/'summary.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(dict(completed=True,seconds=result['seconds'],passed=[s['all_scenes_passed'] for s in summaries])),flush=True)


if __name__=='__main__': main()
