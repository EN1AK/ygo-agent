"""Exact-history combo corrections and independent successful-route retention."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
from scripts.collect_teaching_exercise import execute, load_native
from scripts.eval_capability_exercises import FrozenActor, run as synthetic_run
from scripts.teach_context_coverage import dataset, evaluate
from scripts.teach_correction_retention import exact_evaluate, write
from scripts.teach_three_capabilities import BRANCHES, audit_folder, save
from scripts.teach_material_transfer import material_dataset, admitted_correction, material_evaluate
from ygoai.rl.exercise_materials import material_opening
from ygoai.rl.exercise_core import sha
from ygoai.rl.exercise_retention import parent_kl_rows
from ygoai.rl.scene_batch import pack_scenes


def recorded(a,entry):
    path=a.material_study/entry['path']
    assert sha(path)==entry['sha256']
    return json.loads(path.read_text())


def collect_reserved(a,native,plan):
    folder=a.output/'reserved-references'; folder.mkdir(); rows=[]
    for lineup in plan['reserved_lineups']:
        for seat in (0,1):
            definition=material_opening(a.database,a.code_list,lineup,seat)
            roots=[]
            for branch in ('reference','revive_in_defense'):
                runs=[]
                for repeat in range(2):
                    r,obs=execute(native,definition,a.semantics,branch)
                    save(folder,f'{lineup["name"]}-seat{seat}-{branch}-{repeat}',r,obs); runs.append(r)
                assert runs[0]==runs[1] and r['result']['success']==(branch=='reference')
                roots.append(r['decisions'][r['roots']['position']]['observation_sha256'])
            assert roots[0]==roots[1]
            rows.append(dict(name=f'{lineup["name"]}-seat{seat}',definition=definition))
    audit_folder(a,folder)
    return rows


def admit_new(a,native,actor,old_params,plan):
    folder=a.output/'new-admission'; folder.mkdir(); corrections=[]; errors=[]
    for entry in plan['cases']:
        source=recorded(a,entry); step=entry['step']; root=source['decisions'][step]; hashes=[]
        assert not source['result']['success']
        for label,action,wanted in [('original',root['action'],False),('corrected',entry['action'],True)]:
            runs=[]
            for repeat in range(2):
                r,obs=execute(native,source['definition'],a.semantics,'reference',actor,'teacher',
                    replay_record=source,replay_until=step,root_action=action)
                assert r['result']['success']==wanted, (entry['name'],label,r['result'])
                errors.append(float(np.max(np.abs(np.array(r['decisions'][step]['logits'])-root['logits']))))
                save(folder,f'{entry["name"]}-{label}-{repeat}',r,obs); runs.append(r)
            assert runs[0]==runs[1]
            hashes.append(r['decisions'][step]['observation_sha256'])
            if wanted: corrections.append((r,obs))
        assert hashes[0]==hashes[1]
    parent=actor.params; actor.params=old_params
    source=recorded(a,plan['alternative']); assert source['result']['success']; runs=[]
    last=len(source['decisions'])-1
    for repeat in range(2):
        r,obs=execute(native,source['definition'],a.semantics,'reference',actor,'teacher',
            replay_record=source,replay_until=last,root_action=source['decisions'][last]['action'])
        assert r['result']['success']
        assert len(r['decisions'])==len(source['decisions'])
        for actual,expected in zip(r['decisions'],source['decisions']):
            errors.append(float(np.max(np.abs(np.array(actual['logits'])-expected['logits']))))
        save(folder,f'alternative-retention-{repeat}',r,obs); runs.append(r)
    assert runs[0]==runs[1]; actor.params=parent
    alternative=copy.deepcopy(r)
    for d in alternative['decisions']:
        d['teaching']=d['player']==alternative['definition']['controlled'] and d['step']>=alternative['roots']['opening']
    assert max(errors)<.002
    audit_folder(a,folder)
    write(folder/'admission.json',dict(passed=True,source_logit_max_error=max(errors),
        audit_sha256=sha(folder/'independent-audit.json'),cases=plan['cases'],alternative=plan['alternative'],
        scope='exact-root positive demonstrations; fixed original continuation failure, not proof of no alternative solution'))
    return corrections,(alternative,obs)


def extra_evaluate(a,native,actor,folder,plan,reserved,full):
    folder.mkdir(); results=[]
    requests=[(e['name'],recorded(a,e),e['step'],'exact') for e in plan['cases']]
    requests += [('alternative-prefix',recorded(a,plan['alternative']),plan['alternative']['takeover_step'],'alternative')]
    if full: requests += [(r['name'],r['definition'],None,'reserved') for r in reserved]
    for name,source,step,kind in requests:
        runs=[]
        for repeat in range(2 if full else 1):
            kwargs=dict(replay_record=source,replay_until=step) if step is not None else {}
            definition=source['definition'] if step is not None else source
            r,obs=execute(native,definition,a.semantics,'reference',actor,'correction' if step is not None else 'opening',**kwargs)
            save(folder,f'{name}-{repeat}',r,obs); runs.append(r)
        if full: assert runs[0]==runs[1]
        result=dict(name=name,kind=kind,**r['result'])
        if kind=='exact':
            entry=next(e for e in plan['cases'] if e['name']==name)
            logits=np.array(r['decisions'][step]['logits']); probs=np.exp(logits-logits.max()); probs/=probs.sum()
            result.update(demonstrated_action_probability=float(probs[entry['action']]),root_action=r['decisions'][step]['action'])
        results.append(result)
    if full: audit_folder(a,folder)
    write(folder/'summary.json',results)
    print(json.dumps(dict(stage=str(folder),results=results)),flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','fit'],default='preflight')
    for key in ('release','checkpoint','retention-checkpoint','plan','data','material-data','material-study','previous','failure','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path): setattr(a,k,v.resolve())
    a.native=a.release/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    a.database=a.release/'repo/assets/locale/zh/cards.cdb'; a.scripts=a.release/'repo/scripts/script'
    a.code_list=a.release/'code_list.txt'; a.semantics=a.release/'semantics'
    plan=json.loads(a.plan.read_text()); start=time.monotonic()
    for path,key in [(a.checkpoint,'parent_sha256'),(a.retention_checkpoint,'retention_parent_sha256'),(a.native,'native_sha256')]:
        assert sha(path)==plan[key]
    assert len(plan['cases'])==3 and len(plan['reserved_lineups'])==4
    protected={str(a.release/r['path']):r['sha256'] for r in json.loads((a.release/'manifest.json').read_text())['artifacts']}
    for path in (a.plan,a.checkpoint,a.retention_checkpoint,Path(str(a.checkpoint)+'.metadata.json'),
                 Path(str(a.retention_checkpoint)+'.metadata.json'),a.failure,a.data/'dataset.json',
                 a.previous/'admission/admission.json',a.material_data/'dataset.json'):
        protected[str(path)]=sha(path)
    for entry in plan['cases']+[plan['alternative']]: protected[str(a.material_study/entry['path'])]=entry['sha256']
    assert all(sha(p)==v for p,v in protected.items())
    native=load_native(a,a.release/'opening-02/registration.ydk')
    rows=dataset(a); actual_plan=a.plan
    a.plan=Path(__file__).resolve().parents[1]/'assets/exercises/teaching-v1/material-transfer.json'
    materials=material_dataset(a); protected[str(a.plan)]=sha(a.plan); a.plan=actual_plan
    corrected=admitted_correction(a)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'manifest.json',dict(plan=plan,protected=protected,optimizer_enabled=a.mode=='fit',source_sha256=sha(__file__)))
    reserved=collect_reserved(a,native,plan)
    import jax
    import jax.numpy as jnp
    import optax
    import flax.serialization
    from ygoai.rl.jax.agent import ModelArgs
    from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
    jax.config.update('jax_default_matmul_precision','highest')
    assert any(d.platform=='gpu' for d in jax.devices())
    sample={k:v[:1] for k,v in rows['base','combo'][1].items()}
    actor=FrozenActor(a.checkpoint,a.code_list,a.semantics,sample)
    old_actor=FrozenActor(a.retention_checkpoint,a.code_list,a.semantics,sample)
    parent=actor.params; old_params=old_actor.params
    infer=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3])
    actor.forward=lambda obs,state:infer(actor.params,obs,state)
    corrections,alternative=admit_new(a,native,actor,old_params,plan)
    folder=a.output/'admission'; folder.mkdir(); records=[]
    for repeat in range(2):
        r,obs=synthetic_run(a,native,actor,'combo','reference','policy')
        assert r['status']=='success'; save(folder,f'synthetic-retention-{repeat}',r,obs); records.append(r)
    assert records[0]==records[1]; audit_folder(a,folder)
    r=copy.deepcopy(r); r['definition']=dict(controlled=0); rehearsal=(r,obs)
    @jax.jit
    def sequence(params,batch):
        return actor.agent.apply(params,batch,actor.agent.init_rnn_state(10),jnp.zeros(plan['sequence_steps']*10,bool),None)[1]
    batches=[]; errors=[]; counts=[]
    for i in range(12):
        profile=plan['profiles'][i%6]; lineup=plan['fit_lineups'][i%4]
        chosen=[rows[profile,kind] for kind in BRANCHES]+[corrected,rehearsal]+[materials[lineup,seat] for seat in (0,1)]+[corrections[i%3],alternative]
        tensors,targets,weights,legal=pack_scenes(chosen,plan['sequence_steps'])
        batch=jax.tree.map(jnp.asarray,tensors); anchors=[]
        for params,columns in [(parent,range(9)),(old_params,[9])]:
            actor.params=params; anchor=sequence(params,batch); assert np.isfinite(np.asarray(anchor)).all(); anchors.append(anchor)
            for b in columns:
                record,obs=chosen[b]; actor.reset(); own=0
                for t,d in enumerate(record['decisions']):
                    logits,_,_=actor.predict({k:v[t:t+1] for k,v in obs.items()},d['player'])
                    if d['player']==record['definition']['controlled']:
                        n=len(d['menu']); errors.append(float(np.abs(logits[:n]-np.asarray(anchor)[10*own+b,:n]).max())); own+=1
        anchor=anchors[0].reshape(plan['sequence_steps'],10,-1)
        anchor=anchor.at[:,9,:].set(anchors[1].reshape(plan['sequence_steps'],10,-1)[:,9,:]).reshape(-1,legal.shape[-1])
        batches.append((batch,jnp.asarray(targets),jnp.asarray(weights),jnp.asarray(legal),jax.lax.stop_gradient(anchor)))
        counts.append(dict(profile=profile,lineup=lineup,correction=plan['cases'][i%3]['name'],active=(weights>0).sum(0).tolist()))
    actor.params=parent; assert max(errors)<.002
    write(a.output/'sequence-parity.json',dict(passed=True,max_error=max(errors),checked_columns=120,batches=counts))
    write(a.output/'collection-cost.json',dict(seconds=time.monotonic()-start))
    if a.mode=='preflight':
        assert all(sha(p)==v for p,v in protected.items()); print('Preflight passed, optimizer disabled',flush=True); return
    source=json.loads(a.failure.read_text()); exact_plan=dict(correction_step=corrected[0]['exact_history']['root_step'])
    def full_evaluate(folder):
        folder.mkdir()
        old=evaluate(a,native,actor,folder/'old',rows,True)
        exact=exact_evaluate(a,native,actor,folder/'exact',source,exact_plan)
        material=material_evaluate(a,native,actor,folder/'materials',materials,True)
        extra=extra_evaluate(a,native,actor,folder/'extra',plan,reserved,True)
        result=dict(old=old,exact=exact,materials=material,extra=extra); write(folder/'summary.json',result); return result
    before=full_evaluate(a.output/'before'); summaries=[]
    for arm in plan['arms']:
        folder=a.output/arm['name']; folder.mkdir(); arm_start=time.monotonic(); actor.params=parent
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(plan['learning_rate'])); state=tx.init(parent)
        def objective(params,batch,targets,weights,legal,anchor):
            logits=sequence(params,batch)
            ce=optax.softmax_cross_entropy_with_integer_labels(logits,targets).reshape(-1,10)
            kl=parent_kl_rows(logits,anchor,legal).reshape(-1,10)
            ces=(weights*ce).sum(0); kls=(weights*kl).sum(0)
            value=ces[:4].mean()+plan['main_kl_coef']*kls[:4].mean()+ces[4]+kls[5]+ces[6:8].mean()
            value+=arm['correction_coef']*ces[8]+arm['route_coef']*kls[9]
            return value,(ces,kls)
        @jax.jit
        def update(params,state,*batch):
            (value,terms),grad=jax.value_and_grad(objective,has_aux=True)(params,*batch)
            changes,next_state=tx.update(grad,state,params); candidate=optax.apply_updates(params,changes)
            finite=jnp.all(jnp.stack([jnp.isfinite(v).all() for v in jax.tree.leaves(candidate)]))
            return candidate,next_state,value,terms,optax.global_norm(grad),finite
        with (folder/'metrics.jsonl').open('x') as f:
            for step in range(1,plan['updates_per_arm']+1):
                assert time.monotonic()-start<plan['max_total_seconds'] and time.monotonic()-arm_start<plan['max_seconds_per_arm']
                index=(step-1)%12
                candidate,next_state,value,terms,norm,finite=update(actor.params,state,*batches[index])
                row=dict(update=step,batch_index=index,loss=float(value),teacher_ce=np.asarray(terms[0]).tolist(),parent_kl=np.asarray(terms[1]).tolist(),grad_norm=float(norm))
                assert bool(finite) and np.isfinite([row['loss'],row['grad_norm']]+row['teacher_ce']+row['parent_kl']).all()
                actor.params,state=candidate,next_state; f.write(json.dumps(row)+'\n'); f.flush()
                if step in plan['probe_updates']:
                    evaluate(a,native,actor,folder/f'old-probe-{step}',rows,False)
                    material_evaluate(a,native,actor,folder/f'material-probe-{step}',materials,False)
                    extra_evaluate(a,native,actor,folder/f'extra-probe-{step}',plan,reserved,False)
                    print(json.dumps(dict(arm=arm['name'],**row)),flush=True)
        checkpoint=folder/'model.flax_model'; checkpoint.write_bytes(flax.serialization.to_bytes(actor.params))
        write_checkpoint_metadata(checkpoint,observation_schema='structured-lite-v1',model_args=ModelArgs(observation_schema='structured-lite-v1'),
            code_list_hash=sha(a.code_list),semantic_table_hash=sha(a.semantics/'metadata.json'),
            capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),
            training_context=dict(experiment=plan['id'],parent_sha256=sha(a.checkpoint),retention_sha256=sha(a.retention_checkpoint),arm=arm,updates=plan['updates_per_arm'],promotion_eligible=False))
        actor.params=flax.serialization.from_bytes(parent,checkpoint.read_bytes())
        after=full_evaluate(folder/'endpoint')
        summary=dict(arm=arm,results=after,checkpoint_sha256=sha(checkpoint),seconds=time.monotonic()-arm_start)
        write(folder/'summary.json',summary); summaries.append(summary)
    assert all(sha(p)==v for p,v in protected.items()); dataset(a); admitted_correction(a)
    write(a.output/'summary.json',dict(before=before,arms=summaries,seconds=time.monotonic()-start,protected_unchanged=True,promotion_eligible=False))
    print(json.dumps(dict(completed=True,seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__': main()
