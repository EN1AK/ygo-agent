"""Audited normal-opening material transfer with fixed untouched combinations."""
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
from ygoai.rl.exercise_materials import LINEUPS, material_opening
from ygoai.rl.exercise_core import sha
from ygoai.rl.exercise_retention import parent_kl_rows
from ygoai.rl.scene_batch import pack_scenes


def collect(a,native,plan):
    start=time.monotonic(); a.material_data.mkdir(parents=True,exist_ok=False); proofs=[]
    assert len(LINEUPS)*2==plan['max_candidates']
    for lineup in LINEUPS:
        for seat in (0,1):
            definition=material_opening(a.database,a.code_list,lineup,seat); roots=[]
            for branch in ('reference','revive_in_defense'):
                records=[]
                for repeat in range(2):
                    assert time.monotonic()-start<plan['collection_max_seconds']
                    record,arrays=execute(native,definition,a.semantics,branch)
                    path=save(a.material_data,f'{lineup["name"]}-seat{seat}-{branch}-{repeat}',record,arrays)
                    records.append(record)
                    if repeat==0: proof=dict(path=path.name,sha256=sha(path),observations_sha256=sha(path.with_suffix('.npz')))
                assert records[0]==records[1]
                assert record['result']['success']==(branch=='reference'), (lineup,seat,branch,record['result'])
                roots.append(record['decisions'][record['roots']['position']]['observation_sha256'])
                proofs.append(dict(lineup=lineup['name'],seat=seat,split=lineup['split'],positive=branch=='reference',**proof))
            assert len(set(roots))==1
            print(json.dumps(dict(collected=lineup['name'],seat=seat)),flush=True)
    audited={r['path']:r for r in audit_folder(a,a.material_data)['results']}
    assert all(audited[r['path']]['sha256']==r['sha256'] and audited[r['path']]['passed'] for r in proofs)
    write(a.material_data/'dataset.json',dict(passed=True,native_sha256=sha(a.native),lineups=LINEUPS,
        plan_sha256=sha(a.plan),results=proofs,seconds=time.monotonic()-start))


def material_dataset(a):
    manifest=json.loads((a.material_data/'dataset.json').read_text())
    assert manifest['passed'] and manifest['native_sha256']==sha(a.native) and manifest['lineups']==LINEUPS
    assert manifest['plan_sha256']==sha(a.plan)
    rows={}
    for r in manifest['results']:
        p=a.material_data/r['path']; assert sha(p)==r['sha256'] and sha(p.with_suffix('.npz'))==r['observations_sha256']
        if r['positive']:
            record=json.loads(p.read_text()); assert record['normal_opening'] and record['result']['success']
            with np.load(p.with_suffix('.npz'),allow_pickle=False) as data: rows[r['lineup'],r['seat']]=(record,dict(data))
    assert len(rows)==16
    return rows


def material_evaluate(a,native,actor,folder,rows,full):
    folder.mkdir(); results=[]
    for lineup in LINEUPS:
        if not full and lineup['split']!='fit': continue
        for seat in (0,1):
            definition=rows[lineup['name'],seat][0]['definition']
            for depth in (('opening','combo','position') if full else ('opening',)):
                records=[]
                for repeat in range(2 if full else 1):
                    r,arrays=execute(native,definition,a.semantics,'reference',actor,depth)
                    save(folder,f'{lineup["name"]}-seat{seat}-{depth}-{repeat}',r,arrays); records.append(r)
                if full: assert records[0]==records[1]
                results.append(dict(lineup=lineup['name'],seat=seat,split=lineup['split'],depth=depth,**r['result']))
    if full: audit_folder(a,folder)
    write(folder/'summary.json',results)
    print(json.dumps(dict(stage=str(folder),passed=sum(r['success'] for r in results),total=len(results),
                         failures=[r for r in results if not r['success']])),flush=True)
    return results


def admitted_correction(a):
    folder=a.previous/'admission'; proof=json.loads((folder/'admission.json').read_text())
    assert proof['passed'] and proof['audit_sha256']==sha(folder/'independent-audit.json')
    assert proof['source_sha256']==sha(a.failure)
    audit=json.loads((folder/'independent-audit.json').read_text()); assert audit['passed'] and audit['native_sha256']==sha(a.native)
    r=next(r for r in audit['results'] if r['path']=='exact-stop-0.json'); p=folder/r['path']
    assert sha(p)==r['sha256'] and sha(p.with_suffix('.npz'))==r['observations_sha256']
    record=json.loads(p.read_text()); assert record['result']['success'] and record['exact_history']['prefix_verified']
    with np.load(p.with_suffix('.npz'),allow_pickle=False) as arrays: return record,dict(arrays)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['collect','preflight','fit'],default='preflight')
    for key in ('release','checkpoint','plan','data','material-data','previous','failure','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path): setattr(a,k,v.resolve())
    a.native=a.release/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    a.database=a.release/'repo/assets/locale/zh/cards.cdb'; a.scripts=a.release/'repo/scripts/script'
    a.code_list=a.release/'code_list.txt'; a.semantics=a.release/'semantics'
    plan=json.loads(a.plan.read_text()); start=time.monotonic()
    assert sha(a.checkpoint)==plan['parent_sha256'] and sha(a.native)==plan['native_sha256']
    assert [r['name'] for r in LINEUPS if r['split']=='fit']==plan['fit_lineups']
    assert [r['name'] for r in LINEUPS if r['split']=='reserved-combination']==plan['reserved_combinations']
    assert [r['name'] for r in LINEUPS if r['split']=='reserved-material']==plan['reserved_materials']
    protected={str(a.release/r['path']):r['sha256'] for r in json.loads((a.release/'manifest.json').read_text())['artifacts']}
    for path in (a.plan,a.checkpoint,Path(str(a.checkpoint)+'.metadata.json'),a.failure,a.data/'dataset.json'):
        protected[str(path)]=sha(path)
    assert all(sha(p)==v for p,v in protected.items())
    native=load_native(a,a.release/'opening-02/registration.ydk')
    if a.mode=='collect':
        collect(a,native,plan); assert all(sha(p)==v for p,v in protected.items()); return
    rows=dataset(a); materials=material_dataset(a); corrected=admitted_correction(a)
    protected[str(a.material_data/'dataset.json')]=sha(a.material_data/'dataset.json')
    protected[str(a.previous/'admission/admission.json')]=sha(a.previous/'admission/admission.json')
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'manifest.json',dict(plan=plan,protected=protected,optimizer_enabled=a.mode=='fit',source_sha256=sha(__file__)))
    import jax
    import jax.numpy as jnp
    import optax
    import flax.serialization
    from ygoai.rl.jax.agent import ModelArgs
    from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
    jax.config.update('jax_default_matmul_precision','highest')
    assert any(d.platform=='gpu' for d in jax.devices())
    actor=FrozenActor(a.checkpoint,a.code_list,a.semantics,{k:v[:1] for k,v in rows['base','combo'][1].items()})
    parent=actor.params
    infer=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3])
    actor.forward=lambda obs,state:infer(actor.params,obs,state)
    folder=a.output/'admission'; folder.mkdir(); records=[]
    for repeat in range(2):
        record,arrays=synthetic_run(a,native,actor,'combo','reference','policy')
        assert record['status']=='success'; save(folder,f'synthetic-retention-{repeat}',record,arrays); records.append(record)
    assert records[0]==records[1]; audit_folder(a,folder)
    record=copy.deepcopy(record); record['definition']=dict(controlled=0); rehearsal=(record,arrays)
    @jax.jit
    def sequence(params,batch):
        return actor.agent.apply(params,batch,actor.agent.init_rnn_state(8),jnp.zeros(plan['sequence_steps']*8,bool),None)[1]
    batches=[]; errors=[]; counts=[]
    for i in range(12):
        profile=plan['profiles'][i%6]; lineup=plan['fit_lineups'][i%4]
        chosen=[rows[profile,kind] for kind in BRANCHES]+[corrected,rehearsal]+[materials[lineup,seat] for seat in (0,1)]
        tensors,targets,weights,legal=pack_scenes(chosen,plan['sequence_steps'])
        batch=jax.tree.map(jnp.asarray,tensors); anchor=jax.lax.stop_gradient(sequence(parent,batch))
        assert np.isfinite(np.asarray(anchor)).all()
        for b,(record,obs) in enumerate(chosen):
            actor.reset(); own=0
            for t,d in enumerate(record['decisions']):
                logits,_,_=actor.predict({k:v[t:t+1] for k,v in obs.items()},d['player'])
                if d['player']==record['definition']['controlled']:
                    n=len(d['menu']); errors.append(float(np.abs(logits[:n]-np.asarray(anchor)[8*own+b,:n]).max())); own+=1
        batches.append((batch,jnp.asarray(targets),jnp.asarray(weights),jnp.asarray(legal),anchor))
        counts.append(dict(profile=profile,lineup=lineup,active=(weights>0).sum(0).tolist()))
    assert max(errors)<.002
    write(a.output/'sequence-parity.json',dict(passed=True,max_error=max(errors),checked_columns=96,batches=counts))
    if a.mode=='preflight':
        assert all(sha(p)==v for p,v in protected.items()); print('Preflight passed, optimizer disabled',flush=True); return
    source=json.loads(a.failure.read_text()); exact_plan=dict(correction_step=corrected[0]['exact_history']['root_step'])
    def full_evaluate(folder):
        folder.mkdir()
        old=evaluate(a,native,actor,folder/'old',rows,True)
        exact=exact_evaluate(a,native,actor,folder/'exact',source,exact_plan)
        new=material_evaluate(a,native,actor,folder/'materials',materials,True)
        result=dict(old=old,exact=exact,materials=new)
        write(folder/'summary.json',result); return result
    before=full_evaluate(a.output/'before'); summaries=[]
    for arm in plan['arms']:
        folder=a.output/arm['name']; folder.mkdir(); arm_start=time.monotonic(); actor.params=parent
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(plan['learning_rate'])); state=tx.init(parent)
        def objective(params,batch,targets,weights,legal,anchor):
            logits=sequence(params,batch)
            ce=optax.softmax_cross_entropy_with_integer_labels(logits,targets).reshape(-1,8)
            kl=parent_kl_rows(logits,anchor,legal).reshape(-1,8)
            ces=(weights*ce).sum(0); kls=(weights*kl).sum(0)
            value=ces[:4].mean()+plan['main_kl_coef']*kls[:4].mean()
            value+=plan['correction_coef']*ces[4]+plan['retention_coef']*kls[5]+arm['material_coef']*ces[6:].mean()
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
                row=dict(update=step,batch_index=index,loss=float(value),teacher_ce=np.asarray(terms[0]).tolist(),
                    parent_kl=np.asarray(terms[1]).tolist(),grad_norm=float(norm))
                assert bool(finite) and np.isfinite([row['loss'],row['grad_norm']]+row['teacher_ce']+row['parent_kl']).all()
                actor.params,state=candidate,next_state; f.write(json.dumps(row)+'\n'); f.flush()
                if step in plan['probe_updates']:
                    evaluate(a,native,actor,folder/f'old-probe-{step}',rows,False)
                    material_evaluate(a,native,actor,folder/f'material-probe-{step}',materials,False)
                    print(json.dumps(dict(arm=arm['name'],**row)),flush=True)
        checkpoint=folder/'model.flax_model'; checkpoint.write_bytes(flax.serialization.to_bytes(actor.params))
        write_checkpoint_metadata(checkpoint,observation_schema='structured-lite-v1',model_args=ModelArgs(observation_schema='structured-lite-v1'),
            code_list_hash=sha(a.code_list),semantic_table_hash=sha(a.semantics/'metadata.json'),
            capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),
            training_context=dict(experiment=plan['id'],parent_sha256=sha(a.checkpoint),arm=arm,updates=plan['updates_per_arm'],promotion_eligible=False))
        actor.params=flax.serialization.from_bytes(parent,checkpoint.read_bytes())
        after=full_evaluate(folder/'endpoint')
        summary=dict(arm=arm,results=after,checkpoint_sha256=sha(checkpoint),seconds=time.monotonic()-arm_start)
        write(folder/'summary.json',summary); summaries.append(summary)
    assert all(sha(p)==v for p,v in protected.items()); dataset(a); material_dataset(a); admitted_correction(a)
    write(a.output/'summary.json',dict(before=before,arms=summaries,seconds=time.monotonic()-start,protected_unchanged=True,promotion_eligible=False))
    print(json.dumps(dict(completed=True,seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__': main()
