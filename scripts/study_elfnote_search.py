"""Isolated elfnote search-teacher experiment. Preflight never optimizes."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
from scripts.collect_teaching_exercise import load_native
from scripts.eval_capability_exercises import FrozenActor
from scripts.teach_elfnote_combo import load_row, reserved_cases, save, audit_actor, old_evaluate
from ygoai.rl.elfnote_search import make_cases, run, discover
from ygoai.rl.exercise_core import sha
from ygoai.rl.scene_batch import pack_scenes


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def write(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)


def evaluate(a,native,actor,cases,plan,old,folder,audit=False):
    folder.mkdir();results=[]
    for i,case in enumerate(cases):
        record,obs=run(native,a.semantics,actor,case,plan)
        path=save(folder,f'case-{i:02}',record,obs)
        if audit:
            again,arrays=run(native,a.semantics,actor,case,plan)
            assert again==record and all(np.array_equal(obs[k],arrays[k]) for k in obs)
            audit_actor(a,path)
        results.append(dict(name=case['name'],split=case['split'],status=record['status'],root_step=record['root_step'],outcomes=record.get('outcomes'),intervention_attempted=record['intervention_attempted']))
    old_result=old_evaluate(a,native,actor,old,folder/'old')
    summary=dict(cases=results,old=old_result)
    write(folder/'summary.json',summary)
    print(json.dumps(dict(stage=str(folder),scores={s:sum(r['status']=='success' for r in results if r['split']==s) for s in ['train','reserved','original']},old=sum(x['success'] for x in old_result))),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','source','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--stage',choices=['preflight','collect','train'],default='preflight')
    a=p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    a.release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008'
    a.intake=a.root/'training-runs/elfnote-intake-20261014'
    a.native=a.intake/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    a.database=a.release/'repo/assets/locale/zh/cards.cdb';a.scripts=a.release/'repo/scripts/script'
    a.code_list=a.release/'code_list.txt';a.semantics=a.release/'semantics'
    parent_dir=a.root/'training-runs/elfnote-single-card-study-20261014/teach_retain'
    a.checkpoint=parent_dir/'model.flax_model'
    plan_path=a.source/'assets/exercises/teaching-v1/elfnote-search.json';plan=read(plan_path)
    oldplan=read(a.source/'assets/exercises/teaching-v1/elfnote-single-card.json')
    assert sha(a.native)==plan['native_sha256'] and sha(a.checkpoint)==plan['parent_sha256']
    refs=[load_row(a.intake/'actor-v1'/(e['name']+'.json')) for e in oldplan['demonstrations']]
    old=[load_row(parent_dir/f'endpoint/old/{i}.json') for i in range(5)]
    for i,(record,obs) in enumerate(old):
        if i==4:record['definition']=dict(controlled=0)
        for d in record['decisions']:d['teaching']=d['player']==record['definition']['controlled']
    originals=[dict(name=e['name'],split='original',definition=r[0]['definition'],intervention=None,goals=[__import__('ygoai.rl.elfnote_combo',fromlist=['GOALS']).GOALS[e['goal']]]) for e,r in zip(oldplan['demonstrations'],refs)]
    cases=make_cases(plan,reserved_cases(oldplan,refs))
    native=load_native(a,a.release/'opening-02/registration.ydk')
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_default_matmul_precision','highest')
    assert any(d.platform=='gpu' for d in jax.devices())
    actor=FrozenActor(a.checkpoint,a.code_list,a.semantics,{k:v[:1] for k,v in refs[0][1].items()})
    infer=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3])
    actor.forward=lambda obs,state:infer(actor.params,obs,state)
    parent=actor.params
    protected={str(path):sha(path) for path in [a.native,a.database,a.checkpoint,plan_path]}
    started=time.monotonic()
    if a.stage in ('collect','preflight'):
        a.output.mkdir(exist_ok=False,parents=True)
        write(a.output/'manifest.json',dict(plan=plan,plan_sha256=sha(plan_path),protected=protected,cases=cases,stage=a.stage))
        if a.stage=='preflight':
            # All three first-pair response types must replay through the public root.
            initial_digests=[]
            for i,case in enumerate(cases[:3]):
                record,obs=run(native,a.semantics,actor,case,plan)
                assert record['root_step'] is not None and (not case['intervention'] or record['intervention_attempted'])
                repeated,arr=run(native,a.semantics,actor,case,plan,prefix=record['decisions'])
                assert repeated==record and all(np.array_equal(obs[k],arr[k]) for k in obs)
                initial_digests.append(record['decisions'][0]['observation_sha256'])
                broken=copy.deepcopy(record['decisions']);broken[0]['observation_sha256']='tampered'
                try:run(native,a.semantics,actor,case,plan,prefix=broken)
                except ValueError as exc:assert 'prefix restore mismatch' in str(exc)
                else:raise AssertionError('accepted divergent replay prefix')
                audit_actor(a,save(a.output,f'preflight-{i}',record,obs))
                print(json.dumps(dict(case=case['name'],root=record['root_step'],status=record['status'])),flush=True)
            assert len(set(initial_digests))==1,'hidden response identity leaked into initial observation'
            write(a.output/'summary.json',dict(passed=True,seconds=time.monotonic()-started));return
        evaluate(a,native,actor,cases+originals,plan,old,a.output/'before',audit=True)
        collection=[]
        for mode in ('autonomous','search'):
            folder=a.output/mode;folder.mkdir()
            for i,case in enumerate(cases):
                if case['split']!='train':continue
                assert time.monotonic()-started<plan['max_collection_seconds']
                baseline=read(a.output/f'before/case-{i:02}.json')
                if baseline['root_step'] is None:
                    collection.append(dict(mode=mode,case=case['name'],status='intervention_not_observed',trials=[],decisions=0,seconds=0));continue
                log=folder/(case['name']+'.trials.jsonl')
                with log.open('x') as f:
                    def callback(row):f.write(json.dumps(row)+'\n');f.flush()
                    found,summary=discover(native,a.semantics,actor,case,plan,baseline,mode,callback)
                if found:
                    r,obs=run(native,a.semantics,actor,case,plan,prefix=found['decisions'],baseline=baseline)
                    repeated,obs2=run(native,a.semantics,actor,case,plan,prefix=found['decisions'],baseline=baseline)
                    assert r==repeated and all(np.array_equal(obs[k],obs2[k]) for k in obs)
                    assert r['status']=='success' and all(not d['teaching'] for d in r['decisions'][:r['root_step']])
                    path=save(folder,case['name'],r,obs);audit_actor(a,path)
                    summary['witness']=str(path);summary['record_sha256']=sha(path)
                write(folder/(case['name']+'.search.json'),summary);collection.append(summary)
                print(json.dumps({k:v for k,v in summary.items() if k!='trials'}),flush=True)
        assert all(sha(path)==value for path,value in protected.items())
        write(a.output/'collection.json',dict(results=collection,seconds=time.monotonic()-started,protected_unchanged=True))
        return
    # Training only consumes exact audited discovery rows, never reserved cases.
    manifest=read(a.output/'manifest.json');assert manifest['plan']==plan and manifest['cases']==cases
    collection=read(a.output/'collection.json');assert collection['protected_unchanged']
    import flax.serialization
    import optax
    from ygoai.rl.exercise_retention import parent_kl_rows
    from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
    from ygoai.rl.jax.agent import ModelArgs
    summaries=[]
    for mode in ('autonomous','search'):
        folder=a.output/(mode+'-trained');folder.mkdir();arm_start=time.monotonic();actor.params=parent
        discovered=[]
        for result in collection['results']:
            if result['mode']==mode and 'witness' in result:
                path=Path(result['witness']);assert sha(path)==result['record_sha256']
                proof=read(path.with_name(path.stem+'.audit.json'))
                assert proof['passed'] and proof['record_sha256']==sha(path) and proof['observations_sha256']==sha(path.with_suffix('.npz'))
                discovered.append(load_row(path))
        rows=discovered+refs+old; n=len(rows); lessons=len(discovered)
        tensors,targets,weights,legal=pack_scenes(rows,plan['sequence_steps'])
        batch=jax.tree.map(jnp.asarray,tensors);targets=jnp.asarray(targets);weights=jnp.asarray(weights);legal=jnp.asarray(legal)
        @jax.jit
        def sequence(params,batch):return actor.agent.apply(params,batch,actor.agent.init_rnn_state(n),jnp.zeros(plan['sequence_steps']*n,bool),None)[1]
        anchor=sequence(parent,batch)
        errors=[]
        for b,(record,obs) in enumerate(rows):
            actor.reset();own=0
            for t,d in enumerate(record['decisions']):
                logits,_,_=actor.predict({k:v[t:t+1] for k,v in obs.items()},d['player'])
                if d['player']==record['definition']['controlled']:
                    count=len(d['menu']);errors.append(float(np.max(np.abs(logits[:count]-np.asarray(anchor)[n*own+b,:count]))));own+=1
        assert max(errors)<.002,max(errors)
        write(folder/'admission.json',dict(lessons=lessons,histories=n,max_logit_error=max(errors)))
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(plan['learning_rate']));state=tx.init(parent)
        def objective(params,batch,targets,weights,legal,anchor):
            logits=sequence(params,batch)
            ces=(weights*optax.softmax_cross_entropy_with_integer_labels(logits,targets).reshape(-1,n)).sum(0)
            kls=(weights*parent_kl_rows(logits,anchor,legal).reshape(-1,n)).sum(0)
            value=(ces[:lessons].mean() if lessons else 0.)+plan['retention_coef']*kls[lessons:].mean()
            return value,(ces,kls)
        @jax.jit
        def update(params,state,batch,targets,weights,legal,anchor):
            (loss,terms),grad=jax.value_and_grad(objective,has_aux=True)(params,batch,targets,weights,legal,anchor)
            changes,state=tx.update(grad,state,params);candidate=optax.apply_updates(params,changes)
            finite=jnp.all(jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(candidate)]))
            return candidate,state,loss,terms,optax.global_norm(grad),finite
        with (folder/'metrics.jsonl').open('x') as f:
            for step in range(1,plan['updates_per_arm']+1):
                assert time.monotonic()-arm_start<plan['max_train_seconds_per_arm']
                actor.params,state,loss,terms,norm,finite=update(actor.params,state,batch,targets,weights,legal,anchor)
                metric=dict(update=step,loss=float(loss),ce=np.asarray(terms[0]).tolist(),kl=np.asarray(terms[1]).tolist(),grad_norm=float(norm))
                assert bool(finite) and np.isfinite([metric['loss'],metric['grad_norm']]+metric['ce']+metric['kl']).all()
                f.write(json.dumps(metric)+'\n');f.flush()
                if step in plan['probe_updates']:evaluate(a,native,actor,[c for c in cases if c['split']=='train']+originals,plan,old,folder/f'probe-{step}')
        checkpoint=folder/'model.flax_model';checkpoint.write_bytes(flax.serialization.to_bytes(actor.params))
        write_checkpoint_metadata(checkpoint,observation_schema='structured-lite-v1',model_args=ModelArgs(observation_schema='structured-lite-v1'),code_list_hash=sha(a.code_list),semantic_table_hash=sha(a.semantics/'metadata.json'),capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),training_context=dict(experiment=plan['id'],parent_sha256=sha(a.checkpoint),plan_sha256=sha(plan_path),mode=mode,lessons=lessons,updates=plan['updates_per_arm'],promotion_eligible=False))
        actor.params=flax.serialization.from_bytes(parent,checkpoint.read_bytes())
        result=evaluate(a,native,actor,cases+originals,plan,old,folder/'endpoint',audit=True)
        summaries.append(dict(mode=mode,result=result,checkpoint_sha256=sha(checkpoint),seconds=time.monotonic()-arm_start,lessons=lessons))
    assert all(sha(path)==value for path,value in protected.items())
    write(a.output/'summary.json',dict(arms=summaries,seconds=time.monotonic()-started,protected_unchanged=True))
    print('training complete',flush=True)


if __name__=='__main__':main()
