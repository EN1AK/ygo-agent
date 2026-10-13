"""Bounded exact-history correction and separate successful-policy rehearsal."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np

from scripts.collect_teaching_exercise import execute, load_native
from scripts.eval_capability_exercises import FrozenActor, run as synthetic_run
from scripts.teach_context_coverage import dataset, evaluate
from scripts.teach_three_capabilities import BRANCHES, audit_folder, save
from ygoai.rl.exercise_core import sha
from ygoai.rl.exercise_retention import parent_kl_rows
from ygoai.rl.scene_batch import pack_scenes


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)


def exact_evaluate(a, native, actor, folder, source, plan):
    folder.mkdir(); records=[]
    for repeat in range(2):
        record, arrays = execute(native, source['definition'], a.semantics, 'end_battle',
            actor, 'correction', replay_record=source, replay_until=plan['correction_step'])
        save(folder, f'exact-policy-{repeat}', record, arrays); records.append(record)
    assert records[0] == records[1]
    audit_folder(a, folder)
    write(folder/'summary.json', record['result'])
    return record['result']


def admit(a, native, actor, old_params, source, plan):
    folder=a.output/'admission'; folder.mkdir(); corrected=None
    step=plan['correction_step']; root=source['decisions'][step]
    good=[i for i,m in enumerate(root['menu']) if m['phase']=='Main2']
    assert len(good)==1 and root['player']==source['definition']['controlled']
    roots=[]; logit_error=0.
    for name, action, expected in [('attack',root['action'],False), ('stop',good[0],True)]:
        records=[]
        for repeat in range(2):
            record, arrays=execute(native,source['definition'],a.semantics,'end_battle',actor,
                'correction',replay_record=source,replay_until=step,root_action=action)
            assert record['result']['success']==expected
            error=float(np.abs(np.asarray(record['decisions'][step]['logits'])-root['logits']).max())
            logit_error=max(logit_error,error); assert error<.002, 'source actor not reproduced'
            save(folder,f'exact-{name}-{repeat}',record,arrays); records.append(record)
        assert records[0]==records[1]
        roots.append(record['decisions'][step]['observation_sha256'])
        if expected: corrected=(record,arrays)
    assert len(set(roots))==1
    current=actor.params; actor.params=old_params; records=[]
    for repeat in range(2):
        record, arrays=synthetic_run(a,native,actor,'combo','reference','policy')
        assert record['status']=='success', 'retention anchor must demonstrate success'
        save(folder,f'synthetic-retention-{repeat}',record,arrays); records.append(record)
    assert records[0]==records[1]; actor.params=current
    audit=audit_folder(a,folder)
    # Adapt metadata only after the raw synthetic evidence has been audited.
    rehearsal=copy.deepcopy(record); rehearsal['definition']=dict(controlled=0)
    write(folder/'admission.json',dict(passed=True,source_sha256=sha(a.failure),root_step=step,
        source_checkpoint_sha256=sha(a.checkpoint),retention_checkpoint_sha256=sha(a.retention_checkpoint),
        source_logit_max_error=logit_error,audit_sha256=sha(folder/'independent-audit.json'),
        records=audit['results'],split='development',synthetic_anchor_is_rehearsal=True,
        correction_labels='only Main2 at verified root; prefix has zero correction weight'))
    return corrected,(rehearsal,arrays)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','fit'],default='preflight')
    for name in ('release','checkpoint','retention-checkpoint','plan','data','failure','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path): setattr(a,key,value.resolve())
    a.native=a.release/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    a.database=a.release/'repo/assets/locale/zh/cards.cdb'; a.scripts=a.release/'repo/scripts/script'
    a.code_list=a.release/'code_list.txt'; a.semantics=a.release/'semantics'
    plan=json.loads(a.plan.read_text()); start=time.monotonic()
    for path,key in [(a.checkpoint,'parent_sha256'),(a.retention_checkpoint,'retention_parent_sha256'),
                     (a.native,'native_sha256'),(a.failure,'failure_sha256')]:
        assert sha(path)==plan[key], ('input hash mismatch',path)
    protected={str(a.release/r['path']):r['sha256'] for r in json.loads((a.release/'manifest.json').read_text())['artifacts']}
    for path in (a.plan,a.checkpoint,a.retention_checkpoint,a.failure,a.data/'dataset.json',
                 Path(str(a.checkpoint)+'.metadata.json'),Path(str(a.retention_checkpoint)+'.metadata.json')):
        protected[str(path)]=sha(path)
    assert all(sha(p)==v for p,v in protected.items())
    rows=dataset(a); source=json.loads(a.failure.read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'manifest.json',dict(plan=plan,protected=protected,optimizer_enabled=a.mode=='fit',source_sha256=sha(__file__)))
    native=load_native(a,a.release/'opening-02/registration.ydk')
    import jax
    import jax.numpy as jnp
    import optax
    import flax.serialization
    from ygoai.rl.jax.agent import ModelArgs
    from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
    jax.config.update('jax_default_matmul_precision','highest')
    assert any(d.platform=='gpu' for d in jax.devices()), 'GPU required for bounded fit/preflight'
    sample={k:v[:1] for k,v in rows['base','combo'][1].items()}
    actor=FrozenActor(a.checkpoint,a.code_list,a.semantics,sample)
    previous=FrozenActor(a.retention_checkpoint,a.code_list,a.semantics,sample)
    parent=actor.params; old_params=previous.params
    infer=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3])
    actor.forward=lambda obs,state:infer(actor.params,obs,state)
    corrected,rehearsal=admit(a,native,actor,old_params,source,plan)
    write(a.output/'collection-cost.json',dict(seconds=time.monotonic()-start))
    @jax.jit
    def sequence(params,batch):
        return actor.agent.apply(params,batch,actor.agent.init_rnn_state(6),
            jnp.zeros(plan['sequence_steps']*6,bool),None)[1]
    batches={}; errors=[]; counts={}
    for profile in plan['profiles']:
        chosen=[rows[profile,kind] for kind in BRANCHES]+[corrected,rehearsal]
        tensors,targets,weights,legal=pack_scenes(chosen,plan['sequence_steps'])
        batch=jax.tree.map(jnp.asarray,tensors)
        anchors={}
        for label,params,columns in [('current',parent,range(5)),('old',old_params,[5])]:
            anchor=sequence(params,batch); assert np.isfinite(np.asarray(anchor)).all()
            anchors[label]=anchor; actor.params=params
            for b in columns:
                record,obs=chosen[b]; actor.reset(); own=0
                for i,d in enumerate(record['decisions']):
                    logits,_,_=actor.predict({k:v[i:i+1] for k,v in obs.items()},d['player'])
                    if d['player']==record['definition']['controlled']:
                        n=len(d['menu']); errors.append(float(np.abs(logits[:n]-np.asarray(anchor)[6*own+b,:n]).max())); own+=1
        # Each column's anchor comes from its declared model, with its own history.
        anchor=anchors['current'].reshape(plan['sequence_steps'],6,-1)
        anchor=anchor.at[:,5,:].set(anchors['old'].reshape(plan['sequence_steps'],6,-1)[:,5,:]).reshape(-1,legal.shape[-1])
        batches[profile]=(batch,jnp.asarray(targets),jnp.asarray(weights),jnp.asarray(legal),jax.lax.stop_gradient(anchor))
        counts[profile]=(weights>0).sum(0).tolist()
    actor.params=parent
    assert max(errors)<.002
    write(a.output/'sequence-parity.json',dict(passed=True,max_error=max(errors),checked_columns=36,active_decisions=counts))
    if a.mode=='preflight':
        assert all(sha(p)==v for p,v in protected.items())
        print('Preflight passed, optimizer disabled',flush=True); return
    before=evaluate(a,native,actor,a.output/'before',rows,True)
    before_exact=exact_evaluate(a,native,actor,a.output/'before-exact',source,plan)
    summaries=[]
    for arm in plan['arms']:
        folder=a.output/arm['name']; folder.mkdir(); arm_start=time.monotonic(); actor.params=parent
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(plan['learning_rate'])); state=tx.init(parent)
        def objective(params,batch,targets,weights,legal,anchor):
            logits=sequence(params,batch)
            ce=optax.softmax_cross_entropy_with_integer_labels(logits,targets).reshape(-1,6)
            kl=parent_kl_rows(logits,anchor,legal).reshape(-1,6)
            ces=(weights*ce).sum(0); kls=(weights*kl).sum(0)
            value=ces[:4].mean()+plan['main_kl_coef']*kls[:4].mean()
            value+=arm['correction_coef']*ces[4]+arm['retention_coef']*kls[5]
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
                profile=plan['profiles'][(step-1)%len(plan['profiles'])]
                candidate,next_state,value,terms,norm,finite=update(actor.params,state,*batches[profile])
                row=dict(update=step,profile=profile,loss=float(value),teacher_ce=np.asarray(terms[0]).tolist(),
                    parent_kl=np.asarray(terms[1]).tolist(),grad_norm=float(norm))
                assert bool(finite) and np.isfinite([row['loss'],row['grad_norm']]+row['teacher_ce']+row['parent_kl']).all()
                actor.params,state=candidate,next_state; f.write(json.dumps(row)+'\n'); f.flush()
                if step in plan['probe_updates']:
                    evaluate(a,native,actor,folder/f'probe-{step}',rows,False)
                    exact_evaluate(a,native,actor,folder/f'exact-{step}',source,plan)
                    print(json.dumps(dict(arm=arm['name'],**row)),flush=True)
        checkpoint=folder/'model.flax_model'; checkpoint.write_bytes(flax.serialization.to_bytes(actor.params))
        write_checkpoint_metadata(checkpoint,observation_schema='structured-lite-v1',model_args=ModelArgs(observation_schema='structured-lite-v1'),
            code_list_hash=sha(a.code_list),semantic_table_hash=sha(a.semantics/'metadata.json'),
            capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),
            training_context=dict(experiment=plan['id'],parent_sha256=sha(a.checkpoint),arm=arm,
                updates=plan['updates_per_arm'],synthetic_combo_is_development_rehearsal=True,promotion_eligible=False))
        actor.params=flax.serialization.from_bytes(parent,checkpoint.read_bytes())
        after=evaluate(a,native,actor,folder/'endpoint',rows,True)
        exact=exact_evaluate(a,native,actor,folder/'endpoint-exact',source,plan)
        summary=dict(arm=arm,results=after,exact=exact,all_scenes_passed=all(r['success'] for r in after) and exact['success'],
            checkpoint_sha256=sha(checkpoint),seconds=time.monotonic()-arm_start)
        write(folder/'summary.json',summary); summaries.append(summary)
    assert all(sha(p)==v for p,v in protected.items()); dataset(a)
    result=dict(before=before,before_exact=before_exact,arms=summaries,protected_unchanged=True,
                seconds=time.monotonic()-start,promotion_eligible=False)
    write(a.output/'summary.json',result)
    print(json.dumps(dict(completed=True,seconds=result['seconds'],passed=[s['all_scenes_passed'] for s in summaries])),flush=True)


if __name__=='__main__': main()
