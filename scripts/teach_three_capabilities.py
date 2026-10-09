"""Audited normal-opening joint teaching; default mode never optimizes."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import time
import numpy as np

from scripts.collect_teaching_exercise import execute, load_native
from scripts.eval_capability_exercises import FrozenActor, run as synthetic_run
from ygoai.rl.exercise_core import sha
from ygoai.rl.exercise_teaching import opening
from ygoai.rl.exercise_retention import parent_kl_rows
from ygoai.rl.scene_batch import pack_scenes
from ygoai.rl.exercise_starters import CASES

BRANCHES={'combo':('reference','revive_in_defense'), 'battle':('direct','attack_monster'),
          'battle-negated':('end_battle','attack_monster'), 'interaction':('veiler','ogre','pass')}
DEPTHS={'combo':('opening','combo','position'), 'battle':('opening','battle'),
        'battle-negated':('opening','battle'), 'interaction':('opening','interaction')}


def audit_folder(a,folder):
    env=os.environ.copy()
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    output=folder/'independent-audit.json'
    subprocess.run([sys.executable,'-m','scripts.audit_joint_scenes','--native',str(a.native),
        '--database',str(a.database),'--scripts',str(a.scripts),'--input',str(folder),'--output',str(output)],
        env=env,check=True)
    return json.loads(output.read_text())


def save(folder, name, record, arrays):
    path=folder/(name+'.json')
    with path.open('x') as f: json.dump(record,f,indent=2)
    with path.with_suffix('.npz').open('xb') as f: np.savez_compressed(f,**arrays)
    return path


def collect(a,native):
    a.data.mkdir(parents=True,exist_ok=False)
    proofs=[]
    for kind,branches in BRANCHES.items():
        for variant in range(3):
            definition=opening(a.database,a.code_list,variant,kind)
            roots=[]
            for branch in branches:
                records=[]
                for repeat in range(2):
                    record,arrays=execute(native,definition,a.semantics,branch)
                    path=save(a.data,f'{kind}-v{variant}-{branch}-{repeat}',record,arrays)
                    records.append(record)
                    if repeat==0:
                        proof=dict(path=path.name,sha256=sha(path),observations_sha256=sha(path.with_suffix('.npz')))
                assert records[0]==records[1], 'nonreproducible reference'
                expected=branch==branches[0]
                assert record['result']['success']==expected, (kind,branch,record['result'])
                depth=DEPTHS[kind][-1]
                roots.append(record['decisions'][record['roots'][depth]]['observation_sha256'])
                proofs.append(dict(kind=kind,variant=variant,branch=branch,positive=expected,**proof))
            assert len(set(roots))==1, 'contrast root mismatch'
    audited={r['path']:r for r in audit_folder(a,a.data)['results']}
    for proof in proofs:
        assert audited[proof['path']]['sha256']==proof['sha256']
        proof['passed']=audited[proof['path']]['passed']
    manifest=dict(schema='three-capability-scenes-v1',results=proofs,passed=True,
                  native_sha256=sha(a.native),training_variants=[0],development_variants=[1,2],
                  unseen_families=[],history='complete own-seat from standard opening')
    with (a.data/'dataset.json').open('x') as f: json.dump(manifest,f,indent=2)
    print(json.dumps(dict(collected=len(proofs),passed=True)),flush=True)


def read_dataset(a):
    manifest=json.loads((a.data/'dataset.json').read_text())
    assert manifest['passed'] and manifest['native_sha256']==sha(a.native)
    rows=[]
    for item in manifest['results']:
        path=a.data/item['path']
        assert sha(path)==item['sha256'] and sha(path.with_suffix('.npz'))==item['observations_sha256']
        if item['positive'] and item['variant']==0:
            record=json.loads(path.read_text())
            assert record['normal_opening'] and record['result']['success']
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as f: rows.append((record,dict(f)))
    assert [r['definition']['kind'] for r,_ in rows]==list(BRANCHES)
    return rows


def evaluate(a,native,actor,folder,full):
    folder.mkdir()
    results=[]
    for kind,branches in BRANCHES.items():
        for variant in (range(3) if full else (0,)):
            definition=json.loads((a.data/f'{kind}-v{variant}-{branches[0]}-0.json').read_text())['definition']
            for depth in DEPTHS[kind]:
                records=[]
                for repeat in range(2 if full else 1):
                    record,arrays=execute(native,definition,a.semantics,branches[0],actor,depth)
                    path=save(folder,f'{kind}-v{variant}-{depth}-{repeat}',record,arrays)
                    records.append(record)
                if full: assert records[0]==records[1]
                results.append(dict(kind=kind,variant=variant,depth=depth,**record['result']))
    if full:
        for case,definition in CASES.items():
            records=[]
            for repeat in range(2):
                record,arrays=synthetic_run(a,native,actor,case,next(iter(definition['branches'])),'policy')
                path=save(folder,f'synthetic-{case}-{repeat}',record,arrays)
                records.append(record)
            assert records[0]==records[1]
            results.append(dict(kind='synthetic',case=case,success=record['status']=='success',status=record['status']))
    if full: audit_folder(a,folder)
    with (folder/'summary.json').open('x') as f: json.dump(results,f,indent=2)
    print(json.dumps(dict(stage=folder.name,results=results)),flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['collect','preflight','fit'],default='preflight')
    for key in ('release','checkpoint','plan','data','output'): p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path): setattr(a,key,value.resolve())
    a.native=a.release/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    a.database=a.release/'repo/assets/locale/zh/cards.cdb'
    a.scripts=a.release/'repo/scripts/script'
    a.code_list=a.release/'code_list.txt'; a.semantics=a.release/'semantics'
    plan=json.loads(a.plan.read_text()); start=time.monotonic()
    assert sha(a.checkpoint)==plan['parent_sha256'] and sha(a.native)==plan['native_sha256']
    runtime=json.loads((a.release/'manifest.json').read_text())
    protected={str(a.release/r['path']):r['sha256'] for r in runtime['artifacts']}
    protected.update({str(path):sha(path) for path in (a.checkpoint,Path(str(a.checkpoint)+'.metadata.json'),a.plan)})
    assert all(sha(path)==digest for path,digest in protected.items())
    native=load_native(a,a.release/'opening-02/registration.ydk')
    if a.mode=='collect':
        collect(a,native)
        assert all(sha(path)==digest for path,digest in protected.items())
        return
    rows=read_dataset(a)
    a.output.mkdir(parents=True,exist_ok=False)
    protected[str(a.data/'dataset.json')]=sha(a.data/'dataset.json')
    manifest=dict(plan=plan,protected=protected,optimizer_enabled=a.mode=='fit',source_sha256=sha(__file__))
    with (a.output/'manifest.json').open('x') as f: json.dump(manifest,f,indent=2)
    import jax
    import jax.numpy as jnp
    import optax
    import flax.serialization
    from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
    from ygoai.rl.jax.agent import ModelArgs
    jax.config.update('jax_default_matmul_precision','highest')
    tensors,actions,scene_weights,legal=pack_scenes(rows,plan['sequence_steps'])
    batch=jax.tree.map(jnp.asarray,tensors); targets=jnp.asarray(actions)
    weights=jnp.asarray(scene_weights); legal=jnp.asarray(legal)
    size=len(rows); sample={k:v[:1] for k,v in tensors.items()}
    actor=FrozenActor(a.checkpoint,a.code_list,a.semantics,sample); parent=actor.params
    infer=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3])
    actor.forward=lambda obs,state:infer(actor.params,obs,state)
    @jax.jit
    def sequence(params):
        return actor.agent.apply(params,batch,actor.agent.init_rnn_state(size),jnp.zeros(len(actions),bool),None)[1]
    anchor=jax.lax.stop_gradient(sequence(parent))
    assert np.isfinite(np.asarray(anchor)).all(), 'nonfinite frozen parent output'
    errors=[]
    for b,(record,obs) in enumerate(rows):
        actor.reset(); own=0
        for i,d in enumerate(record['decisions']):
            logits,_,_=actor.predict({k:v[i:i+1] for k,v in obs.items()},d['player'])
            if d['player']==record['definition']['controlled']:
                n=len(d['menu']); errors.append(float(np.abs(logits[:n]-np.asarray(anchor)[size*own+b,:n]).max())); own+=1
    assert max(errors)<.002
    with (a.output/'sequence-parity.json').open('x') as f: json.dump(dict(passed=True,max_error=max(errors)),f)
    if a.mode=='preflight':
        print('Preflight passed; optimizer disabled',flush=True); return
    before=evaluate(a,native,actor,a.output/'before',True)
    summaries=[]
    for arm in plan['arms']:
        folder=a.output/arm['name']; folder.mkdir(); arm_start=time.monotonic()
        actor.params=parent
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(plan['learning_rate']))
        state=tx.init(parent)
        def objective(params):
            logits=sequence(params)
            ce=optax.softmax_cross_entropy_with_integer_labels(logits,targets).reshape(-1,size)
            kl=parent_kl_rows(logits,anchor,legal).reshape(-1,size)
            losses=(weights*ce).sum(0); kls=(weights*kl).sum(0)
            return losses.mean()+arm['parent_kl_coef']*kls.mean(),(losses,kls)
        @jax.jit
        def update(params,state):
            (value,terms),grad=jax.value_and_grad(objective,has_aux=True)(params)
            changes,next_state=tx.update(grad,state,params); candidate=optax.apply_updates(params,changes)
            finite=jnp.all(jnp.stack([jnp.isfinite(v).all() for v in jax.tree.leaves(candidate)]))
            return candidate,next_state,value,terms,optax.global_norm(grad),finite
        with (folder/'metrics.jsonl').open('x') as metrics:
            for step in range(1,plan['updates_per_arm']+1):
                if time.monotonic()-start>plan['max_total_seconds'] or time.monotonic()-arm_start>plan['max_seconds_per_arm']:
                    raise TimeoutError('frozen teaching budget exceeded')
                candidate,next_state,value,terms,norm,finite=update(actor.params,state)
                row=dict(update=step,loss=float(value),teacher_ce=np.asarray(terms[0]).tolist(),parent_kl=np.asarray(terms[1]).tolist(),grad_norm=float(norm))
                assert bool(finite) and np.isfinite([row['loss'],row['grad_norm']]+row['teacher_ce']+row['parent_kl']).all()
                actor.params,state=candidate,next_state
                metrics.write(json.dumps(row)+'\n'); metrics.flush()
                if step in plan['probe_updates']:
                    evaluate(a,native,actor,folder/f'probe-{step}',False)
                    print(json.dumps(dict(arm=arm['name'],**row)),flush=True)
        checkpoint=folder/'model.flax_model'
        checkpoint.write_bytes(flax.serialization.to_bytes(actor.params))
        write_checkpoint_metadata(checkpoint,observation_schema='structured-lite-v1',model_args=ModelArgs(observation_schema='structured-lite-v1'),
            code_list_hash=sha(a.code_list),semantic_table_hash=sha(a.semantics/'metadata.json'),
            capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),
            training_context=dict(experiment=plan['id'],parent_sha256=sha(a.checkpoint),arm=arm,updates=plan['updates_per_arm'],promotion_eligible=False))
        actor.params=flax.serialization.from_bytes(parent,checkpoint.read_bytes())
        after=evaluate(a,native,actor,folder/'endpoint',True)
        summary=dict(arm=arm,results=after,all_scenes_passed=all(r['success'] for r in after),checkpoint_sha256=sha(checkpoint),seconds=time.monotonic()-arm_start)
        with (folder/'summary.json').open('x') as f: json.dump(summary,f,indent=2)
        summaries.append(summary)
    assert all(sha(path)==digest for path,digest in protected.items())
    read_dataset(a)
    result=dict(before=before,arms=summaries,protected_unchanged=True,seconds=time.monotonic()-start,promotion_eligible=False)
    with (a.output/'summary.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
