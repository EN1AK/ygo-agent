"""Frozen four-arm feature ablation on audited full scene histories."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np

from scripts.collect_teaching_exercise import execute, load_native
from scripts.eval_capability_exercises import FrozenActor
from scripts.teach_three_capabilities import audit_folder, evaluate, save, read_dataset
from ygoai.rl.exercise_core import sha
from ygoai.rl.effect_semantics import validate_assets
from ygoai.rl.observation_schema import tensor_contract
from ygoai.rl.scene_batch import pack_scenes

SCHEMA = 'structured-state-v2'
NEW_KEYS = ('action_effect_semantics_', 'public_chain_', 'public_turn_effects_')


def masked(obs, arm):
    out = dict(obs)
    for key in NEW_KEYS:
        enabled = arm['effects'] if key == NEW_KEYS[0] else arm['public_state']
        if not enabled:
            out[key] = out[key] * 0
    return out


def collect(a, native):
    manifest = json.loads((a.source_data/'dataset.json').read_text())
    assert manifest['passed']
    a.data.mkdir(parents=True, exist_ok=False)
    proofs = []
    counts = dict.fromkeys(NEW_KEYS, 0)
    for item in manifest['results']:
        source = a.source_data/item['path']
        assert sha(source) == item['sha256'] and sha(source.with_suffix('.npz')) == item['observations_sha256']
        reference = json.loads(source.read_text())
        with np.load(source.with_suffix('.npz'), allow_pickle=False) as f:
            old = dict(f)
        record, arrays = execute(native, reference['definition'], a.semantics, reference['branch'], observation_schema=SCHEMA)
        for key in tensor_contract('structured-lite-v1'):
            np.testing.assert_array_equal(old[key], arrays[key], err_msg=source.name+':'+key)
        for key in ('core_responses', 'events', 'final_state', 'result', 'roots', 'core_seed'):
            assert record[key] == reference[key], (source.name, key)
        assert len(record['decisions']) == len(reference['decisions'])
        for left, right in zip(record['decisions'], reference['decisions']):
            for key in ('player', 'menu', 'action'):
                assert left[key] == right[key]
        for key in NEW_KEYS:
            counts[key] += int((arrays[key][..., 0] > 0).sum())
        path = save(a.data, source.stem, record, arrays)
        proofs.append(dict(item, sha256=sha(path), observations_sha256=sha(path.with_suffix('.npz'))))
    assert all(counts.values()), counts
    audit_folder(a, a.data)
    result = dict(manifest, native_sha256=sha(a.native), results=proofs,
                  source_dataset_sha256=sha(a.source_data/'dataset.json'), old_tensor_parity=True,
                  new_feature_row_observations=counts)
    (a.data/'dataset.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(collected=len(proofs), old_tensor_parity=True, feature_rows=counts)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('collect','preflight','fit'), default='preflight')
    for key in ('release','native','semantics','source-data','data','checkpoint','plan','output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path): setattr(a, key, value.resolve())
    a.database = a.release/'repo/assets/locale/zh/cards.cdb'
    a.scripts = a.release/'repo/scripts/script'
    a.code_list = a.release/'code_list.txt'
    plan = json.loads(a.plan.read_text())
    assert sha(a.checkpoint) == plan['parent_sha256']
    sem = validate_assets(a.semantics)
    assert sem['source_hashes']['code_list_sha256'] == sha(a.code_list)
    old_sem = json.loads((a.release/'semantics/metadata.json').read_text())
    assert all(old_sem['table_hashes'][k] == sem['table_hashes'][k] for k in ('static','tags','confidence'))
    protected = {str(a.release/r['path']):r['sha256'] for r in json.loads((a.release/'manifest.json').read_text())['artifacts']}
    for path in (a.checkpoint, Path(str(a.checkpoint)+'.metadata.json'), a.plan, a.native,
                 a.source_data/'dataset.json', a.semantics/'metadata.json'):
        protected[str(path)] = sha(path)
    assert all(sha(path) == digest for path,digest in protected.items())
    native = load_native(a, a.release/'opening-02/registration.ydk')
    if a.mode == 'collect':
        collect(a, native)
        assert all(sha(path) == digest for path,digest in protected.items())
        return
    rows = read_dataset(a)
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    (a.output/'manifest.json').write_text(json.dumps(dict(plan=plan, protected=protected,
        dataset_sha256=sha(a.data/'dataset.json'), source_sha256=sha(__file__)), indent=2))
    import jax
    import jax.numpy as jnp
    import flax.serialization
    import optax
    from scripts.migrate_public_state_checkpoint import migrate_model
    from ygoai.rl.jax.agent import ModelArgs, RNNAgent
    from ygoai.rl.checkpoint_compat import load_checkpoint_metadata, write_checkpoint_metadata
    from ygoai.rl.exercise_retention import parent_kl_rows
    jax.config.update('jax_default_matmul_precision', 'highest')
    metadata = load_checkpoint_metadata(a.checkpoint)
    sample = {k:v[:1] for k,v in rows[0][1].items()}
    old_actor = FrozenActor(a.checkpoint, a.code_list, a.release/'semantics', sample)
    migrated, model_args, migration = migrate_model(old_actor.params, ModelArgs(**metadata['model_architecture']),
        sum(bool(s.strip()) for s in a.code_list.read_text().splitlines()), metadata['capacities'])
    class Actor(FrozenActor):
        observation_schema = SCHEMA
        def __init__(self):
            self.jax = jax
            self.agent = RNNAgent(embedding_shape=sum(bool(s.strip()) for s in a.code_list.read_text().splitlines()), **asdict(model_args))
            self.params = migrated
            self.arm = dict(effects=True, public_state=True)
            self.infer = jax.jit(lambda params,obs,state:self.agent.apply(params,obs,state)[:3])
            self.forward = lambda obs,state:self.infer(self.params,masked(obs,self.arm),state)
            self.reset()
    actor = Actor()
    maxima = dict(logits=0., probabilities=0., value=0., recurrent_state=0.)
    for item in json.loads((a.data/'dataset.json').read_text())['results']:
        record = json.loads((a.data/item['path']).read_text())
        with np.load((a.data/item['path']).with_suffix('.npz'), allow_pickle=False) as f: obs = dict(f)
        old_actor.reset(); actor.reset()
        for i,d in enumerate(record['decisions']):
            one = {k:v[i:i+1] for k,v in obs.items()}
            left,lv,_ = old_actor.predict(one,d['player']); right,rv,_ = actor.predict(one,d['player'])
            n = len(d['menu'])
            pairs = [('logits',left[:n],right[:n]),('value',lv,rv),
                     ('probabilities',jax.nn.softmax(left[:n]),jax.nn.softmax(right[:n]))]
            pairs += [('recurrent_state',x,y) for x,y in zip(jax.tree.leaves(old_actor.states[d['player']]),jax.tree.leaves(actor.states[d['player']]))]
            for key,x,y in pairs:
                error = float(np.abs(np.asarray(x)-np.asarray(y)).max())
                maxima[key] = max(maxima[key],error)
                assert np.isfinite(error) and error < .002, (key,error)
    migration.update(real_history_max_errors=maxima, real_history_parity=True)
    (a.output/'migration-report.json').write_text(json.dumps(migration, indent=2))
    def checkpoint(path, params, context):
        path.write_bytes(flax.serialization.to_bytes(params))
        write_checkpoint_metadata(path, observation_schema=SCHEMA, model_args=model_args,
            code_list_hash=sha(a.code_list),semantic_table_hash=sha(a.semantics/'metadata.json'),
            capacities=metadata['capacities'],migrated_from=sha(a.checkpoint),
            training_context=dict(promotion_eligible=False,experiment=plan['id'],**context))
    checkpoint(a.output/'migrated.flax_model', migrated, dict(migration=True))
    tensors,actions,weights,legal = pack_scenes(rows,plan['sequence_steps'])
    batch = jax.tree.map(jnp.asarray,tensors)
    targets,weights,legal = map(jnp.asarray,(actions,weights,legal))
    size = len(rows)
    @jax.jit
    def sequence(params,observations):
        return actor.agent.apply(params,observations,actor.agent.init_rnn_state(size),jnp.zeros(len(actions),bool),None)[1]
    anchor = jax.lax.stop_gradient(sequence(migrated,batch))
    assert np.isfinite(np.asarray(anchor)).all()
    errors = []
    for b,(record,obs) in enumerate(rows):
        actor.reset(); own = 0
        for i,d in enumerate(record['decisions']):
            logits,_,_ = actor.predict({k:v[i:i+1] for k,v in obs.items()},d['player'])
            if d['player'] == record['definition']['controlled']:
                n = len(d['menu'])
                errors.append(float(np.abs(logits[:n]-np.asarray(anchor)[size*own+b,:n]).max())); own += 1
    assert max(errors) < .002
    (a.output/'sequence-parity.json').write_text(json.dumps(dict(max_error=max(errors),passed=True)))
    print(json.dumps(dict(migration=maxima,sequence_error=max(errors),devices=str(jax.devices()))),flush=True)
    if a.mode == 'preflight': return
    before = evaluate(a,native,actor,a.output/'before',True)
    summaries = []
    for arm in plan['arms']:
        folder = a.output/arm['name']; folder.mkdir()
        actor.params = migrated; actor.arm = arm
        arm_batch = masked(batch,arm)
        arm_start = time.monotonic()
        tx = optax.chain(optax.clip_by_global_norm(1.),optax.adam(plan['learning_rate']))
        state = tx.init(migrated)
        def objective(params):
            logits = sequence(params,arm_batch)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits,targets).reshape(-1,size)
            kl = parent_kl_rows(logits,anchor,legal).reshape(-1,size)
            losses,kls = (weights*ce).sum(0),(weights*kl).sum(0)
            return losses.mean()+plan['parent_kl_coef']*kls.mean(),(losses,kls)
        @jax.jit
        def update(params,state):
            (value,terms),grad = jax.value_and_grad(objective,has_aux=True)(params)
            changes,next_state = tx.update(grad,state,params)
            candidate = optax.apply_updates(params,changes)
            finite = jnp.all(jnp.stack([jnp.isfinite(v).all() for v in jax.tree.leaves(candidate)]))
            return candidate,next_state,value,terms,optax.global_norm(grad),finite
        update_seconds = 0.
        with (folder/'metrics.jsonl').open('x') as f:
            for step in range(1,plan['updates_per_arm']+1):
                if time.monotonic()-start > plan['max_total_seconds'] or time.monotonic()-arm_start > plan['max_seconds_per_arm']:
                    raise TimeoutError('frozen ablation budget exceeded')
                tick = time.monotonic()
                candidate,next_state,value,terms,norm,finite = update(actor.params,state)
                row = dict(update=step,loss=float(value),teacher_ce=np.asarray(terms[0]).tolist(),
                           parent_kl=np.asarray(terms[1]).tolist(),grad_norm=float(norm))
                update_seconds += time.monotonic()-tick
                assert bool(finite) and np.isfinite([row['loss'],row['grad_norm']]+row['teacher_ce']+row['parent_kl']).all()
                actor.params,state = candidate,next_state
                f.write(json.dumps(row)+'\n'); f.flush()
                if step in plan['probe_updates']:
                    evaluate(a,native,actor,folder/f'probe-{step}',False)
                    print(json.dumps(dict(arm=arm['name'],**row)),flush=True)
        checkpoint(folder/'model.flax_model',actor.params,dict(arm=arm,updates=plan['updates_per_arm']))
        after = evaluate(a,native,actor,folder/'endpoint',True)
        regressions = [r for old,r in zip(before,after) if old['success'] and not r['success']]
        summary = dict(arm=arm,results=after,parent_solved_regressions=regressions,
            seconds=time.monotonic()-arm_start,update_seconds_including_compile=update_seconds,
            device_memory=jax.devices()[0].memory_stats(),checkpoint_sha256=sha(folder/'model.flax_model'))
        (folder/'summary.json').write_text(json.dumps(summary,indent=2))
        summaries.append(summary)
    assert all(sha(path) == digest for path,digest in protected.items())
    read_dataset(a); validate_assets(a.semantics)
    result = dict(before=before,arms=summaries,migration=maxima,protected_unchanged=True,
                  seconds=time.monotonic()-start,promotion_eligible=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(completed=True,seconds=result['seconds'])),flush=True)


if __name__ == '__main__': main()
