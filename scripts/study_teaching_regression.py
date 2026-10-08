"""Opt-in, preregistered ablations of suffix teaching and prefix retention."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.collect_teaching_exercise import execute, load_native
from scripts.eval_capability_exercises import FrozenActor
from scripts.train_exercise_demonstration import evaluate, pack, verified_record
from ygoai.rl.exercise_core import sha


def component_masks(rows):
    """Time-major masks: combo suffix, battle suffix, combo prefix, Halq root."""
    masks = np.zeros((4, 32, 2), np.float32)
    descriptors = []
    for seat, (record, _) in enumerate(rows):
        root = record['roots']['position' if seat == 0 else 'battle']
        own = [i for i, d in enumerate(record['decisions']) if d['player'] == 1]
        for t, i in enumerate(own):
            d = record['decisions'][i]
            masks[seat, t, seat] = i >= root
            if seat == 0:
                masks[2, t, seat] = i < root
                masks[3, t, seat] = i == record['roots']['combo']
            descriptors.append(dict(batch_index=2*t+seat, kind=record['definition']['kind'],
                                    step=i, action=d['action'], menu=d['menu']))
    flat = masks.reshape(4, -1)
    assert np.all(flat.sum(axis=1) > 0)
    return flat / flat.sum(axis=1, keepdims=True), descriptors


def probabilities(logits):
    p = np.exp(np.asarray(logits) - np.max(logits))
    return p / p.sum()


def save_record(folder, name, record, arrays):
    (folder / (name + '.json')).write_text(json.dumps(record, indent=2))
    np.savez_compressed(folder / (name + '.npz'), **arrays)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('native', 'database', 'code-list', 'scripts', 'semantics', 'checkpoint',
                'combo', 'combo-audit', 'battle', 'battle-audit', 'release', 'plan', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve())
    start = time.monotonic()
    a.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(a.plan.read_text())
    assert sha(a.checkpoint) == plan['parent_sha256']
    assert sha(a.native) == plan['native_sha256']
    runtime = json.loads((a.release/'manifest.json').read_text())
    protected = {str(a.release/r['path']): r['sha256'] for r in runtime['artifacts']}
    protected[str(a.plan)] = sha(a.plan)
    assert all(sha(Path(path)) == digest for path, digest in protected.items())
    rows = [verified_record(a.combo, a.combo_audit, 'v0-reference-0.json', a.native),
            verified_record(a.battle, a.battle_audit, 'v0-direct-0.json', a.native)]
    manifest = dict(plan=plan, plan_sha256=sha(a.plan), parent_sha256=sha(a.checkpoint),
                    release_manifest_sha256=sha(a.release/'manifest.json'),
                    source_sha256=sha(Path(__file__)), optimizer_enabled=a.execute,
                    audits={str(path): sha(path) for path in (a.combo_audit, a.battle_audit)})
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import jax
    import jax.numpy as jnp
    import flax.serialization
    import optax
    from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
    from ygoai.rl.jax.agent import ModelArgs
    jax.config.update('jax_default_matmul_precision', 'highest')
    native = load_native(a, a.combo/'registration.ydk')
    tensors, actions, original_mask = pack(rows)
    masks, descriptors = component_masks(rows)
    assert np.allclose((masks[0]+masks[1])/2, original_mask/original_mask.sum())
    batch = jax.tree.map(jnp.asarray, tensors)
    targets, masks = jnp.asarray(actions), jnp.asarray(masks)
    sample = {k: v[:1] for k, v in tensors.items()}
    actor = FrozenActor(a.checkpoint, a.code_list, a.semantics, sample)
    parent = actor.params
    infer = jax.jit(lambda params, obs, state: actor.agent.apply(params, obs, state)[:3])
    # Parameters must be dynamic inputs; do not close a jitted function over a
    # mutable actor.params and accidentally evaluate the first traced model.
    actor.forward = lambda obs, state: infer(actor.params, obs, state)

    @jax.jit
    def sequence(params):
        return actor.agent.apply(params, batch, actor.agent.init_rnn_state(2),
                                 jnp.zeros(len(actions), dtype=bool), None)[1]

    def components(params):
        ce = optax.softmax_cross_entropy_with_integer_labels(sequence(params), targets)
        return masks @ ce

    component_values = jax.jit(components)
    grad = jax.jit(jax.value_and_grad(lambda params, w: jnp.dot(components(params), w)))

    @jax.jit
    def cosine(g1, g2):
        dot = sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(g1), jax.tree.leaves(g2)))
        return dot / jnp.maximum(optax.global_norm(g1)*optax.global_norm(g2), 1e-30)

    parent_states, errors = {}, []
    logits = np.asarray(sequence(parent))
    for b, (record, obs) in enumerate(rows):
        actor.reset()
        own = 0
        for i, d in enumerate(record['decisions']):
            if b == 0 and i in (record['roots']['combo'], record['roots']['position']):
                parent_states[i] = actor.states[1]
            online, _, _ = actor.predict({k: v[i:i+1] for k, v in obs.items()}, d['player'])
            if d['player'] == 1:
                n = len(d['menu'])
                errors.append(float(np.abs(online[:n]-logits[2*own+b, :n]).max()))
                own += 1
    assert max(errors) < .002
    (a.output/'sequence-parity.json').write_text(json.dumps(dict(passed=True, max_error=max(errors))))
    if not a.execute:
        print('Preflight passed; optimizer disabled', flush=True)
        return

    def gradient_probe():
        gradients = [grad(actor.params, jnp.eye(4)[i])[1] for i in range(4)]
        result = dict(norms=[float(optax.global_norm(g)) for g in gradients],
                      cosines={f'{i}-{j}': float(cosine(gradients[i], gradients[j]))
                               for i, j in ((0, 1), (0, 2), (0, 3), (1, 3), (2, 3))},
                      order=['combo_suffix', 'battle_suffix', 'combo_prefix', 'Halq_NLL'])
        return result

    def probe(folder, update, gradients=False):
        logits = np.asarray(sequence(actor.params))
        decisions = []
        for d in descriptors:
            p = probabilities(logits[d['batch_index'], :len(d['menu'])])
            decisions.append(dict(**d, probabilities=p.tolist(),
                                  reference_probability=float(p[d['action']]), greedy=int(p.argmax())))
        fixed_state = {}
        for step, state in parent_states.items():
            obs = {k: v[step:step+1] for k, v in rows[0][1].items()}
            _, held_logits, _ = infer(actor.params, obs, state)
            d = rows[0][0]['decisions'][step]
            fixed_state[str(step)] = probabilities(np.asarray(held_logits)[0, :len(d['menu'])]).tolist()
        evaluations = []
        for kind, branch, source, depths in [('combo', 'reference', a.combo, ('opening', 'combo', 'position')),
                                              ('battle', 'direct', a.battle, ('opening', 'battle'))]:
            reference = json.loads((source/f'v0-{branch}-0.json').read_text())
            for depth in depths:
                try:
                    record, arrays = execute(native, reference['definition'], a.semantics, branch, actor, depth)
                    name = f'{kind}-v0-{depth}-u{update}-0'
                    save_record(folder, name, record, arrays)
                    evaluations.append(dict(kind=kind, depth=depth, **record['result']))
                except ValueError as error:
                    if str(error) != 'exercise decision budget exhausted':
                        raise
                    evaluations.append(dict(kind=kind, depth=depth, success=False, status='budget_exhausted'))
        result = dict(update=update, component_losses=np.asarray(component_values(actor.params)).tolist(),
                      decisions=decisions, fixed_parent_hidden_state_diagnostic=fixed_state,
                      live=evaluations)
        if gradients:
            result['gradient_probe'] = gradient_probe()
        (folder/f'probe-{update}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(arm=folder.parent.name, update=update,
                              losses=result['component_losses'], live=evaluations)), flush=True)
        return result

    baseline = a.output/'baseline'
    baseline.mkdir()
    probe(baseline, 0, True)
    summaries = []
    for arm in plan['arms']:
        arm_start = time.monotonic()
        if arm_start-start > plan['max_total_seconds']:
            raise TimeoutError('study total budget')
        folder = a.output/arm['name']
        folder.mkdir()
        probes = folder/'probes'
        probes.mkdir()
        actor.params = parent
        weights = jnp.asarray(arm['weights']+[0.], dtype=jnp.float32)
        optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(arm['learning_rate']))
        state = optimizer.init(parent)

        @jax.jit
        def update(params, state):
            value, gradient = jax.value_and_grad(lambda pp: jnp.dot(components(pp), weights))(params)
            changes, state = optimizer.update(gradient, state, params)
            return optax.apply_updates(params, changes), state, value, optax.global_norm(gradient)

        with (folder/'metrics.jsonl').open('x') as metrics:
            for step in range(1, plan['updates_per_arm']+1):
                if time.monotonic()-arm_start > plan['max_seconds_per_arm'] or time.monotonic()-start > plan['max_total_seconds']:
                    raise TimeoutError('study optimization/evaluation budget')
                actor.params, state, value, norm = update(actor.params, state)
                row = dict(update=step, loss=float(value), grad_norm=float(norm), seconds=time.monotonic()-arm_start)
                assert np.isfinite([row['loss'], row['grad_norm']]).all()
                metrics.write(json.dumps(row)+'\n')
                metrics.flush()
                if step in plan['probe_updates']:
                    probe(probes, step, step in plan['gradient_probe_updates'])
        checkpoint = folder/'model.flax_model'
        checkpoint.write_bytes(flax.serialization.to_bytes(actor.params))
        write_checkpoint_metadata(checkpoint, observation_schema='structured-lite-v1',
            model_args=ModelArgs(observation_schema='structured-lite-v1'),
            code_list_hash=sha(a.code_list), semantic_table_hash=sha(a.semantics/'metadata.json'),
            capacities=dict(max_cards=80, max_options=128, history_actions=32, public_events=32, group_references=8),
            training_context=dict(experiment=plan['id'], arm=arm, updates=plan['updates_per_arm'],
                                  parent_sha256=sha(a.checkpoint), promotion_eligible=False))
        # Strict compatibility and serialization checks, retaining the already
        # compiled dynamic-parameter inference executable for the many probes.
        from ygoai.rl.checkpoint_compat import validate_checkpoint_compatibility
        validate_checkpoint_compatibility(checkpoint, observation_schema='structured-lite-v1',
            model_args=ModelArgs(observation_schema='structured-lite-v1'), code_list_hash=sha(a.code_list),
            semantic_table_hash=sha(a.semantics/'metadata.json'),
            capacities=dict(max_cards=80, max_options=128, history_actions=32, public_events=32, group_references=8))
        actor.params = flax.serialization.from_bytes(parent, checkpoint.read_bytes())
        endpoint = evaluate(a, native, actor, arm['name']+'-endpoint')
        for kind, source, branch in [('combo', a.combo, 'reference'), ('battle', a.battle, 'direct')]:
            for variant in range(3):
                definition = json.loads((source/f'v{variant}-{branch}-0.json').read_text())['definition']
                records = []
                for repeat in range(2):
                    try:
                        record, arrays = execute(native, definition, a.semantics, branch, actor, 'opening')
                        save_record(a.output/(arm['name']+'-endpoint'), f'{kind}-v{variant}-opening-{repeat}', record, arrays)
                    except ValueError as error:
                        if str(error) != 'exercise decision budget exhausted':
                            raise
                        record = dict(result=dict(success=False, status='budget_exhausted'))
                    records.append(record)
                assert records[0] == records[1]
                endpoint.append(dict(kind=kind, variant=variant, depth='opening', **record['result']))
        summary = dict(arm=arm, endpoint=endpoint, checkpoint_sha256=sha(checkpoint),
                       seconds=time.monotonic()-arm_start, updates=plan['updates_per_arm'])
        (folder/'summary.json').write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
    assert all(sha(Path(path)) == digest for path, digest in protected.items())
    result = dict(arms=summaries, total_seconds=time.monotonic()-start,
                  protected_unchanged=True, promotion_eligible=False)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
