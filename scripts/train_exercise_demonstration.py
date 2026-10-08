"""Opt-in bounded sequence teaching. Default: audit and frozen evaluation only."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from scripts.collect_teaching_exercise import execute, load_native
from scripts.eval_capability_exercises import FrozenActor, run as synthetic_run
from ygoai.rl.exercise_core import sha
from ygoai.rl.exercise_starters import CASES


def verified_record(directory, audit_path, name, native):
    audit = json.loads(audit_path.read_text())
    assert audit['passed'] and audit['native_sha256'] == sha(native)
    proof = next(row for row in audit['results'] if row['path'] == name)
    path = directory / name
    assert sha(path) == proof['sha256']
    assert sha(path.with_suffix('.npz')) == proof['observations_sha256']
    record = json.loads(path.read_text())
    assert record['normal_opening'] and record['result']['success']
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as f:
        obs = dict(f)
    return record, obs


def pack(rows):
    """Each seat has independent memory: retain its complete observed prefix."""
    length = 32
    selected, masks, actions = [], [], []
    for record, obs in rows:
        indices = [i for i, d in enumerate(record['decisions']) if d['player'] == 1]
        assert 1 < len(indices) <= length
        root = record['roots']['position' if record['definition']['kind'] == 'combo' else 'battle']
        mask = np.array([float(i >= root) for i in indices] + [0.] * (length-len(indices)), np.float32)
        mask /= mask.sum() # fixed equal weight per capability, not per suffix length
        actions.append([record['decisions'][i]['action'] for i in indices] + [0] * (length-len(indices)))
        padded = indices + [indices[-1]] * (length-len(indices))
        selected.append({k: v[padded] for k, v in obs.items()})
        masks.append(mask)
    # The original agent expects time-major flattened observations.
    tensors = {k: np.stack([o[k] for o in selected], axis=1).reshape(
        (-1,) + selected[0][k].shape[1:]) for k in selected[0]}
    return tensors, np.array(actions).T.reshape(-1), np.array(masks).T.reshape(-1)


def evaluate(args, native, actor, label):
    folder = args.output / label
    folder.mkdir()
    results = []
    for kind, source, branch in [('combo', args.combo, 'reference'), ('battle', args.battle, 'direct')]:
        for variant in range(3):
            reference = json.loads((source / f'v{variant}-{branch}-0.json').read_text())
            for depth in (('position', 'combo') if kind == 'combo' else ('battle',)):
                records = []
                for repeat in range(2):
                    r, arrays = execute(native, reference['definition'], args.semantics,
                                        branch, actor, depth=depth)
                    name = f'{kind}-v{variant}-{depth}-{repeat}'
                    (folder / (name + '.json')).write_text(json.dumps(r, indent=2), encoding='utf-8')
                    np.savez_compressed(folder / (name + '.npz'), **arrays)
                    records.append(r)
                assert records[0] == records[1], 'policy repeat mismatch'
                result = dict(kind=kind, variant=variant, depth=depth, **r['result'])
                results.append(result)
                print(json.dumps(dict(stage=label, **result)), flush=True)
    # Original fixtures are evaluation-only; none enters the supervised batch.
    for case, definition in CASES.items():
        branch = next(iter(definition['branches']))
        records = []
        for repeat in range(2):
            record, arrays = synthetic_run(args, native, actor, case, branch, 'policy')
            (folder / f'synthetic-{case}-{repeat}.json').write_text(json.dumps(record, indent=2))
            np.savez_compressed(folder / f'synthetic-{case}-{repeat}.npz', **arrays)
            records.append(record)
        assert records[0] == records[1]
        results.append(dict(kind='synthetic', case=case, status=record['status']))
    (folder / 'summary.json').write_text(json.dumps(results, indent=2))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('native', 'database', 'code-list', 'scripts', 'semantics', 'checkpoint',
                'combo', 'combo-audit', 'battle', 'battle-audit', 'runtime-manifest', 'plan', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--execute', action='store_true', help='Explicitly enable the predeclared 32 updates')
    a = p.parse_args()
    for k, v in vars(a).items():
        if isinstance(v, Path): setattr(a, k, v.resolve())
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    start = time.monotonic()
    plan = json.loads(a.plan.read_text())
    assert sha(a.checkpoint) == plan['parent_sha256']
    assert sha(a.native) == plan['exercise_native_sha256']
    runtime = json.loads(a.runtime_manifest.read_text())
    assert runtime['passed'] and runtime['artifacts_before'] == runtime['artifacts_after']
    assets = {path: digest for path, digest in runtime['artifacts_before'].items()
              if '/scripts/script/' in path or str(a.semantics) in path or path in
              (str(a.database), str(a.code_list), str(a.checkpoint), str(a.checkpoint)+'.metadata.json')}
    assert len(assets) > 1000, 'runtime asset manifest incomplete'
    assert assets == {path: sha(path) for path in assets}, 'runtime asset drift'
    protected = {str(path): sha(path) for path in (a.checkpoint, a.native, a.database, a.code_list,
                  a.combo_audit, a.battle_audit, a.plan)}
    rows = [verified_record(a.combo, a.combo_audit, 'v0-reference-0.json', a.native),
            verified_record(a.battle, a.battle_audit, 'v0-direct-0.json', a.native)]
    manifest = dict(experiment='bulb-position-learnability-v1', seed=8102026,
        updates=32, learning_rate=0.0001, max_grad_norm=1., optimizer='Adam',
        sequence_length=32, batch_size=2, prefix='current parameters, full own-seat prefix, full BPTT',
        loss='equal-weight successful combo suffix and Hayate battle suffix cross entropy',
        fitting_variants=[0], development_variants_not_fitted=[1, 2], heldout_families=[],
        success_gate='all 3 normal position roots complete two attacks and clear board; all 3 Hayate roots retained',
        no_strength_promotion=True, no_reward_changes=True, max_optimization_seconds=600,
        protected=protected, enabled=a.execute, plan_sha256=sha(a.plan),
        controls='parent; ordinary PPO continuation and equal-budget ordinary updates configured separately')
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copyfile(a.checkpoint, a.output / 'parent.flax_model')
    shutil.copyfile(str(a.checkpoint)+'.metadata.json', a.output / 'parent.flax_model.metadata.json')
    native = load_native(a, a.combo / 'registration.ydk')
    tensors, actions, weights = pack(rows)
    sample = {k: v[:1] for k, v in tensors.items()}
    import jax
    # Use full float32 products for the sequential/batched history audit.
    jax.config.update('jax_default_matmul_precision', 'highest')
    actor = FrozenActor(a.output / 'parent.flax_model', a.code_list, a.semantics, sample)
    import jax.numpy as jnp
    import flax.serialization
    import optax
    from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
    from ygoai.rl.jax.agent import ModelArgs
    infer = jax.jit(lambda params, obs, state: actor.agent.apply(params, obs, state)[:3])
    actor.forward = lambda obs, state: infer(actor.params, obs, state)
    print(json.dumps(dict(devices=[str(d) for d in jax.devices()], manifest=manifest)), flush=True)
    before = evaluate(a, native, actor, 'before')
    batch = jax.tree.map(jnp.asarray, tensors)
    targets, mask = jnp.asarray(actions), jnp.asarray(weights)
    def sequence(params):
        return actor.agent.apply(params, batch, actor.agent.init_rnn_state(2),
                                 jnp.zeros(len(actions), dtype=bool), None)[:3]
    sequence_forward = jax.jit(sequence)
    _, sequence_logits, _ = sequence_forward(actor.params)
    errors = []
    for b, (r, obs) in enumerate(rows):
        actor.reset()
        own = 0
        for i, d in enumerate(r['decisions']):
            logits, _, _ = actor.predict({k: v[i:i+1] for k,v in obs.items()}, d['player'])
            if d['player'] == 1:
                n = len(d['menu'])
                errors.append(float(np.abs(logits[:n] - np.asarray(sequence_logits)[own*2+b, :n]).max()))
                own += 1
    parity = max(errors)
    assert parity < 0.002, f'sequence / online RNN mismatch: {parity}'
    (a.output / 'sequence-parity.json').write_text(json.dumps(dict(max_abs_logit_error=parity, passed=True)))
    if not a.execute:
        print('preflight complete; optimizer disabled', flush=True)
        return
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(1e-4))
    state = optimizer.init(actor.params)
    def loss(params):
        _, logits, _ = sequence(params)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask).sum() / mask.sum()
    @jax.jit
    def update(params, state):
        value, grad = jax.value_and_grad(loss)(params)
        changes, state = optimizer.update(grad, state, params)
        return optax.apply_updates(params, changes), state, value, optax.global_norm(grad)
    optimization_start = time.monotonic()
    with (a.output / 'metrics.jsonl').open('x') as f:
        for step in range(32):
            if time.monotonic() - optimization_start > 600:
                raise TimeoutError('declared optimization time budget exceeded')
            actor.params, state, value, norm = update(actor.params, state)
            metric = dict(update=step+1, loss=float(value), grad_norm=float(norm),
                          seconds=time.monotonic()-optimization_start)
            assert np.isfinite([metric['loss'], metric['grad_norm']]).all(), 'nonfinite optimization'
            f.write(json.dumps(metric)+'\n'); f.flush()
            print(json.dumps(metric), flush=True)
    checkpoint = a.output / 'taught.flax_model'
    checkpoint.write_bytes(flax.serialization.to_bytes(actor.params))
    write_checkpoint_metadata(checkpoint, observation_schema='structured-lite-v1', model_args=ModelArgs(observation_schema='structured-lite-v1'),
        code_list_hash=sha(a.code_list), semantic_table_hash=sha(a.semantics/'metadata.json'),
        capacities=dict(max_cards=80, max_options=128, history_actions=32, public_events=32, group_references=8),
        training_context=dict(experiment=manifest['experiment'], parent_sha256=sha(a.checkpoint),
                              manifest_sha256=sha(a.output/'manifest.json'), updates=32, promotion_eligible=False))
    # Reload the written checkpoint using the normal compatibility path.
    actor = FrozenActor(checkpoint, a.code_list, a.semantics, sample)
    optimization_seconds = time.monotonic()-optimization_start
    after = evaluate(a, native, actor, 'after')
    assert protected == {path: sha(path) for path in protected}, 'protected input changed'
    assert assets == {path: sha(path) for path in assets}, 'runtime assets changed'
    passed = all(r['success'] for r in after if r['kind']=='battle' or
                 (r['kind']=='combo' and r['depth']=='position'))
    result = dict(learnability_gate_passed=passed, before=before, after=after, updates=32,
        checkpoint_sha256=sha(checkpoint), total_seconds=time.monotonic()-start,
        optimization_seconds=optimization_seconds, parent_unchanged=True,
        full_game_strength='not established', unseen_family_transfer='not measured')
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
