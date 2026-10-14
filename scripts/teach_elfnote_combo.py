"""Bounded first-turn human-combo learnability study; preflight is the default."""
import argparse
import copy
import hashlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from scripts.collect_teaching_exercise import load_native, execute
from scripts.convert_human_replay import convert
from scripts.eval_capability_exercises import FrozenActor, run as synthetic_run
from scripts.teach_three_capabilities import save, BRANCHES
from ygoai.rl.elfnote_combo import GOALS, grade, rollout
from ygoai.rl.exercise_core import sha
from ygoai.rl.exercise_retention import parent_kl_rows
from ygoai.rl.scene_batch import pack_scenes


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def load_row(path):
    record = json.loads(path.read_text(encoding='utf-8'))
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as f:
        obs = dict(f)
    return record, obs


def audit_actor(a, path):
    proof = path.with_name(path.stem + '.audit.json')
    subprocess.run([sys.executable, '-m', 'scripts.audit_human_replay', '--actor-record',
                    '--native', str(a.native), '--database', str(a.database), '--scripts', str(a.scripts),
                    '--input', str(path), '--output', str(proof)], check=True,
                   cwd=Path(__file__).resolve().parents[1],
                   env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])))
    return json.loads(proof.read_text(encoding='utf-8'))


def old_evaluate(a, native, actor, rows, folder):
    folder.mkdir()
    result = []
    for i, (reference, _) in enumerate(rows):
        if i == 4:
            record, obs = synthetic_run(a, native, actor, 'combo', 'reference', 'policy')
            success = record['status'] == 'success'
        else:
            definition = reference['definition']
            record, obs = execute(native, definition, a.semantics, BRANCHES[definition['kind']][0], actor, 'opening')
            success = record['result']['success']
        save(folder, str(i), record, obs)
        result.append(dict(name=str(i), success=success))
    return result


def reserved_cases(plan, rows):
    references = {entry['name']: row[0]['definition'] for entry, row in zip(plan['demonstrations'], rows)}
    result = []
    for source in plan['reserved']:
        case = copy.deepcopy(source)
        definition = copy.deepcopy(references[case['source_name']])
        if case['kind'] == 'rng_order':
            definition['seed_sequence'] = case['seed_sequence']
        elif case['kind'] == 'two_starters':
            deck = [x for x in definition['initial'] if x[1:3] == [0, 1]]
            replacement = next(x for x in deck[:-5] if x[0] == case['second_starter'])
            replacement[0], deck[-2][0] = deck[-2][0], replacement[0]
        elif case['kind'] == 'disruption':
            deck = [x for x in definition['initial'] if x[1:3] == [1, 1]]
            deck[-1][0] = case['intervention']
        else:
            raise ValueError('unknown reserved transformation')
        assert hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest() == case['definition_sha256']
        case['definition'] = definition
        result.append(case)
    return result


def evaluate(a, native, actor, plan, rows, old, folder, *, reserved=False, audit=False):
    folder.mkdir()
    result = []
    cases = [dict(name=x['name'], kind='original', definition=r[0]['definition'], goals=[GOALS[x['goal']]])
             for x, r in zip(plan['demonstrations'], rows)]
    if reserved:
        cases += reserved_cases(plan, rows)
    for i, case in enumerate(cases):
        record, obs = rollout(native, a.semantics, actor, case['definition'], case['goals'],
                              intervention=case.get('intervention'))
        path = save(folder, f'case-{i:02}', record, obs)
        if audit:
            repeated, _ = rollout(native, a.semantics, actor, case['definition'], case['goals'],
                                   intervention=case.get('intervention'))
            if repeated != record:
                raise ValueError('autonomous rollout repeat mismatch')
            audit_actor(a, path)
        result.append(dict(name=case['name'], kind=case['kind'], status=record['status'],
                           outcomes=record['outcomes'], intervention_attempted=record['intervention_attempted']))
    old_results = old_evaluate(a, native, actor, old, folder/'old')
    write(folder/'summary.json', dict(cases=result, old=old_results))
    print(json.dumps(dict(stage=str(folder), original=sum(x['status']=='success' for x in result[:5]),
                          old=sum(x['success'] for x in old_results))), flush=True)
    return dict(cases=result, old=old_results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('release', 'intake', 'old', 'checkpoint', 'plan', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    for k, v in vars(a).items():
        if isinstance(v, Path): setattr(a, k, v.resolve())
    a.output.mkdir(parents=True, exist_ok=False)
    a.native = a.intake/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    a.database = a.release/'repo/assets/locale/zh/cards.cdb'
    a.scripts = a.release/'repo/scripts/script'
    a.code_list, a.semantics = a.release/'code_list.txt', a.release/'semantics'
    plan = json.loads(a.plan.read_text(encoding='utf-8'))
    assert sha(a.native) == plan['native_sha256'] and sha(a.checkpoint) == plan['parent_sha256']
    protected = {str(x): sha(x) for x in (a.native, a.database, a.checkpoint, a.code_list, a.plan)}
    write(a.output/'manifest.json', dict(plan=plan, plan_sha256=sha(a.plan), protected=protected, execute=a.execute,
                                        optimizer='new Adam state per arm', source_sha256=sha(__file__)))
    start = time.monotonic()
    native = load_native(a, a.release/'opening-02/registration.ydk')
    rows, old = [], []
    admission = a.output/'admission'; admission.mkdir()
    for entry in plan['demonstrations']:
        path = a.intake/'actor-v1'/(entry['name']+'.json')
        assert sha(path) == entry['actor_record_sha256']
        record, obs = load_row(path)
        assert record['status'] == 'converted' and grade(record['final'], GOALS[entry['goal']])['success']
        a.input = a.intake/'replay-v2'/(entry['name']+'.json')
        repeated, repeated_obs = convert(a, native, json.loads(a.input.read_text(encoding='utf-8')))
        assert repeated == record and all(np.array_equal(obs[k], repeated_obs[k]) for k in obs)
        saved = save(admission, entry['name'], record, obs)
        audit_actor(a, saved)
        rows.append((record, obs))
    for i, entry in enumerate(plan['retention_anchors']):
        path = a.old/entry['name']
        assert sha(path) == entry['sha256'] and sha(path.with_suffix('.npz')) == entry['observations_sha256']
        record, obs = load_row(path)
        if i == 4:
            record = copy.deepcopy(record); record['definition'] = dict(controlled=0)
        for decision in record['decisions']:
            decision['teaching'] = decision['player'] == record['definition']['controlled']
        old.append((record, obs))
    import jax
    import jax.numpy as jnp
    import flax.serialization
    import optax
    from ygoai.rl.jax.agent import ModelArgs
    from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
    jax.config.update('jax_default_matmul_precision', 'highest')
    assert any(d.platform == 'gpu' for d in jax.devices())
    tensors, targets, weights, legal = pack_scenes(rows+old, plan['sequence_steps'])
    actor = FrozenActor(a.checkpoint, a.code_list, a.semantics, {k:v[:1] for k,v in tensors.items()})
    parent = actor.params
    infer = jax.jit(lambda params, obs, state: actor.agent.apply(params, obs, state)[:3])
    actor.forward = lambda obs, state: infer(actor.params, obs, state)
    # Prove the rebuilt default RNG/adapter preserves old scene observations.
    for i, (reference, expected) in enumerate(old):
        if i == 4:
            record, obs = synthetic_run(a, native, actor, 'combo', 'reference', 'policy')
        else:
            last = len(reference['decisions'])-1
            record, obs = execute(native, reference['definition'], a.semantics,
                                  BRANCHES[reference['definition']['kind']][0], actor, 'teacher',
                                  replay_record=reference, replay_until=last, root_action=reference['decisions'][last]['action'])
        assert all(np.array_equal(obs[k], expected[k]) for k in expected), 'old adapter observation drift'
        assert record['events'] == reference['events'] and record['core_responses'] == reference['core_responses']
    batch = jax.tree.map(jnp.asarray, tensors)
    targets, weights, legal = jnp.asarray(targets), jnp.asarray(weights), jnp.asarray(legal)
    @jax.jit
    def sequence(params, batch):
        return actor.agent.apply(params, batch, actor.agent.init_rnn_state(10),
                                 jnp.zeros(plan['sequence_steps']*10, bool), None)[1]
    anchor = sequence(parent, batch)
    errors = []
    for b, (record, obs) in enumerate(rows+old):
        actor.reset(); own = 0
        for t, decision in enumerate(record['decisions']):
            logits, _, _ = actor.predict({k:v[t:t+1] for k,v in obs.items()}, decision['player'])
            if decision['player'] == record['definition']['controlled']:
                n = len(decision['menu'])
                errors.append(float(np.max(np.abs(logits[:n]-np.asarray(anchor)[10*own+b,:n])))); own += 1
    assert max(errors) < .002, max(errors)
    write(a.output/'parity.json', dict(passed=True, max_logit_error=max(errors), old_observations_identical=True))
    before = evaluate(a, native, actor, plan, rows, old, a.output/'before', reserved=a.execute, audit=True)
    if not a.execute:
        return
    summaries = []
    for arm in plan['arms']:
        folder = a.output/arm['name']; folder.mkdir(); arm_start = time.monotonic()
        actor.params = parent
        tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(plan['learning_rate']))
        state = tx.init(parent)
        def objective(params, batch, targets, weights, legal, anchor):
            logits = sequence(params, batch)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets).reshape(-1, 10)
            kl = parent_kl_rows(logits, anchor, legal).reshape(-1, 10)
            ces, kls = (weights*ce).sum(0), (weights*kl).sum(0)
            return ces[:5].mean()+arm['retention_coef']*kls[5:].mean(), (ces, kls)
        @jax.jit
        def update(params, state, batch, targets, weights, legal, anchor):
            (value, terms), grad = jax.value_and_grad(objective, has_aux=True)(params, batch, targets, weights, legal, anchor)
            changes, state = tx.update(grad, state, params)
            candidate = optax.apply_updates(params, changes)
            finite = jnp.all(jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(candidate)]))
            return candidate, state, value, terms, optax.global_norm(grad), finite
        with (folder/'metrics.jsonl').open('x') as f:
            for step in range(1, plan['updates_per_arm']+1):
                assert time.monotonic()-start < plan['max_total_seconds']
                assert time.monotonic()-arm_start < plan['max_seconds_per_arm']
                candidate, state, value, terms, norm, finite = update(actor.params, state, batch, targets, weights, legal, anchor)
                metric = dict(update=step, loss=float(value), ce=np.asarray(terms[0]).tolist(),
                              kl=np.asarray(terms[1]).tolist(), grad_norm=float(norm))
                assert bool(finite) and np.isfinite([metric['loss'], metric['grad_norm']]+metric['ce']+metric['kl']).all()
                actor.params = candidate
                f.write(json.dumps(metric)+'\n'); f.flush()
                if step in plan['probe_updates']:
                    evaluate(a, native, actor, plan, rows, old, folder/f'probe-{step}')
                    print(json.dumps(dict(arm=arm['name'], **metric)), flush=True)
        checkpoint = folder/'model.flax_model'
        checkpoint.write_bytes(flax.serialization.to_bytes(actor.params))
        write_checkpoint_metadata(checkpoint, observation_schema='structured-lite-v1',
            model_args=ModelArgs(observation_schema='structured-lite-v1'), code_list_hash=sha(a.code_list),
            semantic_table_hash=sha(a.semantics/'metadata.json'),
            capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),
            training_context=dict(experiment=plan['id'], parent_sha256=sha(a.checkpoint), updates=plan['updates_per_arm'],
                                  arm=arm, plan_sha256=sha(a.plan), promotion_eligible=False))
        actor.params = flax.serialization.from_bytes(parent, checkpoint.read_bytes())
        result = evaluate(a, native, actor, plan, rows, old, folder/'endpoint', reserved=True, audit=True)
        summaries.append(dict(arm=arm, result=result, checkpoint_sha256=sha(checkpoint), seconds=time.monotonic()-arm_start))
    assert all(sha(path) == value for path,value in protected.items())
    write(a.output/'summary.json', dict(before=before, arms=summaries, seconds=time.monotonic()-start, protected_unchanged=True))
    print('study complete', flush=True)


if __name__ == '__main__':
    main()
