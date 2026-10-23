"""Frozen passive-scene coverage comparison; diagnostic routes never become lessons."""
import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scripts.collect_teaching_exercise import load_native
from scripts.eval_capability_exercises import FrozenActor
from scripts.study_elfnote_search import evaluate
from scripts.teach_elfnote_combo import audit_actor, load_row, save
from ygoai.rl.elfnote_combo import GOALS, grade
from ygoai.rl.elfnote_search import discover, run
from ygoai.rl.exercise_actor import array_digest, extract_trace, hidden_identity_audit
from ygoai.rl.exercise_core import sha
from ygoai.rl.observation_schema import tensor_contract
from ygoai.rl.scene_batch import pack_scenes


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def variant(source, settings, name, split, index):
    case = copy.deepcopy(source)
    case.update(name=name, split=split, intervention=None, lineage='elfnote-exposed-development',
                transformation=settings, source_case=source['name'])
    d = case['definition']
    before = Counter(tuple(x) for x in d['initial'])
    deck = [x for x in d['initial'] if x[1:3] == [0, 1]]
    if settings['swap_resource']:
        hand_card = next(x for x in deck[-5:] if x[0] == 59438930)
        replacement = next(x for x in deck[:-5] if x[0] == 14558128)
        hand_card[0], replacement[0] = replacement[0], hand_card[0]
    hand = [x[0] for x in deck[-5:]]
    offset = settings['rotate']
    hand = hand[offset:] + hand[:offset]
    for card, code in zip(deck[-5:], hand):
        card[0] = code
    if settings['seed'] is not None:
        d['seed_sequence'] = np.random.default_rng(settings['seed'] + index).integers(0, 2**32, 8).tolist()
    assert Counter(tuple(x) for x in d['initial']) == before, 'variation changed registration'
    case['definition_sha256'] = digest(d)
    return case


def make_cases(plan, previous):
    cases, bases, diagnostics = [], [], []
    for index, pair in enumerate(plan['pairs_train']):
        source = next(c for c in previous if c['name'] == pair + '-none')
        for i, settings in enumerate(plan['train_variants']):
            case = variant(source, settings, f'{pair}-train-{i}', 'train', index)
            cases.append(case)
            if i == 0:
                bases.append(case['name'])
        for i, settings in enumerate(plan['evaluation_variants']):
            cases.append(variant(source, settings, f'{pair}-fresh-{i}', 'fresh', index))
    for index, pair in enumerate(plan['pairs_diagnostic']):
        source = next(c for c in previous if c['name'] == pair + '-none')
        case = copy.deepcopy(source)
        case.update(name=pair + '-exposed', split='diagnostic', lineage='elfnote-exposed-development')
        diagnostics.append(case)
        cases.append(case)
        for i, settings in enumerate(plan['evaluation_variants']):
            cases.append(variant(source, settings, f'{pair}-fresh-{i}', 'fresh', index + 10))
    assert len(cases) == 38 and len(bases) == 4
    assert len({digest(c['definition']) for c in cases}) == len(cases), 'duplicate condition'
    return cases, bases, diagnostics


def mapped_action(snap, teacher):
    if (snap['message'], snap['player']) != (teacher['message'], teacher['player']):
        raise ValueError('source prompt divergence')
    chosen = teacher['menu'][teacher['action']]
    keys = ['code', 'act', 'phase', 'finish', 'position', 'effect', 'unselect', 'place']
    candidates = [i for i, m in enumerate(snap['menu']) if all(m[k] == chosen[k] for k in keys)]
    exact = [i for i in candidates if snap['menu'][i]['spec'] == chosen['spec']]
    if exact:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    # Numeric deck aliases with identical complete metadata are interchangeable.
    if candidates and str(chosen['spec']).isdigit() and all(str(snap['menu'][i]['spec']).isdigit() for i in candidates):
        return candidates[0]
    raise ValueError('ambiguous or absent source action')


def transfer(native, semantics, actor, case, plan, source):
    d = case['definition']
    env = native.ExerciseActor('', d['initial'], str(semantics), 1, 'structured-lite-v1',
                              d['seed_sequence'], d['duel_flags'], plan['max_decisions'] + 2)
    actor.reset()
    decisions, arrays = [], []
    record = dict(schema='ygo-elfnote-search-route-v1', definition=d, case=case['name'],
                  training_ready=False, source='semantic transfer of frozen search witness')
    try:
        env.reset()
        for step in range(plan['max_decisions']):
            snap = env.snapshot()
            events, state, responses = extract_trace(env, snap)
            outcomes = [grade(state, goal) for goal in case['goals']]
            ended = sum(e['op'] == 40 for e in events) >= 2
            if (ended or snap['message'] == 11) and any(o['success'] for o in outcomes):
                record.update(status='success', outcomes=outcomes)
                break
            if snap['invalid']:
                raise RuntimeError('invalid native fixture')
            if ended or snap['done'] or step >= len(source['decisions']):
                record.update(status='source_not_applicable', outcomes=outcomes)
                break
            try:
                action = mapped_action(snap, source['decisions'][step])
            except ValueError as exc:
                record.update(status='source_not_applicable', error=str(exc), outcomes=outcomes)
                break
            obs = {k: snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
            if step == 0:
                hidden_identity_audit(obs)
            logits, value, rnn = actor.predict(obs, snap['player'])
            decisions.append(dict(step=step, player=snap['player'], message=snap['message'],
                menu=snap['menu'], action=action, observation_sha256=array_digest(obs),
                logits=logits[:len(snap['menu'])].tolist(), value=value, rnn_sha256=rnn,
                core_response_count=len(responses), teaching=snap['player'] == 0))
            arrays.append({k: v[0].copy() for k, v in obs.items()})
            env.step(action)
        else:
            record.update(status='budget_exhausted', outcomes=outcomes)
    finally:
        snap = env.snapshot()
        events, state, responses = extract_trace(env, snap)
        record.update(decisions=decisions, events=events, final=state, core_responses=responses,
                      root_step=0, intervention_attempted=False, core_seed=env.trace()['core_seed'])
        env.close()
    return record, {k: np.stack([x[k] for x in arrays]) for k in arrays[0]} if arrays else {}


def checked(a, native, actor, case, plan, record, obs, folder, name):
    repeated, arr = run(native, a.semantics, actor, case, plan, prefix=record['decisions'])
    # Transferred routes have separate provenance metadata; physical evidence must match.
    keys = ['decisions', 'events', 'final', 'core_responses', 'status', 'root_step', 'core_seed']
    assert all(repeated[k] == record[k] for k in keys), name
    assert all(np.array_equal(obs[k], arr[k]) for k in obs), name
    path = save(folder, name, record, obs)
    audit_actor(a, path)
    return dict(path=str(path), record_sha256=sha(path), observations_sha256=sha(path.with_suffix('.npz')))


def collect(a, native, actor, diagnostic_params, plan, cases, bases, diagnostics, originals, old):
    started = time.monotonic()
    before = evaluate(a, native, actor, cases + originals, plan, old, a.output/'before', audit=True)
    parent = actor.params
    folder = a.output/'diagnostic'; folder.mkdir()
    actor.params = diagnostic_params
    diagnosis = []
    for case in diagnostics:
        baseline, obs = run(native, a.semantics, actor, case, plan)
        checked(a, native, actor, case, plan, baseline, obs, folder, case['name']+'-baseline')
        current = dict(plan, trials_per_case=plan['diagnostic_trials'], seconds_per_case=plan['diagnostic_seconds'])
        with (folder/(case['name']+'.trials.jsonl')).open('x') as f:
            def callback(row):
                f.write(json.dumps(row)+'\n'); f.flush()
            found, result = discover(native, a.semantics, actor, case, current, baseline, 'search', callback)
        if found:
            record, arr = run(native, a.semantics, actor, case, plan, prefix=found['decisions'])
            result.update(checked(a, native, actor, case, plan, record, arr, folder, case['name']+'-witness'))
            mismatch = next((i for i, (x, y) in enumerate(zip(baseline['decisions'], record['decisions'])) if x['action'] != y['action']), None)
            if mismatch is not None:
                b, w = baseline['decisions'][mismatch], record['decisions'][mismatch]
                assert all(b[k] == w[k] for k in ['observation_sha256', 'menu', 'logits', 'rnn_sha256'])
                branches = []
                for label, prefix in [('original', baseline['decisions'][:mismatch+1]), ('replacement', record['decisions'][:mismatch+1])]:
                    branch, arrays = run(native, a.semantics, actor, case, plan, prefix=prefix)
                    proof = checked(a, native, actor, case, plan, branch, arrays, folder, case['name']+'-'+label)
                    branches.append(dict(label=label, status=branch['status'], **proof))
                result['paired_root'] = dict(step=mismatch, original=b['menu'][b['action']], replacement=w['menu'][w['action']],
                    branches=branches, scope='first divergence followed by unchanged greedy policy; not a global preference or optimality label')
        diagnosis.append(result)
        write(folder/(case['name']+'.summary.json'), result)
        print(json.dumps(dict(stage='diagnosis', case=case['name'], status=result['status'], trials=len(result['trials']))), flush=True)
    actor.params = parent
    lessons = []; folder = a.output/'lessons'; folder.mkdir()
    for case in cases:
        if case['split'] != 'train':
            continue
        assert time.monotonic() - started < plan['max_collection_seconds']
        pair = case['source_case'].removesuffix('-none')
        source_path = a.previous/'search'/(pair+'-none.json')
        source = read(source_path)
        record, obs = transfer(native, a.semantics, actor, case, plan, source)
        attempt = dict(case=case['name'], source_sha256=sha(source_path), transfer_status=record['status'])
        if record['status'] != 'success':
            save(folder, case['name']+'-transfer-rejected', record, obs)
            baseline, _ = run(native, a.semantics, actor, case, plan)
            with (folder/(case['name']+'.trials.jsonl')).open('x') as f:
                def callback(row):
                    f.write(json.dumps(row)+'\n'); f.flush()
                found, search = discover(native, a.semantics, actor, case, plan, baseline, 'search', callback)
            attempt['search'] = search
            if found:
                record, obs = run(native, a.semantics, actor, case, plan, prefix=found['decisions'])
        if record['status'] == 'success':
            attempt.update(checked(a, native, actor, case, plan, record, obs, folder, case['name']))
        lessons.append(attempt)
        print(json.dumps(dict(stage='lesson', case=case['name'], admitted='path' in attempt, transfer=attempt['transfer_status'])), flush=True)
    admitted = len(lessons) == 24 and all('path' in x for x in lessons)
    write(a.output/'collection.json', dict(diagnosis=diagnosis, lessons=lessons, admitted=admitted,
          seconds=time.monotonic()-started, before=before))
    print(json.dumps(dict(collection_complete=True, admitted=admitted)), flush=True)


def train(a, native, actor, plan, cases, bases, originals, refs, old):
    import jax
    import jax.numpy as jnp
    import flax.serialization
    import optax
    from ygoai.rl.exercise_retention import parent_kl_rows
    from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
    from ygoai.rl.jax.agent import ModelArgs
    collection = read(a.output/'collection.json')
    assert collection['admitted'], '24 verified lessons required before optimization'
    manifest = read(a.output/'manifest.json')
    assert manifest['cases'] == cases and manifest['plan'] == plan
    parent = actor.params; summaries = []
    lesson_map = {x['case']: x for x in collection['lessons']}
    for mode in ['narrow', 'varied']:
        folder = a.output/mode; folder.mkdir(); started = time.monotonic()
        # Repetition makes the batch size and weighting comparable without adding information.
        names = [name for name in bases for _ in range(6)] if mode == 'narrow' else [c['name'] for c in cases if c['split'] == 'train']
        rows = []
        for name in names:
            entry = lesson_map[name]; path = Path(entry['path']); proof = read(path.with_name(path.stem+'.audit.json'))
            assert proof['passed'] and sha(path) == entry['record_sha256'] == proof['record_sha256']
            assert sha(path.with_suffix('.npz')) == entry['observations_sha256'] == proof['observations_sha256']
            record, obs = load_row(path)
            assert record['status'] == 'success' and record['definition'] == next(c['definition'] for c in cases if c['name'] == name)
            rows.append((record, obs))
        rows += refs + old; n = len(rows); assert n == 34
        tensors, targets, weights, legal = pack_scenes(rows, plan['sequence_steps'])
        batch = jax.tree.map(jnp.asarray, tensors)
        targets, weights, legal = jnp.asarray(targets), jnp.asarray(weights), jnp.asarray(legal)
        @jax.jit
        def sequence(params, batch):
            return actor.agent.apply(params, batch, actor.agent.init_rnn_state(n), jnp.zeros(plan['sequence_steps']*n, bool), None)[1]
        anchor = sequence(parent, batch)
        errors = []
        actor.params = parent
        for b, (record, obs) in enumerate(rows):
            actor.reset(); own = 0
            for t, d in enumerate(record['decisions']):
                logits, _, _ = actor.predict({k:v[t:t+1] for k,v in obs.items()}, d['player'])
                if d['player'] == record['definition']['controlled']:
                    count = len(d['menu']); errors.append(float(np.max(np.abs(logits[:count] - np.asarray(anchor)[n*own+b, :count])))); own += 1
        assert max(errors) < .002, max(errors)
        write(folder/'admission.json', dict(passed=True, lesson_names=names, lessons=24, histories=n, max_logit_error=max(errors)))
        tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(plan['learning_rate'])); state = tx.init(parent)
        def objective(params, batch, targets, weights, legal, anchor):
            logits = sequence(params, batch)
            ce = (weights*optax.softmax_cross_entropy_with_integer_labels(logits, targets).reshape(-1, n)).sum(0)
            kl = (weights*parent_kl_rows(logits, anchor, legal).reshape(-1, n)).sum(0)
            return ce[:24].mean()+plan['retention_coef']*kl[24:].mean(), (ce, kl)
        @jax.jit
        def update(params, state, batch, targets, weights, legal, anchor):
            (loss, terms), grad = jax.value_and_grad(objective, has_aux=True)(params, batch, targets, weights, legal, anchor)
            changes, state = tx.update(grad, state, params)
            params = optax.apply_updates(params, changes)
            finite = jnp.all(jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(params)]))
            return params, state, loss, terms, optax.global_norm(grad), finite
        with (folder/'metrics.jsonl').open('x') as f:
            for step in range(1, plan['updates_per_arm']+1):
                assert time.monotonic()-started < plan['max_train_seconds_per_arm']
                actor.params, state, loss, terms, norm, finite = update(actor.params, state, batch, targets, weights, legal, anchor)
                row = dict(update=step, loss=float(loss), ce=np.asarray(terms[0]).tolist(), kl=np.asarray(terms[1]).tolist(), grad_norm=float(norm))
                assert bool(finite) and np.isfinite([row['loss'], row['grad_norm']]+row['ce']+row['kl']).all()
                f.write(json.dumps(row)+'\n'); f.flush()
                if step in plan['probe_updates']:
                    evaluate(a, native, actor, [c for c in cases if c['name'] in bases]+originals, plan, old, folder/f'probe-{step}')
        path = folder/'model.flax_model'; path.write_bytes(flax.serialization.to_bytes(actor.params))
        write_checkpoint_metadata(path, observation_schema='structured-lite-v1', model_args=ModelArgs(observation_schema='structured-lite-v1'),
            code_list_hash=sha(a.code_list), semantic_table_hash=sha(a.semantics/'metadata.json'),
            capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),
            training_context=dict(experiment=plan['id'], mode=mode, parent_sha256=sha(a.checkpoint), plan_sha256=sha(a.plan), updates=plan['updates_per_arm'], promotion_eligible=False))
        actor.params = flax.serialization.from_bytes(parent, path.read_bytes())
        result = evaluate(a, native, actor, cases+originals, plan, old, folder/'endpoint', audit=True)
        summaries.append(dict(mode=mode, result=result, checkpoint_sha256=sha(path), seconds=time.monotonic()-started))
    write(a.output/'summary.json', dict(completed=True, arms=summaries, production_eligible=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'source', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--stage', choices=['collect', 'train'], required=True)
    p.add_argument('--collection-source', type=Path, help='Explicit immutable collection source for a compilation-only resume')
    a = p.parse_args()
    a.root, a.source, a.output = a.root.resolve(), a.source.resolve(), a.output.resolve()
    a.plan = a.source/'assets/exercises/teaching-v1/elfnote-coverage-study.json'; plan = read(a.plan)
    a.previous = a.root/plan['previous_study']; assert read(a.previous/'final-verification.json')['passed']
    release = a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008'
    intake = a.root/'training-runs/elfnote-selected-scope-intake-03-20261014'
    a.native = intake/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    a.database = release/'repo/assets/locale/zh/cards.cdb'; a.scripts = release/'repo/scripts/script'
    a.code_list, a.semantics = release/'code_list.txt', release/'semantics'
    parent_dir = a.root/'training-runs/elfnote-scope-joint-calibration-20261014/teach_retain'
    a.checkpoint = parent_dir/'model.flax_model'; diagnostic = a.previous/'search-trained/model.flax_model'
    assert sha(a.native) == plan['native_sha256'] and sha(a.checkpoint) == plan['parent_sha256'] and sha(diagnostic) == plan['diagnostic_sha256']
    protected = {str(p):sha(p) for p in [a.native, a.database, a.checkpoint, diagnostic, a.plan]}
    for pair in plan['pairs_train']:
        path = a.previous/'search'/(pair+'-none.json')
        proof = read(path.with_name(path.stem+'.audit.json'))
        assert proof['passed'] and proof['record_sha256'] == sha(path) and proof['observations_sha256'] == sha(path.with_suffix('.npz'))
        for dependency in [path, path.with_suffix('.npz'), path.with_name(path.stem+'.audit.json')]:
            protected[str(dependency)] = sha(dependency)
    source = read(a.source/'source-manifest.json')
    assert all(hashlib.sha256((a.source/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == value for name,value in source['all_files'].items())
    runtime = read(a.root/'training-runs/elfnote-runtime-binding-20261014.json')
    assert all(sha(a.root/name) == value for name,value in runtime.items())
    lesson_plan = read(intake/'joint-calibration-plan.json')
    refs = [load_row(intake/'actor-v1'/(e['name']+'.json')) for e in lesson_plan['demonstrations']]
    old = [load_row(parent_dir/f'endpoint/old/{i}.json') for i in range(5)]
    references = [intake/'actor-v1'/(e['name']+'.json') for e in lesson_plan['demonstrations']]
    references += [parent_dir/f'endpoint/old/{i}.json' for i in range(5)]
    for path in references:
        for dependency in [path, path.with_suffix('.npz')]:
            protected[str(dependency)] = sha(dependency)
    for i, (record, _) in enumerate(old):
        if i == 4:
            record['definition'] = dict(controlled=0)
        for d in record['decisions']:
            d['teaching'] = d['player'] == record['definition']['controlled']
    originals = [dict(name=e['name'], split='original', definition=r[0]['definition'], intervention=None, goals=[GOALS[e['goal']]]) for e,r in zip(lesson_plan['demonstrations'], refs)]
    cases, bases, diagnostics = make_cases(plan, read(a.previous/'manifest.json')['cases'])
    if a.stage == 'collect':
        a.output.mkdir(parents=True, exist_ok=False)
        write(a.output/'manifest.json', dict(plan=plan, plan_sha256=sha(a.plan), protected=protected, cases=cases,
              bases=bases, source=source, runtime=runtime, frozen_before_inference=True))
    else:
        m = read(a.output/'manifest.json')
        if a.collection_source:
            original = a.collection_source.resolve()
            assert read(original/'source-manifest.json') == m['source']
            assert all(hashlib.sha256((original/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in m['source']['all_files'].items())
            changed = {name for name,value in m['source']['all_files'].items() if source['all_files'].get(name)!=value}
            assert changed == {'scripts/study_elfnote_coverage.py', 'scripts/verify_elfnote_coverage.py'}, changed
            assert source['all_files'].keys() == m['source']['all_files'].keys()
            assert read(a.plan)==m['plan'] and sha(a.plan)==m['plan_sha256']
            previous_protected = {path:value for path,value in m['protected'].items() if path!=str(original/'assets/exercises/teaching-v1/elfnote-coverage-study.json')}
            current_protected = {path:value for path,value in protected.items() if path!=str(a.plan)}
            assert previous_protected == current_protected
            assert all(sha(Path(path))==value for path,value in m['protected'].items())
            assert read(a.output/'compile-attempt-01.json')['updates_executed']==0
            assert not (a.output/'narrow').exists() and not (a.output/'varied').exists()
            write(a.output/'training-source.json',dict(source=source,collection_source=str(original),training_source=str(a.source),
                  changed_files=sorted(changed),reason='Dynamic batch arguments replace compile-time constants; no data, objective, budget, architecture or input semantics change',protected=protected))
        else:
            assert m['source'] == source and m['protected'] == protected
    import jax
    import flax.serialization
    jax.config.update('jax_default_matmul_precision', 'highest')
    assert all(d.platform == 'gpu' for d in jax.devices())
    native = load_native(a, release/'opening-02/registration.ydk')
    actor = FrozenActor(a.checkpoint, a.code_list, a.semantics, {k:v[:1] for k,v in refs[0][1].items()})
    infer = jax.jit(lambda params, obs, state:actor.agent.apply(params, obs, state)[:3])
    actor.forward = lambda obs,state:infer(actor.params, obs, state)
    if a.stage == 'collect':
        diagnostic_params = flax.serialization.from_bytes(actor.params, diagnostic.read_bytes())
        collect(a, native, actor, diagnostic_params, plan, cases, bases, diagnostics, originals, old)
    else:
        train(a, native, actor, plan, cases, bases, originals, refs, old)
    assert all(sha(Path(path)) == value for path,value in protected.items())
    print(json.dumps(dict(stage=a.stage, completed=True, protected_unchanged=True)), flush=True)


if __name__ == '__main__':
    main()
