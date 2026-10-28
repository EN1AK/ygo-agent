"""Migrate frozen replay tensors to the isolated selection-counts-v1 runtime.

Every original action, menu, response and endpoint must remain identical. Only
declared select/unselect count fields may change. Raw prompt bytes, rather than
policy behavior, verify the corrected public inputs. No model inference occurs.
"""
import argparse
import copy
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from contracts import read, sha, write

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts/studies/crystron_curriculum'))


def expected_counts(events):
    raw = bytes.fromhex(next(e['raw'] for e in reversed(events) if e['op'] == 26))
    nselect = raw[6]
    offset = 7 + 8 * nselect
    if offset >= len(raw):
        raise ValueError('truncated select/unselect prompt')
    return dict(minimum=raw[4], maximum=raw[5], selected_count=raw[offset],
                must_continue=int(not raw[2] and not raw[3]))


def assert_allowed_difference(old, new, message):
    import numpy as np
    if old.keys() != new.keys():
        raise ValueError('input tensor keys changed')
    for name in old:
        if old[name].shape != new[name].shape or old[name].dtype != new[name].dtype:
            raise ValueError('input tensor shape/dtype changed: ' + name)
        changed = np.not_equal(old[name], new[name])
        if message == 26 and name == 'selection_':
            changed = changed.copy()
            changed[..., [5, 6, 7, 8, 11]] = False
        if changed.any():
            raise ValueError('unexpected input change: ' + name)


def main():
    import numpy as np
    from scripts.collect_teaching_exercise import load_native
    from ygoai.rl.exercise_actor import array_digest, extract_trace, hidden_identity_audit
    from ygoai.rl.human_replay import comparable_game_events
    from ygoai.rl.observation_schema import tensor_contract
    from ygoai.rl import exercise_core
    import ygoai.rl.exercise_actor as trace_module
    trace_module.messages = exercise_core.messages
    from supervision_runtime import Runtime
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen', 'native', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.frozen, args.native, args.output = args.frozen.resolve(), args.native.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = args.output / 'runtime'
    runtime.mkdir()
    plan = read(args.frozen / 'plan.json')
    old_runtime = Path(plan['runtime'])
    if sha(args.native) == plan['native_sha256']:
        raise ValueError('migration requires a new isolated native module')
    for name in ('assets', 'semantics', 'code_list.txt', 'registration.ydk'):
        (runtime / name).symlink_to((old_runtime / name).resolve(), target_is_directory=(old_runtime/name).is_dir())
    native_path = runtime / 'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    shutil.copy2(args.native, native_path)
    config = SimpleNamespace(native=native_path, database=runtime/'assets/cards.cdb',
        scripts=runtime/'assets/script', code_list=runtime/'code_list.txt', semantics=runtime/'semantics')
    native = load_native(config, runtime/'registration.ydk')
    source = Path(plan['source_root'])
    sys.path.insert(0, str(source))
    from modern_core import Core
    independent = SimpleNamespace(Core=Core, source=source, args=config)
    (args.output/'rows').mkdir()
    rows, audits = [], []
    total_probes = total_changed = 0
    for item in plan['rows']:
        path = args.frozen/item['path']
        if sha(path) != item['record_sha256'] or sha(path.with_suffix('.npz')) != item['observations_sha256']:
            raise ValueError('original frozen inputs changed')
        original = read(path)
        record = copy.deepcopy(original)
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
            old = dict(arrays)
        d = record['definition']
        env = native.ExerciseActor(d.get('lua', ''), d['initial'], str(config.semantics),
            d.get('seed', 1), 'structured-lite-v1', d['seed_sequence'], d['duel_flags'], 2048, True)
        observations, probes, changed = [], [], 0
        try:
            env.reset()
            for step, decision in enumerate(record['decisions']):
                snap = env.snapshot()
                if snap['invalid'] or snap['done']:
                    raise ValueError('migration replay ended early')
                if (snap['message'], snap['player'], snap['menu']) != (
                        decision['message'], decision['player'], decision['menu']):
                    raise ValueError('migration changed original legal prompt/menu')
                obs = {k: snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                previous = {k: value[step:step+1] for k, value in old.items()}
                if array_digest(previous) != decision['observation_sha256']:
                    raise ValueError('original record/tensor mismatch')
                assert_allowed_difference(previous, obs, snap['message'])
                if step == 0:
                    hidden_identity_audit(obs)
                if snap['message'] == 26:
                    events, _, _ = extract_trace(env, snap)
                    expected = expected_counts(events)
                    s = obs['selection_'][0]
                    actual = dict(minimum=int(s[5]), maximum=int(s[6]), selected_count=int(s[8]), must_continue=int(s[11]))
                    if actual != expected or s[7] != 0:
                        raise ValueError('new selection counts do not match core')
                    probes.append(dict(step=step, expected=expected, actual=actual))
                current_hash = array_digest(obs)
                changed += current_hash != decision['observation_sha256']
                decision['observation_sha256'] = current_hash
                observations.append({k: value[0].copy() for k, value in obs.items()})
                env.step(decision['action'])
            snap = env.snapshot()
            events, state, responses = extract_trace(env, snap)
            if state != original['final'] or responses != original['core_responses']:
                raise ValueError('migration changed responses or final state')
            if comparable_game_events(events) != comparable_game_events(original['events']):
                raise ValueError('migration changed game events')
            record.update(core_seed=env.trace()['core_seed'], events=events, final=state, core_responses=responses)
        finally:
            env.close()
        proof = Runtime.independent(independent, record)
        if not proof['passed']:
            raise ValueError('independent migration replay failed')
        record.update(native_sha256=sha(native_path), input_behavior='selection-counts-v1',
            input_migration=dict(source_record_sha256=item['record_sha256'],
                source_observations_sha256=item['observations_sha256'],
                changed_decisions=changed, independent_audit=proof))
        target = args.output/'rows'/(item['id']+'.json')
        write(target, record)
        np.savez_compressed(target.with_suffix('.npz'), **{k: np.stack([v[k] for v in observations]) for k in observations[0]})
        rows.append(dict(item, path=str(target.relative_to(args.output)),
            record_sha256=sha(target), observations_sha256=sha(target.with_suffix('.npz'))))
        audits.append(dict(id=item['id'], changed_decisions=changed, probes=probes, independent_audit=proof))
        total_changed += changed
        total_probes += len(probes)
        print(item['id'], 'changed', changed, 'count_probes', len(probes), flush=True)
    plan.update(runtime=str(runtime), native_sha256=sha(native_path), rows=rows,
        input_behavior='selection-counts-v1',
        input_migration=dict(original_plan_sha256=sha(args.frozen/'plan.json'),
            original_native_sha256=read(args.frozen/'plan.json')['native_sha256'],
            source_sha256=sha(__file__), independent_replay=True,
            allowed_selection_columns=[5, 6, 7, 8, 11]))
    write(args.output/'plan.json', plan)
    write(args.output/'migration-audit.json', dict(passed=True, rows=audits,
        input_rows=len(rows), count_probes=total_probes, changed_decisions=total_changed,
        actions_menus_responses_and_endpoints_unchanged=True, no_model_inference=True))
    print('MIGRATION', len(rows), total_probes, total_changed, flush=True)


if __name__ == '__main__':
    main()
