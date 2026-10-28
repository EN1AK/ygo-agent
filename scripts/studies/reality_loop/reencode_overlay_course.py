"""Re-encode an admitted filtered course through the overlay-corrected adapter.

Retain every source row, action, raw response and engine event. Only public
event position features may change. Rebuild all tensors and observation hashes;
independently replay every resulting record before binding the new course.
No model inference, trajectory search or teacher selection occurs here.
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


def main():
    import numpy as np
    from scripts.collect_teaching_exercise import load_native
    from ygoai.rl.exercise_actor import array_digest, extract_trace
    from ygoai.rl.observation_schema import tensor_contract
    from ygoai.rl.core_audit import audit_record, validate_curriculum_binding

    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'native', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.source, a.output = a.source.resolve(), a.output.resolve()
    source_sha = sha(a.source / 'plan.json')
    plan = read(a.source / 'plan.json')
    validate_curriculum_binding(plan)
    if plan.get('full_legal_menu') is not False or plan.get('training_admission_pending'):
        raise ValueError('an admitted filtered course is required')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'rows').mkdir()
    runtime = a.output / 'runtime'
    runtime.mkdir()
    old_runtime = Path(plan['runtime'])
    for name in ('assets', 'semantics', 'code_list.txt', 'registration.ydk'):
        (runtime / name).symlink_to((old_runtime / name).resolve(), target_is_directory=(old_runtime / name).is_dir())
    target_native = runtime / 'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    shutil.copy2(a.native, target_native)
    config = SimpleNamespace(native=target_native, database=runtime / 'assets/cards.cdb',
        scripts=runtime / 'assets/script', code_list=runtime / 'code_list.txt', semantics=runtime / 'semantics')
    native = load_native(config, runtime / 'registration.ydk')
    if getattr(native, 'overlay_event_position_version', None) != 'material-index-is-not-battle-position-v1':
        raise ValueError('overlay-corrected native required')
    revision = copy.deepcopy(plan)
    proofs = []
    changes = []
    for row in revision['rows']:
        path = (a.source / row['path']).resolve()
        if sha(path) != row['record_sha256'] or sha(path.with_suffix('.npz')) != row['observations_sha256']:
            raise ValueError('source teacher changed')
        record = read(path)
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as f:
            old_arrays = dict(f)
        d = record['definition']
        env = native.ExerciseActor(d.get('lua', ''), d['initial'], str(config.semantics),
            d.get('seed', 1), 'structured-lite-v1', d['seed_sequence'], d['duel_flags'], 2048, False)
        arrays, changed_steps = [], []
        try:
            env.reset()
            for index, decision in enumerate(record['decisions']):
                snap = env.snapshot()
                if (snap['message'], snap['player'], snap['menu']) != (decision['message'], decision['player'], decision['menu']):
                    raise ValueError('decision/menu changed: ' + row['id'])
                obs = {k: snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                prior = {k: v[index:index + 1] for k, v in old_arrays.items()}
                if array_digest(prior) != decision['observation_sha256']:
                    raise ValueError('old observation hash mismatch')
                changed = False
                for key, value in obs.items():
                    old = prior[key]
                    if value.shape != old.shape:
                        raise ValueError('tensor shape changed')
                    if np.array_equal(value, old):
                        continue
                    if key != 'public_events_':
                        raise ValueError('unexpected changed tensor: ' + key)
                    allowed = old.copy()
                    allowed[..., 5] = value[..., 5]
                    if not np.array_equal(value, allowed):
                        raise ValueError('non-position event feature changed')
                    changed = True
                if changed:
                    changed_steps.append(index)
                decision['observation_sha256'] = array_digest(obs)
                arrays.append({k: v[0].copy() for k, v in obs.items()})
                env.step(decision['action'])
            events, state, responses = extract_trace(env, env.snapshot())
            if events != record['events'] or state != record['final'] or responses != record['core_responses']:
                raise ValueError('engine trajectory changed: ' + row['id'])
        finally:
            env.close()
        record.update(native_sha256=sha(target_native), input_behavior='filtered-overlay-event-position-v1')
        proof = audit_record(record, plan['source_root'], config.database, config.scripts, plan['independent_library'])
        if not proof['passed']:
            raise ValueError('independent replay rejected: ' + row['id'])
        record['overlay_event_reencoding'] = dict(source_record_sha256=row['record_sha256'],
            source_observations_sha256=row['observations_sha256'], changed_steps=changed_steps,
            identical_actions_responses_events_endpoint=True, model_inference=False)
        target = a.output / 'rows' / (row['id'] + '.json')
        write(target, record)
        np.savez_compressed(target.with_suffix('.npz'), **{k: np.stack([v[k] for v in arrays]) for k in arrays[0]})
        row.update(path=str(target), record_sha256=sha(target), observations_sha256=sha(target.with_suffix('.npz')))
        proofs.append(dict(id=row['id'], record_sha256=row['record_sha256'], audit=proof))
        changes.append(dict(id=row['id'], changed_steps=changed_steps))
        print(row['id'], 'verified', len(changed_steps), flush=True)
    if sha(a.source / 'plan.json') != source_sha:
        raise ValueError('source plan changed during replay')
    proof_path = a.output / 'independent-replay-admission.json'
    write(proof_path, dict(passed=True, source_plan=str(a.source / 'plan.json'), source_plan_sha256=source_sha,
        independent_library=plan['independent_library'], rows=proofs, native_sha256=sha(target_native),
        teacher_data_changed=False, model_inference=False))
    revision.update(runtime=str(runtime), native_sha256=sha(target_native), input_behavior='filtered-overlay-event-position-v1',
        independent_compatibility_admission=dict(path=str(proof_path), sha256=sha(proof_path)),
        overlay_event_reencoding=dict(source_plan=str(a.source / 'plan.json'), source_plan_sha256=source_sha,
            all_rows_retained=True, rows=changes, model_inference=False))
    validate_curriculum_binding(revision)
    write(a.output / 'plan.json', revision)
    print('ADMITTED', len(proofs), 'changed_steps', sum(len(r['changed_steps']) for r in changes), flush=True)


if __name__ == '__main__':
    main()
