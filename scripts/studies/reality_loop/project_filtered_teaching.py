"""Rebuild teaching inputs with the existing policy navigation filter.

Re-index actions and recurrent histories through the real adapter. Preserve
every core response, game event and endpoint. Routes needing a filtered human
undo are quarantined, never silently relabeled or counted as equivalent.
This conservative pass performs no model inference and no trajectory search.
"""
import argparse
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

from contracts import read, sha, write
from navigation_routes import committed_events, loop_erasure

PROJECT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PROJECT), str(PROJECT/'scripts/studies/crystron_curriculum')]


def policy_menu(message, menu, summon_navigation=False):
    """Mirror only filter_policy_navigational_back; actual menus are checked."""
    result = list(menu)
    if message == 18 and any(a['place'] != 0 for a in result):
        result = [a for a in result if not a['finish']]
    if message == 26 and any(a['finish'] or (not a['unselect'] and a['act'] != 'Cancel') for a in result):
        result = [a for a in result if not a['unselect']]
    if summon_navigation and message in (15, 20, 26) and any(a['act'] != 'Cancel' for a in result):
        result = [a for a in result if a['act'] != 'Cancel']
    return result


def main():
    import numpy as np
    from scripts.collect_teaching_exercise import load_native
    from ygoai.rl.exercise_actor import array_digest, extract_trace, hidden_identity_audit
    from ygoai.rl.human_replay import comparable_game_events
    from ygoai.rl.observation_schema import tensor_contract
    from ygoai.rl import exercise_core
    import ygoai.rl.exercise_actor as trace_module
    from supervision_runtime import Runtime
    trace_module.messages = exercise_core.messages
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    a = p.parse_args()
    a.frozen, a.output = a.frozen.resolve(), a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'rows').mkdir()
    plan = read(a.frozen/'plan.json')
    old_rt = Path(plan['runtime'])
    assert sha(old_rt/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so') == plan['native_sha256']
    rt = a.output/'runtime'
    rt.mkdir()
    for name in ('assets','semantics','code_list.txt','registration.ydk'):
        (rt/name).symlink_to((old_rt/name).resolve(),target_is_directory=(old_rt/name).is_dir())
    import shutil
    shutil.copy2(a.native,rt/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so')
    config = SimpleNamespace(native=rt/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so',
        database=rt/'assets/cards.cdb', scripts=rt/'assets/script', code_list=rt/'code_list.txt', semantics=rt/'semantics')
    native = load_native(config, rt/'registration.ydk')
    source = Path(plan['source_root'])
    sys.path.insert(0, str(source))
    from modern_core import Core
    independent = SimpleNamespace(Core=Core, source=source, args=config)
    admitted, coverage = [], []
    for item in plan['rows']:
        path = a.frozen/item['path']
        assert sha(path) == item['record_sha256'] and sha(path.with_suffix('.npz')) == item['observations_sha256']
        original = read(path)
        record = copy.deepcopy(original)
        d = record['definition']
        def environment(full):
            return native.ExerciseActor(d.get('lua', ''),d['initial'],str(config.semantics),
                d.get('seed',1),'structured-lite-v1',d['seed_sequence'],d['duel_flags'],2048,full)
        # First replay the immutable full-menu source through the new header.
        # The diagnostic callback is global, so close it before the next duel.
        frames, contexts = [], []
        env = environment(True)
        result = dict(id=item['id'], split=item['split'], source_record_sha256=item['record_sha256'])
        try:
            env.reset()
            for index in range(len(original['decisions'])+1):
                snap=env.snapshot();events,state,responses=extract_trace(env,snap)
                if 'summon_navigation' not in snap:
                    raise ValueError('contextual-navigation native is required')
                frames.append(dict(state=state,message=snap['message'],player=snap['player'],
                    menu=snap['menu'],committed_events=committed_events(events)))
                contexts.append(snap['summon_navigation'])
                if index==len(original['decisions']):break
                old=original['decisions'][index]
                obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                if (snap['message'],snap['player'],snap['menu'],array_digest(obs)) != (
                    old['message'],old['player'],old['menu'],old['observation_sha256']):
                    raise ValueError('full_source_input_changed')
                env.step(old['action'])
            if responses!=original['core_responses'] or state!=original['final']:
                raise ValueError('full_source_endpoint_changed')
            env.close()
            retained, replacements, loops = loop_erasure(frames)
            expected_menus={i:policy_menu(original['decisions'][i]['message'],
                original['decisions'][i]['menu'],contexts[i]) for i in retained}
            for i in retained:
                old=original['decisions'][i]
                if old['menu'][old['action']] not in expected_menus[i]:
                    raise ValueError('navigation_without_certified_loop')
            env=environment(False)
            decisions, arrays, old_to_new = [], [], {}
            cursor = 0
            env.reset()
            while True:
                snap = env.snapshot()
                events, state, responses = extract_trace(env, snap)
                if cursor == len(retained):
                    break
                if snap['invalid']:
                    raise ValueError('early_endpoint')
                if snap['done']:
                    if all(expected_menus[i]==[original['decisions'][i]['menu'][original['decisions'][i]['action']]] for i in retained[cursor:]):
                        cursor=len(retained)
                        break
                    raise ValueError('early_endpoint')
                matched = None
                for position in range(cursor, len(retained)):
                    index=retained[position];old = original['decisions'][index]
                    if (snap['message'],snap['player'],snap['menu'],state,committed_events(events)) == (
                        old['message'],old['player'],expected_menus[index],frames[index]['state'],frames[index]['committed_events']):
                        matched = (position,index)
                        break
                    # Native can now automatically execute a unique forward
                    # action. No discretionary teacher action may be skipped.
                    if expected_menus[index] != [old['menu'][old['action']]]:
                        raise ValueError('nonforced_decision_disappeared')
                if matched is None:
                    # Every remaining source choice passed the unique-forward
                    # check above. The adapter already consumed this tail;
                    # exact responses/events/endpoint below still must match.
                    cursor = len(retained)
                    break
                position,index=matched
                old = original['decisions'][index]
                obs = {k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                if not decisions:
                    hidden_identity_audit(obs)
                action = snap['menu'].index(old['menu'][old['action']])
                step = len(decisions)
                old_to_new[index] = step
                decisions.append(dict(old, step=step, menu=snap['menu'], action=action,
                    teaching=bool(old['teaching'] and len(snap['menu']) > 1),
                    core_response_count=len(responses), observation_sha256=array_digest(obs)))
                arrays.append({k:v[0].copy() for k,v in obs.items()})
                env.step(action)
                cursor = position + 1
            if state != original['final']:
                raise ValueError('endpoint_changed')
            if committed_events(events) != committed_events(original['events']):
                raise ValueError('committed_game_events_changed')
            root = item.get('root_step', 0)
            seen=set()
            while root in replacements and root not in seen:
                seen.add(root);root=replacements[root]
            if root not in old_to_new:
                raise ValueError('root_became_automatic_requires_new_root_contract')
            record.update(decisions=decisions, events=events, final=state, core_responses=responses,
                core_seed=env.trace()['core_seed'], full_legal_menu=False,
                input_behavior='selection-counts-contextual-navigation-v2',native_sha256=sha(config.native))
            proof = Runtime.independent(independent, record)
            assert proof['passed']
            record['navigation_projection'] = dict(source_record_sha256=item['record_sha256'],
                source_decisions=len(original['decisions']), projected_decisions=len(decisions),
                same_core_responses=responses==original['core_responses'],same_committed_game_events=True,
                erased_loops=loops,same_endpoint=True,independent_audit=proof)
            target = a.output/'rows'/(item['id']+'.json')
            write(target, record)
            np.savez_compressed(target.with_suffix('.npz'), **{k:np.stack([v[k] for v in arrays]) for k in arrays[0]})
            own = sum(x['player'] == d['controlled'] for x in decisions)
            admitted.append(dict(item, path=str(target.relative_to(a.output)), root_step=old_to_new[root],
                own_steps=own, record_sha256=sha(target), observations_sha256=sha(target.with_suffix('.npz'))))
            result.update(status='admitted',decisions=len(decisions),erased_loops=loops,independent_audit=proof)
        except Exception as exc:
            result.update(status='quarantined', reason=str(exc))
        finally:
            env.close()
        coverage.append(result)
        print(item['id'], result['status'], result.get('reason', ''), flush=True)
    write(a.output/'projection-coverage.json', dict(input_rows=len(plan['rows']), admitted=len(admitted),
        quarantined=len(coverage)-len(admitted), rows=coverage, no_model_inference=True,
        no_rejected_route_is_a_negative_label=True, equivalent_route_search_pending=True))
    # This is a preparation artifact. Final curriculum mixing/admission and a
    # new runtime contract are required before training; never reuse old plans.
    plan.update(rows=admitted,runtime=str(rt),native_sha256=sha(config.native),full_legal_menu=False,
        input_behavior='selection-counts-contextual-navigation-v2',
        projection_source_plan_sha256=sha(a.frozen/'plan.json'),
        projection_code_sha256=sha(__file__),navigation_routes_sha256=sha(Path(__file__).with_name('navigation_routes.py')),
        training_admission_pending=True)
    write(a.output/'projected-plan.json', plan)


if __name__ == '__main__':
    main()
