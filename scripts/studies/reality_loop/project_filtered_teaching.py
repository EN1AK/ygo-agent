"""Rebuild teaching inputs with contextual filtering and stable chain menus.

Re-index actions and recurrent histories through the real adapter. Erase only
observed identical-state/menu/event UI loops. Preserve committed game events
and endpoints, then independently audit the new raw-core route. Chain response
indices are aliases, verified against physical card/effect choices. This pass
performs no model inference or trajectory search.
"""
import argparse
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

from contracts import read, sha, write
from navigation_routes import committed_events,loop_erasure,chain_menu_permutation,align_action_rows,selection_navigation_responses
from extra_deck_aliases import unique_extra_cards,board_and_extra_inventory,extra_key,extra_menu_permutation,normalize_extra_events

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
    from goals import achieved
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
    if getattr(native,'chain_menu_order_version',None)!='semantic-spec-card-effect-v1':
        raise ValueError('deterministic chain-menu native required')
    source = Path(plan['source_root'])
    sys.path.insert(0, str(source))
    from modern_core import Core
    independent = SimpleNamespace(Core=Core, source=source, args=config)
    admitted, coverage = [], []
    for item in plan['rows']:
        path = a.frozen/item['path']
        assert sha(path) == item['record_sha256'] and sha(path.with_suffix('.npz')) == item['observations_sha256']
        original = read(path)
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as source_arrays:
            source_inputs=dict(source_arrays)
        record = copy.deepcopy(original)
        d = record['definition']
        unique=unique_extra_cards(d)
        prefix_events=lambda events:committed_events(normalize_extra_events(events,unique,prefix=True))
        final_events=lambda events:committed_events(normalize_extra_events(events,unique))
        def environment(full):
            return native.ExerciseActor(d.get('lua', ''),d['initial'],str(config.semantics),
                d.get('seed',1),'structured-lite-v1',d['seed_sequence'],d['duel_flags'],2048,full)
        # First replay the immutable full-menu source through the new header.
        # The diagnostic callback is global, so close it before the next duel.
        frames, contexts, response_counts = [], [], []
        env = environment(True)
        result = dict(id=item['id'], split=item['split'], source_record_sha256=item['record_sha256'])
        try:
            env.reset()
            for index in range(len(original['decisions'])+1):
                snap=env.snapshot();events,state,responses=extract_trace(env,snap)
                if 'summon_navigation' not in snap:
                    raise ValueError('contextual-navigation native is required')
                frames.append(dict(state=state,message=snap['message'],player=snap['player'],
                    menu=snap['menu'],selection=snap['observation']['selection_'].tolist(),
                    committed_events=prefix_events(events)))
                contexts.append(snap['summon_navigation'])
                response_counts.append(len(responses))
                if index==len(original['decisions']):break
                old=original['decisions'][index]
                obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                if (snap['message'],snap['player']) != (old['message'],old['player']):
                    raise ValueError('full_source_input_changed')
                permutation=chain_menu_permutation(snap['message'],old['menu'],snap['menu'])
                aligned=align_action_rows(obs,permutation)
                if array_digest(aligned)!=old['observation_sha256'] or not all(
                        np.array_equal(v[0],source_inputs[k][index]) for k,v in aligned.items()):
                    raise ValueError('non-permutation source input changed')
                env.step(permutation[old['action']])
            if len(responses)!=len(original['core_responses']) or state!=original['final']:
                raise ValueError('full_source_endpoint_changed')
            allowed={x['core_response_count'] for x in original['decisions'] if x['message']==16}
            if any(x!=y and i not in allowed for i,(x,y) in enumerate(zip(responses,original['core_responses']))):
                raise ValueError('non-chain source response changed')
            if committed_events(events)!=committed_events(original['events']):
                raise ValueError('full source committed events changed')
            env.close()
            undo,cancel=selection_navigation_responses(events,responses)
            def markers(decisions,frames,contexts,counts,undo,cancel):
                result=[]
                for i,decision in enumerate(decisions):
                    action=decision['menu'][decision['action']]
                    pending=contexts[i] or (decision['message']==11 and action['act'] in ('Summon','SpSummon','MSet'))
                    interval=range(counts[i],counts[i+1])
                    result.append(bool(action['unselect'] or action not in policy_menu(decision['message'],frames[i]['menu'],contexts[i]) or
                        any(j in undo or (pending and j in cancel) for j in interval)))
                return result
            removable=markers(original['decisions'],frames,contexts,response_counts,undo,cancel)
            loop_frames=[dict(f,state=board_and_extra_inventory(f['state']),
                menu=sorted([extra_key(a,f['player'],unique) for a in f['menu']],key=lambda a:str(sorted(a.items())))) for f in frames]
            retained, replacements, loops = loop_erasure(loop_frames,removable)
            expected_menus={i:policy_menu(original['decisions'][i]['message'],frames[i]['menu'],contexts[i]) for i in retained}
            for i in retained:
                old=original['decisions'][i]
                if old['menu'][old['action']] not in expected_menus[i]:
                    raise ValueError('navigation_without_certified_loop')
            env=environment(False)
            decisions, arrays, old_to_new = [], [], {}
            filtered_frames,filtered_contexts,filtered_counts=[],[],[]
            cursor = 0
            env.reset()
            while True:
                snap = env.snapshot()
                events, state, responses = extract_trace(env, snap)
                filtered_frames.append(dict(state=board_and_extra_inventory(state),message=snap['message'],player=snap['player'],
                    menu=sorted([extra_key(a,snap['player'],unique) for a in snap['menu']],key=lambda a:str(sorted(a.items()))),
                    selection=snap['observation']['selection_'].tolist(),committed_events=prefix_events(events)))
                filtered_contexts.append(snap['summon_navigation']);filtered_counts.append(len(responses))
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
                    try:permutation=extra_menu_permutation(expected_menus[index],snap['menu'],snap['player'],unique)
                    except ValueError:permutation=None
                    if permutation is not None and (snap['message'],snap['player'],board_and_extra_inventory(state),prefix_events(events)) == (
                        old['message'],old['player'],board_and_extra_inventory(frames[index]['state']),frames[index]['committed_events']):
                        matched = (position,index,permutation)
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
                position,index,permutation=matched
                old = original['decisions'][index]
                obs = {k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                if not decisions:
                    hidden_identity_audit(obs)
                action = permutation[expected_menus[index].index(old['menu'][old['action']])]
                step = len(decisions)
                old_to_new[index] = step
                decisions.append(dict(old, step=step, menu=snap['menu'], action=action,
                    teaching=bool(old['teaching'] and len(snap['menu']) > 1),
                    core_response_count=len(responses),
                    chain_prompt=next((e for e in reversed(events) if e['op']==16),None) if snap['message']==16 else None,
                    observation_sha256=array_digest(obs)))
                arrays.append({k:v[0].copy() for k,v in obs.items()})
                env.step(action)
                cursor = position + 1
            if board_and_extra_inventory(state) != board_and_extra_inventory(original['final']):
                raise ValueError('endpoint_changed')
            if final_events(events) != final_events(original['events']):
                raise ValueError('committed_game_events_changed')
            if not achieved(item['goal'],state,snap,events,d['controlled']):
                raise ValueError('navigation-normalized route does not achieve the original goal')
            remaining_undo,remaining_cancel=selection_navigation_responses(events,responses)
            # Actual menus are needed for marker membership; the stored frames
            # use canonical extra aliases only for observed loop comparison.
            marker_frames=[dict(f,menu=(decisions[i]['menu'] if i<len(decisions) else snap['menu'])) for i,f in enumerate(filtered_frames)]
            remaining_markers=markers(decisions,marker_frames,filtered_contexts,filtered_counts,remaining_undo,remaining_cancel)
            if remaining_undo or loop_erasure(filtered_frames,remaining_markers)[2]:
                raise ValueError('navigation_without_certified_loop')
            root = item.get('root_step', 0)
            seen=set()
            while root in replacements and root not in seen:
                seen.add(root);root=replacements[root]
            if root not in old_to_new:
                raise ValueError('root_became_automatic_requires_new_root_contract')
            record.update(decisions=decisions, events=events, final=state, core_responses=responses,
                core_seed=env.trace()['core_seed'], full_legal_menu=False,
                input_behavior='selection-counts-contextual-navigation-extra-v4',native_sha256=sha(config.native))
            proof = Runtime.independent(independent, record)
            assert proof['passed']
            record.update(training_ready=item['split']=='fit',teacher_scope='verified_local_goal_feasibility',globally_optimal_labels=False)
            record['navigation_projection'] = dict(source_record_sha256=item['record_sha256'],
                source_decisions=len(original['decisions']), projected_decisions=len(decisions),
                same_core_responses=responses==original['core_responses'],same_committed_game_events=True,
                erased_loops=loops,same_endpoint=state==original['final'],
                same_board_and_extra_inventory=True,goal_revalidated=True,
                event_alias_contract='unique-extra-cid-and-pending-xyz-v1',
                exact_rng_realization_equivalence=False,independent_audit=proof)
            record['navigation_projection'].update(remaining_core_unselect_responses=0,remaining_observed_navigation_loops=0)
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
        input_behavior='selection-counts-contextual-navigation-extra-v4',
        projection_source_plan_sha256=sha(a.frozen/'plan.json'),
        projection_code_sha256=sha(__file__),navigation_routes_sha256=sha(Path(__file__).with_name('navigation_routes.py')),
        extra_aliases_sha256=sha(Path(__file__).with_name('extra_deck_aliases.py')),
        training_admission_pending=True)
    write(a.output/'projected-plan.json', plan)


if __name__ == '__main__':
    main()
