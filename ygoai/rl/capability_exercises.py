"""Provenance-preserving candidate intake. Suspicion is never an answer label."""
import hashlib
import json
from pathlib import Path

from ygoai.rl.exercise_starters import CASES

SCHEMA = 'capability-exercise-candidate-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_ref(path):
    p = Path(path)
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def draft(scene, *, case=None, source=None):
    if not isinstance(scene, str) or not scene.strip():
        raise ValueError('scene must be nonempty text')
    if case is not None and case not in CASES:
        raise ValueError('unknown starter template')
    definition = CASES.get(case, {})
    body = dict(schema=SCHEMA, scene=scene.strip(), source=source or {'kind': 'user_scene'},
                template=case, family=definition.get('family'),
                goal=definition.get('goal'), split='development',
                source_kind='synthetic_mechanism_fixture' if case else 'unresolved_scene',
                status='candidate', labels={}, training_ready=False,
                teacher_only=True, actor_interface_verified=False,
                opponent_assumption='starter scripted responses only' if case else None,
                budget={'max_decisions': 160, 'max_process_calls': 4096, 'repeats': 2},
                missing=[] if case else ['legal_root_or_explicit_synthetic_fixture',
                                         'goal_and_stable_boundary', 'opponent_conditions',
                                         'reference_and_counterexample'],
                lineage=[])
    if case:
        body['lineage'] = ['template:' + definition['family']]
    body['id'] = 'exercise-' + digest(body)[:20]
    return body


def validate_candidate(row):
    if row.get('schema') != SCHEMA or row.get('status') != 'candidate':
        raise ValueError('unsupported candidate schema/status')
    if row.get('labels') or row.get('training_ready') or not row.get('teacher_only'):
        raise ValueError('intake cannot grant labels or training eligibility')
    if row.get('template') is not None:
        case = row['template']
        if case not in CASES or row.get('family') != CASES[case]['family'] or row.get('goal') != CASES[case]['goal']:
            raise ValueError('template/goal mismatch')
        if row.get('source_kind') != 'synthetic_mechanism_fixture':
            raise ValueError('starter templates must remain synthetic')
    identity = {k: v for k, v in row.items() if k != 'id'}
    if row.get('id') != 'exercise-' + digest(identity)[:20]:
        raise ValueError('candidate identity mismatch; create a new revision')
    return row


def mine_decisions(path, *, steps=(), max_candidates=48):
    """Works with eval_structured/collect_live_interactions JSONL.

Raw indices aren't portable replay responses. A one-seat WindBot log doesn't
contain opponent history, so it stays quarantined pending engine alignment.
"""
    if max_candidates < 1:
        raise ValueError('candidate cap must be positive')
    path = Path(path)
    source = file_ref(path)
    wanted = set(steps)
    selected, rejected, seen_steps, seen_states = [], [], set(), {}
    deferred = 0
    with path.open(encoding='utf-8-sig') as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get('record_type') != 'decision':
                continue
            step = row.get('step')
            if not isinstance(step, int) or step < 0 or step in seen_steps:
                raise ValueError('duplicate/invalid step; mine one duel per input')
            seen_steps.add(step)
            actions = row.get('legal_actions', [])
            indices = [a['index'] if isinstance(a, dict) else a for a in actions]
            if len(set(indices)) != len(indices) or row.get('selected_action') not in indices:
                rejected.append({'line': line_no, 'step': step, 'reason': 'invalid_recorded_action'})
                continue
            reasons = []
            if step in wanted:
                reasons.append('user_flagged_decision')
            if row.get('cycle_guard_intervened'):
                reasons.append('cycle_guard_intervened')
            if isinstance(row.get('suspicions'), list):
                reasons += ['reported:' + str(s) for s in row['suspicions'] if isinstance(s, str)]
            state_hash = row.get('observation_digest')
            if isinstance(state_hash, str) and len(state_hash) == 64:
                key = (row.get('player'), row.get('turn'), state_hash, digest(actions))
                if key in seen_states:
                    reasons.append(f'repeated_observation_and_menu:step-{seen_states[key]}')
                seen_states[key] = step
            if not reasons:
                continue
            if len(selected) >= max_candidates:
                deferred += 1
                continue
            candidate = draft('复查实战选择；怀疑原因：' + '；'.join(reasons), source={
                'kind': 'decision_log', **source, 'line': line_no, 'step': step,
                'decision_sha256': digest(row), 'checkpoint_sha256': row.get('checkpoint_sha256')})
            candidate.update(source_kind='recorded_decision_candidate',
                             lineage=['duel:' + source['sha256']],
                             suspicion_reasons=reasons,
                             recorded_decision={'player': row.get('player'), 'legal_actions': actions,
                                                'selected_action': row['selected_action']},
                             missing=['both_players_legal_replay_history', 'runtime_and_assets',
                                      'root_observation_menu_alignment', 'goal_and_stable_boundary',
                                      'counterfactual_outcomes', 'current_checkpoint_rnn_reconstruction'])
            metadata = path.parent / 'fixtures' / f'step-{step:06d}.json'
            if metadata.exists():
                candidate['actor_fixture_metadata'] = file_ref(metadata)
                # Hash reference only: old checkpoint RNN is never training input.
            candidate.pop('id')
            candidate['id'] = 'exercise-' + digest(candidate)[:20]
            selected.append(validate_candidate(candidate))
    absent = sorted(wanted - seen_steps)
    if absent:
        raise ValueError(f'requested steps absent: {absent}')
    return {'schema': 'capability-exercise-intake-v1', 'source': source,
            'candidates': selected, 'rejected': rejected, 'candidate_cap': max_candidates,
            'decisions_scanned': len(seen_steps), 'deferred_by_cap': deferred, 'training_ready': False}


def assign_splits(rows):
    """Union all shared lineage keys before assigning a deterministic split.

    Starter/development roots pin their entire connected component to development.
    Callers must include duel, seed group, template and combo-suffix lineage.
    """
    parents = list(range(len(rows)))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate candidate identity')

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        if not row.get('lineage'):
            raise ValueError('missing lineage; cannot split')
        for key in row['lineage']:
            if key in owners:
                parents[find(i)] = find(owners[key])
            owners[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    result = []
    for members in groups.values():
        keys = sorted({k for row in members for k in row['lineage']})
        group = digest(keys)
        bucket = int(group[:8], 16) % 10
        split = 'train' if bucket < 8 else 'validation' if bucket == 8 else 'test'
        if any(r.get('split') == 'development' for r in members):
            split = 'development'
        result.extend({'id': r['id'], 'group': group, 'split': split} for r in members)
    return sorted(result, key=lambda x: x['id'])


def validate_verification(summary, directory, *, expected_runtime=None):
    """Verify evidence integrity; this grants mechanism verification, NOT training."""
    if summary.get('schema') != 'capability-exercise-verification-v1' or not summary.get('passed'):
        raise ValueError('verification did not pass')
    results = summary.get('results', [])
    if not results:
        raise ValueError('missing evidence')
    root = Path(directory).resolve()
    runtime = None
    script_versions = {}
    for result in results:
        expected = 'success' if CASES[result['case']]['branches'][result['branch']] else 'verified_failure'
        if result.get('actual') != expected or not result.get('passed') or not result.get('identical_replays'):
            raise ValueError('branch verification mismatch')
        evidence = result.get('evidence', [])
        if len(evidence) != 2 or evidence[0]['path'] == evidence[1]['path']:
            raise ValueError('two distinct replay records required')
        runs = []
        for ref in evidence:
            path = (root / ref['path']).resolve()
            if not path.is_relative_to(root) or file_ref(path)['sha256'] != ref['sha256']:
                raise ValueError('evidence hash/path mismatch')
            run = json.loads(path.read_text(encoding='utf-8'))
            if (run['status'] != expected or not run['stable_boundary'] or run['root_event_index'] is None
                    or run.get('error') or not run.get('scripts_sha256') or not run.get('runtime')
                    or not run.get('root') or not run.get('decisions')
                    or run['case'] != result['case'] or run['branch'] != result['branch']):
                raise ValueError('invalid replay evidence')
            if runtime is None:
                runtime = run['runtime']
            if run['runtime'] != runtime or (expected_runtime is not None and run['runtime'] != expected_runtime):
                raise ValueError('runtime mismatch')
            for script, version in run['scripts_sha256'].items():
                if script == '__exercise_fixture.lua':
                    continue
                if script in script_versions and script_versions[script] != version:
                    raise ValueError('script runtime mismatch')
                script_versions[script] = version
            runs.append(run)
        if runs[0] != runs[1]:
            raise ValueError('replay drift')
    return {'mechanism_verified': True, 'training_ready': False,
            'remaining_gate': 'original actor observations/actions and current-parameter RNN alignment'}
