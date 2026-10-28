"""Read-only diagnosis of post-PPO regressions; never create wrong-action labels."""
import argparse
from collections import Counter
from pathlib import Path
from contracts import read, write, sha, first_deviation, diagnose


def common_executed_prefix(left, right):
    count = 0
    for actual, reference in zip(left['decisions'], right['decisions']):
        if any(actual.get(k) != reference.get(k) for k in ('observation_sha256', 'menu', 'player', 'action')):
            break
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'exam', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    exam = read(a.exam / 'protocol.json')
    ppo = read(exam['ppo_protocol'])
    if sha(exam['ppo_protocol']) != exam['ppo_protocol_sha256']:
        raise ValueError('fixed PPO protocol changed')
    course = read(Path(ppo['course']) / 'plan.json')
    items = {r['id']: r for r in course['rows']}
    study = Path(ppo['study_output'])
    selected = {r['id'] for r in read(study / 'rehearsal-rehearse.json')['rows']}
    before_path = study / 'before/evaluation/summary.json'
    baseline_proof = read(study / 'before/verification.json')
    if (not baseline_proof['passed'] or baseline_proof['ppo_protocol_sha256'] != exam['ppo_protocol_sha256'] or
            baseline_proof['summary_sha256'] != sha(before_path)):
        raise ValueError('pre-PPO baseline proof changed')
    completion_path = a.exam / 'rehearse-final-development-completion.json'
    completion = read(completion_path)
    if (completion['protocol_sha256'] != sha(a.exam/'protocol.json') or
            sha(completion['summary']) != completion['summary_sha256'] or
            not all(sha(path) == value for path, value in completion['record_artifacts'].items())):
        raise ValueError('completed development evidence changed')
    after_path = Path(completion['summary'])
    key = lambda row: (row['id'], row['seed'])
    before = {key(r): r for r in read(before_path)['results']}
    after = {key(r): r for r in read(after_path)['results']}
    if before.keys() != after.keys():
        raise ValueError('development matrices differ')
    endpoint = read(study / 'rehearse/verification.json')['checkpoint_sha256']
    rows = []
    for k, baseline in before.items():
        if baseline['status'] != 'success' or after[k]['status'] == 'success':
            continue
        item = items[k[0]]
        reference_path = before_path.parent / baseline['path']
        failed_path = after_path.parent / after[k]['path']
        reference, failed = read(reference_path), read(failed_path)
        if (not reference['independent_audit']['passed'] or not failed['independent_audit']['passed'] or
                reference['checkpoint_sha256'] != ppo['parent_sha256'] or failed['checkpoint_sha256'] != endpoint):
            raise ValueError('regression lacks the fixed model or independent replay proof')
        teacher_path = (Path(ppo['course']) / item['path']).resolve()
        if sha(teacher_path) != item['record_sha256']:
            raise ValueError('original teaching record changed')
        root = first_deviation(failed, reference)
        row = dict(id=k[0], seed=k[1], split=item['split'], group=item['group'], in_rehearsal=item['id'] in selected,
            actual_failed_prefix_root=root, shared_executed_steps_with_teacher=common_executed_prefix(failed, read(teacher_path)),
            diagnosis=diagnose(failed), status=after[k]['status'],
            actual_failed_trace=str(failed_path), actual_failed_trace_sha256=sha(failed_path),
            successful_pre_ppo_trace=str(reference_path), successful_pre_ppo_trace_sha256=sha(reference_path),
            fit_recovery_candidate=item['split']=='fit' and k[1] in (None,1) and root is not None,
            wrong_action_label=False, training_ready=False)
        rows.append(row)
    result = dict(schema='post-ppo-regression-diagnosis-v1', checkpoint_sha256=endpoint,
        exam_protocol_sha256=sha(a.exam/'protocol.json'), source_sha256=sha(__file__), rows=rows,
        counts=dict(Counter('selected' if r['in_rehearsal'] else 'unselected' for r in rows)),
        fit_recovery_candidates=sum(r['fit_recovery_candidate'] for r in rows),
        next_intervention=dict(scope='New separately frozen fit-only trial; not a claim of efficacy',
            steps=['Replay actual failed prefixes and search positive goal-reaching continuations.',
                'Require repeated actor execution and independent raw-core proof for every proposed teacher.',
                'Retain all earlier lessons and compare original-course continuation against new repairs at equal update budgets.',
                'Keep reserved seeds, independent components and current frozen duel results out of teacher selection.']),
        model_inference=False, optimizer_updates=0, current_frozen_experiment_changed=False,
        interpretation='Being selected in rehearsal does not prove the autonomous failing history was covered. Different actions are candidates, not globally wrong-action labels.')
    if a.output.exists():
        raise FileExistsError('preserve the previous diagnosis')
    write(a.output, result)
    print({k:result[k] for k in ('counts','fit_recovery_candidates','model_inference','optimizer_updates')}, flush=True)


if __name__ == '__main__':
    main()
