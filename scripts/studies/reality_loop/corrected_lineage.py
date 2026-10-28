"""Verify fresh teaching provenance instead of transferring old input proofs."""
from pathlib import Path
from contracts import read, sha, round_protocol


def teaching_bindings(run, native_sha256):
    run = Path(run).resolve()
    bindings = {}
    previous = None
    for name in ('round1', 'round2'):
        context = round_protocol(run, name)
        if previous is not None and context['initial_checkpoint_sha256'] != previous:
            raise ValueError('round two did not collect from terminal updated round-one policy')
        for arm in ('control', 'repair'):
            folder = run / (name + '-training')
            path = folder / (arm + '-inputs') / 'plan.json'
            plan = read(path)
            if (plan['native_sha256'] != native_sha256 or plan['full_legal_menu'] is not False or
                    plan['parent_sha256'] != context['initial_checkpoint_sha256']):
                raise ValueError('teaching arm used a different runtime, menu or parent')
            admission_path = folder / 'admission.json'
            admission = read(admission_path)
            if (admission['input_plan_sha256'][arm] != sha(path) or
                    admission['parent_sha256'] != context['initial_checkpoint_sha256']):
                raise ValueError('teaching admission does not bind its training inputs')
            bindings.update({str(path): sha(path), str(admission_path): sha(admission_path)})
        previous = read(run / (name + '-training/comparison.json'))['endpoint_sha256']['repair']
    return bindings


def actor_path_bindings(run, course, report_path, native_sha256):
    run = Path(run).resolve()
    report = read(report_path)
    helper = Path(__file__).with_name('audit_historical_actor_paths.py')
    if (not report['passed'] or report['native_sha256'] != native_sha256 or
            report['course_plan_sha256'] != sha(Path(course) / 'plan.json') or
            report['source_sha256'] != sha(helper) or
            len(report['records']) != report['unique_records'] or
            not all(r['passed'] and r['all_policy_inputs_identical'] for r in report['records'])):
        raise ValueError('fresh teaching requires every actual actor path on the corrected inputs')
    summaries = []
    for name in ('round1', 'round2'):
        summaries.append(run / name / 'evaluation/summary.json')
        for arm in ('control', 'repair'):
            summaries.extend([run / (name + '-training') / arm / 'before/summary.json',
                run / (name + '-training') / arm / 'endpoint/evaluation/summary.json'])
    if set(report['summaries']) != {str(p) for p in summaries}:
        raise ValueError('fresh actor proof omitted a training phase')
    expected = sorted(str(p.parent / r['path']) for p in summaries for r in read(p)['results'])
    actual = sorted(p for r in report['records'] for p in r['paths'])
    if expected != actual or len(expected) != report['source_rows']:
        raise ValueError('fresh actor proof omitted a recorded rollout')
    bindings = {str(report_path): sha(report_path), str(helper): sha(helper), **report['summaries']}
    bindings.update({p: r['source_record_sha256'] for r in report['records'] for p in r['paths']})
    if not all(sha(p) == value for p, value in bindings.items()):
        raise ValueError('fresh actor proof or its source changed')
    return bindings
