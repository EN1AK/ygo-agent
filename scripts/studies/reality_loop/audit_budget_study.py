"""Verify inherited attempts and prefix certificates before the full study audit."""
import argparse
from pathlib import Path
from contracts import read, sha, write, ledger
from engine_budget_audit import eligible, trajectory_digest, PROCESS_LIMIT
from freeze_engine_budget_revision import preserved_game_data
import audit_ppo_study as original_audit


def verify_prefix(record, native_sha256, independent_sha256):
    proof = record['independent_audit']
    if (not eligible(record) or record['status'] != 'budget_exhausted' or
            record.get('native_sha256') != native_sha256 or
            record.get('core_process_cutoff', {}).get('limit') != PROCESS_LIMIT or
            not proof['passed'] or proof.get('kind') != 'engine-process-budget-prefix-v1' or
            proof.get('trajectory_sha256') != trajectory_digest(record) or
            proof.get('core_process_calls') != PROCESS_LIMIT or
            proof.get('responses') != len(record['core_responses']) or
            proof.get('recorded_events') != len(record['events']) or
            proof.get('exact_recorded_prefix') is not True or
            proof.get('natural_outcome_certified') is not False or
            proof.get('independent_library_sha256') != independent_sha256):
        raise ValueError('resource cutoff lacks a bound exact-prefix certificate')


def verify_import(exam, protocol):
    path = Path(protocol['completed_attempt_import'])
    if sha(path) != protocol['completed_attempt_import_sha256']:
        raise ValueError('completed-attempt import changed')
    manifest = read(path)
    if manifest['model_inference'] is not False or len(manifest['development']) != 15 or len(manifest['games']) != 60:
        raise ValueError('completed-attempt denominator changed')
    for item in manifest['development']:
        if sha(item['source']) != item['source_sha256']:
            raise ValueError('source development proof changed')
        old, current = read(item['source']), read(exam / (item['endpoint'] + '-development-completion.json'))
        if any(current[k] != old[k] for k in ('summary', 'summary_sha256', 'record_artifacts', 'attempts', 'status_counts', 'checkpoint_sha256')):
            raise ValueError('development results were changed during import')
    course = read(Path(protocol['development_course']) / 'plan.json')
    for item in manifest['games']:
        if (sha(item['source']) != item['source_sha256'] or sha(item['target']) != item['target_sha256'] or
                sha(item['source_marker']) != item['source_marker_sha256'] or
                sha(item['source_plan']) != sha(item['target_plan'])):
            raise ValueError('an inherited game or exact plan changed')
        old, current = read(item['source']), read(item['target'])
        if not preserved_game_data(old, current):
            raise ValueError('policy decisions or engine data changed during import')
        if item['adjudicated_process_cutoff']:
            if old['status'] != 'invalid' or current['status'] != 'budget_exhausted':
                raise ValueError('process-cutoff adjudication changed a natural outcome')
            verify_prefix(current, course['native_sha256'], course['independent_library']['sha256'])
        elif item['source_sha256'] != item['target_sha256']:
            raise ValueError('an already valid game was rewritten')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'exam', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    p = read(a.exam / 'protocol.json')
    if not all(sha(path) == value for path, value in p['protected'].items()):
        raise ValueError('frozen revision evidence changed')
    manifest = verify_import(a.exam, p)
    course = read(Path(p['development_course']) / 'plan.json')
    certified = []
    for endpoint in p['endpoints']:
        name = endpoint['name']
        plan_path = a.exam / (name + '-duel-plan/plan.json')
        execution = read(a.exam / (name + '-natural/execution.json'))
        if execution != dict(exam_sha256=sha(plan_path), source_sha256=sha(p['natural_evaluator'])):
            raise ValueError('natural-duel evaluator source or plan changed')
        for phase in ('independent', 'natural'):
            completion = read(a.exam / (name + '-' + phase + '-completion.json'))
            for path in completion['record_artifacts']:
                record = read(path)
                if record.get('core_process_cutoff'):
                    verify_prefix(record, course['native_sha256'], course['independent_library']['sha256'])
                    certified.append(dict(endpoint=name, phase=phase, id=record['id'], record_sha256=sha(path)))
    # The original full audit still checks all 15 identities, 6 independent rows,
    # every 256-case matrix, raw hashes, both training arms and both SFT rounds.
    original_audit.main()
    proof = dict(passed=True, source_sha256=sha(__file__), inherited_development_sets=15,
        inherited_qualification_attempts=60, inherited_budget_adjudications=manifest['adjudicated_process_cutoffs'],
        certified_unknown_prefixes=certified, process_limit=PROCESS_LIMIT,
        no_inherited_policy_reruns=True, no_natural_outcome_inferred=True,
        scope='Prefix certificates prove all recorded responses/events/field state at the unchanged engine fuse; every such game remains unknown, never a win/loss.')
    target = a.output / 'engine-budget-prefix-audit.json'
    write(target, proof)
    ledger(a.run, 'engine_budget_prefix_completion_audit', [target, a.output / 'report.json'],
        claim='All inherited attempts preserved; exact certified resource prefixes stay unknown; full study audit independently completed')
    print('ENGINE_BUDGET_STUDY_AUDITED', len(certified), flush=True)


if __name__ == '__main__':
    main()
