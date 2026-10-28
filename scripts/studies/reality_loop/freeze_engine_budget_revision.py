"""Carry every completed attempt into an explicit process-budget revision.

All model hashes, input data, cases, opponents and declared duel budgets stay
fixed. Only a captured native-fuse error may gain a strict independently
replayed unknown-prefix certificate. No model inference occurs during import.
"""
import argparse
from collections import Counter
import copy
from pathlib import Path
import shutil
from contracts import read, sha, write, ledger
from duel_exam import completed_case
from engine_budget_audit import adjudicate, eligible
from audit_ppo_study import verify_original_duel_matrix

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]


def preserved_game_data(original, revised):
    keys = ('id', 'definition', 'seed', 'candidate_seat', 'candidate_sha256', 'opponent_sha256',
        'decisions', 'events', 'final', 'core_responses', 'core_seed', 'seconds', 'error', 'winner', 'candidate_result')
    return all(original.get(k) == revised.get(k) for k in keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'previous-exam', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    a.run, a.previous_exam, a.output = a.run.resolve(), a.previous_exam.resolve(), a.output.resolve()
    old_path = a.previous_exam / 'protocol.json'
    old = read(old_path)
    if not old.get('fresh_corrected_sft') or len(old['endpoints']) != 15:
        raise ValueError('the complete fresh corrected-study registry is required')
    if not all(sha(path) == value for path, value in old['protected'].items()):
        raise ValueError('the previous frozen examination changed')
    course = read(Path(old['development_course']) / 'plan.json')
    original_matrix = read(Path(old['previous_exam_protocol']).parent / 'parent-duel-plan/plan.json')
    runtime = Path(course['runtime'])
    a.output.mkdir(parents=True, exist_ok=False)
    bindings = dict(old['protected'])
    bindings[str(old_path)] = sha(old_path)
    imported = dict(schema='engine-budget-exam-import-v1', source_exam_protocol=str(old_path),
        source_exam_protocol_sha256=sha(old_path), model_inference=False, development=[], games=[])
    development = []
    qualifications = []
    new_driver_sha = sha(HERE / 'duel_budget_exam.py')
    old_driver_sha = sha(HERE / 'duel_exam.py')
    for endpoint in old['endpoints']:
        name = endpoint['name']
        source = a.previous_exam / (name + '-development-completion.json')
        proof = read(source)
        if (not proof['completed'] or proof['protocol_sha256'] != sha(old_path) or
                sha(proof['summary']) != proof['summary_sha256'] or
                not all(sha(path) == value for path, value in proof['record_artifacts'].items())):
            raise ValueError('an existing development attempt changed')
        bindings.update({str(source): sha(source), proof['summary']: proof['summary_sha256'], **proof['record_artifacts']})
        imported['development'].append(dict(endpoint=name, source=str(source), source_sha256=sha(source)))
        development.append((name, proof, source))

        source_plan = a.previous_exam / (name + '-duel-plan/plan.json')
        target_plan = a.output / (name + '-duel-plan/plan.json')
        target_plan.parent.mkdir()
        shutil.copy2(source_plan, target_plan)
        plan = read(source_plan)
        verify_original_duel_matrix(plan, original_matrix)
        if (plan['native_sha256'] != course['native_sha256'] or
                plan['candidate_sha256'] != endpoint['checkpoint_sha256'] or
                sha(Path(plan['frozen']) / 'plan.json') != plan['frozen_plan_sha256']):
            raise ValueError('source case runtime or fixed model binding changed')
        if sha(source_plan) != sha(target_plan) or len(plan['cases']) != 256:
            raise ValueError('the exact original case plan must survive import')
        old_target = a.previous_exam / (name + '-natural')
        old_execution = old_target / 'execution.json'
        if read(old_execution) != dict(exam_sha256=sha(source_plan), source_sha256=old_driver_sha):
            raise ValueError('source games were produced by a different frozen evaluator')
        bindings.update({str(source_plan): sha(source_plan), str(target_plan): sha(target_plan), str(old_execution): sha(old_execution)})
        target = a.output / (name + '-natural')
        target.mkdir()
        execution = target / 'execution.json'
        write(execution, dict(exam_sha256=sha(target_plan), source_sha256=new_driver_sha))
        bindings[str(execution)] = sha(execution)
        rows = []
        for case in plan['cases'][:old['natural_duels']['qualification_prefix']]:
            row = completed_case(old_target, case, plan, sha(source_plan))
            if row is None:
                raise ValueError('an original qualification attempt was omitted')
            source_record = old_target / row['path']
            source_marker = old_target / (case['id'] + '.completed.json')
            record = read(source_record)
            revised = record
            changed = not row['audit']['passed'] or row['status'] not in ('natural_terminal', 'budget_exhausted')
            if changed:
                if not eligible(record):
                    raise ValueError('unrelated invalid attempt cannot be adjudicated as a process budget')
                revised = adjudicate(record, course['source_root'], runtime / 'assets/cards.cdb',
                    runtime / 'assets/script', course['independent_library'], course['native_sha256'])
                if not preserved_game_data(record, revised):
                    raise ValueError('adjudication changed an actual game or policy decision')
            target_record = target / row['path']
            if changed:
                write(target_record, revised)
            else:
                shutil.copy2(source_record, target_record)
                if sha(source_record) != sha(target_record):
                    raise ValueError('an already valid attempt changed')
            new_row = dict(row, status=revised['status'], audit=revised['independent_audit'], record_sha256=sha(target_record))
            marker = target / (case['id'] + '.completed.json')
            write(marker, dict(exam_sha256=sha(target_plan), result=new_row))
            if completed_case(target, case, plan, sha(target_plan)) != new_row:
                raise ValueError('imported completed attempt does not bind to the original case')
            bindings.update({str(source_record): sha(source_record), str(source_marker): sha(source_marker),
                str(target_record): sha(target_record), str(marker): sha(marker)})
            imported['games'].append(dict(endpoint=name, id=case['id'], source=str(source_record),
                source_sha256=sha(source_record), source_marker=str(source_marker), source_marker_sha256=sha(source_marker),
                source_plan=str(source_plan), target_plan=str(target_plan), target=str(target_record),
                target_sha256=sha(target_record), adjudicated_process_cutoff=changed,
                original_status=record['status'], revised_status=revised['status']))
            rows.append(new_row)
        summary = dict(completed_attempts=len(rows), planned=256, pending=256-len(rows), qualification_only=True,
            results=rows, status_counts=dict(Counter(r['status'] for r in rows)),
            natural_results=dict(Counter(r['candidate_result'] for r in rows if r['status']=='natural_terminal')),
            invalids_and_timeouts_are_not_losses=True, promotion_eligible=False)
        write(target / 'summary.json', summary)
        write(target / 'progress.json', dict(results=rows, planned=256, requested=len(rows)))
        summary_path = a.output / (name + '-qualification-summary.json')
        write(summary_path, summary)
        bindings[str(summary_path)] = sha(summary_path)
        qualifications.append((name, target, rows, summary_path))

    if len(imported['development']) != 15 or len(imported['games']) != 60:
        raise ValueError('the full completed-attempt denominator must survive revision')
    imported['adjudicated_process_cutoffs'] = sum(r['adjudicated_process_cutoff'] for r in imported['games'])
    manifest = a.output / 'completed-attempt-import.json'
    write(manifest, imported)
    bindings[str(manifest)] = sha(manifest)
    sources = ['freeze_engine_budget_revision.py', 'engine_budget_audit.py', 'audit_engine_budget_prefix.py',
        'budget_runtime.py', 'evaluate_budget_supervision.py', 'duel_budget_exam.py', 'execute_budget_exams.py',
        'dispatch_budget_exams.py', 'audit_budget_study.py', 'audit_core_replay.py']
    for name in sources:
        path = HERE / name
        bindings[str(path)] = sha(path)
    for path in (PROJECT / 'ygoai/rl/core_audit.py', PROJECT / 'scripts/exercises/actor_bridge.cpp',
            Path(course['source_root']) / 'modern_core.py'):
        bindings[str(path)] = sha(path)
    if not all(sha(path) == value for path, value in bindings.items()):
        raise ValueError('source or imported evidence changed during revision')
    protocol = copy.deepcopy(old)
    protocol.update(protected=bindings, revision_source_exam_protocol=str(old_path),
        revision_source_exam_protocol_sha256=sha(old_path), completed_attempt_import=str(manifest),
        completed_attempt_import_sha256=sha(manifest), engine_process_budget=4096,
        engine_budget_classification='strict-exact-prefix-unknown-v1',
        natural_evaluator=str(HERE / 'duel_budget_exam.py'),
        independent_evaluator=str(HERE / 'evaluate_budget_supervision.py'),
        model_inference_during_revision=False,
        exposure_scope=old['exposure_scope'] + ' Completed development and all qualification attempts are inherited without policy inference. A reproduced 4096-call engine fuse is a certified unknown prefix, never a natural win or loss.')
    write(a.output / 'protocol.json', protocol)
    protocol_sha = sha(a.output / 'protocol.json')
    for name, proof, source in development:
        write(a.output / (name + '-development-completion.json'), dict(proof, protocol_sha256=protocol_sha,
            imported_from=str(source), imported_source_sha256=sha(source)))
    for name, target, rows, summary in qualifications:
        if not all(r['audit']['passed'] and r['status'] in ('natural_terminal','budget_exhausted') for r in rows):
            raise ValueError('an imported qualification attempt is unverified')
        write(a.output / (name + '-qualification-completion.json'), dict(completed=True, phase='qualification',
            endpoint=name, checkpoint_sha256=next(e['checkpoint_sha256'] for e in old['endpoints'] if e['name']==name),
            protocol_sha256=protocol_sha, summary=str(summary), summary_sha256=sha(summary), attempts=len(rows),
            record_artifacts={str(target/r['path']):sha(target/r['path']) for r in rows},
            status_counts=dict(Counter(r['status'] for r in rows)), inherited_without_model_inference=True,
            promotion_eligible=False))
    ledger(a.run, 'engine_process_budget_exam_revision', [a.output/'protocol.json', manifest],
        claim='All 15 model weights and 256-case plans unchanged; 15 development sets and 60 qualification attempts carried without rerunning policy; certified process fuse remains unknown')
    print('ENGINE_BUDGET_REVISION_FROZEN', len(imported['development']), len(imported['games']),
        imported['adjudicated_process_cutoffs'], flush=True)


if __name__ == '__main__':
    main()
