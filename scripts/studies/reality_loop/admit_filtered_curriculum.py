"""Freeze a projected course only after per-row engine equivalence admission."""
import argparse
import copy
from pathlib import Path
from contracts import read,sha,validate_splits,write


def admit(projected,source,output,exam=False):
    plan=read(projected/'projected-plan.json')
    original=read(source/'plan.json')
    coverage=read(projected/'projection-coverage.json')
    if plan.get('full_legal_menu') is not False or not plan.get('training_admission_pending'):
        raise ValueError('explicit pending filtered projection required')
    if plan['projection_source_plan_sha256']!=sha(source/'plan.json'):
        raise ValueError('original curriculum changed')
    if coverage['input_rows']!=len(original['rows']) or len(coverage['rows'])!=len(original['rows']):
        raise ValueError('missing original coverage denominator')
    statuses={r['id']:r for r in coverage['rows']}
    originals={r['id']:r for r in original['rows']}
    if set(statuses)!=set(originals) or len(statuses)!=len(coverage['rows']):
        raise ValueError('coverage source identities changed')
    admitted={r['id'] for r in coverage['rows'] if r['status']=='admitted'}
    if admitted!={r['id'] for r in plan['rows']} or coverage['admitted']!=len(plan['rows']):
        raise ValueError('projection admission mismatch')
    if not admitted:raise ValueError('no projected trajectories admitted')
    mapping_failures={'nonforced_decision_disappeared','full_source_input_changed',
        'non-permutation source input changed','full_source_endpoint_changed',
        'non-chain source response changed','full source committed events changed'}
    if any(r.get('reason') in mapping_failures for r in coverage['rows']):
        raise ValueError('unresolved projection mapping mismatch; do not shrink the course to passing rows')
    plan=copy.deepcopy(plan)
    for row in plan['rows']:
        path=(projected/row['path']).resolve()
        if sha(path)!=row['record_sha256'] or sha(path.with_suffix('.npz'))!=row['observations_sha256']:
            raise ValueError('projected inputs changed')
        record=read(path);proof=record['navigation_projection'];old=originals[row['id']]
        if not (record['full_legal_menu'] is False and proof['independent_audit']['passed'] and
                proof['same_endpoint'] and (proof.get('same_committed_game_events') or proof.get('same_game_events'))):
            raise ValueError('missing real-engine equivalence proof')
        if proof['source_record_sha256']!=old['record_sha256']:
            raise ValueError('teacher source changed')
        for key in ('family','source_sha256','split','group','goal'):
            if row[key]!=old[key]:raise ValueError('projection changed curriculum semantics: '+key)
        if not 0<=row['root_step']<len(record['decisions']):
            raise ValueError('projected root is outside the actual trajectory')
        row['path']=str(path)
    validate_splits(plan['rows'])
    if exam and any(r['split']!='holdout' for r in plan['rows']):
        raise ValueError('independent exam cannot contain fit rows')
    for stage in ([] if exam else plan['stages']):
        for group,mass in stage['mix'].items():
            if mass>0 and not any(r['group']==group and r['split']=='fit' for r in plan['rows']):
                raise ValueError('projection removed a declared curriculum group: '+group)
    native=Path(plan['runtime'])/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    if sha(native)!=plan['native_sha256']:raise ValueError('native changed')
    if plan['parent_sha256']!=original['parent_sha256'] or sha(Path(plan['parent']))!=plan['parent_sha256']:
        raise ValueError('projection cannot select a different parent')
    output.mkdir(parents=True,exist_ok=False)
    plan.update(id=original['id']+'-contextual-navigation-chain-v3',training_admission_pending=False,
        admission=dict(coverage_path=str(projected/'projection-coverage.json'),coverage_sha256=sha(projected/'projection-coverage.json'),
                       projected_plan_sha256=sha(projected/'projected-plan.json'),source_plan_sha256=sha(source/'plan.json'),
                       source_rows=len(original['rows']),admitted_rows=len(plan['rows']),quarantined_rows=len(original['rows'])-len(plan['rows']),
                       model_inference=False,no_independent_score_selection=True,exam_only=exam))
    write(output/'plan.json',plan)
    write(output/'admission.json',dict(passed=True,plan_sha256=sha(output/'plan.json'),**plan['admission']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('projected','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--exam-only',action='store_true')
    a=p.parse_args();admit(a.projected.resolve(),a.source.resolve(),a.output.resolve(),a.exam_only)
