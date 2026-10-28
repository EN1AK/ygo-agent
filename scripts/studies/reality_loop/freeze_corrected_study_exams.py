"""Freeze fresh corrected-study exams and retain every earlier fixed model."""
import argparse
import os
from pathlib import Path
import sys
from contracts import read, sha, write, ledger
from corrected_lineage import teaching_bindings, actor_path_bindings
from freeze_revised_exams import terminal_endpoints, retain_historical
from ppo_exams import intake_coverage

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from scripts.run_elfnote_formal import child, verify_runtime
from ygoai.rl.core_audit import validate_curriculum_binding
from ygoai.rl.checkpoint_compat import load_checkpoint_metadata, metadata_path, REQUIRED_STRUCTURED_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'ppo-protocol', 'previous-exam', 'path-proof', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--previous-ppo-protocol', type=Path, nargs='+', required=True)
    a = parser.parse_args()
    run = a.run.resolve()
    p = read(a.ppo_protocol)
    if p['id'] != 'reality-loop-corrected-input-ppo-v1':
        raise ValueError('fresh corrected-input PPO experiment required')
    release = Path(p['release'])
    verify_runtime(release)
    prep = read(release / 'preparation.json')
    rule_path = release.parent / 'real-card-qualification/report.json'
    rule = read(rule_path)
    if (prep.get('core_rule_revision') != 'explicit-nonlevel-material-v1' or not rule['passed'] or
            rule['cases'] != 48 or rule['positive_link1_cases'] != 12 or
            rule['fixed_native_sha256'] != prep['diagnostic_sha256']):
        raise ValueError('corrected real-card runtime qualification required')
    course = read(Path(p['course']) / 'plan.json')
    exam = read(Path(p['independent_exam']) / 'plan.json')
    for plan in (course, exam):
        validate_curriculum_binding(plan)
        if plan['native_sha256'] != prep['diagnostic_sha256'] or plan['full_legal_menu'] is not False:
            raise ValueError('exam inputs differ from the corrected filtered runtime')
    if len(exam['rows']) != 6 or any(r['split'] != 'holdout' for r in exam['rows']):
        raise ValueError('retain the original complete admitted independent set')
    bindings = teaching_bindings(run, prep['diagnostic_sha256'])
    bindings.update(actor_path_bindings(run, p['course'], a.path_proof, prep['diagnostic_sha256']))
    endpoints = [*p['pre_ppo_endpoints'], *terminal_endpoints(a.ppo_protocol)]
    aliases = []
    previous_protocols = []
    historical_registry = []
    for index, path in enumerate(a.previous_ppo_protocol, 1):
        old = read(path)
        fixed = [*old['pre_ppo_endpoints'], *terminal_endpoints(path)]
        historical_registry.extend(fixed)
        tagged = [dict(e, name='ppo' + str(index) + '-' + e['name']) for e in fixed]
        endpoints, extra_aliases = retain_historical(endpoints, tagged)
        aliases.extend(extra_aliases)
        previous_protocols.append(dict(path=str(path.resolve()), sha256=sha(path), endpoints=fixed))
        for filename, digest in old['protected'].items():
            bindings[filename] = digest
        for role in old['roles']:
            for filename in ('verification.json', 'rehearsal-verification.json'):
                evidence = Path(old['study_output']) / role / filename
                bindings[str(evidence)] = sha(evidence)
    previous = read(a.previous_exam / 'protocol.json')
    required = {e['checkpoint_sha256'] for e in [*previous['endpoints'], *historical_registry]}
    if required - {e['checkpoint_sha256'] for e in endpoints}:
        raise ValueError('an earlier fixed model was omitted')
    if len(endpoints) < 7 or len({e['name'] for e in endpoints}) != len(endpoints):
        raise ValueError('invalid endpoint registry')
    metadata = {}
    for e in [*endpoints, *aliases]:
        if sha(e['checkpoint']) != e['checkpoint_sha256']:
            raise ValueError('fixed model weights changed')
        metadata[e['name']] = load_checkpoint_metadata(e['checkpoint'])
    reference = metadata['parent']
    if any(any(value[k] != reference[k] for k in REQUIRED_STRUCTURED_FIELDS) for value in metadata.values()):
        raise ValueError('fixed model inference metadata is incompatible')
    if reference['code_list_hash'] != course['code_list_sha256'] or reference['semantic_table_hash'] != course['semantics_sha256']:
        raise ValueError('corrected course vocabulary differs from fixed models')
    coverage = intake_coverage(run, course, exam)
    protected_paths = [a.ppo_protocol, a.previous_exam / 'protocol.json',
        a.previous_exam / 'parent-duel-plan/plan.json', Path(p['course']) / 'plan.json',
        Path(p['independent_exam']) / 'plan.json', release / 'frozen-manifest.json', Path(__file__),
        rule_path, *a.previous_ppo_protocol,
        *[metadata_path(e['checkpoint']) for e in [*endpoints, *aliases]],
        PROJECT / 'scripts/studies/reality_loop/execute_corrected_exams.py',
        PROJECT / 'scripts/studies/reality_loop/dispatch_corrected_exams.py',
        PROJECT / 'scripts/studies/reality_loop/audit_ppo_study.py',
        PROJECT / 'scripts/studies/reality_loop/ppo_exams.py',
        PROJECT / 'scripts/studies/reality_loop/freeze_revised_exams.py',
        PROJECT / 'scripts/studies/reality_loop/corrected_lineage.py',
        PROJECT / 'scripts/studies/reality_loop/duel_exam.py',
        PROJECT / 'scripts/studies/crystron_curriculum/evaluate_supervision.py',
        PROJECT / 'scripts/studies/crystron_curriculum/supervision_runtime.py']
    bindings.update({str(path.resolve()): sha(path) for path in protected_paths})
    bindings.update(p['protected'])
    bindings.update(p['source_sha256'])
    bindings.update(coverage['protected'])
    for role in p['roles']:
        for filename in ('verification.json', 'rehearsal-verification.json'):
            evidence = Path(p['study_output']) / role / filename
            bindings[str(evidence)] = sha(evidence)
    if not all(sha(path) == value for path, value in bindings.items()):
        raise ValueError('fixed evidence changed before exam registration')
    protocol = dict(format_version=1, ppo_protocol=str(a.ppo_protocol.resolve()),
        ppo_protocol_sha256=sha(a.ppo_protocol), previous_ppo_protocols=previous_protocols,
        previous_exam_protocol=str((a.previous_exam / 'protocol.json').resolve()),
        previous_exam_protocol_sha256=sha(a.previous_exam / 'protocol.json'),
        historical_actor_bridge=str(a.path_proof.resolve()), historical_actor_bridge_sha256=sha(a.path_proof),
        endpoints=endpoints, checkpoint_aliases=aliases, all_original_fixed_models_retained=True,
        independent_exam=p['independent_exam'], independent_all_rows=6, independent_intake_coverage=coverage,
        development_course=p['course'], release=p['release'], protected=bindings,
        development_endpoints=[e['name'] for e in endpoints],
        natural_duels=dict(planned_per_endpoint=256, decks=4, seeds=32, seats=[0, 1],
            opponent=p['evaluation']['opponent_checkpoint'], qualification_prefix=4),
        phases=['development', 'qualification', 'independent', 'natural'], precision=p['precision'],
        jax_precision_environment=p['jax_precision_environment'], invalid_and_budget_are_unknown=True,
        independent_score_checkpoint_selection=False, promotion_eligible=False,
        fresh_corrected_sft=True,
        exposure_scope='Two newly executed corrected-input teaching rounds. All prior fixed weights are historical controls on this same corrected exam. Original independent components were partly observed during a failed old-runtime evaluation; no independent labels enter training, checkpoint selection or recipe tuning. This is not a newly untouched holdout.')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output / 'protocol.json', protocol)
    original = read(a.previous_exam / 'parent-duel-plan/plan.json')
    matrix = None
    env = dict(os.environ, JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for e in endpoints:
        target = a.output / (e['name'] + '-duel-plan')
        child([sys.executable, str(PROJECT / 'scripts/studies/reality_loop/duel_exam.py'), 'freeze',
            '--frozen', p['course'], '--candidate', e['checkpoint'], '--opponent', protocol['natural_duels']['opponent'],
            '--output', str(target), '--decks', '4', '--seeds', '32'], a.output, e['name'] + '-freeze', env)
        duel = read(target / 'plan.json')
        if len(duel['cases']) != 256 or len({c['id'] for c in duel['cases']}) != 256:
            raise ValueError('complete duel matrix missing')
        if matrix is None:
            matrix = duel['cases']
        elif duel['cases'] != matrix:
            raise ValueError('endpoints have unpaired cases')
        if any(duel[k] != original[k] for k in ('cases', 'budget', 'precision', 'opponent_sha256')):
            raise ValueError('original fixed duel matrix or opponent changed')
    ledger(run, 'fresh_corrected_complete_exam_protocol', [a.output / 'protocol.json'],
        unique_endpoints=len(endpoints), explicit_aliases=len(aliases),
        claim='Fresh corrected-input teaching plus every historical fixed weight; full paired matrix and original independent denominator; no model selection')
    print('FRESH_CORRECTED_EXAMS_FROZEN', len(endpoints), len(aliases), flush=True)


if __name__ == '__main__':
    main()
