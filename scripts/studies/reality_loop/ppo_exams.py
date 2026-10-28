"""Audit every fixed endpoint after the PPO recipe and terminal models are fixed.

All seven endpoints receive the same independent set and 256-case natural-duel
matrix. Qualification prefixes remain part of the full denominator. An explicit
resume validates completed games without retrying invalid or timed-out games.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys

from contracts import ledger, read, sha, write

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from scripts.run_elfnote_formal import child, verify_runtime
from ygoai.rl.core_audit import validate_curriculum_binding


def freeze(a):
    run = a.run.resolve()
    p = read(a.ppo_protocol)
    study = Path(p['study_output'])
    release = Path(p['release'])
    verify_runtime(release)
    if not all(sha(Path(f)) == value for f, value in p['protected'].items()):
        raise ValueError('PPO evidence changed')
    endpoints = list(p['pre_ppo_endpoints'])
    for role in p['roles']:
        folder = study / role
        proof = read(folder / 'verification.json')
        rehearsal = read(folder / 'rehearsal-verification.json')
        if (not proof['passed'] or proof['updates'] != p['formal_updates'] or
                not rehearsal['passed'] or rehearsal['ppo_protocol_sha256'] != sha(a.ppo_protocol) or
                proof['checkpoint_sha256'] != rehearsal['checkpoint_sha256'] or
                sha(proof['checkpoint']) != proof['checkpoint_sha256']):
            raise ValueError('both fixed terminal PPO producers must complete first')
        endpoints.append(dict(name=role + '-final', checkpoint=proof['checkpoint'], checkpoint_sha256=proof['checkpoint_sha256']))
    if len(endpoints) != 7 or len({e['name'] for e in endpoints}) != 7:
        raise ValueError('all five pre-PPO and both final endpoints required')
    for e in endpoints:
        if sha(e['checkpoint']) != e['checkpoint_sha256']:
            raise ValueError('fixed endpoint changed')
    exam = Path(p['independent_exam'])
    plan = read(exam / 'plan.json')
    validate_curriculum_binding(plan)
    if len(plan['rows']) != 6 or any(r['split'] != 'holdout' for r in plan['rows']):
        raise ValueError('the entire six-row independent exam is required')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    bindings = {str(f.resolve()): sha(f) for f in [a.ppo_protocol, Path(p['course']) / 'plan.json', exam / 'plan.json',
        release / 'frozen-manifest.json', PROJECT / 'scripts/studies/reality_loop/duel_exam.py', Path(__file__),
        PROJECT / 'scripts/studies/crystron_curriculum/evaluate_supervision.py',
        PROJECT / 'scripts/studies/crystron_curriculum/supervision_runtime.py',
        *[study / r / 'verification.json' for r in p['roles']], *[study / r / 'rehearsal-verification.json' for r in p['roles']]]}
    protocol = dict(format_version=1, ppo_protocol=str(a.ppo_protocol.resolve()), ppo_protocol_sha256=sha(a.ppo_protocol),
        endpoints=endpoints, independent_exam=str(exam), independent_all_rows=6,
        development_course=p['course'], release=p['release'], protected=bindings,
        natural_duels=dict(planned_per_endpoint=256, decks=4, seeds=32, seats=[0, 1],
            opponent=p['evaluation']['opponent_checkpoint'], qualification_prefix=4),
        phases=['development', 'qualification', 'independent', 'natural'], precision=p['precision'],
        jax_precision_environment=p['jax_precision_environment'], invalid_and_budget_are_unknown=True,
        independent_score_checkpoint_selection=False, promotion_eligible=False)
    write(a.output / 'protocol.json', protocol)
    env = dict(os.environ, JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for e in endpoints:
        child([sys.executable, str(PROJECT / 'scripts/studies/reality_loop/duel_exam.py'), 'freeze',
            '--frozen', p['course'], '--candidate', e['checkpoint'], '--opponent', protocol['natural_duels']['opponent'],
            '--output', str(a.output / (e['name'] + '-duel-plan')), '--decks', '4', '--seeds', '32'],
            a.output, e['name'] + '-freeze', env)
        duel = read(a.output / (e['name'] + '-duel-plan') / 'plan.json')
        if len(duel['cases']) != 256:
            raise ValueError('full natural duel denominator changed')
    ledger(run, 'complete_endpoint_exam_protocol', [a.output / 'protocol.json'],
        claim='Every fixed pre-PPO endpoint and both fixed final PPO endpoints, six independent rows and 256 two-seat natural duels each; no model selection')
    print('ALL_ENDPOINT_EXAMS_FROZEN', len(endpoints), flush=True)


def execute(a):
    folder = a.output.resolve()
    p = read(folder / 'protocol.json')
    if not all(sha(Path(f)) == value for f, value in p['protected'].items()):
        raise ValueError('frozen examination evidence or source changed')
    verify_runtime(Path(p['release']))
    for e in p['endpoints']:
        if sha(e['checkpoint']) != e['checkpoint_sha256']:
            raise ValueError('endpoint changed')
    source = str(PROJECT / 'scripts/studies/reality_loop/duel_exam.py')
    env = dict(os.environ, JAX_PLATFORMS='cuda', JAX_DEFAULT_MATMUL_PRECISION=p['jax_precision_environment'],
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    if a.phase in ('independent', 'natural'):
        for endpoint in p['endpoints']:
            proof = read(folder / (endpoint['name'] + '-qualification-completion.json'))
            if proof['protocol_sha256'] != sha(folder / 'protocol.json') or sha(proof['summary']) != proof['summary_sha256']:
                raise ValueError('all fixed endpoints require preserved duel qualification before full exams')
    for e in p['endpoints']:
        name = e['name']
        if a.phase == 'development' and name not in ('control-final', 'rehearse-final'):
            continue
        label = name + '-' + a.phase
        completion = folder / (label + '-completion.json')
        if completion.exists():
            proof = read(completion)
            if not a.resume or proof['protocol_sha256'] != sha(folder / 'protocol.json') or sha(proof['summary']) != proof['summary_sha256']:
                raise ValueError('completed endpoint phase changed or resume not requested')
            continue
        if a.phase in ('development', 'independent'):
            target = folder / label
            frozen = p['independent_exam'] if a.phase == 'independent' else p['development_course']
            child([sys.executable, str(PROJECT / 'scripts/studies/crystron_curriculum/evaluate_supervision.py'),
                '--frozen', frozen, '--checkpoint', e['checkpoint'], '--matmul-precision', p['precision'],
                '--output', str(target)], folder, label, env, 3600)
            summary = target / 'summary.json'
            results = read(summary)['results']
            if not all(r['audit']['passed'] and r['status'] != 'invalid' for r in results):
                raise ValueError('model exam has unaudited or invalid attempts')
            if a.phase == 'independent' and (len(results) != p['independent_all_rows'] or len({r['id'] for r in results}) != p['independent_all_rows']):
                raise ValueError('independent exam omitted a row')
        else:
            target = folder / (name + '-natural')
            argv = [sys.executable, source, 'evaluate', '--exam', str(folder / (name + '-duel-plan')), '--output', str(target)]
            if target.exists():
                if not a.resume and a.phase != 'natural':
                    raise ValueError('qualification already started; explicit resume required')
                argv.append('--resume')
            if a.phase == 'qualification':
                argv += ['--max-cases', str(p['natural_duels']['qualification_prefix'])]
            child(argv, folder, label + ('-resume' if a.resume else ''), env, 36000)
            summary = target / 'summary.json'
            result = read(summary)
            expected = p['natural_duels']['qualification_prefix'] if a.phase == 'qualification' else p['natural_duels']['planned_per_endpoint']
            if result['completed_attempts'] != expected or result['planned'] != 256:
                raise ValueError('natural duel denominator mismatch')
            if a.phase == 'qualification' and not all(r['audit']['passed'] and r['status'] in ('natural_terminal', 'budget_exhausted') for r in result['results']):
                raise ValueError('new-runtime duel qualification failed')
            results = result['results']
            if a.phase == 'qualification':
                summary = folder / (name + '-qualification-summary.json')
                write(summary, result)
        write(completion, dict(completed=True, phase=a.phase, endpoint=name, checkpoint_sha256=e['checkpoint_sha256'],
            protocol_sha256=sha(folder / 'protocol.json'), summary=str(summary), summary_sha256=sha(summary),
            attempts=len(results), status_counts=dict(Counter(r['status'] for r in results)), promotion_eligible=False))
        ledger(a.run.resolve(), 'fixed_endpoint_' + a.phase, [completion, summary], endpoint=name,
            claim='Fixed endpoint exam observation; invalid and budget exhausted attempts never become wins or losses')
    print('EXAM_PHASE_COMPLETED', a.phase, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze', 'execute'])
    for name in ('run', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--ppo-protocol', type=Path)
    p.add_argument('--phase', choices=['development', 'qualification', 'independent', 'natural'])
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.mode == 'freeze' and a.ppo_protocol is None or a.mode == 'execute' and a.phase is None:
        p.error('freeze requires --ppo-protocol; execute requires --phase')
    {'freeze': freeze, 'execute': execute}[a.mode](a)


if __name__ == '__main__':
    main()
