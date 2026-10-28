"""Execute one fixed endpoint, allowing independent useful GPU workers.

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


def execute(a):
    folder = a.output.resolve()
    p = read(folder / 'protocol.json')
    if not all(sha(Path(f)) == value for f, value in p['protected'].items()):
        raise ValueError('frozen examination evidence or source changed')
    verify_runtime(Path(p['release']))
    for e in p['endpoints']:
        if sha(e['checkpoint']) != e['checkpoint_sha256']:
            raise ValueError('endpoint changed')
    source = str(PROJECT / 'scripts/studies/reality_loop/duel_budget_exam.py')
    env = dict(os.environ, JAX_PLATFORMS='cuda', JAX_DEFAULT_MATMUL_PRECISION=p['jax_precision_environment'],
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    if a.phase in ('independent', 'natural'):
        for endpoint in p['endpoints']:
            proof = read(folder / (endpoint['name'] + '-qualification-completion.json'))
            if proof['protocol_sha256'] != sha(folder / 'protocol.json') or sha(proof['summary']) != proof['summary_sha256']:
                raise ValueError('all fixed endpoints require preserved duel qualification before full exams')
    selected = [e for e in p['endpoints'] if e['name'] == a.endpoint]
    if len(selected) != 1:
        raise ValueError('endpoint must name exactly one frozen model')
    for e in selected:
        name = e['name']
        if a.phase == 'development' and name not in p['development_endpoints']:
            continue
        label = name + '-' + a.phase
        completion = folder / (label + '-completion.json')
        if completion.exists():
            proof = read(completion)
            if not a.resume or proof['protocol_sha256'] != sha(folder / 'protocol.json') or sha(proof['summary']) != proof['summary_sha256']:
                raise ValueError('completed endpoint phase changed or resume not requested')
            if not all(sha(Path(f)) == value for f, value in proof['record_artifacts'].items()):
                raise ValueError('completed endpoint records changed')
            continue
        if a.phase in ('development', 'independent'):
            target = folder / label
            frozen = p['independent_exam'] if a.phase == 'independent' else p['development_course']
            child([sys.executable, str(PROJECT / 'scripts/studies/reality_loop/evaluate_budget_supervision.py'),
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
            record_artifacts={str((target / r['path']).resolve()): sha(target / r['path']) for r in results},
            attempts=len(results), status_counts=dict(Counter(r['status'] for r in results)), promotion_eligible=False))
        worker_ledger = folder / 'worker-ledgers' / name
        worker_ledger.mkdir(parents=True, exist_ok=True)
        ledger(worker_ledger, 'fixed_endpoint_' + a.phase, [completion, summary], endpoint=name,
            claim='Fixed endpoint exam observation; invalid and budget exhausted attempts never become wins or losses')
    print('EXAM_PHASE_COMPLETED', a.phase, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--phase', choices=['development', 'qualification', 'independent', 'natural'], required=True)
    parser.add_argument('--resume', action='store_true')
    execute(parser.parse_args())


if __name__ == '__main__':
    main()
