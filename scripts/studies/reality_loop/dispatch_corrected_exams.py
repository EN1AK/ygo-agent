"""Run independent endpoint jobs concurrently without changing their case matrix."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
from contracts import read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--phase', choices=['development', 'qualification', 'independent', 'natural'], required=True)
    a = parser.parse_args()
    if not 1 <= a.workers <= 8:
        parser.error('one to eight actual endpoint workers required')
    protocol = read(a.output / 'protocol.json')
    if not all(sha(path) == value for path, value in protocol['protected'].items()):
        raise ValueError('frozen exam evidence changed before dispatch')
    names = [e['name'] for e in protocol['endpoints']]
    if not names or len(names) != len(set(names)):
        raise ValueError('complete unique endpoint inventory required')
    if a.phase == 'development' and set(names) != set(protocol['development_endpoints']):
        raise ValueError('fresh study development must include all fixed models')
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

    def execute(name):
        label = 'dispatch-' + a.phase + '-' + name
        argv = [sys.executable, str(Path(__file__).with_name('execute_corrected_exams.py')),
            '--run', str(a.run.resolve()), '--output', str(a.output.resolve()), '--endpoint', name, '--phase', a.phase]
        with (a.output / (label + '.log')).open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT)
            write(a.output / (label + '-process.json'), dict(pid=process.pid, command=argv))
            code = process.wait()
        write(a.output / (label + '-exit.json'), dict(code=code, endpoint=name, phase=a.phase))
        return name, code

    failures = []
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        pending = [executor.submit(execute, name) for name in names]
        for future in as_completed(pending):
            name, code = future.result()
            print(a.phase, name, 'exit', code, flush=True)
            if code:
                failures.append(name)
    if failures:
        raise RuntimeError('preserved failed endpoint jobs: ' + ', '.join(failures))
    for name in names:
        proof = read(a.output / (name + '-' + a.phase + '-completion.json'))
        if (not proof['completed'] or proof['protocol_sha256'] != sha(a.output / 'protocol.json') or
                sha(proof['summary']) != proof['summary_sha256']):
            raise ValueError('a dispatched endpoint did not complete its frozen phase')
    print('ALL_FIXED_ENDPOINTS_COMPLETED', a.phase, len(names), flush=True)


if __name__ == '__main__':
    main()
