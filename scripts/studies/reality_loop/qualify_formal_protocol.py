"""Bind actual native boundary tests and protocol coverage to a frozen runtime."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from contracts import read,sha,write


def verify_runtime(release):
    manifest=read(release/'frozen-manifest.json')
    if not all(sha(release/item['path'])==item['sha256'] for item in manifest['artifacts']):
        raise ValueError('frozen runtime changed')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('release','core-source','runtime-smoke','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-core-revision',required=True)
    a=p.parse_args();release=a.release.resolve();a.output=a.output.resolve()
    manifest=verify_runtime(release);prep=read(release/'preparation.json');cpu=read(release/'cpu-admission.json')
    if not cpu['passed'] or len(cpu['decks'])!=4 or {r['deck'] for r in cpu['decks']}!=set(prep['decks']):
        raise ValueError('all four fit decks require actual CPU admission')
    if manifest['native_sha256']!=cpu['native_sha256'] or cpu['native_sha256']!=prep['native_sha256']:
        raise ValueError('CPU/native binding mismatch')
    for item in cpu['decks']:
        if sha(release/item['report'])!=item['sha256']:raise ValueError('CPU report changed')
    a.output.mkdir(parents=True,exist_ok=False)
    repo=release/'repo';adapter=release/'build-provenance/compiled-adapter.h'
    env=dict(os.environ,PYTHONPATH=str(repo/'ygoenv')+':'+str(repo),JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def command(arguments,label):
        with (a.output/(label+'.log')).open('x') as log:
            result=subprocess.run([sys.executable,*arguments],env=env,cwd=release,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        write(a.output/(label+'.exit.json'),dict(returncode=result.returncode,arguments=arguments))
        result.check_returncode()
    for suffix in ('a','b'):
        command([str(repo/'scripts/audit_ygocore_protocol.py'),'--core-source',str(a.core_source),
            '--adapter',str(adapter),'--expected-revision',a.expected_core_revision,
            '--output',str(a.output/('contract-'+suffix+'.json'))],'contract-'+suffix)
    contract=read(a.output/'contract-a.json')
    if contract!=read(a.output/'contract-b.json'):raise ValueError('source contract is not reproducible')
    # Historical evidence remains explicitly historical; it is never relabelled
    # as fifty games on this native, deck pool or model.
    history=a.output/'historical-runtime-smoke-50-v4.json';shutil.copy2(a.runtime_smoke,history)
    command(['-m','unittest','discover','-s',str(repo/'tests'),'-p','test_ygocore_protocol_boundaries.py'],'native-boundaries')
    coverage=[]
    for index,item in enumerate(cpu['decks']):
        path=a.output/('coverage-'+str(index)+'.json')
        command([str(repo/'scripts/build_protocol_coverage.py'),'--contract',str(a.output/'contract-a.json'),
            '--adapter',str(adapter),'--boundary-test',str(repo/'tests/test_ygocore_protocol_boundaries.py'),
            '--runtime-smoke',str(history),'--structured-environment',str(release/item['report']),
            '--output',str(path)],'coverage-'+str(index))
        report=read(path)
        if report['summary']['uncovered_branches'] or report['contract_sha256']!=contract['contract_sha256']:
            raise ValueError('protocol coverage incomplete')
        coverage.append(dict(deck=item['deck'],path=str(path),sha256=sha(path),summary=report['summary']))
    verify_runtime(release)
    write(a.output/'admission.json',dict(passed=True,runtime_manifest_sha256=sha(release/'frozen-manifest.json'),
        native_sha256=prep['native_sha256'],contract_sha256=contract['contract_sha256'],coverage=coverage,
        model_inference=False,gpu_optimization_admitted=False,
        scope='Actual current-native protocol boundaries and all four current CPU-deck reports. Retained fifty-game evidence is historical only.'))
    print('PROTOCOL_ADMITTED',len(coverage),contract['contract_sha256'],flush=True)


if __name__=='__main__':main()
