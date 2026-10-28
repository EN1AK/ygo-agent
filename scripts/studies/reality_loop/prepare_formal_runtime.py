"""Freeze the actual paired normal native and fit decks; qualify every deck.

No checkpoint selection, model optimization or independent-set inference.
Production native and assets stay untouched; failures preserve the new release.
"""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import tarfile

from contracts import digest,read,sha,write

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
from ygoai.rl.core_audit import validate_curriculum_binding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('course','source-archive','native-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--archive-sha256',required=True)
    a=p.parse_args();a.output=a.output.resolve();a.course=a.course.resolve()
    if os.name=='posix':
        import resource
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if sha(a.source_archive)!=a.archive_sha256:raise ValueError('source archive transport mismatch')
    plan=read(a.course/'plan.json');plan_sha=sha(a.course/'plan.json')
    validate_curriculum_binding(plan)
    if plan.get('full_legal_menu') is not False or plan.get('training_admission_pending'):
        raise ValueError('admitted filtered course required')
    a.output.mkdir(parents=True,exist_ok=False);repo=a.output/'repo';repo.mkdir()
    with tarfile.open(a.source_archive) as archive:
        for member in archive:
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts:raise ValueError('unsafe archive member')
            target=repo/member.name
            if member.isdir():target.mkdir(parents=True,exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True,exist_ok=True)
                data=archive.extractfile(member).read()
                target.write_bytes(data.replace(b'\r\n',b'\n') if target.suffix=='.sh' else data)
            else:raise ValueError('only regular tracked files supported')
    import hashlib
    canonical=lambda path:hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    files={str(path.relative_to(repo)):canonical(path) for path in sorted(repo.rglob('*')) if path.is_file()}
    runtime=Path(plan['runtime']);native=a.native_dir/'ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so'
    diagnostic=a.native_dir/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    if sha(diagnostic)!=plan['native_sha256']:raise ValueError('normal/diagnostic pair does not match the course')
    provenance=a.output/'build-provenance';provenance.mkdir()
    for name in ('compiled-adapter.h','compiled-binding.cpp','source-binding.json'):
        shutil.copy2(a.native_dir/name,provenance/name)
    binding=read(provenance/'source-binding.json')
    if sha(provenance/'compiled-adapter.h')!=binding['header_sha256']:raise ValueError('compiled header changed')
    module_path=repo/'ygoenv/ygoenv/ygopro'/native.name
    shutil.copy2(native,module_path)
    spec=importlib.util.spec_from_file_location('ygopro_ygoenv',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.exercise_training_header_sha256!=binding['header_sha256']:raise ValueError('native header binding mismatch')
    destinations={repo/'assets/locale/zh/cards.cdb':runtime/'assets/cards.cdb',
                  repo/'scripts/script':runtime/'assets/script',a.output/'semantics':runtime/'semantics',
                  a.output/'code_list.txt':runtime/'code_list.txt'}
    for target,source in destinations.items():
        if source.is_dir():shutil.copytree(source,target)
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    # Tracked inputs must remain exact after runtime-only assets are supplied.
    if not all(canonical(repo/name)==value for name,value in files.items()):raise ValueError('runtime assets overwrote tracked source')
    decks={}
    for item in plan['rows']:
        if item['split']!='fit':continue
        record_path=a.course/item['path']
        if sha(record_path)!=item['record_sha256']:raise ValueError('source deck record changed')
        definition=read(record_path)['definition'];seat=definition['controlled']
        if definition.get('lua') or any(c[2] not in (1,64) for c in definition['initial']):continue
        main=[c[0] for c in definition['initial'] if c[1:3]==[seat,1]]
        extra=[c[0] for c in definition['initial'] if c[1:3]==[seat,64]]
        if not 40<=len(main)<=60 or len(extra)>15 or max(Counter(main+extra).values())>3:continue
        key=digest([sorted(main),sorted(extra),definition['duel_flags']])
        decks.setdefault(key,dict(main=main,extra=extra,flags=definition['duel_flags'],record_sha256=item['record_sha256']))
    chosen=sorted(decks)[:4]
    if len(chosen)!=4:raise ValueError('four distinct valid fit deck multisets required')
    deck_hashes={};(a.output/'decks').mkdir()
    for key in chosen:
        d=decks[key]
        if d['flags']>>16!=5:raise ValueError('fit deck rule version differs from normal native MR5')
        name='decks/fit-'+key[:12]+'.ydk';path=a.output/name
        path.write_text('#main\n'+'\n'.join(map(str,d['main']))+'\n#extra\n'+'\n'.join(map(str,d['extra']))+'\n!side\n')
        deck_hashes[name]=sha(path)
    primary=next(iter(deck_hashes))
    write(a.output/'source-manifest.json',dict(commit=a.source_commit,archive_sha256=a.archive_sha256,all_files=files))
    preparation=dict(source=a.source_commit,native_sha256=sha(native),diagnostic_sha256=sha(diagnostic),
        header_sha256=binding['header_sha256'],chain_feature_version=module.chain_event_provenance_version,
        course_plan_sha256=plan_sha,course=str(a.course),deck=primary,deck_sha256=deck_hashes[primary],
        decks=deck_hashes,deck_sources={k:decks[k] for k in chosen},full_legal_menu=False,matmul_precision='highest',
        code_list_sha256=sha(a.output/'code_list.txt'),semantic_metadata_sha256=sha(a.output/'semantics/metadata.json'))
    write(a.output/'preparation.json',preparation)
    env=dict(os.environ,JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             PYTHONPATH=str(repo/'ygoenv')+':'+str(repo))
    results=[]
    for index,name in enumerate(deck_hashes):
        attempt='cpu-deck-'+str(index)
        with (a.output/(attempt+'.log')).open('x') as log:
            result=subprocess.run([sys.executable,str(repo/'scripts/validate_formal_runtime.py'),
                '--release',str(a.output),'--deck',name,'--attempt',attempt],env=env,cwd=a.output,
                stdout=log,stderr=subprocess.STDOUT,timeout=300)
        (a.output/(attempt+'.exit')).write_text(str(result.returncode)+'\n')
        result.check_returncode()
        proof=read(a.output/'validation'/attempt/'report.json')
        if not proof['passed'] or proof['native_sha256']!=preparation['native_sha256']:raise ValueError('CPU runtime admission failed')
        results.append(dict(deck=name,report=str(Path('validation')/attempt/'report.json'),sha256=sha(a.output/'validation'/attempt/'report.json')))
    if sha(a.course/'plan.json')!=plan_sha:raise ValueError('course changed during qualification')
    write(a.output/'cpu-admission.json',dict(passed=True,decks=results,source_commit=a.source_commit,
        native_sha256=preparation['native_sha256'],model_inference=False,gpu_optimization_admitted=False))
    artifacts=[dict(path=str(path.relative_to(a.output)),sha256=sha(path),bytes=path.stat().st_size)
        for path in sorted(a.output.rglob('*')) if path.is_file() and '__pycache__' not in path.parts]
    write(a.output/'frozen-manifest.json',dict(format_version=1,source_commit=a.source_commit,
        native_sha256=preparation['native_sha256'],artifacts=artifacts))
    print('CPU_RUNTIME_ADMITTED',len(results),preparation['native_sha256'],flush=True)


if __name__=='__main__':main()
