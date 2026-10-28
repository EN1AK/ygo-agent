"""Launch independent engine checks outside the model/CUDA process."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def validate_curriculum_binding(plan):
    """A changed audit backend requires a hash-bound proof for every row."""
    if plan.get('independent_library') is None:return
    binding=plan.get('independent_compatibility_admission')
    if not binding:raise ValueError('independent core revision lacks curriculum admission')
    import hashlib
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest(binding['path'])!=binding['sha256']:
        raise ValueError('independent curriculum admission changed')
    proof=json.loads(Path(binding['path']).read_text())
    expected={row['id']:row['record_sha256'] for row in plan['rows']}
    actual={row['id']:row['record_sha256'] for row in proof['rows']}
    if (not proof['passed'] or not all(row['audit']['passed'] and
            row['audit']['independent_library_sha256']==plan['independent_library']['sha256'] for row in proof['rows']) or
            len(expected)!=len(plan['rows']) or len(actual)!=len(proof['rows']) or expected!=actual or
            proof['independent_library']!=plan['independent_library'] or
            proof['native_sha256']!=plan['native_sha256'] or proof['teacher_data_changed'] is not False or
            digest(proof['source_plan'])!=proof['source_plan_sha256']):
        raise ValueError('independent curriculum admission does not cover the frozen course')


def audit_record(record,source,database,scripts,library=None):
    worker=Path(__file__).resolve().parents[2]/'scripts/studies/reality_loop/audit_core_replay.py'
    request=dict(record=record,source=str(source),database=str(database),scripts=str(scripts),max_variants=8)
    if library is not None:request['library']=library
    process=subprocess.Popen([sys.executable,'-I',str(worker)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,text=True,start_new_session=True,
        env=dict(os.environ,JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    try:out,err=process.communicate(json.dumps(request),timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM);process.communicate()
        raise ValueError('independent core audit time budget exhausted')
    lines=[line[len('AUDIT_RESULT '):] for line in out.splitlines() if line.startswith('AUDIT_RESULT ')]
    if process.returncode or len(lines)!=1:raise ValueError('independent audit worker failed: '+err[-1500:])
    result=json.loads(lines[0])
    if not result.get('passed'):raise ValueError(result.get('error','independent audit failed'))
    return result
