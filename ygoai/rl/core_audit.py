"""Launch independent engine checks outside the model/CUDA process."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def audit_record(record,source,database,scripts):
    worker=Path(__file__).resolve().parents[2]/'scripts/studies/reality_loop/audit_core_replay.py'
    request=dict(record=record,source=str(source),database=str(database),scripts=str(scripts),max_variants=8)
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
