"""Launch independent engine checks outside the model/CUDA process."""
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_curriculum_binding(plan,parent_path,output):
    """Carry exact parent proofs; independently replay every added record."""
    if plan.get('independent_library') is None:return
    parent_path=Path(parent_path);source_sha=_sha(parent_path)
    parent=json.loads(parent_path.read_text());validate_curriculum_binding(parent)
    if (parent['independent_library']!=plan['independent_library'] or
            parent['native_sha256']!=plan['native_sha256']):
        raise ValueError('derived curriculum changed the replay runtime')
    admitted=json.loads(Path(parent['independent_compatibility_admission']['path']).read_text())
    previous={(row['id'],row['record_sha256']):row for row in admitted['rows']}
    runtime=Path(plan['runtime']);rows=[];fresh=0
    for item in plan['rows']:
        path=Path(item['path'])
        if _sha(path)!=item['record_sha256'] or _sha(path.with_suffix('.npz'))!=item['observations_sha256']:
            raise ValueError('derived curriculum input changed')
        key=(item['id'],item['record_sha256'])
        if key in previous:rows.append(previous[key]);continue
        proof=audit_record(json.loads(path.read_text()),plan['source_root'],runtime/'assets/cards.cdb',
                           runtime/'assets/script',plan['independent_library'])
        rows.append(dict(id=item['id'],record_sha256=item['record_sha256'],audit=proof));fresh+=1
    if _sha(parent_path)!=source_sha:raise ValueError('parent curriculum changed during derivation')
    proof=dict(passed=True,source_plan=str(parent_path.resolve()),source_plan_sha256=source_sha,
        independent_library=plan['independent_library'],rows=rows,model_inference=False,
        native_sha256=plan['native_sha256'],teacher_data_changed=False,
        inherited_rows=len(rows)-fresh,newly_replayed_rows=fresh)
    output=Path(output)
    with output.open('x') as stream:json.dump(proof,stream,indent=2)
    plan['independent_compatibility_admission']=dict(path=str(output.resolve()),sha256=_sha(output))
    validate_curriculum_binding(plan)


def validate_curriculum_binding(plan):
    """A changed audit backend requires a hash-bound proof for every row."""
    if plan.get('independent_library') is None:return
    binding=plan.get('independent_compatibility_admission')
    if not binding:raise ValueError('independent core revision lacks curriculum admission')
    if _sha(binding['path'])!=binding['sha256']:
        raise ValueError('independent curriculum admission changed')
    proof=json.loads(Path(binding['path']).read_text())
    expected={row['id']:row['record_sha256'] for row in plan['rows']}
    actual={row['id']:row['record_sha256'] for row in proof['rows']}
    if (not proof['passed'] or not all(row['audit']['passed'] and
            row['audit']['independent_library_sha256']==plan['independent_library']['sha256'] for row in proof['rows']) or
            len(expected)!=len(plan['rows']) or len(actual)!=len(proof['rows']) or expected!=actual or
            proof['independent_library']!=plan['independent_library'] or
            proof['native_sha256']!=plan['native_sha256'] or proof['teacher_data_changed'] is not False or
            _sha(proof['source_plan'])!=proof['source_plan_sha256']):
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
