"""Bind an isolated audit-core revision only after every source row replays."""
import argparse
import copy
from pathlib import Path
import sys
from contracts import read,sha,write

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
from ygoai.rl.core_audit import audit_record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','library','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.source=a.source.resolve();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    source_sha=sha(a.source/'plan.json')
    plan=read(a.source/'plan.json');binding=dict(path=str(a.library.resolve()),sha256=sha(a.library),
        manifest=str(a.manifest.resolve()),manifest_sha256=sha(a.manifest))
    runtime=Path(plan['runtime']);results=[];revision=copy.deepcopy(plan)
    for row in revision['rows']:
        path=(a.source/row['path']).resolve()
        if sha(path)!=row['record_sha256'] or sha(path.with_suffix('.npz'))!=row['observations_sha256']:
            raise ValueError('source teaching data changed')
        proof=audit_record(read(path),plan['source_root'],runtime/'assets/cards.cdb',runtime/'assets/script',binding)
        if not proof['passed']:raise ValueError('independent compatibility audit rejected '+row['id'])
        row['path']=str(path)
        results.append(dict(id=row['id'],record_sha256=row['record_sha256'],audit=proof))
        print(row['id'],'verified',flush=True)
    if sha(a.source/'plan.json')!=source_sha:
        raise ValueError('source plan changed')
    proof=dict(passed=True,source_plan=str(a.source/'plan.json'),source_plan_sha256=source_sha,
        independent_library=binding,rows=results,model_inference=False,
        native_sha256=plan['native_sha256'],teacher_data_changed=False)
    write(a.output/'independent-replay-admission.json',proof)
    revision.update(independent_library=binding,independent_compatibility_admission=dict(
        path=str(a.output/'independent-replay-admission.json'),sha256=sha(a.output/'independent-replay-admission.json')))
    write(a.output/'plan.json',revision)
    print('ADMITTED',len(results),flush=True)


if __name__=='__main__':main()
