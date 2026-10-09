"""Independent process: the legacy core's reader callbacks are process-global."""
import argparse
import json
from pathlib import Path
from scripts.audit_teaching_exercise import audit
from scripts.verify_exercise_actor_responses import verify
from ygoai.rl.exercise_core import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('native','database','scripts','input','output'): p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for path in sorted(a.input.glob('*-0.json')):
        rows.append(verify(path,a) if path.name.startswith('synthetic-') else audit(path,a))
    assert rows and all(r['passed'] for r in rows)
    with a.output.open('x') as f:
        json.dump(dict(passed=True,native_sha256=sha(a.native),results=rows),f,indent=2)
    print(json.dumps(dict(independent_audit=str(a.output),trajectories=len(rows),passed=True)),flush=True)


if __name__=='__main__': main()
