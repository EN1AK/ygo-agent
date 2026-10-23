"""Turn audited same-root defense failures into reviewable, unadmitted candidates."""
import argparse,json
from pathlib import Path
import _repo_bootstrap
from ygoai.rl.defense_candidates import from_files


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--endpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summary=json.loads((a.endpoint/'summary.json').read_text());a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for case in sorted({r['name'] for r in summary['cases'] if r['split']=='fit'}):
        try:
            candidate=from_files(a.endpoint/(case+'-scripted.json'),a.endpoint/(case+'-policy.json'))
            with (a.output/(case+'.json')).open('x') as f:json.dump(candidate,f,indent=2)
            rows.append(dict(case=case,status=candidate['status'],root_step=candidate['root_step']))
        except (AssertionError,ValueError,KeyError) as exc:
            rows.append(dict(case=case,status='not_admitted',reason=str(exc)))
    with (a.output/'summary.json').open('x') as f:json.dump(dict(candidates=rows,training_started=False),f,indent=2)
    print(json.dumps(dict(candidates=rows,training_started=False)),flush=True)


if __name__=='__main__':main()
