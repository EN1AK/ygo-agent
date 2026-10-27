"""Convert non-FTK terminal evidence and a deterministic partial-record sample."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','catalog','additional','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--prefix-count',type=int,default=12)
    p.add_argument('--runtime',type=Path)
    a=p.parse_args()
    for k in ('source','catalog','additional','output'):setattr(a,k,getattr(a,k).resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    rows=json.loads((a.catalog/'catalog.json').read_text('utf-8'))
    cases=[];prefix=[]
    for row in rows:
        sha=row['source_sha256']
        proof=a.source/'audit-repaired-v7'/(sha+'.json')
        if not proof.exists():proof=a.additional/(sha+'.json')
        if not proof.exists():continue
        r=json.loads(proof.read_text('utf-8'))
        if r['status'] not in ('complete','prefix_only'):continue
        entry=dict(sha=sha,proof=str(proof),proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),status=r['status'])
        if r['status']=='complete':
            seat=r['source']['seat'];turns=sum(e['op']==40 for e in r['events'])
            ftk=r['final'].get('winner')==seat and turns==1 and r['final']['lp'][1-seat]<=0
            if not ftk:cases.append(entry)
        else:prefix.append(entry)
    cases+=sorted(prefix,key=lambda x:hashlib.sha256(('prefix-v1:'+x['sha']).encode()).hexdigest())[:a.prefix_count]
    (a.output/'frozen-intake.json').write_text(json.dumps(cases,indent=2),'utf-8')
    def run(case):
        folder=a.output/case['sha'][:12]
        command=[sys.executable,str(Path(__file__).with_name('convert_replay.py')),'--source',str(a.source),'--catalog',str(a.catalog),'--proof',case['proof'],'--output',str(folder)]
        if a.runtime:command+=['--runtime',str(a.runtime.resolve())]
        try:
            proc=subprocess.run(command,capture_output=True,text=True,timeout=90)
            result=dict(sha=case['sha'],exit_code=proc.returncode,stdout=proc.stdout[-1500:],stderr=proc.stderr[-1500:])
        except subprocess.TimeoutExpired:result=dict(sha=case['sha'],error='90s timeout; unknown')
        print(json.dumps(result),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,cases))
    (a.output/'run-results.json').write_text(json.dumps(results,indent=2),'utf-8')


if __name__=='__main__':main()
