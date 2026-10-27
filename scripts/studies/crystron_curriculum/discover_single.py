"""Bounded proposals from distinct references; failed guides remain unknown."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','runtime','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    for k in ('source','runtime','output'):setattr(a,k,getattr(a,k).resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    proposals=[]
    for path in sorted((a.source/'actor-converted-v4').glob('*.json')):
        r=json.loads(path.read_text('utf-8'));seat=r['definition']['controlled']
        own=[d for d in r['decisions'] if d['player']==seat]
        if not own or own[0]['message']!=11:continue
        action=own[0]['menu'][own[0]['action']]
        if action['code'] not in (4334811,31552317,83443619,99471856,25865565):continue
        proposals.append(dict(sha=path.stem,starter=action['code']))
    (a.output/'proposals.json').write_text(json.dumps(proposals,indent=2),'utf-8')
    def run(row):
        folder=a.output/(str(row['starter'])+'-'+row['sha'][:12])
        command=[sys.executable,str(Path(__file__).with_name('collect_single.py')),'--source',str(a.source),'--runtime',str(a.runtime),'--output',str(folder),'--sha',row['sha'],'--starter',str(row['starter'])]
        try:
            proc=subprocess.run(command,capture_output=True,text=True,timeout=60)
            result=dict(**row,exit_code=proc.returncode,stdout=proc.stdout[-6000:],stderr=proc.stderr[-2000:])
        except subprocess.TimeoutExpired:result=dict(**row,error='60s timeout; unknown')
        print(json.dumps(result),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,proposals))
    (a.output/'run-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),'utf-8')


if __name__=='__main__':main()
