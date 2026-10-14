"""Independent completion gates for the bounded elfnote search study."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from ygoai.rl.exercise_core import sha


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['study','source','native','database','scripts','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();s=a.study
 assert Path(str(s)+'.exit').read_text().strip()=='0'
 assert Path(str(s)+'.collection.exit').read_text().strip()=='0'
 m=read(s/'manifest.json');plan=m['plan'];collection=read(s/'collection.json');summary=read(s/'summary.json')
 assert collection['protected_unchanged'] and summary['protected_unchanged']
 assert all(sha(path)==v for path,v in m['protected'].items())
 assert all(sha(a.source/path)==v for path,v in m['source_files'].items())
 source_manifest=read(a.source/'source-manifest.json')
 assert all(hashlib.sha256((a.source/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==v for path,v in source_manifest['all_files'].items())
 assert all(sha(s.parent/path)==v for path,v in read(s/'reference-hashes.json').items())
 cases=m['cases'];train=[c for c in cases if c['split']=='train'];reserved=[c for c in cases if c['split']=='reserved']
 assert len(train)==12 and len(reserved)==6
 results=collection['results'];assert len(results)==24
 before=read(s/'before/summary.json');lookup={c['name']:i for i,c in enumerate(before['cases'])}
 witnesses=0;cost=[]
 for mode in ['autonomous','search']:
  group=[r for r in results if r['mode']==mode];assert len(group)==12
  assert {r['case'] for r in group}=={c['name'] for c in train}
  for r in group:
   assert len(r['trials'])<=plan['trials_per_case']
   assert r['decisions']==sum(t['decisions'] for t in r['trials'])
   if 'witness' not in r:continue
   witnesses+=1;path=Path(r['witness']);assert sha(path)==r['record_sha256']
   record=read(path);baseline=read(s/f"before/case-{lookup[r['case']]:02}.json")
   assert record['status']=='success' and record['definition']==baseline['definition']
   assert record['root_step']==baseline['root_step']
   root=record['root_step']
   assert record['decisions'][:root]==baseline['decisions'][:root]
   assert all(not d['teaching'] for d in record['decisions'][:root])
   assert all(d['teaching']==(d['player']==0) for d in record['decisions'][root:])
  cost.append(dict(mode=mode,solved=sum('witness' in r for r in group),trials=sum(len(r['trials']) for r in group),actor_decisions_including_prefix_replay=sum(r['decisions'] for r in group),seconds=sum(r['seconds'] for r in group)))
 curves=[]
 for arm in summary['arms']:
  folder=s/(arm['mode']+'-trained');admission=read(folder/'admission.json')
  assert admission['max_logit_error']<.002 and admission['lessons']==arm['lessons']
  metrics=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
  assert [x['update'] for x in metrics]==list(range(1,plan['updates_per_arm']+1))
  assert all(np.isfinite([x['loss'],x['grad_norm']]+x['ce']+x['kl']).all() for x in metrics)
  assert sha(folder/'model.flax_model')==arm['checkpoint_sha256']
  assert read(folder/'endpoint/summary.json')==arm['result']
  assert read(folder/'suffix-endpoint/summary.json')==arm['suffix']
  for step in plan['probe_updates']:
   probe=read(folder/f'probe-{step}/summary.json')
   assert all(c['split']!='reserved' for c in probe['cases'])
   curves.append(dict(mode=arm['mode'],update=step,train=sum(c['status']=='success' for c in probe['cases'] if c['split']=='train'),original=sum(c['status']=='success' for c in probe['cases'] if c['split']=='original'),old=sum(c['success'] for c in probe['old'])))
 proofs=list(s.rglob('*.audit.json'))
 expected=3*len(before['cases'])+sum(c['root_step'] is not None for c in before['cases'][:18])*2+witnesses
 assert len(proofs)==expected,(len(proofs),expected)
 for path in proofs:
  proof=read(path);record=path.with_name(path.name.replace('.audit.json','.json'))
  assert proof['passed'] and proof['record_sha256']==sha(record) and proof['observations_sha256']==sha(record.with_suffix('.npz'))
  assert proof['native_sha256']==sha(a.native)
 old_paths=sorted(s.glob('**/old/[0-4].json'));assert len(old_paths)==65
 old=[]
 for path in old_paths:
  result=subprocess.run([sys.executable,'-m','scripts.verify_elfnote_study',*sys.argv[1:],'--old-record',str(path)],check=True,capture_output=True,text=True,cwd=a.source)
  proof=json.loads(result.stdout.strip().splitlines()[-1]);assert proof['passed'];proof['path']=str(path.relative_to(s));old.append(proof)
 result=dict(passed=True,new_audits=len(proofs),old_audits=old,witnesses=witnesses,discovery=cost,learning_curve=curves,summary_sha256=sha(s/'summary.json'),protected_unchanged=True)
 with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(dict(passed=True,new_audits=len(proofs),old_audits=len(old),discovery=cost)))

if __name__=='__main__':main()
