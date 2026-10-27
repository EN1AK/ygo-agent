"""Require an independent modern-core win before admitting search teaching."""
import hashlib,json,pathlib,subprocess,sys
import numpy as np
root=pathlib.Path(__file__).resolve().parent;project=root.parents[1]/'ygo-agent';sys.path.insert(0,str(project))
from modern_core import Core
from ygoai.rl.human_replay import comparable_game_events
from ygoai.rl.exercise_actor import array_digest
def read(p):return json.loads(p.read_text('utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf-8')
out=root/'recovery-search-v3';admitted=out/'admitted';admitted.mkdir(exist_ok=True)
if len(sys.argv)==1:
    summary=read(out/'summary.json');results=[]
    for item in summary['results']:
        if item['status']!='native_verified_solution':continue
        process=subprocess.run([sys.executable,__file__,item['id']],capture_output=True,text=True,timeout=60)
        result=json.loads(process.stdout) if process.returncode==0 else dict(id=item['id'],passed=False,error=process.stderr[-2000:])
        results.append(result);print(json.dumps(result),flush=True)
    write(out/'independent-certification.json',dict(completed=True,results=results,admitted=sum(r['passed'] for r in results)));sys.exit(0)
identifier=sys.argv[1];source_path=out/(identifier+'.json');result=read(source_path);record=result['record'];case=result['case']
assert result['status']=='native_verified_solution' and case['split']=='fit' and len(record['decisions'])>len(case['prefix_actions'])
assert [d['action'] for d in record['decisions'][:len(case['prefix_actions'])]]==case['prefix_actions']
original=read(root/'actor-converted-v4'/(case['sha']+'.json'))
assert case['prefix_actions'][case['divergence_step']]!=original['decisions'][case['divergence_step']]['action']
assert not any(d['teaching'] for d in record['decisions'][:result['recovery_root']])
assert any(d['teaching'] for d in record['decisions'][result['recovery_root']:])
with np.load(out/(identifier+'.npz'),allow_pickle=False) as data:
    obs=dict(data)
    for i,d in enumerate(record['decisions']):assert array_digest({k:v[i:i+1] for k,v in obs.items()})==d['observation_sha256']
source=read(root/'parsed.json')[case['sha']]
c=Core(root/'resources/libocgcore-current-v2.so',root/'resources/repair-v6/cards.cdb',root/'resources/repair-v6/script',seed_sequence=source['seed_sequence'])
try:
    c.start_standard([[code,p,loc,0,8] for p in (0,1) for loc,deck in ((1,source['main_decks'][p]),(64,source['extra_decks'][p])) for code in deck],duel_flags=source['duel_flags'])
    for response in record['core_responses']:
        prompt=c.next_prompt();assert prompt is not None;c.respond(prompt,bytes.fromhex(response['data']))
    assert c.next_prompt() is None and c.winner==0 and c.lp[1]<=0 and sum(e['op']==40 for e in c.events)==1
    assert comparable_game_events(c.events)==comparable_game_events(record['events'])
    assert c.state()==record['final']
    proof=dict(id=identifier,passed=True,source_sha256=case['sha'],source_result_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),divergence_step=case['divergence_step'],recovery_root=result['recovery_root'],search_branch_steps=len(result['branch']),reference_join=result['reference_join'],final=c.state(),modern_event_sha256=hashlib.sha256(json.dumps(c.events,sort_keys=True).encode()).hexdigest(),teacher_decisions=sum(d['teaching'] for d in record['decisions']))
finally:c.close()
record.update(training_ready=True,status='certified_recovery',independent_proof=proof)
write(admitted/(identifier+'.json'),record)
(admitted/(identifier+'.npz')).write_bytes((out/(identifier+'.npz')).read_bytes())
print(json.dumps(proof))
