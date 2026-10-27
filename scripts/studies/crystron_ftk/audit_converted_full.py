"""Re-execute admitted actor response streams in an independent modern core."""
import json,pathlib,subprocess,sys,hashlib
from modern_core import Core
root=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text('utf-8'))
if len(sys.argv)==1:
    admission=read(root/'conversion-summary-v4.json');assert len(admission['converted'])==110 and not admission['rejected']
    results=[]
    for sha in admission['converted']:
        process=subprocess.run([sys.executable,__file__,sha],capture_output=True,text=True,timeout=40)
        result=json.loads(process.stdout) if process.returncode==0 else dict(sha=sha,passed=False,error=process.stderr[-1500:],exit_code=process.returncode)
        results.append(result);print(json.dumps(result),flush=True)
    (root/'interface-independent-audit-v1.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in results),results=results),indent=2))
    sys.exit(0 if all(r['passed'] for r in results) else 1)
sha=sys.argv[1];record=read(root/'actor-converted-v4'/(sha+'.json'));source=read(root/'parsed.json')[sha];proof=read(root/'audit-repaired-v7'/(sha+'.json'))
c=Core(root/'resources/libocgcore-current-v2.so',root/'resources/repair-v6/cards.cdb',root/'resources/repair-v6/script',seed_sequence=source['seed_sequence'])
try:
    c.start_standard([[code,p,loc,0,8] for p in (0,1) for loc,deck in ((1,source['main_decks'][p]),(64,source['extra_decks'][p])) for code in deck],duel_flags=source['duel_flags'])
    for i,response in enumerate(record['core_responses']):
        prompt=c.next_prompt();assert prompt==proof['decisions'][i]['prompt'],('prompt drift',i)
        c.respond(prompt,bytes.fromhex(response['data']))
    assert c.next_prompt() is None and c.winner==0 and c.lp[1]<=0
    assert c.events==proof['events'] and c.state()==record['final']==proof['final']
    result=dict(sha=sha,passed=True,responses=len(record['core_responses']),model_decisions=len(record['decisions']),events=len(c.events),padding_differences=record['zero_padding_differences'],display_wire_difference=not record['wire_events_equal_excluding_layout'],final_state_sha256=hashlib.sha256(json.dumps(c.state(),sort_keys=True).encode()).hexdigest())
finally:c.close()
print(json.dumps(result))
