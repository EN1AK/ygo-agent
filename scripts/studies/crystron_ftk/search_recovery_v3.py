"""Search from actual off-reference policy states, then certify full suffixes.

CPU-only teacher search. Reference-state matches propose suffixes; only a
complete engine execution proves recovery. Unsearched branches stay unknown.
"""
import collections,hashlib,json,pathlib,subprocess,sys,time,heapq
from types import SimpleNamespace
import numpy as np
root=pathlib.Path(__file__).resolve().parent;deployment=root.parents[1];project=deployment/'ygo-agent'
sys.path[:0]=[str(project),str(project/'scripts')]
from scripts.collect_teaching_exercise import load_native
from ygoai.rl.exercise_actor import array_digest,extract_trace
from ygoai.rl.observation_schema import tensor_contract
import exercise_core,ygoai.rl.exercise_actor as trace_module
trace_module.messages=exercise_core.messages
def read(p):return json.loads(p.read_text('utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf-8')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
plan=read(root/'learning-plan-v1.json');out=root/'recovery-search-v3';out.mkdir(exist_ok=True)
if len(sys.argv)==1:
    summary=read(root/'learning-v1/evaluation/full/summary.json');cases=[]
    for item in summary['results']:
        if item['split']!='fit' or item['status']=='success':continue
        path=pathlib.Path(item['path']);rollout=read(path);reference=read(root/plan['trajectories']/(item['sha']+'.json'))
        for d in rollout['decisions']:
            i=d['step']
            if d['source']!='model':continue
            if i>=len(reference['decisions']):break
            expected=reference['decisions'][i]
            if d['observation_sha256']!=expected['observation_sha256']:break
            if d['selected_action']==expected['action']:continue
            prefix=[x['selected_action'] for x in rollout['decisions'][:i+1]]
            cases.append(dict(id=item['sha'][:12]+'-'+item['mode'],sha=item['sha'],split='fit',rollout_path=str(path),rollout_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),divergence_step=i,wrong_action=d['legal_actions'][d['selected_action']],reference_action=expected['menu'][expected['action']],prefix_actions=prefix,max_nodes=192,max_depth=6,max_children=16))
            break
    unique={}
    for c in cases:unique.setdefault((c['sha'],tuple(c['prefix_actions'])),c)
    cases=list(unique.values());write(out/'plan.json',dict(cases=cases,source_checkpoint_sha256=summary['checkpoint_sha256'],method='reference-guided bounded legal-action search from the state AFTER a model deviation; own-deck alias matches only propose suffixes; each proposed choice is core-legal and the complete execution must certify FTK',training_scope='fit sources only; no holdout optimization'))
    results=[]
    for c in cases:
        path=out/(c['id']+'.json')
        try:
            process=subprocess.run([sys.executable,__file__,c['id']],capture_output=True,text=True,timeout=180)
            r=read(path) if path.exists() else dict(id=c['id'],status='unknown_process_failure',error=process.stderr[-2000:],exit_code=process.returncode)
        except subprocess.TimeoutExpired:
            r=dict(id=c['id'],status='unknown_time_budget')
        results.append({k:v for k,v in r.items() if k not in ('record','events','final','core_responses')});print(json.dumps(results[-1]),flush=True)
    write(out/'summary.json',dict(completed=True,results=results));sys.exit(0)
case=next(c for c in read(out/'plan.json')['cases'] if c['id']==sys.argv[1]);reference=read(root/plan['trajectories']/(case['sha']+'.json'));runtime=root/plan['runtime']
args=SimpleNamespace(native=runtime/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so',database=runtime/'assets/cards.cdb',scripts=runtime/'assets/script',code_list=runtime/'code_list.txt',semantics=runtime/'semantics');native=load_native(args,runtime/'registration.ydk')
definition=reference['definition']
baseline=read(pathlib.Path(case['rollout_path']))
assert hashlib.sha256(pathlib.Path(case['rollout_path']).read_bytes()).hexdigest()==case['rollout_sha256']
def make_env(actions=()):
    e=native.ExerciseActor('',definition['initial'],str(args.semantics),1,'structured-lite-v1',definition['seed_sequence'],definition['duel_flags'],2048,True);e.reset()
    try:
        for i,action in enumerate(actions):
            snap=e.snapshot()
            assert not snap['done'] and not snap['invalid'], ('prefix terminal', i)
            assert 0 <= action < len(snap['menu']), ('illegal prefix action', i, action)
            if i < len(case['prefix_actions']):
                expected=baseline['decisions'][i]
                obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                assert snap['menu']==expected['legal_actions'] and snap['player']==expected['player'], ('baseline menu drift', i)
                assert array_digest(obs)==expected['observation_sha256'], ('baseline observation drift', i)
            e.step(action)
    except Exception:
        e.close()
        raise
    return e
def action_key(action):
    value=dict(action)
    if str(value.get('spec','')).isdecimal():
        value['spec']='#own-deck-card'
        value['response']=0
    return value

def key(e,s):
    events,state,responses=extract_trace(e,s)
    # These relaxed keys only propose a suffix. They never certify a join.
    candidate_state=json.loads(json.dumps(state))
    for card in candidate_state['cards']:
        if card['player']==0 and card['location']==1:card['sequence']=0
    candidate_state['cards'].sort(key=lambda c:json.dumps(c,sort_keys=True))
    candidate_menu=sorted((action_key(x) for x in s['menu']),key=lambda x:json.dumps(x,sort_keys=True))
    return digest(dict(state=candidate_state,menu=candidate_menu,player=s['player'],message=s['message'])),events,state,responses
bank=collections.defaultdict(list);frequencies=collections.Counter();e=make_env()
try:
    for i,d in enumerate(reference['decisions']):
        s=e.snapshot();assert s['menu']==d['menu'] and s['player']==d['player'];k,_,_,_=key(e,s);bank[k].append(i)
        chosen=d['menu'][d['action']];frequencies[(chosen['act'],chosen['code'],chosen['effect'])]+=1;e.step(d['action'])
finally:e.close()
started=time.monotonic();queue=[(0,0,[])];visited=set();expanded=0;sequence=1;solution=None;joins=[];root_state=None;root_menu=None
while queue and expanded<case['max_nodes']:
    _,_,branch=heapq.heappop(queue);actions=case['prefix_actions']+branch;e=make_env(actions)
    try:
        s=e.snapshot();k,events,state,responses=key(e,s);expanded+=1
        if not branch:root_state=state;root_menu=s['menu']
        if s['invalid'] or sum(x['op']==40 for x in events)>1:continue
        if s['done']:
            if state['winner']==0 and state['lp'][1]<=0:solution=dict(actions=actions,branch=branch,reference_join=None)
            if solution:break
            continue
        # Dedup only the complete response history, never an incomplete state
        # hash: hidden counters/RNG may differ despite matching visible cards.
        exact=digest(responses)
        if exact in visited:continue
        visited.add(exact)
        candidates=bank.get(k,[])
        # The adapter's trace callbacks are process-global. Never keep a
        # search-node duel alive while constructing a suffix-test duel.
        if candidates:e.close()
        for index in reversed(candidates[-3:]):
            test=make_env(actions);accepted=False;error=None;tail=[]
            try:
                for d in reference['decisions'][index:]:
                    snap=test.snapshot()
                    if snap['done'] or snap['invalid'] or snap['player']!=d['player'] or snap['message']!=d['message']:raise ValueError('suffix prompt drift')
                    target=action_key(d['menu'][d['action']])
                    choices=[j for j,m in enumerate(snap['menu']) if action_key(m)==target]
                    if not choices:raise ValueError('reference card action unavailable')
                    selected=choices[0];tail.append(selected);test.step(selected)
                final=test.snapshot();ev,st,_=extract_trace(test,final)
                accepted=final['done'] and not final['invalid'] and st['winner']==0 and st['lp'][1]<=0 and sum(x['op']==40 for x in ev)==1
            except Exception as ex:error=str(ex)
            finally:test.close()
            joins.append(dict(depth=len(branch),reference_index=index,accepted=accepted,error=error))
            if accepted:
                solution=dict(actions=actions+tail,branch=branch,reference_join=index);break
        if solution:break
        if len(branch)>=case['max_depth']:continue
        # Rank all core-legal choices by observed action type/card, then keep
        # a bounded beam. Cancel is a legitimate recovery candidate, not forced.
        ranked=[]
        for i,a in enumerate(s['menu']):
            rank=frequencies[(a['act'],a['code'],a['effect'])]+(2 if a['act']=='Cancel' else 0)
            ranked.append((-rank,i))
        for rank,i in sorted(ranked)[:case['max_children']]:
            heapq.heappush(queue,(len(branch)+1+rank*.001,sequence,branch+[i]));sequence+=1
    finally:e.close()
result=dict(id=case['id'],sha=case['sha'],split='fit',status='native_verified_solution' if solution else 'unknown_budget_or_exhausted',expanded=expanded,seconds=time.monotonic()-started,case=case,joins=joins,root_state=root_state,root_menu=root_menu,training_ready=False)
if solution:
    e=make_env();decisions=[];arrays=[]
    try:
        for i,action in enumerate(solution['actions']):
            s=e.snapshot();events,state,responses=extract_trace(e,s);assert not s['invalid'] and not s['done']
            obs={k:s['observation'][k] for k in tensor_contract('structured-lite-v1')}
            decisions.append(dict(step=i,player=s['player'],message=s['message'],menu=s['menu'],action=action,core_response_count=len(responses),observation_sha256=array_digest(obs),teaching=i>=len(case['prefix_actions']) and s['player']==0 and len(s['menu'])>1))
            arrays.append({k:v[0].copy() for k,v in obs.items()});e.step(action)
        s=e.snapshot();events,state,responses=extract_trace(e,s)
        assert s['done'] and not s['invalid'] and state['winner']==0 and state['lp'][1]<=0 and sum(x['op']==40 for x in events)==1
        result.update(branch=solution['branch'],reference_join=solution['reference_join'],recovery_root=len(case['prefix_actions']),events=events,final=state,core_responses=responses)
        result['record']=dict(schema='crystron-search-recovery-v1',definition=definition,source_sha256=case['sha'],full_legal_menu=True,training_ready=False,decisions=decisions,core_responses=responses,events=events,final=state,search_case=case['id'])
        np.savez_compressed(out/(case['id']+'.npz'),**{k:np.stack([v[k] for v in arrays]) for k in arrays[0]})
    finally:e.close()
write(out/(case['id']+'.json'),result);print(json.dumps({k:v for k,v in result.items() if k not in ('record','events','final','core_responses','root_state','root_menu','case','joins')}),flush=True)
