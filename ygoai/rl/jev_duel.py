"""Full-duel model policy and bounded replay search over the very same native interface."""
import copy
import json
import os
import queue
import random
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path


def read_deck(path):
    main,extra=[],[]; zone=None
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line=line.strip()
        if line=='#main': zone=main
        elif line=='#extra': zone=extra
        elif line.startswith('!'): zone=None
        elif line.isdigit() and zone is not None: zone.append(int(line))
    if not 40<=len(main)<=60 or len(extra)>15 or max(Counter(main+extra).values(),default=0)>3:
        raise ValueError('expected 40-60 card deck, <=15 extra, <=3 copies; no banlist validation')
    return main,extra


def deal(config):
    rng=random.Random(config['seed']); initial=[]
    for player,path in enumerate((config['deck1'],config['deck2'])):
        main,extra=read_deck(path); rng.shuffle(main)
        initial.extend([c,player,1,0,8] for c in reversed(main))
        initial.extend([c,player,64,0,8] for c in reversed(extra))
    return initial


class Worker:
    def __init__(self, config, initial, log, prefix=(), timeout=60):
        self.log=Path(log).open('w',encoding='utf-8')
        self.proc=subprocess.Popen([sys.executable,'-u','-m','scripts.jev_duel_worker'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,
            encoding='utf-8',env=dict(os.environ,PYTHONUNBUFFERED='1'))
        self.timeout=timeout; self.lines=queue.Queue()
        def read():
            for line in self.proc.stdout: self.lines.put(line)
            self.lines.put(None)
        self.reader=threading.Thread(target=read,daemon=True); self.reader.start()
        try: self.current=self.call('init',config=config,initial=initial,prefix=list(prefix))
        except BaseException:
            self.close(); raise

    def call(self, cmd, **kwargs):
        self.proc.stdin.write(json.dumps(dict(cmd=cmd,**kwargs))+'\n'); self.proc.stdin.flush()
        try: line=self.lines.get(timeout=self.timeout)
        except queue.Empty: raise TimeoutError('native worker did not respond')
        if line is None: raise RuntimeError('native worker exited; inspect worker log')
        response=json.loads(line)
        if not response['ok']: raise RuntimeError(response['error'])
        return response.get('result')

    def step(self, action):
        self.current=self.call('step',action=action)
        return self.current

    def close(self):
        if self.proc.poll() is None:
            try:
                self.call('close'); self.proc.wait(timeout=5)
            except Exception:
                self.proc.terminate()
                try: self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill(); self.proc.wait()
        for stream in (self.proc.stdin,self.proc.stdout):
            if stream: stream.close()
        self.log.close()


def request(snapshot, branches=(), probe_slots=0):
    if snapshot['done'] or snapshot['invalid'] or not snapshot['menu']:
        raise ValueError('no valid live decision')
    state=copy.deepcopy(snapshot['state'])
    state.update(goal='Win this duel under the engine rules.',simulated_branches=list(branches),
                 probes_remaining=probe_slots,
                 search_information='oracle exact hidden state; development diagnostic only' if probe_slots or branches else 'player-visible observation')
    operations=[('commit',a['index']) for a in snapshot['menu']]
    probed={b['native_action'] for b in branches}
    if probe_slots:
        operations += [('probe',a['index']) for a in snapshot['menu'] if a['index'] not in probed]
    criteria={f'option_{i}': ('Execute: ' if op=='commit' else 'Simulate: ')+snapshot['menu'][idx]['description']+f' [action {idx}]'
              for i,(op,idx) in enumerate(operations)}
    return dict(state=state,model_state=render_state(state),criteria=criteria,operations=operations,
                instruction='Choose one legal operation to win the duel. Execute answers the current engine prompt. Simulate only inspects a hypothetical continuation.')


def render_state(state):
    """Lossless table encoding avoids repeating field labels for every card."""
    state=copy.deepcopy(state)
    def pack(obj):
        if isinstance(obj,dict):
            for key,value in list(obj.items()):
                if key=='cards' and isinstance(value,list) and value:
                    columns=sorted({k for card in value for k in card})
                    obj[key]=dict(columns=columns,rows=[[c.get(k) for k in columns] for c in value])
                else: pack(value)
        elif isinstance(obj,list):
            for item in obj: pack(item)
    pack(state)
    return json.dumps(state,ensure_ascii=False,separators=(',',':'))


def choose(policy, req, greedy):
    encoded=policy.encode(req)
    with policy.torch.no_grad():
        dist=policy.distribution(encoded)
        action=dist.probs.argmax() if greedy else dist.sample()
        i=int(action.item())
        return dict(request=req,encoded=encoded,selected=i,operation=req['operations'][i],
                    old_log_prob=float(dist.log_prob(action)),probabilities=dist.probs.cpu().tolist(),
                    logits=dist.logits.cpu().tolist(),reward=0.)


def simulate(policy, config, initial, prefix, root, action, horizon, log, greedy):
    """Exact replay is explicitly oracle, never labeled fair hidden-information search."""
    worker=Worker(config,initial,log,prefix)
    decisions=[]
    try:
        if worker.current['fingerprint']!=root['fingerprint']:
            raise ValueError('branch replay did not restore exact root')
        worker.step(action)
        for _ in range(horizon-1):
            if worker.current['done']: break
            row=choose(policy,request(worker.current),greedy)
            player=worker.current['player']
            before=worker.current['response_count']
            worker.step(row['operation'][1])
            decisions.append(dict(player=player,**row,responses_before=before,responses_after=worker.current['response_count']))
        if worker.current['invalid']: raise ValueError('invalid simulator branch')
        view=worker.call('observe',viewer=root['player'])
        state=view['state']
        from ygoai.rl.jev_duel_observation import visible_event
        audit=worker.call('audit')
        events=[e for f in audit['frames'][root['frame_count']:] if
                (e:=visible_event(bytes.fromhex(f),root['player'])) is not None]
        # Every changed state field is preserved; unchanged fields inherit the root.
        delta={k:v for k,v in state.items() if k not in ('card_texts','effect_options','recent_events','omitted_older_events')
               and v!=root['state'].get(k)}
        result=dict(native_action=action,action=root['menu'][action]['description'],
                    horizon_decisions=1+len(decisions),terminal=view['done'],
                    cutoff=not view['done'],changes_from_root=delta,new_events=events)
        known={c['code'] for c in root['state'].get('card_texts',[])}
        result['new_card_texts']=[c for c in state.get('card_texts',[]) if c['code'] not in known]
        return result,dict(decisions=decisions,audit=audit,root_fingerprint=root['fingerprint'],final_observation=state)
    finally: worker.close()


def play(policy, config, output, *, greedy=False, max_decisions=1500,
         search_budget=0, probes_per_decision=2, horizon=4, probe_cost=.02):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    initial=deal(config); prefix=[]; training=[]; probes=0; protocol_types=set()
    worker=Worker(config,initial,output/'native.log')
    log=(output/'decisions.jsonl').open('w',encoding='utf-8')
    try:
        for step in range(max_decisions):
            root=worker.current
            if root['invalid']: raise ValueError('native marked duel invalid')
            if root['done']: break
            protocol_types.add(root['message'])
            branches=[]; actor=root['player']
            while True:
                slots=min(probes_per_decision-len(branches),search_budget-probes)
                row=choose(policy,request(root,branches,max(0,slots)),greedy)
                op,action=row['operation']
                row.update(step=step,player=actor,prompt=root['message'],root_fingerprint=root['fingerprint'],
                           responses_before=root['response_count'])
                search_training=[]
                if op=='probe':
                    result,evidence=simulate(policy,config,initial,prefix,root,action,horizon,
                                             output/f'branch-{probes}.log',greedy)
                    # Search must leave the main duel exactly at the original root.
                    if worker.call('snapshot')['fingerprint']!=root['fingerprint']:
                        raise ValueError('search mutated live duel')
                    row['reward']=-probe_cost; row['branch_result']=result
                    branches.append(result)
                    # The learner owns its search policy as well as its real actions.
                    # Credit its sampled continuation choices with subsequent real-duel return,
                    # never with a hypothetical branch win. The other seat is the frozen opponent.
                    if actor==0:
                        search_training=[dict(d,simulation_only=True) for d in evidence['decisions'] if d['player']==0]
                    row['learner_search_decisions']=len(search_training)
                    (output/f'branch-{probes}.json').write_text(json.dumps(evidence,ensure_ascii=False),encoding='utf-8')
                    probes+=1
                else:
                    after=worker.step(action); prefix.append(action)
                    row['responses_after']=after['response_count']
                    row['next_fingerprint']=after['fingerprint']
                log.write(json.dumps(row,ensure_ascii=False)+'\n'); log.flush()
                if actor==0:
                    training.append(row)
                    training.extend(search_training)
                if op=='commit': break
        else: raise TimeoutError('duel decision budget exhausted; not a natural terminal')
        end=worker.current
        if end['invalid'] or not end['done'] or end['termination_reason']!=1:
            raise ValueError('duel did not naturally finish')
        reward=0. if end['winner']==2 else (1. if end['winner']==0 else -1.)
        if not training: raise ValueError('no learner decisions')
        training[-1]['reward']+=reward
        audit=worker.call('audit')
        (output/'audit.json').write_text(json.dumps(dict(initial=initial,prefix=prefix,**audit),ensure_ascii=False),encoding='utf-8')
        summary=dict(success=True,natural_terminal=True,winner=end['winner'],lp=end['state']['lp'],
                     turns=end['state']['turn'],model_decisions=len(prefix),learner_operations=len(training),
                     core_responses=len(audit['responses']),prompt_types=sorted(protocol_types),
                     probes=probes,search_information='oracle_exact_state' if probes else 'no_search',
                     terminal_reward=reward,final_fingerprint=end['fingerprint'],
                     reference_policy_used=False,opponent=f'same {type(policy).__name__} weights, separate visible observations')
        (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        # Terminal reward attachment is explicit; per-step JSONL records environment operations.
        log.write(json.dumps(dict(terminal=True,learner=0,reward=reward))+'\n'); log.flush()
        return dict(steps=training,total=sum(r['reward'] for r in training),success=reward>0),summary
    finally:
        log.close(); worker.close()
