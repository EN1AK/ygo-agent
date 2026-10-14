"""Bounded route discovery on exact replay histories; not an online search policy."""
import copy
import heapq
import math
import time

import numpy as np
from ygoai.rl.elfnote_combo import grade
from ygoai.rl.exercise_actor import array_digest, extract_trace, hidden_identity_audit
from ygoai.rl.observation_schema import tensor_contract


def probability(logits, temperature=1.):
    if temperature <= 0 or not np.isfinite(temperature): raise ValueError("invalid temperature")
    x = np.asarray(logits, np.float64) / temperature
    if not x.size or not np.isfinite(x).all(): raise ValueError("invalid logits")
    x = np.exp(x-x.max())
    return x/x.sum()


def make_cases(plan, sources):
    result = []
    for split in ('train', 'reserved'):
        for pair in plan['pairs_'+split]:
            source = next(c for c in sources if c['name'] == pair)
            for response, card in plan['responses'].items():
                case = copy.deepcopy(source)
                case.update(name=pair+'-'+response, split=split, intervention=card)
                d = case['definition']
                if split == 'reserved':
                    rng = np.random.default_rng(plan['seed']+len(result))
                    d['seed_sequence'] = rng.integers(0, 2**32, 8).tolist()
                    deck = [x for x in d['initial'] if x[1:3] == [0,1]]
                    other = next(x for x in deck[:-5] if x[0] == 14558128 and x[0] != deck[-3][0])
                    other[0], deck[-3][0] = deck[-3][0], other[0]
                if card:
                    deck = [x for x in d['initial'] if x[1:3] == [1,1]]
                    deck[-1][0] = card
                    for goal in case['goals']: goal['hand'] -= 1
                result.append(case)
    return result


def run(native, semantics, actor, case, plan, *, prefix=(), sample_seed=None, baseline=None, detailed=True):
    d = case['definition']
    env = native.ExerciseActor('', d['initial'], str(semantics), 1, 'structured-lite-v1',
                              d['seed_sequence'], d['duel_flags'], plan['max_decisions']+2)
    actor.reset()
    rng = np.random.default_rng(sample_seed)
    decisions, arrays = [], []
    attempted = False
    root = 0 if not case['intervention'] else None
    own = 0
    record = dict(schema='ygo-elfnote-search-route-v1', definition=d, case=case['name'], training_ready=False)
    try:
        env.reset()
        for step in range(plan['max_decisions']):
            snap = env.snapshot()
            events, state, responses = extract_trace(env, snap)
            outcomes = [grade(state, goal) for goal in case['goals']]
            ended = sum(e['op']==40 for e in events) >= 2
            if root is None and attempted and snap['player']==0 and snap['message']==11:
                root = step
            stable = ended or snap['message']==11
            if stable and any(o['success'] for o in outcomes):
                record.update(status='success', outcomes=outcomes); break
            if snap['invalid']: raise ValueError('invalid engine trajectory')
            if ended or snap['done']:
                record.update(status='goal_not_reached', outcomes=outcomes); break
            if own >= plan['max_own_decisions']:
                record.update(status='budget_exhausted', outcomes=outcomes); break
            obs = {k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
            digest = array_digest(obs)
            if step==0: hidden_identity_audit(obs)
            logits, value, rnn = actor.predict(obs, snap['player'])
            n = len(snap['menu']); probs = probability(logits[:n], plan['temperature'])
            if step < len(prefix):
                expected = prefix[step]
                if (digest, snap['menu'], snap['player'], snap['message']) != (expected['observation_sha256'], expected['menu'], expected['player'], expected['message']):
                    raise ValueError('prefix restore mismatch')
                action = expected['action']
            elif baseline is not None and step < baseline['root_step']:
                expected = baseline['decisions'][step]
                assert (digest,snap['menu'],snap['player'],snap['message']) == (expected['observation_sha256'],expected['menu'],expected['player'],expected['message'])
                action = expected['action']
            elif snap['player']==0:
                action = int(rng.choice(n,p=probs)) if sample_seed is not None and root is not None else int(np.argmax(logits[:n]))
            else:
                choices = [i for i,m in enumerate(snap['menu']) if not attempted and case['intervention']
                           and snap['message']==16 and m['code']==case['intervention'] and m['act']=='Activate']
                cancel = [i for i,m in enumerate(snap['menu']) if m['act']=='Cancel']
                action = choices[0] if choices else (cancel[0] if cancel else 0)
            if not 0 <= action < n: raise ValueError('prefix action outside legal menu')
            selected = snap['menu'][action]
            if snap['player']==1 and snap['message']==16 and selected['act']=='Activate' and selected['code']==case['intervention']:
                attempted = True
            row = dict(step=step, player=snap['player'], message=snap['message'], menu=snap['menu'], action=action,
                       observation_sha256=digest, logits=logits[:n].tolist(), value=value, rnn_sha256=rnn,
                       core_response_count=len(responses), teaching=snap['player']==0 and root is not None and step>=root)
            decisions.append(row)
            if detailed: arrays.append({k:v[0].copy() for k,v in obs.items()})
            own += snap['player']==0
            env.step(action)
        else: record.update(status='budget_exhausted',outcomes=outcomes)
    finally:
        snap=env.snapshot();events,state,responses=extract_trace(env,snap)
        record.update(decisions=decisions, events=events, final=state, core_responses=responses,
                      root_step=root, intervention_attempted=attempted, core_seed=env.trace()['core_seed'])
        env.close()
    if case['intervention'] and root is None: record['status']='intervention_not_observed'
    return record, {k:np.stack([x[k] for x in arrays]) for k in arrays[0]} if arrays else {}


def discover(native, semantics, actor, case, plan, baseline, mode, callback=None):
    start=time.monotonic(); trials=[]; serial=0; queue=[]; seen=set(); visited=0
    heapq.heappush(queue,(0.,serial,[],0.))
    best=None
    for trial in range(plan['trials_per_case']):
        if time.monotonic()-start >= plan['seconds_per_case']: break
        if mode=='search':
            if not queue: break
            _,_,prefix,cost=heapq.heappop(queue)
            record,_=run(native,semantics,actor,case,plan,prefix=prefix,baseline=baseline,detailed=False)
        else:
            record,_=run(native,semantics,actor,case,plan,sample_seed=None if trial==0 else plan['seed']+trial,baseline=baseline,detailed=False)
        visited += len(record['decisions'])
        trials.append(dict(trial=trial,status=record['status'],decisions=len(record['decisions']),seconds=time.monotonic()-start))
        if callback: callback(trials[-1])
        if record['status']=='success':
            best=record;break
        if record['status']=='intervention_not_observed': break
        if mode=='search':
            for i,row in enumerate(record['decisions']):
                if i < max(len(prefix),baseline['root_step']) or row['player']!=0: continue
                probs=probability(row['logits'],plan['temperature'])
                choices=sorted(range(len(probs)),key=lambda j:(-probs[j],j))
                signatures=set(); offered=0
                for action in choices:
                    if action==row['action']: continue
                    m=row['menu'][action]
                    # Only identical deck-card aliases are deduplicated; field/grave/hand identity is retained.
                    signature=tuple((k,v) for k,v in sorted(m.items()) if k!='spec' or not str(v).isdigit())
                    if signature in signatures: continue
                    signatures.add(signature)
                    path=record['decisions'][:i]+[dict(row,action=action)]
                    key=tuple(d['action'] for d in path)
                    if key in seen: continue
                    seen.add(key);serial+=1;offered+=1
                    penalty=-math.log(max(float(probs[action]),1e-12))
                    heapq.heappush(queue,(cost+penalty,serial,path,cost+penalty))
                    if offered>=plan['search_branches']: break
            if len(queue)>plan['search_queue_cap']:
                queue=heapq.nsmallest(plan['search_queue_cap'],queue);heapq.heapify(queue)
    return best,dict(mode=mode,case=case['name'],trials=trials,decisions=visited,seconds=time.monotonic()-start,
                     status='success' if best else 'budget_exhausted_or_no_witness',pending=len(queue))
