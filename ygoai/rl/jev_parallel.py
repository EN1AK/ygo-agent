"""Independent duel actors, one GPU batch service, and an on-policy batch tape."""
import concurrent.futures
import queue
import time

from ygoai.rl.jev_duel import play
from ygoai.rl.jev_experiment import leave_one_out_baselines,returns_to_go


class QueuedPolicy:
    name='OpenJevQwenPolicy'

    def __init__(self,policy,inbox,seed):
        self.policy=policy; self.inbox=inbox
        self.generator=policy.torch.Generator(device='cpu').manual_seed(seed)

    def choose(self,req,greedy):
        start=time.perf_counter()
        encoded=self.policy.encode(req)
        encoded_seconds=time.perf_counter()-start
        future=concurrent.futures.Future()
        self.inbox.put((req,encoded,greedy,self.generator,future,encoded_seconds))
        return future.result()


def collect(policy,configs,paths,*,actors=4,batch_size=4,batch_wait_ms=3.,seed=8102027,**play_options):
    """Freeze policy for the entire collection group. Only the service thread touches CUDA."""
    if actors<1 or batch_size<1: raise ValueError('invalid actor/batch capacity')
    inbox=queue.Queue(); tape=[]; sample_id=0
    metrics=dict(inference_seconds=0.,encode_cpu_seconds=0.,batch_sizes=[],prompts=0,tokens=0)
    started=time.perf_counter()
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=actors)
    tasks=[pool.submit(play,QueuedPolicy(policy,inbox,seed+i),cfg,path,**play_options)
           for i,(cfg,path) in enumerate(zip(configs,paths))]
    failure=None
    try:
        while not all(f.done() for f in tasks) or not inbox.empty():
            for f in tasks:
                if f.done() and f.exception(): raise f.exception()
            try: first=inbox.get(timeout=.05)
            except queue.Empty: continue
            pending=[first]; deadline=time.perf_counter()+batch_wait_ms/1000.
            while len(pending)<batch_size:
                try: pending.append(inbox.get(timeout=max(0,deadline-time.perf_counter())))
                except queue.Empty: break
            try:
                enc=[x[1] for x in pending]; begin=time.perf_counter()
                with policy.torch.no_grad():
                    distributions=policy.distributions(enc,microbatch=batch_size)
                    values=[(d.probs.cpu(),d.logits.cpu().tolist()) for d in distributions]
                metrics['inference_seconds']+=time.perf_counter()-begin
                rows=[]
                for item,(prob,logits) in zip(pending,values):
                    req,encoded,greedy,generator,future,encode_seconds=item
                    chosen=int(prob.argmax()) if greedy else int(policy.torch.multinomial(prob,1,generator=generator))
                    row=dict(request=req,encoded=encoded,selected=chosen,operation=req['operations'][chosen],
                             old_log_prob=logits[chosen],probabilities=prob.tolist(),logits=logits,reward=0.,sample_id=sample_id,
                             inference_batch_id=len(tape))
                    sample_id+=1; rows.append(row)
                    metrics['encode_cpu_seconds']+=encode_seconds
                    parts=encoded['leaves']+([encoded['root']] if encoded['root'] is not None else [])
                    metrics['prompts']+=len(parts); metrics['tokens']+=sum(len(x['ids']) for x in parts)
                tape.append(rows); metrics['batch_sizes'].append(len(rows))
                for item,row in zip(pending,rows): item[4].set_result(row)
            except BaseException as exc:
                for item in pending:
                    if not item[4].done(): item[4].set_exception(exc)
                raise
        results=[f.result() for f in tasks]
    except BaseException as exc:
        failure=exc
        # Drain requests until every running actor has unwound its native workers.
        for f in tasks: f.cancel()
        while not all(f.done() for f in tasks):
            try: item=inbox.get(timeout=.05); item[4].set_exception(exc)
            except queue.Empty: pass
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=bool(failure))
    metrics['elapsed_seconds']=time.perf_counter()-started
    metrics['decisions']=sample_id; metrics['decisions_per_second']=sample_id/metrics['elapsed_seconds']
    return results,tape,metrics


def train_tape(policy,optimizer,episodes,tape,*,microbatch=4,baseline_mode='zero',progress=None):
    """Replay the exact inference batches, including zero-loss opponent rows.

    Preserving padding, batch shape and hierarchy avoids silently loosening the
    BF16 on-policy probability check when switching to batched computation.
    """
    t=policy.torch
    if baseline_mode=='zero': baselines=[0.]*len(episodes)
    elif baseline_mode=='leave-one-out': baselines=leave_one_out_baselines([e['total'] for e in episodes])
    else: raise ValueError('unknown baseline')
    weights={}; old={}
    for e,baseline in zip(episodes,baselines):
        for row,future in zip(e['steps'],returns_to_go([r['reward'] for r in e['steps']])):
            key=row['sample_id']
            if key in weights: raise ValueError('duplicate learner operation')
            weights[key]=(future-baseline)/len(episodes); old[key]=row['old_log_prob']
    optimizer.zero_grad(set_to_none=True); start=time.perf_counter(); drift=0.; loss_total=0.; seen=set(); used=0
    for rows in tape:
        selected=[i for i,r in enumerate(rows) if r['sample_id'] in weights]
        if not selected: continue
        used+=1
        ds=policy.distributions([r['encoded'] for r in rows],microbatch=microbatch)
        terms=[]
        for i in selected:
            row=rows[i]; key=row['sample_id']; seen.add(key)
            lp=ds[i].log_prob(t.tensor(row['selected'],device=policy.device))
            delta=abs(float(lp.detach())-old[key]); drift=max(drift,delta)
            if delta>1e-4: raise ValueError('on-policy batch probability drift '+str(delta))
            terms.append(-lp*weights[key])
        loss=t.stack(terms).sum()
        if not t.isfinite(loss): raise ValueError('nonfinite batched loss')
        loss.backward(); loss_total+=float(loss.detach())
        if progress is not None and used%100==0:
            progress(dict(backward_batches=used,learner_operations_seen=len(seen),learner_operations_total=len(weights),
                          elapsed_seconds=time.perf_counter()-start,max_log_prob_drift=drift))
    if seen!=set(weights): raise ValueError('learner operation missing from batch tape')
    parameters=[p for p in policy.model.parameters() if p.requires_grad]
    norm=t.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    tracked=next(((n,p) for n,p in policy.model.named_parameters() if p.grad is not None and bool(t.count_nonzero(p.grad))),None)
    before=tracked[1].detach().clone() if tracked else None
    optimizer.step()
    if str(policy.device).startswith('cuda'): t.cuda.synchronize()
    return dict(loss=loss_total,grad_norm=float(norm),max_log_prob_drift=drift,baseline_mode=baseline_mode,
                tracked_parameter=tracked[0] if tracked else None,
                parameter_max_abs_delta=float((tracked[1].detach()-before).abs().max()) if tracked else 0.,
                elapsed_seconds=time.perf_counter()-start,learner_operations=len(weights),backward_batches=used)
