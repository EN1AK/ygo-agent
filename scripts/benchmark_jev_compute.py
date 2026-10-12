"""Same recorded inputs, frozen weights: measure Qwen batching and checkpoint recomputation."""
import argparse
import gc
import json
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--decisions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=12)
    a=p.parse_args()
    if a.samples<2: p.error('samples must be at least 2')
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from ygoai.rl.jev_qwen_policy import OpenJevQwenPolicy
    torch.set_num_threads(2); torch.manual_seed(8102027)
    rows=[json.loads(x) for x in a.decisions.read_text().splitlines()]
    rows=[r for r in rows if 'encoded' in r]
    rows=[rows[round(i*(len(rows)-1)/(a.samples-1))] for i in range(a.samples)]
    policy=OpenJevQwenPolicy(a.model_dir,'cuda')
    encoded=[r['encoded'] for r in rows]
    with torch.no_grad(): reference=[d.probs.cpu() for e in encoded for d in [policy.distribution(e)]]
    results=[]
    for checkpoint in (True,False):
        if checkpoint: policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else: policy.model.gradient_checkpointing_disable()
        for batch in (1,2,4):
            try:
                policy.model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
                with torch.no_grad(): policy.distributions(encoded[:batch],microbatch=batch)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter(); old=[]; discrepancy=0.
                with torch.no_grad():
                    for i in range(0,len(rows),batch):
                        ds=policy.distributions(encoded[i:i+batch],microbatch=batch)
                        for j,d in enumerate(ds):
                            old.append(float(d.log_prob(torch.tensor(rows[i+j]['selected'],device='cuda'))))
                            discrepancy=max(discrepancy,float((d.probs.cpu()-reference[i+j]).abs().max()))
                torch.cuda.synchronize(); forward=time.perf_counter()-start
                start=time.perf_counter(); drift=0.
                for i in range(0,len(rows),batch):
                    ds=policy.distributions(encoded[i:i+batch],microbatch=batch)
                    lps=[d.log_prob(torch.tensor(rows[i+j]['selected'],device='cuda')) for j,d in enumerate(ds)]
                    drift=max(drift,max(abs(float(lp.detach())-old[i+j]) for j,lp in enumerate(lps)))
                    (-torch.stack(lps).sum()/len(rows)).backward()
                torch.cuda.synchronize(); backward=time.perf_counter()-start
                result=dict(checkpoint=checkpoint,batch=batch,forward_seconds=forward,forward_backward_seconds=backward,
                            inference_decisions_per_second=len(rows)/forward,training_decisions_per_second=len(rows)/backward,
                            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                            same_batch_logprob_drift=drift,scalar_probability_max_delta=discrepancy)
                if drift>1e-4: raise ValueError('same-batch on-policy numerical drift '+str(drift))
                if discrepancy>1e-4: raise ValueError('batched probability differs from scalar reference '+str(discrepancy))
            except torch.cuda.OutOfMemoryError:
                result=dict(checkpoint=checkpoint,batch=batch,error='CUDA out of memory')
                policy.model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
            results.append(result); print(json.dumps(result),flush=True)
            (a.output/'summary.json').write_text(json.dumps(dict(results=results,samples=len(rows),device=torch.cuda.get_device_name(),
                lengths=[max(len(x['ids']) for x in e['leaves']) for e in encoded]),indent=2))


if __name__=='__main__': main()
