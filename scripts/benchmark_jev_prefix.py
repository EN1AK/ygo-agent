"""Fixed real-input comparison of prefix recomputation and exact-token KV reuse."""
import argparse
import json
import time
from pathlib import Path


def main():
    import torch
    from ygoai.rl.jev_qwen_policy import OpenJevQwenPolicy
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--decisions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=48)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads(a.config.read_text()); torch.set_num_threads(2); torch.manual_seed(meta['seed'])
    policy=OpenJevQwenPolicy(meta['model_dir'],'cuda',meta['max_len'])
    policy.enable_prefix_kv('recompute')
    rows=[json.loads(s) for s in a.decisions.read_text().splitlines()][:a.count]
    encoded=[policy.encode(r['request']) for r in rows]
    # Same weights, tokens, attention kernel and split boundary in both arms.
    with torch.no_grad(): policy.distributions(encoded[:2],microbatch=1)
    results={}; reference=None; ref_prob=None
    for mode in ('recompute','reuse'):
        policy.enable_prefix_kv(mode)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter()
        with torch.no_grad():
            probabilities=[policy.distributions([e],microbatch=1)[0].probs.cpu() for e in encoded]
        torch.cuda.synchronize()
        forward=time.perf_counter()-begin; inf=dict(policy.prefix_kv.inference_stats)
        policy.model.zero_grad(set_to_none=True); begin=time.perf_counter()
        with policy.prefix_kv.training_scope():
            for i,e in enumerate(encoded):
                d=policy.distributions([e],microbatch=1)[0]
                (-d.log_prob(torch.tensor(rows[i]['selected'],device='cuda'))/len(rows)).backward()
                policy.prefix_kv.after_backward()
        torch.cuda.synchronize(); backward=time.perf_counter()-begin
        grads=torch.cat([v.grad.detach().float().flatten().cpu() for v in policy.model.parameters() if v.requires_grad])
        if not torch.isfinite(grads).all():raise ValueError('nonfinite gradient')
        if reference is None:reference=grads.clone();ref_prob=probabilities
        result=dict(forward_seconds=forward,forward_backward_seconds=backward,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            inference=inf,training=dict(policy.prefix_kv.training_stats),
            max_probability_difference=max(float((x-y).abs().max()) for x,y in zip(probabilities,ref_prob)),
            gradient_relative_l2=float((grads-reference).norm()/reference.norm()),
            gradient_cosine=float(torch.nn.functional.cosine_similarity(grads,reference,dim=0)),
            gradient_max_abs_difference=float((grads-reference).abs().max()))
        if result['max_probability_difference']!=0:raise ValueError('cached probabilities differ')
        results[mode]=result; print(json.dumps(dict(mode=mode,**result)),flush=True)
        torch.save(grads,a.output/(mode+'-gradients.pt'))
        (a.output/'summary.json').write_text(json.dumps(dict(count=len(rows),results=results),indent=2))


if __name__=='__main__':main()
