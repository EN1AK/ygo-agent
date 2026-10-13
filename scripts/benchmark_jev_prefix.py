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
    a=p.parse_args()
    if a.count<1:p.error('count must be positive')
    a.output.mkdir(parents=True,exist_ok=False)
    from scripts.train_jev_direct_rl import sha
    provenance=dict(config_sha256=sha(a.config),decisions_sha256=sha(a.decisions),
        source_sha256={f:sha(f) for f in ('scripts/benchmark_jev_prefix.py','ygoai/rl/jev_prefix_kv.py',
                                         'ygoai/rl/jev_qwen_policy.py')})
    meta=json.loads(a.config.read_text()); torch.set_num_threads(2); torch.manual_seed(meta['seed'])
    policy=OpenJevQwenPolicy(meta['model_dir'],'cuda',meta['max_len'])
    policy.enable_prefix_kv('recompute')
    rows=[json.loads(s) for s in a.decisions.read_text().splitlines()][:a.count]
    encoded=[policy.encode(r['request']) for r in rows]
    # Same weights, tokens, attention kernel and split boundary in both arms.
    with torch.no_grad(): policy.distributions(encoded[:2],microbatch=1)
    results={}; reference=None; ref_prob=None; whole_prob=None
    for mode in ('singlepass','recompute','reuse'):
        policy.enable_prefix_kv('recompute' if mode=='singlepass' else mode)
        inputs=encoded
        if mode=='singlepass':
            def whole(prompt):
                return None if prompt is None else {k:v for k,v in prompt.items() if k!='prefix_len'}
            inputs=[dict(e,root=whole(e['root']),leaves=[whole(x) for x in e['leaves']]) for e in encoded]
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter()
        with torch.no_grad():
            probabilities=[policy.distributions([e],microbatch=1)[0].probs.cpu() for e in inputs]
        torch.cuda.synchronize()
        forward=time.perf_counter()-begin; inf=dict(policy.prefix_kv.inference_stats)
        policy.model.zero_grad(set_to_none=True); begin=time.perf_counter()
        with policy.prefix_kv.training_scope():
            for i,e in enumerate(inputs):
                d=policy.distributions([e],microbatch=1)[0]
                (-d.log_prob(torch.tensor(rows[i]['selected'],device='cuda'))/len(rows)).backward()
                policy.prefix_kv.after_backward()
        torch.cuda.synchronize(); backward=time.perf_counter()-begin
        grads=torch.cat([v.grad.detach().float().flatten().cpu() for v in policy.model.parameters() if v.requires_grad])
        if not torch.isfinite(grads).all():raise ValueError('nonfinite gradient')
        if mode=='singlepass':whole_prob=probabilities
        if mode!='reuse':reference=grads.clone();ref_prob=probabilities
        # Double precision avoids long FP32 reduction error (cosine > 1).
        g64=grads.double(); r64=reference.double()
        result=dict(forward_seconds=forward,forward_backward_seconds=backward,
            gradient_reference='recompute' if mode=='reuse' else 'self',
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            inference=inf,training=dict(policy.prefix_kv.training_stats),
            max_probability_difference=max(float((x-y).abs().max()) for x,y in zip(probabilities,ref_prob)),
            max_probability_difference_from_singlepass=max(float((x-y).abs().max()) for x,y in zip(probabilities,whole_prob)),
            gradient_relative_l2=float((g64-r64).norm()/r64.norm()),
            gradient_cosine=float(torch.nn.functional.cosine_similarity(g64,r64,dim=0)),
            gradient_max_abs_difference=float((grads-reference).abs().max()))
        if result['max_probability_difference']!=0:raise ValueError('cached probabilities differ')
        results[mode]=result; print(json.dumps(dict(mode=mode,**result)),flush=True)
        torch.save(grads,a.output/(mode+'-gradients.pt'))
        (a.output/'summary.json').write_text(json.dumps(dict(count=len(rows),provenance=provenance,results=results),indent=2))


if __name__=='__main__':main()
