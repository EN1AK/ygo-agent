"""Exact-token prefix reuse, including the prefix's contribution to LoRA gradients."""
from collections import OrderedDict
from contextlib import contextmanager


def install_prefix_attention():
    import torch
    from torch.nn.attention.bias import causal_lower_right
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    def attention(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        if query.shape[0]!=1 or attention_mask is not None or dropout or kwargs.get('sliding_window'):
            raise ValueError('prefix attention requires one unpadded, dropout-free, full-attention sequence')
        q,k=query.shape[-2],key.shape[-2]
        if q>k: raise ValueError('invalid prefix attention positions')
        # Rectangular causal attention must align at the RIGHT, not at position zero.
        out=torch.nn.functional.scaled_dot_product_attention(query,key,value,
            attn_mask=causal_lower_right(q,k) if q!=k else None,is_causal=q==k,
            dropout_p=0.,scale=scaling,enable_gqa=query.shape[1]!=key.shape[1])
        return out.transpose(1,2).contiguous(),None

    ALL_ATTENTION_FUNCTIONS.register('jev_prefix_sdpa',attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register('jev_prefix_sdpa',lambda **kwargs:None)


class PrefixKV:
    def __init__(self,policy,*,reuse=True,max_entries=8,max_tokens=32768):
        if max_entries<1 or max_tokens<1: raise ValueError('invalid prefix cache capacity')
        install_prefix_attention()
        self.policy=policy;self.t=policy.torch;self.reuse=reuse
        self.max_entries=max_entries;self.max_tokens=max_tokens
        self.inference=OrderedDict();self.training=OrderedDict();self.active=False
        self.inference_stats=self.stats();self.training_stats=self.stats()
        self.versions=self.parameter_versions()
        policy.model.gradient_checkpointing_disable()
        policy.model.config._attn_implementation='jev_prefix_sdpa'

    @staticmethod
    def stats():
        return dict(hits=0,misses=0,reused_prefix_tokens=0,prefix_forward_tokens=0,
                    suffix_tokens=0,prefix_backward_tokens=0,prefix_vjps=0,invalidations=0,
                    peak_resident_tokens=0)

    def parameter_versions(self):
        return tuple(p._version for p in self.policy.model.parameters() if p.requires_grad)

    def check_version(self):
        now=self.parameter_versions()
        if now!=self.versions:
            if self.active: raise RuntimeError('parameters changed before prefix gradients were flushed')
            self.inference.clear();self.inference_stats['invalidations']+=1;self.versions=now

    def reset_inference(self):
        if self.active: raise RuntimeError('cannot reset during a prefix gradient transaction')
        self.inference.clear();self.inference_stats=self.stats();self.versions=self.parameter_versions()

    def prefill(self,ids):
        from transformers.cache_utils import DynamicCache
        x=self.t.tensor([ids],device=self.policy.device)
        out=self.policy.model.model(input_ids=x,past_key_values=DynamicCache(),use_cache=True,
                                    cache_position=self.t.arange(len(ids),device=self.policy.device))
        return out.past_key_values.to_legacy_cache()

    def flush(self,key,entry):
        if entry['pending']: raise RuntimeError('backward must finish before prefix eviction')
        gradients=entry['grads']
        if not any(g is not None for g in gradients): return
        # Rebuild the prefix graph once and apply all suffix cotangents. Detaching
        # rollout KV without this VJP would silently omit prefix parameter gradients.
        with self.t.enable_grad():
            outputs=[x for pair in self.prefill(key) for x in pair]
            pairs=[(x,g) for x,g in zip(outputs,gradients) if x.requires_grad and g is not None]
            if pairs:self.t.autograd.backward([x for x,g in pairs],[g for x,g in pairs])
        self.training_stats['prefix_backward_tokens']+=len(key)
        self.training_stats['prefix_vjps']+=1

    def acquire(self,key):
        grad=self.t.is_grad_enabled();train=grad and self.active
        stats=self.training_stats if grad else self.inference_stats
        cache=self.training if train else self.inference
        enabled=self.reuse and (train or not grad) and len(key)<=self.max_tokens
        if enabled and key in cache:
            entry=cache.pop(key);cache[key]=entry
            stats['hits']+=1;stats['reused_prefix_tokens']+=len(key)
            if train:entry['pending']=True
            return entry['kv']
        stats['misses']+=1;stats['prefix_forward_tokens']+=len(key)
        if not enabled:return self.prefill(key)
        while cache and (len(cache)>=self.max_entries or sum(map(len,cache))+len(key)>self.max_tokens):
            old,entry=cache.popitem(last=False)
            if train:self.flush(old,entry)
        with self.t.no_grad():kv=self.prefill(key)
        kv=tuple(tuple(x.detach().requires_grad_(train) for x in pair) for pair in kv)
        cache[key]=dict(kv=kv,pending=train,grads=[None for pair in kv for x in pair])
        stats['peak_resident_tokens']=max(stats['peak_resident_tokens'],sum(map(len,cache)))
        return kv

    def scores(self,prompt):
        from transformers.cache_utils import DynamicCache
        self.check_version()
        n=prompt['prefix_len'];ids=prompt['ids']
        if not 0<n<len(ids):raise ValueError('prefix must leave a nonempty decision suffix')
        kv=self.acquire(tuple(ids[:n]))
        stats=self.training_stats if self.t.is_grad_enabled() else self.inference_stats
        stats['suffix_tokens']+=len(ids)-n
        # A fresh wrapper prevents the suffix (including hypothetical branches)
        # from mutating a shared prefix. Tensor storage is shared read-only.
        cache=DynamicCache.from_legacy_cache(kv)
        out=self.policy.model(input_ids=self.t.tensor([ids[n:]],device=self.policy.device),
            past_key_values=cache,cache_position=self.t.arange(n,len(ids),device=self.policy.device),
            use_cache=True,logits_to_keep=1)
        scores=out.logits[0,-1,self.policy.label_ids[:prompt['count']]].float()
        if not self.t.isfinite(scores).all():raise ValueError('nonfinite prefix option logits')
        return scores

    def after_backward(self):
        for entry in self.training.values():
            for i,p in enumerate(x for pair in entry['kv'] for x in pair):
                if p.grad is not None:
                    if entry['grads'][i] is None:entry['grads'][i]=p.grad.detach().float().clone()
                    else:entry['grads'][i].add_(p.grad.detach().float())
                    p.grad=None
            entry['pending']=False

    @contextmanager
    def training_scope(self):
        if self.active:raise RuntimeError('nested prefix gradient transaction')
        self.check_version();self.inference.clear();self.training.clear()
        self.training_stats=self.stats();self.active=True
        try:
            yield
            self.check_version()
            for key,entry in self.training.items():self.flush(key,entry)
        finally:
            self.training.clear();self.active=False
