import unittest

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    torch = None

from ygoai.rl.jev_qwen_policy import OpenJevQwenPolicy, add_lora


@unittest.skipIf(torch is None, 'requires isolated torch/transformers environment')
class PrefixKVTests(unittest.TestCase):
    def policy(self, reuse=True, entries=8):
        torch.manual_seed(17)
        p = object.__new__(OpenJevQwenPolicy)
        p.torch = torch; p.device = 'cpu'; p.label_ids = [1, 2, 3]
        p.model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32,
            intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.))
        add_lora(p.model, torch, rank=2)
        # Exercise both A and B gradients, including a non-initial adapter.
        with torch.no_grad():
            for n, v in p.model.named_parameters():
                if n.endswith('lora_b'): v.normal_(std=.03)
        p.enable_prefix_kv('reuse' if reuse else 'recompute', entries, 256)
        return p

    def prompts(self):
        return [dict(ids=[4,5,6,7,8,9], prefix_len=4, count=3),
                dict(ids=[4,5,6,7,12,13,14], prefix_len=4, count=3),
                dict(ids=[15,16,17,18,19], prefix_len=3, count=3),
                dict(ids=[4,5,6,7,8,9], prefix_len=4, count=3)]

    def test_causality_branch_isolation_and_reuse(self):
        p = self.policy()
        with torch.no_grad():
            actual = [p._scores(x) for x in self.prompts()]
            expected = [p.model(input_ids=torch.tensor([x['ids']]), use_cache=False,
                logits_to_keep=1).logits[0,-1,p.label_ids] for x in self.prompts()]
        for a,b in zip(actual,expected): torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(actual[0],actual[-1],atol=0,rtol=0)
        self.assertEqual(p.prefix_kv.inference_stats['hits'],2)

    def test_full_parameter_gradients_and_eviction(self):
        for entries in (1,8):
            cached = self.policy(entries=entries); reference = self.policy(False)
            for p in (cached,reference):
                with p.prefix_kv.training_scope():
                    for i,x in enumerate(self.prompts()):
                        loss = -p._scores(x).log_softmax(-1)[i%3]*(i+1)/4
                        loss.backward(); p.prefix_kv.after_backward()
            for (n,a),(m,b) in zip(cached.model.named_parameters(),reference.model.named_parameters()):
                if a.requires_grad:
                    self.assertIsNotNone(a.grad,n)
                    torch.testing.assert_close(a.grad,b.grad,atol=2e-7,rtol=2e-4,msg=n)
            self.assertEqual(cached.prefix_kv.training_stats['prefix_vjps'],3 if entries==1 else 2)

    def test_parameter_update_invalidates_cache(self):
        p = self.policy(); x = self.prompts()[0]
        with torch.no_grad():
            p._scores(x); p._scores(x)
            next(v for v in p.model.parameters() if v.requires_grad).add_(.01)
            p._scores(x)
        self.assertEqual(p.prefix_kv.inference_stats['invalidations'],1)
        self.assertEqual(p.prefix_kv.inference_stats['misses'],2)

    def test_hierarchical_choices_share_prefix_before_one_backward(self):
        cached=self.policy(); reference=self.policy(False)
        a,b=self.prompts()[:2]
        encoded=dict(root=dict(a,count=2),leaves=[a,b],count=6)
        for p in (cached,reference):
            with p.prefix_kv.training_scope():
                d=p.distributions([encoded],microbatch=1)[0]
                (-d.log_prob(torch.tensor(4))).backward()
                p.prefix_kv.after_backward()
        for (n,a),(m,b) in zip(cached.model.named_parameters(),reference.model.named_parameters()):
            if a.requires_grad:torch.testing.assert_close(a.grad,b.grad,atol=2e-7,rtol=2e-4,msg=n)
        self.assertEqual(cached.prefix_kv.training_stats['prefix_vjps'],1)

    def test_unfinished_backward_cannot_be_evicted(self):
        p = self.policy(entries=1)
        with self.assertRaisesRegex(RuntimeError,'backward must finish'):
            with p.prefix_kv.training_scope():
                p._scores(self.prompts()[0]); p._scores(self.prompts()[2])

    def test_parameter_mutation_before_prefix_backward_is_rejected(self):
        p=self.policy()
        with self.assertRaisesRegex(RuntimeError,'parameters changed'):
            with p.prefix_kv.training_scope():
                p._scores(self.prompts()[0]).sum().backward(); p.prefix_kv.after_backward()
                with torch.no_grad():
                    next(v for v in p.model.parameters() if v.requires_grad).add_(.01)


if __name__ == '__main__': unittest.main()
