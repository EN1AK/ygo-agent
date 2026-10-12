import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
except ImportError:
    torch=None

from ygoai.rl.jev_parallel import collect,train_tape
from ygoai.rl.jev_qwen_policy import OpenJevQwenPolicy


@unittest.skipIf(torch is None,'requires isolated torch/transformers environment')
class ParallelPolicyTests(unittest.TestCase):
    def test_real_qwen_causal_right_padding_and_hierarchy_preserve_probabilities(self):
        torch.manual_seed(7)
        policy=object.__new__(OpenJevQwenPolicy)
        policy.torch=torch; policy.device='cpu'; policy.label_ids=list(range(4,30))
        policy.tok=SimpleNamespace(pad_token_id=0,eos_token_id=2)
        policy.model=Qwen3ForCausalLM(Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.)).eval()
        a=dict(ids=[11,12,13],count=3); b=dict(ids=[14,15,16,17,18],count=2)
        inputs=[dict(root=None,leaves=[a]),dict(root=b,leaves=[a,b])]
        reference=[policy.distribution(e) for e in inputs]
        batched=policy.distributions(inputs,microbatch=3)
        for old,new in zip(reference,batched):
            torch.testing.assert_close(old.probs,new.probs,atol=1e-6,rtol=1e-6)
        sum(-d.log_prob(torch.tensor(0)) for d in batched).backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in policy.model.parameters()))

    def make_policy(self):
        class Policy:
            device='cpu'
            def __init__(self):
                self.torch=torch; self.model=torch.nn.Linear(2,3,bias=False)
            def distribution(self,e): return torch.distributions.Categorical(logits=self.model(torch.tensor(e['x'] if isinstance(e,dict) else e)))
            def distributions(self,es,microbatch=4): return [self.distribution(e) for e in es]
            def encode(self,req): return dict(x=req['x'],leaves=[dict(ids=[1,2],count=3)],root=None)
        return Policy()

    def test_exact_batch_training_matches_scalar_gradient_and_ignores_opponent(self):
        from scripts.train_jev_direct_rl import train_group
        torch.manual_seed(3); policy=self.make_policy(); other=self.make_policy()
        other.model.load_state_dict(policy.model.state_dict())
        rows=[]
        for i,(encoded,action,reward) in enumerate([([1.,2.],0,-.02),([2.,1.],2,0.),([1.,1.],1,1.),([0.,1.],2,-1.)]):
            rows.append(dict(sample_id=i,encoded=encoded,selected=action,reward=reward,
                             old_log_prob=float(policy.distribution(encoded).log_prob(torch.tensor(action)).detach())))
        episodes=[dict(total=.98,steps=[rows[0],rows[2]]),dict(total=-1.,steps=[rows[3]])]
        scalar=train_group(other,torch.optim.SGD(other.model.parameters(),lr=.01),episodes,baseline_mode='zero')
        batch=train_tape(policy,torch.optim.SGD(policy.model.parameters(),lr=.01),episodes,[rows[:2],rows[2:]],baseline_mode='zero')
        torch.testing.assert_close(policy.model.weight,other.model.weight)
        self.assertAlmostEqual(scalar['loss'],batch['loss'],places=5)
        self.assertEqual(batch['learner_operations'],3)

    def test_actor_sampling_does_not_depend_on_batch_arrival_order(self):
        policy=self.make_policy()
        def play_stub(proxy,cfg,path,**kwargs):
            return [proxy.choose({'x':[1.,2.],'operations':[('commit',i) for i in range(3)]},False)['selected'] for _ in range(4)]
        with patch('ygoai.rl.jev_parallel.play',play_stub):
            a,_,_=collect(policy,[{}]*3,['a','b','c'],actors=1,batch_size=1,seed=9)
            b,_,_=collect(policy,[{}]*3,['a','b','c'],actors=3,batch_size=3,seed=9)
        self.assertEqual(a,b)

    def test_actor_failure_unwinds_pending_policy_requests(self):
        policy=self.make_policy()
        def play_stub(proxy,cfg,path,**kwargs):
            if path=='bad': raise RuntimeError('native failure')
            return proxy.choose({'x':[1.,2.],'operations':[('commit',i) for i in range(3)]},False)
        with patch('ygoai.rl.jev_parallel.play',play_stub):
            with self.assertRaisesRegex(RuntimeError,'native failure'):
                collect(policy,[{}]*3,['ok','bad','queued'],actors=2,batch_size=2)


if __name__=='__main__': unittest.main()
