import unittest

try:
    import torch
except ImportError:
    torch=None

from ygoai.rl.jev_qwen_policy import add_lora, OpenJevQwenPolicy


@unittest.skipIf(torch is None,'requires isolated torch environment')
class QwenReadoutTests(unittest.TestCase):
    def test_lora_changes_model_choice_without_updating_base_weight(self):
        from scripts.train_jev_direct_rl import train_group
        class Policy:
            device='cpu'
            def __init__(self):
                self.torch=torch
                self.model=torch.nn.Module()
                self.model.layer=torch.nn.Module()
                self.model.layer.q_proj=torch.nn.Linear(4,2,bias=False)
                torch.nn.init.zeros_(self.model.layer.q_proj.weight)
                add_lora(self.model,torch,rank=2)
            def distribution(self,encoded):
                return torch.distributions.Categorical(logits=self.model.layer.q_proj(torch.ones(4)))
        policy=Policy(); base=policy.model.layer.q_proj.base.weight.detach().clone()
        episodes=[]
        for action,reward in ((0,1.),(1,-1.)):
            lp=float(policy.distribution({}).log_prob(torch.tensor(action)).detach())
            episodes.append(dict(total=reward,steps=[dict(encoded={},selected=action,reward=reward,old_log_prob=lp)]))
        opt=torch.optim.SGD([p for p in policy.model.parameters() if p.requires_grad],lr=.1)
        stat=train_group(policy,opt,episodes)
        self.assertGreater(stat['parameter_max_abs_delta'],0.)
        self.assertGreater(float(policy.distribution({}).probs[0]),.5)
        self.assertTrue(torch.equal(base,policy.model.layer.q_proj.base.weight))

    def test_hierarchical_distribution_keeps_every_action_and_gradients(self):
        policy=object.__new__(OpenJevQwenPolicy); policy.torch=torch
        root=torch.tensor([.2,-.3],requires_grad=True)
        a=torch.tensor([.1,.5],requires_grad=True); b=torch.tensor([-.2],requires_grad=True)
        policy._scores=lambda e:e
        d=policy.distribution(dict(root=root,leaves=[a,b]))
        self.assertEqual(len(d.probs),3)
        self.assertAlmostEqual(float(d.probs.sum()),1.,places=6)
        d.log_prob(torch.tensor(0)).backward()
        self.assertGreater(float(root.grad.abs().sum()),0.)
        self.assertGreater(float(a.grad.abs().sum()),0.)


if __name__=='__main__': unittest.main()
