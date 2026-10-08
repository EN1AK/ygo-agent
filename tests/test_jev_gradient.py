"""Small numerical checks; these do not substitute for the pretrained-model pilot."""
import unittest

try:
    import torch
except ImportError:
    torch = None

from scripts.train_jev_direct_rl import train_group


@unittest.skipIf(torch is None, 'requires experimental PyTorch environment')
class DirectGradientTests(unittest.TestCase):
    def policy(self):
        class Policy:
            device = 'cpu'

            def __init__(self):
                self.torch = torch
                self.model = torch.nn.Linear(1, 2, bias=False)
                torch.nn.init.zeros_(self.model.weight)

            def distribution(self, encoded):
                logits = self.model(torch.ones(1))[...]
                return torch.distributions.Categorical(logits=logits)
        return Policy()

    def episodes(self, policy):
        episodes = []
        for chosen, reward in ((0, 1.), (1, -1.)):
            lp = float(policy.distribution({}).log_prob(torch.tensor(chosen)).detach())
            episodes.append(dict(total=reward, steps=[dict(encoded={}, selected=chosen,
                                old_log_prob=lp, reward=reward)]))
        return episodes

    def test_real_parameter_update_favors_rewarded_choice(self):
        policy = self.policy()
        optimizer = torch.optim.SGD(policy.model.parameters(), lr=.1)
        stats = train_group(policy, optimizer, self.episodes(policy))
        self.assertGreater(float(policy.distribution({}).probs[0]), .5)
        self.assertGreater(stats['grad_norm'], 0.)

    def test_stale_rollout_probabilities_are_rejected(self):
        policy = self.policy()
        episodes = self.episodes(policy)
        episodes[0]['steps'][0]['old_log_prob'] += .1
        optimizer = torch.optim.SGD(policy.model.parameters(), lr=.1)
        with self.assertRaises(ValueError):
            train_group(policy, optimizer, episodes)


if __name__ == '__main__':
    unittest.main()
