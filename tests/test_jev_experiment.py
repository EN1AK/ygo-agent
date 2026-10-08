import unittest

from ygoai.rl.jev_experiment import BranchEnvironment, visible_state, returns_to_go, leave_one_out_baselines


class ExperimentTests(unittest.TestCase):
    def environment(self):
        def runner(case, branch, root_only):
            return dict(root_digest='same', runtime={'core': 'fixed'}, visible_root={}, card_texts=[],
                        status='success' if branch == 'veiler' else 'verified_failure',
                        visible_final={'cards': [branch]}, teacher_secret='not for model')
        env = BranchEnvironment('', '', '', runner=runner)
        env.reset('interaction')
        return env

    def test_probe_changes_context_without_committing_or_leaking_reward(self):
        import random
        env = self.environment()
        self.assertEqual(env.step(('probe', 'ogre')), -.02)
        self.assertFalse(env.done)
        r = env.request(random.Random(1))
        self.assertEqual(len(r['state']['simulated_branches']), 1)
        self.assertNotIn('teacher_secret', str(r))
        self.assertNotIn('verified_failure', str(r))
        self.assertNotIn(('probe', 'ogre'), r['operations'])
        self.assertEqual(env.step(('commit', 'veiler')), 1.)
        with self.assertRaises(ValueError): env.step(('commit', 'ogre'))

    def test_logged_request_does_not_acquire_future_probe_results(self):
        import random
        env = self.environment()
        initial = env.request(random.Random(1))
        env.step(('probe', 'ogre'))
        second = env.request(random.Random(1))
        env.step(('probe', 'pass'))
        self.assertEqual(initial['state']['simulated_branches'], [])
        self.assertEqual(len(second['state']['simulated_branches']), 1)
        self.assertEqual(len(env.history), 2)

    def test_probe_budget_and_drift_fail_closed(self):
        env = self.environment()
        env.step(('probe', 'ogre')); env.step(('probe', 'pass'))
        with self.assertRaises(ValueError): env.step(('probe', 'veiler'))
        env = self.environment()
        env.root_digest = 'changed'
        with self.assertRaises(ValueError): env.step(('commit', 'veiler'))

    def test_unknown_never_becomes_negative_label(self):
        env = self.environment()
        original = env.runner
        env.runner = lambda *a: dict(original(*a), status='budget_exhausted')
        with self.assertRaises(ValueError): env.step(('commit', 'pass'))

    def test_hidden_cards_and_deck_order_removed(self):
        cards = [dict(code=i, player=player, location=loc, sequence=0, position=pos)
                 for i,player,loc,pos in [(1,0,1,8),(2,1,2,1),(3,1,8,8),(4,0,2,1),(5,1,4,1)]]
        v = visible_state(dict(cards=cards,lp=[1,2],phase=4,turn_player=0,winner=None),0)
        self.assertEqual([c['code'] for c in v['cards']], [4,5])

    def test_return_and_independent_episode_baseline(self):
        self.assertEqual(returns_to_go([-.02, -.02, 1]), [.96, .98, 1.])
        self.assertEqual(leave_one_out_baselines([1.,-1.,0.]), [-.5,.5,0.])
        with self.assertRaises(ValueError): leave_one_out_baselines([1.])


if __name__ == '__main__':
    unittest.main()
