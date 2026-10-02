from contextlib import contextmanager
import math
import unittest

import numpy as np

from ygoai.rl.candidate_search import (SearchBudget, SearchFailure, SearchPosition,
                                     candidate_search, regularized_policy)


ROOT = SearchPosition(0, (3, 7), (math.log(.8), math.log(.2)), .1, 'root', 'select_idlecmd')


class Backend:
    information_mode = 'oracle_exact_state'
    snapshot_bytes = 32

    def __init__(self, chain=False, fail=False):
        self.opened = self.closed = 0
        self.chain, self.fail = chain, fail

    @contextmanager
    def branch(self, particle, seed):
        self.opened += 1
        parent = self

        class Branch:
            action = None
            closed = False

            def evaluate(self):
                if self.closed:
                    raise SearchFailure('closed_branch')
                if self.action is None:
                    return ROOT
                if parent.fail:
                    raise SearchFailure('backend_failure')
                # The opponent acts at the leaf, hence its negative value is
                # positive for the root player. Root action 7 is better.
                return SearchPosition(1, (2,), (0.,), .5 if self.action==3 else -.5,
                                      'leaf', 'select_chain' if parent.chain else 'select_card')

            def step(self, action):
                if self.closed:
                    raise SearchFailure('closed_branch')
                self.action = action

        branch = Branch()
        try:
            yield branch
        finally:
            branch.closed = True
            self.closed += 1


class CandidateSearchTest(unittest.TestCase):
    def test_regularized_update_matches_hand_computed_odds(self):
        # Odds = (raw odds=4) * exp((q0-q1)/beta) = 4/e.
        result = regularized_policy(ROOT.logits, [0., 1.], policy_kl=1.)
        self.assertAlmostEqual(result[0]/result[1], 4/math.e)
        np.testing.assert_allclose(regularized_policy(ROOT.logits,[0.,0.]), [.8,.2])
        tied=regularized_policy([0.,0.],[.5,.5])
        np.testing.assert_array_equal(tied,[.5,.5])
        self.assertEqual(int(np.argmax(tied)),0)

    def test_magnet_policy_and_extreme_logits_are_finite(self):
        result = regularized_policy([10000.,-10000.], [1.,-1.], policy_kl=.5,
                                    magnet_kl=.5, magnet_logits=[-10000.,10000.])
        np.testing.assert_allclose(result, [1/(1+math.exp(-2)), 1/(1+math.exp(2))])

    def test_paired_seeds_viewpoint_and_cleanup(self):
        backend = Backend()
        result = candidate_search(ROOT,backend,SearchBudget(depth=1),seed=9,policy_kl=.2)
        self.assertTrue(result['completed'])
        self.assertEqual(result['selected_action'],7)
        self.assertEqual([x['mean'] for x in result['action_returns']],[-.5,.5])
        self.assertEqual(backend.opened,backend.closed)
        for i in range(0,len(result['rollouts']),2):
            self.assertEqual(result['rollouts'][i]['rollout_seed'],result['rollouts'][i+1]['rollout_seed'])
        again = candidate_search(ROOT,Backend(),SearchBudget(depth=1),seed=9,policy_kl=.2)
        self.assertEqual(result['rollouts'],again['rollouts'])

    def test_unresolved_chain_falls_back_without_scoring_leaf(self):
        backend = Backend(chain=True)
        result = candidate_search(ROOT,backend,SearchBudget(depth=1),seed=9)
        self.assertFalse(result['completed'])
        self.assertEqual(result['fallback_reason'],'unresolved_chain_at_depth_limit')
        self.assertEqual(result['selected_action'],3)
        self.assertEqual(result['action_returns'],[])
        self.assertEqual(backend.opened,backend.closed)

    def test_partial_backend_failure_cleans_up_and_strict_raises(self):
        backend = Backend(fail=True)
        result = candidate_search(ROOT,backend,SearchBudget(depth=1),seed=9)
        self.assertEqual(result['fallback_reason'],'backend_failure')
        with self.assertRaises(SearchFailure):
            candidate_search(ROOT,backend,SearchBudget(depth=1),seed=9,strict=True)
        self.assertEqual(backend.opened,backend.closed)

    def test_oracle_blocked_before_branch_in_fair_play(self):
        backend = Backend()
        result = candidate_search(ROOT,backend,SearchBudget(depth=1),seed=9,fair_play=True)
        self.assertEqual(result['fallback_reason'],'oracle_forbidden_in_fair_play')
        self.assertEqual(backend.opened,0)

    def test_memory_and_wall_budgets_fall_back(self):
        backend = Backend()
        result = candidate_search(ROOT,backend,SearchBudget(snapshot_bytes=1),seed=9)
        self.assertEqual(result['fallback_reason'],'snapshot_memory_budget')
        times=iter([0.,2.,3.])
        result = candidate_search(ROOT,backend,SearchBudget(wall_seconds=1),seed=9,clock=lambda:next(times))
        self.assertEqual(result['fallback_reason'],'wall_time_budget')
        self.assertEqual(backend.opened,0)

    def test_single_sample_uncertainty_is_unknown(self):
        result=candidate_search(ROOT,Backend(),SearchBudget(depth=1,rollouts_per_action=1),seed=9)
        self.assertIsNone(result['action_returns'][0]['stderr'])

    def test_candidate_subset_preserves_other_action_mass(self):
        result=candidate_search(ROOT,Backend(),SearchBudget(depth=1,candidates=1),seed=9)
        np.testing.assert_allclose(result['updated_policy'],[.8,.2])

    def test_particle_execution_order_does_not_change_aggregation(self):
        budget=SearchBudget(depth=1,particles=3,rollouts_per_action=2)
        a=candidate_search(ROOT,Backend(),budget,seed=9)
        b=candidate_search(ROOT,Backend(),budget,seed=9,particle_order=[2,1,0])
        self.assertEqual(a['action_returns'],b['action_returns'])
        self.assertEqual(a['updated_policy'],b['updated_policy'])
        self.assertEqual(a['selected_action'],b['selected_action'])
        self.assertEqual(a['action_returns'][0]['uncertainty_unit'],'particle')
        with self.assertRaises(ValueError):
            candidate_search(ROOT,Backend(),budget,seed=9,particle_order=[0,0,1])


if __name__ == '__main__':
    unittest.main()
