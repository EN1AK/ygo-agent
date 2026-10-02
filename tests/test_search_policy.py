from dataclasses import replace
import unittest

from ygoai.rl.candidate_search import SearchBudget
from ygoai.rl.search_policy import SearchPolicy
from test_candidate_search import ROOT, Backend


class SearchPolicyTest(unittest.TestCase):
    def test_default_rejects_oracle_and_diagnostic_is_explicit(self):
        backend=Backend()
        policy=SearchPolicy(budget=SearchBudget(depth=1))
        report=policy.choose(ROOT,backend,seed=1)
        self.assertEqual(report['fallback_reason'],'oracle_forbidden_in_fair_play')
        self.assertEqual(backend.opened,0)
        report=replace(policy,fair_play=False,policy_kl=.2).choose(ROOT,backend,seed=1)
        self.assertEqual(report['selected_action'],7)
        self.assertEqual(report['raw_action'],3)

    def test_eligibility_never_opens_backend(self):
        backend=Backend()
        policy=SearchPolicy()
        cases=[(replace(ROOT,legal_actions=(3,),logits=(0.,)),False,'single_legal_action'),
               (ROOT,True,'unsupported_external_opponent_state'),
               (replace(ROOT,prompt='select_place',logits=(0.,-30.)),False,'ineligible_prompt_and_low_entropy')]
        for root,external,reason in cases:
            result=policy.choose(root,backend,seed=1,external_opponent=external)
            self.assertEqual(result['fallback_reason'],reason)
            self.assertEqual(result['selected_action'],result['raw_action'])
        self.assertEqual(backend.opened,0)


if __name__=='__main__': unittest.main()
