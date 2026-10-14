import unittest
from unittest.mock import patch
import numpy as np
from ygoai.rl.elfnote_search import probability, discover

class SearchTests(unittest.TestCase):
    def test_probabilities_fail_closed(self):
        for values,temp in [([],1),([float('nan')],1),([1],0),([1],float('inf'))]:
            with self.assertRaises(ValueError):probability(values,temp)
        np.testing.assert_allclose(probability([1001,1000]),probability([1,0]))

    def test_search_only_branches_after_observed_root(self):
        menu=[dict(code=1,act='Activate',spec='m1'),dict(code=2,act='Activate',spec='m2')]
        decisions=[dict(player=0,action=0,logits=[2,1],menu=menu) for _ in range(2)]
        prefixes=[]
        def fake(*args,**kwargs):
            prefix=kwargs['prefix'];prefixes.append(prefix)
            result=dict(status='success' if prefix else 'goal_not_reached',decisions=prefix or decisions)
            return result,{}
        plan=dict(trials_per_case=3,seconds_per_case=2,seed=1,temperature=1.5,search_branches=4,search_queue_cap=10)
        with patch('ygoai.rl.elfnote_search.run',side_effect=fake):
            witness,result=discover(None,None,None,dict(name='test'),plan,dict(root_step=1),'search')
        self.assertIsNotNone(witness)
        self.assertEqual(len(prefixes),2)
        self.assertEqual([d['action'] for d in prefixes[1]],[0,1])
        self.assertEqual(result['decisions'],4)

    def test_no_success_is_not_unsolvable(self):
        plan=dict(trials_per_case=2,seconds_per_case=2,seed=1,temperature=1.5)
        with patch('ygoai.rl.elfnote_search.run',return_value=(dict(status='goal_not_reached',decisions=[]),{})):
            witness,result=discover(None,None,None,dict(name='test'),plan,dict(root_step=0),'autonomous')
        self.assertIsNone(witness)
        self.assertEqual(result['status'],'budget_exhausted_or_no_witness')
        self.assertEqual(len(result['trials']),2)

if __name__=='__main__':unittest.main()
