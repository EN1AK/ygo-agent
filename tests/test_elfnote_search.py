import unittest
from unittest.mock import patch
import numpy as np
from ygoai.rl.elfnote_search import probability, discover, DiverseFrontier, goal_progress

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

    def test_diverse_preserves_public_root_and_removes_chosen_deck_alias(self):
        menu=[dict(code=1,act='Activate',spec='123'),
              dict(code=1,act='Activate',spec='124'),
              dict(code=1,act='Activate',spec='m1')]
        decisions=[dict(player=0,action=0,logits=[3,2,1],menu=menu) for _ in range(2)]
        prefixes=[]
        def fake(*args,**kwargs):
            prefix=kwargs['prefix'];prefixes.append(prefix)
            return dict(status='success' if prefix else 'goal_not_reached',
                        decisions=prefix or decisions,final=dict(cards=[])),{}
        plan=dict(trials_per_case=3,seconds_per_case=5,seed=1,temperature=1.5,
                  search_branches=1,search_queue_cap=10,search_strategy='rank_progress_mix')
        with patch('ygoai.rl.elfnote_search.run',side_effect=fake):
            witness,_=discover(None,None,None,dict(name='test',goals=[]),plan,dict(root_step=1),'search')
        self.assertIsNotNone(witness)
        self.assertEqual([d['action'] for d in prefixes[1]],[0,2])

    def test_diverse_exploration_is_invariant_to_logit_scale(self):
        routes=[]
        for scale in [1.,1000.]:
            menu=[dict(code=i,act='Activate',spec=f'm{i}') for i in range(3)]
            base=[dict(player=0,action=0,logits=[0,-scale,-4*scale],menu=menu)
                  for _ in range(3)]
            prefixes=[]
            def fake(*args,**kwargs):
                prefix=kwargs['prefix'];prefixes.append(tuple(d['action'] for d in prefix))
                return dict(status='goal_not_reached',decisions=prefix+base[len(prefix):],
                            final=dict(cards=[])),{}
            plan=dict(trials_per_case=12,seconds_per_case=5,seed=10,temperature=1.5,
                      search_branches=2,search_queue_cap=20,search_strategy='rank_progress_mix')
            with patch('ygoai.rl.elfnote_search.run',side_effect=fake):
                witness,_=discover(None,None,None,dict(name='test',goals=[]),plan,dict(root_step=0),'search')
            self.assertIsNone(witness)
            routes.append(prefixes)
        self.assertEqual(routes[0],routes[1])

    def test_progress_ignores_opponent_hidden_inventory(self):
        goal=dict(monsters=[1],spells=[],hand=2)
        own=[dict(player=0,code=1,location=4,position=1)]
        hidden=[dict(player=1,code=56651978,location=2,position=0)]*10
        self.assertEqual(goal_progress(dict(cards=own),[goal]),
                         goal_progress(dict(cards=own+hidden),[goal]))
        self.assertLess(goal_progress(dict(cards=own+hidden),[goal]),1.)

    def test_frontier_capacity_and_exhaustion(self):
        frontier=DiverseFrontier(7,1)
        for i in range(40):frontier.push([i],float(i),i%3,i/40)
        self.assertLessEqual(len(frontier.pending),7)
        popped=[]
        while frontier.pending:popped.append(frontier.pop(len(popped)%3)[0][0])
        self.assertEqual(len(popped),len(set(popped)))
        with self.assertRaises(IndexError):frontier.pop(0)

if __name__=='__main__':unittest.main()
