import unittest
from ygoai.rl.elfnote_recovery import CHALLENGE,NEGATORS,challenge_evidence,board_resources,intervention_evidence
from ygoai.rl.elfnote_search import discover


def chaining(code,player,link):
    raw=bytes([70])+code.to_bytes(4,'little')+bytes(4)+bytes([player,4,0])+bytes(4)+bytes([link])
    return dict(op=70,raw=raw.hex())


def event(op,*payload):return dict(op=op,raw=bytes([op,*payload]).hex())


class RecoveryTests(unittest.TestCase):
    def test_negated_handtrap_is_not_observed_recovery(self):
        chain=[chaining(14558128,1,1),event(73,1),event(74)]
        self.assertTrue(intervention_evidence(chain,14558128)['success'])
        self.assertFalse(intervention_evidence([chain[0],event(75,1),*chain[1:]],14558128)['success'])
        self.assertFalse(intervention_evidence(chain[:-1],14558128)['success'])
        self.assertFalse(intervention_evidence([chaining(14558128,0,1),*chain[1:]],14558128)['success'])

    def test_actual_resolved_negator_and_negated_challenge_are_required(self):
        chain=[chaining(CHALLENGE,1,1),chaining(NEGATORS[0],0,2),event(73,2),event(75,1),event(74)]
        self.assertTrue(challenge_evidence(chain,0)['success'])
        self.assertFalse(challenge_evidence([e for e in chain if e['op']!=75],0)['success'])
        self.assertFalse(challenge_evidence([chain[0],chaining(NEGATORS[0],1,2),*chain[2:]],0)['success'])
        self.assertFalse(challenge_evidence([*chain[:2],event(76,2),*chain[2:]],0)['success'])
        self.assertFalse(challenge_evidence([chain[0],chaining(14558128,0,2),chaining(NEGATORS[0],0,3),event(73,3),event(75,1),event(74)],0)['success'])

    def test_chain_numbers_do_not_leak_across_chains(self):
        events=[chaining(123,1,1),chaining(NEGATORS[0],0,2),event(73,2),event(75,1),event(74),
                chaining(CHALLENGE,1,1),event(73,1),event(74)]
        self.assertFalse(challenge_evidence(events,0)['success'])
        self.assertFalse(challenge_evidence([chaining(CHALLENGE,1,1),chaining(NEGATORS[0],0,2),event(75,1)],0)['completed'])

    def test_board_count_is_only_a_candidate(self):
        resources=board_resources(dict(cards=[dict(code=NEGATORS[0],player=0,location=4,position=1),dict(code=1,player=0,location=2,position=8),dict(code=2,player=1,location=2,position=8)]))
        self.assertEqual(resources,dict(negators=[NEGATORS[0]],hand=1))
        self.assertFalse(challenge_evidence([],0)['success'])

    def test_search_never_changes_scripted_probe_decisions(self):
        menu=[dict(code=1,act='Activate',spec='m1'),dict(code=2,act='Activate',spec='m2')]
        decisions=[dict(player=0,action=0,logits=[2,1],menu=menu,searchable=True),dict(player=0,action=0,logits=[2,1],menu=menu,searchable=False)]
        prefixes=[]
        def runner(*args,**kwargs):
            prefix=kwargs['prefix'];prefixes.append(prefix)
            return dict(status='success' if prefix else 'goal_not_reached',decisions=prefix or decisions,final=dict(cards=[])),{}
        plan=dict(trials_per_case=4,seconds_per_case=5,seed=1,temperature=1.5,search_strategy='rank_progress_mix',search_branches=4,search_queue_cap=10)
        witness,_=discover(None,None,None,dict(name='probe',goals=[]),plan,dict(root_step=0),'search',runner=runner)
        self.assertIsNotNone(witness);self.assertEqual([len(p) for p in prefixes],[0,1])


if __name__=='__main__':unittest.main()
