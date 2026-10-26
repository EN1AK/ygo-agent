import copy
import unittest
from ygoai.rl.retention_candidates import normal_regression,strict_outcomes


def records():
    definition=dict(controlled=0,initial=[1],seed_sequence=[7]);case=dict(name='case',split='rehearsal',definition=definition,goals=[dict(monsters=[9],spells=[],hand=1)])
    cards=[dict(code=9,player=0,location=4,position=1),dict(code=56651978,player=0,location=2,position=0)]
    d=dict(player=0,menu=[0,1],observation_sha256='obs',rnn_sha256='old-memory',action=0)
    before=dict(schema='ygo-elfnote-search-route-v1',case='case',definition=definition,status='success',final=dict(cards=cards),decisions=[d])
    after=copy.deepcopy(before);after['status']='goal_not_reached';after['final']['cards']=after['final']['cards'][1:];after['decisions'][0].update(action=1,rnn_sha256='current-memory')
    for record in [before,after]:record['outcomes']=[dict(checks=c) for c in strict_outcomes(record,case['goals'],0)]
    return before,after,case


class RetentionCandidateTests(unittest.TestCase):
    def test_different_policy_memory_never_turns_prior_actions_into_labels(self):
        values=records();original=copy.deepcopy(values);candidate=normal_regression(*values)
        self.assertFalse(candidate['training_ready']);self.assertEqual(candidate['first_observed_route_divergence']['current_rnn_sha256'],'current-memory')
        self.assertNotIn('correct_action',candidate['first_observed_route_divergence']);self.assertEqual(values,original)

    def test_incomplete_or_misgraded_goals_cannot_be_regressions(self):
        values=records();values[1]['status']='budget_exhausted'
        with self.assertRaises(AssertionError):normal_regression(*values)
        values=records();values[1]['final']=copy.deepcopy(values[0]['final'])
        with self.assertRaises(AssertionError):normal_regression(*values)

    def test_matching_names_do_not_override_different_dealt_contexts(self):
        values=records();values[1]['definition']['initial']=[2]
        with self.assertRaises(AssertionError):normal_regression(*values)


if __name__=='__main__':unittest.main()
