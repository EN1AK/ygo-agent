import copy,unittest
from ygoai.rl.defense_candidates import propose


class DefenseCandidateTests(unittest.TestCase):
    def pair(self):
        row=dict(step=0,player=0,message=16,menu=[dict(code=50954680,act='Activate'),dict(code=0,act='Cancel')],
                 observation_sha256='same',rnn_sha256='same-state',core_response_count=10,logits=[-2.,0.],action=0,teaching=False)
        teacher=dict(schema='ygo-elfnote-functional-recovery-v1',definition={},case='fit',core_seed=7,probe_mode='scripted',status='success',
            functional_evidence=dict(success=True),intervention_evidence=dict(success=True),probe_start_step=0,decisions=[row])
        baseline=copy.deepcopy(teacher);baseline.update(probe_mode='policy',status='goal_not_reached',functional_evidence=dict(success=False));baseline['decisions'][0]['action']=1
        return teacher,baseline

    def test_candidate_does_not_relabel_original_probe_or_start_training(self):
        teacher,baseline=self.pair();original=copy.deepcopy((teacher,baseline));r=propose(teacher,baseline)
        self.assertFalse(r['training_ready']);self.assertEqual(r['status'],'needs_current_model_reconstruction')
        self.assertEqual((teacher,baseline),original)

    def test_equal_visible_observation_with_different_memory_is_not_same_root(self):
        teacher,baseline=self.pair();baseline['decisions'][0]['rnn_sha256']='different-state'
        with self.assertRaises(AssertionError):propose(teacher,baseline)

    def test_budget_exhaustion_is_not_a_verified_negative_example(self):
        teacher,baseline=self.pair();baseline['status']='budget_exhausted'
        with self.assertRaises(AssertionError):propose(teacher,baseline)

    def test_scripted_cancel_cannot_impersonate_the_current_policy(self):
        teacher,baseline=self.pair()
        for record in [teacher,baseline]:record['decisions'][0]['logits']=[1.,0.]
        with self.assertRaises(AssertionError):propose(teacher,baseline)

    def test_negative_python_index_is_not_a_native_legal_action(self):
        teacher,baseline=self.pair();teacher['decisions'][0]['action']=-2
        with self.assertRaises(AssertionError):propose(teacher,baseline)


if __name__=='__main__':unittest.main()
