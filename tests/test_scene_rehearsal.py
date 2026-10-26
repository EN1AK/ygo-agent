import copy,unittest
from ygoai.rl.scene_rehearsal import validate_header


class RehearsalContractTests(unittest.TestCase):
    def config(self):
        return dict(format_version=1,role='control',learning_rate=1e-5,sequence_steps=128,histories=62,gradient_check_pairs=[27,0],
            rows=[dict(ce_coef=0.,kl_coef=1.,teaching_steps=[0,2]) for _ in range(62)])

    def test_only_explicit_bounded_contract_is_accepted(self):
        self.assertEqual(validate_header(self.config())['role'],'control')
        for updates in [dict(role='unknown'),dict(learning_rate=float('nan')),dict(sequence_steps=16),dict(rows=[])]:
            c=self.config();c.update(updates)
            with self.assertRaises(ValueError):validate_header(c)

    def test_invalid_coefficients_never_silently_disable_protection(self):
        for value in [-1.,float('inf'),True]:
            c=self.config();c['rows'][0]['kl_coef']=value
            with self.assertRaises(ValueError):validate_header(c)

    def test_masks_preserve_order_and_are_explicit(self):
        for steps in [[],[2,0],[0,0],[-1],['0']]:
            c=self.config();c['rows'][0]['teaching_steps']=steps
            with self.assertRaises((ValueError,TypeError)):validate_header(c)


if __name__=='__main__':unittest.main()
