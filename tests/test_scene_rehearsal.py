import copy,unittest
from ygoai.rl.scene_rehearsal import validate_header,validate_training_mode,load_rows,sha
from types import SimpleNamespace


class RehearsalContractTests(unittest.TestCase):
    def test_ordinary_args_do_not_need_optional_q_fields(self):
        args=SimpleNamespace(distributed=False,learner_device_ids=[0],bfloat16=False)
        validate_training_mode(args)
        args.q_training_mode='shadow_observation'
        with self.assertRaises(ValueError):validate_training_mode(args)

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

    def test_late_teaching_is_kept_in_complete_longer_histories(self):
        import json,tempfile,numpy as np
        from pathlib import Path
        from ygoai.rl.scene_batch import pack_scenes
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'record.json'
            decisions=[dict(player=0,menu=[{},{}],action=1) for _ in range(193)]
            path.write_text(json.dumps(dict(full_legal_menu=False,definition={'controlled':0},decisions=decisions)))
            np.savez_compressed(path.with_suffix('.npz'),actions_=np.zeros((193,2,12),np.uint8))
            row=dict(path=str(path),record_sha256=sha(path),observations_sha256=sha(path.with_suffix('.npz')),controlled=0,teaching_steps=[129,192])
            config=dict(rows=[row],sequence_steps=128,full_legal_menu=False)
            with self.assertRaisesRegex(ValueError,'never truncate'):load_rows(config)
            config['sequence_steps']=256
            rows=load_rows(config);_,_,weights,_=pack_scenes(rows,256)
            self.assertEqual(weights[129,0],.5);self.assertEqual(weights[192,0],.5)
            self.assertEqual(float(weights[193:,0].sum()),0.)
            c=self.config();c['sequence_steps']=256
            self.assertEqual(validate_header(c)['sequence_steps'],256)

    def test_filtered_rehearsal_cannot_be_applied_to_full_menu_ppo(self):
        args=SimpleNamespace(distributed=False,learner_device_ids=[0],bfloat16=False,full_legal_menu=True)
        with self.assertRaisesRegex(ValueError,'menu contracts differ'):
            validate_training_mode(args,dict(full_legal_menu=False))


if __name__=='__main__':unittest.main()
