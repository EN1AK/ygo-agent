import unittest

import numpy as np

from ygoai.rl.belief_dataset import validate_dataset, split_by_duel, calibration_metrics


def fixture():
    return dict(public_tokens=np.ones((4,2,3),dtype=np.float32),
        public_mask=np.ones((4,2),dtype=bool),hidden_slot_features=np.ones((4,2,2),dtype=np.float32),
        hidden_tokens=np.array([[1,2],[2,1],[1,2],[2,1]],dtype=np.int32),
        slot_mask=np.ones((4,2),dtype=bool),remaining_counts=np.tile([0,1,1],(4,1)),
        duel_ids=np.array(['duel1','duel1','duel2','duel2']))


class BeliefDatasetTest(unittest.TestCase):
    def test_group_split_never_leaks_same_duel(self):
        train,held=split_by_duel(fixture(),seed=8)
        self.assertFalse(set(train['duel_ids'])&set(held['duel_ids']))
        self.assertEqual(len(train['duel_ids'])+len(held['duel_ids']),4)

    def test_privileged_extra_field_and_invalid_labels_fail(self):
        data=fixture(); data['full_state']=np.zeros((4,2))
        with self.assertRaises(ValueError): validate_dataset(data)
        data=fixture(); data['hidden_tokens'][0]=[1,1]
        with self.assertRaisesRegex(ValueError,'constraints'): validate_dataset(data)
        data=fixture(); data['public_mask'][0,1]=False
        with self.assertRaisesRegex(ValueError,'Padding'): validate_dataset(data)
        data=fixture(); data['duel_ids'][:]='duel1'
        with self.assertRaisesRegex(ValueError,'independent'): split_by_duel(data)

    def test_calibration_hand_calculated(self):
        report=calibration_metrics(np.log(np.array([[[.25,.75],[.5,.5]]])),
            np.array([[1,1]]),np.array([[True,True]]))
        self.assertAlmostEqual(report['nll'],-np.log(.375)/2)
        self.assertEqual(report['top1_accuracy'],.5)
        self.assertAlmostEqual(report['ece'],.375)


if __name__=='__main__': unittest.main()
