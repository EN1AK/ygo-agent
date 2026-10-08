import unittest
import numpy as np

from scripts.train_exercise_demonstration import pack
from scripts.study_teaching_regression import component_masks
from ygoai.rl.exercise_teaching import root_matches, verdict
from ygoai.rl.exercise_starters import BULB


class TeachingBoundaryTests(unittest.TestCase):
    def test_rehearsal_mask_adds_prefix_without_labeling_padding_or_other_seat(self):
        combo = dict(definition=dict(kind='combo'), roots=dict(position=3, combo=1),
                     decisions=[dict(player=p, action=0, menu=[{}]) for p in [0,1,0,1,1]])
        battle = dict(definition=dict(kind='battle'), roots=dict(battle=3),
                      decisions=combo['decisions'])
        obs = {'cards_': np.arange(5).reshape(5,1)}
        rows = [(combo, obs), (battle, obs)]
        masks, descriptors = component_masks(rows)
        _, _, old = pack(rows)
        np.testing.assert_allclose((masks[0]+masks[1])/2, old/old.sum())
        self.assertEqual(np.flatnonzero(masks[2]).tolist(), [0])
        self.assertEqual(np.flatnonzero(masks[3]).tolist(), [0])
        self.assertEqual(len(descriptors), 6)
        self.assertTrue(np.all(masks[:,6:] == 0))

    def test_opening_root_starts_at_first_controlled_seat_decision(self):
        self.assertFalse(root_matches('opening', dict(player=0), []))
        self.assertTrue(root_matches('opening', dict(player=1), []))

    def test_full_own_prefix_retained_but_only_suffix_has_loss(self):
        record = dict(definition=dict(kind='combo'), roots=dict(position=3),
                      decisions=[dict(player=p, action=i) for i,p in enumerate([0,1,0,1,1])])
        obs = {'cards_': np.arange(5, dtype=np.uint8).reshape(5,1)}
        batch, actions, weights = pack([(record,obs), (record,obs)])
        self.assertEqual(batch['cards_'][:6,0].tolist(), [1,1,3,3,4,4])
        self.assertEqual(actions[:6].tolist(), [1,1,3,3,4,4])
        self.assertEqual(weights[:6].tolist(), [0,0,.5,.5,.5,.5])
        self.assertTrue(np.all(weights[6:]==0))

    def test_other_seat_or_non_bulb_position_cannot_be_teaching_root(self):
        snap = dict(player=0, message=19, menu=[dict(code=BULB)])
        self.assertFalse(root_matches('position',snap,[]))
        snap['player']=1
        self.assertTrue(root_matches('position',snap,[]))
        snap['menu']=[dict(code=1)]
        self.assertFalse(root_matches('position',snap,[]))

    def test_battle_goal_requires_direct_damage_without_own_loss(self):
        self.assertTrue(verdict(dict(lp=[6500,8000]),[], 'battle')['success'])
        self.assertFalse(verdict(dict(lp=[8000,6500]),[], 'battle')['success'])


if __name__ == '__main__':
    unittest.main()
