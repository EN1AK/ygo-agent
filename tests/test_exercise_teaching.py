import unittest
import numpy as np

from scripts.train_exercise_demonstration import pack
from ygoai.rl.exercise_teaching import root_matches, verdict
from ygoai.rl.exercise_starters import BULB


class TeachingBoundaryTests(unittest.TestCase):
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
