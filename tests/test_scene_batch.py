import unittest
import numpy as np
from ygoai.rl.scene_batch import pack_scenes
from ygoai.rl.exercise_teaching import verdict, root_matches
from ygoai.rl.exercise_starters import ENGAGE, VEILER, OGRE


class SceneTests(unittest.TestCase):
    def test_complete_controlled_seat_history_including_single_decision(self):
        record=dict(definition=dict(controlled=0), decisions=[dict(player=0,action=1,menu=[{},{}]),dict(player=1,action=0,menu=[{}])])
        obs=dict(actions_=np.zeros((2,3,4)), cards_=np.array([[11],[22]]))
        batch, targets, weights, legal=pack_scenes([(record,obs)],steps=3)
        self.assertEqual(batch['cards_'].ravel().tolist(),[11,11,11])
        self.assertEqual(targets.tolist(),[1,0,0])
        self.assertEqual(weights.ravel().tolist(),[1,0,0])
        self.assertEqual(legal.tolist(),[[True,True,False],[True,False,False],[True,False,False]])

    def test_negated_goal_requires_own_turn_end_and_no_damage(self):
        events=[dict(op=40,raw='2800'),dict(op=41,raw='290002')]
        self.assertFalse(verdict(dict(lp=[8000,8000]),events,'battle-negated')['success'])
        events += [dict(op=40,raw='2801'),dict(op=41,raw='290001')]
        self.assertTrue(verdict(dict(lp=[8000,8000]),events,'battle-negated')['success'])
        self.assertFalse(verdict(dict(lp=[8000,6500]),events,'battle-negated')['success'])

    def test_interaction_requires_recovery_stopped_with_at_most_one_hand_trap(self):
        cards=[dict(code=ENGAGE,player=0,location=16),dict(code=VEILER,player=1,location=16)]
        self.assertTrue(verdict(dict(cards=cards),[],'interaction')['success'])
        cards.append(dict(code=OGRE,player=1,location=16))
        self.assertFalse(verdict(dict(cards=cards),[],'interaction')['success'])
        cards[0]['location']=2
        self.assertFalse(verdict(dict(cards=cards),[],'interaction')['success'])


if __name__=='__main__': unittest.main()
