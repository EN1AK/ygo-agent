import copy
import unittest
import numpy as np
from ygoai.rl.exercise_actor import array_digest
from ygoai.rl.exercise_correction import checked_prefix_action
from ygoai.rl.scene_batch import pack_scenes


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.obs={'actions_':np.zeros((1,3,4),np.uint8)}
        self.snap=dict(player=1,message=10,menu=[dict(phase='Battle'),dict(phase='Main2')])
        self.responses=[dict(kind='int',data='00000000')]
        self.record=dict(core_responses=self.responses,decisions=[dict(step=0,**self.snap,
            observation_sha256=array_digest(self.obs),core_response_count=1,action=0)])

    def test_replay_refuses_observation_menu_seat_and_response_drift(self):
        self.assertEqual(checked_prefix_action(self.record,0,self.snap,self.obs,self.responses),0)
        for key,value in [('menu',list(reversed(self.snap['menu']))),('player',0),('message',11)]:
            with self.assertRaises(ValueError):
                checked_prefix_action(self.record,0,dict(self.snap,**{key:value}),self.obs,self.responses)
        with self.assertRaises(ValueError):
            checked_prefix_action(self.record,0,self.snap,{'actions_':self.obs['actions_']+1},self.responses)
        with self.assertRaises(ValueError):
            checked_prefix_action(self.record,0,self.snap,self.obs,[dict(kind='int',data='01000000')])

    def test_only_corrected_root_has_weight_but_prefix_stays_in_history(self):
        decisions=[dict(player=1,action=0,menu=[{},{}],teaching=i==2) for i in range(3)]
        record=dict(definition=dict(controlled=1),decisions=decisions)
        obs=dict(actions_=np.zeros((3,3,4)),history_=np.arange(3).reshape(3,1))
        batch,targets,weights,legal=pack_scenes([(record,obs)],steps=4)
        self.assertEqual(batch['history_'].ravel().tolist(),[0,1,2,2])
        self.assertEqual(weights.ravel().tolist(),[0,0,1,0])


if __name__=='__main__': unittest.main()
