import unittest
import numpy as np
from ygoai.rl.exercise_contexts import ready,context_verdict,ContextReference
from ygoai.rl.scene_batch import pack_scenes


class ContextTests(unittest.TestCase):
    def test_setup_history_is_retained_without_teaching_it(self):
        record=dict(definition=dict(controlled=0),decisions=[
            dict(player=0,action=0,menu=[{},{}],teaching=False),
            dict(player=0,action=1,menu=[{},{}],teaching=True)])
        obs=dict(actions_=np.zeros((2,2,1)),cards_=np.array([[1],[2]]))
        batch,_,weights,_=pack_scenes([(record,obs)],4)
        self.assertEqual(batch['cards_'].ravel().tolist(),[1,2,2,2])
        self.assertEqual(weights.ravel().tolist(),[0,1,0,0])

    def test_seat_zero_root_waits_for_real_opponent_turn(self):
        d=dict(controlled=0,kind='combo'); state=dict(cards=[])
        self.assertFalse(ready(d,{},[dict(op=40,raw='2800')],state))
        events=[dict(op=40,raw=x) for x in ('2800','2801','2800')]
        self.assertTrue(ready(d,{},events,state))
        state['cards']=[dict(player=0,location=2,code=95308449)]
        self.assertFalse(ready(d,{},events,state))

    def test_low_life_goal_uses_post_setup_lp_and_requires_hayate(self):
        d=dict(controlled=0,kind='battle-negated',expected_lp=[1000,8000])
        events=[dict(op=40,raw='2800'),dict(op=41,raw='290001')]
        state=dict(lp=[1000,8000],cards=[])
        self.assertFalse(context_verdict(state,events,d)['success'])
        state['cards']=[dict(player=0,location=4,code=8491308)]
        self.assertTrue(context_verdict(state,events,d)['success'])
        state['lp'][0]=-500
        self.assertFalse(context_verdict(state,events,d)['success'])


if __name__=='__main__': unittest.main()
