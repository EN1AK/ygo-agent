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

    def test_actual_replayed_summon_advances_reference_without_choosing(self):
        from types import SimpleNamespace
        from ygoai.rl.exercise_correction import observe_replayed_action
        from ygoai.rl.exercise_teaching import QUICKDRAW
        from ygoai.rl.exercise_starters import VEILER
        teacher=SimpleNamespace(summoning=None)
        definition=dict(controlled=1,context=dict(materials=dict(starter=45894482,tuner=9742784)))
        snap=dict(player=1,message=11,menu=[dict(act='SpSummon',code=QUICKDRAW),dict(act='Summon',code=9742784)])
        observe_replayed_action(teacher,definition,snap,0)
        self.assertEqual(teacher.summoning,QUICKDRAW)
        observe_replayed_action(teacher,definition,snap,1)
        self.assertEqual(teacher.summoning,VEILER)
        observe_replayed_action(teacher,definition,dict(snap,player=0),0)
        self.assertEqual(teacher.summoning,VEILER)

    def test_forced_replay_never_queries_unsupported_reference(self):
        from types import SimpleNamespace
        from unittest.mock import patch, Mock
        from scripts.collect_teaching_exercise import execute
        obs={'actions_':np.zeros((1,3,4),np.uint8)}
        menu=[dict(act='SpSummon',code=20932152)]
        snap=dict(player=1,message=11,menu=menu,observation=obs,invalid=False,done=False)
        definition=dict(kind='battle',controlled=1,initial=[])
        source=dict(definition=definition,core_responses=[],decisions=[dict(step=0,player=1,
            message=11,menu=menu,observation_sha256=array_digest(obs),core_response_count=0,action=0)])
        class Env:
            def __init__(self): self.stepped=False
            def reset(self): pass
            def snapshot(self): return dict(snap,done=self.stepped)
            def step(self,action): self.stepped=True
            def trace(self): return dict(core_seed=1)
            def close(self): pass
        env=Env(); teacher=SimpleNamespace(summoning=None,choose=Mock(side_effect=AssertionError('must not choose')))
        with patch('scripts.collect_teaching_exercise.OpeningReference',return_value=teacher), \
             patch('scripts.collect_teaching_exercise.tensor_contract',return_value=['actions_']), \
             patch('scripts.collect_teaching_exercise.hidden_identity_audit'), \
             patch('scripts.collect_teaching_exercise.extract_trace',return_value=([],{},[])), \
             patch('scripts.collect_teaching_exercise.verdict',return_value=dict(success=True)):
            r,_=execute(SimpleNamespace(ExerciseActor=lambda *args:env),definition,'.',
                replay_record=source,replay_until=0,root_action=0)
        teacher.choose.assert_not_called()
        self.assertEqual(teacher.summoning,20932152)
        self.assertTrue(r['exact_history']['prefix_verified'])

    def test_only_corrected_root_has_weight_but_prefix_stays_in_history(self):
        decisions=[dict(player=1,action=0,menu=[{},{}],teaching=i==2) for i in range(3)]
        record=dict(definition=dict(controlled=1),decisions=decisions)
        obs=dict(actions_=np.zeros((3,3,4)),history_=np.arange(3).reshape(3,1))
        batch,targets,weights,legal=pack_scenes([(record,obs)],steps=4)
        self.assertEqual(batch['history_'].ravel().tolist(),[0,1,2,2])
        self.assertEqual(weights.ravel().tolist(),[0,0,1,0])


if __name__=='__main__': unittest.main()
