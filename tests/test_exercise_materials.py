import copy
import unittest
from unittest.mock import patch
from ygoai.rl.exercise_materials import LINEUPS, MaterialReference, GILA, JET, FORWARD
from ygoai.rl.exercise_teaching import CYBER
from ygoai.rl.exercise_starters import VEILER


class MaterialTests(unittest.TestCase):
    def test_reserved_compositions_and_material_not_in_fit(self):
        fit=[r for r in LINEUPS if r['split']=='fit']
        held=[r for r in LINEUPS if r['split']!='fit']
        self.assertFalse({(r['starter'],r['tuner']) for r in fit}&{(r['starter'],r['tuner']) for r in held})
        fit_ids={r[k] for r in fit for k in ('starter','tuner')}
        self.assertNotIn(GILA,fit_ids)
        for r in held:
            if r['split']=='reserved-combination': self.assertTrue({r['starter'],r['tuner']}<=fit_ids)

    def test_teacher_role_mapping_never_mutates_real_menu_or_opponent(self):
        lineup=next(r for r in LINEUPS if r['name']=='forward-jet')
        teacher=MaterialReference(dict(kind='combo',controlled=1,context=dict(materials=lineup)))
        snap=dict(player=1,message=11,menu=[dict(code=FORWARD),dict(code=JET)])
        original=copy.deepcopy(snap)
        with patch('ygoai.rl.exercise_contexts.ContextReference.choose',return_value=1) as parent:
            self.assertEqual(teacher.choose(snap,[]),1)
            self.assertEqual([m['code'] for m in parent.call_args.args[0]['menu']],[CYBER,VEILER])
            self.assertEqual(snap,original)
            snap['player']=0; teacher.choose(snap,[])
            self.assertEqual(parent.call_args.args[0]['menu'],original['menu'])


if __name__=='__main__': unittest.main()
