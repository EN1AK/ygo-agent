import copy,unittest
from ygoai.rl.elfnote_curriculum import correction_row


class CurriculumTests(unittest.TestCase):
    def record(self):
        menu=[dict(code=1,spec='h1'),dict(code=2,spec='h2')]
        decisions=[dict(player=0,menu=menu,action=1,logits=[2.,1.],searchable=i==1,teaching=True) for i in range(3)]
        return dict(status='success',definition=dict(controlled=0),functional_evidence=dict(success=True),intervention_evidence=dict(success=True),root_step=1,probe_start_step=2,decisions=decisions)

    def test_only_verified_public_correction_is_supervised(self):
        record=self.record();original=copy.deepcopy(record)
        (masked,obs),steps=correction_row(record,{'history':'preserved'})
        self.assertEqual(steps,[1]);self.assertEqual([d['teaching'] for d in masked['decisions']],[False,True,False])
        self.assertEqual(obs,{'history':'preserved'});self.assertEqual(record,original)

    def test_goal_failure_or_no_corrective_action_is_not_a_lesson(self):
        for field,value in [('status','goal_not_reached'),('functional_evidence',dict(success=False)),('intervention_evidence',dict(success=False)),('root_step',None)]:
            record=self.record();record[field]=value
            with self.assertRaises(ValueError):correction_row(record,{})
        record=self.record();record['decisions'][1]['action']=0
        with self.assertRaises(ValueError):correction_row(record,{})

    def test_probe_or_pre_response_search_weight_is_rejected(self):
        for step in [0,2]:
            record=self.record();record['decisions'][step]['searchable']=True
            with self.assertRaises(ValueError):correction_row(record,{})
        record=self.record();record['decisions'][1]['player']=1
        with self.assertRaises(ValueError):correction_row(record,{})


if __name__=='__main__':unittest.main()
