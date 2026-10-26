import copy
import unittest
from ygoai.rl.defense_lessons import admit


def pair():
    raw = bytearray(17); raw[0] = 70; raw[1:5] = (45812361).to_bytes(4, 'little'); raw[9] = 1; raw[16] = 1
    events = [dict(op=70, raw=raw.hex())]
    menu = [dict(act='Cancel', code=0), dict(act='Activate', code=50954680)]
    row = dict(step=1, player=0, message=16, menu=menu, observation_sha256='obs', rnn_sha256='rnn',
               logits=[1., 0.], value=0., core_response_count=5, action=1, teaching=False, searchable=False)
    prefix = dict(row, step=0, message=11, teaching=True)
    p = dict(schema='ygo-elfnote-functional-recovery-v1', probe_mode='scripted', case='case', definition={'controlled': 0},
             core_seed=1, probe_start_step=1, probe_event_start=0, training_ready=False, status='success', events=events,
             functional_evidence=dict(completed=True, success=True, challenge_negated=True),
             intervention_evidence=dict(success=True), decisions=[prefix, row])
    n = copy.deepcopy(p); n.update(probe_mode='pass', status='goal_not_reached')
    n['decisions'][1]['action'] = 0; n['functional_evidence'].update(success=False, challenge_negated=False)
    context = {key:copy.deepcopy(row[key]) for key in ['player', 'message', 'menu', 'observation_sha256', 'rnn_sha256', 'core_response_count']}
    context.update(turn_player=1, events=events, event_start=0)
    candidate = dict(case='case', training_ready=False, root_step=1,
                     root=dict(correct_action=1, chosen_action=0, observation_sha256='obs', rnn_sha256='rnn'))
    plan = dict(id='test', parent_sha256='parent', label_semantics='local goal only', challenge=45812361,
                negators=[50954680,84815190], cases=[dict(name='case', root_step=1)])
    return p,n,context,candidate,plan


class DefenseLessonTests(unittest.TestCase):
    def test_only_new_response_is_supervised_and_sources_unchanged(self):
        values=pair();before=copy.deepcopy(values)
        lesson,receipt=admit(*values)
        self.assertEqual(values,before)
        self.assertEqual(lesson['schema'],'ygo-public-response-lesson-v1')
        self.assertEqual([i for i,d in enumerate(lesson['decisions']) if d['teaching']],[1])
        self.assertEqual(receipt['teaching_steps'],[1])

    def test_equal_visible_input_cannot_hide_different_recurrent_history(self):
        values=pair();values[1]['decisions'][1]['rnn_sha256']='stale'
        with self.assertRaises(AssertionError):admit(*values)

    def test_own_chain_top_cannot_become_a_defense_lesson(self):
        values=pair();raw=bytearray.fromhex(values[2]['events'][0]['raw']);raw[9]=0
        values[2]['events'][0]['raw']=raw.hex()
        with self.assertRaises(AssertionError):admit(*values)

    def test_unknown_negative_branch_cannot_supply_a_label(self):
        values=pair();values[1]['functional_evidence']['completed']=False
        with self.assertRaises(AssertionError):admit(*values)

    def test_a_current_policy_that_already_responds_is_not_a_failure(self):
        values=pair()
        for record in values[:2]:record['decisions'][1]['logits']=[0.,1.]
        with self.assertRaises(AssertionError):admit(*values)


if __name__=='__main__':unittest.main()
