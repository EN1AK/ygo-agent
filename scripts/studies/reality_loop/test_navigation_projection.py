"""Keep navigation filtering separate from strategic engine decisions."""
import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from project_filtered_teaching import policy_menu
from contracts import write
from run import freeze
from navigation_routes import loop_erasure,chain_menu_permutation,align_action_rows,committed_events,selection_navigation_responses,selection_context,loop_selection
from ygoai.rl.chain_response import chain_response,canonical_responses


def action(**kwargs):
    return dict(dict(act='None', finish=False, unselect=False, place=0), **kwargs)


class PolicyNavigationTests(unittest.TestCase):
    def test_idle_loop_ignores_only_inactive_picker_counts(self):
        a=[[1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0]]
        b=[[1,0,0,0,0,2,2,0,0,0,0,0,0,0,0,0]]
        self.assertEqual(loop_selection(11,a),loop_selection(11,b))
        self.assertNotEqual(loop_selection(15,a),loop_selection(15,b))
        self.assertEqual(a[0][5:7],[1,1])

    def test_chain_decline_and_optional_effect_decline_remain(self):
        menu = [action(act='Activate'), action(act='Cancel')]
        for message in (12, 13, 16):
            self.assertEqual(policy_menu(message, menu), menu)

    def test_selection_undo_and_back_removed_but_finish_remains(self):
        forward, undo, back, finish = action(), action(unselect=True), action(act='Cancel'), action(finish=True)
        self.assertEqual(policy_menu(26, [forward, undo, back, finish], True), [forward, finish])
        self.assertEqual(policy_menu(15, [forward, back, finish], True), [forward, finish])
        self.assertEqual(policy_menu(15, [forward, back, finish], False), [forward, back, finish])
        self.assertEqual(policy_menu(26, [forward, undo, back, finish], False), [forward, back, finish])

    def test_forced_correction_or_cancel_keeps_a_legal_menu(self):
        undo, back = action(unselect=True), action(act='Cancel')
        self.assertEqual(policy_menu(26, [undo]), [undo])
        self.assertEqual(policy_menu(15, [back]), [back])

    def test_empty_place_return_does_not_hide_disable_field_skip(self):
        place, empty = action(place=1), action(finish=True)
        self.assertEqual(policy_menu(18, [place, empty]), [place])
        self.assertEqual(policy_menu(24, [place, empty]), [place, empty])

    def test_new_runs_reject_full_menu_or_unadmitted_projection(self):
        for plan, reason in [({'rows': []}, 'filtered-policy'),
                ({'rows': [], 'full_legal_menu': True}, 'filtered-policy'),
                ({'rows': [], 'full_legal_menu': False, 'training_admission_pending': True}, 'curriculum admission')]:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                write(root/'plan.json', plan)
                args = SimpleNamespace(output=root/'new-run', frozen=root)
                with self.assertRaisesRegex(ValueError, reason):
                    freeze(args)

    def test_loop_erasure_requires_identical_menu_and_committed_events(self):
        initial=dict(message=11,state='same cards',menu=['summon','end'],committed_events=[])
        selecting=dict(message=26,state='same cards',menu=['material','cancel'],committed_events=[])
        path,replacement,loops=loop_erasure([initial,selecting,initial,dict(message=11,state='summoned')],[False,True,False])
        self.assertEqual(path,[2]);self.assertEqual(replacement,{0:2})
        self.assertEqual(len(loops),1)
        for changed in [dict(initial,menu=['choose field','switch graveyard']),
                        dict(initial,committed_events=['activation'])]:
            self.assertEqual(loop_erasure([initial,selecting,changed],[False,True])[0],[0,1])
        chain=dict(initial,message=16)
        self.assertEqual(loop_erasure([chain,chain],[True])[0],[0])
        event=dict(op=16,raw='layout',player=0,forced=False,special_count=0,cards=[])
        self.assertNotEqual(committed_events([event]),committed_events([event,event]))

    def test_chain_permutation_has_exact_action_set_and_tensor_alignment(self):
        import numpy as np
        a,b=dict(code=1),dict(code=2)
        permutation=chain_menu_permutation(16,[a,b],[b,a])
        self.assertEqual(permutation,[1,0])
        with self.assertRaises(ValueError):chain_menu_permutation(15,[a,b],[b,a])
        with self.assertRaises(ValueError):chain_menu_permutation(16,[a,b],[a,dict(code=3)])
        obs=dict(actions_=np.array([[[2],[1],[0]]]),selection_=np.array([[7]]))
        aligned=align_action_rows(obs,permutation)
        self.assertEqual(aligned['actions_'].tolist(),[[[1],[2],[0]]])
        self.assertEqual(aligned['selection_'].tolist(),[[7]])

    def test_independent_chain_rebinding_checks_original_raw_response(self):
        from copy import deepcopy
        a=dict(code=1000,player=0,location=8,sequence=0,description=(1000<<4))
        b=dict(code=2000,player=0,location=8,sequence=1,description=(2000<<4))
        prompt=dict(op=16,player=0,forced=False,cards=[a,b])
        choice=dict(act='Activate',spec='s2',code=2000,effect=10010)
        self.assertEqual(int.from_bytes(chain_response(dict(prompt,cards=[b,a]),choice),'little'),0)
        system=dict(b,description=221)
        self.assertEqual(int.from_bytes(chain_response(dict(prompt,cards=[a,system]),dict(choice,effect=221)),'little'),1)
        record=dict(core_responses=[dict(kind='int',data='01000000')],decisions=[dict(message=16,
            player=0,core_response_count=0,chain_prompt=prompt,menu=[choice],action=0)])
        normalized=canonical_responses(record)
        other=deepcopy(record);other['decisions'][0]['chain_prompt']['cards']=[b,a]
        other['core_responses'][0]['data']='00000000'
        self.assertEqual(canonical_responses(other),normalized)
        bad=deepcopy(record);bad['core_responses'][0]['data']='00000000'
        with self.assertRaisesRegex(ValueError,'declared action'):canonical_responses(bad)

    def test_automatic_core_undo_is_detected_without_a_model_decision(self):
        prompts=[dict(op=26,cards=[{}],unselect=[]),dict(op=26,cards=[],unselect=[{}])]
        responses=[dict(kind='bytes',data='0100'),dict(kind='bytes',data='0100')]
        self.assertEqual(selection_navigation_responses(prompts,responses),({1},set()))
        frame=dict(message=26,state='same',menu=['choose'])
        self.assertEqual(loop_erasure([frame,frame],[True])[0],[])

    def test_identical_material_menus_do_not_merge_different_summon_targets(self):
        def idle():return dict(op=11,special=[dict(code=1000,player=0,location=64,sequence=0),dict(code=2000,player=0,location=64,sequence=1)])
        selector=dict(op=15)
        response=lambda value:dict(kind='int',data=value.to_bytes(4,'little',signed=True).hex())
        unique={(0,1000),(0,2000)}
        a=selection_context([idle(),selector],[response(1)],unique)
        b=selection_context([idle(),selector,idle(),selector],[response(1),response(-1),response((1<<16)|1)],unique)
        self.assertEqual(a['epoch'],b['epoch'])
        self.assertNotEqual(a['pending_summon'],b['pending_summon'])
        ended=selection_context([idle(),selector,idle()],[response(1),response(-1)],unique)
        self.assertEqual(ended,dict(epoch=0,pending_summon=None))


if __name__ == '__main__':
    unittest.main()
