"""Keep navigation filtering separate from strategic engine decisions."""
import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from project_filtered_teaching import policy_menu
from contracts import write
from run import freeze
from navigation_routes import loop_erasure


def action(**kwargs):
    return dict(dict(act='None', finish=False, unselect=False, place=0), **kwargs)


class PolicyNavigationTests(unittest.TestCase):
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
        initial=dict(state='same cards', menu=['summon','end'], committed_events=[])
        selecting=dict(state='same cards', menu=['material','cancel'], committed_events=[])
        path,replacement,loops=loop_erasure([initial,selecting,initial,dict(state='summoned')])
        self.assertEqual(path,[2]);self.assertEqual(replacement,{0:2})
        self.assertEqual(len(loops),1)
        for changed in [dict(initial,menu=['choose field','switch graveyard']),
                        dict(initial,committed_events=['activation'])]:
            self.assertEqual(loop_erasure([initial,selecting,changed])[0],[0,1])


if __name__ == '__main__':
    unittest.main()
