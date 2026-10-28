"""Keep navigation filtering separate from strategic engine decisions."""
import unittest
from project_filtered_teaching import policy_menu


def action(**kwargs):
    return dict(dict(act='None', finish=False, unselect=False, place=0), **kwargs)


class PolicyNavigationTests(unittest.TestCase):
    def test_chain_decline_and_optional_effect_decline_remain(self):
        menu = [action(act='Activate'), action(act='Cancel')]
        for message in (12, 13, 16):
            self.assertEqual(policy_menu(message, menu), menu)

    def test_selection_undo_and_back_removed_but_finish_remains(self):
        forward, undo, back, finish = action(), action(unselect=True), action(act='Cancel'), action(finish=True)
        self.assertEqual(policy_menu(26, [forward, undo, back, finish]), [forward, finish])
        self.assertEqual(policy_menu(15, [forward, back, finish]), [forward, finish])

    def test_forced_correction_or_cancel_keeps_a_legal_menu(self):
        undo, back = action(unselect=True), action(act='Cancel')
        self.assertEqual(policy_menu(26, [undo]), [undo])
        self.assertEqual(policy_menu(15, [back]), [back])

    def test_empty_place_return_does_not_hide_disable_field_skip(self):
        place, empty = action(place=1), action(finish=True)
        self.assertEqual(policy_menu(18, [place, empty]), [place])
        self.assertEqual(policy_menu(24, [place, empty]), [place, empty])


if __name__ == '__main__':
    unittest.main()
