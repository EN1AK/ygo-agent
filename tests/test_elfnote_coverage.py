"""Coverage variations preserve registration and reject ambiguous source actions."""
import unittest
from collections import Counter
from scripts.study_elfnote_coverage import mapped_action, variant


class CoverageTests(unittest.TestCase):
    def test_registration_and_starters_survive_resource_and_order_changes(self):
        source = dict(name='pair-none', intervention=None, definition=dict(
            initial=[[14558128,0,1,0,8], [59438930,0,1,0,8], [97268402,0,1,0,8],
                     [59438930,0,1,0,8], [25661743,0,1,0,8], [59581480,0,1,0,8]],
            seed_sequence=[0]*8))
        settings = dict(seed=123, rotate=2, swap_resource=True)
        result = variant(source, settings, 'new', 'train', 0)
        self.assertEqual(Counter(map(tuple, source['definition']['initial'])), Counter(map(tuple, result['definition']['initial'])))
        self.assertEqual(source['definition']['seed_sequence'], [0]*8)
        self.assertNotEqual(result['definition']['seed_sequence'], [0]*8)
        self.assertEqual(result, variant(source, settings, 'new', 'train', 0))
        self.assertEqual(result['lineage'], 'elfnote-exposed-development')

    def menu(self, spec):
        return dict(code=1, act='Select', phase=0, finish=False, position=0, effect=0, unselect=False, place=0, spec=spec)

    def test_ambiguous_hand_reference_is_rejected(self):
        teacher = dict(message=15, player=0, menu=[self.menu('h1')], action=0)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            mapped_action(dict(message=15,player=0,menu=[self.menu('h2'),self.menu('h3')]), teacher)

    def test_prompt_drift_is_rejected_and_deck_aliases_are_allowed(self):
        teacher = dict(message=15, player=0, menu=[self.menu('1')], action=0)
        self.assertEqual(mapped_action(dict(message=15,player=0,menu=[self.menu('2'),self.menu('3')]),teacher),0)
        with self.assertRaisesRegex(ValueError, 'prompt divergence'):
            mapped_action(dict(message=16,player=0,menu=[self.menu('1')]),teacher)


if __name__ == '__main__':
    unittest.main()
