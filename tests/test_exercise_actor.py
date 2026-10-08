import unittest

import numpy as np

from ygoai.rl.exercise_actor import actor_grade, hidden_identity_audit, settled
from ygoai.rl.exercise_starters import ENGAGE, VEILER, OGRE


class ActorBoundaryTests(unittest.TestCase):
    def test_both_identity_channels_must_hide_opponent_private_cards(self):
        cards = np.zeros((1, 3, 41), dtype=np.uint8)
        cards[0, :, 2] = [1, 2, 7]
        cards[0, :, 4] = 1
        ids = np.zeros((1, 3), dtype=np.uint16)
        obs = dict(cards_=cards, visible_card_ids_=ids)
        self.assertEqual(hidden_identity_audit(obs), 3)
        ids[0, 1] = 42
        with self.assertRaisesRegex(ValueError, 'private identity'):
            hidden_identity_audit(obs)
        ids[:] = 0
        cards[0, 2, 0] = 1
        with self.assertRaisesRegex(ValueError, 'private identity'):
            hidden_identity_audit(obs)

    def test_old_chain_end_cannot_finish_new_exercise(self):
        snap = dict(done=False, invalid=False, phase=4)
        events = [dict(op=74), dict(op=70)]
        self.assertFalse(settled('interaction', snap, events, 1))
        events.append(dict(op=74))
        self.assertTrue(settled('interaction', snap, events, 1))

    def test_two_handtraps_exceed_one_card_response_budget(self):
        cards = [dict(code=ENGAGE, player=0, location=16),
                 dict(code=VEILER, player=1, location=16)]
        self.assertEqual(actor_grade('interaction', dict(cards=cards), [], True), 'success')
        cards.append(dict(code=OGRE, player=1, location=16))
        self.assertEqual(actor_grade('interaction', dict(cards=cards), [], True), 'verified_failure')
        self.assertEqual(actor_grade('interaction', dict(cards=cards), [], False), 'unknown')


if __name__ == '__main__':
    unittest.main()
