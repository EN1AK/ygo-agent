"""Core-emitted menus must not be interpreted as physical deck positions."""
import unittest

from ygoenv.ygopro import ygopro_ygoenv as native


def menu_cards(frame):
    offset = 6 if frame[0] == 15 else 7
    count = frame[offset - 1]
    rows = []
    for pool in range(1 if frame[0] == 15 else 2):
        if pool:
            count = frame[offset]
            offset += 1
        for _ in range(count):
            code = int.from_bytes(bytes(frame[offset:offset + 4]), "little")
            rows.append((code, frame[offset + 6]))
            offset += 8
    assert offset == len(frame)
    return rows


class DeckSelectionIdentityTest(unittest.TestCase):
    def check_fixture(self, *, unselect=False, player=0, location=1,
                      opponent=False, verbose=False, duplicate=False,
                      preserved=False, count=4, choices=(0,)):
        result = native._protocol_selection_identity_fixture(
            unselect, player, location, opponent, verbose, duplicate,
            preserved, count, list(choices))
        frame, snapshots, response, accepted, cleared = result
        self.assertTrue(accepted, "same core must accept unchanged response indices")
        self.assertTrue(cleared, "prompt aliases must not leak into next message")
        candidates = menu_cards(frame)
        remaining = list(range(len(candidates)))
        selected = []
        for choice, snapshot in zip(choices, snapshots):
            self.assertEqual(len(snapshot), len(remaining))
            for candidate, row in zip(remaining, snapshot):
                code, sequence = candidates[candidate]
                own_deck = location == 1 and not opponent
                cid = code - 33456000 + 301 if own_deck else (
                    0 if opponent else 900 + sequence)
                ref = 0 if own_deck else sequence + 1
                self.assertEqual(row[:4], [cid, ref, cid, cid])
                self.assertEqual(row[4:6], [ref, ref])
                self.assertEqual(row[6], ref if not unselect else 0)
                if own_deck:
                    self.assertEqual(row[7], 0, "no bogus selected-card scene refs")
                    self.assertEqual(row[8], cid, "forced-action resolver must agree")
                if unselect:
                    self.assertEqual(row[9], int(candidate >= count // 2))
                    self.assertEqual(row[10], candidate)
            selected.append(remaining.pop(choice))
        self.assertEqual(response, [len(selected), *selected])
        return result

    def test_single_selection_all_seats_modes_and_duplicate_cards(self):
        for player in (0, 1):
            for duplicate in (False, True):
                for choice in range(4):
                    quiet = self.check_fixture(player=player, duplicate=duplicate,
                                               choices=(choice,))
                    verbose = self.check_fixture(player=player, duplicate=duplicate,
                                                 choices=(choice,), verbose=True)
                    self.assertEqual(quiet, verbose)

    def test_staged_multi_selection_keeps_prompt_identity_and_history(self):
        for player in (0, 1):
            self.check_fixture(player=player, choices=(2, 0, 1))
            self.check_fixture(player=player, duplicate=True, choices=(3, 1, 0))

    def test_forced_single_choice(self):
        for player in (0, 1):
            self.check_fixture(player=player, count=1)

    def test_select_unselect_both_pools(self):
        for player in (0, 1):
            for duplicate in (False, True):
                for choice in range(4):
                    quiet = self.check_fixture(unselect=True, player=player,
                        duplicate=duplicate, choices=(choice,))
                    verbose = self.check_fixture(unselect=True, player=player,
                        duplicate=duplicate, choices=(choice,), verbose=True)
                    self.assertEqual(quiet, verbose)

    def test_preserved_sequence_does_not_require_guessing_core_flag(self):
        self.check_fixture(preserved=True)
        self.check_fixture(unselect=True, preserved=True, choices=(3,))

    def test_non_deck_and_opponent_visibility_unchanged(self):
        for unselect in (False, True):
            self.check_fixture(location=2, unselect=unselect)
            self.check_fixture(opponent=True, unselect=unselect)


if __name__ == "__main__":
    unittest.main()
