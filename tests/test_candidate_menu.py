import unittest
from dataclasses import replace

import numpy as np

from ygoai.rl.candidate_menu import (
    CandidateMenuError, capture_candidate_menu, capture_candidate_menu_batch,
    verify_candidate_menu, verify_candidate_menu_batch,
)


def menu_fixture(count, capacity=4):
    obs = {
        "global_": np.zeros((23,), dtype=np.uint8),
        "selection_": np.zeros((14,), dtype=np.uint8),
        "actions_": np.zeros((capacity, 12), dtype=np.uint8),
        "action_features_": np.zeros((capacity, 16), dtype=np.uint8),
        "action_single_refs_": np.zeros((capacity, 4, 3), dtype=np.uint16),
        "action_group_refs_": np.zeros((capacity, 4, 8, 2), dtype=np.uint16),
        "action_group_mask_": np.zeros((capacity, 4, 8), dtype=np.uint8),
    }
    for slot in range(count):
        obs["actions_"][slot, 3] = 3  # Actor-visible prompt marker.
        obs["action_features_"][slot, 1] = slot + 1
        obs["action_single_refs_"][slot, 0, 1] = slot + 1
    return obs


class CandidateMenuTest(unittest.TestCase):
    def test_staged_multiselect_and_reordered_slots(self):
        obs = menu_fixture(3)
        obs["selection_"][4] = 3
        obs["action_group_mask_"][1, 3, 0] = 1
        obs["action_group_refs_"][1, 3, 0, 0] = 2
        captured = capture_candidate_menu(obs, 3, 1)
        self.assertEqual(captured.num_options, 3)
        self.assertEqual(captured.chosen_index, 1)
        np.testing.assert_array_equal(captured.valid_mask, [True, True, True, False])
        verify_candidate_menu(captured, obs, 3, 1)
        reordered = {name: value.copy() for name, value in obs.items()}
        for name in (
            "actions_", "action_features_", "action_single_refs_",
            "action_group_refs_", "action_group_mask_",
        ):
            reordered[name][[0, 1]] = reordered[name][[1, 0]]
        with self.assertRaises(CandidateMenuError):
            verify_candidate_menu(captured, reordered, 3, 1)
        with self.assertRaises(CandidateMenuError):
            verify_candidate_menu(captured, obs, 3, 2)

    def test_chain_forced_and_capacity_boundary(self):
        chain = menu_fixture(2)
        chain["selection_"][2] = 1
        chain["action_features_"][0, 3] = 7
        chain["action_features_"][1, 3] = 9
        self.assertNotEqual(
            capture_candidate_menu(chain, 2, 0).action_digests[0],
            capture_candidate_menu(chain, 2, 0).action_digests[1])
        forced = menu_fixture(1)
        captured = capture_candidate_menu(forced, 1, 0)
        np.testing.assert_array_equal(captured.valid_mask, [True, False, False, False])
        full = menu_fixture(4)
        self.assertEqual(capture_candidate_menu(full, 4, 3).num_options, 4)
        with self.assertRaisesRegex(CandidateMenuError, "outside"):
            capture_candidate_menu(full, 5, 0)
        with self.assertRaisesRegex(CandidateMenuError, "outside"):
            capture_candidate_menu(full, 4, 4)

    def test_same_features_keep_distinct_slots_but_duplicate_ids_fail(self):
        duplicate = menu_fixture(2)
        for name in (
            "actions_", "action_features_", "action_single_refs_",
            "action_group_refs_", "action_group_mask_",
        ):
            duplicate[name][1] = duplicate[name][0]
        captured = capture_candidate_menu(duplicate, 2, 0)
        self.assertEqual(len(set(captured.action_digests)), 2)
        verify_candidate_menu(captured, duplicate, 2, 0)
        corrupted = replace(captured, action_digests=(captured.action_digests[0],) * 2)
        with self.assertRaises(CandidateMenuError):
            verify_candidate_menu(corrupted, duplicate, 2, 0)
        with self.assertRaises(CandidateMenuError):
            verify_candidate_menu(captured, duplicate, 2, 1)
        bad_mask = menu_fixture(2)
        bad_mask["actions_"][1, 3] = 0
        with self.assertRaisesRegex(CandidateMenuError, "mask"):
            capture_candidate_menu(bad_mask, 2, 0)

    def test_batched_transition_identity(self):
        first = menu_fixture(3)
        second = menu_fixture(1)
        observations = {
            name: np.stack((first[name], second[name])) for name in first
        }
        counts = np.array([3, 1], dtype=np.int32)
        selected = np.array([2, 0], dtype=np.int32)
        captured = capture_candidate_menu_batch(observations, counts, selected)
        self.assertEqual(captured.digest.shape, (2, 32))
        np.testing.assert_array_equal(captured.num_options, counts)
        np.testing.assert_array_equal(captured.chosen_index, selected)
        np.testing.assert_array_equal(captured.valid_mask[1], [True, False, False, False])
        verify_candidate_menu_batch(captured, observations, counts, selected)
        changed = {name: value.copy() for name, value in observations.items()}
        changed["action_features_"][0, [0, 1]] = changed["action_features_"][0, [1, 0]]
        with self.assertRaisesRegex(CandidateMenuError, "digest"):
            verify_candidate_menu_batch(captured, changed, counts, selected)
        with self.assertRaisesRegex(CandidateMenuError, "chosen_index"):
            verify_candidate_menu_batch(captured, observations, counts, [1, 0])
        duplicate = {name: value.copy() for name, value in observations.items()}
        for name in (
            "actions_", "action_features_", "action_single_refs_",
            "action_group_refs_", "action_group_mask_",
        ):
            duplicate[name][0, 1] = duplicate[name][0, 0]
        copies = capture_candidate_menu_batch(duplicate, counts, selected)
        verify_candidate_menu_batch(copies, duplicate, counts, selected)
        # Replacing a formerly distinct feature row is still detected.
        with self.assertRaisesRegex(CandidateMenuError, 'digest'):
            verify_candidate_menu_batch(captured, duplicate, counts, selected)


if __name__ == "__main__":
    unittest.main()
