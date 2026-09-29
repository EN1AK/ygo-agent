import unittest

from ygoai.multideck_training import SamplingTelemetry
from ygoai.rl.observation_schema import manifest


class MultiDeckTrainingTest(unittest.TestCase):
    def test_telemetry_reconciles_games_and_decisions(self):
        telemetry = SamplingTelemetry()
        info = {
            "invalid_game": [0], "deck_cluster": [[1, 2]],
            "deck_family": [[3, 4]], "deck_member": [[5, 6]],
        }
        for seat in (0, 1, 0):
            telemetry.decision(seat, info, 0)
        telemetry.game(info, 0)
        report = telemetry.report()
        self.assertEqual(report["completed_games"], 1)
        self.assertEqual(sum(report["deck_counts"].values()), 2)
        self.assertEqual(sum(report["seat_decision_counts"].values()), 3)
        self.assertEqual(sum(report["deck_decision_counts"].values()), 3)

    def test_invalid_game_is_not_counted_as_completed(self):
        telemetry = SamplingTelemetry()
        telemetry.game({"invalid_game": [1], "termination_reason": [2]}, 0)
        report = telemetry.report()
        self.assertEqual(report["invalid_games"], 1)
        self.assertEqual(report["completed_games"], 0)
        self.assertEqual(report["invalid_termination_counts"], {"2": 1})

    def test_private_deck_ids_are_not_policy_inputs(self):
        tensors = manifest("structured-lite-v1")["tensors"]
        self.assertFalse(any("deck" in name or "cluster" in name or "family" in name
                             for name in tensors))


if __name__ == "__main__":
    unittest.main()
