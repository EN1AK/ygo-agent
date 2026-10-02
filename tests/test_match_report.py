import unittest

import numpy as np

from ygoai.rl.match_report import battle_report, summarize_matches, terminal_outcome


class MatchReportTest(unittest.TestCase):
    def test_invalid_positive_reward_cannot_win_and_zero_is_not_a_loss(self):
        self.assertEqual(terminal_outcome(10.0, 1, 2), "invalid")
        self.assertEqual(terminal_outcome(-1.0, 0, 3), "invalid")
        self.assertEqual(terminal_outcome(1.0, 0, 0), "invalid")
        self.assertEqual(terminal_outcome(0.0, 0, 1), "draw")
        with self.assertRaises(ValueError):
            terminal_outcome(float("nan"), 0, 1)
        with self.assertRaises(ValueError):
            terminal_outcome(1.0, 0, 99)

    def test_all_invalid_has_no_win_rate_or_nan_json(self):
        report = summarize_matches([{"outcome": "invalid", "termination_reason": 2}])
        self.assertIsNone(report["win_rate"])
        self.assertIsNone(report["win_rate_wilson95"])
        self.assertIsNone(report["mean_return"])
        self.assertEqual(report["invalid_games"], 1)

    def test_perspective_and_repeated_finished_environment(self):
        class FakeEnv:
            num_envs = 4
            calls = 0

            def reset(self):
                return {}, {"to_play": np.array([0, 1, 0, 1])}

            def step(self, actions):
                self.calls += 1
                # The next-player field changes at termination. Use the actor
                # before step to orient the terminal reward, not this field.
                info = {"to_play": np.array([1, 0, 1, 0]),
                        "r": np.array([1., 1., -1., 9.]),
                        "l": np.array([2, 3, 4, 1000]),
                        "invalid_game": np.array([0, 0, 0, 1]),
                        "termination_reason": np.array([1, 1, 1, 2])}
                done = np.array([True, True, True, self.calls == 2])
                return {}, np.zeros(4), done, info

        def predict(obs, s1, s2, main, done):
            return s1, s2, np.zeros(4, dtype=int)

        result = battle_report(FakeEnv(), 4, predict)
        self.assertEqual([r["outcome"] for r in result["episodes"]],
                         ["win", "loss", "win", "invalid"])
        self.assertEqual(result["attempts"], 4)
        self.assertEqual(result["valid_games"], 3)
        self.assertAlmostEqual(result["win_rate"], 2 / 3)
        self.assertEqual(result["by_seat"]["0"]["win_rate"], 0.5)
        self.assertEqual(result["by_seat"]["1"]["win_rate"], 1.0)

    def test_missing_validity_fields_fail_closed(self):
        class FakeEnv:
            num_envs = 2

            def reset(self):
                return {}, {"to_play": np.array([0, 1])}

            def step(self, actions):
                return {}, np.zeros(2), np.ones(2, dtype=bool), {
                    "to_play": np.array([0, 1]), "r": np.ones(2), "l": np.ones(2)}

        with self.assertRaises(KeyError):
            battle_report(FakeEnv(), 2, lambda *args: (None, None, [0, 0]))


if __name__ == "__main__":
    unittest.main()
