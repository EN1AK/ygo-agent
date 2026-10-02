import unittest

from scripts.eval_vrpo_baseline import parse_episodes
from ygoai.rl.match_report import summarize_matches


def line(number, reward, invalid=0, reason=1):
    return (f"Episode {number}: length=12, reward={reward}, "
            f"win={int(float(reward) > 0)}, reason=1, "
            f"invalid_game={invalid}, termination_reason={reason}\n")


class BaselineReportTest(unittest.TestCase):
    def test_natural_outcomes_and_timeouts_are_distinct(self):
        rows = parse_episodes(line(1, 1) + line(2, -1) + line(3, 0)
                              + line(4, 3, reason=2) + line(5, 1, invalid=1))
        report = summarize_matches(rows)
        self.assertEqual(report["valid_games"], 3)
        self.assertEqual(report["invalid_games"], 2)
        self.assertEqual([report[k] for k in ("wins", "losses", "draws")], [1, 1, 1])

    def test_nonfinite_and_inconsistent_records_fail_closed(self):
        for text in (line(1, "nan"), line(1, "inf"), line(2, 1),
                     line(1, 1) * 2, line(1, 1).replace("win=1", "win=0")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_episodes(text)


if __name__ == "__main__":
    unittest.main()
