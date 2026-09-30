import unittest
from dataclasses import asdict

from scripts.cleanba import (
    Args, CandidateQArgs, parse_training_args, validate_candidate_q_mode,
)


class CandidateQModeTest(unittest.TestCase):
    def test_unflagged_parser_keeps_legacy_arguments(self):
        parsed = parse_training_args([])
        self.assertIs(type(parsed), Args)
        self.assertEqual(asdict(parsed), asdict(Args()))
        validate_candidate_q_mode(parsed)

    def test_explicit_modes_fail_closed_until_wired(self):
        for mode in (
            "shadow_observation", "qboost_observation", "vrpo_centralized",
        ):
            with self.subTest(mode=mode):
                parsed = parse_training_args(["--q-training-mode", mode])
                self.assertIs(type(parsed), CandidateQArgs)
                if mode != "shadow_observation":
                    parsed.upgo = False
                with self.assertRaisesRegex(NotImplementedError, "not wired"):
                    validate_candidate_q_mode(parsed)

    def test_incompatible_flags_rejected(self):
        parsed = CandidateQArgs(q_training_mode="qboost_observation")
        with self.assertRaisesRegex(ValueError, "upgo"):
            validate_candidate_q_mode(parsed)
        parsed.upgo = False
        parsed.value = "vtrace"
        with self.assertRaisesRegex(ValueError, "value must be gae"):
            validate_candidate_q_mode(parsed)


if __name__ == "__main__":
    unittest.main()
