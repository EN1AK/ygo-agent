import importlib.util
import sys
import unittest
from pathlib import Path


SCHEMA = Path(__file__).resolve().parents[1] / "ygoai" / "rl" / "observation_schema.py"
SPEC = importlib.util.spec_from_file_location("observation_schema", SCHEMA)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ModelInputDomainTest(unittest.TestCase):
    def test_legacy_action_embedding_domains_are_pinned(self):
        domains = MODULE.model_input_domains()["actions_"]
        self.assertEqual(
            [domains[name] for name in (
                "prompt_type", "act", "finish", "effect", "phase",
                "position", "number", "place", "attribute",
            )],
            [30, 10, 3, 256, 4, 9, 13, 31, 10],
        )

    def test_card_embedding_domains_are_pinned(self):
        domains = MODULE.model_input_domains()["cards_"]
        self.assertEqual(domains["location"], 9)
        self.assertEqual(domains["sequence"], 76)
        self.assertEqual(domains["race"], 27)
        self.assertEqual(domains["level_0_to_12_or_13_plus"], 14)
        self.assertEqual(domains["type_bits"], 2)

    def test_callers_cannot_mutate_the_contract(self):
        copy = MODULE.model_input_domains()
        copy["actions_"]["prompt_type"] = 1
        self.assertEqual(MODULE.model_input_domains()["actions_"]["prompt_type"], 30)

    def test_public_event_actor_includes_padding_self_and_opponent(self):
        self.assertEqual(
            MODULE.model_input_domains()["structured"]["actor_relative"], 3
        )


if __name__ == "__main__":
    unittest.main()
