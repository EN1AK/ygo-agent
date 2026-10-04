"""Requires an isolated native built with YGO_CHAIN_EVENT_PROVENANCE_V2."""
import unittest
from ygoenv.ygopro import ygopro_ygoenv as native


def activation(code, actor, link, location=4):
    return [70, *int(code).to_bytes(4, "little"), actor, location, 0, 1,
            actor, location, 0, 0, 0, 0, 0, link]


class ChainProvenanceTest(unittest.TestCase):
    def test_version_is_explicit(self):
        self.assertEqual(native.chain_event_provenance_version, "chain-source-by-link-v2")

    def test_two_links_retain_source_after_reverse_resolution(self):
        rows = native._chain_provenance_fixture([
            activation(97268402, 1, 1, 2), activation(26077387, 0, 2),
            [72, 2], [73, 2], [72, 1], [73, 1]])
        self.assertEqual(rows[-4:], [[10, 2, 26077387, 0, 0], [11, 2, 26077387, 0, 0],
                                    [10, 1, 97268402, 1, 0], [11, 1, 97268402, 1, 0]])

    def test_three_links_and_earlier_negation_use_message_link(self):
        rows = native._chain_provenance_fixture([
            activation(11, 0, 1), activation(22, 1, 2), activation(33, 0, 3),
            [75, 1], [76, 2], [73, 3], [73, 2], [73, 1]])
        self.assertEqual(rows[-5:], [[3, 1, 11, 0, 1], [3, 2, 22, 1, 2],
                                    [11, 3, 33, 0, 0], [11, 2, 22, 1, 0], [11, 1, 11, 0, 0]])

    def test_chain_end_clears_source(self):
        rows = native._chain_provenance_fixture([activation(11, 0, 1), [74], [72, 1]])
        self.assertEqual(rows[-1], [10, 1, 0, 255, 0])

    def test_reset_helper_clears_source(self):
        rows = native._chain_provenance_fixture([activation(11, 0, 1), [0], [73, 1]])
        self.assertEqual(rows[-1], [11, 1, 0, 255, 0])

    def test_new_chain_cannot_reuse_previous_second_link(self):
        rows = native._chain_provenance_fixture([
            activation(11, 0, 1), activation(22, 1, 2), [74],
            activation(33, 1, 1), [72, 2], [72, 1]])
        self.assertEqual(rows[-2:], [[10, 2, 0, 255, 0], [10, 1, 33, 1, 0]])


if __name__ == "__main__":
    unittest.main()
