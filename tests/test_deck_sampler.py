import unittest

from ygoai.deck_corpus import CorpusError
from ygoai.deck_sampler import HierarchicalDeckSampler, validate_sampling_manifest


MANIFEST = {
    "anchor": {"deck_id": "elfnote", "reserve_ppm_per_seat": 250000},
    "non_anchor_probability_ppm": 750000,
    "manifest_hash": "fixture",
    "clusters": [
        {"cluster_id": "c0", "cluster_index": 0, "families": [
            {"family_id": "f0", "family_index": 0,
             "decks": [{"deck_id": "d0", "deck_index": 0},
                       {"deck_id": "d1", "deck_index": 1}]},
        ]},
        {"cluster_id": "c1", "cluster_index": 1, "families": [
            {"family_id": "f1", "family_index": 0,
             "decks": [{"deck_id": "d2", "deck_index": 0}]},
            {"family_id": "f2", "family_index": 1,
             "decks": [{"deck_id": "d3", "deck_index": 0}]},
        ]},
    ],
}


class DeckSamplerTest(unittest.TestCase):
    def test_probabilities_normalize(self):
        validate_sampling_manifest(MANIFEST)
        invalid = {**MANIFEST, "non_anchor_probability_ppm": 1}
        with self.assertRaises(CorpusError):
            validate_sampling_manifest(invalid)

    def test_fixed_seed_and_resume_are_exact(self):
        uninterrupted = HierarchicalDeckSampler(MANIFEST, 7)
        expected = [uninterrupted.draw_game() for _ in range(20)]
        interrupted = HierarchicalDeckSampler(MANIFEST, 7)
        prefix = [interrupted.draw_game() for _ in range(8)]
        state = interrupted.state_dict()
        resumed = HierarchicalDeckSampler(MANIFEST, 7)
        resumed.load_state_dict(state)
        suffix = [resumed.draw_game() for _ in range(12)]
        self.assertEqual(expected, prefix + suffix)

    def test_seats_are_independent_draws(self):
        sampler = HierarchicalDeckSampler(MANIFEST, 99)
        games = [sampler.draw_game() for _ in range(100)]
        self.assertTrue(any(left["deck_id"] != right["deck_id"] for left, right in games))


if __name__ == "__main__":
    unittest.main()
