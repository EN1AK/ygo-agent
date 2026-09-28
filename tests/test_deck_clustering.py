import unittest

import numpy as np

from ygoai.deck_clustering import build_family_split, build_tfidf_features, cluster_ready_families
from ygoai.deck_corpus import CorpusError


def row(deck_id, family, main, extra=None, side=None):
    return {"deck": {
        "canonical_id": deck_id, "family_id": family,
        "zones": {"main": main, "extra": extra or {}, "side": side or {}},
    }}


class DeckClusteringTest(unittest.TestCase):
    def test_feature_build_is_deterministic_and_ignores_side(self):
        document = {"decks": [
            row("b", "f2", {"1": 3, "2": 1}, {"9": 1}, {"100": 3}),
            row("a", "f1", {"1": 3, "3": 1}, {"8": 1}, {"101": 3}),
        ]}
        left_matrix, left = build_tfidf_features(document)
        right_matrix, right = build_tfidf_features(document)
        self.assertEqual(left["matrix_hash"], right["matrix_hash"])
        np.testing.assert_array_equal(left_matrix.toarray(), right_matrix.toarray())
        self.assertFalse(any(column.endswith(":100") or column.endswith(":101")
                             for column in left["column_ids"]))
        self.assertEqual(left["row_ids"], ["a", "b"])

    def test_common_card_has_lower_idf(self):
        document = {"decks": [
            row("a", "f1", {"1": 1, "2": 1}),
            row("b", "f2", {"1": 1, "3": 1}),
        ]}
        _, artifact = build_tfidf_features(document)
        idf = dict(zip(artifact["column_ids"], artifact["idf"]))
        self.assertLess(idf["main:1"], idf["main:2"])

    def test_family_split_never_crosses(self):
        clusters = {"clusters": [
            {"families": ["f1", "f2", "f3"]},
            {"families": ["f4", "f5", "f6"]},
        ], "assignments": {
            "a": {"family_id": "f1"}, "b": {"family_id": "f1"},
            "c": {"family_id": "f2"}, "d": {"family_id": "f3"},
            "e": {"family_id": "f4"}, "f": {"family_id": "f5"},
            "g": {"family_id": "f6"},
        }}
        split = build_family_split(clusters)
        self.assertFalse(set(split["train_families"]) & set(split["held_out_families"]))
        self.assertFalse(set(split["train_decks"]) & set(split["held_out_decks"]))

    def test_degenerate_cluster_constraints_reject_candidates(self):
        document = {"decks": [row(chr(97 + index), f"f{index}", {str(index + 1): 40})
                              for index in range(6)]}
        matrix, features = build_tfidf_features(document)
        families = {"families": [
            {"family_id": f"f{index}", "members": [chr(97 + index)]}
            for index in range(6)
        ]}
        with self.assertRaises(CorpusError):
            cluster_ready_families(matrix, features, document, families,
                                   candidate_k=[3], min_families=3,
                                   max_family_fraction=0.3, max_deck_fraction=0.3)


if __name__ == "__main__":
    unittest.main()
