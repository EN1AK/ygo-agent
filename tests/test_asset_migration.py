import tempfile
import unittest
from pathlib import Path

import numpy as np

from ygoai.asset_migration import append_code_list, migrate_parameter_tree, verify_semantic_append
from ygoai.deck_corpus import CorpusError


class AssetMigrationTest(unittest.TestCase):
    def test_code_list_append_preserves_prefix_and_sorts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "old.txt"
            source.write_bytes(b"1 0\r\n2 1\r\n")
            scripts = root / "scripts"
            scripts.mkdir()
            (scripts / "c4.lua").write_text("-- fixture")
            output = root / "new.txt"
            append_code_list(source, output, [4, 3], scripts)
            self.assertTrue(output.read_bytes().startswith(source.read_bytes()))
            self.assertEqual(output.read_bytes(), b"1 0\r\n2 1\r\n3 0\r\n4 1\r\n")

    def test_semantic_append_requires_byte_identical_old_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, new = root / "old", root / "new"
            old.mkdir(); new.mkdir()
            for name, width in (("card-semantics.u8", 32), ("effect-tags.u8", 16),
                                ("effect-tag-confidence.u8", 16)):
                (old / name).write_bytes(bytes([1]) * (2 * width))
                (new / name).write_bytes(bytes([1]) * (2 * width) + bytes([2]) * width)
            result = verify_semantic_append(old, new, 2, 3)
            self.assertTrue(all(item["old_bytes_identical"] for item in result.values()))
            (new / "effect-tags.u8").write_bytes(bytes([9]) + (new / "effect-tags.u8").read_bytes()[1:])
            with self.assertRaises(CorpusError):
                verify_semantic_append(old, new, 2, 3)

    def test_embedding_migration_copies_all_old_values_exactly(self):
        source = {
            ("params", "dense", "kernel"): np.arange(6, dtype=np.float32).reshape(2, 3),
            ("params", "Embed_0", "embedding"): np.arange(12, dtype=np.float32).reshape(3, 4),
        }
        destination = {
            ("params", "dense", "kernel"): np.zeros((2, 3), dtype=np.float32),
            ("params", "Embed_0", "embedding"): np.zeros((5, 4), dtype=np.float32),
        }
        left, report = migrate_parameter_tree(source, destination, 3, 5, 77)
        right, _ = migrate_parameter_tree(source, destination, 3, 5, 77)
        np.testing.assert_array_equal(left[("params", "dense", "kernel")], source[("params", "dense", "kernel")])
        np.testing.assert_array_equal(left[("params", "Embed_0", "embedding")][:3],
                                      source[("params", "Embed_0", "embedding")])
        np.testing.assert_array_equal(left[("params", "Embed_0", "embedding")][3:],
                                      right[("params", "Embed_0", "embedding")][3:])
        self.assertEqual(report["embedding_paths"], ["params/Embed_0/embedding"])


if __name__ == "__main__":
    unittest.main()
