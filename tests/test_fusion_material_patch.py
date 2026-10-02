import unittest

from scripts.patch_fusion_material_search import patch_procedure


class FusionPatchTest(unittest.TestCase):
    def test_unknown_or_already_patched_input_is_never_overwritten(self):
        for value in (b"", b"function Auxiliary.FSelectMixRep() end"):
            with self.assertRaisesRegex(ValueError, "Unknown procedure"):
                patch_procedure(value)


if __name__ == "__main__":
    unittest.main()
