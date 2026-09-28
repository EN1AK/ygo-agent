import tempfile
import unittest
from pathlib import Path

from ygoai.rl.checkpoint_compat import (
    CheckpointCompatibilityError,
    validate_checkpoint_compatibility,
    write_checkpoint_metadata,
)


class CheckpointTrainingContextTest(unittest.TestCase):
    def test_cross_manifest_context_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "fixture.flax_model"
            checkpoint.write_bytes(b"fixture")
            context = {"corpus_hash": "a", "cluster_hash": "b", "curriculum_hash": "c"}
            write_checkpoint_metadata(
                checkpoint, observation_schema="structured-lite-v1",
                model_args={"observation_schema": "structured-lite-v1"},
                semantic_table_hash="semantic", code_list_hash="codes",
                capacities={"max_cards": 80}, training_context=context,
            )
            validate_checkpoint_compatibility(
                checkpoint, observation_schema="structured-lite-v1",
                training_context=context,
            )
            with self.assertRaises(CheckpointCompatibilityError):
                validate_checkpoint_compatibility(
                    checkpoint, observation_schema="structured-lite-v1",
                    training_context={**context, "cluster_hash": "different"},
                )


if __name__ == "__main__":
    unittest.main()
