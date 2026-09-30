import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import flax.serialization
import jax.numpy as jnp
import numpy as np

from ygoai.rl.candidate_q_checkpoint import (
    QCheckpointContext, import_ppo_actor_weights, load_q_checkpoint,
    validate_q_checkpoint, write_q_checkpoint,
)
from ygoai.rl.checkpoint_compat import (
    CheckpointCompatibilityError, write_checkpoint_metadata,
)


class CandidateQCheckpointTest(unittest.TestCase):
    def context(self):
        return QCheckpointContext(
            mode="shadow_observation",
            critic_input_schema="actor-observation-v1",
            observation_schema="structured-lite-v1",
            reward_convention="two-seat-zero-sum-v1",
            actor_architecture={"channels": 32},
            critic_architecture={"channels": 16, "heads": 2},
            code_list_hash="codes", semantic_table_hash="semantics",
            capacities={"max_options": 128},
            training_context={"corpus_hash": "corpus"},
            source_commit="10e3469", native_sha256="native",
        )

    def test_matching_resume_and_mismatch_rejection(self):
        template = {
            "actor_variables": {"params": {"weight": jnp.array([1., 2.])}},
            "actor_optimizer_state": {"step": jnp.array(7)},
            "critic_variables": {"params": {"weight": jnp.array([3., 4.])}},
            "critic_optimizer_state": {"step": jnp.array(9)},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.candidate_q.flax_model"
            context = self.context()
            write_q_checkpoint(path, context=context, **template)
            loaded = load_q_checkpoint(path, expected=context, template=template)
            for name in template:
                self.assertEqual(
                    flax.serialization.to_bytes(loaded[name]),
                    flax.serialization.to_bytes(template[name]),
                )
            for mismatch in (
                replace(context, mode="qboost_observation"),
                replace(context, critic_input_schema="central-truth-v1"),
                replace(context, observation_schema="legacy-v2"),
                replace(context, reward_convention="acting-seat-v0"),
                replace(context, training_context={"corpus_hash": "other"}),
            ):
                with self.subTest(mismatch=mismatch):
                    with self.assertRaises(CheckpointCompatibilityError):
                        validate_q_checkpoint(path, mismatch)
            path.write_bytes(path.read_bytes() + b"corruption")
            with self.assertRaisesRegex(CheckpointCompatibilityError, "hash mismatch"):
                validate_q_checkpoint(path, context)

    def test_old_ppo_actor_import_is_explicit_and_legacy_reader_still_loads(self):
        actor = {"params": {"weight": jnp.array([0.25, -0.5])}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.flax_model"
            path.write_bytes(flax.serialization.to_bytes(actor))
            write_checkpoint_metadata(
                path, observation_schema="structured-lite-v1",
                model_args={"channels": 32}, semantic_table_hash="semantics",
                code_list_hash="codes", capacities={"max_options": 128},
                training_context={"corpus_hash": "corpus"},
            )
            imported = import_ppo_actor_weights(
                path, actor_template=actor,
                observation_schema="structured-lite-v1",
                actor_architecture={"channels": 32},
                code_list_hash="codes", semantic_table_hash="semantics",
                capacities={"max_options": 128},
                training_context={"corpus_hash": "corpus"},
            )
            direct_old_reader = flax.serialization.from_bytes(actor, path.read_bytes())
            np.testing.assert_array_equal(imported["params"]["weight"], actor["params"]["weight"])
            self.assertEqual(
                flax.serialization.to_bytes(imported),
                flax.serialization.to_bytes(direct_old_reader),
            )


if __name__ == "__main__":
    unittest.main()
