import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import flax.serialization
import jax.numpy as jnp
import numpy as np
import optax

from ygoai.rl.candidate_q_checkpoint import (
    QCheckpointContext, export_ppo_actor_checkpoint, import_ppo_actor_weights,
    load_q_checkpoint, q_metadata_path, validate_q_checkpoint, write_q_checkpoint,
)
from ygoai.rl.checkpoint_compat import (
    CheckpointCompatibilityError, load_checkpoint_metadata, write_checkpoint_metadata,
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
                replace(context, actor_architecture={"channels": 64}),
                replace(context, critic_architecture={"channels": 64}),
                replace(context, capacities={"max_options": 64}),
                replace(context, code_list_hash="other"),
                replace(context, semantic_table_hash=None),
                replace(context, source_commit="other"),
                replace(context, native_sha256="other"),
            ):
                with self.subTest(mismatch=mismatch):
                    with self.assertRaises(CheckpointCompatibilityError):
                        validate_q_checkpoint(path, mismatch)
            path.write_bytes(path.read_bytes() + b"corruption")
            with self.assertRaisesRegex(CheckpointCompatibilityError, "hash mismatch"):
                validate_q_checkpoint(path, context)

    def state(self):
        return {
            "actor_variables": {"params": {"weight": jnp.array([0.25, -0.5])}},
            "actor_optimizer_state": {"step": jnp.array(7)},
            "critic_variables": {"params": {"private_weight": jnp.array([23., 42.])}},
            "critic_optimizer_state": {"step": jnp.array(9)},
        }

    def test_actor_export_is_byte_identical_for_every_q_mode(self):
        for mode in ("shadow_observation", "qboost_observation", "vrpo_centralized"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                context = replace(self.context(), mode=mode)
                state = self.state()
                source = Path(directory) / "pilot.q"
                output = Path(directory) / "actor.flax_model"
                write_q_checkpoint(source, context=context, **state)
                export_ppo_actor_checkpoint(source, output, expected=context, template=state)
                self.assertEqual(output.read_bytes(), flax.serialization.to_bytes(state["actor_variables"]))
                metadata = load_checkpoint_metadata(output)
                self.assertEqual(metadata["model_architecture"], context.actor_architecture)
                self.assertTrue(metadata["migrated_from"].startswith("candidate-q:"))
                self.assertIsNone(metadata["runtime_state"])
                self.assertFalse(q_metadata_path(output).exists())
                imported = import_ppo_actor_weights(
                    output, actor_template=state["actor_variables"],
                    observation_schema=context.observation_schema,
                    actor_architecture=context.actor_architecture,
                    code_list_hash=context.code_list_hash,
                    semantic_table_hash=context.semantic_table_hash,
                    capacities=context.capacities, training_context=context.training_context,
                )
                np.testing.assert_array_equal(
                    imported["params"]["weight"], state["actor_variables"]["params"]["weight"])
                with self.assertRaises(FileExistsError):
                    export_ppo_actor_checkpoint(source, output, expected=context, template=state)

    def test_missing_malformed_metadata_and_wrong_template_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.q"
            state = self.state()
            context = replace(self.context(), semantic_table_hash=None)
            with self.assertRaisesRegex(CheckpointCompatibilityError, "missing"):
                load_q_checkpoint(path, expected=context, template=state)
            write_q_checkpoint(path, context=context, **state)
            sidecar = q_metadata_path(path)
            original_metadata = sidecar.read_text()
            missing = json.loads(original_metadata)
            del missing["context"]["semantic_table_hash"]
            for bad in ("not JSON", "[]", json.dumps(missing)):
                sidecar.write_text(bad)
                with self.subTest(metadata=bad):
                    with self.assertRaises(CheckpointCompatibilityError):
                        load_q_checkpoint(path, expected=context, template=state)
            sidecar.write_text(original_metadata)
            with self.assertRaisesRegex(CheckpointCompatibilityError, "four-state"):
                load_q_checkpoint(path, expected=context, template={"actor_variables": state["actor_variables"]})
            for wrong in (jnp.zeros((3,), dtype=jnp.float32), jnp.zeros((2,), dtype=jnp.int32)):
                template = self.state()
                template["critic_variables"]["params"]["private_weight"] = wrong
                with self.assertRaisesRegex(CheckpointCompatibilityError, "shape/dtype"):
                    load_q_checkpoint(path, expected=context, template=template)
            output = Path(directory) / "refused.flax_model"
            with self.assertRaises(CheckpointCompatibilityError):
                export_ppo_actor_checkpoint(
                    path, output, expected=replace(context, mode="vrpo_centralized"), template=state)
            self.assertFalse(output.exists())

    def test_optax_resume_next_update_matches_uninterrupted(self):
        state = self.state()
        optimizer = optax.adam(1e-3)
        for prefix in ("actor", "critic"):
            params = state[f"{prefix}_variables"]["params"]
            state[f"{prefix}_optimizer_state"] = optimizer.init(params)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.q"
            # Advance both Adam states so this tests moments/counters, not just
            # reinitialization masquerading as a successful resume.
            for prefix in ("actor", "critic"):
                params = state[f"{prefix}_variables"]["params"]
                _, advanced = optimizer.update(params, state[f"{prefix}_optimizer_state"], params)
                state[f"{prefix}_optimizer_state"] = advanced
            write_q_checkpoint(path, context=self.context(), **state)
            restored = load_q_checkpoint(path, expected=self.context(), template=state)
            for prefix in ("actor", "critic"):
                params = state[f"{prefix}_variables"]["params"]
                expected = optimizer.update(params, state[f"{prefix}_optimizer_state"], params)
                actual = optimizer.update(params, restored[f"{prefix}_optimizer_state"], params)
                self.assertEqual(flax.serialization.to_bytes(expected), flax.serialization.to_bytes(actual))

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
