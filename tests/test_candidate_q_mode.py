import unittest
from dataclasses import asdict

import flax
import jax
import jax.numpy as jnp
import numpy as np

from scripts.cleanba import (
    Args, CandidateQArgs, advantage_fn, create_agent, parse_training_args,
    validate_candidate_q_mode,
)
from ygoai.rl.jax.agent import ModelArgs
from ygoai.rl.observation_schema import tensor_contract


class CandidateQModeTest(unittest.TestCase):
    def test_unflagged_parser_keeps_legacy_arguments(self):
        parsed = parse_training_args([])
        self.assertIs(type(parsed), Args)
        self.assertEqual(asdict(parsed), asdict(Args()))
        validate_candidate_q_mode(parsed)

    def test_explicit_modes_fail_closed_until_wired(self):
        for mode in ("qboost_observation", "vrpo_centralized"):
            with self.subTest(mode=mode):
                parsed = parse_training_args(["--q-training-mode", mode])
                self.assertIs(type(parsed), CandidateQArgs)
                if mode != "shadow_observation":
                    parsed.upgo = False
                with self.assertRaisesRegex(NotImplementedError, "not wired"):
                    validate_candidate_q_mode(parsed)

    def test_shadow_single_device_selfplay_gate(self):
        args = CandidateQArgs(q_training_mode='shadow_observation', learner_device_ids=[0])
        validate_candidate_q_mode(args)
        args.collect_steps = args.num_steps * 2
        with self.assertRaisesRegex(ValueError, 'collect_steps'):
            validate_candidate_q_mode(args)

    def test_incompatible_flags_rejected(self):
        parsed = CandidateQArgs(q_training_mode="qboost_observation")
        with self.assertRaisesRegex(ValueError, "upgo"):
            validate_candidate_q_mode(parsed)
        parsed.upgo = False
        parsed.value = "vtrace"
        with self.assertRaisesRegex(ValueError, "value must be gae"):
            validate_candidate_q_mode(parsed)

    def test_unflagged_inference_gae_and_checkpoint_bytes(self):
        legacy, parsed = Args(), parse_training_args([])
        model_args = ModelArgs(
            num_layers=1, num_channels=32, rnn_channels=32, rnn_type="none",
            film=False, noam=False, observation_schema="structured-lite-v1")
        for args in (legacy, parsed):
            args.m1 = model_args
            args.num_embeddings = 8
        specs = tensor_contract(
            "structured-lite-v1", max_cards=4, max_options=4,
            history_actions=4, public_events=4, group_references=2)
        obs = {
            name: jnp.zeros((2,) + spec.shape, dtype=np.dtype(spec.dtype))
            for name, spec in specs.items()
        }
        obs["actions_"] = obs["actions_"].at[:, :2, 3].set(1)
        key = jax.random.PRNGKey(711)
        outputs = []
        checkpoint_bytes = []
        for args in (legacy, parsed):
            agent = create_agent(args)
            state = agent.init_rnn_state(2)
            variables = agent.init(key, obs, state)
            outputs.append(agent.apply(variables, obs, state)[1])
            checkpoint_bytes.append(flax.serialization.to_bytes(variables))
        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertEqual(checkpoint_bytes[0], checkpoint_bytes[1])

        next_v = jnp.array([0.])
        values = jnp.zeros((2, 1))
        rewards = jnp.array([[0.], [1.]])
        next_dones = jnp.array([[False], [True]])
        mains = jnp.array([[True], [True]])
        old_targets = advantage_fn(legacy, next_v, values, rewards, next_dones, mains)
        new_targets = advantage_fn(parsed, next_v, values, rewards, next_dones, mains)
        jax.tree.map(np.testing.assert_array_equal, old_targets, new_targets)


if __name__ == "__main__":
    unittest.main()
