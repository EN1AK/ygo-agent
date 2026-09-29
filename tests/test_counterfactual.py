import math

import numpy as np

from ygoai.rl.counterfactual import (
    DecisionPoint, aggregate_samples, distillation_loss, evaluate_decision,
    mine_errors, observation_digest, preference_loss, RolloutSample, unpack_step,
)
from ygoai.rl.cycle_guard import PolicyCycleGuard, public_state_digest


def point():
    return DecisionPoint("duel-1", "duel-1:7", 7, 0, (0, 1, 2), 0,
                         (2.0, 1.0, 0.0), 0.1, {"seed": 3, "actions": [1]})


def test_q_regret_soft_target_and_preferences():
    rows = []
    values = {0: (0.0, 0.2), 1: (0.7, 0.9), 2: (-0.5, -0.3)}
    for action, outcomes in values.items():
        for particle, value in enumerate(outcomes):
            rows.append(RolloutSample(action, particle, 10 + particle, value))
    result = aggregate_samples(point(), rows, temperature=0.2,
                               preference_margin=0.1)
    assert result.best_action == 1
    assert math.isclose(result.regret, 0.7)
    assert result.regret_lower95 > 0
    assert math.isclose(sum(result.soft_policy_target), 1.0)
    assert result.soft_policy_target[1] > result.soft_policy_target[0]
    assert {p.rejected for p in result.preferences} == {0, 2}
    assert mine_errors([result], min_regret=0.5) == [result]
    assert mine_errors([result], min_regret=0.5,
                       require_confident=True) == [result]
    assert distillation_loss((0.0, 2.0, -1.0),
                             result.soft_policy_target) > 0
    good = preference_loss((0.0, 2.0, -1.0), point().legal_actions,
                           result.preferences)
    bad = preference_loss((2.0, 0.0, -1.0), point().legal_actions,
                          result.preferences)
    assert good < bad


class Beliefs:
    def sample(self, snapshot, count, seed):
        return list(range(count))


class Backend:
    def restore(self, snapshot, belief):
        return {"belief": belief}

    def apply_action(self, state, action):
        state["action"] = action
        return state

    def rollout(self, state, seed):
        return state["action"] + state["belief"] * 0.01

    def close(self, state):
        state["closed"] = True


def test_cartesian_rollout_is_complete_and_deterministic():
    a = evaluate_decision(point(), Backend(), Beliefs(), particles=2,
                          rollout_seeds=3, seed=42)
    b = evaluate_decision(point(), Backend(), Beliefs(), particles=2,
                          rollout_seeds=3, seed=42)
    assert a == b
    assert [e.samples for e in a.action_estimates] == [6, 6, 6]
    assert a.best_action == 2


def test_observation_digest_is_order_independent_and_sensitive():
    a = {"b": [[1, 2]], "a": [3]}
    b = {"a": [3], "b": [[1, 2]]}
    assert observation_digest(a) == observation_digest(b)
    b["b"][0][1] = 4
    assert observation_digest(a) != observation_digest(b)


def test_step_api_normalization():
    assert unpack_step(("obs", 1, False, {})) == ("obs", 1, False, {})
    obs, reward, done, info = unpack_step(
        ("obs", 1, np.asarray([False]), np.asarray([True]), {}))
    assert (obs, reward, info) == ("obs", 1, {})
    assert bool(done[0])


def test_public_state_digest_ignores_history_but_includes_legal_menu():
    obs = {
        "cards_": np.asarray([[[1, 2], [3, 4]]], dtype=np.uint8),
        "actions_": np.asarray([[[1, 0], [2, 0], [9, 9]]], dtype=np.uint8),
        "h_actions_": np.asarray([[[1], [2]]], dtype=np.uint8),
        "public_events_": np.asarray([[[3], [4]]], dtype=np.uint8),
    }
    a = public_state_digest(obs, env_index=0, num_options=2, player=0)
    obs["h_actions_"][0, 0, 0] = 8
    obs["public_events_"][0, 0, 0] = 8
    assert public_state_digest(
        obs, env_index=0, num_options=2, player=0) == a
    obs["actions_"][0, 1, 0] = 7
    assert public_state_digest(
        obs, env_index=0, num_options=2, player=0) != a


def test_cycle_guard_detects_a_b_a_and_keeps_raw_mode_unchanged():
    guard = PolicyCycleGuard(enabled=False, max_cycle_period=4)
    first = guard.select(env_index=0, step=0, state_fingerprint="A",
                         policy_logits=(3.0, 2.0), num_options=2)
    guard.select(env_index=0, step=1, state_fingerprint="B",
                 policy_logits=(0.0, 4.0), num_options=2)
    repeated = guard.select(env_index=0, step=2, state_fingerprint="A",
                            policy_logits=(3.0, 2.0), num_options=2)
    assert not first.detected
    assert repeated.detected and repeated.cycle_period == 2
    assert repeated.raw_action == repeated.selected_action == 0
    assert not repeated.intervened


def test_cycle_guard_does_not_flag_a_distant_revisit():
    guard = PolicyCycleGuard(enabled=True, max_cycle_period=2)
    guard.select(env_index=0, step=1, state_fingerprint="A",
                 policy_logits=(2.0, 1.0), num_options=2)
    distant = guard.select(env_index=0, step=5, state_fingerprint="A",
                           policy_logits=(2.0, 1.0), num_options=2)
    assert not distant.detected
    assert distant.selected_action == distant.raw_action == 0


def test_cycle_guard_uses_next_ranked_action_only_when_enabled():
    guard = PolicyCycleGuard(enabled=True, max_cycle_period=4)
    guard.select(env_index=3, step=4, state_fingerprint="A",
                 policy_logits=(1.0, 5.0, 3.0), num_options=3)
    repeated = guard.select(env_index=3, step=6, state_fingerprint="A",
                            policy_logits=(1.0, 5.0, 3.0), num_options=3)
    assert repeated.detected and repeated.intervened
    assert repeated.raw_action == 1
    assert repeated.selected_action == 2
    repeated_again = guard.select(
        env_index=3, step=8, state_fingerprint="A",
        policy_logits=(1.0, 5.0, 3.0), num_options=3)
    assert repeated_again.selected_action == 0
    guard.reset(3)
    after_reset = guard.select(env_index=3, step=7, state_fingerprint="A",
                               policy_logits=(1.0, 5.0, 3.0), num_options=3)
    assert not after_reset.detected
