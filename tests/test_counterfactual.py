import math

from ygoai.rl.counterfactual import (
    DecisionPoint, aggregate_samples, distillation_loss, evaluate_decision,
    mine_errors, preference_loss, RolloutSample,
)


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
    assert math.isclose(sum(result.soft_policy_target), 1.0)
    assert result.soft_policy_target[1] > result.soft_policy_target[0]
    assert {p.rejected for p in result.preferences} == {0, 2}
    assert mine_errors([result], min_regret=0.5) == [result]
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
