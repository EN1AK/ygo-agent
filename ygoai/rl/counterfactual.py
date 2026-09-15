"""Counterfactual self-play analysis and distillation targets.

The module deliberately separates engine state restoration from statistics.  A
slow replay-based restorer can therefore be used today and replaced by a native
ocgcore snapshot implementation without changing trace or training artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np


SCHEMA = "ygo-counterfactual-v1"


class SnapshotBackend(Protocol):
    """Minimal engine contract needed by counterfactual rollouts."""

    def restore(self, snapshot: Any, belief: Any) -> Any: ...

    def apply_action(self, state: Any, action: int) -> Any: ...

    def rollout(self, state: Any, seed: int) -> float: ...

    def close(self, state: Any) -> None: ...


class BeliefSampler(Protocol):
    def sample(self, snapshot: Any, count: int, seed: int) -> Sequence[Any]: ...


@dataclass(frozen=True)
class DecisionPoint:
    trace_id: str
    decision_id: str
    step: int
    player: int
    legal_actions: tuple[int, ...]
    selected_action: int
    policy_logits: tuple[float, ...]
    state_value: float
    snapshot: Any
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.legal_actions:
            raise ValueError("a decision point must contain legal actions")
        if self.selected_action not in self.legal_actions:
            raise ValueError("selected_action is not legal")
        if len(self.policy_logits) != len(self.legal_actions):
            raise ValueError("policy logits must align with legal actions")


@dataclass(frozen=True)
class RolloutSample:
    action: int
    particle: int
    rollout_seed: int
    value: float


@dataclass(frozen=True)
class ActionEstimate:
    action: int
    q: float
    stderr: float
    samples: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class Preference:
    preferred: int
    rejected: int
    margin: float
    weight: float


@dataclass(frozen=True)
class CounterfactualResult:
    schema: str
    trace_id: str
    decision_id: str
    selected_action: int
    best_action: int
    regret: float
    regret_stderr: float
    regret_lower95: float
    action_estimates: tuple[ActionEstimate, ...]
    soft_policy_target: tuple[float, ...]
    preferences: tuple[Preference, ...]
    metadata: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def stable_seed(*parts: Any) -> int:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return int.from_bytes(hashlib.sha256(blob.encode()).digest()[:8], "little")


def observation_digest(observation: Mapping[str, Any]) -> str:
    """Stable digest for proving a restored engine observation is identical."""
    digest = hashlib.sha256()
    for key in sorted(observation):
        value = np.asarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def unpack_step(result: Sequence[Any]) -> tuple[Any, Any, Any, Any]:
    """Normalize Gym's 4-tuple and Gymnasium's 5-tuple step APIs."""
    if len(result) == 4:
        observation, reward, done, info = result
        return observation, reward, done, info
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, np.logical_or(terminated, truncated), info
    raise ValueError(f"unsupported environment step tuple of length {len(result)}")


def probabilities(logits: Sequence[float]) -> tuple[float, ...]:
    if not logits:
        return ()
    top = max(logits)
    weights = [math.exp(float(v) - top) for v in logits]
    total = sum(weights)
    return tuple(v / total for v in weights)


def decision_priority(point: DecisionPoint) -> float:
    """Rank points where the policy is uncertain or the value is fragile."""
    ps = probabilities(point.policy_logits)
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in ps)
    norm_entropy = entropy / math.log(len(ps)) if len(ps) > 1 else 0.0
    ordered = sorted(ps, reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
    fragility = 1.0 - min(1.0, abs(float(point.state_value)))
    return 0.55 * norm_entropy + 0.30 * (1.0 - margin) + 0.15 * fragility


def select_decision_points(
    points: Iterable[DecisionPoint], *, limit: int | None = None,
    min_actions: int = 2, min_priority: float = 0.0,
) -> list[DecisionPoint]:
    ranked = [p for p in points if len(p.legal_actions) >= min_actions]
    ranked = [p for p in ranked if decision_priority(p) >= min_priority]
    ranked.sort(key=lambda p: (-decision_priority(p), p.trace_id, p.step))
    return ranked if limit is None else ranked[:limit]


def _estimate(action: int, values: Sequence[float]) -> ActionEstimate:
    if not values:
        raise ValueError(f"action {action} has no valid rollout samples")
    mean = sum(values) / len(values)
    if len(values) == 1:
        stderr = 0.0
    else:
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        stderr = math.sqrt(variance / len(values))
    return ActionEstimate(action, mean, stderr, len(values), tuple(values))


def aggregate_samples(
    point: DecisionPoint, samples: Iterable[RolloutSample], *,
    temperature: float = 0.25, preference_margin: float = 0.05,
) -> CounterfactualResult:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    grouped: dict[int, list[float]] = {a: [] for a in point.legal_actions}
    sample_rows = list(samples)
    for row in sample_rows:
        if row.action not in grouped:
            raise ValueError(f"rollout contains unknown action {row.action}")
        if not math.isfinite(row.value):
            raise ValueError("rollout values must be finite")
        grouped[row.action].append(float(row.value))
    estimates = tuple(_estimate(a, grouped[a]) for a in point.legal_actions)
    q_by_action = {row.action: row.q for row in estimates}
    best_action = max(point.legal_actions, key=lambda a: (q_by_action[a], -a))
    regret = max(0.0, q_by_action[best_action] - q_by_action[point.selected_action])
    by_action_seed = {
        action: {(row.particle, row.rollout_seed): row.value
                 for row in sample_rows if row.action == action}
        for action in point.legal_actions
    }
    paired_keys = sorted(set(by_action_seed[best_action]).intersection(
        by_action_seed[point.selected_action]))
    differences = [by_action_seed[best_action][key]
                   - by_action_seed[point.selected_action][key]
                   for key in paired_keys]
    if len(differences) <= 1 or best_action == point.selected_action:
        regret_stderr = 0.0
    else:
        mean_difference = sum(differences) / len(differences)
        variance = sum((v - mean_difference) ** 2 for v in differences) \
            / (len(differences) - 1)
        regret_stderr = math.sqrt(variance / len(differences))
    regret_lower95 = max(0.0, regret - 1.96 * regret_stderr)
    target = probabilities([q_by_action[a] / temperature for a in point.legal_actions])
    prefs = []
    for rejected in point.legal_actions:
        margin = q_by_action[best_action] - q_by_action[rejected]
        if rejected != best_action and margin >= preference_margin:
            prefs.append(Preference(best_action, rejected, margin, margin))
    return CounterfactualResult(
        schema=SCHEMA, trace_id=point.trace_id,
        decision_id=point.decision_id, selected_action=point.selected_action,
        best_action=best_action, regret=regret,
        regret_stderr=regret_stderr, regret_lower95=regret_lower95,
        action_estimates=estimates,
        soft_policy_target=target, preferences=tuple(prefs),
        metadata={"temperature": temperature,
                  "preference_margin": preference_margin,
                  "particles": len({r.particle for r in sample_rows}),
                  "rollout_seeds": len({r.rollout_seed for r in sample_rows}),
                  "paired_samples": len(paired_keys)},
    )


def evaluate_decision(
    point: DecisionPoint, backend: SnapshotBackend, belief_sampler: BeliefSampler,
    *, particles: int, rollout_seeds: int, seed: int,
    temperature: float = 0.25, preference_margin: float = 0.05,
) -> CounterfactualResult:
    """Evaluate legal action x belief particle x rollout seed."""
    if particles < 1 or rollout_seeds < 1:
        raise ValueError("particles and rollout_seeds must be positive")
    beliefs = belief_sampler.sample(point.snapshot, particles, seed)
    if len(beliefs) != particles:
        raise ValueError("belief sampler returned the wrong number of particles")
    samples: list[RolloutSample] = []
    for particle_idx, belief in enumerate(beliefs):
        for action in point.legal_actions:
            for rollout_idx in range(rollout_seeds):
                state = backend.restore(point.snapshot, belief)
                try:
                    state = backend.apply_action(state, action)
                    rollout_seed = stable_seed(seed, point.decision_id, particle_idx,
                                               action, rollout_idx)
                    value = float(backend.rollout(state, rollout_seed))
                    samples.append(RolloutSample(action, particle_idx,
                                                 rollout_seed, value))
                finally:
                    backend.close(state)
    return aggregate_samples(point, samples, temperature=temperature,
                             preference_margin=preference_margin)


class ReplaySnapshotBackend:
    """Portable snapshot fallback: rebuild and replay an action prefix.

    ``snapshot`` must contain ``seed`` and ``actions``. ``env_factory`` receives
    ``(seed, belief)`` and returns a fresh environment with ``reset``, ``step``
    and ``close`` methods. This is slower than native rollback but deterministic
    and useful for correctness gates and development.
    """

    def __init__(self, env_factory: Callable[[int, Any], Any],
                 rollout_policy: Callable[[Any, random.Random], int],
                 value_from_terminal: Callable[[Any, int], float],
                 *, max_steps: int = 1000):
        self.env_factory = env_factory
        self.rollout_policy = rollout_policy
        self.value_from_terminal = value_from_terminal
        self.max_steps = max_steps

    def restore(self, snapshot: Mapping[str, Any], belief: Any) -> dict[str, Any]:
        env = self.env_factory(int(snapshot["seed"]), belief)
        obs, info = env.reset()
        for action in snapshot.get("actions", ()):
            obs, reward, done, info = unpack_step(env.step(action))
            if bool(done):
                env.close()
                raise RuntimeError("snapshot action prefix terminated early")
        return {"env": env, "obs": obs, "info": info,
                "perspective": int(snapshot.get("player", 0))}

    def apply_action(self, state: dict[str, Any], action: int) -> dict[str, Any]:
        state["obs"], reward, state["done"], state["info"] = unpack_step(
            state["env"].step(action))
        return state

    def rollout(self, state: dict[str, Any], seed: int) -> float:
        rng = random.Random(seed)
        for _ in range(self.max_steps):
            if bool(state.get("done", False)):
                return float(self.value_from_terminal(state, state["perspective"]))
            action = self.rollout_policy(state, rng)
            self.apply_action(state, action)
        raise RuntimeError("counterfactual rollout exceeded max_steps")

    def close(self, state: Mapping[str, Any]) -> None:
        env = state.get("env")
        if env is not None:
            env.close()


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_decision_points(path: str | Path) -> list[DecisionPoint]:
    points = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            actions = tuple(int(a) for a in row.get(
                "legal_actions", row.get("legal_action_indices", ())))
            points.append(DecisionPoint(
                trace_id=str(row.get("trace_id", Path(path).stem)),
                decision_id=str(row.get("decision_id", f"{Path(path).stem}:{line_no}")),
                step=int(row["step"]), player=int(row["player"]),
                legal_actions=actions, selected_action=int(row["selected_action"]),
                policy_logits=tuple(float(v) for v in row["policy_logits"]),
                state_value=float(row["state_value"]), snapshot=row.get("snapshot"),
                context=row.get("context", {}),
            ))
    return points


def mine_errors(results: Iterable[CounterfactualResult], *,
                min_regret: float = 0.15,
                require_confident: bool = False) -> list[CounterfactualResult]:
    rows = [r for r in results if r.regret >= min_regret
            and r.best_action != r.selected_action
            and (not require_confident or r.regret_lower95 > 0)]
    return sorted(rows, key=lambda r: (-r.regret, r.trace_id, r.decision_id))


def distillation_loss(policy_logits: Sequence[float],
                      soft_policy_target: Sequence[float]) -> float:
    """Cross entropy used to retrain a policy from the soft search target."""
    if len(policy_logits) != len(soft_policy_target) or not policy_logits:
        raise ValueError("logits and target must be non-empty and aligned")
    target_sum = sum(float(v) for v in soft_policy_target)
    if not math.isclose(target_sum, 1.0, abs_tol=1e-6):
        raise ValueError("soft policy target must sum to one")
    top = max(float(v) for v in policy_logits)
    log_z = top + math.log(sum(math.exp(float(v) - top)
                               for v in policy_logits))
    return -sum(float(t) * (float(logit) - log_z)
                for t, logit in zip(soft_policy_target, policy_logits))


def preference_loss(policy_logits: Sequence[float], legal_actions: Sequence[int],
                    preferences: Sequence[Preference]) -> float:
    """Weighted pairwise logistic loss over preferred/rejected actions."""
    if len(policy_logits) != len(legal_actions):
        raise ValueError("logits and legal actions must be aligned")
    if not preferences:
        return 0.0
    logits = {int(a): float(v) for a, v in zip(legal_actions, policy_logits)}
    total_weight = sum(float(p.weight) for p in preferences)
    if total_weight <= 0:
        raise ValueError("preference weights must have positive total")
    total = 0.0
    for pref in preferences:
        if pref.preferred not in logits or pref.rejected not in logits:
            raise ValueError("preference contains an unknown action")
        delta = logits[pref.preferred] - logits[pref.rejected]
        total += pref.weight * math.log1p(math.exp(-abs(delta)))
        total += pref.weight * max(-delta, 0.0)
    return total / total_weight
