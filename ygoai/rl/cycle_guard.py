"""Auditable detection and optional mitigation of deterministic policy loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "ygo-policy-cycle-v1"

# These tensors intentionally describe prior decisions.  Including them would
# make an otherwise identical reversible state look different on every visit.
DEFAULT_IGNORED_KEYS = frozenset(("h_actions_", "public_events_"))


def public_state_digest(
    observation: Mapping[str, Any], *, env_index: int, num_options: int,
    player: int, ignored_keys: Sequence[str] = tuple(DEFAULT_IGNORED_KEYS),
) -> str:
    """Hash one environment's public state and its bounded legal-action menu."""
    ignored = set(ignored_keys)
    digest = hashlib.sha256()
    digest.update(f"player:{int(player)}\noptions:{int(num_options)}\n".encode())
    for key in sorted(observation):
        if key in ignored:
            continue
        batch = np.asarray(observation[key])
        if batch.ndim == 0:
            value = batch
        else:
            if env_index < 0 or env_index >= batch.shape[0]:
                raise IndexError(f"environment index {env_index} outside {key}")
            value = batch[env_index]
        if key == "actions_" or key.startswith("action_"):
            value = value[:num_options]
        value = np.ascontiguousarray(value)
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class CycleDecision:
    schema: str
    env_index: int
    step: int
    state_fingerprint: str
    raw_action: int
    selected_action: int
    detected: bool
    intervened: bool
    cycle_period: int | None
    max_cycle_period: int
    legal_actions: tuple[int, ...]
    policy_logits: tuple[float, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class PolicyCycleGuard:
    """Detect repeated state/action pairs and optionally choose a fallback.

    Detection is active in both raw and guard modes.  Only ``enabled=True`` may
    change an action, and the caller receives the full decision record either
    way so raw and assisted evaluation can never be silently mixed.
    """

    def __init__(self, *, enabled: bool = False, max_cycle_period: int = 8):
        if max_cycle_period < 1:
            raise ValueError("max_cycle_period must be positive")
        self.enabled = bool(enabled)
        self.max_cycle_period = int(max_cycle_period)
        self._last_seen: dict[int, dict[tuple[str, int], int]] = {}
        self._last_selected: dict[int, dict[str, int]] = {}
        self._blocked: dict[int, dict[str, set[int]]] = {}

    def reset(self, env_index: int) -> None:
        self._last_seen.pop(int(env_index), None)
        self._last_selected.pop(int(env_index), None)
        self._blocked.pop(int(env_index), None)

    def select(
        self, *, env_index: int, step: int, state_fingerprint: str,
        policy_logits: Sequence[float], num_options: int,
    ) -> CycleDecision:
        if num_options < 1:
            raise ValueError("num_options must be positive")
        logits = np.asarray(policy_logits, dtype=np.float64).reshape(-1)
        if logits.size < num_options:
            raise ValueError("policy logits do not cover every legal action")
        legal_logits = logits[:num_options]
        if not np.all(np.isfinite(legal_logits)):
            raise ValueError("legal policy logits must be finite")
        # Stable sorting preserves the lower index for equal logits, matching
        # numpy argmax while making the fallback deterministic.
        ranking = np.argsort(-legal_logits, kind="stable")
        raw_action = int(ranking[0])
        key = (str(state_fingerprint), raw_action)
        seen = self._last_seen.setdefault(int(env_index), {})
        prior_step = seen.get(key)
        period = None if prior_step is None else int(step) - prior_step
        detected = period is not None and 0 < period <= self.max_cycle_period
        selected = raw_action
        intervened = False
        env = int(env_index)
        last_selected = self._last_selected.setdefault(env, {})
        blocked_by_state = self._blocked.setdefault(env, {})
        if not detected:
            blocked_by_state.pop(str(state_fingerprint), None)
        elif self.enabled:
            blocked = blocked_by_state.setdefault(str(state_fingerprint), set())
            blocked.add(last_selected.get(str(state_fingerprint), raw_action))
            alternatives = [int(action) for action in ranking
                            if int(action) not in blocked]
            if alternatives:
                selected = alternatives[0]
                intervened = selected != raw_action
        seen[key] = int(step)
        last_selected[str(state_fingerprint)] = selected
        return CycleDecision(
            schema=SCHEMA, env_index=int(env_index), step=int(step),
            state_fingerprint=str(state_fingerprint), raw_action=raw_action,
            selected_action=selected, detected=detected,
            intervened=intervened, cycle_period=period if detected else None,
            max_cycle_period=self.max_cycle_period,
            legal_actions=tuple(range(num_options)),
            policy_logits=tuple(float(v) for v in legal_logits),
        )
