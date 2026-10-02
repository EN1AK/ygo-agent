"""Bounded candidate rollout with an explicit KL-regularized policy update.

The update follows Ataraxos Appendix D.7 (arxiv:2511.07312), with
configurable coefficients rather than assumed transferable Stratego settings.
The engine/evaluator adapter owns recurrent state and hidden-state application.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import time
from typing import ContextManager, Protocol

import numpy as np

from ygoai.rl.counterfactual import stable_seed


class SearchFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchPosition:
    player: int | None
    legal_actions: tuple[int, ...]
    logits: tuple[float, ...]
    value: float | None  # Acting-player value, not an action Q estimate.
    observation_digest: str
    prompt: str
    terminal_root_return: float | None = None


class SearchBranch(Protocol):
    def evaluate(self) -> SearchPosition: ...
    def step(self, action: int) -> None: ...


class SearchBackend(Protocol):
    information_mode: str
    snapshot_bytes: int

    def branch(self, particle: int, seed: int) -> ContextManager[SearchBranch]: ...


@dataclass(frozen=True)
class SearchBudget:
    candidates: int = 4
    particles: int = 1
    rollouts_per_action: int = 4
    depth: int = 16
    wall_seconds: float = 5.0
    snapshot_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for value in (self.candidates, self.particles, self.rollouts_per_action,
                      self.depth, self.snapshot_bytes):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError('Budget counts must be positive integers')
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError('Wall budget must be finite and positive')


def _probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise SearchFailure('nonfinite_or_empty_policy')
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def regularized_policy(raw_logits, returns, *, policy_kl=1.0, magnet_kl=0.0,
                       magnet_logits=None):
    """Maximize E[return] - beta KL(p,raw) - alpha KL(p,magnet)."""
    if (not math.isfinite(policy_kl) or policy_kl <= 0 or
            not math.isfinite(magnet_kl) or magnet_kl < 0):
        raise ValueError('A positive raw-policy KL coefficient is required')
    raw = np.asarray(raw_logits, dtype=np.float64)
    values = np.asarray(returns, dtype=np.float64)
    _probabilities(raw)
    if values.shape != raw.shape or not np.isfinite(values).all():
        raise SearchFailure('invalid_action_returns')
    magnet = np.zeros_like(raw) if magnet_logits is None else np.asarray(magnet_logits, dtype=np.float64)
    if magnet.shape != raw.shape:
        raise ValueError('Magnet policy must align with the legal menu')
    _probabilities(magnet)
    # Additive log-normalizers cancel in the final softmax.
    return _probabilities((values + policy_kl * (raw - raw.max()) +
                           magnet_kl * (magnet - magnet.max())) / (policy_kl + magnet_kl))


def candidate_search(root: SearchPosition, backend: SearchBackend, budget: SearchBudget,
                     *, seed: int, policy_kl=1.0, magnet_kl=0.0, magnet_logits=None,
                     fair_play=False, strict=False, sample_action=False,
                     particle_order=None,
                     clock=time.monotonic):
    """Compare complete paired rollouts; any incomplete comparison falls back.

    Candidate subsets retain their original total policy mass. Other legal
    actions retain their raw probability. Wall time is checked cooperatively
    around backend calls; an adapter must bound its own blocking operations.
    """
    if root.player not in (0, 1) or root.terminal_root_return is not None:
        raise ValueError('Search root must be a live player decision')
    if root.value is None or not math.isfinite(root.value) or not root.observation_digest:
        raise ValueError('Search root requires a finite critic value and identity')
    menu = root.legal_actions
    if not menu or len(menu) != len(root.logits) or len(set(menu)) != len(menu):
        raise ValueError('Root menu/logits must be unique and aligned')
    if any(not isinstance(a, int) or isinstance(a, bool) or a < 0 for a in menu):
        raise ValueError('Actions must be nonnegative integer engine indices')
    raw = _probabilities(root.logits)
    # Validate update settings even when a later budget gate triggers fallback.
    regularized_policy(root.logits, np.zeros(len(menu)), policy_kl=policy_kl,
                       magnet_kl=magnet_kl, magnet_logits=magnet_logits)
    raw_index = int(np.argmax(raw))
    indices = sorted(sorted(range(len(menu)), key=lambda i: (-raw[i], i))[:budget.candidates])
    start = clock()
    rows = []
    order = list(range(budget.particles)) if particle_order is None else list(particle_order)
    if sorted(order) != list(range(budget.particles)):
        raise ValueError('Particle execution order must be a complete permutation')
    result = {'schema':'ygo-candidate-search-v1', 'method':'candidate-rollout',
              'information_mode':backend.information_mode, 'budget':asdict(budget),
              'root_observation_digest':root.observation_digest, 'root_player':root.player,
              'legal_actions':list(menu), 'raw_logits':list(root.logits),
              'raw_value':root.value, 'raw_policy':raw.tolist(),
              'raw_action':menu[raw_index], 'selected_action':menu[raw_index],
              'updated_policy':raw.tolist(), 'policy_kl':policy_kl, 'magnet_kl':magnet_kl,
              'magnet_logits':None if magnet_logits is None else list(magnet_logits),
              'candidate_actions':[menu[i] for i in indices], 'seed':seed,
              'particle_execution_order':order,
              'selection_rule':'sample' if sample_action else 'argmax',
              'fallback_reason':None, 'completed':False, 'rollouts':rows,
              'action_returns':[]}

    def check_time():
        if clock() - start >= budget.wall_seconds:
            raise SearchFailure('wall_time_budget')

    try:
        if backend.information_mode not in ('oracle_exact_state', 'belief_particles'):
            raise SearchFailure('unknown_information_mode')
        if fair_play and backend.information_mode != 'belief_particles':
            raise SearchFailure('oracle_forbidden_in_fair_play')
        if backend.snapshot_bytes < 0 or backend.snapshot_bytes > budget.snapshot_bytes:
            raise SearchFailure('snapshot_memory_budget')
        for particle in order:
            for trial in range(budget.rollouts_per_action):
                # Excludes candidate action: every candidate shares random draws.
                rollout_seed = stable_seed(seed, root.observation_digest, particle, trial)
                for index in indices:
                    check_time()
                    with backend.branch(particle, rollout_seed) as branch:
                        check_time()
                        restored = branch.evaluate()
                        if (restored.observation_digest != root.observation_digest or
                                restored.legal_actions != menu or restored.player != root.player or
                                restored.terminal_root_return is not None):
                            raise SearchFailure('root_identity_mismatch')
                        rng = random.Random(rollout_seed)
                        action = menu[index]
                        prefix = [action]
                        branch.step(action)
                        for depth in range(1, budget.depth + 1):
                            check_time()
                            position = branch.evaluate()
                            check_time()
                            if position.terminal_root_return is not None:
                                value = float(position.terminal_root_return)
                                if value not in (-1., 0., 1.):
                                    raise SearchFailure('terminal_reward_scale')
                                break
                            if position.player not in (0, 1):
                                raise SearchFailure('invalid_player')
                            if depth == budget.depth:
                                if position.prompt == 'select_chain':
                                    raise SearchFailure('unresolved_chain_at_depth_limit')
                                value = float(position.value)
                                value *= 1 if position.player == root.player else -1
                                break
                            if len(position.legal_actions) != len(position.logits):
                                raise SearchFailure('branch_menu_alignment')
                            action = rng.choices(position.legal_actions,
                                                 weights=_probabilities(position.logits), k=1)[0]
                            branch.step(action)
                            prefix.append(action)
                        if not math.isfinite(value):
                            raise SearchFailure('nonfinite_leaf')
                        rows.append({'action':menu[index], 'particle':particle, 'trial':trial,
                                     'rollout_seed':rollout_seed, 'root_return':value,
                                     'action_prefix':prefix, 'leaf':asdict(position)})
                    check_time()
        means = []
        for index in indices:
            values = [r['root_return'] for r in rows if r['action']==menu[index]]
            mean = math.fsum(values)/len(values)
            particle_means = [math.fsum(r['root_return'] for r in rows
                if r['action']==menu[index] and r['particle']==p)/budget.rollouts_per_action
                for p in range(budget.particles)]
            # Repeated rollouts of one hidden-state particle are not independent
            # draws of hidden state. Cluster uncertainty by particle when present.
            uncertainty_values = particle_means if budget.particles>1 else values
            count = len(uncertainty_values)
            error = (math.sqrt(math.fsum((v-mean)**2 for v in uncertainty_values)/(count-1)/count)
                     if count>1 else None)
            means.append(mean)
            result['action_returns'].append({'action':menu[index], 'mean':mean,
                'stderr':error, 'samples':len(values),'particle_means':particle_means,
                'uncertainty_unit':'particle' if budget.particles>1 else 'rollout-conditional-on-one-state'})
        updated = raw.copy()
        magnet_subset = None if magnet_logits is None else np.asarray(magnet_logits)[indices]
        updated[indices] = raw[indices].sum() * regularized_policy(
            np.asarray(root.logits)[indices], means, policy_kl=policy_kl,
            magnet_kl=magnet_kl, magnet_logits=magnet_subset)
        if sample_action:
            chosen = random.Random(stable_seed(seed,'selection')).choices(menu,weights=updated,k=1)[0]
        else:
            chosen = menu[int(np.argmax(updated))]
        result.update(completed=True, updated_policy=updated.tolist(), selected_action=chosen)
    except SearchFailure as error:
        if strict:
            raise
        result['fallback_reason'] = str(error)
    result['elapsed_seconds'] = clock()-start
    return result
