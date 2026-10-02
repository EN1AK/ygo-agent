"""Evaluation-only, validity-aware results for paired checkpoint matches."""
import math
from collections import Counter

import numpy as np


class FirstEpisodeCollector:
    """Count each environment's initial duel, independent of completion order."""

    def __init__(self, num_envs):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.collected = np.zeros(num_envs, dtype=bool)

    def take(self, done):
        done = np.asarray(done, dtype=bool)
        if done.shape != self.collected.shape:
            raise ValueError("terminal batch shape changed")
        indices = np.flatnonzero(done & ~self.collected)
        self.collected[indices] = True
        return indices


def terminal_outcome(reward, invalid_game, termination_reason):
    """Reward must already be in the candidate player's perspective."""
    if not math.isfinite(float(reward)):
        raise ValueError("Non-finite terminal reward")
    if invalid_game not in (0, 1) or termination_reason not in (0, 1, 2, 3):
        raise ValueError("Unrecognized terminal validity fields")
    if invalid_game or termination_reason != 1:
        return "invalid"
    return "win" if reward > 0 else "loss" if reward < 0 else "draw"


def summarize_matches(episodes):
    counts = Counter(row["outcome"] for row in episodes)
    valid = [row for row in episodes if row["outcome"] != "invalid"]
    n, wins = len(valid), counts["win"]
    interval = None
    if n:
        p, z = wins / n, 1.959963984540054
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        interval = [max(0.0, center - half), min(1.0, center + half)]
    return {
        "attempts": len(episodes), "valid_games": n,
        "invalid_games": counts["invalid"], "wins": wins,
        "losses": counts["loss"], "draws": counts["draw"],
        "win_rate": wins / n if n else None,
        "win_rate_wilson95": interval,
        "mean_return": sum(row["reward"] for row in valid) / n if n else None,
        "mean_episode_length": sum(row["length"] for row in valid) / n if n else None,
        "termination_reasons": dict(sorted(Counter(str(row["termination_reason"]) for row in episodes).items())),
    }


def battle_report(envs, num_episodes, predict_fn, rstate1=None, rstate2=None,
                  candidate_seat=None):
    """Collect one episode per environment, including invalid attempts exactly once.

    This separate entrypoint leaves training-time ``battle`` unchanged.
    By default half of the environments assign the candidate to each seat.
    An explicit seat lets the caller run the identical environment-index/seed
    batch twice, switching model seats while preserving initial duel deals.
    Terminal validity fields are mandatory; absence must not imply a valid game.
    """
    if candidate_seat not in (None, 0, 1):
        raise ValueError("candidate_seat must be 0, 1, or None")
    if num_episodes != envs.num_envs or num_episodes < (2 if candidate_seat is None else 1):
        raise ValueError("One episode per environment and both seats are required")
    obs, infos = envs.reset()
    to_play = infos["to_play"]
    seats = np.concatenate([np.zeros(num_episodes // 2, dtype=int),
                            np.ones(num_episodes - num_episodes // 2, dtype=int)])
    if candidate_seat is not None:
        seats[:] = candidate_seat
    done = np.zeros(num_episodes, dtype=bool)
    collector = FirstEpisodeCollector(num_episodes)
    episodes = []
    while not collector.collected.all():
        candidate_acting = np.asarray(to_play) == seats
        rstate1, rstate2, actions = predict_fn(
            obs, rstate1, rstate2, candidate_acting, done)
        obs, _, done, infos = envs.step(np.asarray(actions))
        to_play = infos["to_play"]
        for idx in collector.take(done):
            reward = float(infos["r"][idx]) * (1 if candidate_acting[idx] else -1)
            invalid = int(infos["invalid_game"][idx])
            reason = int(infos["termination_reason"][idx])
            episodes.append({
                "environment_index": int(idx), "candidate_seat": int(seats[idx]),
                "reward": reward, "length": int(infos["l"][idx]),
                "invalid_game": invalid, "termination_reason": reason,
                "outcome": terminal_outcome(reward, invalid, reason),
            })
    episodes.sort(key=lambda row: row["environment_index"])
    return {**summarize_matches(episodes), "episodes": episodes,
            "by_seat": {str(seat): summarize_matches([
                row for row in episodes if row["candidate_seat"] == seat])
                for seat in (0, 1)}}
