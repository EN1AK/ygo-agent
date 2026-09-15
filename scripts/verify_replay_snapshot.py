"""Verify replay snapshots reproduce exact YGOPro decision observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _repo_bootstrap  # noqa: F401
import numpy as np
import ygoenv

from ygoai.rl.counterfactual import observation_digest
from ygoai.utils import init_ygopro


def make_env(seed: int, deck_path: str, code_list: str):
    deck, _ = init_ygopro("YGOPro-v1", "chinese", deck_path, code_list,
                          return_deck_names=True)
    env = ygoenv.make(
        task_id="YGOPro-v1", env_type="gymnasium", num_envs=1,
        num_threads=1, seed=seed, deck1=deck, deck2=deck, player=-1,
        max_options=24, n_history_actions=32, play_mode="self",
        async_reset=False, verbose=False, record=False)
    env.num_envs = 1
    return env


def replay(row, deck_path: str, code_list: str):
    snapshot = row["snapshot"]
    env = make_env(int(snapshot["seed"]), deck_path, code_list)
    obs, info = env.reset()
    try:
        for action in snapshot["actions"]:
            obs, _reward, done, info = env.step(np.asarray([int(action)]))
            if bool(done[0]):
                raise RuntimeError("snapshot prefix terminated before decision")
        return observation_digest(obs), int(info["to_play"][0]), \
            int(info["num_options"][0])
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--deck", required=True)
    parser.add_argument("--code-list-file", required=True)
    parser.add_argument("--max-points", type=int, default=16)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()
            if line.strip()]
    if not rows:
        raise RuntimeError("trace contains no decisions")
    if any("observation_digest" not in row for row in rows):
        raise RuntimeError("trace predates observation digest support")
    if args.max_points < 1:
        raise ValueError("max-points must be positive")
    if len(rows) <= args.max_points:
        selected = rows
    else:
        indices = np.linspace(0, len(rows) - 1, args.max_points,
                              dtype=np.int64)
        selected = [rows[int(i)] for i in indices]
    checked = []
    for row in selected:
        digest, player, legal_count = replay(
            row, args.deck, args.code_list_file)
        expected = (row["observation_digest"], int(row["player"]),
                    int(row["legal_count"]))
        actual = (digest, player, legal_count)
        if actual != expected:
            raise AssertionError(
                f"snapshot mismatch at {row['decision_id']}: "
                f"expected={expected}, actual={actual}")
        checked.append(row["decision_id"])
    print(json.dumps({"status": "pass", "trace": str(args.trace),
                      "decisions": len(rows), "restored": checked}, indent=2))


if __name__ == "__main__":
    main()
