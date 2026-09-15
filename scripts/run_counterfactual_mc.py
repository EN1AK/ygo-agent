"""Run a bounded K=1 Monte-Carlo counterfactual experiment on YGOPro."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import _repo_bootstrap  # noqa: F401
import numpy as np
import ygoenv

from ygoai.rl.counterfactual import stable_seed, unpack_step, write_jsonl
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


def scalar(value) -> int | float | bool:
    return np.asarray(value).reshape(-1)[0].item()


def rollout(row, root_action: int, rollout_seed: int, *, deck: str,
            code_list: str, max_steps: int) -> tuple[float, int]:
    snapshot = row["snapshot"]
    root_player = int(row["player"])
    env = make_env(int(snapshot["seed"]), deck, code_list)
    obs, info = env.reset()
    try:
        for action in snapshot["actions"]:
            obs, reward, done, info = unpack_step(
                env.step(np.asarray([int(action)])))
            if bool(scalar(done)):
                raise RuntimeError("snapshot prefix terminated early")
        rng = random.Random(rollout_seed)
        action = int(root_action)
        for step in range(max_steps):
            acting_player = int(scalar(info["to_play"]))
            obs, reward, done, info = unpack_step(
                env.step(np.asarray([action])))
            if bool(scalar(done)):
                terminal = float(scalar(reward))
                if acting_player != root_player:
                    terminal = -terminal
                return terminal, step + 1
            legal_count = int(scalar(info["num_options"]))
            if legal_count < 1:
                raise RuntimeError("nonterminal state has no legal action")
            action = rng.randrange(legal_count)
        raise RuntimeError("rollout exceeded max_steps")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deck", required=True)
    parser.add_argument("--code-list-file", required=True)
    parser.add_argument("--rollout-seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.decisions.read_text().splitlines()
            if line.strip()]
    output = []
    for row in rows:
        for action in row["legal_actions"]:
            for index in range(args.rollout_seeds):
                seed = stable_seed(args.seed, row["decision_id"], action, index)
                value, steps = rollout(row, int(action), seed, deck=args.deck,
                                       code_list=args.code_list_file,
                                       max_steps=args.max_steps)
                output.append({"decision_id": row["decision_id"],
                               "action": int(action), "particle": 0,
                               "rollout_seed": seed, "value": value,
                               "steps": steps, "policy": "uniform-random"})
                print(json.dumps(output[-1]), flush=True)
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()
