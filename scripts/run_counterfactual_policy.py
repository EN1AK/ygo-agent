"""Evaluate snapshot root actions with the original checkpoint policies."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import _repo_bootstrap  # noqa: F401
import flax
import jax
import jax.numpy as jnp
import numpy as np
import ygoenv

from ygoai.rl.counterfactual import stable_seed, unpack_step, write_jsonl
from ygoai.rl.jax.agent import ModelArgs, RNNAgent
from ygoai.utils import init_ygopro


def scalar(value):
    return np.asarray(value).reshape(-1)[0].item()


def make_env(seed, deck_path, code_list):
    deck, _ = init_ygopro("YGOPro-v1", "chinese", deck_path, code_list,
                          return_deck_names=True)
    env = ygoenv.make(task_id="YGOPro-v1", env_type="gymnasium", num_envs=1,
                      num_threads=1, seed=seed, deck1=deck, deck2=deck,
                      player=-1, max_options=24, n_history_actions=32,
                      play_mode="self", async_reset=False, verbose=False,
                      record=False)
    env.num_envs = 1
    return env


def load_agent(checkpoint, obs_space, embeddings, key):
    agent = RNNAgent(**asdict(ModelArgs()), embedding_shape=embeddings)
    sample = jax.tree.map(lambda x: jnp.array([x]), obs_space.sample())
    state = agent.init_rnn_state(1)
    params = agent.init(key, sample, state)
    params = flax.serialization.from_bytes(params, Path(checkpoint).read_bytes())

    @jax.jit
    def forward(obs, rstate):
        next_state, logits, value = agent.apply(params, obs, rstate)[:3]
        return next_state, logits, value

    return agent, forward


def prepare_root(row, args, forward_a, forward_b, agent_a, agent_b):
    snap = row["snapshot"]
    env = make_env(int(snap["seed"]), args.deck, args.code_list_file)
    obs, info = env.reset()
    ra, rb = agent_a.init_rnn_state(1), agent_b.init_rnn_state(1)
    try:
        for action in snap["actions"]:
            player = int(scalar(info["to_play"]))
            if player == args.player_a:
                ra, _logits, _value = forward_a(obs, ra)
            else:
                rb, _logits, _value = forward_b(obs, rb)
            obs, _reward, done, info = unpack_step(
                env.step(np.asarray([int(action)])))
            if bool(scalar(done)):
                raise RuntimeError("snapshot prefix terminated early")
        return ra, rb
    finally:
        env.close()


def rollout(row, root_action, rollout_seed, args, forward_a, forward_b,
            root_ra, root_rb):
    snap = row["snapshot"]
    root_player = int(row["player"])
    env = make_env(int(snap["seed"]), args.deck, args.code_list_file)
    obs, info = env.reset()
    try:
        for action in snap["actions"]:
            obs, _reward, done, info = unpack_step(
                env.step(np.asarray([int(action)])))
            if bool(scalar(done)):
                raise RuntimeError("snapshot prefix terminated early")
        ra, rb = root_ra, root_rb
        rng = random.Random(rollout_seed)
        forced = True
        for step in range(args.max_steps):
            player = int(scalar(info["to_play"]))
            if player == args.player_a:
                ra, logits, _value = forward_a(obs, ra)
            else:
                rb, logits, _value = forward_b(obs, rb)
            count = int(scalar(info["num_options"]))
            if forced:
                action = int(root_action)
                forced = False
            else:
                values = np.asarray(logits[0, :count], dtype=np.float64)
                values -= values.max()
                probs = np.exp(values)
                probs /= probs.sum()
                action = rng.choices(range(count), weights=probs, k=1)[0]
            obs, reward, done, info = unpack_step(
                env.step(np.asarray([action])))
            if bool(scalar(done)):
                value = float(scalar(reward))
                if player != root_player:
                    value = -value
                return value, step + 1
        raise RuntimeError("policy rollout exceeded max_steps")
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decision", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint-a", required=True)
    p.add_argument("--checkpoint-b", required=True)
    p.add_argument("--player-a", type=int, choices=(0, 1), required=True)
    p.add_argument("--deck", required=True)
    p.add_argument("--code-list-file", required=True)
    p.add_argument("--rollout-seeds", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=1000)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(x) for x in args.decision.read_text().splitlines()
            if x.strip()]
    if len(rows) != 1:
        raise ValueError("decision file must contain exactly one point")
    row = rows[0]
    embeddings = sum(1 for line in open(args.code_list_file,
                                        encoding="utf-8-sig") if line.strip())
    probe = make_env(int(row["snapshot"]["seed"]), args.deck,
                     args.code_list_file)
    obs_space = probe.observation_space
    probe.close()
    key_a, key_b = jax.random.split(jax.random.PRNGKey(args.seed))
    agent_a, forward_a = load_agent(args.checkpoint_a, obs_space, embeddings,
                                    key_a)
    agent_b, forward_b = load_agent(args.checkpoint_b, obs_space, embeddings,
                                    key_b)
    root_ra, root_rb = prepare_root(row, args, forward_a, forward_b,
                                    agent_a, agent_b)
    output = []
    for action in row["legal_actions"]:
        for index in range(args.rollout_seeds):
            seed = stable_seed(args.seed, row["decision_id"], index)
            value, steps = rollout(row, action, seed, args, forward_a,
                                   forward_b, root_ra, root_rb)
            result = {"decision_id": row["decision_id"], "action": action,
                      "particle": 0, "rollout_seed": seed, "value": value,
                      "steps": steps, "policy": "checkpoint-sampling"}
            output.append(result)
            print(json.dumps(result), flush=True)
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()
