"""Evaluate two checkpoints against each other and optionally record replays."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random

import _repo_bootstrap  # noqa: F401
import flax
import jax
import jax.numpy as jnp
import numpy as np
import ygoenv

from ygoai.rl.jax.agent import ModelArgs, RNNAgent
from ygoai.rl.utils import RecordEpisodeStatistics
from ygoai.utils import init_ygopro


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-a", required=True)
    p.add_argument("--checkpoint-b", required=True)
    p.add_argument("--player-a", type=int, choices=(0, 1), required=True)
    p.add_argument("--deck", required=True)
    p.add_argument("--code-list-file", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--record", action="store_true")
    args = p.parse_args()

    args.output = args.output.resolve()
    args.checkpoint_a = str(Path(args.checkpoint_a).resolve())
    args.checkpoint_b = str(Path(args.checkpoint_b).resolve())
    args.deck = str(Path(args.deck).resolve())
    args.code_list_file = str(Path(args.code_list_file).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "replay").mkdir()
    os.chdir(args.output)
    num_embeddings = sum(1 for line in open(args.code_list_file, encoding="utf-8-sig") if line.strip())
    env_id = "YGOPro-v1"
    model_args = ModelArgs()
    deck, _ = init_ygopro(env_id, "chinese", args.deck, args.code_list_file,
                          return_deck_names=True)
    random.seed(args.seed + 100000)
    seed = random.randint(0, int(1e8))
    env = ygoenv.make(task_id=env_id, env_type="gymnasium", num_envs=1,
                      num_threads=1, seed=seed, deck1=deck, deck2=deck, player=-1,
                      max_options=24, n_history_actions=32,
                      play_mode="self", async_reset=False, verbose=False,
                      record=args.record)
    obs_space = env.observation_space
    env.num_envs = 1
    env = RecordEpisodeStatistics(env)
    agent_a = RNNAgent(**asdict(model_args), embedding_shape=num_embeddings)
    agent_b = RNNAgent(**asdict(model_args), embedding_shape=num_embeddings)
    key = jax.random.PRNGKey(seed)
    sample = jax.tree.map(lambda x: jnp.array([x]), obs_space.sample())
    state0 = agent_a.init_rnn_state(1)
    params_a = agent_a.init(key, sample, state0)
    params_b = agent_b.init(key, sample, state0)
    params_a = flax.serialization.from_bytes(params_a, Path(args.checkpoint_a).read_bytes())
    params_b = flax.serialization.from_bytes(params_b, Path(args.checkpoint_b).read_bytes())
    rstate_a = agent_a.init_rnn_state(1)
    rstate_b = agent_b.init_rnn_state(1)

    @jax.jit
    def act(pa, pb, obs, ra, rb, use_a, done):
        na, la, va = agent_a.apply(pa, obs, ra)[:3]
        nb, lb, vb = agent_b.apply(pb, obs, rb)[:3]
        logits = jnp.where(use_a[:, None], la, lb)
        value = jnp.where(use_a[:, None], va, vb)
        ra = jax.tree.map(lambda n, o: jnp.where(use_a[:, None], n, o), na, ra)
        rb = jax.tree.map(lambda n, o: jnp.where(use_a[:, None], o, n), nb, rb)
        ra, rb = jax.tree.map(lambda x: jnp.where(done[:, None], 0, x), (ra, rb))
        return ra, rb, logits, value

    obs, info = env.reset()
    to_play = info["to_play"]
    done = np.zeros(1, dtype=np.bool_)
    steps = 0
    with (args.output / "decisions.jsonl").open("w", encoding="utf-8") as decisions:
        while not done[0]:
            use_a = np.asarray(to_play == args.player_a)
            rstate_a, rstate_b, logits, value = act(
                params_a, params_b, obs, rstate_a, rstate_b, use_a, done)
            logits = np.asarray(logits[0])
            count = int(info["num_options"][0])
            legal_logits = logits[:count]
            probabilities = np.exp(legal_logits - legal_logits.max())
            probabilities /= probabilities.sum()
            action = int(legal_logits.argmax())
            acting_a = bool(use_a[0])
            decisions.write(json.dumps({
                "step": steps, "player": int(to_play[0]),
                "model": "A" if acting_a else "B", "legal_count": count,
                "legal_action_indices": list(range(count)), "selected_action": action,
                "policy_logits": legal_logits.tolist(),
                "policy_probabilities": probabilities.tolist(),
                "state_value": float(np.asarray(value).reshape(-1)[0]),
            }) + "\n")
            obs, reward, done, info = env.step(np.asarray([action]))
            to_play = info["to_play"]
            steps += 1
    terminal = float(info["r"][0]) * (1 if acting_a else -1)
    result = {"seed": args.seed, "player_a": args.player_a, "winner": "A" if terminal > 0 else "B",
              "reward_a": terminal, "length": int(info["l"][0]),
              "win_reason": int(info["win_reason"][0]), "steps": steps,
              "checkpoint_a": args.checkpoint_a, "checkpoint_b": args.checkpoint_b,
              "checkpoint_a_sha256": hashlib.sha256(Path(args.checkpoint_a).read_bytes()).hexdigest(),
              "checkpoint_b_sha256": hashlib.sha256(Path(args.checkpoint_b).read_bytes()).hexdigest()}
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
