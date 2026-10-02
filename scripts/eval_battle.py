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

from ygoai.rl.checkpoint_compat import sha256_file, validate_checkpoint_compatibility
from ygoai.rl.env import VersionedObservation
from ygoai.rl.jax.agent import ModelArgs, RNNAgent
from ygoai.rl.observation_schema import (
    DEFAULT_GROUP_REFERENCES,
    DEFAULT_PUBLIC_EVENTS,
    LEGACY_SCHEMA,
)
from ygoai.rl.counterfactual import observation_digest
from ygoai.rl.cycle_guard import PolicyCycleGuard, public_state_digest
from ygoai.rl.match_report import terminal_outcome
from ygoai.rl.utils import RecordEpisodeStatistics
from ygoai.utils import init_ygopro


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-a", required=True)
    p.add_argument("--checkpoint-b", required=True)
    p.add_argument("--player-a", type=int, choices=(0, 1), required=True)
    p.add_argument("--deck", required=True)
    p.add_argument("--deck1")
    p.add_argument("--deck2")
    p.add_argument("--code-list-file", required=True)
    p.add_argument("--observation-schema", default=LEGACY_SCHEMA)
    p.add_argument("--semantic-asset-dir", default="")
    p.add_argument("--n-public-events", type=int, default=DEFAULT_PUBLIC_EVENTS)
    p.add_argument("--max-group-references", type=int, default=DEFAULT_GROUP_REFERENCES)
    p.add_argument("--max-options", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--n-history-actions", type=int, default=32)
    p.add_argument("--checkpoint-a-schema", default=LEGACY_SCHEMA)
    p.add_argument("--checkpoint-b-schema", default=LEGACY_SCHEMA)
    p.add_argument("--checkpoint-a-variant", default="full")
    p.add_argument("--checkpoint-b-variant", default="full")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--record", action="store_true")
    p.add_argument("--verbose", action="store_true",
                   help="emit the complete human-readable engine/action log")
    p.add_argument("--cycle-guard", action="store_true",
                   help="replace repeated argmax choices with ranked alternatives")
    p.add_argument("--cycle-max-period", type=int, default=8)
    args = p.parse_args()

    args.output = args.output.resolve()
    args.checkpoint_a = str(Path(args.checkpoint_a).resolve())
    args.checkpoint_b = str(Path(args.checkpoint_b).resolve())
    args.deck = str(Path(args.deck).resolve())
    args.code_list_file = str(Path(args.code_list_file).resolve())
    if args.semantic_asset_dir:
        args.semantic_asset_dir = str(Path(args.semantic_asset_dir).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "replay").mkdir()
    os.chdir(args.output)
    num_embeddings = sum(1 for line in open(args.code_list_file, encoding="utf-8-sig") if line.strip())
    env_id = "YGOPro-v1"
    model_args_a = ModelArgs(
        observation_schema=args.checkpoint_a_schema,
        structured_variant=args.checkpoint_a_variant,
    )
    model_args_b = ModelArgs(
        observation_schema=args.checkpoint_b_schema,
        structured_variant=args.checkpoint_b_variant,
    )
    deck, _ = init_ygopro(env_id, "chinese", args.deck, args.code_list_file,
                          return_deck_names=True)
    deck1 = args.deck1 or deck
    deck2 = args.deck2 or deck
    random.seed(args.seed + 100000)
    seed = random.randint(0, int(1e8))
    env = ygoenv.make(task_id=env_id, env_type="gymnasium", num_envs=1,
                      num_threads=1, seed=seed, deck1=deck1, deck2=deck2, player=-1,
                      max_options=args.max_options,
                      max_steps=args.max_steps,
                      n_history_actions=args.n_history_actions,
                      play_mode="self", async_reset=False, verbose=args.verbose,
                      record=args.record,
                      observation_schema=args.observation_schema,
                      semantic_asset_dir=args.semantic_asset_dir,
                      n_public_events=args.n_public_events,
                      max_group_references=args.max_group_references)
    env.num_envs = 1
    env = VersionedObservation(env, args.observation_schema)
    obs_space = env.observation_space
    env = RecordEpisodeStatistics(env)
    agent_a = RNNAgent(**asdict(model_args_a), embedding_shape=num_embeddings)
    agent_b = RNNAgent(**asdict(model_args_b), embedding_shape=num_embeddings)
    key = jax.random.PRNGKey(seed)
    sample = jax.tree.map(lambda x: jnp.array([x]), obs_space.sample())
    state0 = agent_a.init_rnn_state(1)
    params_a = agent_a.init(key, sample, state0)
    params_b = agent_b.init(key, sample, state0)
    semantic_hash = None
    if args.semantic_asset_dir:
        metadata = Path(args.semantic_asset_dir) / "metadata.json"
        if metadata.exists():
            semantic_hash = sha256_file(metadata)
    capacities = {
        "max_cards": 80,
        "max_options": args.max_options,
        "history_actions": args.n_history_actions,
        "public_events": args.n_public_events,
        "group_references": args.max_group_references,
    }
    validate_checkpoint_compatibility(
        args.checkpoint_a,
        observation_schema=args.checkpoint_a_schema,
        model_args=model_args_a,
        semantic_table_hash=semantic_hash if args.checkpoint_a_schema == args.observation_schema else None,
        code_list_hash=sha256_file(args.code_list_file),
        capacities=capacities if args.checkpoint_a_schema == args.observation_schema else None,
    )
    validate_checkpoint_compatibility(
        args.checkpoint_b,
        observation_schema=args.checkpoint_b_schema,
        model_args=model_args_b,
        semantic_table_hash=semantic_hash if args.checkpoint_b_schema == args.observation_schema else None,
        code_list_hash=sha256_file(args.code_list_file),
        capacities=capacities if args.checkpoint_b_schema == args.observation_schema else None,
    )
    params_a = flax.serialization.from_bytes(params_a, Path(args.checkpoint_a).read_bytes())
    params_b = flax.serialization.from_bytes(params_b, Path(args.checkpoint_b).read_bytes())
    checkpoint_a_sha256 = hashlib.sha256(Path(args.checkpoint_a).read_bytes()).hexdigest()
    checkpoint_b_sha256 = hashlib.sha256(Path(args.checkpoint_b).read_bytes()).hexdigest()
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
    action_prefix = []
    cycle_guard = PolicyCycleGuard(
        enabled=args.cycle_guard, max_cycle_period=args.cycle_max_period)
    cycle_detections = 0
    cycle_interventions = 0
    trace_id = f"battle-{args.seed}-seat{args.player_a}"
    with (args.output / "decisions.jsonl").open("w", encoding="utf-8") as decisions, \
            (args.output / "cycles.jsonl").open("w", encoding="utf-8") as cycles:
        while not done[0]:
            use_a = np.asarray(to_play == args.player_a)
            rstate_a, rstate_b, logits, value = act(
                params_a, params_b, obs, rstate_a, rstate_b, use_a, done)
            logits = np.asarray(logits[0])
            count = int(info["num_options"][0])
            legal_logits = logits[:count]
            probabilities = np.exp(legal_logits - legal_logits.max())
            probabilities /= probabilities.sum()
            raw_action = int(legal_logits.argmax())
            fingerprint = public_state_digest(
                obs, env_index=0, num_options=count, player=int(to_play[0]))
            cycle = cycle_guard.select(
                env_index=0, step=steps, state_fingerprint=fingerprint,
                policy_logits=legal_logits, num_options=count)
            action = cycle.selected_action
            if cycle.detected:
                cycle_detections += 1
                cycle_interventions += int(cycle.intervened)
                cycle_row = cycle.to_json()
                cycle_row.update({
                    "record_type": "policy_cycle",
                    "trace_id": trace_id,
                    "decision_id": f"{trace_id}:{steps}",
                    "player": int(to_play[0]),
                    "model": "A" if bool(use_a[0]) else "B",
                    "player_a": args.player_a,
                    "checkpoint_a": args.checkpoint_a,
                    "checkpoint_b": args.checkpoint_b,
                    "checkpoint_a_sha256": checkpoint_a_sha256,
                    "checkpoint_b_sha256": checkpoint_b_sha256,
                    "checkpoint_a_schema": args.checkpoint_a_schema,
                    "checkpoint_b_schema": args.checkpoint_b_schema,
                    "checkpoint_a_variant": args.checkpoint_a_variant,
                    "checkpoint_b_variant": args.checkpoint_b_variant,
                    "state_value": float(np.asarray(value).reshape(-1)[0]),
                    "observation_digest": observation_digest(obs),
                    "snapshot": {
                        "seed": seed, "actions": list(action_prefix),
                        "player": int(to_play[0]), "play_mode": "self",
                        "replayable": True, "deck1": deck1, "deck2": deck2,
                        "max_options": args.max_options,
                        "n_history_actions": args.n_history_actions,
                        "observation_schema": args.observation_schema,
                        "semantic_asset_dir": args.semantic_asset_dir,
                        "n_public_events": args.n_public_events,
                        "max_group_references": args.max_group_references,
                    },
                })
                cycles.write(json.dumps(cycle_row, ensure_ascii=False) + "\n")
            acting_a = bool(use_a[0])
            global_features = np.asarray(obs["global_"][0])
            action_features = np.asarray(obs["actions_"][0, :count])
            phase_names = {
                0: "draw", 1: "standby", 2: "main1",
                3: "battle_start", 4: "battle_step", 5: "damage",
                6: "damage_calculation", 7: "battle", 8: "main2", 9: "end",
            }
            legal_actions = []
            for index in range(count):
                item = {
                    "index": index,
                    "legacy_features": action_features[index].tolist(),
                }
                if "action_features_" in obs:
                    item["structured_features"] = np.asarray(
                        obs["action_features_"][0, index]).tolist()
                    item["single_references"] = np.asarray(
                        obs["action_single_refs_"][0, index]).tolist()
                    item["group_references"] = np.asarray(
                        obs["action_group_refs_"][0, index]).tolist()
                    item["group_reference_mask"] = np.asarray(
                        obs["action_group_mask_"][0, index]).tolist()
                legal_actions.append(item)
            decisions.write(json.dumps({
                "record_type": "decision",
                "trace_id": trace_id, "decision_id": f"{trace_id}:{steps}",
                "step": steps, "player": int(to_play[0]),
                "model": "A" if acting_a else "B", "legal_count": count,
                "checkpoint_sha256": checkpoint_a_sha256 if acting_a else checkpoint_b_sha256,
                "turn": int(global_features[4]),
                "phase_id": int(global_features[5]),
                "phase": phase_names.get(int(global_features[5]), "unknown"),
                "phase_context": {
                    "self_went_first": bool(global_features[6]),
                    "is_self_turn": bool(global_features[7]),
                    "selection": np.asarray(obs.get("selection_", np.zeros((1, 0), dtype=np.uint8))[0]).tolist(),
                    "global_features": global_features.tolist(),
                },
                "legal_actions": legal_actions,
                "raw_selected_action": raw_action,
                "selected_action": action,
                "cycle_guard_intervened": action != raw_action,
                "policy_logits": legal_logits.tolist(),
                "policy_probabilities": probabilities.tolist(),
                "state_value": float(np.asarray(value).reshape(-1)[0]),
                "observation_digest": observation_digest(obs),
                "snapshot": {"seed": seed, "actions": action_prefix,
                             "player": int(to_play[0])},
            }) + "\n")
            obs, reward, done, info = env.step(np.asarray([action]))
            action_prefix.append(action)
            to_play = info["to_play"]
            steps += 1
    terminal = float(info["r"][0]) * (1 if acting_a else -1)
    invalid = int(info["invalid_game"][0])
    termination_reason = int(info["termination_reason"][0])
    outcome = terminal_outcome(terminal, invalid, termination_reason)
    winner = "A" if outcome == "win" else "B" if outcome == "loss" else None
    result = {"seed": args.seed, "player_a": args.player_a, "winner": winner,
              "outcome_a": outcome, "invalid_game": invalid,
              "termination_reason": termination_reason, "max_steps": args.max_steps,
              "reward_a": terminal, "length": int(info["l"][0]),
              "win_reason": int(info["win_reason"][0]), "steps": steps,
              "checkpoint_a": args.checkpoint_a, "checkpoint_b": args.checkpoint_b,
              "checkpoint_a_sha256": checkpoint_a_sha256,
              "checkpoint_b_sha256": checkpoint_b_sha256,
              "cycle_mode": "next_ranked_guard" if args.cycle_guard else "raw",
              "cycle_detections": cycle_detections,
              "cycle_interventions": cycle_interventions,
              "observation_schema": args.observation_schema,
              "checkpoint_a_schema": args.checkpoint_a_schema,
              "checkpoint_b_schema": args.checkpoint_b_schema,
              "semantic_table_hash": semantic_hash,
              "code_list_hash": sha256_file(args.code_list_file),
              "deck": args.deck, "deck1": deck1, "deck2": deck2}
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "decisions.jsonl").open("a", encoding="utf-8") as decisions:
        decisions.write(json.dumps({
            "record_type": "terminal", "trace_id": trace_id,
            "terminal_reward_a": terminal, "winner": result["winner"],
            "outcome_a": outcome, "invalid_game": invalid,
            "termination_reason": termination_reason,
            "win_reason": result["win_reason"], "steps": steps,
            "checkpoint_a": args.checkpoint_a, "checkpoint_b": args.checkpoint_b,
            "checkpoint_a_sha256": checkpoint_a_sha256,
            "checkpoint_b_sha256": checkpoint_b_sha256,
        }) + "\n")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
