"""Compare bounded terminal enumeration with leaf-value PUCT on a replay snapshot.

The duel engine has no cheap public clone operation, so a search node is an
action prefix.  Restoring a node deterministically replays the snapshot and
that prefix, including both players' recurrent model states.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
from statistics import fmean, stdev
from typing import Any

import _repo_bootstrap  # noqa: F401
import flax
import jax
import jax.numpy as jnp
import numpy as np
import ygoenv

from ygoai.rl.counterfactual import observation_digest, stable_seed, unpack_step
from ygoai.rl.jax.agent import ModelArgs, RNNAgent
from ygoai.utils import init_ygopro


def scalar(value):
    return np.asarray(value).reshape(-1)[0].item()


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max()
    probs = np.exp(values)
    return probs / probs.sum()


def stderr(values: list[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def validate_terminal_reward(value: float) -> None:
    if not np.isclose(abs(value), 1.0):
        raise RuntimeError(
            "terminal reward scale mismatch: expected win/loss ±1 with "
            f"greedy_reward=False, got {value}")


PROMPT_NAMES = {
    1: "select_idlecmd", 2: "select_chain", 3: "select_card",
    4: "select_tribute", 5: "select_position", 6: "select_effectyn",
    7: "select_yesno", 8: "select_battlecmd", 9: "select_unselect_card",
    10: "select_option", 11: "select_place", 12: "select_sum",
    13: "select_disfield", 14: "announce_attrib", 15: "announce_number",
    16: "announce_card",
}


def prompt_from_obs(obs) -> tuple[int, str]:
    prompt_id = 0
    if "selection_" in obs:
        prompt_id = int(np.asarray(obs["selection_"])[0, 0])
    # Legacy checkpoints expose a zero-filled Structured-lite selection block;
    # fall back to the message feature shared by all non-padding legal actions.
    if prompt_id == 0:
        message_ids = np.asarray(obs["actions_"])[0, :, 3]
        nonzero = message_ids[message_ids != 0]
        if len(nonzero):
            prompt_id = int(nonzero[0])
    return prompt_id, PROMPT_NAMES.get(prompt_id, f"unknown-{prompt_id}")


@dataclass
class RestoredState:
    env: Any
    obs: Any
    info: Any
    rstate_a: Any
    rstate_b: Any
    terminal: bool = False
    terminal_value: float | None = None


@dataclass
class Evaluation:
    player: int | None
    priors: list[float]
    value: float
    raw_value: float | None
    terminal: bool
    observation_digest: str
    prompt_id: int | None
    prompt: str


class SnapshotModel:
    def __init__(self, row, args):
        self.row = row
        self.args = args
        self.root_player = int(row["player"])
        self.deck, _ = init_ygopro(
            "YGOPro-v1", "chinese", args.deck, args.code_list_file,
            return_deck_names=True)
        probe = self.make_env()
        obs_space = probe.observation_space
        probe.close()
        embeddings = sum(
            1 for line in open(args.code_list_file, encoding="utf-8-sig")
            if line.strip())
        key_a, key_b = jax.random.split(jax.random.PRNGKey(args.seed))
        self.agent_a, self.forward_a = self.load_agent(
            args.checkpoint_a, obs_space, embeddings, key_a)
        self.agent_b, self.forward_b = self.load_agent(
            args.checkpoint_b, obs_space, embeddings, key_b)
        if self.root_player == args.player_a:
            self.search_agent, self.search_forward = self.agent_a, self.forward_a
            self.search_checkpoint = args.checkpoint_a
        else:
            self.search_agent, self.search_forward = self.agent_b, self.forward_b
            self.search_checkpoint = args.checkpoint_b

    def make_env(self):
        return ygoenv.make(
            task_id="YGOPro-v1", env_type="gymnasium", num_envs=1,
            num_threads=1, seed=int(self.row["snapshot"]["seed"]),
            deck1=self.deck, deck2=self.deck, player=-1, max_options=24,
            n_history_actions=32, play_mode="self", async_reset=False,
            greedy_reward=False, verbose=self.args.verbose, record=False)

    @staticmethod
    def load_agent(checkpoint, obs_space, embeddings, key):
        agent = RNNAgent(**asdict(ModelArgs()), embedding_shape=embeddings)
        sample = jax.tree.map(lambda x: jnp.array([x]), obs_space.sample())
        state = agent.init_rnn_state(1)
        params = agent.init(key, sample, state)
        params = flax.serialization.from_bytes(
            params, Path(checkpoint).read_bytes())

        @jax.jit
        def forward(obs, rstate):
            return agent.apply(params, obs, rstate)[:3]

        return agent, forward

    def advance_model(self, obs, info, ra, rb, *, evaluator="per-player"):
        player = int(scalar(info["to_play"]))
        if evaluator == "root":
            if player == self.args.player_a:
                ra, logits, value = self.search_forward(obs, ra)
            else:
                rb, logits, value = self.search_forward(obs, rb)
        elif player == self.args.player_a:
            ra, logits, value = self.forward_a(obs, ra)
        else:
            rb, logits, value = self.forward_b(obs, rb)
        return player, ra, rb, logits, value

    def restore(self, prefix: tuple[int, ...], *, evaluator="root") -> RestoredState:
        env = self.make_env()
        env.num_envs = 1
        obs, info = env.reset()
        if evaluator == "root":
            ra = self.search_agent.init_rnn_state(1)
            rb = self.search_agent.init_rnn_state(1)
        else:
            ra = self.agent_a.init_rnn_state(1)
            rb = self.agent_b.init_rnn_state(1)
        all_actions = tuple(self.row["snapshot"]["actions"]) + prefix
        for action in all_actions:
            player, ra, rb, _logits, _value = self.advance_model(
                obs, info, ra, rb, evaluator=evaluator)
            count = int(scalar(info["num_options"]))
            if action < 0 or action >= count:
                env.close()
                raise ValueError(
                    f"illegal replay action {action}; legal range is [0,{count})")
            obs, reward, done, info = unpack_step(
                env.step(np.asarray([action])))
            if bool(scalar(done)):
                value = float(scalar(reward))
                validate_terminal_reward(value)
                if player != self.root_player:
                    value = -value
                return RestoredState(env, obs, info, ra, rb, True, value)
        return RestoredState(env, obs, info, ra, rb)

    def evaluate(self, prefix: tuple[int, ...]) -> Evaluation:
        state = self.restore(prefix, evaluator="root")
        try:
            if state.terminal:
                return Evaluation(
                    None, [], float(state.terminal_value), None, True,
                    observation_digest(state.obs), None, "terminal")
            player, _ra, _rb, logits, value = self.advance_model(
                state.obs, state.info, state.rstate_a, state.rstate_b,
                evaluator="root")
            count = int(scalar(state.info["num_options"]))
            raw_value = float(scalar(value))
            root_value = raw_value if player == self.root_player else -raw_value
            prompt_id, prompt = prompt_from_obs(state.obs)
            return Evaluation(
                player, softmax(np.asarray(logits[0, :count])).tolist(),
                root_value, raw_value, False, observation_digest(state.obs),
                prompt_id, prompt)
        finally:
            state.env.close()

    def rollout(self, prefix: tuple[int, ...], rollout_seed: int,
                max_steps: int) -> tuple[float, int]:
        state = self.restore(prefix, evaluator="per-player")
        try:
            if state.terminal:
                return float(state.terminal_value), 0
            rng = random.Random(rollout_seed)
            for step in range(max_steps):
                player, state.rstate_a, state.rstate_b, logits, _value = \
                    self.advance_model(state.obs, state.info, state.rstate_a,
                                       state.rstate_b, evaluator="per-player")
                count = int(scalar(state.info["num_options"]))
                probs = softmax(np.asarray(logits[0, :count]))
                action = rng.choices(range(count), weights=probs, k=1)[0]
                state.obs, reward, done, state.info = unpack_step(
                    state.env.step(np.asarray([action])))
                if bool(scalar(done)):
                    value = float(scalar(reward))
                    validate_terminal_reward(value)
                    if player != self.root_player:
                        value = -value
                    return value, step + 1
            raise RuntimeError("terminal rollout exceeded max_steps")
        finally:
            state.env.close()


def bounded_frontier(model: SnapshotModel, depth: int, max_leaves: int):
    frontier: list[tuple[int, ...]] = [()]
    truncated = False
    for _ in range(depth):
        next_frontier = []
        for prefix in frontier:
            ev = model.evaluate(prefix)
            if ev.terminal:
                next_frontier.append(prefix)
                continue
            for action in range(len(ev.priors)):
                if len(next_frontier) >= max_leaves:
                    truncated = True
                    break
                next_frontier.append(prefix + (action,))
            if truncated:
                break
        frontier = next_frontier
        if truncated:
            break
    return frontier, truncated


def run_terminal_enumeration(model, args):
    frontier, truncated = bounded_frontier(
        model, args.enumeration_depth, args.max_leaves)
    samples = []
    for leaf_index, prefix in enumerate(frontier):
        for rollout_index in range(args.rollouts_per_leaf):
            seed = stable_seed(
                args.seed, model.row["decision_id"], rollout_index)
            value, steps = model.rollout(prefix, seed, args.max_steps)
            samples.append({
                "prefix": list(prefix), "root_action": prefix[0],
                "rollout_seed": seed, "value": value, "steps": steps})
    actions = {}
    for action in model.row["legal_actions"]:
        rows = [x for x in samples if x["root_action"] == action]
        values = [x["value"] for x in rows]
        actions[str(action)] = {
            "frontier_leaves": len({tuple(x["prefix"]) for x in rows}),
            "samples": len(values), "mean": fmean(values),
            "stderr": stderr(values), "minimum": min(values),
            "maximum": max(values),
            "positive_terminal_paths": sum(v > 0 for v in values),
            "best_prefix": max(rows, key=lambda x: x["value"])["prefix"],
        }
    return {
        "method": "bounded-enumeration-terminal-rollout",
        "depth": args.enumeration_depth, "frontier_size": len(frontier),
        "max_leaves": args.max_leaves, "truncated": truncated,
        "rollouts_per_leaf": args.rollouts_per_leaf,
        "common_random_seeds_across_leaves": True,
        "actions": actions, "samples": samples,
    }


@dataclass
class Edge:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class Node:
    prefix: tuple[int, ...]
    player: int | None = None
    value: float | None = None
    raw_value: float | None = None
    terminal: bool = False
    observation_digest: str | None = None
    prompt_id: int | None = None
    prompt: str | None = None
    edges: dict[int, Edge] = field(default_factory=dict)

    @property
    def expanded(self):
        return self.terminal or bool(self.edges)


def expand(model, node):
    ev = model.evaluate(node.prefix)
    node.player = ev.player
    node.value = ev.value
    node.raw_value = ev.raw_value
    node.terminal = ev.terminal
    node.observation_digest = ev.observation_digest
    node.prompt_id = ev.prompt_id
    node.prompt = ev.prompt
    if not ev.terminal:
        node.edges = {i: Edge(p) for i, p in enumerate(ev.priors)}
    return ev.value


def choose_edge(node, root_player, c_puct):
    parent_visits = sum(edge.visits for edge in node.edges.values())
    perspective = 1.0 if node.player == root_player else -1.0
    return max(
        node.edges,
        key=lambda action: (
            perspective * node.edges[action].q
            + c_puct * node.edges[action].prior
            * math.sqrt(parent_visits + 1) / (node.edges[action].visits + 1),
            -action,
        ),
    )


def run_puct(model, args):
    nodes = {(): Node(())}
    root_initial_value = expand(model, nodes[()])
    simulation_audit = []
    for simulation_index in range(args.simulations):
        node = nodes[()]
        path = []
        while not node.terminal and len(node.prefix) < args.search_depth:
            newly_expanded = not node.expanded
            if newly_expanded:
                expand(model, node)
                if node.terminal:
                    break
                # A chain-response prompt is a protocol boundary, not a stable
                # position to score. Continue the same simulation until the
                # chain resolves to another prompt type.
                if node.prompt_id != 2:
                    break
            action = choose_edge(node, model.root_player, args.c_puct)
            path.append((node, action))
            prefix = node.prefix + (action,)
            node = nodes.setdefault(prefix, Node(prefix))
        value = node.value if node.expanded else expand(model, node)
        backups = []
        for parent, action in path:
            edge = parent.edges[action]
            visits_before = edge.visits
            q_before = edge.q
            edge.visits += 1
            edge.value_sum += value
            backups.append({
                "parent_prefix": list(parent.prefix), "action": action,
                "prior": edge.prior, "visits_before": visits_before,
                "visits_after": edge.visits, "q_before": q_before,
                "q_after": edge.q, "backed_up_value": value,
            })
        simulation_audit.append({
            "simulation": simulation_index,
            "action_prefix": list(node.prefix),
            "leaf_player": node.player, "terminal": node.terminal,
            "raw_value": node.raw_value,
            "root_perspective_value": value,
            "observation_digest": node.observation_digest,
            "leaf_prompt_id": node.prompt_id,
            "leaf_prompt": node.prompt,
            "resolved_chain_boundary": node.terminal or node.prompt_id != 2,
            "backups": backups,
        })
    root = nodes[()]
    total_visits = sum(edge.visits for edge in root.edges.values())
    actions = {
        str(action): {
            "prior": edge.prior, "visits": edge.visits,
            "visit_probability": edge.visits / total_visits,
            "q_leaf_value": edge.q,
        }
        for action, edge in root.edges.items()
    }
    return {
        "method": "puct-resolved-chain-leaf-value",
        "simulations": args.simulations,
        "search_depth": args.search_depth, "c_puct": args.c_puct,
        "root_initial_value": root_initial_value,
        "expanded_nodes": sum(node.expanded for node in nodes.values()),
        "actions": actions, "simulation_audit": simulation_audit,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decision", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint-a", required=True)
    p.add_argument("--checkpoint-b", required=True)
    p.add_argument("--player-a", type=int, choices=(0, 1), required=True)
    p.add_argument("--deck", required=True)
    p.add_argument("--code-list-file", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--enumeration-depth", type=int, default=2)
    p.add_argument("--max-leaves", type=int, default=512)
    p.add_argument("--rollouts-per-leaf", type=int, default=2)
    p.add_argument("--simulations", type=int, default=96)
    p.add_argument("--search-depth", type=int, default=12)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--restore-tolerance", type=float, default=1e-3)
    p.add_argument("--method", choices=("both", "enumeration", "puct"),
                   default="both")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.decision.read_text().splitlines()
            if line.strip()]
    if len(rows) != 1:
        raise ValueError("decision file must contain exactly one point")
    if args.enumeration_depth < 1:
        raise ValueError("enumeration depth must include at least the root action")
    model = SnapshotModel(rows[0], args)
    root = model.restore(())
    try:
        restored = {
            "to_play": int(scalar(root.info["to_play"])),
            "num_options": int(scalar(root.info["num_options"])),
            "observation_digest": observation_digest(root.obs),
        }
    finally:
        root.env.close()
    expected_actions = list(rows[0]["legal_actions"])
    actual_actions = list(range(restored["num_options"]))
    if actual_actions != expected_actions:
        raise RuntimeError(
            "snapshot restore mismatch: expected legal actions "
            f"{expected_actions}, got {actual_actions}")
    root_eval = model.evaluate(())
    restored["state_value"] = root_eval.raw_value
    restored["policy_probabilities"] = root_eval.priors
    expected_value = rows[0].get("state_value")
    expected_logits = rows[0].get("policy_logits")
    if expected_value is not None and abs(root_eval.raw_value - expected_value) \
            > args.restore_tolerance:
        raise RuntimeError(
            f"root value mismatch: expected {expected_value}, "
            f"got {root_eval.raw_value}")
    if expected_logits is not None:
        expected_probs = softmax(np.asarray(expected_logits)).tolist()
        restored["expected_policy_probabilities"] = expected_probs
        max_policy_error = max(
            abs(a - b) for a, b in zip(root_eval.priors, expected_probs))
        restored["max_policy_probability_error"] = max_policy_error
        if max_policy_error > args.restore_tolerance:
            raise RuntimeError(
                "root policy mismatch: maximum probability error is "
                f"{max_policy_error}")
    report = {
        "schema": "ygo-counterfactual-search-v1",
        "decision_id": rows[0]["decision_id"],
        "reward_mode": "terminal-win-loss",
        "greedy_reward": False,
        "terminal_reward_scale": {"win": 1.0, "loss": -1.0},
        "puct_evaluator": "single-root-checkpoint",
        "puct_evaluator_checkpoint": model.search_checkpoint,
        "root_player": model.root_player, "restored": restored,
    }
    if args.method in ("both", "enumeration"):
        report["terminal_enumeration"] = run_terminal_enumeration(model, args)
    if args.method in ("both", "puct"):
        report["leaf_value_puct"] = run_puct(model, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(report)
    if "terminal_enumeration" in report:
        summary["terminal_enumeration"] = dict(report["terminal_enumeration"])
        summary["terminal_enumeration"].pop("samples")
    if "leaf_value_puct" in report:
        summary["leaf_value_puct"] = dict(report["leaf_value_puct"])
        summary["leaf_value_puct"].pop("simulation_audit")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
