"""Select decision points and aggregate counterfactual rollout results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _repo_bootstrap  # noqa: F401

from ygoai.rl.counterfactual import (
    RolloutSample, aggregate_samples, load_decision_points, mine_errors,
    select_decision_points, write_jsonl,
)


def select(args) -> None:
    points = select_decision_points(load_decision_points(args.trace),
                                    limit=args.limit,
                                    min_priority=args.min_priority)
    write_jsonl(args.output, ({
        "trace_id": p.trace_id, "decision_id": p.decision_id, "step": p.step,
        "player": p.player, "legal_actions": p.legal_actions,
        "selected_action": p.selected_action, "policy_logits": p.policy_logits,
        "state_value": p.state_value, "snapshot": p.snapshot,
        "context": p.context,
    } for p in points))


def aggregate(args) -> None:
    points = {p.decision_id: p for p in load_decision_points(args.decisions)}
    grouped: dict[str, list[RolloutSample]] = {key: [] for key in points}
    with args.rollouts.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped[row["decision_id"]].append(RolloutSample(
                action=int(row["action"]), particle=int(row["particle"]),
                rollout_seed=int(row["rollout_seed"]), value=float(row["value"])))
    results = [aggregate_samples(points[key], grouped[key],
                                 temperature=args.temperature,
                                 preference_margin=args.preference_margin)
               for key in points]
    write_jsonl(args.output, (r.to_json() for r in results))
    if args.errors:
        write_jsonl(args.errors, (r.to_json() for r in mine_errors(
            results, min_regret=args.min_regret)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    p = commands.add_parser("select")
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--min-priority", type=float, default=0.0)
    p.set_defaults(func=select)
    p = commands.add_parser("aggregate")
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--errors", type=Path)
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--preference-margin", type=float, default=0.05)
    p.add_argument("--min-regret", type=float, default=0.15)
    p.set_defaults(func=aggregate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

