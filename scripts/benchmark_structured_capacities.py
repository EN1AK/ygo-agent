"""Measure Structured-lite capacity demand on deterministic random-play duels.

The measurement environment deliberately uses capacities much larger than the
candidate values.  Candidate overflow rates are then computed from the
untruncated per-decision demand, so changing the candidate list does not require
rerunning the duels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import _repo_bootstrap  # noqa: F401
import numpy as np
import ygoenv

from ygoai.rl.env import VersionedObservation
from ygoai.rl.observation_schema import (
    DEFAULT_GROUP_REFERENCES,
    DEFAULT_PUBLIC_EVENTS,
    STRUCTURED_LITE_SCHEMA,
    observation_nbytes,
    tensor_contract,
)
from ygoai.utils import init_ygopro


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_capacities(value: str) -> list[int]:
    capacities = sorted({int(item) for item in value.split(",") if item.strip()})
    if not capacities or capacities[0] <= 0:
        raise argparse.ArgumentTypeError("capacities must be positive integers")
    return capacities


def percentile_summary(values: list[int]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.int64)
    if data.size == 0:
        return {"count": 0, "max": 0}
    result: dict[str, float | int] = {"count": int(data.size), "max": int(data.max())}
    for percentile in (50, 90, 95, 99, 99.9):
        key = f"p{str(percentile).replace('.', '_')}"
        result[key] = float(np.percentile(data, percentile, method="higher"))
    return result


def candidate_summary(
    demands: list[int], episode_maxima: list[int], candidates: list[int]
) -> dict[str, dict[str, float | int]]:
    decision_values = np.asarray(demands, dtype=np.int64)
    episode_values = np.asarray(episode_maxima, dtype=np.int64)
    total_demand = int(decision_values.sum())
    result = {}
    for capacity in candidates:
        decision_overflows = int(np.count_nonzero(decision_values > capacity))
        episode_overflows = int(np.count_nonzero(episode_values > capacity))
        truncated_items = int(np.maximum(decision_values - capacity, 0).sum())
        result[str(capacity)] = {
            "decision_overflows": decision_overflows,
            "decision_overflow_rate": (
                decision_overflows / int(decision_values.size)
                if decision_values.size else 0.0
            ),
            "episode_overflows": episode_overflows,
            "episode_overflow_rate": (
                episode_overflows / int(episode_values.size)
                if episode_values.size else 0.0
            ),
            "truncated_items": truncated_items,
            "truncated_item_rate": truncated_items / total_demand if total_demand else 0.0,
        }
    return result


def retained_decision_spans(
    event_sequences: list[list[int]], capacity: int
) -> dict[str, float | int]:
    spans: list[int] = []
    for values in event_sequences:
        sequence = np.asarray(values, dtype=np.int64)
        for index, current in enumerate(sequence):
            threshold = max(0, int(current) - capacity)
            left = int(np.searchsorted(sequence, threshold, side="left"))
            spans.append(index - left + 1)
    data = np.asarray(spans, dtype=np.int64)
    result: dict[str, float | int] = {
        "count": int(data.size), "min": int(data.min()), "max": int(data.max())
    }
    for percentile in (1, 5, 50, 95, 99):
        result[f"p{percentile}"] = float(
            np.percentile(data, percentile, method="higher")
        )
    return result


def git_revision(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--code-list", type=Path, required=True)
    parser.add_argument("--semantic-assets", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=63003)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--event-candidates", type=parse_capacities,
                        default=parse_capacities("8,16,24,32,48,64,96,128"))
    parser.add_argument("--group-candidates", type=parse_capacities,
                        default=parse_capacities("1,2,4,8,12,16"))
    parser.add_argument("--measurement-public-events", type=int, default=8192)
    parser.add_argument("--measurement-group-references", type=int, default=64)
    parser.add_argument("--measurement-max-options", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    deck_root = args.deck_root.resolve()
    code_list = args.code_list.resolve()
    semantic_assets = args.semantic_assets.resolve()
    deck_name, deck_names = init_ygopro(
        "YGOPro-v1", "chinese", str(deck_root), str(code_list),
        return_deck_names=True,
    )
    if deck_name != "random":
        raise ValueError("--deck-root must be a directory containing a deck pool")

    native = ygoenv.make(
        task_id="YGOPro-v1", env_type="gymnasium", num_envs=1, num_threads=1,
        seed=args.seed, deck1="random", deck2="random", player=-1,
        play_mode="random", max_options=args.measurement_max_options,
        n_history_actions=32, max_steps=args.max_steps, greedy_reward=False,
        observation_schema=STRUCTURED_LITE_SCHEMA,
        semantic_asset_dir=str(semantic_assets),
        n_public_events=args.measurement_public_events,
        max_group_references=args.measurement_group_references,
    )
    native.num_envs = 1
    env = VersionedObservation(native, STRUCTURED_LITE_SCHEMA)
    obs, infos = env.reset()
    rng = np.random.default_rng(args.seed)

    event_demands: list[int] = []
    group_demands: list[int] = []
    action_demands: list[int] = []
    event_episode_maxima: list[int] = []
    event_episode_sequences: list[list[int]] = []
    group_episode_maxima: list[int] = []
    current_event_sequence: list[int] = []
    current_event_max = 0
    current_group_max = 0
    measurement_event_overflow_samples = 0
    measurement_group_overflow_samples = 0
    action_overflow_samples = 0
    negative_win_reason_games = 0
    episodes = 0
    start = time.perf_counter()

    while episodes < args.episodes:
        events = np.asarray(obs["public_events_"])[0]
        event_demand = int(np.count_nonzero(events[:, 0]))
        group_mask = np.asarray(obs["action_group_mask_"])[0]
        mask_demand = int(group_mask.sum(axis=-1).max(initial=0))
        selected_demand = int(np.asarray(obs["selection_"])[0, 8])
        group_demand = max(mask_demand, selected_demand)
        diagnostics = np.asarray(obs["structured_diagnostics_"])[0]
        action_demand = int(np.asarray(infos["num_options"])[0])

        event_demands.append(event_demand)
        current_event_sequence.append(event_demand)
        group_demands.append(group_demand)
        action_demands.append(action_demand)
        current_event_max = max(current_event_max, event_demand)
        current_group_max = max(current_group_max, group_demand)
        measurement_event_overflow_samples += int(diagnostics[6] != 0)
        measurement_group_overflow_samples += int(np.any(diagnostics[2:6] != 0))
        action_overflow_samples += int(diagnostics[0] != 0)

        action = int(rng.integers(action_demand))
        obs, _, terminated, truncated, infos = env.step(
            np.asarray([action], dtype=np.int32)
        )
        if bool(np.logical_or(terminated, truncated)[0]):
            event_episode_maxima.append(current_event_max)
            event_episode_sequences.append(current_event_sequence)
            group_episode_maxima.append(current_group_max)
            current_event_sequence = []
            current_event_max = 0
            current_group_max = 0
            negative_win_reason_games += int(
                np.asarray(infos.get("win_reason", [0]))[0] < 0
            )
            episodes += 1

    elapsed = time.perf_counter() - start
    env.close()

    if measurement_event_overflow_samples or measurement_group_overflow_samples:
        raise RuntimeError(
            "measurement capacity overflowed; increase measurement capacities before "
            "using the candidate rates: "
            f"public_event_samples={measurement_event_overflow_samples}, "
            f"group_reference_samples={measurement_group_overflow_samples}"
        )
    if action_overflow_samples:
        raise RuntimeError("max_options overflowed during the capacity audit")

    deck_files = sorted(deck_root.glob("*.ydk"))
    semantic_metadata = semantic_assets / "metadata.json"
    contract = tensor_contract(STRUCTURED_LITE_SCHEMA)
    event_candidates = candidate_summary(
        event_demands, event_episode_maxima, args.event_candidates
    )
    for capacity, values in event_candidates.items():
        values["observation_bytes_with_default_group"] = observation_nbytes(
            STRUCTURED_LITE_SCHEMA,
            public_events=int(capacity),
            group_references=DEFAULT_GROUP_REFERENCES,
        )
        values["retained_decision_span"] = retained_decision_spans(
            event_episode_sequences, int(capacity)
        )
    group_candidates = candidate_summary(
        group_demands, group_episode_maxima, args.group_candidates
    )
    for capacity, values in group_candidates.items():
        values["observation_bytes_with_default_events"] = observation_nbytes(
            STRUCTURED_LITE_SCHEMA,
            public_events=DEFAULT_PUBLIC_EVENTS,
            group_references=int(capacity),
        )

    report = {
        "format_version": 1,
        "git_revision": git_revision(repo_root),
        "schema": STRUCTURED_LITE_SCHEMA,
        "inputs": {
            "seed": args.seed,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "deck_root": str(deck_root),
            "deck_count": len(deck_names),
            "deck_names": sorted(deck_names),
            "deck_sha256": {path.name: sha256_file(path) for path in deck_files},
            "code_list": str(code_list),
            "code_list_sha256": sha256_file(code_list),
            "semantic_assets": str(semantic_assets),
            "semantic_metadata_sha256": (
                sha256_file(semantic_metadata) if semantic_metadata.exists() else None
            ),
        },
        "measurement_capacities": {
            "max_options": args.measurement_max_options,
            "public_events": args.measurement_public_events,
            "group_references": args.measurement_group_references,
            "measurement_overflow_samples": {
                "public_events": measurement_event_overflow_samples,
                "group_references": measurement_group_overflow_samples,
                "actions": action_overflow_samples,
            },
        },
        "coverage": {
            "episodes": episodes,
            "decisions": len(event_demands),
            "negative_win_reason_games": negative_win_reason_games,
            "elapsed_seconds": elapsed,
            "decisions_per_second": len(event_demands) / elapsed,
        },
        "observed_demand": {
            "public_events": percentile_summary(event_demands),
            "group_references": percentile_summary(group_demands),
            "legal_actions": percentile_summary(action_demands),
        },
        "candidate_overflow": {
            "public_events": event_candidates,
            "group_references": group_candidates,
        },
        "current_defaults": {
            "public_events": DEFAULT_PUBLIC_EVENTS,
            "group_references": DEFAULT_GROUP_REFERENCES,
            "observation_bytes": observation_nbytes(STRUCTURED_LITE_SCHEMA),
            "manifest_shapes": {
                "public_events_": list(contract["public_events_"].shape),
                "action_group_refs_": list(contract["action_group_refs_"].shape),
            },
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

