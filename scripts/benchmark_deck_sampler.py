"""Benchmark native manifest reset selection against native uniform random."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import _repo_bootstrap  # noqa: F401
import ygoenv

from ygoai.utils import init_ygopro


def make(args, deck_name: str):
    env = ygoenv.make(
        task_id="YGOPro-v1", env_type="gymnasium", num_envs=1, num_threads=1,
        seed=args.seed, deck1=deck_name, deck2=deck_name, player=-1, play_mode="self",
        async_reset=False, max_options=64, n_history_actions=32,
        observation_schema="structured-lite-v1",
        semantic_asset_dir=str(args.semantic_assets.resolve()),
        n_public_events=32, max_group_references=8,
        deck_sampling_manifest=str(args.manifest.resolve()),
        deck_sampler_seed=str(args.sampler_seed), deck_sampler_counters="0",
    )
    env.num_envs = 1
    return env


def measure(env, warmup: int, resets: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        env.reset()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(resets):
            env.reset()
        samples.append((time.perf_counter_ns() - started) / resets / 1000.0)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decks", type=Path, required=True)
    parser.add_argument("--code-list", type=Path, required=True)
    parser.add_argument("--semantic-assets", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--resets", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2309)
    parser.add_argument("--sampler-seed", type=int, default=23092026)
    parser.add_argument("--max-overhead-us", type=float, default=500.0)
    parser.add_argument("--max-overhead-ratio", type=float, default=1.10)
    args = parser.parse_args()
    init_ygopro("YGOPro-v1", "chinese", str(args.decks.resolve()),
                str(args.code_list.resolve()))
    uniform = make(args, "random")
    manifest = make(args, "manifest")
    uniform_samples = measure(uniform, args.warmup, args.resets, args.repeats)
    manifest_samples = measure(manifest, args.warmup, args.resets, args.repeats)
    uniform.close()
    manifest.close()
    uniform_median = statistics.median(uniform_samples)
    manifest_median = statistics.median(manifest_samples)
    overhead = manifest_median - uniform_median
    report = {
        "schema_version": 1, "unit": "microseconds_per_reset",
        "warmup": args.warmup, "resets_per_repeat": args.resets, "repeats": args.repeats,
        "uniform_samples": uniform_samples, "manifest_samples": manifest_samples,
        "uniform_median": uniform_median, "manifest_median": manifest_median,
        "overhead": overhead, "overhead_ratio": manifest_median / uniform_median,
        "acceptance_budget_us": args.max_overhead_us,
        "acceptance_ratio": args.max_overhead_ratio,
        "passed": overhead <= args.max_overhead_us and manifest_median / uniform_median <= args.max_overhead_ratio,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
