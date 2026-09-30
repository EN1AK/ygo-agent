"""Freeze a paired, both-seat greedy baseline before Candidate-Q pilots."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


EPISODE = re.compile(
    r"^Episode \d+: .*?reward=([-+0-9.eE]+), win=(\d+), .*?"
    r"invalid_game=(\d+), termination_reason=(-?\d+)", re.MULTILINE)
SPS = re.compile(r"SPS:\s*(\d+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_telemetry(log: Path) -> dict:
    recent_sps = []
    timeout_lines = 0
    with log.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            match = SPS.search(line)
            if match:
                recent_sps.append(int(match.group(1)))
                recent_sps = recent_sps[-100:]
            if "Env max steps:" in line or "environment step timeout" in line.lower():
                timeout_lines += 1
    return {
        "last_100_logged_sps": recent_sps,
        "timeout_diagnostic_lines": timeout_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--decks", type=Path, required=True)
    parser.add_argument("--code-list", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--native-module", type=Path, required=True)
    parser.add_argument("--deck-names", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--episodes-per-cell", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to reuse baseline output: {output}")
    for path in (
        args.repo, args.checkpoint, args.training_log, args.decks,
        args.code_list, args.semantics, args.native_module,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for name in args.deck_names:
        if not (args.decks / f"{name}.ydk").is_file():
            raise FileNotFoundError(args.decks / f"{name}.ydk")
    output.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_metadata_sha256": sha256(Path(str(args.checkpoint) + ".metadata.json")),
        "source_commit": args.source_commit,
        "native_module": str(args.native_module.resolve()),
        "native_sha256": sha256(args.native_module),
        "code_list_sha256": sha256(args.code_list),
        "training_log_sha256": sha256(args.training_log),
        "training_telemetry": training_telemetry(args.training_log),
        "matrix": {
            "decks": args.deck_names, "seeds": args.seeds,
            "seats": [0, 1], "episodes_per_cell": args.episodes_per_cell,
            "opponent": "in-process-greedy", "max_steps": 1000,
            "search_or_belief": False,
        },
        "cells": [],
        "replays": [],
        "combo_completion": None,
        "interruption_quality": None,
        "notes": "Combo/interruption annotations require separate decision-log review.",
    }
    manifest_path = output / "baseline-manifest.json"

    def save():
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    save()
    common = [
        "--deck", str(args.decks), "--code-list-file", str(args.code_list),
        "--observation-schema", "structured-lite-v1",
        "--semantic-asset-dir", str(args.semantics),
        "--max-options", "128", "--n-history-actions", "32",
        "--n-public-events", "32", "--max-group-references", "8",
        "--max-steps", "1000", "--bot-type", "greedy",
        "--checkpoint", str(args.checkpoint),
    ]
    try:
        for deck in args.deck_names:
            for seed in args.seeds:
                for seat in (0, 1):
                    tag = f"{deck}-seed{seed}-seat{seat}"
                    log = output / f"{tag}.log"
                    command = [
                        sys.executable, "-u", str(args.repo / "scripts/eval_structured.py"),
                        *common, "--deck1", deck, "--deck2", deck,
                        "--seed", str(seed), "--player", str(seat),
                        "--num-envs", "32", "--env-threads", "32",
                        "--num-episodes", str(args.episodes_per_cell),
                    ]
                    start = time.monotonic()
                    with log.open("wb") as stream:
                        completed = subprocess.run(
                            command, cwd=args.repo, stdout=stream, stderr=stream,
                            timeout=args.timeout_seconds, check=False)
                    text = log.read_text(encoding="utf-8", errors="replace")
                    rows = EPISODE.findall(text)
                    cell = {
                        "deck": deck, "deck_sha256": sha256(args.decks / f"{deck}.ydk"),
                        "seed": seed, "seat": seat, "games": len(rows),
                        "valid_games": sum(int(row[2]) == 0 for row in rows),
                        "valid_wins": sum(int(row[1]) == 1 and int(row[2]) == 0 for row in rows),
                        "invalid_games": sum(int(row[2]) != 0 for row in rows),
                        "termination_reasons": {
                            str(reason): sum(row[3] == reason for row in rows)
                            for reason in sorted(set(row[3] for row in rows))
                        },
                        "seconds": time.monotonic() - start,
                        "exit_code": completed.returncode,
                        "log": str(log), "log_sha256": sha256(log),
                    }
                    manifest["cells"].append(cell)
                    save()
                    if completed.returncode or len(rows) != args.episodes_per_cell:
                        raise RuntimeError(f"Baseline cell failed: {tag}: {cell}")
        manifest["status"] = "matrix_complete_replays_pending"
        save()
        for deck in (args.deck_names[0], args.deck_names[-1]):
            for seat in (0, 1):
                seed = args.seeds[0]
                tag = f"replay-{deck}-seed{seed}-seat{seat}"
                log = output / f"{tag}.log"
                decisions = output / f"{tag}.jsonl"
                replay_dir = args.repo / "replay"
                before = set(replay_dir.glob("*.yrp")) if replay_dir.exists() else set()
                command = [
                    sys.executable, "-u", str(args.repo / "scripts/eval_structured.py"),
                    *common, "--deck1", deck, "--deck2", deck,
                    "--seed", str(seed), "--player", str(seat),
                    "--num-envs", "1", "--num-episodes", "1", "--record",
                    "--decision-log", str(decisions),
                ]
                with log.open("wb") as stream:
                    completed = subprocess.run(
                        command, cwd=args.repo, stdout=stream, stderr=stream,
                        timeout=args.timeout_seconds, check=False)
                new_replays = set(replay_dir.glob("*.yrp")) - before
                if completed.returncode or len(new_replays) != 1:
                    raise RuntimeError(f"Replay capture failed: {tag}")
                replay = output / f"{tag}.yrp"
                shutil.copy2(new_replays.pop(), replay)
                manifest["replays"].append({
                    "deck": deck, "seat": seat, "seed": seed,
                    "replay": str(replay), "replay_sha256": sha256(replay),
                    "log": str(log), "log_sha256": sha256(log),
                    "decisions": str(decisions), "decisions_sha256": sha256(decisions),
                })
                save()
        manifest["status"] = "evidence_complete_manual_metrics_pending"
        save()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        save()
        raise


if __name__ == "__main__":
    main()
