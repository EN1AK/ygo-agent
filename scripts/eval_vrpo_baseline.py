"""Freeze a paired, both-seat greedy baseline before Candidate-Q pilots."""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

from ygoai.rl.match_report import summarize_matches, terminal_outcome


EPISODE = re.compile(
    r"^Episode (\d+): length=(\d+), reward=([^,]+), win=(\d+), (.*?)"
    r"invalid_game=(\d+), termination_reason=(-?\d+)([^\r\n]*)", re.MULTILINE)
SPS = re.compile(r"SPS:\s*(\d+)")


def parse_episodes(text: str) -> list[dict]:
    """Only natural finite outcomes enter strength metrics, not timeout rewards."""
    episodes = []
    for number, length, reward, logged_win, detail, invalid, reason, tail in EPISODE.findall(text):
        reward, invalid, reason = float(reward), int(invalid), int(reason)
        outcome = terminal_outcome(reward, invalid, reason)
        if int(logged_win) != int(reward > 0):
            raise ValueError("Episode win flag disagrees with terminal reward")
        episodes.append({
            "episode": int(number), "length": int(length), "reward": reward,
            "invalid_game": invalid, "termination_reason": reason,
            "outcome": outcome,
        })
        environment = re.search(r"(?:^|,\s*)environment_index=(\d+)(?:,|$)", tail)
        if environment:
            episodes[-1]["environment_index"] = int(environment.group(1))
        win_reason = re.search(r"(?:^|,\s*)win_reason=(-?\d+)(?:,|$)", detail)
        if win_reason:
            episodes[-1]["win_reason"] = int(win_reason.group(1))
    if [row["episode"] for row in episodes] != list(range(1, len(episodes) + 1)):
        raise ValueError("Missing, reordered or duplicate episode records")
    return episodes


def validate_first_episodes(rows: list[dict], num_envs: int) -> None:
    indices = [row.get("environment_index") for row in rows]
    if None in indices or sorted(indices) != list(range(num_envs)):
        raise ValueError("Expected exactly one initial episode per environment index")


def validate_replay_decisions(path: Path, checkpoint_hash: str, seat: int, episode: dict) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) < 2 or rows[-1].get("record_type") != "terminal":
        raise ValueError("Replay decision log is incomplete")
    for index, row in enumerate(rows[:-1]):
        if (row.get("record_type") != "decision" or row.get("step") != index
                or row.get("player") != seat or row.get("checkpoint_sha256") != checkpoint_hash):
            raise ValueError("Replay decision identity mismatch")
        actions, logits, probabilities = row["legal_actions"], row["policy_logits"], row["policy_probabilities"]
        if (not actions or len(actions) != len(logits) or len(actions) != len(probabilities)
                or [a["index"] for a in actions] != list(range(len(actions)))):
            raise ValueError("Replay legal menu/logits alignment mismatch")
        if (not all(math.isfinite(x) for x in [*logits, *probabilities, row["state_value"]])
                or any(p < 0 or p > 1 for p in probabilities)
                or not math.isclose(sum(probabilities), 1., abs_tol=2e-5)):
            raise ValueError("Replay contains invalid policy/value numbers")
        if (row["selected_action"] not in range(len(actions))
                or row["selected_action"] != row["raw_selected_action"]
                or row["cycle_guard_intervened"]):
            raise ValueError("Raw baseline replay contains an invalid or assisted action")
    terminal = rows[-1]
    if (terminal["checkpoint_sha256"] != checkpoint_hash
            or terminal["steps"] != len(rows) - 1
            or terminal["terminal_reward"] != episode["reward"]
            or terminal["invalid_game"] != episode["invalid_game"]
            or terminal["termination_reason"] != episode["termination_reason"]):
        raise ValueError("Replay terminal disagrees with episode log")
    if "win_reason" in episode and terminal.get("win_reason") != episode["win_reason"]:
        raise ValueError("Replay raw win reason disagrees with episode log")
    return {"decisions": len(rows) - 1, "complete_raw_policy_log": True}


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
    if args.episodes_per_cell < 1 or args.timeout_seconds <= 0:
        parser.error("episodes-per-cell and timeout-seconds must be positive")
    if len(set(args.deck_names)) != len(args.deck_names) or len(set(args.seeds)) != len(args.seeds):
        parser.error("deck names and seeds must be unique")
    for name in ("repo", "checkpoint", "training_log", "decks", "code_list", "semantics", "native_module"):
        setattr(args, name, getattr(args, name).resolve())
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
        "schema_version": 3,
        "status": "running",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_metadata_sha256": sha256(Path(str(args.checkpoint) + ".metadata.json")),
        "source_commit": args.source_commit,
        "native_module": str(args.native_module.resolve()),
        "native_sha256": sha256(args.native_module),
        "code_list_sha256": sha256(args.code_list),
        "semantic_metadata_sha256": sha256(args.semantics / "metadata.json"),
        "procedure_sha256": sha256(args.repo / "scripts/script/procedure.lua"),
        "card_database_sha256": sha256(args.repo / "assets/locale/zh/cards.cdb"),
        "training_log_sha256": sha256(args.training_log),
        "training_telemetry": training_telemetry(args.training_log),
        "matrix": {
            "decks": args.deck_names, "seeds": args.seeds,
            "seats": [0, 1], "episodes_per_cell": args.episodes_per_cell,
            "opponent": "in-process-greedy", "max_steps": 1000,
            "search_or_belief": False,
            "collection": "first_episode_per_environment",
            "pair_key": ["deck", "seed", "environment_index"],
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
        "--no-cycle-guard",  # Keep the frozen baseline an unassisted raw-policy measurement.
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
                        "--num-envs", str(args.episodes_per_cell), "--env-threads", "32",
                        "--num-episodes", str(args.episodes_per_cell),
                        "--first-episode-per-env",
                    ]
                    start = time.monotonic()
                    with log.open("wb") as stream:
                        completed = subprocess.run(
                            command, cwd=args.repo, stdout=stream, stderr=stream,
                            timeout=args.timeout_seconds, check=False)
                    text = log.read_text(encoding="utf-8", errors="replace")
                    rows = parse_episodes(text)
                    summary = summarize_matches(rows)
                    cell = {
                        **summary,
                        "deck": deck, "deck_sha256": sha256(args.decks / f"{deck}.ydk"),
                        "seed": seed, "seat": seat, "games": len(rows),
                        "valid_wins": summary["wins"],
                        "episodes": rows,
                        "seconds": time.monotonic() - start,
                        "exit_code": completed.returncode,
                        "log": str(log), "log_sha256": sha256(log),
                    }
                    manifest["cells"].append(cell)
                    save()
                    if completed.returncode or len(rows) != args.episodes_per_cell:
                        raise RuntimeError(f"Baseline cell failed: {tag}: {cell}")
                    validate_first_episodes(rows, args.episodes_per_cell)
        manifest["status"] = "matrix_complete_replays_pending"
        save()
        for deck in dict.fromkeys((args.deck_names[0], args.deck_names[-1])):
            for seat in (0, 1):
                seed = args.seeds[0]
                tag = f"replay-{deck}-seed{seed}-seat{seat}"
                duel_dir = output / tag
                duel_dir.mkdir()
                log = duel_dir / "engine.log"
                decisions = duel_dir / "decisions.jsonl"
                replay_dir = duel_dir / "replay"
                command = [
                    sys.executable, "-u", str(args.repo / "scripts/eval_structured.py"),
                    *common, "--deck1", deck, "--deck2", deck,
                    "--seed", str(seed), "--player", str(seat),
                    "--num-envs", "1", "--num-episodes", "1", "--record",
                    "--decision-log", str(decisions),
                ]
                with log.open("wb") as stream:
                    completed = subprocess.run(
                        command, cwd=duel_dir, stdout=stream, stderr=stream,
                        timeout=args.timeout_seconds, check=False)
                new_replays = list(replay_dir.glob("*.yrp"))
                if completed.returncode or len(new_replays) != 1:
                    raise RuntimeError(f"Replay capture failed: {tag}")
                replay = new_replays[0]
                replay_rows = parse_episodes(log.read_text(encoding="utf-8", errors="replace"))
                if len(replay_rows) != 1:
                    raise RuntimeError(f"Replay terminal record missing: {tag}")
                decision_check = validate_replay_decisions(
                    decisions, manifest["checkpoint_sha256"], seat, replay_rows[0])
                manifest["replays"].append({
                    "deck": deck, "seat": seat, "seed": seed,
                    "outcome": replay_rows[0],
                    "commentary": None,
                    "decision_validation": decision_check,
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
