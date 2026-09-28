"""Run a resumable real-engine self-play smoke duel for every static-ready deck."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import _repo_bootstrap  # noqa: F401
import numpy as np
import ygoenv

from ygoai.deck_corpus import materialize_canonical_decks, sha256_file
from ygoai.rl.env import VersionedObservation
from ygoai.rl.observation_schema import STRUCTURED_LITE_SCHEMA
from ygoai.utils import init_ygopro


def scalar(value):
    return np.asarray(value).reshape(-1)[0]


def deck_seed(base_seed: int, deck_id: str) -> int:
    return (base_seed + int(deck_id.removeprefix("deck-")[:8], 16)) % (2**31 - 1)


def observation_digest(observation) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        value = np.asarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def run_duel(args, deck_id: str, seed: int) -> dict[str, object]:
    started = time.monotonic()
    env = None
    try:
        native = ygoenv.make(
            task_id="YGOPro-v1", env_type="gymnasium", num_envs=1, num_threads=1,
            seed=seed, deck1=deck_id, deck2=deck_id, player=-1, play_mode="self",
            async_reset=False, max_options=args.max_options,
            n_history_actions=32, max_steps=args.max_steps, timeout=args.timeout,
            observation_schema=STRUCTURED_LITE_SCHEMA,
            semantic_asset_dir=str(args.semantic_assets.resolve()),
            n_public_events=32, max_group_references=8,
        )
        native.num_envs = 1
        env = VersionedObservation(native, STRUCTURED_LITE_SCHEMA)
        reset = env.reset()
        if not isinstance(reset, tuple) or len(reset) != 2:
            raise RuntimeError("incomplete_reset: expected (observation, info)")
        observation, info = reset
        if "num_options" not in info or "to_play" not in info:
            raise RuntimeError("incomplete_reset: missing num_options/to_play")
        initial_digest = observation_digest(observation)
        seat_decisions = Counter()
        done = False
        terminal_reward = 0.0
        for step in range(args.max_steps):
            option_count = int(scalar(info["num_options"]))
            player = int(scalar(info["to_play"]))
            if option_count < 1 or option_count > args.max_options:
                raise RuntimeError(
                    f"illegal_action_space: num_options={option_count}, max={args.max_options}"
                )
            if player not in (0, 1):
                raise RuntimeError(f"invalid_acting_seat: {player}")
            actions = np.asarray(observation["actions_"])[0]
            if np.any(actions[option_count:, 3] != 0):
                raise RuntimeError(
                    "stale_action_padding: nonzero prompt id beyond num_options"
                )
            cards = np.asarray(observation["cards_"])[0]
            padding = cards[:, 2] == 0
            if np.any(cards[padding] != 0):
                raise RuntimeError(
                    "stale_card_padding: nonzero data in unused card row"
                )
            seat_decisions[player] += 1
            result = env.step(np.asarray([0], dtype=np.int32))
            if len(result) == 5:
                observation, reward, terminated, truncated, info = result
                done = bool(scalar(terminated)) or bool(scalar(truncated))
            else:
                observation, reward, raw_done, info = result
                done = bool(scalar(raw_done))
            terminal_reward = float(scalar(reward))
            if done:
                break
        if not done:
            raise RuntimeError(f"incomplete_game: exceeded {args.max_steps} decisions")
        missing_seats = sorted({0, 1} - set(seat_decisions))
        if missing_seats:
            raise RuntimeError(f"incomplete_seat_coverage: missing={missing_seats}")
        return {
            "status": "passed", "seed": seed, "steps": step + 1,
            "seat_decisions": {str(key): seat_decisions[key] for key in (0, 1)},
            "terminal_reward": terminal_reward, "initial_observation_sha256": initial_digest,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    except Exception as exc:  # quarantine all native/Python failures instead of scoring them
        message = str(exc)
        if message.startswith("incomplete_reset"):
            reason = "incomplete_reset"
        elif message.startswith("illegal_action") or message.startswith("invalid_acting"):
            reason = "illegal_action"
        elif message.startswith("incomplete_game") or message.startswith("incomplete_seat"):
            reason = "incomplete_game"
        elif "timeout" in message.casefold():
            reason = "timeout"
        else:
            reason = "engine_exception"
        return {
            "status": "quarantined", "seed": seed, "reason": reason,
            "error_type": type(exc).__name__, "error": message,
            "traceback": traceback.format_exc(limit=8),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    finally:
        if env is not None:
            env.close()


def write_report(path: Path, report: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_worker(args) -> None:
    worklist = json.loads(args.worklist.read_text(encoding="utf-8"))
    _, registered = init_ygopro(
        "YGOPro-v1", "chinese", str(args.deck_dir.resolve()),
        str(args.code_list.resolve()), return_deck_names=True,
    )
    missing = sorted(set(worklist) - set(registered))
    if missing:
        raise RuntimeError(f"native registration missed decks: {missing[:8]}")
    with args.worker_results.open("a", encoding="utf-8", buffering=1) as stream:
        for deck_id in worklist:
            seed = deck_seed(args.seed, deck_id)
            result = run_duel(args, deck_id, seed)
            stream.write(json.dumps({"canonical_id": deck_id, "result": result}) + "\n")
            stream.flush()


def worker_command(args) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--canonical", str(args.canonical.resolve()),
        "--static-validation", str(args.static_validation.resolve()),
        "--deck-dir", str(args.deck_dir.resolve()),
        "--code-list", str(args.code_list.resolve()),
        "--semantic-assets", str(args.semantic_assets.resolve()),
        "--output", str(args.output.resolve()),
        "--worklist", str(args.worklist.resolve()),
        "--worker-results", str(args.worker_results.resolve()),
        "--seed", str(args.seed), "--max-steps", str(args.max_steps),
        "--max-options", str(args.max_options), "--timeout", str(args.timeout),
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--static-validation", type=Path, required=True)
    parser.add_argument("--deck-dir", type=Path, required=True)
    parser.add_argument("--code-list", type=Path, required=True)
    parser.add_argument("--semantic-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23092026)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-options", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worklist", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-results", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    canonical = json.loads(args.canonical.read_text(encoding="utf-8"))
    validation = json.loads(args.static_validation.read_text(encoding="utf-8"))
    ready = sorted(item["canonical_id"] for item in validation["decks"] if item["status"] == "ready")
    if args.limit is not None:
        ready = ready[:args.limit]
    deck_hashes = materialize_canonical_decks(canonical, ready, args.deck_dir)
    report = {
        "schema_version": 1,
        "validation_stage": "native-self-play-both-seats-v1",
        "inputs": {
            "canonical_decks_sha256": sha256_file(args.canonical),
            "static_validation_sha256": sha256_file(args.static_validation),
            "code_list_sha256": sha256_file(args.code_list),
            "semantic_metadata_sha256": sha256_file(args.semantic_assets / "metadata.json"),
        },
        "configuration": {
            "base_seed": args.seed, "max_steps": args.max_steps,
            "max_options": args.max_options, "timeout_seconds": args.timeout,
            "action_policy": "always-first-legal", "play_mode": "self",
        },
        "deck_hashes": deck_hashes,
        "decks": {},
    }
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("inputs") != report["inputs"] or previous.get("configuration") != report["configuration"]:
            raise RuntimeError("refusing to resume runtime validation with different inputs/config")
        report["decks"] = previous.get("decks", {})
        report["decks"] = {
            deck_id: result for deck_id, result in report["decks"].items()
            if not (
                result.get("reason") == "engine_exception"
                and str(result.get("error", "")).startswith("Seed should be in range of int32")
            )
        }

    args.worklist = args.output.with_suffix(".worklist.json")
    args.worker_results = args.output.with_suffix(".worker.jsonl")
    while True:
        pending = [deck_id for deck_id in ready if deck_id not in report["decks"]]
        if not pending:
            break
        args.worklist.write_text(json.dumps(pending) + "\n", encoding="utf-8")
        args.worker_results.write_text("", encoding="utf-8")
        completed = subprocess.run(worker_command(args), text=True, capture_output=True)
        newly_completed = []
        for line in args.worker_results.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            report["decks"][row["canonical_id"]] = row["result"]
            newly_completed.append(row["canonical_id"])
        if completed.returncode != 0:
            crashed = next((deck_id for deck_id in pending if deck_id not in newly_completed), None)
            if crashed is None:
                raise RuntimeError(f"worker failed after completing worklist: {completed.stderr[-2000:]}")
            report["decks"][crashed] = {
                "status": "quarantined", "reason": "engine_process_abort",
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
            }
            print(f"quarantined {crashed}: worker exit {completed.returncode}", flush=True)
        elif len(newly_completed) != len(pending):
            raise RuntimeError("worker exited successfully without completing its worklist")
        write_report(args.output, report)
        counts = Counter(item["status"] for item in report["decks"].values())
        print(f"{len(report['decks'])}/{len(ready)} {dict(sorted(counts.items()))}", flush=True)
    counts = Counter(item["status"] for item in report["decks"].values())
    reasons = Counter(
        item.get("reason") for item in report["decks"].values()
        if item["status"] != "passed"
    )
    report["status_counts"] = dict(sorted(counts.items()))
    report["quarantine_reason_counts"] = dict(sorted(reasons.items()))
    write_report(args.output, report)
    print(json.dumps({"status_counts": report["status_counts"],
                      "quarantine_reason_counts": report["quarantine_reason_counts"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
