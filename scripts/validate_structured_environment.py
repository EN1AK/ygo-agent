"""Real-engine smoke and visibility checks for both observation schemas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import _repo_bootstrap  # noqa: F401
import numpy as np
import ygoenv

from ygoai.rl.env import VersionedObservation
from ygoai.rl.observation_schema import LEGACY_SCHEMA, STRUCTURED_LITE_SCHEMA, tensor_contract
from ygoai.utils import init_ygopro


def digest(observation: dict[str, np.ndarray]) -> str:
    result = hashlib.sha256()
    for key in sorted(observation):
        value = np.asarray(observation[key])
        result.update(key.encode())
        result.update(str(value.dtype).encode())
        result.update(str(value.shape).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def make_env(args, schema: str, max_options: int = 24):
    native = ygoenv.make(
        task_id="YGOPro-v1", env_type="gymnasium", num_envs=1, num_threads=1,
        seed=args.seed, deck1=args.deck_name, deck2=args.deck_name,
        player=0, play_mode="bot", max_options=max_options,
        n_history_actions=32, max_steps=1000, greedy_reward=False,
        observation_schema=schema, semantic_asset_dir=str(args.semantic_assets),
        n_public_events=32, max_group_references=8,
    )
    native.num_envs = 1
    return VersionedObservation(native, schema)


def unpack_reset(result):
    return result[0] if isinstance(result, tuple) else result


def unpack_step(result):
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, np.logical_or(terminated, truncated), info
    obs, reward, done, info = result
    return obs, reward, done, info


def run_duel(args, schema: str) -> dict[str, object]:
    env = make_env(args, schema)
    obs = unpack_reset(env.reset())
    expected_keys = set(tensor_contract(schema))
    if set(obs) != expected_keys:
        raise AssertionError(f"{schema} keys mismatch: {set(obs) ^ expected_keys}")
    trace = [digest(obs)]
    event_types: set[int] = set()
    hidden_rows = hidden_semantic_violations = unsafe_event_refs = 0
    unsafe_examples = []
    overflow_samples = 0
    for step in range(1000):
        if schema == STRUCTURED_LITE_SCHEMA:
            cards = np.asarray(obs["cards_"])[0]
            visible_ids = np.asarray(obs["visible_card_ids_"])[0]
            hidden = np.logical_and(cards[:, 2] != 0,
                                    np.all(visible_ids == 0, axis=1))
            hidden_rows += int(hidden.sum())
            semantics = np.asarray(obs["card_semantics_"])[0]
            tags = np.asarray(obs["effect_tags_"])[0]
            hidden_semantic_violations += int(np.any(semantics[hidden] != 0, axis=1).sum())
            hidden_semantic_violations += int(np.any(tags[hidden] != 0, axis=1).sum())
            events = np.asarray(obs["public_events_"])[0]
            event_types.update(int(x) for x in events[:, 0] if x)
            refs = np.asarray(obs["public_event_refs_"])[0, :, :, 1]
            for event_index, event_refs in enumerate(refs):
                for index in event_refs[event_refs != 0]:
                    if index > len(cards):
                        unsafe_event_refs += 1
                        if len(unsafe_examples) < 8:
                            unsafe_examples.append({
                                "reason": "out_of_range", "index": int(index),
                                "event": events[event_index].tolist(),
                            })
                    elif (np.any(events[event_index, 10:12] != 0) and
                          not np.array_equal(visible_ids[index - 1],
                                             events[event_index, 10:12])):
                        unsafe_event_refs += 1
                        if len(unsafe_examples) < 8:
                            unsafe_examples.append({
                                "reason": "identity_mismatch", "index": int(index),
                                "event": events[event_index].tolist(),
                                "visible_card_id": visible_ids[index - 1].tolist(),
                            })
            overflow_samples += int(np.asarray(obs["structured_diagnostics_"])[0, 0] != 0)
        obs, reward, done, info = unpack_step(env.step(np.asarray([0], dtype=np.int32)))
        trace.append(digest(obs))
        if bool(np.asarray(done)[0]):
            break
    else:
        raise AssertionError(f"{schema} duel did not terminate")
    env.close()
    return {
        "schema": schema, "steps": len(trace) - 1, "trace_sha256": digest_trace(trace),
        "terminal_reward": float(np.asarray(reward)[0]), "event_types": sorted(event_types),
        "hidden_rows_checked": hidden_rows,
        "hidden_semantic_violations": hidden_semantic_violations,
        "unsafe_event_references": unsafe_event_refs,
        "unsafe_event_examples": unsafe_examples,
        "action_overflow_samples": overflow_samples,
    }


def digest_trace(trace: list[str]) -> str:
    return hashlib.sha256("\n".join(trace).encode()).hexdigest()


def overflow_probe(args) -> dict[str, object]:
    env = make_env(args, STRUCTURED_LITE_SCHEMA, max_options=2)
    obs = unpack_reset(env.reset())
    observed = bool(np.asarray(obs["structured_diagnostics_"])[0, 0])
    steps = 0
    while not observed and steps < 128:
        obs, _, done, _ = unpack_step(env.step(np.asarray([0], dtype=np.int32)))
        observed = bool(np.asarray(obs["structured_diagnostics_"])[0, 0])
        steps += 1
        if bool(np.asarray(done)[0]):
            break
    env.close()
    if not observed:
        raise AssertionError("max_options overflow was silently accepted")
    return {"capacity": 2, "overflow_observed": observed, "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--code-list", type=Path, required=True)
    parser.add_argument("--semantic-assets", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71900845)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.deck_name = init_ygopro(
        "YGOPro-v1", "chinese", str(args.deck.resolve()), str(args.code_list.resolve()))
    legacy_a = run_duel(args, LEGACY_SCHEMA)
    legacy_b = run_duel(args, LEGACY_SCHEMA)
    structured_a = run_duel(args, STRUCTURED_LITE_SCHEMA)
    structured_b = run_duel(args, STRUCTURED_LITE_SCHEMA)
    if legacy_a["trace_sha256"] != legacy_b["trace_sha256"]:
        raise AssertionError("legacy fixed-seed duel is not deterministic")
    if structured_a["trace_sha256"] != structured_b["trace_sha256"]:
        raise AssertionError("Structured-lite fixed-seed duel is not deterministic")
    for report in (structured_a, structured_b):
        if report["hidden_semantic_violations"] or report["unsafe_event_references"]:
            raise AssertionError(f"Structured-lite visibility check failed: {report}")
    report = {
        "legacy": legacy_a, "structured_lite": structured_a,
        "overflow_probe": overflow_probe(args), "passed": True,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
