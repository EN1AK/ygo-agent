"""Bounded retrospective evidence inventory, not a training input exporter.

Old engine logs lack link numbers on SOLVED/DISABLED. Never attribute these
messages to the latest source or use absence of a movement as 'cannot escape'.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ACTIVATE = re.compile(r"^([01]) Activating (\w+) \((.+)\)$")
TARGET = re.compile(r"^0 .+ targets (\w+) \((.+)\)$")
MOVE = re.compile(r"^([01]) (?:Your card|You tribute|You discarded) (\w+) \(([^)]+)\)(.*)$")
DESTROY = re.compile(r"^([01]) Card (\w+) \(([^)]+)\) destroyed\.$")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bucket(value):
    return int(digest(value.encode())[:8], 16) % 5


def role(spec, perspective):
    return (1 - perspective, spec[1:]) if spec.startswith("o") else (perspective, spec)


def chains(lines):
    """Only complete chains; all evidence uses one-based source line numbers."""
    current = None
    index = 0
    for number, line in enumerate(lines, 1):
        if line.startswith("Message chaining,") and current is None:
            current = {"chain_index": index, "start_line": number, "events": [], "first_solving": None}
            index += 1
        if current is None:
            continue
        entry = {"line": number, "quote": line}
        if match := ACTIVATE.match(line):
            current["events"].append(dict(entry, kind="activation", actor=int(match[1]),
                                          spec=match[2], card=match[3]))
        elif match := TARGET.match(line):
            owner, spec = role(match[1], 0)
            sources = [e for e in current["events"] if e["kind"] == "activation"]
            source = sources[-1] if sources else None
            current["events"].append(dict(entry, kind="target", actor=owner, spec=spec, card=match[2],
                                          source_actor=source["actor"] if source else None,
                                          source_card=source["card"] if source else None,
                                          relation="self" if source and owner == source["actor"] else
                                                   "opponent" if source else "unknown"))
        elif match := MOVE.match(line):
            # Only retain public field departures or cards revealed by activation.
            actor, spec, card = int(match[1]), match[2], match[3]
            revealed = any(e["kind"] == "activation" and
                           (e["actor"], e["card"]) == (actor, card) for e in current["events"])
            if spec.startswith("m") or revealed:
                kind = ("tribute" if "You tribute" in line else
                        "discard" if "You discarded" in line else
                        "to_grave" if "sent to the graveyard" in line else "movement_other")
                current["events"].append(dict(entry, kind=kind, actor=actor, spec=spec, card=card))
        elif match := DESTROY.match(line):
            owner, spec = role(match[2], int(match[1]))
            if int(match[1]) == 0 and spec.startswith(("m", "s")):
                current["events"].append(dict(entry, kind="destroyed", actor=owner, spec=spec, card=match[3]))
        elif line.startswith(("Message chain_disabled,", "Message chain_negated,")):
            current["events"].append(dict(entry, kind="chain_disabled" if "disabled" in line else "chain_negated"))
        elif line.startswith("Message chain_solving,") and current["first_solving"] is None:
            current["first_solving"] = number
        elif line.startswith("Message chain_end,"):
            current["end_line"] = number
            yield current
            current = None


def label(window):
    events = window["events"]
    starts = [e for e in events if e["kind"] == "activation"]
    targets = [e for e in events if e["kind"] == "target"]
    cutoff = window["first_solving"] or window["end_line"]
    departures = [e for e in events if e["kind"] in ("tribute", "to_grave") and e["line"] < cutoff]
    escaped = [(t, d) for t in targets for d in departures
               if t["spec"].startswith("m") and t["line"] < d["line"] and
               (t["actor"], t["spec"], t["card"]) == (d["actor"], d["spec"], d["card"])]
    discarded = [e for e in events if e["kind"] == "discard" and e["line"] < cutoff]
    disabled = [e for e in events if e["kind"] in ("chain_disabled", "chain_negated")]
    destroyed = [e for e in events if e["kind"] == "destroyed" and e["line"] > cutoff]
    # Co-occurrence, NOT causal attribution or expected strategic utility.
    facts = {
        "target_left_before_first_resolution": True if escaped else None,
        "opponent_target_left_before_first_resolution": True if any(t["relation"] == "opponent" for t, d in escaped) else None,
        "own_target_left_before_first_resolution": True if any(t["relation"] == "self" for t, d in escaped) else None,
        "disable_or_negate_message_in_chain": bool(disabled),
        "revealed_discard_then_disable_message": bool(discarded and disabled),
        "revealed_discard_then_field_destruction": bool(discarded and destroyed),
        "effect_success": None, "cost_reason": None,
        "cannot_escape": None, "strategic_value": None,
    }
    return dict(window, labels=facts, activation_count=len(starts),
                combination=" -> ".join(e["card"] for e in starts) + " | targets=" +
                            ",".join(sorted({t["card"] for t in targets})),
                escape_evidence_lines=[[t["line"], d["line"]] for t, d in escaped],
                student_input_exported=False)


def partition(seed, combination):
    if seed == 2026104421:
        return "development_root"
    group = "test" if bucket(f"seed:{seed}") == 0 else "validation" if bucket(f"seed:{seed}") == 1 else "train"
    combo = "test" if bucket(f"combo:{combination}") == 0 else "validation" if bucket(f"combo:{combination}") == 1 else "train"
    return group if group == combo else "quarantine_cross_partition"


def collect(root, output, limit=500):
    if not 1 <= limit <= 500:
        raise ValueError("Window budget must be 1..500")
    original = json.loads((root / "delivery-verification.json").read_text())["sha256"]
    output.mkdir(parents=True, exist_ok=False)
    queues, manifest, game_ids = [], {}, set()
    for model in ("old121m", "new186m"):
        for folder in sorted((root / model).glob("attempt-*")):
            def checked(name):
                path = folder / name
                relative = path.relative_to(root).as_posix()
                data = path.read_bytes()
                if digest(data) != original[relative]:
                    raise ValueError(f"Changed evidence: {relative}")
                manifest[relative] = digest(data)
                return data
            result = json.loads(checked("result.json"))
            if result["status"] != "valid":
                raise ValueError(f"Invalid source attempt: {folder}")
            cmd = json.loads(checked("command.json"))
            seed = int(cmd[cmd.index("--seed") + 1])
            seat = int(cmd[cmd.index("--player") + 1])
            game = f"{model}/{folder.name}"
            key = (model, seed, seat)
            if key in game_ids:
                raise ValueError(f"Duplicate game identity: {key}")
            game_ids.add(key)
            lines = checked("eval.log").decode("utf-8").splitlines()
            rows = []
            for window in chains(lines):
                row = label(window)
                if not row["activation_count"]:
                    continue
                row.update(id=f"{game}/chain-{row['chain_index']:04d}", game=game,
                           seed=seed, seat=seat, group=f"seed:{seed}",
                           evidence_log=f"{game}/eval.log")
                # Outcome-enriched discovery sample; NEVER population calibration.
                score = (bool(row["labels"]["target_left_before_first_resolution"]),
                         row["labels"]["revealed_discard_then_field_destruction"] or
                         row["labels"]["revealed_discard_then_disable_message"],
                         row["activation_count"] > 1)
                row["discovery_priority"] = list(score)
                rows.append(row)
            queues.append(sorted(rows, key=lambda r: (tuple(-int(v) for v in r["discovery_priority"]), digest(r["id"].encode()))))
    selected = []
    while len(selected) < limit and any(queues):
        for queue in queues:
            if queue and len(selected) < limit:
                row = queue.pop(0)
                row["split"] = partition(row["seed"], row["combination"])
                selected.append(row)
    partitions = ("train", "validation", "test")
    for field in ("group", "combination"):
        sets = [{r[field] for r in selected if r["split"] == s} for s in partitions]
        assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode()
    (output / "windows.jsonl").write_bytes(data)
    coverage = {key: dict(Counter("unknown" if r["labels"][key] is None else str(r["labels"][key])
                                 for r in selected)) for key in selected[0]["labels"]}
    summary = {"schema": "interaction-window-inventory-v1", "windows": len(selected),
               "source_valid_duels": len(game_ids), "represented_duels": len({r["game"] for r in selected}),
               "independent_seed_clusters": len({r["seed"] for r in selected}),
               "split_counts": dict(Counter(r["split"] for r in selected)),
               "combination_counts": dict(Counter(r["combination"] for r in selected)),
               "label_coverage": coverage, "source_file_hashes": manifest,
               "source_delivery_index_sha256": digest((root / "delivery-verification.json").read_bytes()),
               "windows_sha256": digest(data), "split_rule": "sha256 seed/combo mod5: 0 test, 1 validation, else train; mismatches quarantine; 4421 development",
               "student_inputs_ready": False, "fitting_performed": False,
               "promotion": "blocked: no cannot-escape/effect-status truth or actor-visible fixtures",
               "sampling": "round-robin by duel, outcome-enriched; no population calibration claims",
               "manual_review_complete": False}
    (output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("source_file_hashes", "combination_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    collect(args.root, args.output, args.limit)
