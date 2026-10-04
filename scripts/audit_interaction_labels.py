"""Curate the three existing development branches; not a general label miner.

Only explicit public log statements are facts. No actor, engine, or GPU loads.
Full engine logs are evidence, NEVER model inputs (they can expose private data).
"""
import argparse
import hashlib
import json
from pathlib import Path


ACTOR = "dc29c11254c0418a4b43cc29083db3d05e17b5c8e3dacd5c27b17d63c8670721"
GROUP = "seed2026104421-player1-shared-root"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(lines, fragment, start=1, end=400):
    matches = [i + 1 for i, text in enumerate(lines)
               if start <= i + 1 <= end and fragment in text]
    if len(matches) != 1:
        raise ValueError(f"Expected one evidence anchor: {fragment!r}: {matches}")
    number = matches[0]
    return {"line": number, "quote": lines[number - 1]}


def fact(value, *anchors):
    if not anchors:
        raise ValueError("An observed fact needs evidence")
    return {"value": value, "status": "observed", "evidence": list(anchors)}


def unknown(reason):
    return {"value": None, "status": "unknown", "reason": reason}


def curate(root, output):
    index = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for name, expected in index.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError(f"Evidence hash/path failed: {name}")
    records = []
    for branch, steps in (("use_now", [1]), ("hold", [1, 2]), ("hold_two", [1, 2])):
        folder = root / branch
        log = folder / "eval.log"
        lines = log.read_text(encoding="utf-8").splitlines()
        decisions = [json.loads(x) for x in (folder / "decisions.jsonl").read_text().splitlines()]
        rows = {x["step"]: x for x in decisions if x.get("record_type") == "decision"}
        selected = []
        for step in steps:
            row = rows[step]
            assert row["checkpoint_sha256"] == ACTOR and row["player"] == 1
            action = next(x for x in row["legal_actions"] if x["index"] == row["selected_action"])
            selected.append({"step": step, "selected_action": row["selected_action"],
                             "cancel": bool(action["structured_features"][12]),
                             "intervened": row["raw_selected_action"] != row["selected_action"]})
        assert [x["cancel"] for x in selected] == {
            "use_now": [False], "hold": [True, False], "hold_two": [True, True]}[branch]
        labels = {
            "cost_attribution": unknown("Public movement alone does not establish a cost reason; no cost-reason packet or Lua audit attached."),
            "negation_applied_to_raye": unknown("CHAIN_SOLVED is not effect success; no status-query proof attached."),
            "strategically_optimal": unknown("One continuation per branch is not expected value."),
            "later_effective_use_of_held_veiler": unknown("Not demonstrated by this local window."),
        }
        if branch != "hold_two":
            sent = evidence(lines, "1 Your card h4 (效果遮蒙者) was sent to the graveyard.")
            target = evidence(lines, "1 Agent targets om2 (闪刀姬-零衣)")
            response = evidence(lines, "0 Activating m2 (闪刀姬-零衣)")
            release = evidence(lines, "0 You tribute m2 (闪刀姬-零衣).")
            summon = evidence(lines, "0 You special summoning 闪刀姬-燎里")
            recovered = evidence(lines, "0 Card g2 (闪刀机-大黄蜂浮游单元) returned to hand.")
            ending = evidence(lines, "Message chain_end", release["line"], summon["line"] + 8)
            assert sent["line"] < target["line"] < response["line"] < release["line"] < summon["line"] < ending["line"] < recovered["line"]
            labels.update({
                "own_veiler_moved_hand_to_grave": fact(True, sent),
                "opponent_response": fact("raye_activated_and_tributed", response, release),
                "selected_target_left_before_chain_end": fact(True, target, release, ending),
                "public_followup": fact("kagari_summoned_and_hornet_returned", summon, recovered),
            })
        else:
            cancels = [evidence(lines, "Player 1 chose \"{act=Cancel}\"", a, b)
                       for a, b in ((229, 235), (256, 262))]
            turn = evidence(lines, "1 Your turn.", 262, 280)
            held = evidence(lines, "Summon 效果遮蒙者 in face-up attack position", 377, 385)
            assert cancels[-1]["line"] < turn["line"] < held["line"]
            labels.update({
                "declined_both_windows": fact(True, *cancels),
                "veiler_present_in_own_next_turn_hand_menu": fact(True, turn, held),
                "target_escape_after_veiler": {"value": None, "status": "not_applicable",
                                               "reason": "No Veiler activation in these two windows."},
            })
        records.append({"id": branch, "duel_group": GROUP, "split": "development_only",
                        "checkpoint_sha256": ACTOR, "decisions": selected,
                        "evidence_log": {"path": f"{branch}/eval.log", "sha256": sha(log)},
                        "decision_log_sha256": sha(folder / "decisions.jsonl"),
                        "labels": labels, "student_input_exported": False,
                        "runtime_semantics": "legacy-a2db; labels from explicit public log statements, not event source rows"})
    result = {"schema": "interaction-facts-v1", "independent_duel_roots": 1,
              "branches": 3, "training_ready": False,
              "reason": "One development root; no independent counterexample or held-out combination.",
              "source_index_sha256": sha(root / "SHA256SUMS.json"),
              "verified_source_files": len(index), "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(output), "sha256": sha(output),
                      "independent_roots": 1, "branches": 3, "training_ready": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    curate(args.root, args.output)
