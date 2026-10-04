"""Verify one packet-grounded positive Veiler example, not a strength test."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(original, replay):
    def decisions(root):
        return [x for line in (root / "decisions.jsonl").read_text().splitlines()
                if (x := json.loads(line))["record_type"] == "decision"]
    old, new = decisions(original), decisions(replay)
    assert len(old) == len(new) == 82
    for a, b in zip(old, new):
        for key in ("step", "player", "legal_actions", "selected_action", "checkpoint_sha256"):
            assert a[key] == b[key], (a["step"], key)
    manifest = json.loads((replay / "manifest.json").read_text())
    assert manifest["returncode"] == 0 and manifest["device"] == "cpu"
    for key in ("native_sha256", "checkpoint_sha256", "frozen_native_sha256"):
        assert manifest[key] == manifest[f"{key}_after"]
    links, chain, found, current = {}, 0, [], []
    for line in (replay / "chain-packets.jsonl").read_text().splitlines():
        row = json.loads(line)
        packet = bytes.fromhex(row["packet_hex"])
        message = packet[1]
        current.append(row)
        if message == 70:
            assert len(packet) == 18
            links[packet[-1]] = {"code": int.from_bytes(packet[2:6], "little"),
                                  "actor": packet[6], "packet_index": row["index"]}
        elif message == 76 and links.get(1, {}).get("code") == 63288573 and links.get(2, {}).get("code") == 97268402:
            assert packet == bytes([1, 76, 1])
            assert links[1]["actor"] == 1 and links[2]["actor"] == 0
            found.append({"chain": chain, "sources": dict(links), "disabled_link": 1,
                          "disabled_packet": row})
        elif message == 74:
            links, current = {}, []
            chain += 1
    assert len(found) == 1, found
    fixtures = []
    for path in sorted((replay / "fixtures").glob("*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == meta["sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            assert all(not a.dtype.hasobject and np.isfinite(a).all() for a in arrays.values())
        fixtures.append({"file": path.name, "sha256": sha(path)})
    assert len(fixtures) == 5
    result = {"verified": True, "same_menu_and_selected_actions": 82, "packet_evidence": found,
              "fixtures": fixtures, "policy_logits_bitwise_equal_required": False,
              "veiler_actor": "WindBot, not the model", "model_actor": "Kagari controller",
              "conclusion": "Kagari link 1 actually receives CHAIN_DISABLED after Veiler link 2 resolves.",
              "cannot_escape_proven": False, "strategic_optimality_proven": False,
              "training_performed": False,
              "hashes": {name: sha(replay / name) for name in
                         ("manifest.json", "chain-packets.jsonl", "decisions.jsonl", "engine.log")},
              "original_decisions_sha256": sha(original / "decisions.jsonl")}
    with (replay / "counterexample-verification.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("replay", type=Path)
    args = parser.parse_args()
    verify(args.original, args.replay)
