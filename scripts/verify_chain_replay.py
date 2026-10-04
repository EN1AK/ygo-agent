"""Compare frozen/candidate replay evidence without equating policy outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(old, new):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    old_manifest, new_manifest = read(old / "manifest.json"), read(new / "manifest.json")
    assert old_manifest["checkpoint_sha256"] == new_manifest["checkpoint_sha256"]
    assert new_manifest["returncode"] == 0
    assert new_manifest["candidate"]
    assert new_manifest["frozen_native_sha256_after"] == old_manifest["native_sha256"]
    assert new_manifest["native_sha256_after"] == new_manifest["native_sha256"]
    assert new_manifest["checkpoint_sha256_after"] == new_manifest["checkpoint_sha256"]
    packets = [p.read_bytes() for p in (old / "chain-packets.jsonl", new / "chain-packets.jsonl")]
    assert packets[0] == packets[1], "Different packet trajectories; compare windows manually"
    decisions = [[json.loads(line) for line in (root / "decisions.jsonl").read_text().splitlines()
                  if json.loads(line).get("record_type") == "decision"] for root in (old, new)]
    assert len(decisions[0]) == len(decisions[1]) > 0
    for before, after in zip(*decisions):
        for key in ("step", "player", "checkpoint_sha256", "legal_actions", "selected_action"):
            assert before[key] == after[key], (before["step"], key)
    fixture_count = 0
    changed_event_fixtures = []
    for file in sorted((old / "fixtures").glob("*.npz")):
        with np.load(file) as before, np.load(new / "fixtures" / file.name) as after:
            assert set(before.files) == set(after.files)
            for key in before.files:
                if key.startswith("obs__") and key != "obs__public_events_":
                    assert np.array_equal(before[key], after[key]), (file.name, key)
            if not np.array_equal(before["obs__public_events_"], after["obs__public_events_"]):
                changed_event_fixtures.append(file.name)
        fixture_count += 1
    old_rows = read(old / "event-comparison.json")["observed_by_fixture"]["step-000002"]
    new_rows = read(new / "event-comparison.json")["observed_by_fixture"]["step-000002"]
    for index, event in ((19, 10), (20, 11)):
        before = next(x for x in old_rows if x["row"] == index)
        after = next(x for x in new_rows if x["row"] == index)
        assert (before["code"], before["actor"]) == (26077387, 0)
        assert (after["event"], after["link"], after["code"], after["actor"]) == (event, 1, 97268402, 1)
    hashes = {str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest()
              for root in (old, new)
              for name in ("manifest.json", "chain-packets.jsonl", "decisions.jsonl", "event-comparison.json")}
    result = {"passed": True, "decisions_same_menu_and_action": len(decisions[0]),
              "fixtures_same_non_event_observations": fixture_count,
              "changed_event_fixtures": changed_event_fixtures,
              "chain_packets_identical": True, "veiler_rows_19_20_corrected": True,
              "policy_logits_equal_required": False, "evidence_sha256": hashes,
              "scope": "One replay only; no strength or general policy equivalence claim."}
    (new / "paired-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args()
    verify(args.old, args.new)
