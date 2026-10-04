"""Decode public chain capture and fixture rows; never infer strategic value."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(root, code_list):
    codes = [0] + [int(x.split()[0]) for x in code_list.read_text().splitlines()]
    expected, links, chain = [], {}, 0
    for line in (root / "chain-packets.jsonl").read_text().splitlines():
        record = json.loads(line)
        p = bytes.fromhex(record["packet_hex"])
        msg = p[1]
        if msg == 70:
            assert len(p) == 18, len(p)
            links[p[-1]] = {"code": int.from_bytes(p[2:6], "little"), "actor": p[6]}
        elif msg in (72, 73, 75, 76):
            assert len(p) == 3
            expected.append({"packet_index": record["index"], "chain": chain,
                             "event": {72: 10, 73: 11, 75: 3, 76: 3}[msg],
                             "link": p[2], "source": links.get(p[2]), "raw": record["packet_hex"]})
        elif msg == 74:
            links = {}
            chain += 1
    observed = {}
    for f in sorted((root / "fixtures").glob("*.npz")):
        meta = json.loads(f.with_suffix(".json").read_text())
        with np.load(f) as fixture:
            rows = fixture["obs__public_events_"].reshape(-1, 12)
        observed[f.stem] = [
            {"row": i, "event": int(x[0]), "link": int(x[2]),
             "code": codes[int(x[10]) * 256 + int(x[11])],
             "actor": (meta["player"] if int(x[1]) == 1 else
                       1 - meta["player"] if int(x[1]) == 2 else None),
             "raw": x.tolist()}
            for i, x in enumerate(rows) if int(x[0]) in (1, 3, 10, 11, 12)]
    result = {"expected_from_packets": expected, "observed_by_fixture": observed,
              "alignment": "Ordered evidence; repeated histories are not independent samples. Audit chain boundaries before comparison."}
    (root / "event-comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"expected_first": expected[:8], "observed_first": dict(list(observed.items())[:5])}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("code_list", type=Path)
    args = parser.parse_args()
    audit(args.root, args.code_list)
