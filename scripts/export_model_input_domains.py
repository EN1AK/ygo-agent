#!/usr/bin/env python3
"""Export the model-facing categorical domain contract deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from ygoai.rl.observation_schema import model_input_domains

    payload = {
        "schema_version": 1,
        "bound_semantics": "exclusive_upper_bound_unless_range_string",
        "domains": model_input_domains(),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["contract_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
