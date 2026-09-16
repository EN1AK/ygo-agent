"""Generate and validate committed observation-schema manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _repo_bootstrap  # noqa: F401

from ygoai.rl.observation_schema import (
    LEGACY_SCHEMA,
    STRUCTURED_LITE_SCHEMA,
    manifest,
    observation_nbytes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("assets/observation-schema"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for schema in (LEGACY_SCHEMA, STRUCTURED_LITE_SCHEMA):
        data = manifest(schema)
        data["observation_bytes"] = observation_nbytes(schema)
        output = args.output_dir / f"{schema}.json"
        output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{schema}: {data['observation_bytes']} bytes -> {output}")


if __name__ == "__main__":
    main()
