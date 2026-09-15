"""Import YRP2 recordings into the versioned human-demonstration manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from ygoai.rl.demonstrations import make_manifest_record, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--kind", choices=("combo", "full_duel"), required=True)
    parser.add_argument("--deck-tag")
    args = parser.parse_args()
    paths = sorted(args.input.glob("*.yrp")) if args.input.is_dir() else [args.input]
    if not paths:
        raise ValueError("no .yrp files found")
    records = [make_manifest_record(path, args.kind, args.deck_tag) for path in paths]
    write_manifest(records, args.output)
    print(f"imported {len(records)} {args.kind} demonstrations to {args.output}")


if __name__ == "__main__":
    main()
