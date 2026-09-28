"""Build the frozen 25%-elfnote hierarchical deck curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _repo_bootstrap  # noqa: F401

from ygoai.deck_sampler import build_sampling_manifest, write_sampling_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--anchor-deck", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-native", type=Path, required=True)
    parser.add_argument("--reserve-ppm", type=int, default=250_000)
    args = parser.parse_args()
    manifest = build_sampling_manifest(
        args.clusters, args.split, args.canonical, args.anchor_deck, args.reserve_ppm)
    write_sampling_manifest(manifest, args.output_json, args.output_native)
    print(json.dumps({"manifest_hash": manifest["manifest_hash"],
                      "clusters": len(manifest["clusters"]),
                      "families": sum(len(row["families"]) for row in manifest["clusters"]),
                      "decks": sum(len(family["decks"]) for row in manifest["clusters"]
                                   for family in row["families"]),
                      "anchor": manifest["anchor"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
