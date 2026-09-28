"""Freeze immutable ready/limited/excluded corpus-v1 eligibility tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _repo_bootstrap  # noqa: F401

from ygoai.deck_corpus import freeze_corpus_revision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--static-validation", type=Path, required=True)
    parser.add_argument("--runtime-validation", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--acquisition-rejected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="deck-corpus-v1")
    args = parser.parse_args()
    manifest = freeze_corpus_revision(
        args.canonical, args.static_validation, args.runtime_validation,
        args.source_manifest, args.families, args.acquisition_rejected,
        args.output, args.revision,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
