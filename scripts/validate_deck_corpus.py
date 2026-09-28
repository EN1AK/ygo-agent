"""Classify canonical decks against frozen engine and observation assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _repo_bootstrap  # noqa: F401

from ygoai.deck_corpus import validate_corpus_static


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--cards-db", type=Path, required=True)
    parser.add_argument("--code-list", type=Path, required=True)
    parser.add_argument("--semantic-assets", type=Path, required=True)
    parser.add_argument("--script-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_corpus_static(
        args.canonical, args.cards_db, args.code_list, args.semantic_assets, args.script_dir
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "status_counts": report["status_counts"],
        "reason_counts": report["reason_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
