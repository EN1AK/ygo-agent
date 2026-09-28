"""Inventory existing repository decks and historical multi-deck artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _repo_bootstrap  # noqa: F401

from ygoai.deck_corpus import parse_ydk, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--deck-dir", type=Path, required=True)
    parser.add_argument("--legacy-path", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    deck_dir = args.deck_dir.resolve()
    decks = []
    by_hash: dict[str, list[str]] = {}
    for path in sorted(deck_dir.glob("*.ydk"), key=lambda item: item.name.casefold()):
        deck = parse_ydk(path)
        digest = deck.canonical_hash()
        by_hash.setdefault(digest, []).append(path.stem)
        decks.append({
            "name": path.stem,
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "canonical_hash": digest,
            "zone_sizes": {name: sum(values.values()) for name, values in deck.zone_counts().items()},
            "source_status": "repository_top_level",
        })
    aliases = [
        {"canonical_hash": digest, "names": sorted(names)}
        for digest, names in sorted(by_hash.items()) if len(names) > 1
    ]
    artifacts = []
    for legacy in args.legacy_path:
        path = legacy.resolve()
        candidates = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
        for item in candidates:
            artifacts.append({
                "path": item.relative_to(root).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
                "source_status": "historical_read_only",
            })
    result = {
        "format_version": 1,
        "deck_count": len(decks),
        "decks": decks,
        "exact_alias_groups": aliases,
        "historical_artifact_count": len(artifacts),
        "historical_artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"deck_count": len(decks), "historical_artifact_count": len(artifacts)}))


if __name__ == "__main__":
    main()
