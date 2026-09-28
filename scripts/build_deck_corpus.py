import argparse
import json
from pathlib import Path

import _repo_bootstrap  # noqa: F401

from ygoai.deck_corpus import build_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and canonicalize a pinned YDK corpus")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    args = parser.parse_args()
    result = build_corpus(
        args.registry.resolve(),
        args.project_root.resolve(),
        args.output.resolve(),
        args.cache.resolve(),
        args.near_duplicate_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

