"""Build deterministic deck TF-IDF features, clusters, and family split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _repo_bootstrap  # noqa: F401

from ygoai.deck_clustering import write_cluster_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-weight", type=float, default=1.0)
    parser.add_argument("--extra-weight", type=float, default=1.5)
    parser.add_argument("--candidate-k", type=int, nargs="+", default=list(range(32, 65, 4)))
    parser.add_argument("--min-families", type=int, default=1)
    parser.add_argument("--max-family-fraction", type=float, default=0.25)
    parser.add_argument("--max-deck-fraction", type=float, default=0.30)
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    args = parser.parse_args()
    manifest = write_cluster_artifacts(
        args.ready, args.families, args.output, args.main_weight, args.extra_weight,
        args.candidate_k, args.min_families, args.max_family_fraction,
        args.max_deck_fraction, args.holdout_fraction,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
