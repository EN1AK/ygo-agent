"""Record the immutable 40M parent and assets for multi-deck continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-metadata", type=Path, required=True)
    parser.add_argument("--semantic-metadata", type=Path, required=True)
    parser.add_argument("--code-list", type=Path, required=True)
    parser.add_argument("--elfnote-deck", type=Path, required=True)
    parser.add_argument("--corpus-report", type=Path, required=True)
    parser.add_argument("--compute-manifest", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifacts = {
        "checkpoint": describe(args.checkpoint),
        "checkpoint_metadata": describe(args.checkpoint_metadata),
        "semantic_metadata": describe(args.semantic_metadata),
        "code_list": describe(args.code_list),
        "elfnote_deck": describe(args.elfnote_deck),
        "corpus_report": describe(args.corpus_report),
        "compute_manifest": describe(args.compute_manifest),
    }
    if artifacts["checkpoint"]["sha256"] != args.expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint hash mismatch: {artifacts['checkpoint']['sha256']} != "
            f"{args.expected_checkpoint_sha256}"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    metadata = json.loads(args.checkpoint_metadata.read_text(encoding="utf-8"))
    manifest = {
        "format_version": 1,
        "purpose": "clustered-multideck-continuation-parent",
        "repository_revision": revision,
        "observation_schema": metadata.get("observation_schema"),
        "structured_variant": metadata.get("model_architecture", {}).get("structured_variant"),
        "checkpoint_declared_hashes": {
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "code_list_hash": metadata.get("code_list_hash"),
            "semantic_table_hash": metadata.get("semantic_table_hash"),
        },
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
