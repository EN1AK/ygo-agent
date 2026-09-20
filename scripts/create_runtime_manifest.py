"""Write a deterministic manifest for an immutable runtime artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="root containing every frozen artifact")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(),
                        help="Git checkout whose revision produced the release")
    parser.add_argument("--git-revision",
                        help="explicit revision for an exported checkout without .git")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--artifact", action="append", default=[],
                        help="relative artifact path; repeat for every required file")
    parser.add_argument("--metadata", action="append", default=[],
                        help="KEY=VALUE metadata; repeat as needed")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    if not args.artifact:
        raise ValueError("at least one --artifact is required")
    metadata = {}
    for item in args.metadata:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid metadata item: {item!r}")
        metadata[key] = value

    artifacts = []
    for relative in sorted(set(args.artifact)):
        path = (root / relative).resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"artifact escapes root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    manifest = {
        "format_version": 1,
        "git_revision": args.git_revision or git_revision(args.repo_root.resolve()),
        "observation_schema": args.schema,
        "metadata": dict(sorted(metadata.items())),
        "artifacts": artifacts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
