"""Capture a reproducible compute/runtime fingerprint for benchmark upgrades."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command, cwd=cwd, check=True, capture_output=True, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_inventory() -> list[dict[str, str]]:
    fields = (
        "name", "uuid", "driver_version", "memory.total", "pci.bus_id",
        "compute_cap", "power.limit",
    )
    output = run([
        "nvidia-smi", f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return []
    return [
        dict(zip(fields, (item.strip() for item in row)))
        for row in csv.reader(output.splitlines())
    ]


def jax_inventory() -> dict[str, object]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        import jax
    except Exception as error:  # pragma: no cover - diagnostic fallback
        return {"error": repr(error)}
    devices = []
    for device in jax.devices():
        devices.append({
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
            "process_index": int(device.process_index),
        })
    return {"backend": jax.default_backend(), "devices": devices}


def snapshot_tree_sha256(root: Path) -> str:
    """Hash a source snapshot when deployment intentionally omits .git."""
    excluded_parts = {".git", ".xmake", "build", "__pycache__"}
    digest = hashlib.sha256()
    for path in sorted(
        item for item in root.rglob("*")
        if item.is_file()
        and not excluded_parts.intersection(item.relative_to(root).parts)
        and path_suffix_allowed(item)
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def path_suffix_allowed(path: Path) -> bool:
    return path.suffix not in {".pyc", ".so", ".o", ".a"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    revision = run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if revision:
        status = run(["git", "status", "--porcelain=v1"], cwd=repo_root) or ""
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "."], cwd=repo_root,
            check=True, capture_output=True,
        ).stdout
        repository = {
            "root": str(repo_root),
            "source_state": "git",
            "git_revision": revision,
            "dirty": bool(status),
            "status_porcelain": status.splitlines(),
            "tracked_diff_sha256": sha256_bytes(diff),
            "snapshot_tree_sha256": None,
        }
    else:
        repository = {
            "root": str(repo_root),
            "source_state": "snapshot",
            "git_revision": None,
            "dirty": None,
            "status_porcelain": [],
            "tracked_diff_sha256": None,
            "snapshot_tree_sha256": snapshot_tree_sha256(repo_root),
        }
    lscpu = run(["lscpu", "--json"])
    meminfo = Path("/proc/meminfo")
    artifacts = []
    for value in args.artifact:
        path = value.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append({
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    manifest = {
        "format_version": 1,
        "label": args.label,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "python_executable": sys.executable,
            "wsl_distribution": os.getenv("WSL_DISTRO_NAME"),
            "kernel": run(["uname", "-a"]),
            "cpu": json.loads(lscpu) if lscpu else None,
            "meminfo": meminfo.read_text(encoding="utf-8").splitlines()
            if meminfo.exists() else None,
        },
        "accelerator": {
            "nvidia_smi": gpu_inventory(),
            "cuda_compiler": run(["nvcc", "--version"]),
            "jax": jax_inventory(),
        },
        "packages": package_versions([
            "jax", "jaxlib", "flax", "numpy", "optax", "orbax-checkpoint",
            "gymnasium", "envpool",
        ]),
        "runtime_environment": {
            name: os.getenv(name) for name in (
                "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "JAX_ENABLE_X64",
                "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION", "LD_LIBRARY_PATH",
            )
        },
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
