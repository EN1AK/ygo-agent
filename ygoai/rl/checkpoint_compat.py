"""Fail-fast metadata validation for versioned model checkpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from ygoai.rl.observation_schema import LEGACY_SCHEMA, SUPPORTED_SCHEMAS


METADATA_SUFFIX = ".metadata.json"
REQUIRED_STRUCTURED_FIELDS = (
    "observation_schema", "model_architecture", "semantic_table_hash",
    "code_list_hash", "capacities",
)


class CheckpointCompatibilityError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    return path.with_name(path.name + METADATA_SUFFIX)


def architecture_dict(model_args: Any) -> dict[str, Any]:
    if is_dataclass(model_args):
        return asdict(model_args)
    if isinstance(model_args, Mapping):
        return dict(model_args)
    raise TypeError("model_args must be a dataclass or mapping")


def write_checkpoint_metadata(
    checkpoint: str | Path,
    *,
    observation_schema: str,
    model_args: Any,
    semantic_table_hash: str | None,
    code_list_hash: str,
    capacities: Mapping[str, int],
    migrated_from: str | None = None,
) -> Path:
    if observation_schema not in SUPPORTED_SCHEMAS:
        raise CheckpointCompatibilityError(f"unsupported schema {observation_schema!r}")
    checkpoint = Path(checkpoint)
    payload = {
        "format_version": 1,
        "checkpoint_sha256": sha256_file(checkpoint),
        "observation_schema": observation_schema,
        "model_architecture": architecture_dict(model_args),
        "semantic_table_hash": semantic_table_hash,
        "code_list_hash": code_list_hash,
        "capacities": dict(sorted(capacities.items())),
        "migrated_from": migrated_from,
    }
    output = metadata_path(checkpoint)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_checkpoint_metadata(checkpoint: str | Path) -> dict[str, Any]:
    sidecar = metadata_path(checkpoint)
    if not sidecar.exists():
        # Historical checkpoints predate metadata and are legacy-only.
        return {"format_version": 0, "observation_schema": LEGACY_SCHEMA, "legacy_inferred": True}
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    missing = [name for name in REQUIRED_STRUCTURED_FIELDS if name not in data]
    if missing:
        raise CheckpointCompatibilityError(f"checkpoint metadata missing: {', '.join(missing)}")
    actual_hash = sha256_file(checkpoint)
    if data.get("checkpoint_sha256") != actual_hash:
        raise CheckpointCompatibilityError("checkpoint hash does not match metadata")
    return data


def validate_checkpoint_compatibility(
    checkpoint: str | Path,
    *,
    observation_schema: str,
    model_args: Any | None = None,
    semantic_table_hash: str | None = None,
    code_list_hash: str | None = None,
    capacities: Mapping[str, int] | None = None,
    allow_migration: bool = False,
) -> dict[str, Any]:
    metadata = load_checkpoint_metadata(checkpoint)
    actual_schema = metadata["observation_schema"]
    if actual_schema != observation_schema and not allow_migration:
        raise CheckpointCompatibilityError(
            f"checkpoint schema {actual_schema!r} is incompatible with requested "
            f"schema {observation_schema!r}; run an explicit migration"
        )
    if model_args is not None and not metadata.get("legacy_inferred"):
        if metadata["model_architecture"] != architecture_dict(model_args):
            raise CheckpointCompatibilityError("model architecture metadata mismatch")
    for key, expected in (
        ("semantic_table_hash", semantic_table_hash),
        ("code_list_hash", code_list_hash),
        ("capacities", dict(sorted(capacities.items())) if capacities is not None else None),
    ):
        if expected is not None and not metadata.get("legacy_inferred") and metadata.get(key) != expected:
            raise CheckpointCompatibilityError(f"checkpoint {key} mismatch")
    return metadata
