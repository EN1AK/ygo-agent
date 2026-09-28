"""Append-only runtime assets and embedding-table checkpoint migration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ygoai.deck_corpus import CorpusError, read_code_list_metadata, sha256_file


def required_new_codes(ready_path: Path, code_list_path: Path) -> list[int]:
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    existing, _ = read_code_list_metadata(code_list_path)
    required = {
        int(code) for row in ready["decks"]
        for zone in ("main", "extra", "side")
        for code in row["deck"]["zones"].get(zone, {})
    }
    return sorted(required - set(existing))


def append_code_list(source: Path, destination: Path, new_codes: list[int], script_dir: Path) -> None:
    old_codes, _ = read_code_list_metadata(source)
    additions = sorted(set(int(code) for code in new_codes))
    if set(old_codes) & set(additions):
        raise CorpusError("append list contains existing card codes")
    payload = source.read_bytes()
    newline = b"\r\n" if b"\r\n" in payload else b"\n"
    if payload and not payload.endswith((b"\n", b"\r")):
        payload += newline
    for code in additions:
        has_script = int((script_dir / f"c{code}.lua").is_file())
        payload += f"{code} {has_script}".encode("ascii") + newline
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    if not destination.read_bytes().startswith(source.read_bytes()):
        raise RuntimeError("append-only code-list prefix changed")


def verify_semantic_append(old_dir: Path, new_dir: Path, old_rows: int, new_rows: int) -> dict[str, Any]:
    widths = {"card-semantics.u8": 32, "effect-tags.u8": 16,
              "effect-tag-confidence.u8": 16}
    results = {}
    for name, width in widths.items():
        old = (old_dir / name).read_bytes()
        new = (new_dir / name).read_bytes()
        if len(old) != old_rows * width or len(new) != new_rows * width:
            raise CorpusError(f"semantic dimensions mismatch for {name}")
        if new[:len(old)] != old:
            raise CorpusError(f"old semantic prefix changed for {name}")
        results[name] = {"old_sha256": sha256_file(old_dir / name),
                         "new_sha256": sha256_file(new_dir / name),
                         "old_bytes_identical": True,
                         "appended_bytes": len(new) - len(old)}
    return results


def migrate_parameter_tree(
    source_flat: dict[tuple[str, ...], np.ndarray],
    destination_flat: dict[tuple[str, ...], np.ndarray],
    old_embedding_rows: int,
    new_embedding_rows: int,
    seed: int,
) -> tuple[dict[tuple[str, ...], np.ndarray], dict[str, Any]]:
    if new_embedding_rows <= old_embedding_rows:
        raise CorpusError("expanded embedding rows must exceed old rows")
    output = {}
    inventory = []
    embedding_paths = []
    rng = np.random.default_rng(seed)
    for path, destination in destination_flat.items():
        label = "/".join(path)
        if path not in source_flat:
            raise CorpusError(f"destination parameter missing from source: {label}")
        source = np.asarray(source_flat[path])
        destination = np.asarray(destination)
        if source.shape == destination.shape:
            output[path] = source.copy()
            inventory.append({"path": label, "mode": "exact", "shape": list(source.shape)})
            continue
        if (path[-1] == "embedding" and source.ndim == 2 and destination.ndim == 2
                and source.shape[0] == old_embedding_rows
                and destination.shape[0] == new_embedding_rows
                and source.shape[1:] == destination.shape[1:]):
            mean = source.mean(axis=0)
            scale = source.std(axis=0)
            scale = np.where(scale == 0, np.asarray(1e-6, dtype=scale.dtype), scale)
            appended = rng.normal(mean, scale, size=(new_embedding_rows - old_embedding_rows,
                                                     source.shape[1])).astype(source.dtype)
            output[path] = np.concatenate([source, appended], axis=0)
            embedding_paths.append(label)
            inventory.append({"path": label, "mode": "append", "old_shape": list(source.shape),
                              "new_shape": list(destination.shape),
                              "appended_rows": new_embedding_rows - old_embedding_rows})
            continue
        raise CorpusError(f"unexplained parameter shape change {label}: {source.shape} -> {destination.shape}")
    missing_destination = sorted("/".join(path) for path in source_flat if path not in destination_flat)
    if missing_destination:
        raise CorpusError(f"source parameters absent from destination: {missing_destination[:4]}")
    if len(embedding_paths) != 1:
        raise CorpusError(f"expected exactly one expanded embedding, found {embedding_paths}")
    return output, {"seed": seed, "embedding_paths": embedding_paths,
                    "parameters": inventory,
                    "exact_parameter_count": sum(item["mode"] == "exact" for item in inventory)}


def asset_gap_report(ready_path: Path, code_list_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    additions = required_new_codes(ready_path, code_list_path)
    return {
        "schema_version": 1, "ready_table_sha256": sha256_file(ready_path),
        "source_code_list_sha256": sha256_file(code_list_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "required_new_codes": additions, "required_new_code_count": len(additions),
        "decision": "migration-required" if additions else "no-migration-required",
        "checkpoint_output_created": False,
    }
