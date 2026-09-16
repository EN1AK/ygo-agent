"""Explicit legacy -> Structured-lite warm-start checkpoint migration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _repo_bootstrap  # noqa: F401
import flax
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np

from ygoai.rl.checkpoint_compat import sha256_file, write_checkpoint_metadata
from ygoai.rl.jax.agent import ModelArgs, RNNAgent
from ygoai.rl.observation_schema import manifest


def fixture(schema: str, embeddings: int) -> dict[str, jax.Array]:
    return {
        name: jnp.zeros((1,) + tuple(spec["shape"]), dtype=spec["dtype"])
        for name, spec in manifest(schema)["tensors"].items()
    }


def initialise(model_args: ModelArgs, embeddings: int):
    agent = RNNAgent(embedding_shape=embeddings, **asdict(model_args))
    state = agent.init_rnn_state(1)
    params = agent.init(jax.random.PRNGKey(0), fixture(model_args.observation_schema, embeddings), state)
    return agent, state, params


def migrate(source_tree, destination_tree):
    source = flatten_dict(flax.core.unfreeze(source_tree))
    destination = flatten_dict(flax.core.unfreeze(destination_tree))
    output = dict(destination)
    copied = []
    incompatible = []
    excluded = []
    for path, value in source.items():
        label = "/".join(path)
        if path not in destination:
            excluded.append({"path": label, "source_shape": list(np.shape(value))})
        elif np.shape(value) != np.shape(destination[path]):
            incompatible.append({
                "path": label, "source_shape": list(np.shape(value)),
                "destination_shape": list(np.shape(destination[path])),
            })
        else:
            output[path] = value
            copied.append({"path": label, "shape": list(np.shape(value))})
    new = [
        {"path": "/".join(path), "shape": list(np.shape(value))}
        for path, value in destination.items() if path not in source
    ]
    report = {
        "source_parameters": len(source), "destination_parameters": len(destination),
        "copied": copied, "newly_initialized": new,
        "shape_incompatible": incompatible, "intentionally_excluded": excluded,
        "source_accounted": len(copied) + len(incompatible) + len(excluded),
        "destination_accounted": len(copied) + len(incompatible) + len(new),
    }
    if report["source_accounted"] != len(source) or report["destination_accounted"] != len(destination):
        raise RuntimeError("migration inventory does not account for both parameter trees")
    return flax.core.freeze(unflatten_dict(output)), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-list", type=Path, required=True)
    parser.add_argument("--semantic-metadata", type=Path, required=True)
    parser.add_argument("--num-embeddings", type=int, required=True)
    parser.add_argument("--variant", choices=("relationship-only", "full"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be new or empty")

    source_hash_before = sha256_file(args.source)
    legacy_args = ModelArgs(observation_schema="legacy-v2")
    structured_args = ModelArgs(
        observation_schema="structured-lite-v1", structured_variant=args.variant)
    _, _, legacy_template = initialise(legacy_args, args.num_embeddings)
    _, state, destination = initialise(structured_args, args.num_embeddings)
    source = flax.serialization.from_bytes(legacy_template, args.source.read_bytes())
    migrated, report = migrate(source, destination)
    report.update({
        "source": str(args.source.resolve()), "source_sha256": source_hash_before,
        "observation_schema": "structured-lite-v1", "variant": args.variant,
        "dry_run": args.dry_run,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "migration-report.json"
    if not args.dry_run:
        output = args.output_dir / "warm-start.flax_model"
        output.write_bytes(flax.serialization.to_bytes(migrated))
        write_checkpoint_metadata(
            output, observation_schema="structured-lite-v1", model_args=structured_args,
            semantic_table_hash=sha256_file(args.semantic_metadata),
            code_list_hash=sha256_file(args.code_list),
            capacities={"max_cards": 80, "max_options": 24, "history_actions": 32,
                        "public_events": 32, "group_references": 8},
            migrated_from=source_hash_before,
        )
        _, logits, values, _ = RNNAgent(
            embedding_shape=args.num_embeddings, **asdict(structured_args)
        ).apply(migrated, fixture("structured-lite-v1", args.num_embeddings), state)
        report["output"] = str(output.resolve())
        report["output_sha256"] = sha256_file(output)
        report["forward_finite"] = bool(
            np.isfinite(np.asarray(logits)).all() and np.isfinite(np.asarray(values)).all())
    if sha256_file(args.source) != source_hash_before:
        raise RuntimeError("source checkpoint changed during migration")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
