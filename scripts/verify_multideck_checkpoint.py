"""Verify the selected 40M-derived checkpoint and manifest compatibility gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import _repo_bootstrap  # noqa: F401
import flax
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np

from ygoai.rl.checkpoint_compat import (
    CheckpointCompatibilityError, load_checkpoint_metadata,
    validate_checkpoint_compatibility, write_checkpoint_metadata,
)
from ygoai.rl.jax.agent import ModelArgs, RNNAgent
from ygoai.rl.observation_schema import manifest


def fixture(schema: str) -> dict[str, jax.Array]:
    return {name: jnp.zeros((1,) + tuple(spec["shape"]), dtype=spec["dtype"])
            for name, spec in manifest(schema)["tensors"].items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--code-list", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--cluster-manifest", type=Path, required=True)
    parser.add_argument("--curriculum-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = load_checkpoint_metadata(args.checkpoint)
    model_args = ModelArgs(**metadata["model_architecture"])
    embedding_count = sum(1 for line in args.code_list.read_text(
        encoding="utf-8-sig").splitlines() if line.strip())
    agent = RNNAgent(embedding_shape=embedding_count, **asdict(model_args))
    state = agent.init_rnn_state(1)
    template = agent.init(jax.random.PRNGKey(0), fixture(metadata["observation_schema"]), state)
    variables = flax.serialization.from_bytes(template, args.checkpoint.read_bytes())
    first = agent.apply(variables, fixture(metadata["observation_schema"]), state)
    second = agent.apply(variables, fixture(metadata["observation_schema"]), state)
    logits_a, values_a = np.asarray(first[1]), np.asarray(first[2])
    logits_b, values_b = np.asarray(second[1]), np.asarray(second[2])
    serialized = flax.serialization.to_bytes(variables)
    reloaded = flax.serialization.from_bytes(template, serialized)
    flattened = flatten_dict(flax.core.unfreeze(variables))
    reloaded_flat = flatten_dict(flax.core.unfreeze(reloaded))
    reload_exact = all(np.array_equal(np.asarray(value), np.asarray(reloaded_flat[path]))
                       for path, value in flattened.items())
    training_context = {
        "corpus_hash": hashlib.sha256(args.corpus_manifest.read_bytes()).hexdigest(),
        "cluster_hash": hashlib.sha256(args.cluster_manifest.read_bytes()).hexdigest(),
        "curriculum_hash": hashlib.sha256(args.curriculum_manifest.read_bytes()).hexdigest(),
    }
    with tempfile.TemporaryDirectory() as temp:
        probe = Path(temp) / "probe.flax_model"
        probe.write_bytes(serialized)
        write_checkpoint_metadata(
            probe, observation_schema=metadata["observation_schema"], model_args=model_args,
            semantic_table_hash=metadata["semantic_table_hash"],
            code_list_hash=metadata["code_list_hash"], capacities=metadata["capacities"],
            migrated_from=metadata["checkpoint_sha256"], training_context=training_context,
        )
        validate_checkpoint_compatibility(
            probe, observation_schema=metadata["observation_schema"],
            training_context=training_context,
        )
        cross_manifest_rejected = False
        try:
            validate_checkpoint_compatibility(
                probe, observation_schema=metadata["observation_schema"],
                training_context={**training_context, "cluster_hash": "mismatch"},
            )
        except CheckpointCompatibilityError:
            cross_manifest_rejected = True
    report = {
        "schema_version": 1, "checkpoint_sha256": metadata["checkpoint_sha256"],
        "migration_decision": "no-migration-required",
        "finite_logits": bool(np.isfinite(logits_a).all()),
        "finite_values": bool(np.isfinite(values_a).all()),
        "deterministic_logits": bool(np.array_equal(logits_a, logits_b)),
        "deterministic_values": bool(np.array_equal(values_a, values_b)),
        "deterministic_action": int(logits_a.argmax()) == int(logits_b.argmax()),
        "serialized_sha256": hashlib.sha256(serialized).hexdigest(),
        "save_reload_exact": reload_exact,
        "parameter_leaf_count": len(flattened),
        "training_context": training_context,
        "cross_manifest_rejected": cross_manifest_rejected,
    }
    report["passed"] = all(report[key] for key in (
        "finite_logits", "finite_values", "deterministic_logits", "deterministic_values",
        "deterministic_action", "save_reload_exact", "cross_manifest_rejected"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
