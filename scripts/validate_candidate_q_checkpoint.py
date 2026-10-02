"""CPU-only real PPO weight round-trip; not a Candidate-Q training pilot."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import flax.serialization
import numpy as np
import optax

from ygoai.rl.candidate_q_checkpoint import (
    QCheckpointContext, export_ppo_actor_checkpoint, import_ppo_actor_weights,
    load_q_checkpoint, write_q_checkpoint,
)
from ygoai.rl.checkpoint_compat import load_checkpoint_metadata, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--native-sha256", required=True)
    args = parser.parse_args()
    metadata = load_checkpoint_metadata(args.checkpoint)
    if metadata.get("legacy_inferred"):
        raise ValueError("This provenance check requires an explicit PPO metadata sidecar")
    raw = args.checkpoint.read_bytes()
    actor_template = flax.serialization.msgpack_restore(raw)
    actor = import_ppo_actor_weights(
        args.checkpoint, actor_template=actor_template,
        observation_schema=metadata["observation_schema"],
        actor_architecture=metadata["model_architecture"],
        code_list_hash=metadata["code_list_hash"],
        semantic_table_hash=metadata["semantic_table_hash"],
        capacities=metadata["capacities"], training_context=metadata["training_context"],
    )
    context = QCheckpointContext(
        mode="shadow_observation", critic_input_schema="serialization-fixture-only-v1",
        observation_schema=metadata["observation_schema"],
        reward_convention="two-seat-zero-sum-v1",
        actor_architecture=metadata["model_architecture"],
        critic_architecture={"fixture_only": True, "shape": [2, 2]},
        code_list_hash=metadata["code_list_hash"],
        semantic_table_hash=metadata["semantic_table_hash"],
        capacities=metadata["capacities"], training_context=metadata["training_context"] or {},
        source_commit=args.source_commit, native_sha256=args.native_sha256,
    )
    critic = {"params": {"fixture": np.zeros((2, 2), dtype=np.float32)}}
    state = dict(
        actor_variables=actor, actor_optimizer_state=optax.sgd(1e-3).init(actor["params"]),
        critic_variables=critic, critic_optimizer_state=optax.adam(1e-3).init(critic["params"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    envelope = args.output_dir / "serialization-fixture.candidate_q"
    write_q_checkpoint(envelope, context=context, **state)
    restored = load_q_checkpoint(envelope, expected=context, template=state)
    assert flax.serialization.to_bytes(state) == flax.serialization.to_bytes(restored)
    exported = export_ppo_actor_checkpoint(
        envelope, args.output_dir / "actor.flax_model", expected=context, template=state)
    assert exported.read_bytes() == raw, "actor export changed original checkpoint bytes"
    legacy_reader = flax.serialization.from_bytes(actor_template, exported.read_bytes())
    assert flax.serialization.to_bytes(legacy_reader) == raw
    report = {
        "schema_version": 1, "validation_only": True, "q_training_performed": False,
        "critic_is_synthetic_fixture": True, "context": asdict(context),
        "input_checkpoint": str(args.checkpoint), "input_sha256": sha256_file(args.checkpoint),
        "envelope_sha256": sha256_file(envelope), "export_sha256": sha256_file(exported),
        "complete_state_roundtrip": True, "actor_export_byte_identical": True,
        "unchanged_legacy_reader_passed": True,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
