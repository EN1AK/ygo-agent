"""Opt-in checkpoint envelope for Candidate-Q experiments.

Legacy PPO checkpoint readers are deliberately untouched. Metadata is checked
before any parameter deserialization or environment rollout is started.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import flax.serialization

from ygoai.rl.checkpoint_compat import (
    CheckpointCompatibilityError, sha256_file, validate_checkpoint_compatibility,
)


Q_FORMAT_VERSION = 2
Q_METADATA_SUFFIX = ".candidate_q.json"
Q_MODES = frozenset((
    "shadow_observation", "qboost_observation", "vrpo_centralized",
))


@dataclass(frozen=True)
class QCheckpointContext:
    mode: str
    critic_input_schema: str
    observation_schema: str
    reward_convention: str
    actor_architecture: dict[str, Any]
    critic_architecture: dict[str, Any]
    code_list_hash: str
    semantic_table_hash: str | None
    capacities: dict[str, int]
    training_context: dict[str, Any]
    source_commit: str
    native_sha256: str

    def __post_init__(self):
        if self.mode not in Q_MODES:
            raise CheckpointCompatibilityError(f"not a Candidate-Q mode: {self.mode!r}")
        if not self.critic_input_schema or not self.reward_convention:
            raise CheckpointCompatibilityError("critic schema and reward convention are required")


def q_metadata_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    return path.with_name(path.name + Q_METADATA_SUFFIX)


def write_q_checkpoint(
    checkpoint: str | Path,
    *,
    context: QCheckpointContext,
    actor_variables: Any,
    actor_optimizer_state: Any,
    critic_variables: Any,
    critic_optimizer_state: Any,
) -> Path:
    """Save complete experimental state without changing PPO serialization."""
    checkpoint = Path(checkpoint)
    payload = {
        "actor_variables": actor_variables,
        "actor_optimizer_state": actor_optimizer_state,
        "critic_variables": critic_variables,
        "critic_optimizer_state": critic_optimizer_state,
    }
    checkpoint.write_bytes(flax.serialization.to_bytes(payload))
    metadata = {
        "format_version": Q_FORMAT_VERSION,
        "checkpoint_sha256": sha256_file(checkpoint),
        "context": asdict(context),
    }
    q_metadata_path(checkpoint).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checkpoint


def validate_q_checkpoint(checkpoint: str | Path, expected: QCheckpointContext) -> dict:
    """Reject incompatible resume before touching a rollout or Q target."""
    checkpoint = Path(checkpoint)
    sidecar = q_metadata_path(checkpoint)
    if not checkpoint.is_file() or not sidecar.is_file():
        raise CheckpointCompatibilityError("Candidate-Q checkpoint or metadata is missing")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("format_version") != Q_FORMAT_VERSION:
        raise CheckpointCompatibilityError("Candidate-Q checkpoint format mismatch")
    actual_context = metadata.get("context")
    requested_context = asdict(expected)
    if not isinstance(actual_context, dict):
        raise CheckpointCompatibilityError("Candidate-Q checkpoint context is missing")
    for name, value in requested_context.items():
        if actual_context.get(name) != value:
            raise CheckpointCompatibilityError(f"Candidate-Q checkpoint {name} mismatch")
    if metadata.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise CheckpointCompatibilityError("Candidate-Q checkpoint hash mismatch")
    return metadata


def load_q_checkpoint(checkpoint: str | Path, *, expected: QCheckpointContext, template: Any):
    validate_q_checkpoint(checkpoint, expected)
    return flax.serialization.from_bytes(template, Path(checkpoint).read_bytes())


def import_ppo_actor_weights(
    checkpoint: str | Path,
    *,
    actor_template: Any,
    observation_schema: str,
    actor_architecture: Any,
    code_list_hash: str,
    semantic_table_hash: str | None,
    capacities: dict[str, int],
    training_context: dict[str, Any] | None,
):
    """Explicit actor-only import; never pretends to restore Q or optimizers."""
    validate_checkpoint_compatibility(
        checkpoint,
        observation_schema=observation_schema,
        model_args=actor_architecture,
        code_list_hash=code_list_hash,
        semantic_table_hash=semantic_table_hash,
        capacities=capacities,
        training_context=training_context,
    )
    return flax.serialization.from_bytes(actor_template, Path(checkpoint).read_bytes())
