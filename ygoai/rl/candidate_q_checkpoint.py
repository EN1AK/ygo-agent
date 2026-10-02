"""Opt-in checkpoint envelope for Candidate-Q experiments.

Legacy PPO checkpoint readers are deliberately untouched. Metadata is checked
before any parameter deserialization or environment rollout is started.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import flax.serialization
import numpy as np

from ygoai.rl.checkpoint_compat import (
    CheckpointCompatibilityError, metadata_path, sha256_file,
    validate_checkpoint_compatibility, write_checkpoint_metadata,
)


Q_FORMAT_VERSION = 2
Q_METADATA_SUFFIX = ".candidate_q.json"
Q_MODES = frozenset((
    "shadow_observation", "qboost_observation", "vrpo_centralized",
))
Q_STATE_FIELDS = frozenset((
    "actor_variables", "actor_optimizer_state", "critic_variables", "critic_optimizer_state",
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
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise CheckpointCompatibilityError("invalid Candidate-Q metadata JSON") from exc
    if not isinstance(metadata, dict):
        raise CheckpointCompatibilityError("Candidate-Q metadata must be an object")
    if metadata.get("format_version") != Q_FORMAT_VERSION:
        raise CheckpointCompatibilityError("Candidate-Q checkpoint format mismatch")
    actual_context = metadata.get("context")
    requested_context = asdict(expected)
    if not isinstance(actual_context, dict):
        raise CheckpointCompatibilityError("Candidate-Q checkpoint context is missing")
    for name, value in requested_context.items():
        if name not in actual_context or actual_context[name] != value:
            raise CheckpointCompatibilityError(f"Candidate-Q checkpoint {name} mismatch")
    if metadata.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise CheckpointCompatibilityError("Candidate-Q checkpoint hash mismatch")
    return metadata


def load_q_checkpoint(checkpoint: str | Path, *, expected: QCheckpointContext, template: Any):
    validate_q_checkpoint(checkpoint, expected)
    if not isinstance(template, dict) or template.keys() != Q_STATE_FIELDS:
        raise CheckpointCompatibilityError("Candidate-Q resume requires the complete four-state template")
    try:
        restored = flax.serialization.from_bytes(template, Path(checkpoint).read_bytes())
    except (ValueError, TypeError, KeyError) as exc:
        raise CheckpointCompatibilityError("Candidate-Q payload cannot restore template") from exc
    # Flax restores arrays without requiring their shape/dtype to match the
    # template. Reject an incompatible optimizer or parameter layout here,
    # before the learner can use it in a rollout or compilation.
    _validate_state_layout(
        flax.serialization.to_state_dict(template),
        flax.serialization.to_state_dict(restored),
    )
    return restored


def _validate_state_layout(expected: Any, actual: Any, path: str = "state"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise CheckpointCompatibilityError(f"Candidate-Q {path} structure mismatch")
        for key in expected:
            _validate_state_layout(expected[key], actual[key], f"{path}.{key}")
    elif hasattr(expected, "shape") and hasattr(expected, "dtype"):
        if (not hasattr(actual, "shape") or actual.shape != expected.shape
                or np.dtype(actual.dtype) != np.dtype(expected.dtype)):
            raise CheckpointCompatibilityError(f"Candidate-Q {path} shape/dtype mismatch")
    elif type(expected) is not type(actual):
        raise CheckpointCompatibilityError(f"Candidate-Q {path} type mismatch")


def export_ppo_actor_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    *,
    expected: QCheckpointContext,
    template: Any,
) -> Path:
    """Export validated actor variables only, readable by unchanged evaluators.

    This is an inference/actor-weight artifact, not a Q training resume. No Q
    parameters, optimizer, privileged inputs, or sampler state are exported.
    Refuse existing targets so a research export cannot overwrite the baseline.
    """
    output = Path(output)
    if output.exists() or metadata_path(output).exists() or q_metadata_path(output).exists():
        raise FileExistsError(f"actor export target already exists: {output}")
    restored = load_q_checkpoint(checkpoint, expected=expected, template=template)
    actor_bytes = flax.serialization.to_bytes(restored["actor_variables"])
    with output.open("xb") as stream:
        stream.write(actor_bytes)
    write_checkpoint_metadata(
        output,
        observation_schema=expected.observation_schema,
        model_args=expected.actor_architecture,
        semantic_table_hash=expected.semantic_table_hash,
        code_list_hash=expected.code_list_hash,
        capacities=expected.capacities,
        training_context=expected.training_context,
        migrated_from=f"candidate-q:{sha256_file(checkpoint)}",
    )
    return output


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
