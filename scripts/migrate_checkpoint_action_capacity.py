"""Copy a structured checkpoint with a larger action-row capacity.

The actor scores action rows with shared weights, so this changes the runtime
observation envelope and checkpoint metadata, not the learned parameters.
The source checkpoint and sidecar remain immutable.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ygoai.rl.checkpoint_compat import (
    load_checkpoint_metadata,
    metadata_path,
    sha256_file,
    write_checkpoint_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-options", type=int, required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if source == output or output.exists() or metadata_path(output).exists():
        raise ValueError("output must be a new, distinct checkpoint")
    metadata = load_checkpoint_metadata(source)
    if metadata.get("legacy_inferred"):
        raise ValueError("source must have verified structured metadata")
    old_capacity = metadata["capacities"]["max_options"]
    if args.max_options <= old_capacity:
        raise ValueError("new max_options must exceed source capacity")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    capacities = dict(metadata["capacities"])
    capacities["max_options"] = args.max_options
    sidecar = write_checkpoint_metadata(
        output,
        observation_schema=metadata["observation_schema"],
        model_args=metadata["model_architecture"],
        semantic_table_hash=metadata["semantic_table_hash"],
        code_list_hash=metadata["code_list_hash"],
        capacities=capacities,
        migrated_from=metadata["checkpoint_sha256"],
        training_context=metadata.get("training_context"),
        runtime_state=metadata.get("runtime_state"),
    )
    if sha256_file(output) != metadata["checkpoint_sha256"]:
        raise RuntimeError("copied weights differ from immutable source")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_metadata_sha256": sha256_file(metadata_path(source)),
        "source_weight_sha256": metadata["checkpoint_sha256"],
        "output": str(output),
        "output_metadata": str(sidecar),
        "output_weight_sha256": sha256_file(output),
        "old_max_options": old_capacity,
        "new_max_options": args.max_options,
        "weight_transform": "none; byte-identical copy",
        "reason": "25 legal idle actions exceeded frozen 24-row input capacity",
    }
    report_path = output.parent / "action-capacity-migration.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
