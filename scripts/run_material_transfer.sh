#!/usr/bin/env bash
set -euo pipefail
root=/root/ygo-agent-gpu-20260910
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$root/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
timeout 5400 "$root/.venv/bin/python" -m scripts.teach_material_transfer \
  --mode "${1:-preflight}" --output "${2:?unused output}" \
  --release "$root/dist/runtime-releases/exercise-study-e9994b4-20261008" \
  --checkpoint "$root/training-runs/correction-retention-study-20261013/correct_retain/model.flax_model" \
  --data "$root/training-runs/correction-retention-inputs-20261013/context-coverage-scenes-v2-20261009" \
  --previous "$root/training-runs/correction-retention-study-20261013" \
  --failure "$root/training-runs/correction-retention-inputs-20261013/failure.json" \
  --material-data "${3:?material dataset}" \
  --plan assets/exercises/teaching-v1/material-transfer.json
