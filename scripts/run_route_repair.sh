#!/usr/bin/env bash
set -euo pipefail
root=/root/ygo-agent-gpu-20260910
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$root/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
timeout 7200 "$root/.venv/bin/python" -m scripts.teach_route_repair \
  --mode "${1:-preflight}" --output "${2:?unused output}" \
  --release "$root/dist/runtime-releases/exercise-study-e9994b4-20261008" \
  --checkpoint "$root/training-runs/material-transfer-study-20261013/add_materials/model.flax_model" \
  --retention-checkpoint "$root/training-runs/correction-retention-study-20261013/correct_retain/model.flax_model" \
  --data "$root/training-runs/correction-retention-inputs-20261013/context-coverage-scenes-v2-20261009" \
  --previous "$root/training-runs/correction-retention-study-20261013" \
  --failure "$root/training-runs/correction-retention-inputs-20261013/failure.json" \
  --material-data "$root/training-runs/material-transfer-scenes-20261013" \
  --material-study "$root/training-runs/material-transfer-study-20261013" \
  --plan assets/exercises/teaching-v1/route-repair.json
