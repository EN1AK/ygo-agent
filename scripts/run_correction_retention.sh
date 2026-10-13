#!/usr/bin/env bash
set -euo pipefail
root=/root/ygo-agent-gpu-20260910
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$root/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
timeout 6500 "$root/.venv/bin/python" -m scripts.teach_correction_retention \
  --mode "${1:-preflight}" --output "${2:?supply unused output}" \
  --release "$root/dist/runtime-releases/exercise-study-e9994b4-20261008" \
  --checkpoint "${3:?supply restored 94e7 checkpoint}" \
  --retention-checkpoint "$root/training-runs/scene-parent-kl-study-20261009/prefix_parent_kl/model.flax_model" \
  --data "${4:?supply verified context dataset}" \
  --failure "${5:?supply original failure JSON}" \
  --plan assets/exercises/teaching-v1/correction-retention.json
