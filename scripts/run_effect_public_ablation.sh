#!/usr/bin/env bash
set -euo pipefail
root=/root/ygo-agent-gpu-20260910
job=$root/training-runs/effect-ablation-82a364b
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$root/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
mode=${1:-preflight}
output=${2:?Supply an unused output directory}
timeout 7200 "$root/.venv/bin/python" -m scripts.study_effect_public_state \
  --mode "$mode" --output "$output" \
  --release "$root/dist/runtime-releases/exercise-study-e9994b4-20261008" \
  --native "$job/exercise_actor_native.cpython-310-x86_64-linux-gnu.so" \
  --checkpoint "$root/training-runs/scene-parent-kl-study-20261009/prefix_parent_kl/model.flax_model" \
  --semantics "$job/parent-compatible-semantics" \
  --source-data "$root/training-runs/three-capability-scenes-v3-20261009" \
  --data "$job/scenes" --plan assets/exercises/teaching-v1/effect-public-ablation.json
