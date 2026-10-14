#!/usr/bin/env bash
set -euo pipefail
root=/root/ygo-agent-gpu-20260910
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$root/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
timeout 14400 "$root/.venv/bin/python" -m scripts.teach_elfnote_combo \
 --intake "$root/training-runs/elfnote-intake-20261014" \
 --release "$root/dist/runtime-releases/exercise-study-e9994b4-20261008" \
 --old "$root/training-runs/route-repair-study-20261013/correct_retain/endpoint/old" \
 --checkpoint "$root/training-runs/route-repair-study-20261013/correct_retain/model.flax_model" \
 --plan assets/exercises/teaching-v1/elfnote-single-card.json --output "${1:?new output directory}" "${@:2}"
