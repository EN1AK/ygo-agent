#!/usr/bin/env bash
set -euo pipefail
# Use the immutable audited runtime from the previous teaching study; only the
# source checkout changes. Never replace its model, engine or card assets.
root=/root/ygo-agent-gpu-20260910
release=$root/dist/runtime-releases/exercise-study-e9994b4-20261008
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$root/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
mode=${1:-preflight}
output=${2:?Supply an unused output directory}
flags=()
case "$mode" in
  preflight) ;;
  execute) flags+=(--execute) ;;
  *) echo 'mode must be preflight or execute' >&2; exit 2 ;;
esac
"$root/.venv/bin/python" -m unittest tests.test_exercise_teaching tests.test_exercise_retention -v
timeout 5400 "$root/.venv/bin/python" -m scripts.study_teaching_regression \
  --native "$release/exercise_actor_native.cpython-310-x86_64-linux-gnu.so" \
  --database "$release/repo/assets/locale/zh/cards.cdb" --scripts "$release/repo/scripts/script" \
  --code-list "$release/code_list.txt" --semantics "$release/semantics" --checkpoint "$release/parent.flax_model" \
  --combo "$release/opening-02" --combo-audit "$release/opening-02-audit.json" \
  --battle "$release/battle-01" --battle-audit "$release/battle-01-audit.json" \
  --release "$release" --plan assets/exercises/teaching-v1/scene-parent-kl.json \
  --output "$output" "${flags[@]}"
