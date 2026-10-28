#!/usr/bin/env bash
# Fresh feedback rounds after the historical input-equivalence gate failed.
set -euo pipefail
base=/root/ygo-agent-gpu-20260910
repo=$base/ygo-agent
root=$base/training-runs/reality-loop-20261028
filtered=$root/filtered-policy-v5
revision=$filtered/synchro-core-v1
course=$revision/base235-projection
run=$filtered/research-v6
python=$base/.venv/bin/python
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$base/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=float32
trap 'echo $? > /root/ygo-agent-gpu-20260910/training-runs/reality-loop-20261028/corrected-sft-v6.exit' EXIT
test "$(cat "$root/ppo-core-revision-v1.exit")" = 0
test "$(cat "$root/synchro-runtime-admission-v1.exit")" = 0
test "$(cat "$root/historical-path-bridge-v1.exit")" = 1
test ! -e "$course"
test ! -e "$run"
if pgrep -f '[r]eality_loop/(run|recover|train_pair).py.*research-v6'; then exit 1; fi
# Original 235 human rows only. Old seven repairs remain historical evidence.
"$python" "$repo/scripts/studies/reality_loop/reencode_overlay_course.py" --source "$filtered/course-independent-v3" --native "$revision/native/exercise_actor_native.cpython-310-x86_64-linux-gnu.so" --output "$course"
parent=$root/round1-training/control/endpoint/model.flax_model
"$python" "$repo/scripts/studies/reality_loop/run.py" freeze --frozen "$course" --checkpoint "$parent" --independent-exam "$revision/independent-exam" --output "$run"
for round in round1 round2; do
    JAX_PLATFORMS=cuda "$python" "$repo/scripts/studies/reality_loop/run.py" observe --run "$run" --checkpoint "$parent" --name "$round"
    "$python" "$repo/scripts/studies/reality_loop/recover.py" --run "$run" --observation "$round"
    JAX_PLATFORMS=cuda "$python" "$repo/scripts/studies/reality_loop/train_pair.py" --run "$run" --observation "$round" --checkpoint "$parent" --prepare-only
    JAX_PLATFORMS=cuda "$python" "$repo/scripts/studies/reality_loop/train_pair.py" --run "$run" --observation "$round" --checkpoint "$parent" --execute-prepared
    "$python" "$repo/scripts/studies/reality_loop/compare.py" --run "$run" --observation "$round"
    if test "$round" = round1; then
        "$python" "$repo/scripts/studies/reality_loop/advance_round.py" --run "$run"
        parent=$run/round1-training/repair/endpoint/model.flax_model
    fi
done
