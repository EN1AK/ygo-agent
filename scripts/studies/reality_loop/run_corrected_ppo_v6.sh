#!/usr/bin/env bash
set -euo pipefail
producer_pid=${1:?live corrected SFT producer PID required}
base=/root/ygo-agent-gpu-20260910
repo=$base/ygo-agent
root=$base/training-runs/reality-loop-20261028
filtered=$root/filtered-policy-v5
revision=$filtered/synchro-core-v1
run=$filtered/research-v6
python=$base/.venv/bin/python
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$base/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=float32
trap 'echo $? > /root/ygo-agent-gpu-20260910/training-runs/reality-loop-20261028/corrected-ppo-v6.exit' EXIT
while test ! -e "$root/corrected-sft-v6.exit"; do
    if ! kill -0 "$producer_pid" 2>/dev/null; then
        echo 'SFT producer absent without a terminal exit; inspect rather than restart'
        exit 1
    fi
    sleep 30
done
test "$(cat "$root/corrected-sft-v6.exit")" = 0
"$python" "$repo/scripts/studies/reality_loop/test_corrected_lineage.py"
proof=$revision/fresh-sft-v6-actor-paths
"$python" "$repo/scripts/studies/reality_loop/audit_historical_actor_paths.py" --run "$run" --course "$run/round2-training/repair-inputs" --native "$revision/native/exercise_actor_native.cpython-310-x86_64-linux-gnu.so" --output "$proof"
parent=$run/round2-training/repair/endpoint/model.flax_model
JAX_PLATFORMS=cuda "$python" "$revision/formal-runtime/repo/scripts/probe_elfnote_formal_gpu.py" --release "$revision/formal-runtime" --checkpoint "$parent" --output "$revision/gpu-reload-v6.json"
"$python" "$repo/scripts/studies/reality_loop/ppo_corrected_study.py" freeze --run "$run" --runtime-root "$revision" --gpu-reload "$revision/gpu-reload-v6.json" --path-proof "$proof/report.json" --duel-opponent "$filtered/research-v5/round2-training/control/endpoint/model.flax_model" --name ppo-study-v1
protocol=$run/ppo-study-v1-protocol.json
"$python" "$repo/scripts/studies/reality_loop/ppo_corrected_study.py" execute --run "$run" --protocol "$protocol" --phase baseline
for role in control rehearse; do
    "$python" "$repo/scripts/studies/reality_loop/ppo_corrected_study.py" execute --run "$run" --protocol "$protocol" --phase smoke --role "$role"
done
for role in control rehearse; do
    "$python" "$repo/scripts/studies/reality_loop/ppo_corrected_study.py" execute --run "$run" --protocol "$protocol" --phase train --role "$role"
done
