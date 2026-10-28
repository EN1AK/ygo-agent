#!/usr/bin/env bash
set -euo pipefail
producer_pid=${1:?live corrected PPO producer PID required}
base=/root/ygo-agent-gpu-20260910
repo=$base/ygo-agent
root=$base/training-runs/reality-loop-20261028
filtered=$root/filtered-policy-v5
revision=$filtered/synchro-core-v1
run=$filtered/research-v6
old=$filtered/research-v5
python=$base/.venv/bin/python
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$base/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=float32
trap 'echo $? > /root/ygo-agent-gpu-20260910/training-runs/reality-loop-20261028/corrected-exams-v6.exit' EXIT
while test ! -e "$root/corrected-ppo-v6.exit"; do
    if ! kill -0 "$producer_pid" 2>/dev/null; then
        echo 'PPO producer absent without terminal exit; inspect rather than restart'
        exit 1
    fi
    sleep 30
done
test "$(cat "$root/corrected-ppo-v6.exit")" = 0
exam=$run/all-endpoint-exams-v1
"$python" "$repo/scripts/studies/reality_loop/freeze_corrected_study_exams.py" --run "$run" --ppo-protocol "$run/ppo-study-v1-protocol.json" --previous-exam "$old/all-endpoint-exams-v3" --previous-ppo-protocol "$old/ppo-study-v2-protocol.json" "$old/ppo-study-v3-protocol.json" --path-proof "$revision/fresh-sft-v6-actor-paths/report.json" --output "$exam"
for phase in development qualification independent natural; do
    "$python" "$repo/scripts/studies/reality_loop/dispatch_corrected_exams.py" --run "$run" --output "$exam" --phase "$phase" --workers 8
done
"$python" "$repo/scripts/studies/reality_loop/audit_ppo_study.py" --run "$run" --exam "$exam" --output "$root/corrected-study-audit-v6"
