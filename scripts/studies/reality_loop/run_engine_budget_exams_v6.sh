#!/usr/bin/env bash
set -euo pipefail
base=/root/ygo-agent-gpu-20260910
repo=$base/ygo-agent
root=$base/training-runs/reality-loop-20261028
run=$root/filtered-policy-v5/research-v6
python=$base/.venv/bin/python
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$base/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=float32
trap 'echo $? > /root/ygo-agent-gpu-20260910/training-runs/reality-loop-20261028/engine-budget-exams-v6.exit' EXIT
test "$(cat "$root/corrected-exams-v6.exit")" = 1
exam=$run/all-endpoint-exams-v2
"$python" "$repo/scripts/studies/reality_loop/freeze_engine_budget_revision.py" --run "$run" --previous-exam "$run/all-endpoint-exams-v1" --output "$exam"
for phase in independent natural; do
    "$python" "$repo/scripts/studies/reality_loop/dispatch_budget_exams.py" --run "$run" --output "$exam" --phase "$phase" --workers 8
done
"$python" "$repo/scripts/studies/reality_loop/audit_budget_study.py" --run "$run" --exam "$exam" --output "$root/corrected-study-audit-v6"
