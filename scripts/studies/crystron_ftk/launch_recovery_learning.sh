#!/usr/bin/env bash
set -euo pipefail
cd /root/ygo-agent-gpu-20260910/training-runs/crystron-ftk-20261026
export OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:/root/ygo-agent-gpu-20260910/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib
python=/root/ygo-agent-gpu-20260910/.venv/bin/python
trap 'echo $? > recovery-learning-launch.exit' EXIT
test "$(cat learning-launch.exit)" = 0
"$python" evaluate_recovery_learning.py --label before --checkpoint learning-v1/full/model.flax_model > recovery-learning-before.log 2>&1
"$python" train_recovery_learning.py > recovery-learning-train.log 2>&1
"$python" evaluate_recovery_learning.py --label after --checkpoint learning-v1/recovery-augmented/model.flax_model > recovery-learning-after.log 2>&1
"$python" evaluate_learning.py --label recovery-augmented --checkpoint learning-v1/recovery-augmented/model.flax_model > recovery-learning-retention.log 2>&1