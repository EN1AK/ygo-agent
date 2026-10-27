#!/usr/bin/env bash
set -euo pipefail
cd /root/ygo-agent-gpu-20260910/training-runs/crystron-ftk-20261026
export OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:/root/ygo-agent-gpu-20260910/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib
python=/root/ygo-agent-gpu-20260910/.venv/bin/python
trap 'echo $? > learning-launch.exit' EXIT
"$python" freeze_learning.py > learning-freeze.log
"$python" evaluate_learning.py --label before --checkpoint trial-v1/model.flax_model > learning-before.log 2>&1
for stage in suffix32 suffix96 full; do
  "$python" train_learning.py --stage "$stage" > "learning-$stage-train.log" 2>&1
  "$python" evaluate_learning.py --label "$stage" --checkpoint "learning-v1/$stage/model.flax_model" > "learning-$stage-eval.log" 2>&1
done
