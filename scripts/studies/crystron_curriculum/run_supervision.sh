#!/usr/bin/env bash
# Frozen inputs and a new output directory are required; no production changes.
set -euo pipefail
test "$#" -eq 4 # deployment, source checkout, frozen dataset, new output
deployment=$(realpath "$1")
project=$(realpath "$2")
frozen=$(realpath "$3")
output="$4"
test ! -e "$output"
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:"$deployment/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
export OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
trap 'echo $? > "$output.exit"' EXIT
"$deployment/.venv/bin/python" "$project/scripts/studies/crystron_curriculum/train_supervision.py" \
  --frozen "$frozen" --output "$output"
