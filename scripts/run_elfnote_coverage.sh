#!/usr/bin/env bash
set -euo pipefail
root=${ROOT:-/root/ygo-agent-gpu-20260910}
src=${SOURCE:?set immutable SOURCE directory}
out=${OUTPUT:?set new OUTPUT directory}
test ! -e "$out"
cd "$src"
export PYTHONPATH=. JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$root/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
"$root/.venv/bin/python" -m unittest discover -s tests -p 'test_elfnote*.py'
date +%s > "$out.started"
study() {
 timeout "$2" "$root/.venv/bin/python" -c 'import os,runpy;os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8]);runpy.run_module("scripts.study_elfnote_coverage",run_name="__main__")' --root "$root" --source "$src" --output "$out" --stage "$1"
}
set +e
study collect 6000
status=$?
printf '%s\n' "$status" > "$out.collection.exit"
if test "$status" = 0; then
 study train 7800
 status=$?
fi
printf '%s\n' "$status" > "$out.exit"
if test "$status" = 0; then
 JAX_PLATFORMS=cpu "$root/.venv/bin/python" -m scripts.verify_elfnote_coverage --root "$root" --source "$src" --study "$out"
 status=$?
fi
printf '%s\n' "$status" > "$out.verification.exit"
exit "$status"
