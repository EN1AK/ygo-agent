#!/usr/bin/env bash
set -euo pipefail
root="$1"
src="$2"
out="$3"
plan="${4:-$src/assets/exercises/teaching-v1/elfnote-functional-teaching.json}"
cd "$src"
export PYTHONPATH=. JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$root/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
"$root/.venv/bin/python" -m unittest discover -s tests -p 'test_elfnote*.py'
set +e
timeout 2700 "$root/.venv/bin/python" -c 'import os,runpy;os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8]);runpy.run_module("scripts.study_elfnote_functional_teaching",run_name="__main__")' --root "$root" --source "$src" --output "$out" --plan "$plan" --stage collect
status=$?
printf '%s\n' "$status" > "$out.collection.exit"
if test "$status" = 0; then
 timeout 7500 "$root/.venv/bin/python" -c 'import os,runpy;os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8]);runpy.run_module("scripts.study_elfnote_functional_teaching",run_name="__main__")' --root "$root" --source "$src" --output "$out" --plan "$plan" --stage train
 status=$?
fi
printf '%s\n' "$status" > "$out.exit"
if test "$status" = 0; then
 JAX_PLATFORMS=cpu "$root/.venv/bin/python" -m scripts.verify_elfnote_functional_teaching --root "$root" --source "$src" --study "$out"
 status=$?
fi
printf '%s\n' "$status" > "$out.verification.exit"
exit "$status"
