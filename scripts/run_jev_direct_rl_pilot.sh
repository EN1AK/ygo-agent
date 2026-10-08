#!/usr/bin/env bash
# Bounded, isolated mechanism pilot. No dependency installation or production edits.
set -eu
: "${JEV_ROOT:?Set the absolute experiment root}"
: "${JEV_CORE:?Set the existing exercise bridge library}"
: "${JEV_DATABASE:?Set the pinned cards.cdb}"
: "${JEV_SCRIPTS:?Set the pinned card script directory}"
JEV_RUN_NAME=${JEV_RUN_NAME:-pilot-001}
case "$JEV_RUN_NAME" in *[!a-zA-Z0-9_-]*|'') exit 2;; esac
JEV_RUN_DIR="$JEV_ROOT/results/$JEV_RUN_NAME"
test ! -e "$JEV_RUN_DIR"
test ! -e "$JEV_RUN_DIR.log"
mkdir -p "$JEV_ROOT/results"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$JEV_ROOT/source"
set +e
timeout --signal=TERM --kill-after=15s 900 "$JEV_ROOT/venv/bin/python" -u -m scripts.train_jev_direct_rl \
  --model-dir "$JEV_ROOT/laya-multilingual" --core "$JEV_CORE" \
  --database "$JEV_DATABASE" --scripts "$JEV_SCRIPTS" --output "$JEV_RUN_DIR" \
  --updates 3 --group-size 4 --max-probes 2 --device cuda > "$JEV_RUN_DIR.log" 2>&1
JEV_EXIT=$?
set -e
printf '%s\n' "$JEV_EXIT" > "$JEV_RUN_DIR.exit-code"
tail -n 12 "$JEV_RUN_DIR.log"
exit "$JEV_EXIT"
