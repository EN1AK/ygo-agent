#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/d/workspace/ygo-agent
runtime=/home/ygo/ygo-agent
run="$repo/training-runs/multideck-protocolgate100k-final-20260926"
python="$runtime/.venv-wsl/bin/python"
checkpoint="$runtime/training-runs/structured-lite-full-cont-39m-20260922-night/checkpoints/1790011620_step_000040000000.flax_model"

test ! -e "$run/completed.txt"
mkdir -p "$run/checkpoints"
cd "$repo"
export PYTHONPATH="$repo/ygoenv:$repo"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

sha256sum \
  assets/protocol/ygopro-core-f969296-protocol.json \
  ygoenv/ygoenv/ygopro/ygopro.h \
  ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so \
  "$checkpoint" > "$run/input-sha256.txt"

args=(
  --checkpoint "$checkpoint"
  --ckpt-dir "$run/checkpoints"
  --run-name multideck-protocolgate100k-final__26092026
  --deck "$runtime/training-runs/deck-corpus-runtime-v1-20260923/decks"
  --deck1 manifest
  --deck2 manifest
  --deck-sampling-manifest "$repo/training-runs/deck-clusters-v1-20260923/sampling-manifest.tsv"
  --deck-sampler-seed 23092026
  --corpus-manifest "$repo/training-runs/deck-corpus-v1-20260923/manifest.json"
  --cluster-manifest "$repo/training-runs/deck-clusters-v1-20260923/manifest.json"
  --curriculum-manifest "$repo/training-runs/deck-clusters-v1-20260923/sampling-manifest.json"
  --elfnote-reserve 0.25
  --code-list-file "$repo/training-runs/deck-corpus-raw-20260923/corpus/frozen-parent-assets/code_list.crlf.txt"
  --semantic-asset-dir "$repo/training-runs/structured-lite-local-validation-20260921/pinned-semantics"
  --observation-schema structured-lite-v1
  --local-num-envs 16
  --local-env-threads 16
  --num-actor-threads 1
  --actor-device-ids 0
  --learner-device-ids 0
  --num-steps 64
  --collect-steps 64
  --num-minibatches 8
  --total-timesteps 100352
  --save-interval 49
  --max-checkpoints 4
  --eval-interval 0
  --tb-dir None
  --log-frequency 5
  --no-concurrency
)

printf '%q ' "$python" scripts/cleanba.py "${args[@]}" > "$run/launch-command.txt"
printf '\n' >> "$run/launch-command.txt"
set +e
"$python" -u scripts/cleanba.py "${args[@]}" 2>&1 | tee "$run/train.log"
status=${PIPESTATUS[0]}
set -e
if [[ $status -eq 0 ]]; then
  find "$run/checkpoints" -maxdepth 1 -type f -print0 \
    | sort -z | xargs -0 sha256sum > "$run/checkpoint-sha256.txt"
  printf 'completed UTC=%s\n' "$(date -u +%FT%TZ)" > "$run/completed.txt"
else
  printf 'failed exit=%s UTC=%s\n' "$status" "$(date -u +%FT%TZ)" > "$run/failed.txt"
fi
exit "$status"
