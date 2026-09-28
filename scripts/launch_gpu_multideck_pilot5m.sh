#!/usr/bin/env bash
set -euo pipefail

root=/root/ygo-agent-gpu-20260910
repo="$root/ygo-agent"
run="${RUN_DIR:-$root/training-runs/multideck-scratch-pilot5m-protocol64-cap128-20260926}"
python="$root/.venv/bin/python"
checkpoint="${CHECKPOINT:-none}"
contract="$repo/assets/protocol/ygopro-core-f969296-protocol.json"
run_name="${RUN_NAME:-multideck-scratch-pilot5m-protocol64-cap128__26092026}"
total_timesteps="${TOTAL_TIMESTEPS:-5000192}"
timeout_seconds="${TIMEOUT_SECONDS:-600}"
log_frequency="${LOG_FREQUENCY:-50}"
save_interval="${SAVE_INTERVAL:-257}"
max_checkpoints="${MAX_CHECKPOINTS:-20}"
run_label="${RUN_LABEL:-multideck-scratch-pilot5m-protocol64-cap128-20260926}"
max_options="${MAX_OPTIONS:-128}"

if [[ -e "$run" ]]; then
  echo "Refusing to reuse existing pilot directory: $run" >&2
  exit 1
fi
mkdir -p "$run/checkpoints"
cp "$0" "$run/launch.sh"
coverage="$repo/training-runs/protocol-gate-v1-20260923/protocol-coverage-current.json"
"$python" -c 'import hashlib,json,sys; c=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2])); assert r["summary"]["uncovered_branches"] == 0; assert r["contract_sha256"] == c["contract_sha256"]; assert r["inputs"]["adapter"]["sha256"] == hashlib.sha256(open(sys.argv[3],"rb").read()).hexdigest(); assert r["inputs"]["boundary_test"]["sha256"] == hashlib.sha256(open(sys.argv[4],"rb").read()).hexdigest()' "$contract" "$coverage" "$repo/ygoenv/ygoenv/ygopro/ygopro.h" "$repo/tests/test_ygocore_protocol_boundaries.py"
contract_internal_sha256=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["contract_sha256"])' "$contract")
contract_file_sha256=$(sha256sum "$contract" | cut -d ' ' -f 1)
test "$contract_internal_sha256" = ed4259cbad49c1b740d8b6e74d05ffaad1d54344bfdca611e5b70f3d137a01fe
printf '%s\n' \
  "contract_internal_sha256=$contract_internal_sha256" \
  "contract_file_sha256=$contract_file_sha256" \
  "coverage_file_sha256=$(sha256sum "$coverage" | cut -d ' ' -f 1)" \
  > "$run/protocol-contract.txt"

if [[ "$checkpoint" == none ]]; then
  printf 'initialization=random-from-scratch\nseed=1\n' > "$run/initialization.txt"
else
  echo 'a6ddf4b08d5dba2cbe786b3df8b465cc4b01d8d8b85104187a98cb166a14496f  '"$checkpoint" | sha256sum -c -
  printf 'initialization=checkpoint\ncheckpoint=%s\n' "$checkpoint" > "$run/initialization.txt"
fi
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$root/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib
export LD_PRELOAD="$repo/libcompat_glibc.so"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$repo/ygoenv:$repo"

cd "$repo"
compute_args=(
  --repo-root "$repo" \
  --label "$run_label"
  --output "$run/compute-manifest.json"
)
if [[ "$checkpoint" != none ]]; then
  compute_args+=(--artifact "$checkpoint")
fi
"$python" scripts/capture_compute_manifest.py "${compute_args[@]}"

args=(
  --ckpt-dir "$run/checkpoints"
  --run-name "$run_name"
  --timeout "$timeout_seconds"
  --deck "$root/training-runs/deck-corpus-runtime-v1-20260923/decks"
  --deck1 manifest
  --deck2 manifest
  --deck-sampling-manifest "$root/training-runs/deck-clusters-v1-20260923/sampling-manifest.tsv"
  --deck-sampler-seed 23092026
  --corpus-manifest "$root/training-runs/deck-corpus-v1-20260923/manifest.json"
  --cluster-manifest "$root/training-runs/deck-clusters-v1-20260923/manifest.json"
  --curriculum-manifest "$root/training-runs/deck-clusters-v1-20260923/sampling-manifest.json"
  --elfnote-reserve 0.25
  --code-list-file "$root/training-runs/deck-corpus-raw-20260923/corpus/frozen-parent-assets/code_list.crlf.txt"
  --semantic-asset-dir "$root/training-runs/structured-lite-local-validation-20260921/pinned-semantics"
  --observation-schema structured-lite-v1
  --max-options "$max_options"
  --local-num-envs 16
  --local-env-threads 16
  --num-actor-threads 1
  --actor-device-ids 0
  --learner-device-ids 0
  --num-steps 64
  --collect-steps 64
  --num-minibatches 8
  --total-timesteps "$total_timesteps"
  --save-interval "$save_interval"
  --max-checkpoints "$max_checkpoints"
  --eval-interval 0
  --tb-dir None
  --log-frequency "$log_frequency"
  --no-concurrency
)
if [[ "$checkpoint" != none ]]; then
  args+=(--checkpoint "$checkpoint")
fi

printf '%q ' "$python" scripts/cleanba.py "${args[@]}" > "$run/launch-command.txt"
printf '\n' >> "$run/launch-command.txt"

if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then
  args+=(--config-only)
fi

set +e
"$python" -u scripts/cleanba.py "${args[@]}" 2>&1 | tee "$run/train.log"
status=${PIPESTATUS[0]}
set -e
if [[ $status -ne 0 ]]; then
  printf 'failed exit=%s UTC=%s\n' "$status" "$(date -u +%FT%TZ)" > "$run/failed.txt"
  exit "$status"
fi

if [[ "${CONFIG_ONLY:-0}" != 1 ]]; then
  find "$run/checkpoints" -maxdepth 1 -type f -print0 \
    | sort -z | xargs -0 sha256sum > "$run/checkpoint-sha256.txt"
  printf 'completed UTC=%s\n' "$(date -u +%FT%TZ)" > "$run/completed.txt"
else
  printf 'config-only-passed UTC=%s\n' "$(date -u +%FT%TZ)" > "$run/config-only-passed.txt"
fi
