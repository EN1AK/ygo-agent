#!/usr/bin/env bash
set -euo pipefail

root=/root/ygo-agent-gpu-20260910
repo="$root/ygo-agent"
run="${RUN_DIR:-$root/training-runs/multideck-unselect-forward-45m-to100m-f751782-20261002}"
python="$root/.venv/bin/python"
parent="${PARENT_CHECKPOINT:-$root/training-runs/multideck-maxstep-loss-40m-plus5m-a1f79e0-20261001/checkpoints/1102026_step_000045015040.flax_model}"
module="$repo/ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so"
contract="$repo/assets/protocol/ygopro-core-f969296-protocol.json"
coverage="$repo/training-runs/protocol-gate-v1-20260923/protocol-coverage-current.json"
run_name="${RUN_NAME:-multideck-unselect-forward-45m-to100m-f751782__02102026}"
source_commit="${SOURCE_COMMIT:?Set SOURCE_COMMIT to the deployed full Git revision}"
native_commit="${NATIVE_COMMIT:-f7517828bae524a90f6e7fdbff7c0939abf5381c}"
native_sha256="${NATIVE_SHA256:-748f969a5e112b90b536f3b2e55042fe079ca173d26d7cbbdc8342f5ce52d22b}"
parent_sha256="${PARENT_SHA256:-28b4bd084a92e760fd54905887dda92c4cdac9d2ba39cc963114de98d9144322}"
source_marker="$root/training-runs/source-deploy-${source_commit:0:7}.txt"
parent_steps="${PARENT_STEPS:-45015040}"
continuation_steps="${CONTINUATION_STEPS:-54988800}"
target_steps="${TARGET_STEPS:-100003840}"
actor_threads="${ACTOR_THREADS:-15}"
num_minibatches="${NUM_MINIBATCHES:-80}"
save_interval="${SAVE_INTERVAL:-65}"
log_frequency="${LOG_FREQUENCY:-10}"
local_envs="${LOCAL_NUM_ENVS:-16}"
env_threads="${LOCAL_ENV_THREADS:-11}"
procedure="$repo/scripts/script/procedure.lua"
procedure_sha256="${PROCEDURE_SHA256:-}"

test "$((parent_steps + continuation_steps))" -eq "$target_steps"
test ! -e "$run"
test -f "$parent"
test -f "$module"
test -f "$coverage"
grep -Fxq "commit=$source_commit" "$source_marker"
grep -Fxq "commit=$native_commit" "$root/training-runs/native-deploy-${native_commit:0:7}.txt"
echo "$parent_sha256  $parent" | sha256sum -c -
echo "$native_sha256  $module" | sha256sum -c -
if [[ -n "$procedure_sha256" ]]; then
  echo "$procedure_sha256  $procedure" | sha256sum -c -
else
  procedure_sha256="$(sha256sum "$procedure" | cut -d' ' -f1)"
fi
echo '6fe9f7c92ee2ed68c9b4b9cd81c7574d12b279ffa139e1f664427b9e54b77933  '"$coverage" | sha256sum -c -
test "$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "$parent.metadata.json")" = "$parent_sha256"
"$python" -c 'import hashlib,json,sys; c=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2])); assert r["summary"]["uncovered_branches"] == 0; assert r["contract_sha256"] == c["contract_sha256"]; assert r["inputs"]["adapter"]["sha256"] == hashlib.sha256(open(sys.argv[3],"rb").read()).hexdigest(); assert r["inputs"]["boundary_test"]["sha256"] == hashlib.sha256(open(sys.argv[4],"rb").read()).hexdigest()' "$contract" "$coverage" "$repo/ygoenv/ygoenv/ygopro/ygopro.h" "$repo/tests/test_ygocore_protocol_boundaries.py"
test -d "$root/training-runs/deck-corpus-runtime-v1-20260923/decks"
test -f "$root/training-runs/deck-clusters-v1-20260923/sampling-manifest.tsv"
test -f "$root/training-runs/deck-corpus-v1-20260923/manifest.json"
test -f "$root/training-runs/deck-clusters-v1-20260923/manifest.json"
test -f "$root/training-runs/deck-clusters-v1-20260923/sampling-manifest.json"
test -f "$root/training-runs/deck-corpus-raw-20260923/corpus/frozen-parent-assets/code_list.crlf.txt"
test -d "$root/training-runs/structured-lite-local-validation-20260921/pinned-semantics"
if ps -eo args | grep -Eq '[s]cripts/(cleanba|eval_structured)\.py'; then
  echo 'Refusing to start alongside train/eval' >&2
  exit 1
fi

mkdir -p "$run/checkpoints"
cp "$0" "$run/launch.sh"
printf '%s\n' \
  "source_commit=$source_commit" \
  "native_sha256=$native_sha256" \
  "procedure_sha256=$procedure_sha256" \
  'protocol_coverage_sha256=6fe9f7c92ee2ed68c9b4b9cd81c7574d12b279ffa139e1f664427b9e54b77933' \
  "parent_checkpoint_sha256=$parent_sha256" \
  "parent_checkpoint=$parent" \
  "parent_cumulative_steps=$parent_steps" \
  "continuation_steps=$continuation_steps" \
  "target_cumulative_steps=$target_steps" \
  'continuation=weights-plus-sampler-state-fresh-optimizer' \
  'candidate_q_mode=off' \
  'learning_rate=0.0001' \
  'max_step_loss=2.0' \
  "num_actor_threads=$actor_threads" \
  "local_num_envs=$local_envs" \
  "local_env_threads=$env_threads" \
  "sampler_repartition=${SAMPLER_REPARTITION:-0}" \
  "num_minibatches=$num_minibatches" \
  > "$run/initialization.txt"
cp "$source_marker" "$run/source.txt"

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$root/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_PRELOAD="$repo/libcompat_glibc.so"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$repo/ygoenv:$repo"
export PYTHONFAULTHANDLER=1

cd "$repo"
"$python" scripts/capture_compute_manifest.py \
  --repo-root "$repo" \
  --label "$(basename "$run")" \
  --output "$run/compute-manifest.json" \
  --artifact "$parent"

args=(
  --ckpt-dir "$run/checkpoints"
  --run-name "$run_name"
  --checkpoint "$parent"
  --tb-offset "$parent_steps"
  --learning-rate 0.0001
  --max-step-loss 2.0
  --timeout 600
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
  --max-options 128
  --local-num-envs "$local_envs"
  --local-env-threads "$env_threads"
  --num-actor-threads "$actor_threads"
  --actor-device-ids 0
  --learner-device-ids 0
  --num-steps 64
  --collect-steps 64
  --num-minibatches "$num_minibatches"
  --total-timesteps "$continuation_steps"
  --save-interval "$save_interval"
  --max-checkpoints 12
  --eval-interval 0
  --tb-dir None
  --log-frequency "$log_frequency"
)
if [[ "${SAMPLER_REPARTITION:-0}" == 1 ]]; then
  args+=(--deck-sampler-repartition)
fi
printf '%q ' "$python" scripts/cleanba.py "${args[@]}" > "$run/launch-command.txt"
printf '\n' >> "$run/launch-command.txt"

if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then
  set +e
  "$python" -u scripts/cleanba.py "${args[@]}" --config-only > "$run/train.log" 2>&1
  status=$?
  set -e
else
  set +e
  gdb --batch --return-child-result \
    -ex 'set pagination off' \
    -ex 'set confirm off' \
    -ex 'set print thread-events off' \
    -ex 'handle SIGPIPE nostop noprint pass' \
    -ex run \
    -ex 'info sharedlibrary' \
    -ex 'info registers rdi rsi rip' \
    -ex 'p $_siginfo' \
    -ex 'thread apply all bt 24' \
    --args "$python" -u scripts/cleanba.py "${args[@]}" > "$run/train.log" 2>&1
  status=$?
  set -e
fi

if [[ $status -ne 0 ]]; then
  printf 'failed exit=%s UTC=%s\n' "$status" "$(date -u +%FT%TZ)" > "$run/failed.txt"
  exit "$status"
fi

if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then
  printf 'config-only-passed UTC=%s\n' "$(date -u +%FT%TZ)" > "$run/config-only-passed.txt"
else
  find "$run/checkpoints" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$run/checkpoint-sha256.txt"
  printf 'completed UTC=%s\n' "$(date -u +%FT%TZ)" > "$run/completed.txt"
fi
