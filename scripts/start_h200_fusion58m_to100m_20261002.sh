#!/usr/bin/env bash
set -euo pipefail
root=/root/ygo-agent-gpu-20260910
export SOURCE_COMMIT="${SOURCE_COMMIT:?Use the verified four-way source revision}"
export NATIVE_COMMIT=92154ab7a165c5749cead7d10f5f862aeb63f3bf
export NATIVE_SHA256=5962816be0362007c751c43a515d18a50abcd7716c00be579aac0aba1e6705a8
export PROCEDURE_SHA256=c2742f65f750092ef4298889391bc718c52ece5fad8d07c5a6e2ed69e0fe1322
export PARENT_CHECKPOINT="$root/training-runs/multideck-fastest-56m-to100m-913b40d-20261002/checkpoints/2102026_step_000057994240.flax_model"
export PARENT_SHA256=ac5dcc0c8ab533f5341f89d78f6fed365568563afa123bb45a766eb3ca560fee
export PARENT_STEPS=57994240 CONTINUATION_STEPS=42009600 TARGET_STEPS=100003840
export ACTOR_THREADS=5 LOCAL_NUM_ENVS=48 LOCAL_ENV_THREADS=11 NUM_MINIBATCHES=80
export SAMPLER_REPARTITION=0
export RUN_NAME="multideck-fusion-canonical-58m-to100m-${SOURCE_COMMIT:0:7}__02102026"
export RUN_DIR="${RUN_DIR:-$root/training-runs/multideck-fusion-canonical-58m-to100m-${SOURCE_COMMIT:0:7}-20261002}"
if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then
  exec bash "$root/ygo-agent/scripts/launch_h200_45m_to_100m_20261002.sh"
fi
exec bash "$root/ygo-agent/scripts/start_h200_45m_to_100m_20261002.sh"
