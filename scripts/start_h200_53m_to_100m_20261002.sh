#!/usr/bin/env bash
set -euo pipefail

root=/root/ygo-agent-gpu-20260910
repo="$root/ygo-agent"
export RUN_DIR="${RUN_DIR:-$root/training-runs/multideck-disablecheck-53m-to100m-92154ab-20261002}"
export RUN_NAME="${RUN_NAME:-multideck-disablecheck-53m-to100m-92154ab__02102026}"
export PARENT_CHECKPOINT="$root/training-runs/multideck-unselect-forward-45m-to100m-f751782-20261002/checkpoints/2102026_step_000053002240.flax_model"
export PARENT_SHA256=efaa6448201e49778b1604f56efa294bd05c1518861aed30bdf55051078c6fda
export PARENT_STEPS=53002240
export CONTINUATION_STEPS=47001600
export NATIVE_COMMIT=92154ab7a165c5749cead7d10f5f862aeb63f3bf
export NATIVE_SHA256=5962816be0362007c751c43a515d18a50abcd7716c00be579aac0aba1e6705a8

if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then
  exec bash "$repo/scripts/launch_h200_45m_to_100m_20261002.sh"
fi
exec bash "$repo/scripts/start_h200_45m_to_100m_20261002.sh"
