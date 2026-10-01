#!/usr/bin/env bash
set -euo pipefail

root=/root/ygo-agent-gpu-20260910
repo="$root/ygo-agent"
run="$root/training-runs/multideck-unselect-forward-45m-to100m-f751782-20261002"
launcher_log="$root/training-runs/multideck-unselect-forward-45m-to100m-f751782-20261002-launcher.log"
pid_file="$root/training-runs/multideck-unselect-forward-45m-to100m-f751782-20261002-launcher.pid"
source_commit="${SOURCE_COMMIT:?Set SOURCE_COMMIT to the deployed full Git revision}"

test ! -e "$run"
test ! -e "$launcher_log"
test ! -e "$pid_file"
if ps -eo args | grep -Eq '[s]cripts/(cleanba|eval_structured)\.py'; then
  echo 'Refusing to start alongside train/eval' >&2
  exit 1
fi

SOURCE_COMMIT="$source_commit" RUN_DIR="$run" \
  setsid bash "$repo/scripts/launch_h200_45m_to_100m_20261002.sh" \
  > "$launcher_log" 2>&1 < /dev/null &
pid=$!
case "$pid" in
  ''|*[!0-9]*) echo "Invalid launcher PID: $pid" >&2; exit 1 ;;
esac
printf '%s\n' "$pid" > "$pid_file"
sleep 5
kill -0 "$pid"
printf 'launcher_pid=%s\nrun=%s\nlog=%s\n' "$pid" "$run" "$launcher_log"
