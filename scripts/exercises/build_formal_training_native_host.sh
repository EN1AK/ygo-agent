#!/usr/bin/env bash
set -euo pipefail
job="${3:-/root/elfnote-formal-native-v3-20261023}"
source="${1:-/root/elfnote-selected-scope-build-20261014}"
cache="${2:-/root/elfnote-replay-build-20261014/output}"
builder="$(cd -- "$(dirname -- "$0")" && pwd)/build_formal_training_native_container.sh"
test ! -e "$job"
test "$(docker image inspect ygo-build-cache:replay-fix --format='{{.Id}}')" = sha256:be59db1ab77bfada7a5b111e873520ae2fbe96b70cf754d2e1c656fe1781f96a
mkdir "$job"
sha256sum "$source/ygoenv/ygoenv/ygopro/ygopro.h" "$source/output03/exercise_actor_native.cpython-310-x86_64-linux-gnu.so" > "$job/protected-before.sha256"
cp "$builder" "$job/build-container.sh"
sed -i 's/\r$//' "$job/build-container.sh"
bash -n "$job/build-container.sh"
set +e
docker run --rm --name elfnote-formal-native-v3-20261023 -v "$source:/src:ro" -v "$cache:/cache:ro" -v "$job:/out" -v "$job/build-container.sh:/run-build.sh:ro" --entrypoint bash ygo-build-cache:replay-fix /run-build.sh > "$job/build.log" 2>&1
status=$?
printf '%s\n' "$status" > "$job/exit"
sha256sum -c "$job/protected-before.sha256"
exit "$status"
