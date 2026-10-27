#!/usr/bin/env bash
# Build in the existing portable toolchain; never replace a production module.
set -euo pipefail
test "$#" -eq 3 # exact source archive, new build directory, SQLiteCpp archive
archive=$(realpath "$1")
base="$2"
sqlite=$(realpath "$3")
test ! -e "$base"
mkdir -p "$base/source"
base=$(realpath "$base")
tar -xf "$archive" -C "$base/source"
find "$base/source/scripts/exercises" -name '*.sh' -exec sed -i 's/\r$//' {} +
sha256sum "$archive" > "$base/source-archive.sha256"
trap 'echo $? > "$base/build.exit"' EXIT
docker run --rm -v "$base/source:/work/crystron-curriculum" -v "$base:/result" \
  -v "$sqlite:/sqlitecpp.tar.gz:ro" ygo-build-cache:replay-fix \
  bash -lc 'cd /work/crystron-curriculum && bash scripts/exercises/build_actor_bridge.sh /sqlitecpp.tar.gz /result/native' \
  > "$base/build.log" 2>&1
