#!/usr/bin/env bash
# Run in the pinned ygo-build-cache:replay-fix image, in a fresh source archive.
set -euo pipefail
test "$#" -eq 2 # SQLiteCpp 3.2.1 tarball, new output directory
test ! -e "$2"
mkdir -p "$2"
out=$(realpath "$2")
sqlite_archive=$(realpath "$1")
echo '70c67d5680c47460f82a7abf8e6b0329bf2fb10795a982a6d8abc06adb42d693  '"$sqlite_archive" | sha256sum -c -
tar -xf "$sqlite_archive" -C "$out"
python3 scripts/exercises/prepare_actor_header.py ygoenv/ygoenv/ygopro/ygopro.h "$out/exercise_ygopro.h"
cp -a /root/.xmake/cache/packages/2609/y/ygopro-core/0.0.2/source/ygopro-core "$out/core"
ln -s core "$out/ygopro-core"
patch -d "$out/core" -p1 < repo/packages/y/ygopro-core/patches/0.0.3/empty-deck-guards.patch
patch -d "$out/core" -p1 < repo/packages/y/ygopro-core/patches/0.0.4/disable-check-pointer-guards.patch
lua=/root/.xmake/packages/l/lua/v5.3.6/e36ca01661cc4da38a40db5fbb826426
includes=(-I "$out" -I ygoenv -I "$out/core" -I "$out/SQLiteCpp-3.2.1/include" -I /usr/include/python3.10 -I "$lua/include/lua")
for path in /root/.xmake/packages/{f/fmt,g/glog,g/gflags,c/concurrentqueue,u/unordered_dense,p/pybind11,l/lua}/*/*/include; do
  includes+=(-I "$path")
done
includes+=(-I /root/.xmake/packages/s/sqlite3/3.43.0+200/95fcc1aa9fa6461d859b22bc16ee0811/include)
mkdir -p "$out/objects"
for file in "$out/core/"*.cpp "$out/SQLiteCpp-3.2.1/src/"*.cpp; do
  g++ -std=c++17 -O2 -fPIC -DNDEBUG "${includes[@]}" -c "$file" -o "$out/objects/$(basename "$file").o"
done
g++ -std=c++17 -O2 -fPIC -DNDEBUG -shared "${includes[@]}" scripts/exercises/actor_bridge.cpp scripts/exercises/core_bridge.cpp \
  "$out/objects/"*.o \
  /root/.xmake/packages/f/fmt/10.2.1/*/lib/libfmt.a \
  /root/.xmake/packages/g/glog/v0.6.0/*/lib/libglog.a \
  /root/.xmake/packages/g/gflags/v2.3.1/*/lib/libgflags.a \
  /root/.xmake/packages/s/sqlite3/3.43.0+200/*/lib/libsqlite3.a \
  "$lua/lib/liblua.a" -ldl -lm -lpthread -o "$out/exercise_actor_native.cpython-310-x86_64-linux-gnu.so"
sha256sum "$out/exercise_actor_native.cpython-310-x86_64-linux-gnu.so"
