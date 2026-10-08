#!/usr/bin/env bash
set -euo pipefail
# core install prefix, Lua static archive, new isolated output path
test "$#" -eq 3
test ! -e "$3"
g++ -std=c++17 -shared -fPIC -O2 \
  -I "$1/include/ygopro-core" "$(dirname "$0")/core_bridge.cpp" \
  "$1/lib/libygopro-core.a" "$2" -ldl -lm -o "$3"
