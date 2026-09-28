#!/usr/bin/env bash
set -euo pipefail

root=/root/ygo-agent-gpu-20260910
repo="$root/ygo-agent"
bundle=/tmp/protocol64-source-bundle-20260926.tar.gz
coverage=/tmp/protocol-coverage-current.json
native=/tmp/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so
backup="$root/training-runs/deploy-backup-protocol64-20260926.tar.gz"
native_backup="$root/training-runs/deploy-backup-protocol64-20260926.so"
file_list="$root/training-runs/deploy-filelist-protocol64-20260926.txt"

test -d "$repo"
test ! -e "$backup"
test ! -e "$native_backup"
printf '%s  %s\n' \
  c692a8c7092b853c4ab284220d60d1524b08100d4a96a844cac6603b5710e039 "$bundle" \
  7404b19650f6a99f5aa64fb2e8926cf7ff623d1878ff81883fd1d6cdbe9399af "$coverage" \
  27f40ba7807f2cbb321ab3172676265feb4a80f927de3145d15fe1f76d263667 "$native" \
  | sha256sum -c -

tar -tzf "$bundle" > "$file_list"
if grep -Eq '(^/|(^|/)\.\.(/|$))' "$file_list"; then
  echo "Unsafe archive member" >&2
  exit 1
fi
tar -C "$repo" --ignore-failed-read -czf "$backup" -T "$file_list"
cp "$repo/ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so" "$native_backup"

tar -C "$repo" -xzf "$bundle"
mkdir -p "$repo/training-runs/protocol-gate-v1-20260923"
cp "$coverage" "$repo/training-runs/protocol-gate-v1-20260923/protocol-coverage-current.json"
install -m 755 "$native" \
  "$repo/ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so"
sha256sum \
  "$repo/ygoenv/ygoenv/ygopro/ygopro.h" \
  "$repo/ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so" \
  "$repo/assets/protocol/ygopro-core-f969296-protocol.json" \
  "$repo/training-runs/protocol-gate-v1-20260923/protocol-coverage-current.json"
