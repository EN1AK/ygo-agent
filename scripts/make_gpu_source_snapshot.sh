#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/mnt/d/workspace/ygo-agent}"
output="${2:-/mnt/d/workspace/dist/ygo-agent-source-snapshot-20260924.tar.gz}"

mkdir -p "$(dirname "$output")"
tar -C "$repo_root" -czf "$output" \
  --exclude=.git \
  --exclude=training-runs \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  .
sha256sum "$output"
tar -tzf "$output" | wc -l
