#!/usr/bin/env bash
set -euo pipefail

runtime_root="${1:-/home/ygo/ygo-agent}"
workspace_root="${2:-/mnt/d/workspace/ygo-agent}"
output="${3:-/mnt/d/workspace/dist/multideck-pilot-assets-20260924.tar.gz}"

mkdir -p "$(dirname "$output")"
tar -czf "$output" \
  -C "$runtime_root" \
  training-runs/structured-lite-full-cont-39m-20260922-night/checkpoints/1790011620_step_000040000000.flax_model \
  training-runs/deck-corpus-runtime-v1-20260923/decks \
  -C "$workspace_root" \
  training-runs/deck-clusters-v1-20260923 \
  training-runs/deck-corpus-v1-20260923 \
  training-runs/deck-corpus-raw-20260923/corpus/frozen-parent-assets \
  training-runs/structured-lite-local-validation-20260921/pinned-semantics

sha256sum "$output"
tar -tzf "$output" | wc -l
