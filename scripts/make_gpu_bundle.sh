#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/mnt/d/workspace/ygo-agent}"
output="${2:-/mnt/d/workspace/dist/add-clustered-multideck-training-20260924.tar.gz}"
list_file=$(mktemp)
trap 'rm -f "$list_file"' EXIT

cd "$repo_root"
{
  git diff --name-only
  git ls-files --others --exclude-standard
} | awk 'NF && $0 !~ /^training-runs\// && $0 !~ /^\.codex\//' | sort -u > "$list_file"

mkdir -p "$(dirname "$output")"
tar -T "$list_file" -czf "$output"
sha256sum "$output"
printf 'files='
wc -l < "$list_file"
