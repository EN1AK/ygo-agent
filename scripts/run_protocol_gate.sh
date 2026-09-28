#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/mnt/d/workspace/ygo-agent}"
runtime_root="${2:-/home/ygo/ygo-agent}"
gate_dir="$repo_root/training-runs/protocol-gate-v1-20260923"
python_bin="$runtime_root/.venv-wsl/bin/python"

mkdir -p "$repo_root/assets/protocol" "$gate_dir"
cd "$repo_root"

audit_args=(
  --core-source training-runs/protocol-source-ygopro-core-f969296
  --source-bundle training-runs/ygopro-core-f969296.bundle
  --adapter ygoenv/ygoenv/ygopro/ygopro.h
  --expected-revision f96929650ff8685b82fd48670126eae406366734
)

"$python_bin" scripts/audit_ygocore_protocol.py \
  "${audit_args[@]}" --output "$gate_dir/protocol-a.json"
"$python_bin" scripts/audit_ygocore_protocol.py \
  "${audit_args[@]}" --output "$gate_dir/protocol-b.json"
cmp "$gate_dir/protocol-a.json" "$gate_dir/protocol-b.json"
cp "$gate_dir/protocol-a.json" \
  assets/protocol/ygopro-core-f969296-protocol.json

PYTHONPATH="$repo_root/ygoenv:$repo_root" "$python_bin" \
  scripts/export_model_input_domains.py \
  --output assets/protocol/model-input-domains.json

native_module=$(PYTHONPATH="$repo_root/ygoenv:$repo_root" "$python_bin" - <<'PY'
import ygoenv.ygopro.ygopro_ygoenv as native
print(native.__file__)
PY
)

sha256sum \
  assets/protocol/ygopro-core-f969296-protocol.json \
  assets/protocol/model-input-domains.json \
  ygoenv/ygoenv/ygopro/ygopro.h \
  training-runs/ygopro-core-f969296.bundle \
  "$native_module" \
  > "$gate_dir/artifact-sha256.txt"

"$python_bin" - "$repo_root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
contract = json.loads(
    (root / "assets/protocol/ygopro-core-f969296-protocol.json").read_text()
)
print("contract_sha256=" + contract["contract_sha256"])
print("counts=" + json.dumps(contract.get("counts", {}), sort_keys=True))
print((root / "training-runs/protocol-gate-v1-20260923/artifact-sha256.txt").read_text(), end="")
PY
