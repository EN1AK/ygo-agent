"""Build a separate, hash-pinned chain repair with legacy selection semantics.

Never install into an existing runtime. The archived source is patched only in
the newly created build directory; active training and the checkout are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(path, before, after):
    content = path.read_text()
    if content.count(before) != 1:
        raise RuntimeError(f"Unexpected source in {path}")
    path.write_text(content.replace(before, after))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--archive-sha256", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--xmake", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    args = p.parse_args()
    assert sha(args.archive) == args.archive_sha256
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.archive) as archive:
        for item in archive.getmembers():
            target = (root / item.name).resolve()
            assert target.is_relative_to(root)
            assert item.isfile() or item.isdir(), item.name
        archive.extractall(root)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".h", ".cpp", ".lua", ".sh"}:
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    header = root / "ygoenv/ygoenv/ygopro/ygopro.h"
    binding = root / "ygoenv/ygoenv/ygopro/ygopro.cpp"
    original = {str(x.relative_to(root)): sha(x) for x in (header, binding)}
    replace_once(header,
        'selection(9) = static_cast<uint8_t>(std::any_of(\n'
        '        legal_actions_.begin(), legal_actions_.end(),\n'
        '        [](const LegalAction &action) { return action.finish_; }));',
        '// Explicit legacy compatibility for previously trained checkpoints.\n'
        '    selection(9) = static_cast<uint8_t>(selection_finishable_);')
    replace_once(binding, 'PYBIND11_MODULE(ygopro_ygoenv, m) {',
        'PYBIND11_MODULE(ygopro_ygoenv, m) {\n'
        '  m.attr("selection_finish_version") = "legacy-prompt-field-v1";')
    env = dict(os.environ, PYTHON=str(args.python), CCACHE_DISABLE="1",
               PATH=str(args.python.parent) + os.pathsep + os.environ["PATH"])
    configure = [str(args.xmake), "f", "-P", str(root), "-c", "-m", "release",
                 "--native_optimization=n", "--ccache=n", "--confirm=yes",
                 "--cxxflags=-DYGO_CHAIN_EVENT_PROVENANCE_V2"]
    build = [str(args.xmake), "b", "-P", str(root), "-r", "-j", "2", "-v",
             "--confirm=yes", "ygopro_ygoenv"]
    record = {"source_commit": args.source_commit,
              "archive_sha256": args.archive_sha256,
              "chain_event_provenance_version": "chain-source-by-link-v2",
              "selection_finish_version": "legacy-prompt-field-v1",
              "original_source_sha256": original,
              "patched_source_sha256": {str(x.relative_to(root)): sha(x) for x in (header, binding)},
              "configure": configure, "build": build,
              "compiler_cache_disabled": True,
              "scope": "Isolated compatibility candidate; replay acceptance required."}
    manifest = root / "compat-build.json"
    manifest.write_text(json.dumps(record, indent=2))
    for label, command in (("configure", configure), ("build", build)):
        with (root / f"compat-{label}.log").open("x") as log:
            result = subprocess.run(command, cwd=root, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        record[label + "_exit"] = result.returncode
        manifest.write_text(json.dumps(record, indent=2))
        result.check_returncode()
    native = root / "ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so"
    record["native_sha256"] = sha(native)
    record["status"] = "built_not_promoted"
    manifest.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
