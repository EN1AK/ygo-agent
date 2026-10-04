"""Read-only frozen-runtime chain capture. Never installs or changes native code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def worker(source, output, arguments):
    sys.path[:0] = [str(source / "ygoenv"), str(source)]
    import ygoai.windbot_protocol as protocol
    translate = protocol.translate_server_packet
    # Only public chain messages: no hand/deck packet capture.
    with (output / "chain-packets.jsonl").open("x", encoding="utf-8") as trace:
        def capture(packet):
            if len(packet) > 1 and packet[0] == 1 and 70 <= packet[1] <= 76:
                trace.write(json.dumps({"index": capture.count, "message": packet[1],
                                        "packet_hex": packet.hex()}) + "\n")
                trace.flush()
                capture.count += 1
            return translate(packet)
        capture.count = 0
        protocol.translate_server_packet = capture
        sys.argv = [str(source / "scripts/eval_structured.py"), *arguments]
        runpy.run_path(sys.argv[0], run_name="__main__")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-command", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cmd = json.loads(args.prior_command.read_text())
    start = next(i for i, value in enumerate(cmd) if value.endswith("/scripts/eval_structured.py"))
    source = Path(cmd[start]).parent.parent
    python = cmd[start - 2] if cmd[start - 1] == "-u" else cmd[start - 1]
    options = cmd[start + 1:]
    options = options[:options.index("--diagnostic-steps")]
    for option, path in (("--diagnostic-dir", out / "fixtures"),
                         ("--windbot-log-dir", out / "windbot"),
                         ("--windbot-metadata", out / "metadata.json"),
                         ("--decision-log", out / "decisions.jsonl")):
        options[options.index(option) + 1] = str(path)
    options += ["--diagnostic-steps", *map(str, range(12))]
    cp = Path(options[options.index("--checkpoint") + 1])
    native = source / "ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so"
    assert sha(cp) == "dc29c11254c0418a4b43cc29083db3d05e17b5c8e3dacd5c27b17d63c8670721"
    assert sha(native) == "a2dbd2604fec0e01ad722d887c639c00ed4c5aa74c912bd8f37d69f57a815486"
    evidence = json.loads((args.evidence / "SHA256SUMS.json").read_text())
    for name, expected in evidence.items():
        assert sha(args.evidence / name) == expected, name
    manifest = {"checkpoint": str(cp), "checkpoint_sha256": sha(cp),
                "native": str(native), "native_sha256": sha(native),
                "source": str(source), "source_header_sha256": sha(source / "ygoenv/ygoenv/ygopro/ygopro.h"),
                "evidence_files_verified": len(evidence), "script_sha256": sha(__file__),
                "device": "cpu", "production_modified": False}
    manifest["processes_before"] = subprocess.run(["pgrep", "-af", "cleanba"], capture_output=True, text=True).stdout
    dump(out / "manifest.json", manifest)
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH=f"{source}/ygoenv:{source}",
               LD_PRELOAD=str(args.release / "libcompat_glibc.so"))
    cpus = sorted(os.sched_getaffinity(0))[-2:]
    command = ["nice", "-n", "15", "taskset", "-c", ",".join(map(str, cpus)),
               python, "-u", str(Path(__file__).resolve()), "--worker", str(source), str(out), *options]
    dump(out / "command.json", command)
    with (out / "engine.log").open("x") as log:
        proc = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=360)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise RuntimeError("Diagnostic timeout; evidence retained")
    manifest["returncode"] = rc
    manifest["native_sha256_after"] = sha(native)
    manifest["checkpoint_sha256_after"] = sha(cp)
    manifest["processes_after"] = subprocess.run(["pgrep", "-af", "cleanba"], capture_output=True, text=True).stdout
    dump(out / "manifest.json", manifest)
    assert rc == 0, rc
    assert manifest["native_sha256_after"] == manifest["native_sha256"]
    assert manifest["checkpoint_sha256_after"] == manifest["checkpoint_sha256"]
    (out / "capture-completed.txt").write_text("Capture complete; provenance audit still required.\n")


if __name__ == "__main__":
    main()
