import argparse
import contextlib
import hashlib
import json
import os
import stat
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZH_CDB_URL = "https://cdn02.moecube.com:444/ygopro-database/zh-CN/cards.cdb"
DEFAULT_SCRIPTS_REPO = "https://github.com/mycard/ygopro-scripts"
DEFAULT_MANIFEST = ROOT / "assets" / "asset_manifest.json"
TMP_PARENT = ROOT / "assets" / ".asset-update-tmp"
REQUIRED_SCRIPT_FILES = ("constant.lua", "utility.lua", "procedure.lua")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "assets": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def run_git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def download_file(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "ygo-agent-asset-updater/1.0"})
    with urlopen(request, timeout=60) as response:
        with dest.open("wb") as f:
            shutil.copyfileobj(response, f)
        return {
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "content_length": response.headers.get("Content-Length"),
        }


def validate_cards_cdb(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"cards.cdb is missing or empty: {path}")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise ValueError(f"SQLite integrity_check failed for {path}: {integrity}")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = {"datas", "texts"} - tables
        if missing:
            raise ValueError(f"cards.cdb missing required tables: {sorted(missing)}")
        card_count = int(conn.execute("SELECT COUNT(*) FROM datas").fetchone()[0])
    return card_count


def find_script_root(repo_path: Path) -> Path:
    candidates = [repo_path, repo_path / "script"]
    for candidate in candidates:
        if all((candidate / name).exists() for name in REQUIRED_SCRIPT_FILES):
            return candidate
    raise ValueError(
        "Unable to locate YGOPro script root containing "
        + ", ".join(REQUIRED_SCRIPT_FILES)
    )


def validate_scripts(path: Path) -> int:
    if not path.is_dir():
        raise ValueError(f"script path is not a directory: {path}")
    missing = [name for name in REQUIRED_SCRIPT_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"script path missing required files: {missing}")
    card_scripts = [p for p in path.glob("c*.lua") if p.stem[1:].isdigit()]
    if not card_scripts:
        raise ValueError(f"script path contains no c*.lua card scripts: {path}")
    return len(card_scripts)


def promote_file(staged: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    promotion_tmp = dest.with_name(dest.name + ".update-tmp")
    with contextlib.suppress(FileNotFoundError):
        promotion_tmp.unlink()
    shutil.copy2(staged, promotion_tmp)
    try:
        replace_with_retry(promotion_tmp, dest)
    finally:
        with contextlib.suppress(FileNotFoundError):
            promotion_tmp.unlink()


def replace_with_retry(source: Path, dest: Path, attempts: int = 10, delay: float = 0.25) -> None:
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, dest)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay)


def promote_dir(staged: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    backup = dest.with_name(dest.name + f".backup-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    moved_existing = False
    try:
        if dest.exists() or dest.is_symlink():
            if dest.is_dir() and not dest.is_symlink():
                dest.rename(backup)
            else:
                replace_with_retry(dest, backup)
            moved_existing = True
        staged.rename(dest)
    except Exception:
        if moved_existing and not (dest.exists() or dest.is_symlink()) and backup.exists():
            backup.rename(dest)
        raise
    else:
        if backup.exists() or backup.is_symlink():
            try:
                remove_path(backup)
            except OSError as exc:
                print(f"Warning: activated {dest} but could not remove backup {backup}: {exc}")


def remove_path(path: Path) -> None:
    def make_writable(func, target, exc_info):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, onerror=make_writable)
    else:
        path.unlink()


def update_zh_cdb(args: argparse.Namespace, stage_root: Path, manifest: dict) -> None:
    staged = stage_root / "zh" / "cards.cdb"
    headers = download_file(args.zh_cdb_url, staged)
    checksum = sha256_file(staged)
    if args.expected_zh_cdb_sha256 and checksum.lower() != args.expected_zh_cdb_sha256.lower():
        raise ValueError(
            f"Chinese cards.cdb checksum mismatch: expected {args.expected_zh_cdb_sha256}, got {checksum}"
        )
    card_count = validate_cards_cdb(staged)
    dest = ROOT / "assets" / "locale" / "zh" / "cards.cdb"
    promote_file(staged, dest)
    manifest["assets"]["zh_cards_cdb"] = {
        "source_url": args.zh_cdb_url,
        "destination": str(dest.relative_to(ROOT)),
        "sha256": checksum,
        "size": dest.stat().st_size,
        "card_count": card_count,
        "updated_at": now_utc(),
        "http": headers,
    }
    print(f"Updated {dest.relative_to(ROOT)} ({card_count} cards, sha256={checksum})")


def update_scripts(args: argparse.Namespace, stage_root: Path, manifest: dict) -> None:
    repo_stage = stage_root / "ygopro-scripts-repo"
    run_git(["clone", "--depth", "1", args.scripts_repo, str(repo_stage)])
    if args.scripts_ref:
        run_git(["fetch", "--depth", "1", "origin", args.scripts_ref], cwd=repo_stage)
        run_git(["checkout", "--detach", "FETCH_HEAD"], cwd=repo_stage)
    commit = run_git(["rev-parse", "HEAD"], cwd=repo_stage)
    script_root = find_script_root(repo_stage)
    staged_scripts = stage_root / "script"
    shutil.copytree(script_root, staged_scripts, ignore=shutil.ignore_patterns(".git"))
    script_count = validate_scripts(staged_scripts)
    dest = ROOT / "scripts" / "script"
    promote_dir(staged_scripts, dest)
    manifest["assets"]["ygopro_scripts"] = {
        "source_url": args.scripts_repo,
        "destination": str(dest.relative_to(ROOT)),
        "git_commit": commit,
        "script_count": script_count,
        "updated_at": now_utc(),
    }
    print(f"Updated {dest.relative_to(ROOT)} ({script_count} card scripts, commit={commit})")


def validate_active_assets(args: argparse.Namespace) -> None:
    zh_cdb = ROOT / "assets" / "locale" / "zh" / "cards.cdb"
    scripts_dir = ROOT / "scripts" / "script"
    card_count = validate_cards_cdb(zh_cdb)
    script_count = validate_scripts(scripts_dir)
    print(f"Validated {zh_cdb.relative_to(ROOT)} ({card_count} cards)")
    print(f"Validated {scripts_dir.relative_to(ROOT)} ({script_count} card scripts)")
    if args.run_code_list:
        output_path = Path(args.code_list_output) if args.code_list_output else ROOT / "scripts" / "code_list.txt"
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "card" / "code_list.py"),
                "--cdb",
                str(zh_cdb),
                "--script-dir",
                str(scripts_dir),
                "--output",
                str(output_path),
            ],
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update external YGO runtime assets explicitly.")
    parser.add_argument(
        "--asset",
        choices=("all", "zh-cdb", "scripts"),
        default="all",
        help="Which asset group to update.",
    )
    parser.add_argument("--zh-cdb-url", default=DEFAULT_ZH_CDB_URL)
    parser.add_argument("--expected-zh-cdb-sha256", default=None)
    parser.add_argument("--scripts-repo", default=DEFAULT_SCRIPTS_REPO)
    parser.add_argument("--scripts-ref", default=None, help="Optional commit, tag, or ref for ygopro-scripts.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--validate-only", action="store_true", help="Validate active local assets without updating.")
    parser.add_argument("--run-code-list", action="store_true", help="Run scripts/card/code_list.py after validation.")
    parser.add_argument("--code-list-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if args.validate_only:
        validate_active_assets(args)
        return 0

    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    with tempfile.TemporaryDirectory(prefix="update-", dir=TMP_PARENT, ignore_cleanup_errors=True) as tmp:
        stage_root = Path(tmp)
        if args.asset in ("all", "zh-cdb"):
            update_zh_cdb(args, stage_root, manifest)
        if args.asset in ("all", "scripts"):
            update_scripts(args, stage_root, manifest)
    write_manifest(manifest_path, manifest)
    print(f"Wrote manifest {manifest_path.relative_to(ROOT)}")
    if args.run_code_list:
        validate_active_assets(args)
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        raise SystemExit(main())
    raise SystemExit(130)
