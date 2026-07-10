from pathlib import Path
import sys
from typing import Optional

import _repo_bootstrap  # noqa: F401
import tyro

from ygoai.windbot import (
    WindBotConfig,
    WindBotError,
    WindBotProcess,
    allocate_port,
    run_connection_smoke,
    validate_config,
    write_metadata,
)


def _make_config(
    executable: str,
    workdir: Optional[str],
    deck: str,
    name: str,
    host: str,
    port: int,
    host_info: str,
    password: str,
    dialog: bool,
    timeout: float,
    log_dir: str,
    mono: Optional[str],
    source_revision: Optional[str],
    server_mode: bool,
    server_host: str,
    server_port: int,
) -> WindBotConfig:
    if port == 0:
        port = allocate_port(host)
    return WindBotConfig(
        executable=executable,
        workdir=workdir,
        deck=deck,
        name=name,
        host=host,
        port=port,
        host_info=host_info,
        password=password,
        dialog=dialog,
        timeout=timeout,
        log_dir=log_dir,
        mono=mono,
        source_revision=source_revision,
        server_mode=server_mode,
        server_host=server_host,
        server_port=server_port,
    )


def validate(
    executable: str,
    workdir: Optional[str] = None,
    deck: str = "AI_Default",
    name: str = "WindBot",
    host: str = "127.0.0.1",
    port: int = 7911,
    host_info: str = "",
    password: str = "",
    dialog: bool = False,
    timeout: float = 30.0,
    log_dir: str = "logs/windbot",
    mono: Optional[str] = None,
    source_revision: Optional[str] = None,
    server_mode: bool = False,
    server_host: str = "127.0.0.1",
    server_port: int = 2399,
    metadata: Optional[str] = None,
) -> None:
    """Validate local WindBot executable, cards.cdb, deck, and port settings."""
    config = _make_config(
        executable, workdir, deck, name, host, port, host_info, password, dialog, timeout, log_dir, mono,
        source_revision, server_mode, server_host, server_port
    )
    info = validate_config(config)
    if metadata:
        write_metadata(config, metadata)
    print(info)


def launch(
    executable: str,
    workdir: Optional[str] = None,
    deck: str = "AI_Default",
    name: str = "WindBot",
    host: str = "127.0.0.1",
    port: int = 7911,
    host_info: str = "",
    password: str = "",
    dialog: bool = False,
    timeout: float = 30.0,
    log_dir: str = "logs/windbot",
    mono: Optional[str] = None,
    source_revision: Optional[str] = None,
    server_mode: bool = False,
    server_host: str = "127.0.0.1",
    server_port: int = 2399,
    metadata: Optional[str] = None,
) -> None:
    """Launch WindBot and wait until it exits; intended for setup debugging."""
    config = _make_config(
        executable, workdir, deck, name, host, port, host_info, password, dialog, timeout, log_dir, mono,
        source_revision, server_mode, server_host, server_port
    )
    if metadata:
        write_metadata(config, metadata)
    with WindBotProcess(config) as proc:
        print(f"Started WindBot pid={proc.process.pid} logs={proc.stdout_path},{proc.stderr_path}")
        proc.process.wait()
        print(f"WindBot exited with code {proc.process.returncode}")


def smoke(
    executable: str,
    workdir: Optional[str] = None,
    deck: str = "AI_Default",
    name: str = "WindBot",
    host: str = "127.0.0.1",
    port: int = 0,
    host_info: str = "",
    password: str = "",
    dialog: bool = False,
    timeout: float = 30.0,
    log_dir: str = "logs/windbot",
    mono: Optional[str] = None,
    source_revision: Optional[str] = None,
    server_mode: bool = False,
    server_host: str = "127.0.0.1",
    server_port: int = 2399,
    metadata: str = "logs/windbot/windbot-metadata.json",
) -> None:
    """Validate setup and confirm WindBot can connect to a minimal YGOPro TCP host."""
    config = _make_config(
        executable, workdir, deck, name, host, port, host_info, password, dialog, timeout, log_dir, mono,
        source_revision, server_mode, server_host, server_port
    )
    info = validate_config(config)
    write_metadata(config, metadata)
    result = run_connection_smoke(config)
    print({"validation": info, "connection": result})


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        tyro.extras.subcommand_cli_from_dict(
            {
                "validate": validate,
                "launch": launch,
                "smoke": smoke,
            }
        )
    except WindBotError as exc:
        raise SystemExit(f"WindBot error: {exc}") from exc


if __name__ == "__main__":
    main()
