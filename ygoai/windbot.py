import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from http.client import HTTPConnection
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode


class WindBotError(RuntimeError):
    pass


class WindBotAdapterUnavailable(WindBotError):
    pass


@dataclass
class WindBotConfig:
    executable: str
    workdir: Optional[str] = None
    deck: str = "AI_Default"
    name: str = "WindBot"
    host: str = "127.0.0.1"
    port: int = 7911
    host_info: str = ""
    password: str = ""
    dialog: bool = False
    timeout: float = 30.0
    log_dir: str = "logs/windbot"
    mono: Optional[str] = None
    source_revision: Optional[str] = None
    server_mode: bool = False
    server_host: str = "127.0.0.1"
    server_port: int = 2399

    @property
    def executable_path(self) -> Path:
        return Path(self.executable).expanduser()

    @property
    def working_directory(self) -> Path:
        return Path(self.workdir).expanduser() if self.workdir else self.executable_path.parent

    @property
    def cards_cdb(self) -> Path:
        return self.working_directory / "cards.cdb"

    def command(self) -> list[str]:
        exe = self.executable_path
        command = [str(exe)]
        if self.mono:
            command = [self.mono, str(exe)]
        elif exe.suffix.lower() == ".exe" and os.name != "nt":
            command = ["mono", str(exe)]
        if self.server_mode:
            command.extend([f"ServerMode=true", f"ServerPort={self.server_port}"])
            return command
        command.extend(
            [
                f"Deck={self.deck}",
                f"Name={self.name}",
                f"Host={self.host}",
                f"Port={self.port}",
                f"Dialog={str(self.dialog).lower()}",
            ]
        )
        if self.host_info:
            command.append(f"HostInfo={self.host_info}")
        if self.password:
            command.append(f"Password={self.password}")
        return command

    def metadata(self) -> dict:
        data = asdict(self)
        data["workdir"] = str(self.working_directory)
        data["executable"] = str(self.executable_path)
        return data

    def add_bot_path(self) -> str:
        query = urlencode(
            {
                "name": self.name,
                "deck": self.deck,
                "host": self.host,
                "port": self.port,
                "dialog": str(self.dialog).lower(),
                "password": self.password or self.host_info,
            }
        )
        return f"/?{query}"


def allocate_port(host: str = "127.0.0.1") -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def check_port_available(host: str, port: int, label: str = "host") -> None:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise WindBotError(f"WindBot {label} port is not available: {host}:{port}") from exc


def validate_cards_cdb(path: Path) -> int:
    if not path.exists():
        raise WindBotError(f"WindBot cards.cdb not found: {path}")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = {"datas", "texts"} - tables
        if missing:
            raise WindBotError(f"WindBot cards.cdb missing required tables: {sorted(missing)}")
        return int(conn.execute("SELECT COUNT(*) FROM datas").fetchone()[0])


def parse_ydk_codes(path: Path) -> set[int]:
    codes: set[int] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                codes.add(int(line))
    return codes


def validate_deck(config: WindBotConfig) -> dict:
    deck_path = Path(config.deck).expanduser()
    if not deck_path.is_absolute():
        deck_path = config.working_directory / "Decks" / config.deck
    if not deck_path.suffix and not deck_path.exists():
        deck_path = deck_path.with_suffix(".ydk")
    if not deck_path.exists():
        return {
            "deck": config.deck,
            "deck_path": str(deck_path),
            "checked": False,
            "warning": "Deck was treated as a WindBot deck name and was not found as a local .ydk file.",
        }
    codes = parse_ydk_codes(deck_path)
    cdb_codes = read_cdb_codes(config.cards_cdb)
    missing_in_cdb = sorted(codes.difference(cdb_codes))
    if missing_in_cdb:
        raise WindBotError(f"WindBot deck contains cards missing in cards.cdb: {missing_in_cdb[:20]}")
    script_dir = config.working_directory / "script"
    if script_dir.exists():
        missing_scripts = sorted(code for code in codes if not (script_dir / f"c{code}.lua").exists())
        if missing_scripts:
            raise WindBotError(f"WindBot deck contains cards missing Lua scripts: {missing_scripts[:20]}")
    return {
        "deck": config.deck,
        "deck_path": str(deck_path),
        "checked": True,
        "card_count": len(codes),
    }


def read_cdb_codes(path: Path) -> set[int]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return {int(row[0]) for row in conn.execute("SELECT id FROM datas")}


def validate_config(config: WindBotConfig, check_port: bool = True) -> dict:
    exe = config.executable_path
    if not exe.exists():
        raise WindBotError(f"WindBot executable not found: {exe}")
    if not config.working_directory.exists():
        raise WindBotError(f"WindBot working directory not found: {config.working_directory}")
    card_count = validate_cards_cdb(config.cards_cdb)
    if check_port:
        check_port_available(config.host, config.port, "YGOPro host")
        if config.server_mode:
            check_port_available(config.server_host, config.server_port, "server-mode HTTP")
    deck_info = validate_deck(config)
    return {
        "executable": str(exe),
        "workdir": str(config.working_directory),
        "cards_cdb": str(config.cards_cdb),
        "card_count": card_count,
        "deck": deck_info,
        "host": config.host,
        "port": config.port,
        "source_revision": config.source_revision,
    }


class WindBotProcess:
    def __init__(self, config: WindBotConfig):
        self.config = config
        self.process: subprocess.Popen | None = None
        self.stdout_path: Path | None = None
        self.stderr_path: Path | None = None
        self._stdout = None
        self._stderr = None

    def start(self) -> subprocess.Popen:
        validate_config(self.config, check_port=False)
        log_dir = Path(self.config.log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.stdout_path = log_dir / f"windbot-{stamp}.stdout.log"
        self.stderr_path = log_dir / f"windbot-{stamp}.stderr.log"
        self._stdout = self.stdout_path.open("ab")
        self._stderr = self.stderr_path.open("ab")
        self.process = subprocess.Popen(
            self.config.command(),
            cwd=str(self.config.working_directory),
            stdout=self._stdout,
            stderr=self._stderr,
            stdin=subprocess.DEVNULL,
        )
        return self.process

    def stop(self, timeout: float = 5.0) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout)
        for handle in (self._stdout, self._stderr):
            if handle:
                handle.close()

    def add_bot(self) -> None:
        if not self.config.server_mode:
            return
        deadline = time.monotonic() + self.config.timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                raise WindBotError(f"WindBot server exited before bot add: code={self.process.returncode}")
            try:
                conn = HTTPConnection(self.config.server_host, self.config.server_port, timeout=1.0)
                conn.request("GET", self.config.add_bot_path())
                response = conn.getresponse()
                response.read()
                conn.close()
                if response.status < 500:
                    return
                last_error = WindBotError(f"WindBot server returned HTTP {response.status}")
            except Exception as exc:
                last_error = exc
                time.sleep(0.1)
        raise WindBotError(
            f"Timed out adding WindBot via server mode at "
            f"{self.config.server_host}:{self.config.server_port}: {last_error}"
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


def require_windbot_adapter() -> None:
    raise WindBotAdapterUnavailable(
        "WindBot process/configuration support is available, but the local YGOPro host "
        "adapter needed to use WindBot as an automated training opponent is not implemented yet."
    )


CTOS_NAMES = {
    0x01: "RESPONSE",
    0x02: "UPDATE_DECK",
    0x03: "HAND_RESULT",
    0x04: "TP_RESULT",
    0x10: "PLAYER_INFO",
    0x11: "CREATE_GAME",
    0x12: "JOIN_GAME",
    0x13: "LEAVE_GAME",
    0x14: "SURRENDER",
    0x15: "TIME_CONFIRM",
    0x16: "CHAT",
    0x20: "HS_TO_DUELIST",
    0x21: "HS_TO_OBSERVER",
    0x22: "HS_READY",
    0x23: "HS_NOT_READY",
    0x24: "HS_KICK",
    0x25: "HS_START",
}


@dataclass
class WindBotPacket:
    proto: int
    name: str
    payload_size: int


class WindBotSmokeHost:
    """Minimal YGOPro TCP host used only to verify that WindBot connects."""

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.packets: list[WindBotPacket] = []
        self.client_address: tuple[str, int] | None = None
        self.error: Exception | None = None
        self._ready = threading.Event()
        self._done = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="windbot-smoke-host", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise WindBotError(f"WindBot smoke host did not start on {self.host}:{self.port}")

    def wait_for_connection(self) -> dict:
        if not self._done.wait(timeout=self.timeout):
            raise WindBotError(
                f"WindBot did not connect to smoke host within {self.timeout:.1f}s "
                f"at {self.host}:{self.port}"
            )
        if self.error:
            raise WindBotError(f"WindBot smoke host failed: {self.error}") from self.error
        if self.client_address is None:
            raise WindBotError(
                f"WindBot did not connect to smoke host within {self.timeout:.1f}s "
                f"at {self.host}:{self.port}"
            )
        return {
            "client_address": self.client_address,
            "packets": [asdict(packet) for packet in self.packets],
        }

    def stop(self) -> None:
        self._stop.set()
        if self._socket:
            self._socket.close()
        if self._thread:
            self._thread.join(timeout=2.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def _run(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                self._socket = server
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self.host, self.port))
                server.listen(1)
                server.settimeout(0.2)
                self._ready.set()
                deadline = time.monotonic() + self.timeout
                conn = None
                while time.monotonic() < deadline and not self._stop.is_set():
                    try:
                        conn, self.client_address = server.accept()
                        break
                    except socket.timeout:
                        continue
                if conn is None:
                    return
                with conn:
                    conn.settimeout(1.0)
                    data = conn.recv(65535)
                    self.packets.extend(parse_ygopro_packets(data))
        except Exception as exc:
            if not self._stop.is_set():
                self.error = exc
        finally:
            self._done.set()


def parse_ygopro_packets(data: bytes) -> list[WindBotPacket]:
    packets: list[WindBotPacket] = []
    offset = 0
    while offset + 3 <= len(data):
        length = int.from_bytes(data[offset:offset + 2], "little")
        end = offset + 2 + length
        if length < 1 or end > len(data):
            break
        proto = data[offset + 2]
        packets.append(WindBotPacket(proto=proto, name=CTOS_NAMES.get(proto, f"UNKNOWN_{proto}"), payload_size=length - 1))
        offset = end
    return packets


def run_connection_smoke(config: WindBotConfig) -> dict:
    validate_config(config, check_port=True)
    with WindBotSmokeHost(config.host, config.port, config.timeout) as host:
        with WindBotProcess(config) as proc:
            if config.server_mode:
                proc.add_bot()
            result = host.wait_for_connection()
            result["pid"] = proc.process.pid if proc.process else None
            result["stdout"] = str(proc.stdout_path) if proc.stdout_path else None
            result["stderr"] = str(proc.stderr_path) if proc.stderr_path else None
            result["server_mode"] = config.server_mode
            return result


def write_metadata(config: WindBotConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(config.metadata(), f, indent=2, sort_keys=True)
        f.write("\n")
