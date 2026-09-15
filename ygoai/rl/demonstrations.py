"""Portable metadata layer for human YRP2 demonstrations.

Raw replays are deliberately kept outside Git.  This module turns them into a
small, versioned manifest while preserving the response bytes required by a
future engine-backed observation/action converter.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA = "ygo-human-demonstration-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _name(data: bytes, offset: int) -> str:
    raw = data[offset:offset + 40]
    return raw.decode("utf-16-le", errors="replace").split("\0", 1)[0]


def _deck(data: bytes, offset: int) -> tuple[list[int], int]:
    count = _u32(data, offset)
    offset += 4
    end = offset + count * 4
    if end > len(data):
        raise ValueError("truncated deck array")
    return list(struct.unpack_from(f"<{count}I", data, offset)), end


def _responses(data: bytes, offset: int) -> list[str]:
    result = []
    while offset < len(data):
        size = data[offset]
        offset += 1
        end = offset + size
        if end > len(data):
            raise ValueError("truncated replay response")
        result.append(data[offset:end].hex())
        offset = end
    return result


@dataclass(frozen=True)
class Yrp2:
    version: int
    flags: int
    start_time: int
    start_player: int
    initial_lp: int
    initial_hand: int
    draw_count: int
    duel_flags: int
    names: tuple[str, str]
    main_decks: tuple[list[int], list[int]]
    extra_decks: tuple[list[int], list[int]]
    responses: list[str]


def parse_yrp2(path: Path) -> Yrp2:
    raw = path.read_bytes()
    if len(raw) < 0x50 or raw[:4] != b"yrp2":
        raise ValueError(f"{path} is not a supported YRP2 replay")
    size = _u32(raw, 16)
    stream = raw[24:29] + struct.pack("<Q", size) + raw[0x50:]
    decoded = lzma.decompress(stream, format=lzma.FORMAT_ALONE)
    if len(decoded) != size:
        raise ValueError(f"decoded size mismatch: expected {size}, got {len(decoded)}")
    offset = 96
    main0, offset = _deck(decoded, offset)
    extra0, offset = _deck(decoded, offset)
    main1, offset = _deck(decoded, offset)
    extra1, offset = _deck(decoded, offset)
    return Yrp2(
        version=_u32(raw, 4), flags=_u32(raw, 8),
        start_time=_u32(raw, 20), start_player=_u32(raw, 0x40),
        initial_lp=_u32(decoded, 80), initial_hand=_u32(decoded, 84),
        draw_count=_u32(decoded, 88), duel_flags=_u32(decoded, 92),
        names=(_name(decoded, 0), _name(decoded, 40)),
        main_decks=(main0, main1), extra_decks=(extra0, extra1),
        responses=_responses(decoded, offset),
    )


def make_manifest_record(path: Path, kind: str, deck_tag: str | None = None) -> dict:
    replay = parse_yrp2(path)
    companion = path.with_name(path.name.replace("-Game1.yrp", ".yrp3d"))
    source_id = _sha256(path)
    decks = []
    for player in range(2):
        cards = replay.main_decks[player] + replay.extra_decks[player]
        deck_hash = hashlib.sha256(
            ",".join(map(str, sorted(cards))).encode("ascii")
        ).hexdigest()
        decks.append({"main": replay.main_decks[player],
                      "extra": replay.extra_decks[player], "sha256": deck_hash})
    return {
        "schema": SCHEMA,
        "demonstration_id": source_id[:20],
        "trajectory_kind": kind,
        "deck_tag": deck_tag,
        "source": {"yrp2": path.name, "yrp2_sha256": source_id,
                   "yrp3d": companion.name if companion.exists() else None,
                   "yrp3d_sha256": _sha256(companion) if companion.exists() else None},
        "engine": {"version": replay.version, "flags": replay.flags,
                   "duel_flags": replay.duel_flags},
        "players": list(replay.names), "start_player": replay.start_player,
        "initial_lp": replay.initial_lp, "initial_hand": replay.initial_hand,
        "draw_count": replay.draw_count, "decks": decks,
        "responses_hex": replay.responses,
        "supervision": {"policy": True, "value": kind == "full_duel",
                        "terminal_required": kind == "full_duel"},
        "conversion": {"status": "needs_engine_alignment",
                       "decision_schema": "ygo-human-decision-v1"},
    }


def write_manifest(records: Iterable[dict], output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
