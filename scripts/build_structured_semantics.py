"""Build deterministic Structured-lite card semantics from pinned local assets.

The output is a raw row-major byte table so identical inputs produce identical
bytes on every platform. Row zero is permanently reserved for unknown cards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import _repo_bootstrap  # noqa: F401

from ygoai.rl.observation_schema import EFFECT_TAGS, STATIC_SEMANTIC_FIELDS


TYPE_MONSTER = 0x1
TYPE_SPELL = 0x2
TYPE_TRAP = 0x4
TYPE_NORMAL = 0x10
TYPE_EFFECT = 0x20
TYPE_FUSION = 0x40
TYPE_RITUAL = 0x80
TYPE_SYNCHRO = 0x2000
TYPE_XYZ = 0x800000
TYPE_PENDULUM = 0x1000000
TYPE_LINK = 0x4000000

EXACT = 3
FALLBACK = 2
UNKNOWN = 1

TAG_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "activate": (re.compile(r"EFFECT_TYPE_ACTIVATE"),),
    "destroy": (re.compile(r"CATEGORY_DESTROY|Duel\.Destroy\s*\("),),
    "negate": (re.compile(r"CATEGORY_NEGATE|Duel\.Negate(?:Activation|Effect)\s*\("),),
    "send_to_grave": (re.compile(r"CATEGORY_TOGRAVE|Duel\.SendtoGrave\s*\("),),
    "banish": (re.compile(r"CATEGORY_REMOVE|Duel\.Remove\s*\("),),
    "draw": (re.compile(r"CATEGORY_DRAW|Duel\.Draw\s*\("),),
    "search_or_add": (re.compile(r"CATEGORY_SEARCH|Duel\.Search\s*\(|Duel\.SendtoHand\s*\("),),
    "summon": (re.compile(r"CATEGORY_SUMMON|Duel\.Summon\s*\("),),
    "special_summon": (re.compile(r"CATEGORY_SPECIAL_SUMMON|Duel\.SpecialSummon\s*\("),),
    "target": (re.compile(r"SetTarget\s*\(|Duel\.SelectTarget\s*\("),),
    "change_atk_def": (re.compile(r"UPDATE_ATTACK|SET_ATTACK|UPDATE_DEFENSE|SET_DEFENSE"),),
    "move": (re.compile(r"Duel\.MoveToField\s*\(|Duel\.MoveSequence\s*\("),),
    "cost_release": (re.compile(r"Duel\.Release\s*\("),),
    "cost_discard": (re.compile(r"REASON_DISCARD"),),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def script_tree_hash(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    if root.exists():
        for path in sorted(root.glob("c*.lua"), key=lambda p: p.name):
            digest.update(path.name.encode("ascii", errors="strict"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
            count += 1
    return digest.hexdigest(), count


def read_code_list(path: Path) -> list[int]:
    codes: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            code = int(line.split()[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"invalid code list line {line_number}: {line!r}") from exc
        codes.append(code)
    if len(codes) != len(set(codes)):
        raise ValueError("code list contains duplicate card codes")
    return codes


def u16_bytes(value: int) -> tuple[int, int]:
    value = max(0, min(65535, value))
    return value >> 8, value & 0xFF


def u32_bytes(value: int) -> tuple[int, int, int, int]:
    value &= 0xFFFFFFFF
    return value >> 24, (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF


def static_row(record: sqlite3.Row | None) -> bytes:
    row = bytearray(len(STATIC_SEMANTIC_FIELDS))
    if record is None:
        return bytes(row)
    card_type = int(record["type"])
    attack = int(record["atk"])
    defense_raw = int(record["def"])
    packed_level = int(record["level"])
    level = packed_level & 0xFF
    monster = bool(card_type & TYPE_MONSTER)
    xyz = bool(card_type & TYPE_XYZ)
    link = bool(card_type & TYPE_LINK)
    pendulum = bool(card_type & TYPE_PENDULUM)
    row[0] = 1
    row[1:5] = bytes(u32_bytes(card_type))
    row[5:9] = bytes(u32_bytes(int(record["race"])))
    row[9] = int(record["attribute"]) & 0xFF
    row[10] = int(monster and attack >= 0)
    row[11:13] = bytes(u16_bytes(attack if attack >= 0 else 0))
    row[13] = int(monster and not link and defense_raw >= 0)
    row[14:16] = bytes(u16_bytes(defense_raw if row[13] else 0))
    row[16] = int(monster and not xyz and not link)
    row[17] = level if row[16] else 0
    row[18] = int(xyz)
    row[19] = level if xyz else 0
    row[20] = int(link)
    row[21] = level if link else 0
    row[22] = int(pendulum)
    row[23] = (packed_level >> 24) & 0xFF if pendulum else 0
    row[24] = int(pendulum)
    row[25] = (packed_level >> 16) & 0xFF if pendulum else 0
    row[26] = int(link)
    row[27:31] = bytes(u32_bytes(defense_raw if link else 0))
    return bytes(row)


def tag_rows(script: str | None, card_type: int | None) -> tuple[bytes, bytes]:
    values = bytearray(len(EFFECT_TAGS))
    confidence = bytearray(len(EFFECT_TAGS))
    if script is None:
        values[0] = 1
        confidence[0] = UNKNOWN
        return bytes(values), bytes(confidence)
    for name, patterns in TAG_PATTERNS.items():
        if any(pattern.search(script) for pattern in patterns):
            index = EFFECT_TAGS.index(name)
            values[index] = 1
            confidence[index] = EXACT
    if card_type is not None and card_type & 0x20000:
        index = EFFECT_TAGS.index("continuous")
        values[index] = 1
        confidence[index] = EXACT
    if not any(values[1:]):
        values[0] = 1
        confidence[0] = FALLBACK
    return bytes(values), bytes(confidence)


def build(args: argparse.Namespace) -> dict[str, object]:
    codes = read_code_list(args.code_list)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.cards_db)
    connection.row_factory = sqlite3.Row
    query = connection.cursor()
    semantic = bytearray(len(STATIC_SEMANTIC_FIELDS))
    tags = bytearray(len(EFFECT_TAGS))
    confidences = bytearray(len(EFFECT_TAGS))
    missing_db = missing_script = 0
    tag_counts = {name: 0 for name in EFFECT_TAGS}
    for code in codes:
        record = query.execute(
            "SELECT id, type, atk, def, level, race, attribute FROM datas WHERE id = ?", (code,)
        ).fetchone()
        if record is None:
            missing_db += 1
        semantic.extend(static_row(record))
        script_path = args.scripts / f"c{code}.lua"
        script = script_path.read_text(encoding="utf-8", errors="replace") if script_path.exists() else None
        if script is None:
            missing_script += 1
        tag_row, confidence_row = tag_rows(script, int(record["type"]) if record else None)
        tags.extend(tag_row)
        confidences.extend(confidence_row)
        for i, value in enumerate(tag_row):
            tag_counts[EFFECT_TAGS[i]] += int(bool(value))
    connection.close()

    outputs = {
        "static": (args.output_dir / "card-semantics.u8", bytes(semantic)),
        "tags": (args.output_dir / "effect-tags.u8", bytes(tags)),
        "confidence": (args.output_dir / "effect-tag-confidence.u8", bytes(confidences)),
    }
    table_hashes: dict[str, str] = {}
    for name, (path, payload) in outputs.items():
        path.write_bytes(payload)
        table_hashes[name] = hashlib.sha256(payload).hexdigest()
    scripts_hash, script_count = script_tree_hash(args.scripts)
    metadata: dict[str, object] = {
        "format": "structured-lite-semantics-v1",
        "rows": len(codes) + 1,
        "unknown_row": 0,
        "static_width": len(STATIC_SEMANTIC_FIELDS),
        "effect_tag_width": len(EFFECT_TAGS),
        "static_fields": list(STATIC_SEMANTIC_FIELDS),
        "effect_tags": list(EFFECT_TAGS),
        "source_hashes": {
            "code_list_sha256": sha256_file(args.code_list),
            "cards_db_sha256": sha256_file(args.cards_db),
            "scripts_sha256": scripts_hash,
            "config_sha256": hashlib.sha256(json.dumps({
                "effect_tags": EFFECT_TAGS,
                "patterns": {key: [p.pattern for p in value] for key, value in TAG_PATTERNS.items()},
            }, sort_keys=True).encode()).hexdigest(),
        },
        "table_hashes": table_hashes,
        "coverage": {
            "code_list_cards": len(codes), "database_missing": missing_db,
            "script_files": script_count, "scripts_missing_for_code_list": missing_script,
            "tag_counts": tag_counts,
        },
    }
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    (args.output_dir / "metadata.json").write_bytes(metadata_bytes)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards-db", type=Path, default=Path("assets/locale/zh/cards.cdb"))
    parser.add_argument("--code-list", type=Path, default=Path("scripts/code_list.txt"))
    parser.add_argument("--scripts", type=Path, default=Path("scripts/script"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
