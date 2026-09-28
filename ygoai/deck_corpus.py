"""Deterministic acquisition and canonicalization for YDK deck corpora."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


class CorpusError(ValueError):
    pass


TYPE_TOKEN = 0x4000
TYPE_EXTRA_DECK = 0x40 | 0x2000 | 0x800000 | 0x4000000


ARTIFACT_REQUIRED_FIELDS = {
    "sourceRegistry": ("schema_version", "acquired_at", "sources"),
    "canonicalDeck": ("canonical_id", "canonical_hash", "zones", "zone_sizes", "aliases", "family_id"),
    "validation": ("canonical_id", "status", "reasons"),
    "family": ("family_id", "members"),
    "feature": ("schema_version", "matrix_hash", "row_ids", "column_ids"),
    "cluster": ("cluster_id", "representative", "members", "metrics"),
    "split": ("revision", "train_families", "held_out_families"),
    "corpusManifest": (
        "revision", "source_registry_hash", "canonical_decks_hash",
        "validation_hash", "family_hash",
    ),
}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_fields(value: dict[str, Any], fields: Iterable[str], label: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise CorpusError(f"{label} missing required fields: {', '.join(missing)}")


def validate_artifact(kind: str, value: dict[str, Any]) -> None:
    if kind not in ARTIFACT_REQUIRED_FIELDS:
        raise CorpusError(f"unknown artifact kind: {kind}")
    require_fields(value, ARTIFACT_REQUIRED_FIELDS[kind], kind)
    if kind == "sourceRegistry":
        validate_registry(value)
    elif kind == "validation" and value["status"] not in {"ready", "limited", "excluded"}:
        raise CorpusError(f"invalid validation status: {value['status']}")


def validate_registry(registry: dict[str, Any]) -> None:
    require_fields(registry, ("schema_version", "acquired_at", "sources"), "registry")
    if registry["schema_version"] != 1 or not isinstance(registry["sources"], list):
        raise CorpusError("unsupported registry schema")
    seen: set[str] = set()
    for source in registry["sources"]:
        require_fields(
            source,
            ("id", "kind", "location", "revision", "license", "provenance", "include"),
            "source",
        )
        if source["id"] in seen:
            raise CorpusError(f"duplicate source id: {source['id']}")
        seen.add(source["id"])
        if source["kind"] not in {"local", "git", "archive"}:
            raise CorpusError(f"unsupported source kind: {source['kind']}")
        if not source["revision"]:
            raise CorpusError(f"source {source['id']} is not pinned")
        if source["kind"] == "archive" and len(source["revision"]) != 64:
            raise CorpusError(f"archive {source['id']} revision must be a SHA-256")


@dataclass(frozen=True)
class ParsedDeck:
    main: tuple[int, ...]
    extra: tuple[int, ...]
    side: tuple[int, ...]

    def zone_counts(self) -> dict[str, dict[str, int]]:
        return {
            name: {str(code): count for code, count in sorted(Counter(cards).items())}
            for name, cards in (("main", self.main), ("extra", self.extra), ("side", self.side))
        }

    def canonical_hash(self) -> str:
        return sha256_bytes(stable_json(self.zone_counts()).encode("utf-8"))


def parse_ydk_text(text: str, label: str = "<memory>") -> ParsedDeck:
    zones: dict[str, list[int]] = {"main": [], "extra": [], "side": []}
    section: str | None = None
    for lineno, raw in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered == "#main":
            section = "main"
            continue
        if lowered == "#extra":
            section = "extra"
            continue
        if lowered == "!side":
            section = "side"
            continue
        if line.startswith("#"):
            continue
        if section is None:
            raise CorpusError(f"{label}:{lineno}: card before a zone header")
        if not line.isascii() or not line.isdigit():
            raise CorpusError(f"{label}:{lineno}: invalid card code {line!r}")
        code = int(line)
        if code <= 0 or code > 999_999_999_999:
            raise CorpusError(f"{label}:{lineno}: card code out of range")
        zones[section].append(code)
    if not zones["main"]:
        raise CorpusError(f"{label}: main deck is empty")
    return ParsedDeck(tuple(zones["main"]), tuple(zones["extra"]), tuple(zones["side"]))


def parse_ydk(path: Path) -> ParsedDeck:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    return parse_ydk_text(text, str(path))


def read_code_list_metadata(path: Path) -> tuple[list[int], dict[int, int | None]]:
    """Read the frozen dense-card list without normalizing its line endings."""
    codes: list[int] = []
    has_script: dict[int, int | None] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        try:
            code = int(parts[0])
            script = int(parts[1]) if len(parts) > 1 else None
        except ValueError as exc:
            raise CorpusError(f"invalid code-list row {lineno}: {raw!r}") from exc
        if code <= 0 or code in has_script or script not in {None, 0, 1}:
            raise CorpusError(f"invalid code-list row {lineno}: {raw!r}")
        codes.append(code)
        has_script[code] = script
    return codes, has_script


def load_card_database(path: Path) -> dict[int, dict[str, int]]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT id, alias, type FROM datas").fetchall()
    return {
        int(code): {"alias": int(alias), "type": int(card_type)}
        for code, alias, card_type in rows
    }


def load_semantic_known_rows(asset_dir: Path, expected_rows: int) -> tuple[set[int], dict[str, Any]]:
    metadata_path = asset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = int(metadata.get("rows", -1))
    width = int(metadata.get("static_width", -1))
    table_path = asset_dir / "card-semantics.u8"
    if rows != expected_rows or width <= 0:
        raise CorpusError(
            f"semantic row contract mismatch: rows={rows}, expected={expected_rows}, width={width}"
        )
    table = table_path.read_bytes()
    if len(table) != rows * width:
        raise CorpusError(
            f"semantic table size mismatch: bytes={len(table)}, expected={rows * width}"
        )
    known = {index for index in range(1, rows) if table[index * width] != 0}
    return known, metadata


def _reason(code: str, severity: str, card_codes: Iterable[int], detail: str) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "card_codes": sorted(set(int(value) for value in card_codes)),
        "detail": detail,
    }


def validate_canonical_deck(
    deck: dict[str, Any],
    database: dict[int, dict[str, int]],
    code_positions: dict[int, int],
    code_scripts: dict[int, int | None],
    semantic_known_rows: set[int],
    script_dir: Path,
) -> dict[str, Any]:
    """Classify one canonical deck against the exact frozen runtime assets."""
    reasons: list[dict[str, Any]] = []
    sizes = deck["zone_sizes"]
    invalid_sizes = []
    if not 40 <= int(sizes.get("main", 0)) <= 60:
        invalid_sizes.append(f"main={sizes.get('main', 0)} (expected 40..60)")
    if not 0 <= int(sizes.get("extra", 0)) <= 15:
        invalid_sizes.append(f"extra={sizes.get('extra', 0)} (expected 0..15)")
    if not 0 <= int(sizes.get("side", 0)) <= 15:
        invalid_sizes.append(f"side={sizes.get('side', 0)} (expected 0..15)")
    if invalid_sizes:
        reasons.append(_reason("invalid_deck_size", "excluded", [], "; ".join(invalid_sizes)))

    all_codes = sorted({
        int(code)
        for zone in ("main", "extra", "side")
        for code in deck["zones"].get(zone, {})
    })
    missing_db = [code for code in all_codes if code not in database]
    if missing_db:
        reasons.append(_reason("missing_card_database_row", "excluded", missing_db,
                               "card code is absent from cards.cdb"))

    token_cards = [
        code for code in all_codes
        if code in database and database[code]["type"] & TYPE_TOKEN
    ]
    if token_cards:
        reasons.append(_reason("token_card_in_deck", "excluded", token_cards,
                               "token-only cards cannot be placed in a playable deck"))

    invalid_zone_cards = []
    for zone in ("main", "extra"):
        for raw_code in deck["zones"].get(zone, {}):
            code = int(raw_code)
            if code not in database:
                continue
            is_extra = bool(database[code]["type"] & TYPE_EXTRA_DECK)
            if (zone == "extra") != is_extra:
                invalid_zone_cards.append(code)
    if invalid_zone_cards:
        reasons.append(_reason("invalid_zone_card_type", "excluded", invalid_zone_cards,
                               "main/extra placement disagrees with the card database type"))

    missing_code_list = [code for code in all_codes if code not in code_positions]
    if missing_code_list:
        reasons.append(_reason("missing_code_list_entry", "limited", missing_code_list,
                               "card lacks a dense observation/code-list row"))

    missing_scripts = []
    for code in all_codes:
        if code not in code_positions:
            continue
        declared = code_scripts.get(code)
        if declared != 1:
            continue
        direct = script_dir / f"c{code}.lua"
        alias = database.get(code, {}).get("alias", 0)
        alias_path = script_dir / f"c{alias}.lua" if alias else None
        if not direct.is_file() and not (alias_path and alias_path.is_file()):
            missing_scripts.append(code)
    if missing_scripts:
        reasons.append(_reason("missing_required_script", "limited", missing_scripts,
                               "code-list requires a script but neither card nor alias script exists"))

    missing_semantics = [
        code for code in all_codes
        if code in code_positions and code_positions[code] not in semantic_known_rows
    ]
    if missing_semantics:
        reasons.append(_reason("missing_semantic_row", "limited", missing_semantics,
                               "card semantic row is absent or marked unknown"))

    status = "ready"
    if any(item["severity"] == "excluded" for item in reasons):
        status = "excluded"
    elif reasons:
        status = "limited"
    result = {
        "canonical_id": deck["canonical_id"],
        "status": status,
        "reasons": reasons,
        "zone_sizes": sizes,
        "card_count": len(all_codes),
    }
    validate_artifact("validation", result)
    return result


def validate_corpus_static(
    canonical_path: Path,
    cards_db: Path,
    code_list: Path,
    semantic_assets: Path,
    script_dir: Path,
) -> dict[str, Any]:
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    codes, code_scripts = read_code_list_metadata(code_list)
    code_positions = {code: index + 1 for index, code in enumerate(codes)}
    semantic_known, semantic_metadata = load_semantic_known_rows(
        semantic_assets, len(codes) + 1
    )
    database = load_card_database(cards_db)
    entries = [
        validate_canonical_deck(
            deck, database, code_positions, code_scripts, semantic_known, script_dir
        )
        for deck in canonical["decks"]
    ]
    status_counts = dict(sorted(Counter(item["status"] for item in entries).items()))
    reason_counts = dict(sorted(Counter(
        reason["code"] for item in entries for reason in item["reasons"]
    ).items()))
    return {
        "schema_version": 1,
        "validation_stage": "static-runtime-assets-v1",
        "inputs": {
            "canonical_decks_sha256": sha256_file(canonical_path),
            "cards_db_sha256": sha256_file(cards_db),
            "code_list_sha256": sha256_file(code_list),
            "semantic_metadata_sha256": sha256_file(semantic_assets / "metadata.json"),
            "semantic_table_sha256": sha256_file(semantic_assets / "card-semantics.u8"),
        },
        "semantic_format": semantic_metadata.get("format"),
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "decks": entries,
    }


def canonical_deck_text(deck: dict[str, Any]) -> str:
    lines = ["#created by ygo-agent deck-corpus-v1", "#main"]
    for zone, header in (("main", None), ("extra", "#extra"), ("side", "!side")):
        if header:
            lines.append(header)
        for code, count in sorted(
            ((int(code), int(count)) for code, count in deck["zones"].get(zone, {}).items())
        ):
            lines.extend([str(code)] * count)
    return "\n".join(lines) + "\n"


def materialize_canonical_decks(
    canonical: dict[str, Any], deck_ids: Iterable[str], output_dir: Path
) -> dict[str, str]:
    """Write stable engine-ready YDK names and return their content hashes."""
    wanted = set(deck_ids)
    by_id = {item["canonical_id"]: item for item in canonical["decks"]}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise CorpusError(f"unknown canonical deck ids: {missing[:8]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for deck_id in sorted(wanted):
        payload = canonical_deck_text(by_id[deck_id]).encode("ascii")
        path = output_dir / f"{deck_id}.ydk"
        if path.exists() and path.read_bytes() != payload:
            raise CorpusError(f"materialized deck drift: {path}")
        path.write_bytes(payload)
        hashes[deck_id] = sha256_bytes(payload)
    return hashes


def freeze_corpus_revision(
    canonical_path: Path,
    static_validation_path: Path,
    runtime_validation_path: Path,
    source_manifest_path: Path,
    family_path: Path,
    acquisition_rejected_path: Path,
    output_dir: Path,
    revision: str = "deck-corpus-v1",
) -> dict[str, Any]:
    if output_dir.exists():
        raise CorpusError(f"immutable corpus output already exists: {output_dir}")
    canonical_doc = json.loads(canonical_path.read_text(encoding="utf-8"))
    static_doc = json.loads(static_validation_path.read_text(encoding="utf-8"))
    runtime_doc = json.loads(runtime_validation_path.read_text(encoding="utf-8"))
    static_by_id = {item["canonical_id"]: item for item in static_doc["decks"]}
    runtime_by_id = runtime_doc["decks"]
    tables: dict[str, list[dict[str, Any]]] = {key: [] for key in ("ready", "limited", "excluded")}
    combined = []
    for deck in canonical_doc["decks"]:
        deck_id = deck["canonical_id"]
        static = static_by_id[deck_id]
        runtime = runtime_by_id.get(deck_id)
        status = static["status"]
        reasons = list(static["reasons"])
        if status == "ready":
            if runtime is None:
                status = "limited"
                reasons.append(_reason("runtime_validation_missing", "limited", [],
                                       "native smoke result is absent"))
            elif runtime["status"] != "passed":
                status = "excluded"
                reasons.append(_reason(
                    runtime.get("reason", "runtime_validation_failed"), "excluded", [],
                    runtime.get("error") or runtime.get("stderr_tail", "native smoke failed")[-500:],
                ))
        validation = {
            "canonical_id": deck_id, "status": status, "reasons": reasons,
            "static_status": static["status"],
            "runtime_status": runtime["status"] if runtime else "not_run",
        }
        combined.append(validation)
        tables[status].append({"deck": deck, "validation": validation})

    output_dir.mkdir(parents=True)
    table_hashes = {}
    for status, rows in tables.items():
        document = {"schema_version": 1, "revision": revision, "status": status, "decks": rows}
        path = output_dir / f"{status}.json"
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        table_hashes[status] = sha256_file(path)
    validation_doc = {"schema_version": 1, "revision": revision, "decks": combined}
    validation_path = output_dir / "validation.json"
    validation_path.write_text(json.dumps(validation_doc, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    sampling_dir = output_dir / "sampling-decks"
    ready_ids = [row["deck"]["canonical_id"] for row in tables["ready"]]
    deck_hashes = materialize_canonical_decks(canonical_doc, ready_ids, sampling_dir)
    sampling = {
        "schema_version": 1, "revision": revision, "eligible_status": "ready",
        "deck_count": len(ready_ids),
        "decks": [{"canonical_id": deck_id, "ydk_sha256": deck_hashes[deck_id],
                   "path": f"sampling-decks/{deck_id}.ydk"} for deck_id in ready_ids],
    }
    sampling_path = output_dir / "sampling-input.json"
    sampling_path.write_text(json.dumps(sampling, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    rejected_sources = json.loads(acquisition_rejected_path.read_text(encoding="utf-8"))
    rejected_sources_path = output_dir / "acquisition-rejected-sources.json"
    rejected_sources_path.write_text(
        json.dumps(rejected_sources, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1, "revision": revision, "state": "eligibility-frozen",
        "source_registry_hash": sha256_file(source_manifest_path),
        "canonical_decks_hash": sha256_file(canonical_path),
        "validation_hash": sha256_file(validation_path),
        "family_hash": sha256_file(family_path),
        "static_validation_hash": sha256_file(static_validation_path),
        "runtime_validation_hash": sha256_file(runtime_validation_path),
        "sampling_input_hash": sha256_file(sampling_path),
        "table_hashes": table_hashes,
        "counts": {status: len(rows) for status, rows in tables.items()},
        "acquisition_rejected_source_count": len(rejected_sources.get("decks", [])),
    }
    validate_artifact("corpusManifest", manifest)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return manifest


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _safe_extract_zip(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            resolved = (target / info.filename).resolve()
            if target.resolve() not in resolved.parents and resolved != target.resolve():
                raise CorpusError(f"unsafe archive member: {info.filename}")
        bundle.extractall(target)


def _safe_extract_tar(archive: Path, target: Path) -> None:
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            resolved = (target / member.name).resolve()
            if target.resolve() not in resolved.parents and resolved != target.resolve():
                raise CorpusError(f"unsafe archive member: {member.name}")
        bundle.extractall(target)


def materialize_source(source: dict[str, Any], project_root: Path, cache_dir: Path) -> Path:
    source_dir = cache_dir / source["id"]
    if source["kind"] == "local":
        path = (project_root / source["location"]).resolve()
        if not path.is_dir():
            raise CorpusError(f"local source does not exist: {path}")
        return path
    if source["kind"] == "git":
        if not source_dir.exists():
            _run_git(["clone", "--no-checkout", source["location"], str(source_dir)])
            _run_git(["checkout", "--detach", source["revision"]], source_dir)
        head = _run_git(["rev-parse", "HEAD"], source_dir)
        if head != source["revision"]:
            raise CorpusError(
                f"git source {source['id']} is at {head}, expected {source['revision']}"
            )
        remote = _run_git(["remote", "get-url", "origin"], source_dir)
        expected_remote = str(source["location"]).rstrip("/").removesuffix(".git").casefold()
        actual_remote = remote.rstrip("/").removesuffix(".git").casefold()
        if actual_remote != expected_remote:
            raise CorpusError(
                f"git source {source['id']} remote {remote} != {source['location']}"
            )
        return source_dir
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / "source.archive"
    if not archive_path.exists():
        location = source["location"]
        local = (project_root / location).resolve()
        if local.is_file():
            shutil.copy2(local, archive_path)
        else:
            urllib.request.urlretrieve(location, archive_path)
    actual = sha256_file(archive_path)
    if actual != source["revision"]:
        raise CorpusError(f"archive {source['id']} hash {actual} != {source['revision']}")
    extracted = source_dir / "extracted"
    if not extracted.exists():
        extracted.mkdir()
        if zipfile.is_zipfile(archive_path):
            _safe_extract_zip(archive_path, extracted)
        elif tarfile.is_tarfile(archive_path):
            _safe_extract_tar(archive_path, extracted)
        else:
            raise CorpusError(f"unsupported archive: {archive_path}")
    return extracted


def _selected_files(root: Path, include: list[str], exclude: list[str]) -> list[Path]:
    selected = []
    for path in root.rglob("*.ydk"):
        relative = path.relative_to(root).as_posix()
        if include and not any(fnmatch.fnmatch(relative, item) for item in include):
            continue
        if any(fnmatch.fnmatch(relative, item) for item in exclude):
            continue
        selected.append(path)
    return sorted(selected, key=lambda item: item.relative_to(root).as_posix().casefold())


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def weighted_jaccard(left: dict[str, dict[str, int]], right: dict[str, dict[str, int]]) -> float:
    numerator = denominator = 0.0
    for zone, weight in (("main", 1.0), ("extra", 1.5)):
        keys = set(left[zone]) | set(right[zone])
        for key in keys:
            a, b = left[zone].get(key, 0), right[zone].get(key, 0)
            numerator += weight * min(a, b)
            denominator += weight * max(a, b)
    return numerator / denominator if denominator else 0.0


def assign_families(canonical: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    ids = [item["canonical_id"] for item in canonical]
    union = UnionFind(ids)
    for index, left in enumerate(canonical):
        for right in canonical[index + 1 :]:
            if weighted_jaccard(left["zones"], right["zones"]) >= threshold:
                union.union(left["canonical_id"], right["canonical_id"])
    groups: dict[str, list[str]] = defaultdict(list)
    for value in ids:
        groups[union.find(value)].append(value)
    families = []
    for members in sorted((sorted(group) for group in groups.values()), key=lambda x: x[0]):
        family_hash = sha256_bytes("\n".join(members).encode("ascii"))
        families.append({"family_id": f"family-{family_hash[:16]}", "members": members})
    return families


def build_corpus(
    registry_path: Path,
    project_root: Path,
    output_dir: Path,
    cache_dir: Path,
    near_duplicate_threshold: float = 0.90,
) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_registry(registry)
    if output_dir.exists():
        raise CorpusError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()

    source_entries = []
    parsed_entries = []
    rejected_entries = []
    for source in sorted(registry["sources"], key=lambda item: item["id"]):
        root = materialize_source(source, project_root, cache_dir)
        files = _selected_files(root, source["include"], source.get("exclude", []))
        manifest_files = []
        for path in files:
            relative = path.relative_to(root).as_posix()
            destination = raw_dir / source["id"] / PurePosixPath(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            file_hash = sha256_file(destination)
            manifest_files.append({"path": relative, "sha256": file_hash})
            try:
                parsed = parse_ydk(destination)
                zones = parsed.zone_counts()
                parsed_entries.append(
                    {
                        "source_id": source["id"],
                        "source_path": relative,
                        "source_sha256": file_hash,
                        "name": path.stem,
                        "zones": zones,
                        "zone_sizes": {name: sum(values.values()) for name, values in zones.items()},
                        "canonical_hash": parsed.canonical_hash(),
                    }
                )
            except CorpusError as exc:
                rejected_entries.append(
                    {
                        "source_id": source["id"],
                        "source_path": relative,
                        "source_sha256": file_hash,
                        "reason": "parse_error",
                        "detail": str(exc),
                    }
                )
        source_entries.append(
            {
                **source,
                "file_count": len(manifest_files),
                "files": manifest_files,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in parsed_entries:
        grouped[item["canonical_hash"]].append(item)
    canonical = []
    for digest, aliases in sorted(grouped.items()):
        first = sorted(aliases, key=lambda x: (x["source_id"], x["source_path"]))[0]
        canonical.append(
            {
                "canonical_id": f"deck-{digest[:16]}",
                "canonical_hash": digest,
                "zones": first["zones"],
                "zone_sizes": first["zone_sizes"],
                "aliases": [
                    {
                        key: item[key]
                        for key in ("source_id", "source_path", "source_sha256", "name")
                    }
                    for item in sorted(aliases, key=lambda x: (x["source_id"], x["source_path"]))
                ],
            }
        )
    families = assign_families(canonical, near_duplicate_threshold)
    family_by_deck = {
        deck: family["family_id"] for family in families for deck in family["members"]
    }
    for item in canonical:
        item["family_id"] = family_by_deck[item["canonical_id"]]

    source_manifest = {
        "schema_version": 1,
        "acquired_at": registry["acquired_at"],
        "sources": source_entries,
    }
    report = {
        "schema_version": 1,
        "registry_sha256": sha256_file(registry_path),
        "source_manifest_sha256": sha256_bytes(stable_json(source_manifest).encode("utf-8")),
        "source_count": len(source_entries),
        "raw_file_count": sum(item["file_count"] for item in source_entries),
        "parsed_file_count": len(parsed_entries),
        "rejected_file_count": len(rejected_entries),
        "canonical_deck_count": len(canonical),
        "exact_duplicate_count": len(parsed_entries) - len(canonical),
        "family_count": len(families),
        "near_duplicate_threshold": near_duplicate_threshold,
    }
    artifacts = {
        "source-manifest.json": source_manifest,
        "canonical-decks.json": {"schema_version": 1, "decks": canonical},
        "families.json": {
            "schema_version": 1,
            "similarity": "zone-weighted-jaccard-v1",
            "threshold": near_duplicate_threshold,
            "families": families,
        },
        "rejected.json": {"schema_version": 1, "decks": rejected_entries},
        "report.json": report,
    }
    for name, value in artifacts.items():
        (output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report
