#!/usr/bin/env python3
"""Build and verify a source-derived protocol contract for legacy ygopro-core."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


MESSAGE_RE = re.compile(r"^\s*#define\s+(MSG_[A-Z0-9_]+)\s+([^\s/]+)", re.MULTILINE)
EMIT_RE = re.compile(r"write_buffer8\(\s*(MSG_[A-Z0-9_]+)\s*\)\s*;")
WRITE_RE = re.compile(r"write_buffer(8|16|32)\((.*?)\)\s*;", re.DOTALL)
FUNCTION_RE = re.compile(
    r"^[\w:<>,~*&\s]+\s+([\w:~]+)\s*\([^;{}]*\)\s*(?:const\s*)?\{\s*$",
    re.MULTILINE,
)


POLICY_INTERACTIVE = {
    "MSG_SELECT_BATTLECMD",
    "MSG_SELECT_IDLECMD",
    "MSG_SELECT_EFFECTYN",
    "MSG_SELECT_YESNO",
    "MSG_SELECT_OPTION",
    "MSG_SELECT_CARD",
    "MSG_SELECT_CHAIN",
    "MSG_SELECT_PLACE",
    "MSG_SELECT_POSITION",
    "MSG_SELECT_TRIBUTE",
    "MSG_SELECT_SUM",
    "MSG_SELECT_DISFIELD",
    "MSG_SELECT_UNSELECT_CARD",
    "MSG_ANNOUNCE_RACE",
    "MSG_ANNOUNCE_ATTRIB",
    "MSG_ANNOUNCE_CARD",
    "MSG_ANNOUNCE_NUMBER",
}

AUTOMATIC_INTERACTIVE = {
    "MSG_SELECT_COUNTER",
    "MSG_SORT_CARD",
    "MSG_ROCK_PAPER_SCISSORS",
}

KNOWN_NOTIFICATIONS = {
    "MSG_HINT", "MSG_CONFIRM_DECKTOP", "MSG_CONFIRM_CARDS", "MSG_SHUFFLE_DECK",
    "MSG_SHUFFLE_HAND", "MSG_SWAP_GRAVE_DECK", "MSG_SHUFFLE_SET_CARD",
    "MSG_REVERSE_DECK", "MSG_DECK_TOP", "MSG_SHUFFLE_EXTRA", "MSG_NEW_TURN",
    "MSG_NEW_PHASE", "MSG_CONFIRM_EXTRATOP", "MSG_MOVE", "MSG_POS_CHANGE",
    "MSG_SET", "MSG_SWAP", "MSG_FIELD_DISABLED", "MSG_SUMMONING", "MSG_SUMMONED",
    "MSG_SPSUMMONING", "MSG_SPSUMMONED", "MSG_FLIPSUMMONING",
    "MSG_FLIPSUMMONED", "MSG_CHAINING", "MSG_CHAINED", "MSG_CHAIN_SOLVING",
    "MSG_CHAIN_SOLVED", "MSG_CHAIN_END", "MSG_CHAIN_NEGATED",
    "MSG_CHAIN_DISABLED", "MSG_RANDOM_SELECTED", "MSG_BECOME_TARGET", "MSG_DRAW",
    "MSG_DAMAGE", "MSG_RECOVER", "MSG_EQUIP", "MSG_LPUPDATE", "MSG_CARD_TARGET",
    "MSG_CANCEL_TARGET", "MSG_PAY_LPCOST", "MSG_ADD_COUNTER", "MSG_REMOVE_COUNTER",
    "MSG_ATTACK", "MSG_BATTLE", "MSG_ATTACK_DISABLED", "MSG_DAMAGE_STEP_START",
    "MSG_DAMAGE_STEP_END", "MSG_MISSED_EFFECT", "MSG_TOSS_COIN", "MSG_TOSS_DICE",
    "MSG_HAND_RES", "MSG_CARD_HINT", "MSG_TAG_SWAP", "MSG_RELOAD_FIELD",
    "MSG_AI_NAME", "MSG_SHOW_HINT", "MSG_PLAYER_HINT", "MSG_MATCH_KILL",
    "MSG_CUSTOM_MSG",
}

RESPONSE_FORMS: dict[str, dict[str, Any]] = {
    "MSG_SELECT_BATTLECMD": {"storage": "int32", "encoding": "low16=command,high16=index"},
    "MSG_SELECT_IDLECMD": {"storage": "int32", "encoding": "low16=command,high16=index"},
    "MSG_SELECT_EFFECTYN": {"storage": "int32", "encoding": "0_or_1"},
    "MSG_SELECT_YESNO": {"storage": "int32", "encoding": "0_or_1"},
    "MSG_SELECT_OPTION": {"storage": "int32", "encoding": "option_index"},
    "MSG_SELECT_CARD": {"storage": "bytes", "encoding": "count_then_unique_indices_or_int32_minus_one"},
    "MSG_SELECT_CHAIN": {"storage": "int32", "encoding": "chain_index_or_minus_one"},
    "MSG_SELECT_PLACE": {"storage": "bytes", "encoding": "count_triplets_player_location_sequence"},
    "MSG_SELECT_POSITION": {"storage": "int32", "encoding": "one_allowed_position_bit"},
    "MSG_SELECT_TRIBUTE": {"storage": "bytes", "encoding": "count_then_unique_indices_or_int32_minus_one"},
    "MSG_SELECT_COUNTER": {"storage": "uint16_array", "encoding": "allocation_per_candidate"},
    "MSG_SELECT_SUM": {"storage": "bytes", "encoding": "count_then_unique_optional_indices"},
    "MSG_SELECT_DISFIELD": {"storage": "bytes", "encoding": "count_triplets_player_location_sequence"},
    "MSG_SORT_CARD": {"storage": "bytes", "encoding": "permutation_or_first_byte_ff"},
    "MSG_SELECT_UNSELECT_CARD": {"storage": "bytes", "encoding": "one_index_or_int32_minus_one"},
    "MSG_ROCK_PAPER_SCISSORS": {"storage": "int32", "encoding": "choice_1_to_3"},
    "MSG_ANNOUNCE_RACE": {"storage": "int32", "encoding": "allowed_mask_with_exact_popcount"},
    "MSG_ANNOUNCE_ATTRIB": {"storage": "int32", "encoding": "allowed_mask_with_exact_popcount"},
    "MSG_ANNOUNCE_CARD": {"storage": "int32", "encoding": "declarable_card_code"},
    "MSG_ANNOUNCE_NUMBER": {"storage": "int32", "encoding": "option_index"},
}

PARAMETER_BRANCHES: dict[str, list[str]] = {
    "MSG_SELECT_BATTLECMD": ["activate", "attack", "to_main2", "to_end"],
    "MSG_SELECT_IDLECMD": ["summon", "special_summon", "reposition", "monster_set", "spell_trap_set", "activate", "to_battle", "to_end", "shuffle_hand"],
    "MSG_SELECT_EFFECTYN": ["yes", "no"],
    "MSG_SELECT_YESNO": ["yes", "no"],
    "MSG_SELECT_OPTION": ["one_or_many_options"],
    "MSG_SELECT_CARD": ["cancelable", "min_zero", "min_to_max", "max_zero_or_empty_short_circuit"],
    "MSG_SELECT_CHAIN": ["forced", "optional_cancel", "zero_or_many_chains"],
    "MSG_SELECT_PLACE": ["count_zero_sentinel", "count_one", "count_many", "both_players", "monster_spell_trap_pendulum_zones"],
    "MSG_SELECT_DISFIELD": ["count_zero_sentinel", "count_one", "count_many", "both_players", "monster_spell_trap_pendulum_zones"],
    "MSG_SELECT_POSITION": ["single_position_short_circuit", "composite_allowed_mask"],
    "MSG_SELECT_TRIBUTE": ["cancelable", "min_zero", "weighted_release_values", "max_capped_to_five"],
    "MSG_SELECT_COUNTER": ["zero_short_circuit", "one_or_many_cards", "arbitrary_exact_allocation"],
    "MSG_SELECT_SUM": ["exact_sum_mode", "sum_limit_mode", "mandatory_cards", "dual_value_cards", "min_max_count"],
    "MSG_SORT_CARD": ["permutation", "no_sort_ff"],
    "MSG_SELECT_UNSELECT_CARD": ["cancelable", "finishable", "select_and_unselect_pools"],
    "MSG_ANNOUNCE_RACE": ["exact_popcount", "available_mask_up_to_32_bits"],
    "MSG_ANNOUNCE_ATTRIB": ["exact_popcount", "available_mask"],
    "MSG_ANNOUNCE_CARD": ["rpn_filter_expression", "arbitrary_declarable_code"],
    "MSG_ANNOUNCE_NUMBER": ["arbitrary_uint32_options", "response_is_option_index"],
    "MSG_ROCK_PAPER_SCISSORS": ["choice_1_to_3", "repeat_on_draw"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source_file(path: Path) -> str:
    """Hash source text canonically so checkout line endings do not drift."""
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def parse_int(token: str) -> int:
    token = token.rstrip("uUlL")
    return int(token, 0)


def parse_messages(common_h: Path) -> list[dict[str, Any]]:
    text = common_h.read_text(encoding="utf-8", errors="replace")
    messages = []
    seen_values: dict[int, str] = {}
    for match in MESSAGE_RE.finditer(text):
        name, raw_value = match.groups()
        value = parse_int(raw_value)
        if value in seen_values:
            raise ValueError(f"duplicate MSG value {value}: {seen_values[value]} and {name}")
        seen_values[value] = name
        messages.append({"name": name, "value": value, "definition_line": text.count("\n", 0, match.start()) + 1})
    if not messages:
        raise ValueError(f"no MSG_* definitions found in {common_h}")
    return messages


def enclosing_function(text: str, offset: int) -> str | None:
    result = None
    for match in FUNCTION_RE.finditer(text, 0, offset):
        result = match.group(1)
    return result


def emission_payload(text: str, emission: re.Match[str], source_line: int) -> list[dict[str, Any]]:
    tail = text[emission.end() :]
    stops = [len(tail)]
    next_msg = EMIT_RE.search(tail)
    if next_msg:
        stops.append(next_msg.start())
    next_return = re.search(r"\breturn\s+(?:TRUE|FALSE)\s*;", tail)
    if next_return:
        stops.append(next_return.start())
    block = tail[: min(stops)]
    writes = []
    for write in WRITE_RE.finditer(block):
        expression = " ".join(write.group(2).split())
        writes.append(
            {
                "width_bits": int(write.group(1)),
                "expression": expression,
                "source_line": source_line + block.count("\n", 0, write.start()),
            }
        )
    return writes


def scan_writers(core_source: Path) -> dict[str, list[dict[str, Any]]]:
    writers: dict[str, list[dict[str, Any]]] = {}
    for source in sorted(core_source.rglob("*.cpp")):
        text = source.read_text(encoding="utf-8", errors="replace")
        for emission in EMIT_RE.finditer(text):
            name = emission.group(1)
            line = text.count("\n", 0, emission.start()) + 1
            writers.setdefault(name, []).append(
                {
                    "source": source.relative_to(core_source).as_posix(),
                    "line": line,
                    "function": enclosing_function(text, emission.start()),
                    "payload_fields": emission_payload(text, emission, line),
                }
            )
    return writers


def adapter_message_names(adapter: Path) -> set[str]:
    text = adapter.read_text(encoding="utf-8", errors="replace")
    marker = "static std::string msg_to_string"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"msg_to_string not found in {adapter}")
    end = text.find("// system string", start)
    if end < 0:
        raise ValueError(f"msg_to_string end marker not found in {adapter}")
    return set(re.findall(r"case\s+(MSG_[A-Z0-9_]+)\s*:", text[start:end]))


def adapter_handler_message_names(adapter: Path) -> set[str]:
    """Return messages with a real production ``handle_message`` branch.

    A name in ``msg_to_string`` only proves that diagnostics can print it.  It
    does not prove the binary payload is consumed, which is the property needed
    to keep concatenated core messages aligned.
    """
    text = adapter.read_text(encoding="utf-8", errors="replace")
    marker = "void handle_message()"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"handle_message not found in {adapter}")
    end = text.find("void _damage(", start)
    if end < 0:
        raise ValueError(f"handle_message end marker not found in {adapter}")
    body = text[start:end]
    return set(re.findall(r"msg_\s*==\s*(MSG_[A-Z0-9_]+)", body))


def git_revision(core_source: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(core_source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def classify(name: str) -> str:
    if name in POLICY_INTERACTIVE:
        return "interactive_policy"
    if name in AUTOMATIC_INTERACTIVE:
        return "interactive_automatic"
    if name == "MSG_RETRY":
        return "response_retry"
    if name == "MSG_WIN":
        return "terminal"
    if name in KNOWN_NOTIFICATIONS:
        return "notification"
    raise ValueError(f"unclassified core message: {name}")


def portable_adapter_path(adapter: Path) -> str:
    parts = adapter.parts
    try:
        start = parts.index("ygoenv")
    except ValueError:
        return adapter.name
    return Path(*parts[start:]).as_posix()


def build_contract(core_source: Path, adapter: Path, expected_revision: str) -> dict[str, Any]:
    common_h = core_source / "common.h"
    playerop_cpp = core_source / "playerop.cpp"
    if not common_h.is_file() or not playerop_cpp.is_file():
        raise ValueError("core source must contain common.h and playerop.cpp")
    actual_revision = git_revision(core_source)
    if actual_revision != "unknown" and actual_revision != expected_revision:
        raise ValueError(f"core revision drift: expected {expected_revision}, got {actual_revision}")

    messages = parse_messages(common_h)
    writers = scan_writers(core_source)
    adapter_names = adapter_message_names(adapter)
    handler_names = adapter_handler_message_names(adapter)
    core_names = {item["name"] for item in messages}
    missing_adapter = sorted(core_names - adapter_names)
    if missing_adapter:
        raise ValueError(f"adapter msg_to_string misses core messages: {missing_adapter}")
    extra_adapter = sorted(adapter_names - core_names)
    if extra_adapter:
        raise ValueError(f"adapter msg_to_string has messages absent from core: {extra_adapter}")
    missing_handlers = sorted(
        name for name, records in writers.items()
        if records and name not in handler_names)
    if missing_handlers:
        raise ValueError(
            "adapter handle_message misses core-emitted messages: "
            f"{missing_handlers}")

    interactive = POLICY_INTERACTIVE | AUTOMATIC_INTERACTIVE
    missing_response = sorted(interactive - RESPONSE_FORMS.keys())
    if missing_response:
        raise ValueError(f"interactive messages lack response contracts: {missing_response}")
    missing_writers = sorted(name for name in interactive if not writers.get(name))
    if missing_writers:
        raise ValueError(f"interactive messages lack core writers: {missing_writers}")

    records = []
    for item in messages:
        name = item["name"]
        records.append(
            {
                **item,
                "class": classify(name),
                "writers": writers.get(name, []),
                "writer_status": "direct_core_writer" if writers.get(name) else "defined_without_direct_writer",
                "response": RESPONSE_FORMS.get(name),
                "parameter_branches": PARAMETER_BRANCHES.get(name, []),
                "adapter_declared": name in adapter_names,
                "adapter_handler_present": name in handler_names,
            }
        )

    contract: dict[str, Any] = {
        "schema_version": 1,
        "source_hash_normalization": "crlf-and-cr-to-lf",
        "core": {
            "repository": "https://github.com/Fluorohydride/ygopro-core.git",
            "expected_revision": expected_revision,
            "observed_revision": actual_revision,
            "common_h_sha256": sha256_source_file(common_h),
            "playerop_cpp_sha256": sha256_source_file(playerop_cpp),
        },
        "adapter": {
            "path": portable_adapter_path(adapter),
            "sha256": sha256_source_file(adapter),
        },
        "counts": {
            "messages": len(records),
            "interactive_policy": len(POLICY_INTERACTIVE),
            "interactive_automatic": len(AUTOMATIC_INTERACTIVE),
            "messages_with_core_writers": sum(bool(record["writers"]) for record in records),
            "core_emitted_messages_with_adapter_handlers": sum(
                bool(record["writers"]) and record["adapter_handler_present"]
                for record in records),
        },
        "messages": records,
    }
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    contract["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-source", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = build_contract(args.core_source.resolve(), args.adapter.resolve(), args.expected_revision)
    if args.source_bundle:
        source_bundle = args.source_bundle.resolve()
        contract["core"]["source_bundle"] = source_bundle.name
        contract["core"]["source_bundle_sha256"] = sha256_file(source_bundle)
        contract.pop("contract_sha256")
        canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        contract["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": contract["counts"], "contract_sha256": contract["contract_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
