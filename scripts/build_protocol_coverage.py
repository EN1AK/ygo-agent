#!/usr/bin/env python3
"""Build a fail-closed coverage report for the pinned interactive protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EVIDENCE = {
    "MSG_SELECT_BATTLECMD": [
        "test_command_response_preserves_command_and_full_uint16_index"],
    "MSG_SELECT_IDLECMD": [
        "test_command_response_preserves_command_and_full_uint16_index"],
    "MSG_SELECT_EFFECTYN": [
        "test_scalar_yes_no_option_chain_and_rps_responses"],
    "MSG_SELECT_YESNO": [
        "test_scalar_yes_no_option_chain_and_rps_responses"],
    "MSG_SELECT_OPTION": [
        "test_scalar_yes_no_option_chain_and_rps_responses"],
    "MSG_SELECT_CARD": ["test_optional_empty_and_index_responses"],
    "MSG_SELECT_CHAIN": [
        "test_scalar_yes_no_option_chain_and_rps_responses"],
    "MSG_SELECT_PLACE": [
        "test_multi_place_mapping_covers_both_players_and_zone_types"],
    "MSG_SELECT_POSITION": [
        "test_composite_position_mask_yields_only_atomic_positions"],
    "MSG_SELECT_TRIBUTE": [
        "test_tribute_uses_core_count_and_weight_rules"],
    "MSG_SELECT_COUNTER": [
        "test_counter_allocation_is_exact_and_capacity_bounded"],
    "MSG_SELECT_SUM": [
        "test_exact_sum_sequences_match_core_validator",
        "test_sum_limit_combinations_are_complete",
    ],
    "MSG_SELECT_DISFIELD": [
        "test_multi_place_mapping_covers_both_players_and_zone_types"],
    "MSG_SORT_CARD": ["test_sort_permutations_and_no_sort_sentinel"],
    "MSG_SELECT_UNSELECT_CARD": ["test_optional_empty_and_index_responses"],
    "MSG_ROCK_PAPER_SCISSORS": [
        "test_scalar_yes_no_option_chain_and_rps_responses"],
    "MSG_ANNOUNCE_RACE": [
        "test_race_and_attribute_masks_preserve_exact_popcount"],
    "MSG_ANNOUNCE_ATTRIB": [
        "test_race_and_attribute_masks_preserve_exact_popcount"],
    "MSG_ANNOUNCE_CARD": [
        "test_announce_card_rpn_filters",
        "test_large_announce_card_is_staged_without_losing_any_code",
    ],
    "MSG_ANNOUNCE_NUMBER": [
        "test_announced_number_preserves_full_uint32_option_value",
        "test_scalar_yes_no_option_chain_and_rps_responses",
    ],
}

# A helper-predicate assertion is not an adapter/core integration test.  Only
# list branches whose request was emitted by the pinned core, parsed by the
# production adapter, encoded by its callback, and accepted by that same core
# instance.  Keep this map explicit so a new branch cannot inherit coverage
# just because another test happens to mention its message name.
CORE_ADAPTER_EVIDENCE = {
    "MSG_SELECT_BATTLECMD": {
        branch: "test_battle_and_idle_commands_through_adapter_and_same_core"
        for branch in ("activate", "attack", "to_main2", "to_end")
    },
    "MSG_SELECT_IDLECMD": {
        branch: "test_battle_and_idle_commands_through_adapter_and_same_core"
        for branch in (
            "summon", "special_summon", "reposition", "monster_set",
            "spell_trap_set", "activate", "to_battle", "to_end",
            "shuffle_hand")
    },
    "MSG_SELECT_CHAIN": {
        branch: "test_chain_forced_optional_and_zero_through_adapter_and_same_core"
        for branch in ("forced", "optional_cancel", "zero_or_many_chains")
    },
    "MSG_ANNOUNCE_CARD": {
        branch: "test_announce_card_rpn_and_arbitrary_code_through_same_core"
        for branch in ("rpn_filter_expression", "arbitrary_declarable_code")
    },
    "MSG_SELECT_EFFECTYN": {
        branch: "test_yesno_effectyn_and_option_frames_through_adapter_and_core"
        for branch in ("yes", "no")
    },
    "MSG_SELECT_YESNO": {
        branch: "test_yesno_effectyn_and_option_frames_through_adapter_and_core"
        for branch in ("yes", "no")
    },
    "MSG_SELECT_OPTION": {
        "one_or_many_options":
            "test_yesno_effectyn_and_option_frames_through_adapter_and_core",
    },
    "MSG_SELECT_CARD": {
        branch: "test_select_card_optional_and_cancel_through_adapter_and_core"
        for branch in ("cancelable", "min_zero", "min_to_max")
    },
    "MSG_SELECT_PLACE": {
        branch: "test_place_and_disfield_frames_through_adapter_and_core"
        for branch in (
            "count_zero_sentinel", "count_one", "count_many", "both_players",
            "monster_spell_trap_pendulum_zones")
    },
    "MSG_SELECT_DISFIELD": {
        branch: "test_place_and_disfield_frames_through_adapter_and_core"
        for branch in (
            "count_zero_sentinel", "count_one", "count_many", "both_players",
            "monster_spell_trap_pendulum_zones")
    },
    "MSG_SELECT_TRIBUTE": {
        "weighted_release_values": "test_weighted_frames_through_actual_adapter_and_same_core_instance",
        "cancelable": "test_tribute_cancel_zero_min_and_core_max_cap",
        "min_zero": "test_tribute_cancel_zero_min_and_core_max_cap",
        "max_capped_to_five": "test_tribute_cancel_zero_min_and_core_max_cap",
    },
    "MSG_SELECT_COUNTER": {
        "one_or_many_cards": "test_counter_frame_through_actual_adapter_and_same_core_instance",
        "arbitrary_exact_allocation": "test_counter_frame_through_actual_adapter_and_same_core_instance",
    },
    "MSG_SELECT_SUM": {
        branch: "test_weighted_frames_through_actual_adapter_and_same_core_instance"
        for branch in (
            "exact_sum_mode", "sum_limit_mode", "mandatory_cards",
            "dual_value_cards", "min_max_count")
    },
    "MSG_SORT_CARD": {
        "permutation": "test_sort_frame_through_actual_adapter_and_same_core_instance",
        "no_sort_ff": "test_sort_frame_through_actual_adapter_and_same_core_instance",
    },
    "MSG_SELECT_UNSELECT_CARD": {
        branch: "test_select_unselect_frame_through_actual_adapter_and_same_core"
        for branch in ("cancelable", "finishable", "select_and_unselect_pools")
    },
    "MSG_ROCK_PAPER_SCISSORS": {
        branch: "test_rps_frames_through_actual_adapter_and_same_core"
        for branch in ("choice_1_to_3", "repeat_on_draw")
    },
    "MSG_ANNOUNCE_RACE": {
        branch: "test_mask_frames_through_actual_adapter_and_same_core"
        for branch in ("exact_popcount", "available_mask_up_to_32_bits")
    },
    "MSG_ANNOUNCE_ATTRIB": {
        branch: "test_mask_frames_through_actual_adapter_and_same_core"
        for branch in ("exact_popcount", "available_mask")
    },
    "MSG_SELECT_POSITION": {
        "composite_allowed_mask": "test_announced_numbers_and_position_through_adapter_and_core",
    },
    "MSG_ANNOUNCE_NUMBER": {
        branch: "test_announced_numbers_and_position_through_adapter_and_core"
        for branch in ("arbitrary_uint32_options", "response_is_option_index")
    },
}

CORE_AUTOMATIC_EVIDENCE = {
    ("MSG_SELECT_CARD", "max_zero_or_empty_short_circuit"):
        "test_core_automatic_short_circuits_emit_no_adapter_frame",
    ("MSG_SELECT_POSITION", "single_position_short_circuit"):
        "test_core_automatic_short_circuits_emit_no_adapter_frame",
    ("MSG_SELECT_COUNTER", "zero_short_circuit"):
        "test_core_automatic_short_circuits_emit_no_adapter_frame",
}

NOTIFICATION_BOUNDARY_TEST = (
    "test_core_emitted_notifications_parse_exactly_and_preserve_boundaries")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--boundary-test", type=Path, required=True)
    parser.add_argument("--runtime-smoke", type=Path, required=True)
    parser.add_argument("--structured-environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    adapter = args.adapter.read_text(encoding="utf-8")
    boundary_test = args.boundary_test.read_text(encoding="utf-8")
    runtime_smoke = json.loads(args.runtime_smoke.read_text(encoding="utf-8"))
    structured = json.loads(
        args.structured_environment.read_text(encoding="utf-8"))

    interactive = [
        message for message in contract["messages"]
        if message["class"].startswith("interactive_")
    ]
    expected = {message["name"] for message in interactive}
    if expected != set(EVIDENCE):
        missing = sorted(expected - set(EVIDENCE))
        stale = sorted(set(EVIDENCE) - expected)
        raise ValueError(
            f"interactive evidence map drift: missing={missing}, stale={stale}")

    records = []
    uncovered = []
    for message in interactive:
        name = message["name"]
        if name not in adapter:
            raise ValueError(f"adapter has no handler token for {name}")
        tests = EVIDENCE[name]
        absent = [test for test in tests if f"def {test}(" not in boundary_test]
        if absent:
            raise ValueError(f"missing boundary tests for {name}: {absent}")
        branches = message.get("parameter_branches") or []
        if not branches:
            uncovered.append({"message": name, "branch": None})
        for branch in branches:
            core_test = CORE_ADAPTER_EVIDENCE.get(name, {}).get(branch)
            automatic_test = CORE_AUTOMATIC_EVIDENCE.get((name, branch))
            if core_test and f"def {core_test}(" not in boundary_test:
                raise ValueError(
                    f"missing adapter/core fixture for {name}/{branch}: {core_test}")
            if automatic_test and f"def {automatic_test}(" not in boundary_test:
                raise ValueError(
                    f"missing automatic core fixture for {name}/{branch}: {automatic_test}")
            if not core_test and not automatic_test:
                uncovered.append({"message": name, "branch": branch})
            records.append({
                "message": name,
                "class": message["class"],
                "branch": branch,
                "adapter_handler_present": True,
                "core_response_form": message["response"],
                "boundary_tests": tests,
                "adapter_core_test": core_test,
                "automatic_core_test": automatic_test,
                "covered": bool(core_test or automatic_test),
            })

    notification_fixture_present = (
        f"def {NOTIFICATION_BOUNDARY_TEST}(" in boundary_test)
    notifications = []
    for message in contract["messages"]:
        if message["class"] != "notification" or not message.get("writers"):
            continue
        name = message["name"]
        handler_present = bool(message.get("adapter_handler_present"))
        covered = handler_present and notification_fixture_present
        if not covered:
            uncovered.append({"message": name, "branch": "notification_payload"})
        notifications.append({
            "message": name,
            "class": message["class"],
            "writer_count": len(message["writers"]),
            "adapter_handler_present": handler_present,
            "boundary_test": (
                NOTIFICATION_BOUNDARY_TEST
                if notification_fixture_present else None),
            "covered": covered,
        })

    report = {
        "schema_version": 1,
        "contract_sha256": contract["contract_sha256"],
        "inputs": {
            "contract": {"path": str(args.contract), "sha256": sha256(args.contract)},
            "adapter": {"path": str(args.adapter), "sha256": sha256(args.adapter)},
            "boundary_test": {
                "path": str(args.boundary_test), "sha256": sha256(args.boundary_test)},
            "runtime_smoke": {
                "path": str(args.runtime_smoke), "sha256": sha256(args.runtime_smoke)},
            "structured_environment": {
                "path": str(args.structured_environment),
                "sha256": sha256(args.structured_environment),
            },
        },
        "runtime_evidence": {
            "deck_corpus_smoke": runtime_smoke,
            "structured_environment": structured,
        },
        "summary": {
            "interactive_messages": len(interactive),
            "core_emitted_notifications": len(notifications),
            "normal_interactive_branches": len(records),
            "covered_interactive_branches": sum(
                bool(record["covered"]) for record in records),
            "covered_notification_messages": sum(
                bool(record["covered"]) for record in notifications),
            "covered_branches": (
                sum(bool(record["covered"]) for record in records) +
                sum(bool(record["covered"]) for record in notifications)),
            "uncovered_branches": len(uncovered),
        },
        "uncovered": uncovered,
        "branches": records,
        "notifications": notifications,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    if uncovered:
        raise ValueError(
            f"{len(uncovered)} normal interactive branches lack adapter/core fixtures")


if __name__ == "__main__":
    main()
