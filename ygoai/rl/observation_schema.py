"""Versioned observation contracts for YGO Agent.

The native environment uses compact uint8 tensors.  This module is the
human- and machine-readable source of truth used by model/checkpoint tooling.
Generated manifests are deliberately small and are committed; generated card
semantic tables are not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Mapping


LEGACY_SCHEMA = "legacy-v2"
STRUCTURED_LITE_SCHEMA = "structured-lite-v1"
SUPPORTED_SCHEMAS = (LEGACY_SCHEMA, STRUCTURED_LITE_SCHEMA)

DEFAULT_MAX_CARDS = 80
DEFAULT_MAX_OPTIONS = 24
DEFAULT_HISTORY_ACTIONS = 32
DEFAULT_PUBLIC_EVENTS = 32
DEFAULT_GROUP_REFERENCES = 8


class Confidence(IntEnum):
    ABSENT = 0
    UNKNOWN = 1
    FALLBACK = 2
    EXACT = 3


class ReferenceRole(IntEnum):
    UNKNOWN = 0
    ACTION_SOURCE = 1
    ACTIVE_EFFECT_SOURCE = 2
    CANDIDATE = 3
    TARGET = 4
    COST = 5
    MATERIAL = 6
    TRIBUTE = 7
    SELECTED = 8


class PublicEvent(IntEnum):
    NONE = 0
    ACTIVATION = 1
    TARGET = 2
    NEGATION = 3
    DESTRUCTION = 4
    MOVEMENT = 5
    DRAW = 6
    SEARCH = 7
    SUMMON = 8
    SPECIAL_SUMMON = 9
    CHAIN_SOLVING = 10
    CHAIN_SOLVED = 11
    CHAIN_END = 12


class OverflowFlag(IntEnum):
    ACTIONS = 0
    ACTION_SINGLE_REFS = 1
    ACTION_COST_REFS = 2
    ACTION_MATERIAL_REFS = 3
    ACTION_TRIBUTE_REFS = 4
    ACTION_SELECTED_REFS = 5
    PUBLIC_EVENTS = 6
    INVALID_REFERENCE = 7


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[str | int, ...]
    dtype: str
    fields: tuple[str, ...] = ()
    range: str = ""
    padding: str = "zero"
    truncation: str = "none"


LEGACY_CARD_FIELDS = (
    "card_id_hi", "card_id_lo", "location", "sequence", "controller",
    "position", "overlay", "attribute", "race", "level", "counter",
    "disabled", "attack_hi", "attack_lo", "defense_hi", "defense_lo",
    "type_monster", "type_spell", "type_trap", "type_normal", "type_effect",
    "type_fusion", "type_ritual", "type_trapmonster", "type_spirit",
    "type_union", "type_dual", "type_tuner", "type_synchro", "type_token",
    "type_quickplay", "type_continuous", "type_equip", "type_field",
    "type_counter", "type_flip", "type_toon", "type_xyz", "type_pendulum",
    "type_spsummon", "type_link",
)

LEGACY_GLOBAL_FIELDS = (
    "self_lp_hi", "self_lp_lo", "opponent_lp_hi", "opponent_lp_lo", "turn",
    "phase", "self_went_first", "is_self_turn", "self_deck", "self_hand",
    "self_monster", "self_spell_trap", "self_grave", "self_banished",
    "self_extra", "opponent_deck", "opponent_hand", "opponent_monster",
    "opponent_spell_trap", "opponent_grave", "opponent_banished",
    "opponent_extra", "terminal_or_invalid",
)

LEGACY_ACTION_FIELDS = (
    "scene_card_index", "card_id_hi", "card_id_lo", "prompt_type", "act",
    "finish", "effect", "phase", "position", "number", "place", "attribute",
)

STATIC_SEMANTIC_FIELDS = (
    "known", "type_b0", "type_b1", "type_b2", "type_b3", "race_b0",
    "race_b1", "race_b2", "race_b3", "attribute", "attack_applicable",
    "attack_hi", "attack_lo", "defense_applicable", "defense_hi", "defense_lo",
    "level_applicable", "level", "rank_applicable", "rank", "link_applicable",
    "link", "left_scale_applicable", "left_scale", "right_scale_applicable",
    "right_scale", "link_marker_applicable", "link_marker_b0", "link_marker_b1",
    "link_marker_b2", "link_marker_b3", "reserved_0",
)

EFFECT_TAGS = (
    "unknown", "activate", "destroy", "negate", "send_to_grave", "banish",
    "draw", "search_or_add", "summon", "special_summon", "target",
    "change_atk_def", "move", "cost_release", "cost_discard", "continuous",
)

SELECTION_FIELDS = (
    "prompt_type", "chain_depth", "is_chain_response", "is_forced",
    "selection_role", "minimum", "maximum", "must_select", "selected_count",
    "finishable", "cancelable", "must_continue", "active_source_index",
    "active_source_confidence", "reserved_0", "reserved_1",
)

STRUCTURED_ACTION_FIELDS = (
    "prompt_type", "act", "finish", "effect", "phase", "position", "number",
    "place", "attribute", "source_semantic_row_hi", "source_semantic_row_lo",
    "source_confidence", "is_cancel", "is_continue", "reserved_0", "reserved_1",
)

PUBLIC_EVENT_FIELDS = (
    "event_type", "actor_relative", "chain_link", "location_from",
    "location_to", "position", "effect", "result", "turn_age", "phase",
    "card_id_hi", "card_id_lo",
)


def tensor_contract(
    schema: str,
    *,
    max_cards: int = DEFAULT_MAX_CARDS,
    max_options: int = DEFAULT_MAX_OPTIONS,
    history_actions: int = DEFAULT_HISTORY_ACTIONS,
    public_events: int = DEFAULT_PUBLIC_EVENTS,
    group_references: int = DEFAULT_GROUP_REFERENCES,
) -> dict[str, TensorSpec]:
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported observation schema: {schema!r}")
    common = {
        "cards_": TensorSpec((max_cards * 2, 41), "uint8", LEGACY_CARD_FIELDS),
        "global_": TensorSpec((23,), "uint8", LEGACY_GLOBAL_FIELDS),
        "actions_": TensorSpec((max_options, 12), "uint8", LEGACY_ACTION_FIELDS,
                               truncation="overflow flag required"),
        "h_actions_": TensorSpec((history_actions, 14), "uint8",
                                  LEGACY_ACTION_FIELDS + ("turn_age", "phase"),
                                  truncation="oldest-first ring truncation"),
        "mask_": TensorSpec((max_cards * 2, 14), "uint8", range="0..1"),
    }
    if schema == LEGACY_SCHEMA:
        return common
    return common | {
        "visible_card_ids_": TensorSpec((max_cards * 2, 2), "uint8",
                                           ("card_id_hi", "card_id_lo"),
                                           range="0 is hidden/padding; otherwise code-list row"),
        "card_semantics_": TensorSpec((max_cards * 2, len(STATIC_SEMANTIC_FIELDS)),
                                       "uint8", STATIC_SEMANTIC_FIELDS),
        "effect_tags_": TensorSpec((max_cards * 2, len(EFFECT_TAGS)), "uint8",
                                    EFFECT_TAGS, range="0..1"),
        "effect_tag_confidence_": TensorSpec((max_cards * 2, len(EFFECT_TAGS)),
                                               "uint8", EFFECT_TAGS,
                                               range="Confidence enum"),
        "selection_": TensorSpec((len(SELECTION_FIELDS),), "uint8", SELECTION_FIELDS),
        "action_features_": TensorSpec((max_options, len(STRUCTURED_ACTION_FIELDS)),
                                        "uint8", STRUCTURED_ACTION_FIELDS,
                                        truncation="diagnostics[ACTIONS] set"),
        "action_single_refs_": TensorSpec((max_options, 4, 3), "uint16",
                                           ("role", "scene_card_index", "confidence")),
        "action_group_refs_": TensorSpec((max_options, 4, group_references, 2),
                                          "uint16", ("scene_card_index", "confidence"),
                                          truncation="role-specific overflow flag set"),
        "action_group_mask_": TensorSpec((max_options, 4, group_references),
                                          "uint8", range="0..1"),
        "public_events_": TensorSpec((public_events, len(PUBLIC_EVENT_FIELDS)),
                                      "uint8", PUBLIC_EVENT_FIELDS,
                                      truncation="drop oldest; diagnostics[PUBLIC_EVENTS] set"),
        "public_event_refs_": TensorSpec((public_events, 4, 3), "uint16",
                                          ("role", "scene_card_index", "confidence")),
        "structured_diagnostics_": TensorSpec((len(OverflowFlag),), "uint8",
                                                tuple(x.name.lower() for x in OverflowFlag),
                                                range="0..255 saturating count"),
    }


def manifest(schema: str, **capacities: int) -> dict[str, Any]:
    tensors = tensor_contract(schema, **capacities)
    result: dict[str, Any] = {
        "observation_schema": schema,
        "semantics_version": "cdb-effect-tags-v1" if schema == STRUCTURED_LITE_SCHEMA else None,
        "partial_observation": True,
        "reference_indexing": "1-based visible scene-card index; 0 means absent",
        "confidence": {x.name.lower(): int(x) for x in Confidence},
        "reference_roles": {x.name.lower(): int(x) for x in ReferenceRole},
        "public_events": {x.name.lower(): int(x) for x in PublicEvent},
        "overflow_flags": {x.name.lower(): int(x) for x in OverflowFlag},
        "effect_tags": {name: i for i, name in enumerate(EFFECT_TAGS)},
        "tensors": {name: asdict(spec) for name, spec in tensors.items()},
    }
    validate_manifest(result)
    return result


def validate_manifest(data: Mapping[str, Any]) -> None:
    schema = data.get("observation_schema")
    expected = tensor_contract(str(schema))
    tensors = data.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != set(expected):
        raise ValueError("manifest tensor names do not match schema contract")
    for name, spec in expected.items():
        actual = tensors[name]
        if tuple(actual["shape"]) != spec.shape or actual["dtype"] != spec.dtype:
            raise ValueError(f"manifest tensor mismatch for {name}")
        fields = tuple(actual.get("fields", ()))
        if fields and len(fields) != spec.shape[-1]:
            raise ValueError(f"field count does not match final dimension for {name}")


def observation_nbytes(schema: str, **capacities: int) -> int:
    sizes = {"uint8": 1, "uint16": 2, "int32": 4, "float32": 4}
    total = 0
    for spec in tensor_contract(schema, **capacities).values():
        count = 1
        for dim in spec.shape:
            if not isinstance(dim, int):
                raise ValueError(f"unresolved dimension {dim!r}")
            count *= dim
        total += count * sizes[spec.dtype]
    return total
