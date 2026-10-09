"""Conservative, static effect-description features; never executes card Lua.

Only unconditional initial_effect registrations with literal Stringid bindings
are accepted. Features describe declarations, not successful resolution or
remaining legal uses. Missing/ambiguous bindings are all-zero; the existing
whole-card representation remains available to the model.
"""
from __future__ import annotations

import re
import hashlib
import json
from pathlib import Path

VERSION = "effect-description-declarations-v1"
SLOTS = 16
FIELDS = (
    "bound", "destroy", "negate", "to_grave", "banish", "draw", "to_hand",
    "search", "summon", "special_summon", "damage", "recover", "disable_effect",
    "activate", "ignition", "quick", "trigger", "continuous", "single",
    "range_hand", "range_monster", "range_spell", "range_grave", "range_banished",
    "range_deck", "range_extra", "has_condition", "has_cost", "has_target",
    "has_operation", "declared_count_limit", "targets_card",
)
WIDTH = len(FIELDS)


def validate_assets(directory):
    """Reject damaged or unversioned effect assets before a v2 run."""
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    description = metadata.get('effect_descriptions', {})
    if (description.get('version'), description.get('slots'), description.get('width')) != (VERSION, SLOTS, WIDTH):
        raise ValueError('incompatible effect-description asset contract')
    for key, filename in {'static':'card-semantics.u8', 'tags':'effect-tags.u8',
                          'confidence':'effect-tag-confidence.u8', 'effect_descriptions':'effect-descriptions.u8'}.items():
        data = (directory / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != metadata['table_hashes'][key]:
            raise ValueError('semantic asset digest mismatch: ' + filename)
        if key == 'effect_descriptions':
            if len(data) != metadata['rows'] * SLOTS * WIDTH or any(data[:SLOTS * WIDTH]):
                raise ValueError('invalid effect-description rows or unknown row')
            if any(value > 1 for value in data):
                raise ValueError('effect declarations must be binary')
    return metadata


_FUNCTION = re.compile(r"(?m)^\s*function\s+([\w]+)\.initial_effect\s*\(c\)")
# Remove strings as well as comments, so fake code inside either cannot bind.
_LITERALS = re.compile(r"--\[(=*)\[.*?\]\1\]|--[^\n]*|\[(=*)\[.*?\]\2\]|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", re.S)
_CREATE = re.compile(r"(?:local\s+)?(\w+)\s*=\s*(?:Effect\.CreateEffect\(c\)|(\w+):Clone\(\))")
_SET = re.compile(r"(\w+):Set(\w+)\((.*)\)")
_REGISTER = re.compile(r"c:RegisterEffect\((\w+)\)")


def parse_effects(script: str, code: int) -> tuple[bytes, dict]:
    clean = _LITERALS.sub(lambda m: "\n" * m.group().count("\n"), script)
    starts = list(_FUNCTION.finditer(clean))
    blank = bytes(SLOTS * WIDTH)
    if len(starts) != 1:
        return blank, {"bound": 0, "reason": "missing_or_multiple_initial_effect"}
    start = starts[0]
    namespace = start.group(1)
    tail = clean[start.end():]
    # Unconditional straight-line registration only. Reject control flow and
    # anonymous functions rather than guessing execution order or scope.
    end = re.search(r"\bend\b", tail)
    if end is None:
        return blank, {"bound": 0, "reason": "missing_end"}
    body = tail[:end.start()]
    if re.search(r"\b(if|for|while|repeat|function|return|goto|do)\b", body):
        return blank, {"bound": 0, "reason": "dynamic_initial_effect"}
    effects, registered = {}, []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        create = _CREATE.fullmatch(line)
        setter = _SET.fullmatch(line)
        registration = _REGISTER.fullmatch(line)
        if create:
            name, parent = create.groups()
            if name in effects or (parent and parent not in effects):
                return blank, {"bound": 0, "reason": "ambiguous_effect_variable"}
            effects[name] = dict(effects[parent]) if parent else {}
        elif setter and setter[1] in effects:
            effects[setter[1]][setter[2]] = setter[3].strip()
        elif registration and registration[1] in effects:
            registered.append(dict(effects[registration[1]]))
        elif any(re.search(r"\b" + re.escape(name) + r"\b", line) for name in effects):
            return blank, {"bound": 0, "reason": "unsupported_effect_mutation"}
    by_slot = {}
    for effect in registered:
        desc = re.fullmatch(r"aux.Stringid\(\s*(id|\d+)\s*,\s*(\d+)\s*\)", effect.get("Description", ""))
        if not desc:
            continue
        owner = code if desc[1] == "id" and namespace == "s" and re.search(r"\blocal\s+s\s*,\s*id\s*=\s*GetID\(\)", clean[:start.start()]) else desc[1]
        if str(owner) != str(code) or not 0 <= int(desc[2]) < SLOTS:
            continue
        slot = int(desc[2])
        by_slot.setdefault(slot, []).append(effect)
    output = bytearray(blank)
    accepted = []
    for slot, definitions in sorted(by_slot.items()):
        # A description shared by multiple registrations is not a unique effect.
        if len(definitions) != 1:
            continue
        e = definitions[0]
        def has(field, *tokens):
            expression = e.get(field, "")
            # Constants may be combined, but calls, arithmetic and variables
            # are not evaluated. A zero means no positively identified token.
            if not re.fullmatch(r"[A-Z_0-9\s+|()]*", expression):
                return 0
            return int(any(re.search(r"\b" + token + r"\b", expression) for token in tokens))
        row = [1]
        row += [has("Category", "CATEGORY_" + token) for token in (
            "DESTROY", "NEGATE", "TOGRAVE", "REMOVE", "DRAW", "TOHAND", "SEARCH",
            "SUMMON", "SPECIAL_SUMMON", "DAMAGE", "RECOVER", "DISABLE")]
        row += [has("Type", *["EFFECT_TYPE_" + x for x in tokens]) for tokens in (
            ("ACTIVATE",), ("IGNITION",), ("QUICK_O", "QUICK_F"),
            ("TRIGGER_O", "TRIGGER_F"), ("CONTINUOUS",), ("SINGLE",))]
        row += [has("Range", "LOCATION_" + token) for token in (
            "HAND", "MZONE", "SZONE", "GRAVE", "REMOVED", "DECK", "EXTRA")]
        row += [int(field in e) for field in ("Condition", "Cost", "Target", "Operation", "CountLimit")]
        row += [has("Property", "EFFECT_FLAG_CARD_TARGET")]
        output[slot * WIDTH:(slot + 1) * WIDTH] = bytes(row)
        accepted.append(slot)
    return bytes(output), {"bound": len(accepted), "slots": accepted,
                           "ambiguous_slots": [k for k, v in by_slot.items() if len(v) > 1]}
