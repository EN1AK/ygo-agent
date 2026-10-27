"""Teacher-only inverse mapping of audited YRP2 responses to native actions."""

import itertools


def comparable_game_events(events):
    """Normalize versioned display packets without moving game mutations.

    16/31 have separate legacy/modern wire layouts. Card-hint notifications
    may use unordered iteration; sort only the adjacent notification block.
    Message 2 is a sticky UI hint, and repeated identical hints are idempotent.
    Keep its deduplicated value order as a separate equality requirement.
    """
    game, hints = [], []
    for event in events:
        if event['op'] in (2, 16, 31):
            continue
        if event['op'] == 160:
            hints.append(event['raw'])
            continue
        if hints:
            game.append(tuple(sorted(hints)))
            hints = []
        game.append(event['raw'])
    if hints:
        game.append(tuple(sorted(hints)))
    display = [value for value, _ in itertools.groupby(
        event['raw'] for event in events if event['op'] == 2)]
    return game, display


def canonical_response(prompt, response):
    """Remove only documented zero padding after a counted card selection.

    The core consumes the count and that many byte indices. Preserve the
    source bytes separately; this is semantic comparison, not byte equality.
    """
    data = bytes.fromhex(response)
    if prompt['op'] in (15, 20, 23) and data != b'\xff' * 4 and data:
        end = data[0] + 1
        if len(data) < end or any(data[end:]):
            raise ValueError('invalid recorded selection length or padding')
        data = data[:end]
    return data.hex()


def card_spec(card, player):
    zones = {1: '', 2: 'h', 4: 'm', 8: 's', 16: 'g', 32: 'r', 64: 'x'}
    location = card['location']
    spec = ('o' if card['player'] != player else '') + zones[location & 0x7f] + str(card['sequence'] + 1)
    if location & 0x80:
        position = card.get('position', card.get('extra'))
        if position is None or not 0 <= position < 26:
            raise ValueError('missing or unsupported overlay material index')
        spec += chr(ord('a') + position)
    return spec


def response_action(snapshot, prompt, response):
    if snapshot['message'] != prompt['op'] or snapshot['player'] != prompt['player']:
        raise ValueError('actor/core prompt mismatch')
    menu, op = snapshot['menu'], prompt['op']
    data = bytes.fromhex(canonical_response(prompt, response))
    value = int.from_bytes(data, 'little', signed=True)

    def find(predicate):
        rows = [i for i, row in enumerate(menu) if predicate(row)]
        if len(rows) != 1:
            raise ValueError(f'ambiguous/missing response mapping: op={op}, response={response}, menu={menu}')
        return rows[0]

    if op == 10:
        kind, index = value & 65535, value >> 16
        if kind in (0, 1):
            acts = ('Activate',) if kind == 0 else ('Attack', 'DirectAttack')
            rows = [i for i, row in enumerate(menu) if row['act'] in acts]
            if not 0 <= index < len(rows):
                raise ValueError('battle response index outside legal menu')
            return rows[index]
        if kind in (2, 3) and index == 0:
            return find(lambda a: a['phase'] == ('Main2' if kind == 2 else 'End'))
        raise ValueError('invalid battle command response')
    elif op == 11:
        kind, index = value & 65535, value >> 16
        if kind < 6:
            act = ('Summon', 'SpSummon', 'Repo', 'MSet', 'Set', 'Activate')[kind]
            rows = [i for i, row in enumerate(menu) if row['act'] == act]
            return rows[index]
        if kind in (6, 7):
            return find(lambda a: a['phase'] == ('Battle' if kind == 6 else 'End'))
        if kind == 8:
            return find(lambda a: a['finish'])
    elif op in (12, 13):
        if value not in (0, 1):
            raise ValueError('invalid yes/no response')
        return find(lambda a: (a['act'] == 'Activate') if value == 1 else (a['act'] == 'Cancel'))
    elif op == 14:
        return value
    elif op == 16:
        return find(lambda a: a['act'] == 'Cancel') if value == -1 else value
    elif op == 19:
        return find(lambda a: a['position'] == value)
    elif op in (18, 24):
        if len(data) != 3:
            raise ValueError('multi-place source requires a separate staged mapping')
        player, location, sequence = data
        place = (15 if player != prompt['player'] else 0) + (1 if location == 4 else 8) + sequence
        return find(lambda a: a['place'] == place)
    elif op in (15, 20, 23):
        if value == -1:
            return find(lambda a: a['act'] == 'Cancel')
        if len(data) != data[0] + 1:
            raise ValueError('invalid recorded selection length')
        must = len(prompt.get('must', []))
        indices = data[1 + must:]
        if op == 23 or (op == 20 and (prompt['min'] != prompt['max'] or
                                      prompt['min'] == 0 or
                                      any(c['extra'] != 1 for c in prompt['cards']))):
            indices = sorted(indices)
        desired = [card_spec(prompt['cards'][i], prompt['player']) for i in indices]
        # Weighted selection exposes only choices with a legal completion and
        # advances in canonical index order. Ordinary selection removes picks.
        for spec in desired:
            matches = [i for i, a in enumerate(menu) if a['spec'] == spec and not a['finish']]
            if len(matches) == 1:
                return matches[0]
        return find(lambda a: a['finish'])
    elif op == 26:
        if value == -1:
            return find(lambda a: a['finish'] or a['act'] == 'Cancel')
        if len(data) != 2 or data[0] != 1:
            raise ValueError('invalid unselect response')
        return find(lambda a: not a['finish'] and a['act'] != 'Cancel' and a['response'] == data[1])
    elif op == 25:
        if data == b'\xff':
            return find(lambda a: a['finish'])
        count = len(prompt['cards'])
        if len(data) != count or sorted(data) != list(range(count)):
            raise ValueError('invalid recorded sort permutation')
        for index in data:
            spec = card_spec(prompt['cards'][index], prompt['player'])
            matches = [i for i, a in enumerate(menu) if a['spec'] == spec and not a['finish']]
            if len(matches) == 1:
                return matches[0]
        raise ValueError('sort response has no remaining legal card')
    elif op in (140, 141):
        if len(data) != 4:
            raise ValueError('invalid announced mask response')
        key = 'race' if op == 140 else 'attribute'
        return find(lambda a: a.get(key) == int.from_bytes(data, 'little'))
    elif op == 143:
        if len(data) != 4 or not 0 <= value < len(menu):
            raise ValueError('invalid announced number index')
        return value
    elif op == 142:
        if len(data) != 4:
            raise ValueError('invalid announced card response')
        code = int.from_bytes(data, 'little')
        return find(lambda a: code in a['announce_codes'] if 'announce_codes' in a
                    else a['response'] == code)
    raise ValueError(f'unsupported response mapping {op}')
