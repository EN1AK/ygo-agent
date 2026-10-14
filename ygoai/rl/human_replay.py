"""Teacher-only inverse mapping of audited YRP2 responses to native actions."""


def card_spec(card, player):
    zones = {1: '', 2: 'h', 4: 'm', 8: 's', 16: 'g', 32: 'r', 64: 'x'}
    return ('o' if card['player'] != player else '') + zones[card['location']] + str(card['sequence'] + 1)


def response_action(snapshot, prompt, response):
    if snapshot['message'] != prompt['op'] or snapshot['player'] != prompt['player']:
        raise ValueError('actor/core prompt mismatch')
    menu, op = snapshot['menu'], prompt['op']
    data = bytes.fromhex(response)
    value = int.from_bytes(data, 'little', signed=True)

    def find(predicate):
        rows = [i for i, row in enumerate(menu) if predicate(row)]
        if len(rows) != 1:
            raise ValueError(f'ambiguous/missing response mapping: op={op}, response={response}, menu={menu}')
        return rows[0]

    if op == 11:
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
    elif op in (15, 23):
        if value == -1:
            return find(lambda a: a['act'] == 'Cancel')
        if len(data) != data[0] + 1:
            raise ValueError('invalid recorded selection length')
        must = len(prompt.get('must', []))
        desired = [card_spec(prompt['cards'][i], prompt['player']) for i in sorted(data[1 + must:])]
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
    raise ValueError(f'unsupported response mapping {op}')
