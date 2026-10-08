"""Three development families, real card scripts and bounded reference policies.

Initial fields are synthetic. Every action after setup is an engine response.
Goals inspect settled results, not equality to the reference response sequence.
"""
HALQ, BULB, SWORD = 50588353, 67441435, 85289965
HAYATE, KAGARI, ENGAGE = 8491308, 63288573, 63166095
VEILER, OGRE, BLUE = 97268402, 59438930, 89631139
FADER, MAXXC, FILLER = 19665973, 23434538, 46986414

CASES = {
    'combo': {'family': 'halq-bulb-sword', 'goal': 'two_completed_sword_attacks',
              'branches': {'reference': True, 'effect_after_first_attack': True, 'revive_in_defense': False}},
    'battle': {'family': 'hayate-direct-lethal', 'goal': 'win_this_battle_phase',
               'branches': {'direct': True, 'attack_monster': False}},
    'battle-negated': {'family': 'hayate-direct-lethal', 'goal': 'survive_this_battle_phase',
                       'branches': {'end_battle': True, 'attack_monster': False}},
    'interaction': {'family': 'kagari-recovery', 'goal': 'engage_stays_in_grave_after_chain',
                    'branches': {'veiler': True, 'ogre': False, 'pass': False}},
}


def fixture(case):
    if case not in CASES:
        raise ValueError('unknown fixture')
    lp = (1000, 1500) if case.startswith('battle') else (8000, 8000)
    cards = []

    def add(code, player, location, sequence=0, position=1):
        cards.append((code, player, location, sequence, position))

    for p in (0, 1):
        for _ in range(3):
            add(FILLER, p, 1, position=8)
    if case == 'combo':
        for seq, code in enumerate((VEILER, MAXXC, FADER)):
            add(code, 0, 4, seq)
        add(BULB, 0, 1, position=8)
        add(HALQ, 0, 64, position=8)
        add(SWORD, 0, 64, position=8)
        for seq in (0, 1):
            add(BLUE, 1, 4, seq, 4)
    elif case.startswith('battle'):
        add(HAYATE, 0, 4, 5)
        add(BLUE, 1, 4)
        if case == 'battle-negated':
            add(VEILER, 1, 2)
    else:
        add(HAYATE, 0, 4, 5)
        add(KAGARI, 0, 64, position=8)
        add(ENGAGE, 0, 16)
        add(VEILER, 1, 2)
        add(OGRE, 1, 2)
    lines = ['Debug.ReloadFieldBegin(DUEL_ATTACK_FIRST_TURN)']
    lines += [f'Debug.SetPlayerInfo({p},{lp[p]},0,0)' for p in (0, 1)]
    lines += [f'Debug.AddCard({c},{p},{p},{loc},{seq},{pos})' for c, p, loc, seq, pos in cards]
    lines.append('Debug.ReloadFieldEnd()')
    return '\n'.join(lines), lp


def index_of(cards, code):
    return next((i for i, c in enumerate(cards) if c['code'] == code), None)


def completed_attacks(events, attacker_locations=None):
    # Pair each attack with its damage-step end; declarations alone don't score.
    active, count = None, 0
    for m in events:
        if m['op'] == 110:
            active = bytes.fromhex(m['raw'])[1:5]
            if attacker_locations is not None and tuple(active[:3]) not in attacker_locations:
                active = None
        elif m['op'] == 112:
            active = None
        elif m['op'] == 114 and active is not None:
            count += 1
            active = None
    return count


class ReferencePolicy:
    def __init__(self, case, branch):
        if branch not in CASES[case]['branches']:
            raise ValueError('unknown branch')
        self.case, self.branch = case, branch
        self.summoning = None
        self.root_event = None
        self.chain_ended = False
        self.used_response = False
        self.last_attack_card = None

    def choose(self, core, m):
        op, case, branch = m['op'], self.case, self.branch
        attacks = completed_attacks(core.events)
        if op in (10, 11):
            if op == 11:
                for code in ((HALQ, SWORD) if case == 'combo' else ((KAGARI,) if case == 'interaction' else ())):
                    i = index_of(m['special'], code)
                    if i is not None:
                        self.summoning = code
                        return (i << 16) | 1
                if case == 'combo':
                    for code in (BULB, SWORD):
                        i = index_of(m['activate'], code)
                        if i is not None and not (code == SWORD and branch == 'effect_after_first_attack' and attacks == 0):
                            return (i << 16) | 5
                if m['battle']:
                    return 6
                raise ValueError('reference reached unexpected idle boundary')
            if self.root_event is None and case.startswith('battle'):
                self.root_event = len(core.events) - 1
            if case == 'battle-negated' and branch == 'end_battle':
                return 2 if m['main2'] else 3
            if case == 'combo':
                i = index_of(m['activate'], SWORD)
                if i is not None and (branch != 'effect_after_first_attack' or attacks >= 1):
                    return i << 16
            attacker = SWORD if case == 'combo' else HAYATE
            i = index_of(m['attack'], attacker)
            if i is not None:
                self.last_attack_card = attacker
                return (i << 16) | 1
            return 2 if m['main2'] else 3
        if op == 12:
            if case == 'combo' and m['code'] == HALQ and self.root_event is None:
                self.root_event = len(core.events) - 1
            return int(m['code'] in (HALQ, KAGARI))
        if op == 13:
            # Description 31 = direct attack despite opposing monsters.
            if m['description'] == 31:
                return int(branch == 'direct')
            return 1
        if op == 16:
            if case == 'interaction' and m['player'] == 1 and not self.used_response:
                if any(e['op'] == 70 and int.from_bytes(bytes.fromhex(e['raw'])[1:5], 'little') == KAGARI for e in core.events):
                    if self.root_event is None:
                        self.root_event = len(core.events) - 1
                    desired = VEILER if branch == 'veiler' else OGRE if branch == 'ogre' else None
                    i = index_of(m['cards'], desired)
                    if i is not None:
                        self.used_response = True
                        return i
            if case == 'battle-negated' and m['player'] == 1 and core.phase == 4:
                i = index_of(m['cards'], VEILER)
                if i is not None:
                    return i
            if case == 'combo':
                for code in (HALQ, SWORD):
                    i = index_of(m['cards'], code)
                    if i is not None and (code != SWORD or branch != 'effect_after_first_attack' or attacks >= 1):
                        if code == HALQ and self.root_event is None:
                            self.root_event = len(core.events) - 1
                        return i
            return 0 if m['forced'] else -1
        if op in (15, 26):
            # Engine determines valid material sets; prefer preserving Fader for
            # the Sword summon, and choose Bulb for the position-change target.
            preferred = ([VEILER, MAXXC] if self.summoning == HALQ else [])
            preferred += [BULB, HALQ, FADER, ENGAGE, KAGARI, HAYATE, BLUE]
            order = sorted(range(len(m['cards'])), key=lambda i: preferred.index(m['cards'][i]['code'])
                           if m['cards'][i]['code'] in preferred else len(preferred))
            if op == 26:
                if m['finish']:
                    return -1
                if not order:
                    raise ValueError('no selectable material')
                return bytes([1, order[0]])
            count = m['min']
            return bytes([count] + order[:count])
        if op == 18:
            choices = []
            for relative in (0, 1):
                for loc, shift in ((4, 0), (8, 8)):
                    for seq in range(7 if loc == 4 else 8):
                        bit = relative * 16 + shift + seq
                        if not (m['mask'] >> bit) & 1:
                            choices.append((m['player'] ^ relative, loc, seq))
            if len(choices) < m['count']:
                raise ValueError('insufficient zones')
            return bytes(x for c in choices[:m['count']] for x in c)
        if op == 19:
            desired = 4 if branch == 'revive_in_defense' else 1
            return desired if m['positions'] & desired else next(p for p in (1, 4, 2, 8) if m['positions'] & p)
        if op == 14:
            return 0
        raise ValueError(f'unhandled reference prompt {op}')


def finished(core, policy, prompt):
    if core.winner is not None:
        return True
    if policy.case == 'interaction':
        return policy.root_event is not None and any(m['op'] == 74 for m in core.events[policy.root_event:])
    return core.phase in (0x100, 0x200) or (policy.case == 'combo' and completed_attacks(core.events) >= 2 and prompt is not None)


def grade(case, state, events, *, stable):
    if not stable:
        return 'unknown'
    if case == 'combo':
        sword = [c for c in state['cards'] if c['code'] == SWORD and c['player'] == 0 and c['location'] == 4]
        # Only Sword attacks are commanded by the starter reference. Check raw
        # attack sources against its actual zone too, so another card can't score.
        allowed = {(c['player'], c['location'], c['sequence']) for c in sword}
        ok = bool(sword) and completed_attacks(events, allowed) >= 2
    elif case == 'battle':
        ok = state['winner'] == 0
    elif case == 'battle-negated':
        ok = state['lp'][0] > 0 and state['winner'] != 1 and state['phase'] in (0x100, 0x200)
    elif case == 'interaction':
        ok = any(c['code'] == ENGAGE and c['player'] == 0 and c['location'] == 16 for c in state['cards'])
    else:
        raise ValueError('unknown goal')
    return 'success' if ok else 'verified_failure'
