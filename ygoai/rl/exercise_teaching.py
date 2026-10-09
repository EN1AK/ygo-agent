"""Normal-opening development exercises; fixed deals, unmodified core rules."""
from collections import Counter
import sqlite3

from ygoai.rl.exercise_actor import ActorReference
from ygoai.rl.exercise_starters import BULB, HALQ, SWORD, VEILER, BLUE, HAYATE, KAGARI, ENGAGE, OGRE, completed_attacks

ANCIENT, CYBER, QUICKDRAW = 10667321, 70095154, 20932152
RAYE = 26077387


def opening(database, code_list, variant=0, kind='combo'):
    codes = set(map(int, code_list.read_text().split()))
    with sqlite3.connect(database) as db:
        filler = [r[0] for r in db.execute(
            'select id from datas where type=17 and (level & 255)>=5 and alias=0 order by id')
            if r[0] in codes and r[0] != BLUE]
    assert len(filler) >= 42
    # Variants change visible hand filler and irrelevant deck order within one
    # development family; none of them is an unseen-family test.
    filler = filler[variant:] + filler[:variant]
    opponent = [ANCIENT, ANCIENT, BLUE, BLUE, filler[0]] + filler[1:36]
    learner = [CYBER, QUICKDRAW, VEILER, filler[0], filler[1], filler[2], BULB] + filler[3:36]
    extras = [[], [HALQ, SWORD]]
    if kind in ('battle', 'battle-negated'):
        opponent = [ANCIENT, BLUE] + filler[:38]
        if kind == 'battle-negated':
            opponent = [ANCIENT, BLUE, VEILER] + filler[:37]
        learner = [RAYE] + filler[:39]
        extras = [[], [HAYATE]]
    elif kind == 'interaction':
        opponent = [RAYE, ENGAGE] + filler[:37] + [RAYE]
        learner = [VEILER, OGRE] + filler[:38]
        extras = [[KAGARI], []]
    initial = []
    for p, deck in enumerate((opponent, learner)):
        assert len(deck) == 40 and max(Counter(deck).values()) <= 3
        assert set(deck + extras[p]) <= codes
        initial += [[code, p, 1, 0, 8] for code in reversed(deck)]
        initial += [[code, p, 64, 0, 8] for code in reversed(extras[p])]
    return dict(initial=initial, decks=[opponent, learner], extra=extras, filler=filler,
                family='halq-bulb-sword' if kind == 'combo' else 'kagari-recovery' if kind == 'interaction' else 'hayate-direct-lethal',
                split='development', variant=variant, kind=kind,
                rules='core MR4, 8000 LP, 5 opening cards, 1 draw, first-turn restrictions, no banlist',
                opening_kind='fixed legal deal from decks only', controlled=1)


class OpeningReference(ActorReference):
    def __init__(self, definition, branch='reference'):
        super().__init__('combo' if definition['kind'] == 'combo' else 'battle', branch)
        self.definition = definition

    def choose(self, snap, events):
        menu, op, player = snap['menu'], snap['message'], snap['player']
        branch = self.branch
        def find(**fields):
            return next((i for i, a in enumerate(menu) if all(a[k] == v for k, v in fields.items())), None)
        if self.definition['kind'] == 'interaction':
            if player == 0:
                if op == 11:
                    for code, act in ((ENGAGE, 'Activate'), (RAYE, 'Summon'), (KAGARI, 'SpSummon')):
                        i = find(code=code, act=act)
                        if i is not None:
                            return i
                    return find(phase='End')
                if op in (15, 26):
                    i = find(finish=True)
                    if i is not None:
                        return i
                    for code in (ENGAGE, RAYE):
                        i = find(code=code, unselect=False)
                        if i is not None:
                            return i
                if op in (12, 13):
                    return find(act='Activate')
            elif op == 16:
                if root_matches('interaction', snap, events):
                    code = VEILER if branch == 'veiler' else OGRE if branch == 'ogre' else None
                    i = find(code=code, act='Activate')
                    if i is not None:
                        return i
                return find(act='Cancel')
            elif op in (15, 26):
                return find(code=KAGARI)
            if op == 16:
                return find(act='Cancel')
            if op in (18, 14):
                return 0
            raise ValueError(f'unsupported interaction prompt {snap}')
        if player == 0:
            if op == 11:
                i = find(code=ANCIENT, act='Activate')
                return i if i is not None else find(phase='End')
            if op in (12, 16):
                if self.definition['kind'] == 'battle-negated' and any(
                        c['code'] == HAYATE and c['player'] == 1 and c['location'] == 4
                        for c in getattr(self, 'state', {}).get('cards', [])):
                    i = find(code=VEILER, act='Activate')
                    if i is not None:
                        return i
                return find(act='Cancel')
            if op == 15:
                if self.definition['kind'] == 'battle-negated':
                    i = find(code=HAYATE)
                    if i is not None:
                        return i
                return find(code=BLUE)
            if op == 19:
                return find(position=4 if self.definition['kind'] == 'combo' else 1)
            if op in (18, 14):
                return 0
            if op == 13:
                return find(act='Activate')
        else:
            if self.definition['kind'] in ('battle', 'battle-negated'):
                if op == 10 and branch == 'end_battle':
                    return find(phase='Main2')
                if op == 11:
                    for code, act in ((RAYE, 'Summon'), (HAYATE, 'SpSummon')):
                        i = find(code=code, act=act)
                        if i is not None:
                            return i
                if op in (15, 26):
                    i = find(finish=True)
                    return i if i is not None else find(code=RAYE)
                return super().choose(snap, events)
            if op == 11:
                for code, act in ((CYBER, 'SpSummon'), (QUICKDRAW, 'SpSummon'), (VEILER, 'Summon')):
                    i = find(code=code, act=act)
                    if i is not None:
                        self.summoning = code
                        return i
            if op in (15, 26):
                i = find(finish=True)
                if i is not None:
                    return i
                if self.summoning == QUICKDRAW:
                    priority = self.definition['filler']
                elif self.summoning == HALQ:
                    priority = [VEILER, CYBER, QUICKDRAW, BULB]
                else:
                    priority = [BULB, HALQ, QUICKDRAW, BLUE]
                for code in priority:
                    i = find(code=code, unselect=False)
                    if i is not None:
                        return i
            return super().choose(snap, events)
        raise ValueError(f'unsupported opening teacher prompt {snap}')


def root_matches(depth, snapshot, events):
    if snapshot['player'] != 1:
        return False
    if depth == 'opening':
        return True
    if depth == 'combo':
        return snapshot['message'] == 12 and any(a['code'] == HALQ for a in snapshot['menu'])
    if depth == 'position':
        return snapshot['message'] == 19 and any(a['code'] == BULB for a in snapshot['menu'])
    if depth == 'battle':
        return snapshot['message'] == 10
    if depth == 'interaction':
        return snapshot['message'] == 16 and any(
            e['op'] == 70 and int.from_bytes(bytes.fromhex(e['raw'])[1:5], 'little') == KAGARI
            for e in events)
    raise ValueError('unknown depth')


def turn_boundary_crossed(events, root_event):
    """A native step may auto-skip Main2/End before exposing another menu."""
    for event in events[root_event:]:
        if event['op'] == 40:
            return True
        if event['op'] == 41:
            phase = int.from_bytes(bytes.fromhex(event['raw'])[1:3], 'little')
            if phase in (256, 512):
                return True
    return False


def verdict(state, events, kind='combo'):
    if kind == 'interaction':
        retained = any(c['code'] == ENGAGE and c['player'] == 0 and c['location'] == 16 for c in state['cards'])
        spent = sum(c['player'] == 1 and c['location'] == 16 and c['code'] in (VEILER, OGRE) for c in state['cards'])
        return dict(success=retained and spent <= 1, engage_stays_in_grave=retained, hand_traps_spent=spent)
    if kind == 'battle-negated':
        own_turns = [i for i,e in enumerate(events) if e['op'] == 40 and bytes.fromhex(e['raw'])[1] == 1]
        ended = bool(own_turns) and turn_boundary_crossed(events, own_turns[-1]+1)
        return dict(success=ended and state['lp'][1] == 8000, battle_ended=ended, own_damage=8000-state['lp'][1])
    if kind == 'battle':
        return dict(success=state['lp'][0] == 6500 and state['lp'][1] == 8000,
                    opponent_damage=8000-state['lp'][0], own_damage=8000-state['lp'][1])
    locations = {(c['player'], c['location'], c['sequence']) for c in state['cards']
                 if c['code'] == SWORD and c['player'] == 1 and c['location'] == 4}
    attacks = completed_attacks(events, locations)
    cleared = not any(c['player'] == 0 and c['location'] == 4 for c in state['cards'])
    return dict(success=bool(locations) and attacks >= 2 and cleared, sword_attacks=attacks,
                opponent_board_cleared=cleared)
