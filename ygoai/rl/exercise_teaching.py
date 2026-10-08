"""Normal-opening development exercises; fixed deals, unmodified core rules."""
from collections import Counter
import sqlite3

from ygoai.rl.exercise_actor import ActorReference
from ygoai.rl.exercise_starters import BULB, HALQ, SWORD, VEILER, BLUE, HAYATE, completed_attacks

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
    if kind == 'battle':
        opponent = [ANCIENT, BLUE] + filler[:38]
        learner = [RAYE] + filler[:39]
        extras = [[], [HAYATE]]
    initial = []
    for p, deck in enumerate((opponent, learner)):
        assert len(deck) == 40 and max(Counter(deck).values()) <= 3
        assert set(deck + extras[p]) <= codes
        initial += [[code, p, 1, 0, 8] for code in reversed(deck)]
        initial += [[code, p, 64, 0, 8] for code in reversed(extras[p])]
    return dict(initial=initial, decks=[opponent, learner], extra=extras, filler=filler,
                family='halq-bulb-sword' if kind == 'combo' else 'hayate-direct-lethal',
                split='development', variant=variant, kind=kind,
                rules='core MR4, 8000 LP, 5 opening cards, 1 draw, first-turn restrictions, no banlist',
                opening_kind='fixed legal deal from decks only', controlled=1)


class OpeningReference(ActorReference):
    def __init__(self, definition, branch='reference'):
        super().__init__('combo' if definition['kind'] == 'combo' else 'battle', branch)
        self.definition = definition

    def choose(self, snap, events):
        menu, op, player = snap['menu'], snap['message'], snap['player']
        def find(**fields):
            return next((i for i, a in enumerate(menu) if all(a[k] == v for k, v in fields.items())), None)
        if player == 0:
            if op == 11:
                i = find(code=ANCIENT, act='Activate')
                return i if i is not None else find(phase='End')
            if op in (12, 16):
                return find(act='Cancel')
            if op == 15:
                return find(code=BLUE)
            if op == 19:
                return find(position=4 if self.definition['kind'] == 'combo' else 1)
            if op in (18, 14):
                return 0
            if op == 13:
                return find(act='Activate')
        else:
            if self.definition['kind'] == 'battle':
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
    if depth == 'combo':
        return snapshot['message'] == 12 and any(a['code'] == HALQ for a in snapshot['menu'])
    if depth == 'position':
        return snapshot['message'] == 19 and any(a['code'] == BULB for a in snapshot['menu'])
    if depth == 'battle':
        return snapshot['message'] == 10
    raise ValueError('unknown depth')


def verdict(state, events, kind='combo'):
    if kind == 'battle':
        return dict(success=state['lp'][0] == 6500 and state['lp'][1] == 8000,
                    opponent_damage=8000-state['lp'][0], own_damage=8000-state['lp'][1])
    locations = {(c['player'], c['location'], c['sequence']) for c in state['cards']
                 if c['code'] == SWORD and c['player'] == 1 and c['location'] == 4}
    attacks = completed_attacks(events, locations)
    cleared = not any(c['player'] == 0 and c['location'] == 4 for c in state['cards'])
    return dict(success=bool(locations) and attacks >= 2 and cleared, sword_attacks=attacks,
                opponent_board_cleared=cleared)
