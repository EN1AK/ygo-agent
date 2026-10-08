"""Reference routing and result extraction for the original actor adapter."""
import hashlib
import re

import numpy as np

from ygoai.rl.exercise_core import Reader, messages
from ygoai.rl.exercise_starters import (BULB, ENGAGE, FADER, HALQ, HAYATE, KAGARI,
                                       MAXXC, OGRE, SWORD, VEILER, BLUE,
                                       completed_attacks, grade)


def array_digest(arrays):
    h = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        a = np.asarray(array)
        h.update(name.encode())
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def initial_cards(lua):
    rows = []
    for args in re.findall(r'Debug.AddCard\(([^)]+)\)', lua):
        code, owner, controller, location, sequence, position = map(int, args.split(','))
        if owner != controller:
            raise ValueError('starter owner/controller mismatch')
        rows.append([code, owner, location, sequence, position])
    return rows


def extract_trace(native, snapshot):
    trace = native.trace()
    events = [m for packet in trace['packets'] for m in messages(bytes(packet))]
    cards = []
    for field in trace['fields']:
        r = Reader(bytes(field))
        while r.i < len(r.data):
            size = r.u32()
            if size == 4:
                continue
            b = Reader(r.take(size - 4))
            if b.u32() != 3:
                raise ValueError('query flag mismatch')
            cards.append(dict(code=b.u32(), player=b.u8(), location=b.u8(), sequence=b.u8(), position=b.u8()))
    state = {k: snapshot[k] for k in ('phase', 'turn_player', 'lp')}
    state.update(cards=cards, winner=None if snapshot['winner'] < 0 else snapshot['winner'])
    return events, state, [dict(kind=r['kind'], data=bytes(r['data']).hex()) for r in trace['responses']]


def hidden_identity_audit(obs):
    """Starter initial/root windows have no revealed opponent private cards."""
    cards = np.asarray(obs['cards_'])[0]
    # Legacy location IDs 1/2/7 = deck/hand/extra, controller 1 = opponent.
    private = (cards[:, 4] == 1) & np.isin(cards[:, 2], [1, 2, 7])
    ids = cards[:, :2].astype(np.int32)
    exposed = np.any(ids[private] != 0)
    if 'visible_card_ids_' in obs:
        exposed = exposed or np.any(np.asarray(obs['visible_card_ids_'])[0, private] != 0)
    if exposed:
        raise ValueError('opponent private identity exposed at starter boundary')
    return int(private.sum())


class ActorReference:
    def __init__(self, case, branch):
        self.case, self.branch = case, branch
        self.summoning = None

    def is_root(self, snap, events):
        op, menu = snap['message'], snap['menu']
        if self.case == 'combo':
            return op == 12 and any(a['code'] == HALQ for a in menu)
        if self.case.startswith('battle'):
            return op == 10 and snap['player'] == 0
        return op == 16 and snap['player'] == 1 and any(
            e['op'] == 70 and int.from_bytes(bytes.fromhex(e['raw'])[1:5], 'little') == KAGARI for e in events)

    def choose(self, snap, events):
        menu, op, case, branch = snap['menu'], snap['message'], self.case, self.branch
        def find(**fields):
            return next((i for i, a in enumerate(menu) if all(a[k] == v for k, v in fields.items())), None)
        def required(**fields):
            i = find(**fields)
            if i is None:
                raise ValueError(f'reference missing action {fields}, menu={menu}')
            return i
        attacks = completed_attacks(events)
        if op in (10, 11):
            for code in ((HALQ, SWORD) if case == 'combo' else (KAGARI,) if case == 'interaction' else ()):
                i = find(code=code, act='SpSummon')
                if i is not None:
                    self.summoning = code
                    return i
            if case == 'combo':
                for code in (BULB, SWORD):
                    i = find(code=code, act='Activate')
                    if i is not None and not (code == SWORD and branch == 'effect_after_first_attack' and attacks == 0):
                        return i
            if op == 11:
                return required(phase='Battle')
            if case == 'battle-negated' and branch == 'end_battle':
                return required(phase='Main2')
            attacker = SWORD if case == 'combo' else HAYATE
            for action in ('Attack', 'DirectAttack'):
                i = find(code=attacker, act=action)
                if i is not None:
                    return i
            return required(phase='Main2')
        if op == 12:
            return required(act='Activate') if any(a['code'] in (HALQ, KAGARI) for a in menu) else required(act='Cancel')
        if op == 13:
            # Direct-attack choice may be encoded as an anonymous yes/no menu.
            direct = any(a['effect'] == 31 for a in menu)
            return required(act='Cancel') if direct and branch != 'direct' else required(act='Activate')
        if op == 16:
            if case == 'interaction' and self.is_root(snap, events):
                target = VEILER if branch == 'veiler' else OGRE if branch == 'ogre' else None
                i = find(code=target, act='Activate')
                if i is not None:
                    return i
            if case == 'battle-negated' and snap['player'] == 1 and snap['phase'] == 4:
                i = find(code=VEILER, act='Activate')
                if i is not None:
                    return i
            if case == 'combo':
                for code in (HALQ, SWORD):
                    i = find(code=code, act='Activate')
                    if i is not None and not (code == SWORD and branch == 'effect_after_first_attack' and attacks == 0):
                        return i
            return required(act='Cancel')
        if op in (15, 26):
            i = find(finish=True)
            if i is not None:
                return i
            preferred = ([VEILER, MAXXC] if self.summoning == HALQ else [])
            for code in preferred + [BULB, HALQ, FADER, ENGAGE, KAGARI, HAYATE, BLUE]:
                i = find(code=code, unselect=False)
                if i is not None:
                    return i
            raise ValueError('no reference selection')
        if op == 18:
            return 0
        if op == 19:
            return required(position=4 if branch == 'revive_in_defense' else 1)
        if op == 14:
            return 0
        raise ValueError(f'unsupported reference prompt {op}')


def settled(case, snapshot, events, root_event):
    if snapshot['done']:
        return not snapshot['invalid']
    if root_event is None:
        return False
    if case == 'interaction':
        return any(m['op'] == 74 for m in events[root_event:])
    return snapshot['phase'] in (0x100, 0x200) or (case == 'combo' and completed_attacks(events) >= 2)


def actor_grade(case, state, events, stable):
    result = grade(case, state, events, stable=stable)
    if case == 'interaction' and result == 'success':
        spent = sum(c['player'] == 1 and c['location'] == 16 and c['code'] in (VEILER, OGRE) for c in state['cards'])
        if spent > 1:
            result = 'verified_failure'
    return result
