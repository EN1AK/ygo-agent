"""Legal history variations of the three development exercise families."""
from copy import deepcopy
from ygoai.rl.exercise_teaching import opening, OpeningReference
from ygoai.rl.exercise_starters import HALQ, VEILER, HAYATE

COUNTDOWN, TOON, OOKAZI = 95308449, 15259703, 19523799
PROFILES = [
    dict(name='base', split='fit'),
    dict(name='zones', place_last=True, split='fit'),
    dict(name='route', alternate=True, split='fit'),
    dict(name='seat', controlled=0, split='fit'),
    dict(name='low', low=True, split='fit'),
    dict(name='seat-low', controlled=0, low=True, split='fit'),
    dict(name='zones-route', place_last=True, alternate=True, variant=1, split='development-combination'),
    dict(name='seat-route', controlled=0, alternate=True, variant=2, split='development-combination'),
    dict(name='seat-low-zones-route', controlled=0, low=True, place_last=True, alternate=True,
         variant=3, split='development-combination'),
]


def context_opening(database, code_list, kind, profile):
    d=opening(database,code_list,profile.get('variant',0),kind)
    d['context']=deepcopy(profile)
    d['expected_lp']=[8000,8000]
    if profile.get('low'):
        f=d['filler']
        if kind=='combo':
            old=d['decks'][1]
            d['decks'][1]=old[:3]+[COUNTDOWN,COUNTDOWN,f[0]]+old[6:38]+f[36:38]
            d['expected_lp'][1]=4000
        elif kind in ('battle','battle-negated'):
            d['decks'][1]=[26077387,COUNTDOWN,COUNTDOWN,COUNTDOWN,TOON]+f[:35]
            d['expected_lp'][1]=1000
        else:
            d['decks'][0]=d['decks'][0][:2]+[OOKAZI,OOKAZI]+f[:35]+[26077387]
            d['expected_lp'][1]=6400
    if profile.get('controlled',1)==0:
        d['decks'].reverse(); d['extra'].reverse(); d['expected_lp'].reverse(); d['controlled']=0
    d['initial']=[]
    for p in (0,1):
        assert len(d['decks'][p])==40
        from collections import Counter
        assert max(Counter(d['decks'][p]).values())<=3
        d['initial'] += [[c,p,1,0,8] for c in reversed(d['decks'][p])]
        d['initial'] += [[c,p,64,0,8] for c in reversed(d['extra'][p])]
    codes=set(map(int,code_list.read_text().split()))
    assert all(row[0] in codes for row in d['initial'])
    d['rules']+='; optional LP costs/damage only through legal card effects'
    d['opening_kind']='fixed legal deal; setup prefix excluded from teaching labels'
    return d


def current_turn(events):
    turns=[bytes.fromhex(e['raw'])[1] for e in events if e['op']==40]
    return turns[-1] if turns else None, len(turns)


def ready(definition,snap,events,state):
    own=definition['controlled']; turn,n=current_turn(events)
    if definition['kind']=='interaction':
        return turn==1-own and n >= (2 if own==0 else 1)
    if turn!=own or n < (3 if own==0 else 2): return False
    return not any(c['player']==own and c['location']==2 and c['code'] in (COUNTDOWN,TOON)
                   for c in state['cards'])


class ContextReference(OpeningReference):
    def choose(self,snap,events):
        d=self.definition; own=d['controlled']; turn,n=current_turn(events)
        menu=snap['menu']; op=snap['message']; player=snap['player']
        def find(**fields):
            return next((i for i,a in enumerate(menu) if all(a[k]==v for k,v in fields.items())),None)
        if d['context'].get('place_last') and op==18: return len(menu)-1
        if own==0 and n==1:
            # A normal first turn, retained in the recurrent history.
            for fields in (dict(phase='End'),dict(act='Cancel')):
                i=find(**fields)
                if i is not None: return i
            raise ValueError('unsupported first-turn setup prompt')
        if op==11:
            for code in ((COUNTDOWN,TOON) if player==own else (OOKAZI,)):
                i=find(code=code,act='Activate')
                if i is not None: return i
        if d['kind']!='interaction' and player==own and turn!=own and op==16:
            return find(act='Cancel')
        if d['context'].get('alternate') and player==own:
            if d['kind'] in ('battle','battle-negated'):
                if op==16:
                    i=find(code=26077387,act='Activate')
                    if i is not None: return i
                if op in (15,26):
                    i=find(code=HAYATE)
                    if i is not None: return i
            if d['kind']=='combo' and op in (15,26) and self.summoning==HALQ:
                i=find(finish=True)
                if i is not None: return i
                for code in (VEILER,20932152,70095154):
                    i=find(code=code,unselect=False)
                    if i is not None: return i
        normalized=dict(snap,player=1 if player==own else 0)
        state=self.state
        self.state=dict(state,cards=[dict(c,player=1 if c['player']==own else 0) for c in state['cards']])
        try: return super().choose(normalized,events)
        finally: self.state=state


def context_verdict(state,events,definition):
    from ygoai.rl.exercise_teaching import verdict
    own=definition['controlled']; kind=definition['kind']
    if kind in ('battle','battle-negated'):
        expected=definition['expected_lp']; turn,n=current_turn(events)
        if kind=='battle':
            return dict(success=state['lp'][1-own]==expected[1-own]-1500 and state['lp'][own]==expected[own],
                        opponent_damage=expected[1-own]-state['lp'][1-own],own_damage=expected[own]-state['lp'][own])
        from ygoai.rl.exercise_teaching import turn_boundary_crossed
        turns=[i for i,e in enumerate(events) if e['op']==40 and bytes.fromhex(e['raw'])[1]==own]
        ended=bool(turns) and turn_boundary_crossed(events,turns[-1]+1)
        # A required Hayate on the final board prevents passing by skipping the exercise.
        hayate=any(c['code']==HAYATE and c['player']==own and c['location']==4 for c in state['cards'])
        return dict(success=ended and hayate and state['lp'][own]==expected[own],battle_ended=ended,
                    hayate_present=hayate,own_damage=expected[own]-state['lp'][own])
    normalized=dict(state,cards=[dict(c,player=1 if c['player']==own else 0) for c in state['cards']])
    if kind=='interaction': return verdict(normalized,events,kind)
    from ygoai.rl.exercise_starters import SWORD,completed_attacks
    locations={(own,4,c['sequence']) for c in state['cards'] if c['code']==SWORD and c['player']==own and c['location']==4}
    attacks=completed_attacks(events,locations)
    cleared=not any(c['player']==1-own and c['location']==4 for c in state['cards'])
    return dict(success=bool(locations) and attacks>=2 and cleared,sword_attacks=attacks,opponent_board_cleared=cleared)
