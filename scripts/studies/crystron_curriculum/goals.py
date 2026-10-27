"""Outcome predicates for resource modules and local practical skills."""
from collections import Counter


def own_attacks(events, seat):
    active=False;count=0
    for e in events:
        if e['op']==40:active=False;count=0
        elif e['op']==110:active=bytes.fromhex(e['raw'])[1]==seat
        elif e['op']==112:active=False
        elif e['op']==114:
            count+=int(active);active=False
    return count


def achieved(goal,state,snapshot,events,seat):
    kind=goal['kind']
    if kind=='win_after_first_own_main':return state['winner']==seat
    turn=sum(e['op']==40 for e in events)
    if turn>goal.get('max_turn',1):return False
    if kind=='settled_resource_inventory':
        if snapshot['message']!=11 or snapshot['player']!=seat:return False
        own=Counter((c['location'],c['code']) for c in state['cards'] if c['player']==seat and (c['location'] in (2,16,32) or c['location'] in (4,8) and c['position']&5))
        return all(own[(x['location'],x['code'])]>=x['count'] for x in goal['cards'])
    if state['winner']==seat:return True
    if state['lp'][seat]<=0:return False
    stable=snapshot['message'] in (10,11) and snapshot['player']==seat
    if not stable:return False
    if kind=='remove_public_threat':
        field=Counter(c['code'] for c in state['cards'] if c['player']==1-seat and c['location'] in (4,8))
        return field[goal['code']]<=goal['max_remaining']
    if kind in ('battle_damage','attack_and_lp_reduction'):
        return own_attacks(events,seat)>=goal['min_completed_attacks'] and state['lp'][1-seat]<=goal['opponent_lp_at_most']
    if kind=='postbattle_resources':
        own=[c for c in state['cards'] if c['player']==seat]
        return state['phase']==0x100 and own_attacks(events,seat)>=1 and sum(c['location']==2 or c['location']==4 and bool(c['position']&5) for c in own)>=goal['min_resources']
    if kind=='recovery_board':
        cards=Counter(c['code'] for c in state['cards'] if c['player']==seat and c['location']==4 and c['position']&5)
        return all(cards[int(k)]>=v for k,v in goal['monsters'].items())
    raise ValueError('unknown curriculum goal '+kind)
