"""Typed extra-deck aliases for navigation-normalized positive witnesses.

An extra-deck permutation is not an identical RNG realization. These helpers
preserve the board, ordered main deck/hand and exact extra inventory. A pending
XYZ material destination is bound to its subsequent actual summon, and only
when that code has one physical extra-deck copy in the declared opening.
"""
from collections import Counter
import copy
import re
from contracts import digest


def unique_extra_cards(definition):
    counts=Counter((row[1],row[0]) for row in definition['initial'] if row[2]==64)
    return {key for key,count in counts.items() if count==1}


def board_and_extra_inventory(state):
    result=copy.deepcopy(state)
    extra=[{k:v for k,v in c.items() if k!='sequence'} for c in result['cards'] if c['location']==64]
    result['cards']=[c for c in result['cards'] if c['location']!=64]
    result['extra_inventory']=sorted(extra,key=digest)
    return result


def extra_key(action,player,unique):
    result=dict(action);match=re.fullmatch(r'(o?)x[0-9]+',result.get('spec',''))
    if match:
        owner=1-player if match[1] else player
        if (owner,result.get('code',0)) in unique:
            result['spec']='unique-extra:'+str(owner)+':'+str(result['code'])
            result.pop('response',None)
    return result


def extra_menu_permutation(expected,actual,player,unique):
    if expected==actual:return list(range(len(expected)))
    old=[digest(extra_key(a,player,unique)) for a in expected]
    new=[digest(extra_key(a,player,unique)) for a in actual]
    if len(set(old))!=len(old) or len(set(new))!=len(new) or set(old)!=set(new):
        raise ValueError('extra-deck choices changed or physical copy is ambiguous')
    # Ordinary choices must retain their own relative order.
    ordinary=lambda menu:[a for a in menu if not re.fullmatch(r'o?x[0-9]+',a.get('spec',''))]
    if ordinary(expected)!=ordinary(actual):raise ValueError('non-extra choices changed')
    return [new.index(key) for key in old]


def pending_xyz_parent(events,start,unique):
    first=bytes.fromhex(events[start]['raw']);owner,slot=first[9],first[11]
    block=[]
    for event in events[start+1:]:
        if event['op']==62:
            raw=bytes.fromhex(event['raw']);code=int.from_bytes(raw[1:5],'little')
            if raw[5]!=owner or raw[6]!=4 or (owner,code) not in unique:return None
            for move in block:
                if move[10]==192:
                    if move[9]!=owner or move[11]!=slot or not int.from_bytes(move[13:17],'little')&0x200000:return None
                elif not (int.from_bytes(move[1:5],'little')==code and move[6]==64 and move[10]==4):return None
            return owner,code
        if event['op']==50:block.append(bytes.fromhex(event['raw']))
        elif event['op'] not in (2,3,18,19,26):return None
    return None


def normalize_extra_events(events,unique,prefix=False):
    result=[]
    for index,event in enumerate(events):
        if event['op']!=50:
            result.append(event);continue
        raw=bytearray.fromhex(event['raw']);code=int.from_bytes(raw[1:5],'little');bindings=[]
        if raw[6]==64 and (raw[5],code) in unique:
            raw[7]=0;bindings.append(('from-extra',raw[5],code))
        if raw[10]==64 and (raw[9],code) in unique:
            raw[11]=0;bindings.append(('to-extra',raw[9],code))
        if raw[10]==192 and int.from_bytes(raw[13:17],'little')&0x200000:
            parent=('pending',raw[9]) if prefix else pending_xyz_parent(events,index,unique)
            # Unresolved targets remain byte-exact. Prefix alignment is only
            # provisional; a changed unresolved alias cannot pass final proof.
            if parent is not None:
                raw[11]=0;bindings.append(('xyz-parent',*parent))
        if bindings:
            result.append(dict(event,raw=raw.hex()+'|extra-bindings:'+digest(bindings)))
        else:result.append(event)
    return result
