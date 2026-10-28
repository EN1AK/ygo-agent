"""Remove only observed UI loops returning to an identical engine/menu state."""
from contracts import digest

# These packets request human navigation or display hints. Keep activation,
# yes/no, chain and every game mutation in the equivalence transcript.
NAVIGATION_PACKETS = {2, 3, 11, 15, 20, 26}


def selection_navigation_responses(events,responses):
    from ygoai.rl.exercise_core import PROMPTS
    prompts=[e for e in events if e['op'] in PROMPTS]
    if len(prompts) not in (len(responses),len(responses)+1):
        raise ValueError('cannot bind navigation responses to core prompts')
    undo,cancel=set(),set()
    for index,(prompt,response) in enumerate(zip(prompts,responses)):
        raw=bytes.fromhex(response['data'])
        if prompt['op'] in (15,20,26) and raw==b'\xff'*4:cancel.add(index)
        if prompt['op']==26 and len(raw)==2 and raw[0]==1:
            if not raw[1]<len(prompt['cards'])+len(prompt['unselect']):
                raise ValueError('unselect response outside core pools')
            if raw[1]>=len(prompt['cards']):undo.add(index)
    return undo,cancel


def selection_context(events,responses,unique_extra):
    """Transaction identity omitted by identical material menus/field views.

    A canceled attempt to summon A followed by opening B is not an undo loop
    inside the material picker, even when A and B use the same materials.
    An idle command is tentative until its selection commits or game messages
    show the summon; canceling it back to idle leaves the epoch unchanged.
    """
    from ygoai.rl.exercise_core import PROMPTS
    pending=None;epoch=0;index=0
    for event in events:
        op=event['op']
        if op in (5,40,54,60,62,64,70,72) or op==11:pending=None
        if op not in PROMPTS:continue
        if index==len(responses):break
        raw=bytes.fromhex(responses[index]['data']);index+=1
        if op==11:
            value=int.from_bytes(raw,'little',signed=True);kind,slot=value&65535,value>>16
            group={0:'summon',1:'special',3:'set_monster'}.get(kind)
            if group is not None:
                if not 0<=slot<len(event[group]):raise ValueError('summon response outside raw command list')
                card=event[group][slot]
                pending={k:card[k] for k in ('code','player','location')};pending['command']=kind
                if card['location']!=64 or (card['player'],card['code']) not in unique_extra:
                    pending['sequence']=card['sequence']
            else:epoch+=1
        elif op==26:
            if raw==b'\xff'*4 and (event['finish'] or pending is None):epoch+=1
        elif op in (15,20):
            if raw!=b'\xff'*4 or pending is None:epoch+=1
        elif op==18 and pending is not None and len(raw)==3 and raw[1:]==b'\x00\x00':
            pass  # empty-place return inside an uncommitted summon
        else:epoch+=1
    return dict(epoch=epoch,pending_summon=pending)


def committed_events(events):
    from ygoai.rl.human_replay import comparable_game_events
    stable=[]
    for event in events:
        if event['op'] in NAVIGATION_PACKETS:continue
        if event['op']==16:
            # Generic cross-core event comparison omits chain wire layouts.
            # Loop detection must still preserve every priority window: two
            # consecutive declines can resolve a chain without moving cards.
            payload={k:v for k,v in event.items() if k not in ('raw','op','cards')}
            payload['cards']=sorted(event['cards'],key=digest)
            stable.append(dict(op=-16,raw='chain-window:'+digest(payload)))
        elif event['op']==31:
            stable.append(dict(event,op=-31))  # preserve information reveals
        else:stable.append(event)
    return comparable_game_events(stable)


def chain_menu_permutation(message, expected, actual):
    """Only chain order is an acknowledged core-container permutation.

    Every full action descriptor must match bijectively. No gameplay choice or
    response value is erased, and ambiguous duplicate rows remain rejected.
    Returns actual indices in the expected order.
    """
    if expected==actual:return list(range(len(expected)))
    if message!=16:raise ValueError('non-chain menu changed')
    old,new=[digest(a) for a in expected],[digest(a) for a in actual]
    if len(set(old))!=len(old) or len(set(new))!=len(new) or set(old)!=set(new):
        raise ValueError('chain choices changed or are ambiguous')
    return [new.index(key) for key in old]


def align_action_rows(observations, permutation):
    """Reorder only action candidates; all other native inputs stay exact."""
    result={k:v.copy() for k,v in observations.items()}
    for key in result:
        if key=='actions_' or key.startswith('action_'):
            result[key][:,:len(permutation)]=observations[key][:,permutation]
    return result


def loop_erasure(frames, removable):
    """Frames include the final endpoint; decisions occupy all earlier frames.

    Equal field state alone is insufficient: a strategic engine event or a
    different legal menu prevents erasure, even if cards have not moved yet.
    """
    if len(removable)!=len(frames)-1:raise ValueError('navigation evidence missing')
    seen, removed, replacements, loops = {}, set(), {}, []
    for index, frame in enumerate(frames):
        key = digest(frame)
        if key in seen and any(removable[seen[key]:index]) and all(
                f['message'] in (11,15,18,20,26) for f in frames[seen[key]:index]):
            start = seen[key]
            removed.update(range(start, index))
            replacements[start] = index
            loops.append(dict(start=start, end=index, same_frame_sha256=key))
            seen = {k:v for k,v in seen.items() if v < start}
        seen[key] = index
    return [i for i in range(len(frames)-1) if i not in removed], replacements, loops
