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
