"""Remove only observed UI loops returning to an identical engine/menu state."""
from contracts import digest

# These packets request human navigation or display hints. Keep activation,
# yes/no, chain and every game mutation in the equivalence transcript.
NAVIGATION_PACKETS = {2, 3, 11, 15, 20, 26}


def committed_events(events):
    from ygoai.rl.human_replay import comparable_game_events
    return comparable_game_events([e for e in events if e['op'] not in NAVIGATION_PACKETS])


def loop_erasure(frames):
    """Frames include the final endpoint; decisions occupy all earlier frames.

    Equal field state alone is insufficient: a strategic engine event or a
    different legal menu prevents erasure, even if cards have not moved yet.
    """
    seen, removed, replacements, loops = {}, set(), {}, []
    for index, frame in enumerate(frames):
        key = digest(frame)
        if key in seen:
            start = seen[key]
            removed.update(range(start, index))
            replacements[start] = index
            loops.append(dict(start=start, end=index, same_frame_sha256=key))
            seen = {k:v for k,v in seen.items() if v < start}
        seen[key] = index
    return [i for i in range(len(frames)-1) if i not in removed], replacements, loops
