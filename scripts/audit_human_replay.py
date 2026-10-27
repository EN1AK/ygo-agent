"""Replay YRP2 input exactly; stop on drift, never invent fallback responses."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

try:
    import _repo_bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _repo_bootstrap  # noqa: F401
from ygoai.rl.demonstrations import parse_yrp2
from ygoai.rl.exercise_core import Core, sha


def initial_cards(replay):
    return [[code, p, loc, 0, 8] for p in (0, 1)
            for loc, deck in ((1, replay.main_decks[p]), (64, replay.extra_decks[p]))
            for code in deck]


def audit(args):
    core_type = Core
    if getattr(args, 'core_abi', 'legacy') == 'modern':
        from ygoai.rl.modern_replay_core import Core as core_type
    replay = parse_yrp2(args.input)
    record = dict(schema='ygo-human-core-replay-v1', source_sha256=sha(args.input),
                  source=asdict(replay), training_ready=False)
    core = core_type(args.native, args.database, args.scripts, seed_sequence=replay.seed_sequence)
    try:
        core.start_standard(initial_cards(replay), duel_flags=replay.duel_flags,
                            lp=replay.initial_lp, hand=replay.initial_hand, draw=replay.draw_count)
        for response in replay.responses:
            prompt = core.next_prompt()
            if prompt is None:
                raise ValueError('engine ended before recorded inputs were consumed')
            core.respond(prompt, bytes.fromhex(response))
        next_prompt = core.next_prompt()
        record.update(status='responses_replayed', next_prompt=next_prompt)
    except (ValueError, TimeoutError) as exc:
        record.update(status='rejected', error=str(exc))
    finally:
        record.update(runtime=core.runtime, assets=core.assets, decisions=core.decisions,
                      events=core.events, final=core.state())
        core.close()
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('native', 'database', 'scripts', 'input', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--actor-record', action='store_true')
    p.add_argument('--core-abi', choices=('legacy', 'modern'), default='legacy',
                   help='Modern uses the pinned 80-byte CardData/create_duel_v2 ABI')
    args = p.parse_args()
    record = audit_actor(args) if args.actor_record else audit(args)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status=record['status'], error=record.get('error'))))
    if record['status'] not in ('responses_replayed', 'actor_audited'):
        raise SystemExit(1)


def audit_actor(args):
    import numpy as np
    from ygoai.rl.exercise_actor import array_digest
    record = json.loads(args.input.read_text(encoding='utf-8'))
    with np.load(args.input.with_suffix('.npz'), allow_pickle=False) as f:
        arrays = dict(f)
    if any(len(v) != len(record['decisions']) for v in arrays.values()):
        raise ValueError('observation count mismatch')
    for i, decision in enumerate(record['decisions']):
        if decision['observation_sha256'] != array_digest({k: v[i:i+1] for k, v in arrays.items()}):
            raise ValueError('observation digest mismatch')
    definition = record['definition']
    core = Core(args.native, args.database, args.scripts, seed_sequence=definition['seed_sequence'])
    try:
        core.start_standard(definition['initial'], duel_flags=definition['duel_flags'])
        for response in record['core_responses']:
            prompt = core.next_prompt()
            if prompt is None:
                raise ValueError('core ended before actor trace')
            core.respond(prompt, bytes.fromhex(response['data']))
        core.next_prompt()
        if core.events != record['events'] or core.state() != record['final']:
            raise ValueError('independent actor/core events or final state mismatch')
        return dict(status='actor_audited', passed=True, path=args.input.name,
                    record_sha256=sha(args.input), observations_sha256=sha(args.input.with_suffix('.npz')),
                    native_sha256=sha(args.native), runtime=core.runtime, assets=core.assets)
    finally:
        core.close()


if __name__ == '__main__':
    main()
