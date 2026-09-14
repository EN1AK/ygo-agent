"""Run bounded WindBot evaluation attempts under Linux process supervision."""

import argparse
import json
import math
from pathlib import Path
import sys

import _repo_bootstrap  # noqa: F401
from ygoai.windbot_supervisor import run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=1, help='Total attempts, including invalid games')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--player', type=int, choices=[0, 1], default=1)
    parser.add_argument('--alternate-seats', action='store_true', help='Run each seed in both seats')
    parser.add_argument('--learner-deck', default='k9vs')
    parser.add_argument('--opponent-deck', default='AI_OldSchool')
    parser.add_argument('--timeout', type=float, default=120, help='Wall seconds per worker, including startup')
    parser.add_argument('--output', required=True, type=Path, help='New result directory; must not already exist')
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('eval_args', nargs=argparse.REMAINDER, help='After --, options for scripts/eval.py')
    args = parser.parse_args()
    if args.games < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--games and --timeout must be positive')
    forwarded = args.eval_args[1:] if args.eval_args[:1] == ['--'] else args.eval_args
    owned = {'bot-type', 'num-envs', 'num-episodes', 'player', 'seed', 'deck1', 'deck2',
             'windbot-port', 'windbot-log-dir', 'windbot-metadata', 'windbot-server-mode', 'play'}
    for token in forwarded:
        if token.startswith('--') and token[2:].split('=')[0].replace('_', '-') in owned:
            parser.error(f'{token} is controlled by the supervisor')
    root = args.repo_root.resolve()
    output = args.output.resolve()

    def jobs():
        for index in range(args.games):
            player = (args.player + index) % 2 if args.alternate_seats else args.player
            seed = args.seed + (index // 2 if args.alternate_seats else index)
            decks = [args.opponent_deck, args.learner_deck] if player else [args.learner_deck, args.opponent_deck]
            folder = output / f'attempt-{index + 1:04d}'
            command = [sys.executable, '-u', str(root / 'scripts/eval.py'), *forwarded,
                       '--bot-type', 'windbot', '--num-envs', '1', '--num-episodes', '1',
                       '--player', str(player), '--seed', str(seed),
                       '--deck1', decks[0], '--deck2', decks[1], '--windbot-port', '0',
                       '--windbot-log-dir', str(folder / 'windbot'),
                       '--windbot-metadata', str(folder / 'metadata.json')]
            yield {'command': command, 'seed': seed, 'player': player,
                   'learner_deck': args.learner_deck, 'opponent_deck': args.opponent_deck}

    summary = run_batch(jobs(), output, root, args.timeout)
    print(json.dumps(summary, indent=2))
    return 130 if summary['interrupted'] else (1 if summary['invalid'] else 0)


if __name__ == '__main__':
    raise SystemExit(main())
