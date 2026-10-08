"""Scene/log -> versioned candidate; candidate does not imply a correct answer."""
import argparse
import json
from pathlib import Path

from ygoai.rl.capability_exercises import (draft, mine_decisions, validate_candidate,
                                         validate_verification, assign_splits)
from ygoai.rl.exercise_starters import CASES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('draft')
    d.add_argument('--scene', required=True)
    d.add_argument('--case', choices=list(CASES))
    m = sub.add_parser('mine')
    m.add_argument('--decisions', type=Path, required=True)
    m.add_argument('--step', type=int, action='append', default=[])
    m.add_argument('--max-candidates', type=int, default=48)
    s = sub.add_parser('split')
    s.add_argument('--candidates', type=Path, required=True)
    v = sub.add_parser('audit')
    v.add_argument('--summary', type=Path, required=True)
    for command in (d, m, s, v):
        command.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'draft':
        result = validate_candidate(draft(args.scene, case=args.case))
    elif args.command == 'mine':
        result = mine_decisions(args.decisions, steps=args.step, max_candidates=args.max_candidates)
    elif args.command == 'split':
        data = json.loads(args.candidates.read_text(encoding='utf-8'))
        result = assign_splits(data['candidates'])
    else:
        result = validate_verification(json.loads(args.summary.read_text(encoding='utf-8')), args.summary.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(str(args.output))


if __name__ == '__main__':
    main()
