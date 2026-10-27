"""Inventory every downloaded Crystron replay without an FTK or deck-size gate.

This is an evidence inventory, not automatic admission to imitation training.
An incomplete recording supplies no terminal win/loss label.
"""
import argparse
import collections
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ygoai.rl.demonstrations import parse_yrp2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def turn_segments(events, seat):
    turns, current = [], None
    for i, event in enumerate(events):
        raw = bytes.fromhex(event['raw'])
        if event['op'] == 40:
            if current is not None:
                current.update(end_event=i, boundary_complete=True)
            current = dict(turn=len(turns)+1, player=raw[1], start_event=i,
                           end_event=len(events), boundary_complete=False,
                           attacks=0, opponent_damage=0)
            turns.append(current)
        elif current is not None:
            if event['op'] == 110:
                current['attacks'] += 1
            # MSG_DAMAGE does not itself identify effect versus battle damage.
            # MSG_LPUPDATE (94) is an absolute LP assignment, not damage.
            if event['op'] == 91 and raw[1] == 1-seat:
                current['opponent_damage'] += int.from_bytes(raw[2:6], 'little')
    return turns


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--additional-audit', type=Path, help='Verified proofs for previously unaudited recordings')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    index = json.loads((a.source/'index.json').read_text('utf-8'))
    db = sqlite3.connect(f'file:{(a.source/"actor-runtime-v3/assets/cards.cdb").resolve().as_posix()}?mode=ro', uri=True)
    names = dict(db.execute('select id,name from texts'))
    db.close()
    parsed, rows, candidates = {}, [], []
    for entry in index['selected']:
        sha = entry['digest']
        path = a.source/'replays'/entry['match_id']/(sha+'.yrp')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        replay = parse_yrp2(path)
        parsed[sha] = asdict(replay)
        proof_path = a.source/'audit-repaired-v7'/(sha+'.json')
        if a.additional_audit:
            additional=a.additional_audit/(sha+'.json')
            if additional.exists():
                if proof_path.exists():raise ValueError('overlapping audit versions for '+sha)
                proof_path=additional
        proof = json.loads(proof_path.read_text('utf-8')) if proof_path.exists() else {}
        for seat in entry['crystron_seats']:
            deck = replay.main_decks[seat]
            deck_hash = digest([sorted(deck), sorted(replay.extra_decks[seat])])
            candidate = dict(source_sha256=sha, match_id=entry['match_id'], seat=seat,
                             relative_path=path.relative_to(a.source).as_posix(),
                             responses=len(replay.responses), main=deck,
                             extra=replay.extra_decks[seat], names=replay.names,
                             deck_sha256=deck_hash,
                             missing_ids=sorted({c for d in replay.main_decks+replay.extra_decks for c in d if c not in names}),
                             training_ready=False)
            candidates.append(candidate)
            events, final = proof.get('events', []), proof.get('final', {})
            turns = turn_segments(events, seat)
            completed = proof.get('status') == 'complete'
            ftk = completed and final.get('winner') == seat and len(turns) == 1 and final.get('lp', [8000,8000])[1-seat] <= 0
            rows.append(dict(source_sha256=sha, match_id=entry['match_id'], seat=seat,
                             player=replay.names[seat], deck_size=len(deck), deck_sha256=deck_hash,
                             audit_status=proof.get('status', 'not_audited'),
                             audit_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest() if proof else None,
                             own_turns=[t for t in turns if t['player']==seat],
                             going_second=bool(turns and turns[0]['player']!=seat),
                             first_own_turn_observed=any(t['player']==seat for t in turns),
                             terminal_outcome=('win' if final.get('winner')==seat else 'loss' if final.get('winner') in (0,1) else 'draw') if completed else 'unknown',
                             ftk=ftk, training_ready=False,
                             family_split_key=deck_hash,
                             admission_required=['actor_tensor_reconstruction','independent_replay','controlled_seat_mask','quality_review'],
                             prefix_usage='local imitation only; terminal outcome remains unknown' if not completed else 'terminal recording; strategic quality not implied'))
    status = collections.Counter(r['audit_status'] for r in rows)
    summary = dict(replays=len(parsed), seat_records=len(rows), audit_status=dict(status),
                   deck_sizes=dict(collections.Counter(r['deck_size'] for r in rows)),
                   verified_ftk=sum(r['ftk'] for r in rows),
                   non_ftk_complete=sum(r['audit_status']=='complete' and not r['ftk'] for r in rows),
                   observed_going_second=sum(r['going_second'] and r['first_own_turn_observed'] for r in rows),
                   observed_nonopening_own_turns=sum(sum(t['turn']>1 for t in r['own_turns']) for r in rows),
                   note='No sample is admitted to training by this inventory; unknown is not failure.')
    for name, value in [('catalog.json',rows),('parsed.json',parsed),('candidates.json',candidates),('summary.json',summary)]:
        (a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
