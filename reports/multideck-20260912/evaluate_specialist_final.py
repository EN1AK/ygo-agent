"""Sequential GPU evaluation after successful specialist continuation."""
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

root = Path('/root/ygo-agent-gpu-20260910/ygo-agent')
area = root / 'reports/multideck-20260912'
run = Path(sys.argv[1])
out = run / 'evaluation'
out.mkdir(exist_ok=False)
final = sorted((run / 'checkpoints').glob('*_step_000009994240.flax_model'))
if len(final) != 1:
    raise RuntimeError('Expected the completed cumulative 9,994,240-step checkpoint')
baseline = root.parent / 'training-runs/h200-k9vs-mixed-10m-20260911T020646Z/checkpoints/1789092408_step_000009994240.flax_model'
rows = []
for label, checkpoint in [('stage3', baseline), ('specialist', final[0])]:
    for deck in ['k9vs', 'AI_BlueEyes', 'AI_Salamangreat', 'AI_Swordsoul', 'AI_Labrynth']:
        for seat in [0, 1]:
            decks = ['k9vs', deck] if seat == 0 else [deck, 'k9vs']
            log = out / f'{label}-{deck}-{seat}.log'
            command = [sys.executable, '-u', str(root / 'scripts/eval.py'),
                       '--lang', 'chinese', '--deck', str(area / 'training-decks'),
                       '--deck1', decks[0], '--deck2', decks[1],
                       '--code-list-file', str(root / 'scripts/code_list.txt'),
                       '--player', str(seat), '--seed', '13', '--num-envs', '8',
                       '--num-episodes', '128', '--env-threads', '8',
                       '--bot-type', 'greedy', '--checkpoint', str(checkpoint)]
            with log.open('wb') as stream:
                proc = subprocess.Popen(command, cwd=root, env=os.environ.copy(),
                                        stdout=stream, stderr=stream, start_new_session=True)
                timed_out = False
                try:
                    proc.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            content = log.read_text(errors='replace')
            episodes = re.findall(r'^Episode \d+: length=([^,]+), reward=([^,]+), win=(\d+), win_reason=(-?\d+)', content, re.M)
            valid = proc.returncode == 0 and len(episodes) >= 128 and not any(
                error in content for error in ['Traceback', 'terminate called', 'Card not found'])
            row = dict(model=label, checkpoint=str(checkpoint), deck=deck, seat=seat,
                       seed=13, valid=valid, exit_code=proc.returncode, timed_out=timed_out,
                       completed_episodes=len(episodes), log=str(log), command=command)
            if valid:
                # A vector step can finish extra games; compare exactly the first 128.
                samples = episodes[:128]
                wins = sum(int(e[2]) for e in samples)
                n = len(samples)
                p = wins / n
                z = 1.96
                center = (p + z*z/(2*n))/(1+z*z/n)
                radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                row.update(games=n, wins=wins, non_wins=n-wins, win_rate=p,
                           wilson95=[center-radius, center+radius],
                           average_length=sum(float(e[0]) for e in samples)/n)
            rows.append(row)
            (out / 'results.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
    selected = [r for r in rows if r['model'] == label and r['valid']]
    print(f'{label}: {len(selected)}/10 valid matchups', flush=True)
(out / 'completed.json').write_text(json.dumps({'all_valid': all(r['valid'] for r in rows),
    'matrix_rows': len(rows), 'note': 'Greedy evaluation only; WindBot gates remain pending.'}, indent=2))
