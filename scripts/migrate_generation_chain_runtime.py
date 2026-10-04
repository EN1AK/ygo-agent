"""Finish generation four, then resume full PPO state on an isolated runtime."""
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time

OLD = Path('/home/ygo/ygo-agent/training-runs/skystriker-generations-20m-v2-20261004')
NEW = Path('/home/ygo/ygo-agent/training-runs/skystriker-generations-20m-chain-v3-20261004')
COMPAT = Path('/home/ygo/chain-compat-b81516e-20261004')
NATIVE = '7bc098653b5336b159215168a4680b5f77b0015939bae6cf061e732016e79358'
NATIVE_REL = 'ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so'


def clone_runtime(source, target):
    target.mkdir()
    for item in source.iterdir():
        dst = target / item.name
        if item.name == 'ygoenv':
            shutil.copytree(item, dst)
        elif item.name == 'scripts':
            dst.mkdir()
            for child in item.iterdir():
                out = dst / child.name
                if child.is_dir():
                    out.symlink_to(child, target_is_directory=True)
                else:
                    shutil.copy2(child, out)
        elif item.name not in ('.git', '.xmake', 'build'):
            dst.symlink_to(item, target_is_directory=item.is_dir())
    shutil.copy2(COMPAT / NATIVE_REL, target / NATIVE_REL)


def process_state(pid):
    path = Path(f'/proc/{pid}/stat')
    if not path.exists():
        return None
    fields = path.read_text().rsplit(')', 1)[1].split()
    return fields[0], int(fields[49])


def finish_training(m, run, record):
    cp = run / 'checkpoints' / f"{104205000 + record['index']}_step_{record['target_step']:012d}.flax_model"
    proof = m.verify(cp)
    statefile = Path(str(cp) + '.ppo_state')
    meta = json.loads(Path(str(statefile) + '.json').read_text())
    assert meta['actor_sha256'] == proof['sha256']
    assert meta['state_sha256'] == m.sha(statefile)
    assert meta['optimizer_step'] == record['index'] * (m.BLOCK // m.BATCH) * 64
    log = (run / 'train.log').read_text()
    finite = re.findall(r'optimizer_finite=\[([^]]+)\], notfinite_count=\[([^]]+)\], total_notfinite=\[([^]]+)\]', log)
    assert len(finite) == m.BLOCK // m.BATCH
    assert all(a.strip() == 'True' and b.strip() == c.strip() == '0' for a, b, c in finite)
    assert 'ppo_optimizer_restored=' in log and '"actor_exact": true' in log
    assert '"roundtrip_exact": true' in log
    record.update(status='trained', endpoint=proof, optimizer_state=meta, optimizer_updates=len(finite))
    m.state['latest_checkpoint'] = proof
    m.save()
    return cp, proof


def main():
    spec = importlib.util.spec_from_file_location('generation_schedule', OLD / 'launcher.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    snapshot = json.loads((OLD / 'schedule.json').read_text())
    assert snapshot['status'] == 'training' and snapshot['generations'][-1]['index'] == 4
    assert snapshot['active']['pid'] == 284147
    assert process_state(244489)[0] == 'T', 'old scheduler must be suspended at authorized handoff'
    assert m.sha(COMPAT / NATIVE_REL) == NATIVE
    NEW.mkdir(exist_ok=False)
    shutil.copy2(__file__, NEW / 'migration-launcher.py')
    m.write(NEW / 'handoff-start.json', snapshot)
    m.write(NEW / 'migration-status.json', {'status': 'waiting_generation_004', 'pid': os.getpid()})
    # Existing child is allowed to finish naturally; no training is interrupted.
    while True:
        status = process_state(284147)
        assert status is not None, 'old child disappeared before exit evidence was captured'
        if status[0] == 'Z':
            assert status[1] == 0, f'training exit status {status[1]}'
            break
        if time.time() - (OLD / 'generation-004/train.log').stat().st_mtime > 1800:
            raise RuntimeError('old training stalled; keep evidence and do not migrate')
        time.sleep(5)
    m.write(NEW / 'old-training-exit.json', {'pid': 284147, 'exit_code': 0})
    # Wake the suspended scheduler with a pending graceful stop. Its child has
    # already exited, so its cleanup cannot interrupt a training update.
    os.kill(244489, signal.SIGTERM)
    os.kill(244489, signal.SIGCONT)
    for _ in range(60):
        status = process_state(244489)
        if status is None or status[0] == 'Z':
            break
        time.sleep(1)
    else:
        raise RuntimeError('old scheduler did not stop')
    shutil.copy2(OLD / 'schedule.json', NEW / 'old-scheduler-stop.json')
    m.state = snapshot
    m.state['active'] = None
    m.state['last_process_exit'] = 0
    m.state['handoff_controller_pid'] = os.getpid()
    run = OLD / 'generation-004'
    record = m.state['generations'][-1]
    cp, proof = finish_training(m, run, record)
    result = m.evaluate(run, cp, Path(record['parent']['checkpoint']), 4, 'screen', 8)
    record.update(status='evaluated', transition='user requested corrected runtime for next 20M')
    m.write(run / 'run-manifest.json', record)
    m.state.update(status='runtime_transition_completed', active=None,
                   successor_root=str(NEW), finished=time.time())
    m.save()
    m.write(OLD / 'runtime-handoff.json', {'successor_root': str(NEW), 'checkpoint': proof,
            'old_generation_004_screen': result, 'intentional_scheduler_stop': True,
            'old_patience_not_carried': True})
    original_repo, original_eval = m.REPO, m.EVAL
    clone_runtime(original_repo, NEW / 'training-runtime')
    clone_runtime(original_eval, NEW / 'evaluation-runtime')
    m.REPO, m.EVAL = NEW / 'training-runtime', NEW / 'evaluation-runtime'
    shim = m.EVAL / 'scripts/eval_mixed_match.py'
    source = shim.read_text()
    assert source.count(str(OLD)) == 1
    shim.write_text(source.replace(str(OLD), str(NEW)))
    for name in ('cleanba_with_ppo_state.py', 'ppo_state_io.py'):
        shutil.copy2(OLD / name, NEW / name)
        assert m.sha(OLD / name) == m.sha(NEW / name)
    for root in (m.REPO, m.EVAL):
        assert m.sha(root / NATIVE_REL) == NATIVE
    m.ROOT, m.NATIVE = NEW, NATIVE
    m.state = dict(snapshot)
    m.state.update(status='transition_ready', started=time.time(), active=None,
                   generations=[], checkpoints=[], no_gain_streak=0, native_sha256=NATIVE,
                   parent=proof, previous_run=str(OLD), controller_pid=os.getpid(),
                   event_version='chain-source-by-link-v2', selection_version='legacy-prompt-field-v1',
                   runtime_transition='generation 005 onward; both evaluation models use corrected runtime',
                   optimizer='restore full TrainState and learner RNG from generation 004',
                   generation_number_offset=4, old_patience_not_carried=True)
    m.save()
    parent, offset, index = cp, proof['step'], 4
    signal.signal(signal.SIGTERM, m.interrupt)
    while True:
        if (NEW / 'STOP').exists():
            raise InterruptedError('operator STOP')
        index += 1
        run = NEW / f'generation-{index:03d}'
        run.mkdir()
        command = m.training_command(run, parent, offset, 104205000 + index)
        record = dict(index=index, status='training', parent=m.verify(parent),
                      target_step=offset + m.BLOCK, command=command, native_sha256=NATIVE,
                      event_version='chain-source-by-link-v2', selection_version='legacy-prompt-field-v1')
        m.state['generations'].append(record)
        m.state['status'] = 'training'
        m.save()
        m.write(run / 'run-manifest.json', record)
        m.execute(command, m.REPO, run / 'train.log', training=True, checkpoints=run / 'checkpoints')
        cp, proof = finish_training(m, run, record)
        result = m.evaluate(run, cp, parent, index, 'screen', 8)
        m.state['no_gain_streak'] = 0 if result['clear_gain'] else m.state['no_gain_streak'] + 1
        record['status'] = 'evaluated'
        if m.state['no_gain_streak'] >= 3:
            result = m.evaluate(run, cp, parent, index, 'confirmation', 16)
            if result['excludes_five_point_gain']:
                record['status'] = 'plateau_confirmed'
                m.state.update(status='plateau_stopped', finished=time.time(),
                    stop_reason='three corrected-runtime screens without clear gain; confirmation upper95 below 55%')
                m.write(run / 'run-manifest.json', record)
                m.save()
                (NEW / 'completed.txt').write_text(m.state['stop_reason'] + '\n')
                return
            m.state['no_gain_streak'] = 0
            record['confirmation_decision'] = 'gain or uncertainty remains; continue'
        m.write(run / 'run-manifest.json', record)
        m.save()
        parent, offset = cp, proof['step']


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if NEW.exists():
            (NEW / 'migration-failed.json').write_text(json.dumps({'error': repr(error), 'time': time.time()}))
        raise
