"""Linux process supervision for the existing single-duel evaluator.

Keep this module independent of JAX/native imports: it must survive a worker's
native abort and still clean the WindBot process in the same process group.
"""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import threading


_EPISODE = re.compile(
    r'^Episode 1: length=(\d+), reward=([^,\s]+), win=([01]), win_reason=(-?\d+)'
    r'(?:, invalid_game=(\d+), termination_reason=(-?\d+), '
    r'episode_steps=(\d+), turn_count=(\d+))?'
    r'(?:, environment_index=(\d+))?$', re.M
)


def classify_result(exit_code, text):
    result = {'status': 'invalid', 'reason': 'worker_exit', 'episode': None}
    if exit_code != 0:
        return result
    matches = list(_EPISODE.finditer(text))
    if len(matches) != 1 or len(re.findall(r'^Episode ', text, re.M)) != 1:
        result['reason'] = 'missing_or_multiple_episodes'
        return result
    if not re.search(r'^len=[^\n]+, win_rate=[^\n]+$', text, re.M):
        result['reason'] = 'missing_summary'
        return result
    match = matches[0]
    if ((match[5] is not None and (int(match[5]) != 0 or int(match[6]) != 1))
            or (match[9] is not None and int(match[9]) != 0)):
        result['reason'] = 'invalid_episode'
        return result
    try:
        reward = float(match[2])
    except ValueError:
        reward = float('nan')
    if not math.isfinite(reward) or int(match[3]) != int(reward > 0):
        result['reason'] = 'invalid_episode'
        return result
    return {'status': 'valid', 'reason': None, 'episode': {
        'length': int(match[1]), 'reward': reward,
        'win': int(match[3]), 'win_reason': int(match[4]),
    }}


def _live_group(pgid):
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(') ', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                members.append(int(entry.name))
        except (OSError, IndexError, ValueError):
            continue
    return members


def _signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _cleanup_group(process):
    # WindBot inherits eval.py's group. SIGCONT lets a stopped child receive
    # SIGTERM; SIGKILL handles children that cannot exit gracefully.
    _signal_group(process.pid, signal.SIGTERM)
    _signal_group(process.pid, signal.SIGCONT)
    deadline = time.monotonic() + .5
    while _live_group(process.pid) and time.monotonic() < deadline:
        time.sleep(.02)
    if _live_group(process.pid):
        _signal_group(process.pid, signal.SIGKILL)
    process.wait(timeout=2)
    deadline = time.monotonic() + 2
    while _live_group(process.pid) and time.monotonic() < deadline:
        time.sleep(.02)
    return _live_group(process.pid)


@contextmanager
def _interrupt_handlers():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_attempt(command, directory, cwd, timeout, env=None, *, stop_event=None):
    """Run one worker and persist its result, including cleanup on native abort."""
    if sys.platform != 'linux':
        raise RuntimeError('WindBot process supervision currently requires Linux')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be a positive finite number')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    command = [str(arg) for arg in command]
    (directory / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    started = time.monotonic()
    process = None
    reason = None
    launch_error = None
    remaining = []
    try:
        with (directory / 'eval.log').open('wb') as log:
            try:
                process = subprocess.Popen(command, cwd=cwd, env=env,
                                           stdout=log, stderr=log,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
            except OSError as exc:
                reason, launch_error = 'launch_error', str(exc)
            if process is not None:
                (directory / 'worker.json').write_text(json.dumps({'pid': process.pid, 'pgid': process.pid}))
                deadline = time.monotonic() + timeout
                while True:
                    if stop_event is not None and stop_event.is_set():
                        reason = 'interrupted'
                        break
                    remaining_time = deadline - time.monotonic()
                    if remaining_time <= 0:
                        reason = 'timeout'
                        break
                    try:
                        process.wait(timeout=min(.1, remaining_time))
                        break
                    except subprocess.TimeoutExpired:
                        pass
    except KeyboardInterrupt:
        reason = 'interrupted'
    finally:
        if process is not None:
            remaining = _cleanup_group(process)
    exit_code = process.returncode if process is not None else None
    text = (directory / 'eval.log').read_text(encoding='utf-8', errors='replace')
    result = classify_result(exit_code, text)
    child_errors = []
    for log in (directory / 'windbot').glob('*.log'):
        if re.search(r'\bSystem\.[\w.]*Exception\b', log.read_text(encoding='utf-8', errors='replace')):
            child_errors.append(str(log))
    if child_errors:
        result.update(status='invalid', reason='windbot_exception', episode=None,
                      windbot_error_logs=child_errors)
    if reason or remaining:
        result.update(status='invalid', reason='cleanup_failed' if remaining else reason, episode=None)
    result.update(exit_code=exit_code, wall_seconds=time.monotonic() - started,
                  cleanup_complete=not remaining, remaining_pids=remaining,
                  log=str(directory / 'eval.log'))
    if launch_error:
        result['error'] = launch_error
    (directory / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def _parallel_attempts(jobs, directory, cwd, timeout, env, workers):
    """Bounded submission: stopping never launches the rest of the schedule."""
    stop = threading.Event()
    iterator = iter(enumerate(jobs, 1))
    pending = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit_one():
            item = next(iterator, None)
            if item is None or stop.is_set():
                return False
            index, job = item
            folder = directory / f'attempt-{index:04d}'
            (directory / f'schedule-{index:04d}.json').write_text(
                json.dumps({key: value for key, value in job.items() if key != 'command'}),
                encoding='utf-8')
            future = pool.submit(run_attempt, job['command'], folder, folder,
                                 timeout, env, stop_event=stop)
            pending[future] = (index, job)
            return True

        try:
            for _ in range(workers):
                if not submit_one():
                    break
            while pending:
                finished, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                batch = []
                for future in finished:
                    index, job = pending.pop(future)
                    result = future.result()
                    if not result['cleanup_complete'] or result['reason'] == 'interrupted':
                        stop.set()
                    batch.append((index, job, result))
                for item in sorted(batch, key=lambda item: item[0]):
                    yield item
                if not stop.is_set():
                    for _ in batch:
                        if not submit_one():
                            break
        finally:
            # Includes SIGINT/SIGTERM and exceptions in the consumer. Workers
            # poll this event and clean their own independent process groups.
            stop.set()
            for future in pending:
                future.result()


def run_batch(jobs, directory, cwd, timeout, env=None, *, isolate_workers=False,
              workers=1):
    """Execute a bounded schedule. Each invalid attempt consumes its slot."""
    if not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if workers > 1 and not isolate_workers:
        raise ValueError('parallel evaluation requires isolated worker directories')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    summary = {}
    interrupted = False
    started = time.monotonic()
    try:
        with _interrupt_handlers():
            if workers > 1:
                try:
                    for index, job, result in _parallel_attempts(
                            jobs, directory, cwd, timeout, env, workers):
                        result.update(attempt=index, metadata={
                            key: value for key, value in job.items() if key != 'command'})
                        results.append(result)
                        with (directory / 'results.jsonl').open('a', encoding='utf-8') as out:
                            out.write(json.dumps(result) + '\n')
                        print(json.dumps({'attempt': index, 'status': result['status'],
                                          'reason': result['reason']}), flush=True)
                except KeyboardInterrupt:
                    interrupted = True
                finally:
                    # Also retain every in-flight interrupted result; never
                    # treat manual cancellation as if these attempts vanished.
                    known = {r['attempt'] for r in results}
                    for path in sorted(directory.glob('attempt-*/result.json')):
                        index = int(path.parent.name.split('-')[-1])
                        if index not in known:
                            result = json.loads(path.read_text(encoding='utf-8'))
                            result['attempt'] = index
                            result['metadata'] = json.loads(
                                (directory / f'schedule-{index:04d}.json').read_text(encoding='utf-8'))
                            results.append(result)
                            with (directory / 'results.jsonl').open('a', encoding='utf-8') as out:
                                out.write(json.dumps(result) + '\n')
            else:
                for index, job in enumerate(jobs, 1):
                    attempt_dir = directory / f'attempt-{index:04d}'
                    worker_cwd = attempt_dir if isolate_workers else cwd
                    result = run_attempt(job['command'], attempt_dir, worker_cwd, timeout, env)
                    result['attempt'] = index
                    result['metadata'] = {key: value for key, value in job.items() if key != 'command'}
                    results.append(result)
                    with (directory / 'results.jsonl').open('a', encoding='utf-8') as out:
                        out.write(json.dumps(result) + '\n')
                    print(json.dumps({'attempt': index, 'status': result['status'], 'reason': result['reason']}), flush=True)
                    if not result['cleanup_complete'] or result['reason'] == 'interrupted':
                        break
    finally:
        valid = [r for r in results if r['status'] == 'valid']
        wins = sum(r['episode']['win'] for r in valid)
        summary = {'attempted': len(results), 'valid': len(valid),
                   'invalid': len(results) - len(valid), 'wins': wins,
                   'losses': len(valid) - wins,
                   'win_rate': wins / len(valid) if valid else None,
                   'workers': workers, 'wall_seconds': time.monotonic() - started,
                   'interrupted': interrupted or any(r['reason'] == 'interrupted' for r in results),
                   'cleanup_complete': all(r['cleanup_complete'] for r in results)}
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
