"""Certify the exact recorded prefix at the existing actor process fuse.

This does not extend the fuse, replay policy inference, infer a winner, or
accept an incomplete response/event/state comparison.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ygoai.rl.human_replay import comparable_game_events

ACTOR_ERROR = 'exercise process budget exhausted'
CORE_ERROR = 'core process budget exhausted'
PROCESS_LIMIT = 4096


def trajectory_digest(record):
    keys = ('id', 'definition', 'seed', 'candidate_seat', 'candidate_sha256', 'opponent_sha256',
        'checkpoint_sha256', 'core_seed', 'decisions', 'events', 'final', 'core_responses', 'error',
        'core_process_cutoff', 'native_sha256')
    payload = {k: record.get(k) for k in keys}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def eligible(record):
    return (record.get('error') == ACTOR_ERROR and 'final' in record and
        not record.get('capture_error') and record['final'].get('winner') is None and
        record.get('candidate_result') is None and record.get('status') in ('invalid', 'budget_exhausted'))


def finish_prefix(core, record):
    """Only an exhausted tail after every response can become a prefix proof."""
    try:
        core.next_prompt(limit=PROCESS_LIMIT)
    except TimeoutError as error:
        if (str(error) != CORE_ERROR or core.calls != PROCESS_LIMIT or
                len(core.decisions) != len(record['core_responses'])):
            raise ValueError('cutoff did not occur after all recorded responses at the fixed fuse') from error
    else:
        raise ValueError('independent core did not reproduce the recorded process cutoff')
    if comparable_game_events(core.events) != comparable_game_events(record['events']):
        raise ValueError('budget prefix event mismatch')
    if core.state() != record['final']:
        raise ValueError('budget prefix full state mismatch')
    return dict(passed=True, validated_variants=1, ambiguity_nodes=0, semantic_chain_index_remaps=0,
        events=len(core.events), recorded_events=len(record['events']), core_process_calls=core.calls, exact_recorded_prefix=True,
        natural_outcome_certified=False)


def audit_prefix(record, source, database, scripts, library):
    request = dict(record=record, source=str(source), database=str(database), scripts=str(scripts),
        library=library, max_variants=8)
    worker = Path(__file__).with_name('audit_engine_budget_prefix.py')
    process = subprocess.Popen([sys.executable, '-I', str(worker)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
        env=dict(os.environ, JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
    try:
        out, err = process.communicate(json.dumps(request), timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate()
        raise ValueError('independent budget-prefix audit time budget exhausted')
    lines = [line[len('AUDIT_RESULT '):] for line in out.splitlines() if line.startswith('AUDIT_RESULT ')]
    if process.returncode or len(lines) != 1:
        raise ValueError('budget-prefix audit worker failed: ' + err[-1500:])
    proof = json.loads(lines[0])
    if not proof.get('passed') or proof.get('trajectory_sha256') != trajectory_digest(record):
        raise ValueError(proof.get('error', 'budget-prefix proof is not bound to this trajectory'))
    return proof


def adjudicate(record, source, database, scripts, library, native_sha256):
    if not eligible(record):
        raise ValueError('record is not a captured native process-budget cutoff')
    if record.get('native_sha256', native_sha256) != native_sha256:
        raise ValueError('native process-cutoff record belongs to another runtime')
    revised = copy.deepcopy(record)
    revised.update(status='budget_exhausted', native_sha256=native_sha256,
        core_process_cutoff=dict(limit=PROCESS_LIMIT, native_error=ACTOR_ERROR,
            original_status=record['status'], original_audit=record.get('independent_audit'),
            classification='resource-bounded recorded prefix; no natural outcome'))
    revised['independent_audit'] = audit_prefix(revised, source, database, scripts, library)
    return revised
