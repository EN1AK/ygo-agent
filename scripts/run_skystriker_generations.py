"""Home-only PPO continuation with serial, paired parent-generation evaluation.

Uses the immutable existing runtime; never changes serving models or native code.
Stop criteria and all seeds are recorded before the first optimizer update.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
import flax.serialization
import numpy as np

BASE = Path('/home/ygo/ygo-agent')
ROOT = BASE / 'training-runs/skystriker-generations-20m-20261004'
RELEASE = BASE / 'dist/runtime-releases/skystriker-place-forward-505b855-20261003'
REPO = RELEASE / 'source'
EVAL = BASE / 'training-runs/q2m-source-65443f7'
TEMPLATE = BASE / 'training-runs/skystriker-place-forward-107m-to121m-fast-4x32-20261003/run-manifest.json'
PARENT = BASE / 'training-runs/skystriker-121m-eight-hours-20261004/segment-04/checkpoints/104202614_step_000186277888.flax_model'
PARENT_SHA = 'dc29c11254c0418a4b43cc29083db3d05e17b5c8e3dacd5c27b17d63c8670721'
NATIVE = 'a2dbd2604fec0e01ad722d887c639c00ed4c5aa74c912bd8f37d69f57a815486'
BATCH = 8192
BLOCK = math.ceil(20_000_000 / BATCH) * BATCH
SAVE_UPDATES = math.ceil(5_000_000 / BATCH)
TRAINER_SHA = 'a26907a94d780884ac75e17a302c8bdf782f2bc57f99121db1396b0f995e65f0'

# This small sidecar module is written into the exclusive run directory. The
# immutable trainer receives only save/restore hooks; its PPO update is unchanged.
PPO_STATE_IO = r'''
import hashlib
import json
import os
from pathlib import Path
import flax.serialization
import jax
import numpy as np

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def context(args):
    assert not args.anneal_lr and not args.distributed
    fields = ('learning_rate', 'max_grad_norm', 'update_epochs', 'num_minibatches',
              'gamma', 'gae_lambda', 'value', 'ppo_clip', 'clip_coef', 'dual_clip_coef',
              'norm_adv', 'ent_coef', 'vf_coef', 'vloss_clip', 'spo_kld_max',
              'logits_threshold', 'sep_value', 'upgo', 'bfloat16', 'num_steps',
              'collect_steps', 'segment_length', 'burn_in_steps', 'max_step_loss',
              'observation_schema', 'max_options', 'n_history_actions',
              'n_public_events', 'max_group_references', 'local_num_envs',
              'num_actor_threads', 'train_opponent', 'mixed_self_actors',
              'mixed_history_actors', 'mixed_bot_actors', 'historical_checkpoints')
    result = {name: getattr(args, name) for name in fields}
    result['trainer_sha256'] = digest(Path(os.environ['GENERATION_TRAINER']))
    result['helper_sha256'] = digest(Path(__file__))
    return result

def same(left, right):
    la, lt = jax.tree_util.tree_flatten(left)
    ra, rt = jax.tree_util.tree_flatten(right)
    assert lt == rt and len(la) == len(ra)
    for a, b in zip(la, ra):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)

def atomic(path, data):
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)

def restore_ppo(args, state, keys, devices):
    checkpoint = Path(args.checkpoint)
    path = Path(str(checkpoint) + '.ppo_state')
    if not path.exists():
        assert digest(checkpoint) == os.environ['GENERATION_INITIAL_ACTOR_SHA'], 'missing optimizer state for later generation'
        print('ppo_optimizer_initialization=fresh_from_legacy186m', flush=True)
        return state, keys
    metadata = json.loads(Path(str(path) + '.json').read_text())
    assert metadata['schema'] == 'ygo-ppo-training-state-v1'
    assert metadata['actor_sha256'] == digest(checkpoint)
    assert metadata['state_sha256'] == digest(path)
    assert metadata['context'] == context(args), 'PPO resume context mismatch'
    template = {'state': state, 'learner_keys': np.asarray(keys)}
    restored = flax.serialization.from_bytes(template, path.read_bytes())
    same(restored['state'].params, state.params)
    same(restored['state'].batch_stats, state.batch_stats)
    assert int(restored['state'].step) == metadata['optimizer_step']
    assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree_util.tree_leaves(restored))
    assert restored['learner_keys'].shape == np.asarray(keys).shape
    keys = jax.device_put_sharded(list(restored['learner_keys']), devices)
    print('ppo_optimizer_restored=' + json.dumps({'optimizer_step': metadata['optimizer_step'],
          'state_sha256': metadata['state_sha256'], 'actor_exact': True}), flush=True)
    return restored['state'], keys

def save_ppo(args, state, keys, checkpoint):
    checkpoint = Path(checkpoint)
    payload = {'state': state, 'learner_keys': np.asarray(keys)}
    assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree_util.tree_leaves(payload))
    data = flax.serialization.to_bytes(payload)
    path = Path(str(checkpoint) + '.ppo_state')
    atomic(path, data)
    restored = flax.serialization.from_bytes(payload, path.read_bytes())
    same(payload, restored)
    metadata = dict(schema='ygo-ppo-training-state-v1', actor_sha256=digest(checkpoint),
                    state_sha256=digest(path), optimizer_step=int(state.step), context=context(args),
                    scope='TrainState and learner RNG; environment, actor RNG and live rollout are reset')
    atomic(Path(str(path) + '.json'), (json.dumps(metadata, indent=2) + '\n').encode())
    print('ppo_optimizer_saved=' + json.dumps({'optimizer_step': metadata['optimizer_step'],
          'state_sha256': metadata['state_sha256'], 'roundtrip_exact': True}), flush=True)
'''


def prepare_trainer():
    original = REPO / 'scripts/cleanba.py'
    assert sha(original) == TRAINER_SHA
    source = original.read_text()
    restore_hook = '    agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)'
    save_hook = '            ckpt_maneger.save(unreplicated_params, ckpt_name)'
    assert source.count(restore_hook) == source.count(save_hook) == 1
    source = source.replace(restore_hook,
        '    from ppo_state_io import restore_ppo, save_ppo\n'
        '    agent_state, learner_keys = restore_ppo(args, agent_state, learner_keys, learner_devices)\n' + restore_hook)
    source = source.replace(save_hook, save_hook + '\n'
        '            save_ppo(args, flax.jax_utils.unreplicate(agent_state), learner_keys, Path(args.ckpt_dir) / ckpt_name)')
    (ROOT / 'cleanba_with_ppo_state.py').write_text(source)
    (ROOT / 'ppo_state_io.py').write_text(PPO_STATE_IO)
    # Compilation catches malformed generated source before an expensive launch.
    compile(source, str(ROOT / 'cleanba_with_ppo_state.py'), 'exec')
    compile(PPO_STATE_IO, str(ROOT / 'ppo_state_io.py'), 'exec')


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leaves(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from leaves(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            yield from leaves(value)
    else:
        yield np.asarray(tree)


def verify(path):
    digest = sha(path)
    assert json.loads(Path(str(path) + '.metadata.json').read_text())['checkpoint_sha256'] == digest
    arrays = list(leaves(flax.serialization.msgpack_restore(path.read_bytes())))
    assert len(arrays) == 165 and all(np.isfinite(a).all() for a in arrays)
    return dict(checkpoint=str(path), sha256=digest, arrays=165,
                step=int(path.stem.split('_')[-1]), finite=True)


def save():
    state['updated'] = time.time()
    write(ROOT / 'schedule.json', state)


def interrupt(signum, frame):
    raise InterruptedError(f'received signal {signum}')


def execute(command, repo, log, *, training=False, checkpoints=None):
    env = dict(os.environ, JAX_PLATFORMS='cuda', CUDA_VISIBLE_DEVICES='0',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONFAULTHANDLER='1',
               PYTHONPATH=f'{ROOT}:{repo}/scripts:{repo}/ygoenv:{repo}',
               GENERATION_TRAINER=str(ROOT / 'cleanba_with_ppo_state.py'),
               GENERATION_INITIAL_ACTOR_SHA=PARENT_SHA,
               LD_PRELOAD=str(RELEASE / 'libcompat_glibc.so'))
    with log.open('x') as stream:
        proc = subprocess.Popen(command, cwd=repo, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        state['active'] = dict(pid=proc.pid, command=command, log=str(log))
        save()
        started = time.time()
        try:
            while proc.poll() is None:
                time.sleep(5)
                if (ROOT / 'STOP').exists():
                    raise InterruptedError('operator STOP file')
                if time.time() - max(started, log.stat().st_mtime) > 1800:
                    raise RuntimeError('child produced no output for 30 minutes')
                if not training and time.time() - started > 1800:
                    raise RuntimeError('evaluation/config wall limit exceeded')
                if training:
                    with log.open('rb') as reader:
                        reader.seek(max(0, log.stat().st_size - 32000))
                        tail = reader.read().decode(errors='replace')
                    if re.search(r'optimizer_finite=\[\s*False|total_notfinite=\[\s*[1-9]|NONFINITE_', tail):
                        raise RuntimeError('nonfinite training metrics')
                if checkpoints:
                    seen = {item['checkpoint'] for item in state['checkpoints']}
                    for cp in sorted(checkpoints.glob('*.flax_model')):
                        if str(cp) not in seen and Path(str(cp) + '.metadata.json').exists():
                            proof = verify(cp)
                            state['checkpoints'].append(proof)
                            state['latest_checkpoint'] = proof
                            save()
            if proc.returncode != 0:
                raise RuntimeError(f'child exit {proc.returncode}: {log}')
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
            state['last_process_exit'] = proc.returncode
            state['active'] = None
            save()


def training_command(run, parent, offset, seed, config=False):
    cmd = list(json.loads(TEMPLATE.read_text())['command'])
    cmd[cmd.index('scripts/cleanba.py')] = str(ROOT / 'cleanba_with_ppo_state.py')
    settings = {'--seed': seed, '--ckpt-dir': run / 'checkpoints',
                '--run-name': run.name + '__' + str(seed), '--checkpoint': parent,
                '--tb-offset': offset, '--total-timesteps': BLOCK,
                '--save-interval': SAVE_UPDATES, '--max-checkpoints': 10}
    for key, value in settings.items():
        cmd[cmd.index(key) + 1] = str(value)
    if config:
        cmd.append('--config-only')
    return cmd


def summarize(rows):
    # Both seats of each (seed, environment) remain together in every resample.
    valid = [row for row in rows if row['outcome'] != 'invalid']
    counts = {name: sum(r['outcome'] == name for r in rows)
              for name in ('win', 'loss', 'draw', 'invalid')}
    result = dict(attempts=len(rows), counts=counts,
                  win_rate=counts['win'] / len(valid) if valid else None)
    if counts['invalid']:
        return result
    groups = {}
    for row in rows:
        groups.setdefault((row['seed'], row['environment_index']), []).append(row)
    assert all(len(pair) == 2 and {r['candidate_seat'] for r in pair} == {0, 1}
               for pair in groups.values())
    scores = np.array([np.mean([{'win': 1., 'draw': .5, 'loss': 0.}[r['outcome']]
                               for r in pair]) for pair in groups.values()])
    rng = np.random.default_rng(2026100407)
    boot = np.concatenate([rng.choice(scores, (250, len(scores)), replace=True).mean(1)
                           for _ in range(40)])
    interval = np.quantile(boot, [.025, .975]).tolist()
    score = float(scores.mean())
    result.update(score_rate=score, paired_bootstrap95=interval, paired_deals=len(scores),
                  clear_gain=bool(score >= .55 and interval[0] > .5),
                  excludes_five_point_gain=bool(interval[1] < .55),
                  by_seat={str(seat): {'wins': sum(r['outcome'] == 'win' for r in rows if r['candidate_seat'] == seat),
                                      'attempts': sum(r['candidate_seat'] == seat for r in rows)}
                           for seat in (0, 1)})
    return result


def evaluate(run, cp, parent, index, kind, seed_count):
    directory = run / kind
    directory.mkdir()
    # Disjoint seed blocks per generation and confirmation; never used in training.
    seed_start = 61050000 + index * 1000 + (100 if kind == 'confirmation' else 0)
    rows = []
    record = dict(status='running', candidate=verify(cp), opponent=verify(parent),
                  seed_start=seed_start, seed_count=seed_count, reports=[])
    state['status'] = 'evaluating'
    state['generations'][-1][kind] = record
    save()
    for seed in range(seed_start, seed_start + seed_count):
        for seat in (0, 1):
            output = directory / f'{seed}-seat{seat}.json'
            command = [sys.executable, '-u', str(EVAL / 'scripts/eval_mixed_match.py'),
                       '--checkpoint', str(cp), '--opponent-checkpoint', str(parent),
                       '--output', str(output), '--seed', str(seed), '--num-episodes', '32',
                       '--candidate-seat', str(seat), '--env-threads', '4',
                       '--deck', str(RELEASE / 'SkyStriker.ydk'),
                       '--code-list-file', str(RELEASE / 'code_list.crlf.txt'),
                       '--semantic-asset-dir', str(RELEASE / 'semantics'),
                       '--observation-schema', 'structured-lite-v1',
                       '--checkpoint-schema', 'structured-lite-v1', '--opponent-schema', 'structured-lite-v1',
                       '--max-options', '128', '--max-steps', '1000',
                       '--n-history-actions', '32', '--n-public-events', '32', '--max-group-references', '8']
            execute(command, EVAL, output.with_suffix('.log'))
            report = json.loads(output.read_text())
            assert report['checkpoint_sha256'] == record['candidate']['sha256']
            assert report['opponent_checkpoint_sha256'] == record['opponent']['sha256']
            assert report['seed'] == seed and report['candidate_seat'] == seat
            assert report['attempts'] == 32 and len(report['episodes']) == 32
            assert {r['environment_index'] for r in report['episodes']} == set(range(32))
            assert all(r['candidate_seat'] == seat for r in report['episodes'])
            rows.extend(dict(row, seed=seed) for row in report['episodes'])
            record['reports'].append(str(output))
            save()
            if report['invalid_games']:
                raise RuntimeError('invalid evaluation games; do not count as plateau or replace attempts')
    summary = summarize(rows)
    write(directory / 'episodes.json', rows)
    write(directory / 'summary.json', summary)
    record.update(status='completed', summary=summary)
    save()
    return summary


def main():
    global state
    ROOT.mkdir(exist_ok=False)
    shutil.copy2(__file__, ROOT / 'launcher.py')
    state = dict(status='preflight', started=time.time(), parent=verify(PARENT),
                 launcher_sha256=sha(Path(__file__)), training_source='505b855d9e1e0a2363a3d81cf49d6097f7539555',
                 native_sha256=NATIVE, additional_steps_per_generation=BLOCK,
                 save_every_steps=BATCH * SAVE_UPDATES, optimizer='first generation fresh Adam; later generations restore full TrainState and learner RNG',
                 training_opponents=['self', 'self', 'history100m', 'bot'],
                 evaluation_opponent='immediate previous 20M endpoint; first parent 186277888',
                 protocol=dict(screen_games=512, confirmation_games=1024,
                               meaningful_score_rate=.55, clear_gain='score >= .55 and paired bootstrap lower95 > .50',
                               plateau='3 consecutive screens without clear gain AND fresh confirmation upper95 < .55',
                               inconclusive_confirmation='continue training; restart patience counter',
                               draws='half point in score; actual win rate reported separately',
                               invalid='stop for investigation, no replacements',
                               caveat='operational sequential stopping rule; not a multiple-testing-adjusted global convergence claim'),
                 generations=[], checkpoints=[], no_gain_streak=0, active=None)
    save()
    signal.signal(signal.SIGTERM, interrupt)
    try:
        assert state['parent']['sha256'] == PARENT_SHA
        context = json.loads((RELEASE / 'context.json').read_text())
        for repo in (REPO, EVAL):
            assert sha(repo / 'ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so') == NATIVE
        assert sha(REPO / 'scripts/script/procedure.lua') == context['procedure_sha256']
        assert sha(RELEASE / 'SkyStriker.ydk') == context['input_deck_sha256']
        assert sha(RELEASE / 'code_list.crlf.txt') == context['code_list_sha256']
        state['eval_script_sha256'] = sha(EVAL / 'scripts/eval_mixed_match.py')
        state['training_script_sha256'] = sha(REPO / 'scripts/cleanba.py')
        prepare_trainer()
        state['generated_trainer_sha256'] = sha(ROOT / 'cleanba_with_ppo_state.py')
        state['ppo_state_helper_sha256'] = sha(ROOT / 'ppo_state_io.py')
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                args = (proc / 'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            if args and b'python' in args[0] and any(Path(a.decode(errors='replace')).name in
                    ('cleanba.py', 'eval_mixed_match.py', 'eval_windbot.py') for a in args[1:]):
                raise RuntimeError('another training/evaluation job active: ' + proc.name)
        config = ROOT / 'config'
        config.mkdir()
        execute(training_command(config, PARENT, state['parent']['step'], 104205000, True), REPO, config / 'config.log')
        parent, offset, index = PARENT, state['parent']['step'], 0
        while True:
            if (ROOT / 'STOP').exists():
                raise InterruptedError('operator STOP file')
            index += 1
            run = ROOT / f'generation-{index:03d}'
            run.mkdir()
            seed = 104205000 + index
            command = training_command(run, parent, offset, seed)
            record = dict(index=index, status='training', parent=verify(parent),
                          target_step=offset + BLOCK, command=command)
            state['generations'].append(record)
            state['status'] = 'training'
            save()
            write(run / 'run-manifest.json', record)
            execute(command, REPO, run / 'train.log', training=True, checkpoints=run / 'checkpoints')
            cp = run / 'checkpoints' / f'{seed}_step_{offset + BLOCK:012d}.flax_model'
            proof = verify(cp)
            optimizer_path = Path(str(cp) + '.ppo_state')
            optimizer_metadata = json.loads(Path(str(optimizer_path) + '.json').read_text())
            assert optimizer_metadata['actor_sha256'] == proof['sha256']
            assert optimizer_metadata['state_sha256'] == sha(optimizer_path)
            expected_optimizer_step = index * (BLOCK // BATCH) * 64
            assert optimizer_metadata['optimizer_step'] == expected_optimizer_step
            log_text = (run / 'train.log').read_text()
            if index > 1:
                assert 'ppo_optimizer_restored=' in log_text
            record['optimizer_state'] = optimizer_metadata
            finite = re.findall(r'optimizer_finite=\[([^]]+)\], notfinite_count=\[([^]]+)\], total_notfinite=\[([^]]+)\]', (run / 'train.log').read_text())
            assert len(finite) == BLOCK // BATCH
            assert all(a.strip() == 'True' and b.strip() == c.strip() == '0' for a, b, c in finite)
            record.update(status='trained', endpoint=proof, optimizer_updates=len(finite))
            state['latest_checkpoint'] = proof
            save()
            result = evaluate(run, cp, parent, index, 'screen', 8)
            state['no_gain_streak'] = 0 if result['clear_gain'] else state['no_gain_streak'] + 1
            record['status'] = 'evaluated'
            if state['no_gain_streak'] >= 3:
                confirmation = evaluate(run, cp, parent, index, 'confirmation', 16)
                if confirmation['excludes_five_point_gain']:
                    state.update(status='plateau_stopped', stop_reason='three screens without clear gain; independent confirmation upper95 below 55% score')
                    record['status'] = 'plateau_confirmed'
                    save()
                    (ROOT / 'completed.txt').write_text(state['stop_reason'] + '\n')
                    break
                state['no_gain_streak'] = 0
                record['confirmation_decision'] = 'gain or uncertainty remains; continue'
            write(run / 'run-manifest.json', record)
            save()
            parent, offset = cp, offset + BLOCK
    except BaseException as error:
        state.update(status='stopped' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'failed', error=repr(error))
        (ROOT / 'failed.txt').write_text(repr(error) + '\n')
        raise
    finally:
        state['finished'] = time.time()
        save()


if __name__ == '__main__':
    main()
