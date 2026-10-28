"""Freeze and execute matched ordinary PPO/full-history rehearsal continuations.

The terminal round-two repair checkpoint is fixed before unseen inference.
Discard smoke updates; both formal arms restart from that exact checkpoint.
The compatible corrected runtime is a separately measured study revision.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from contracts import ledger, read, sha, write
from compare import audit_endpoint

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from scripts.run_elfnote_formal import child, command, verify_runtime, verify_training
from ygoai.rl.core_audit import validate_curriculum_binding


def protected(paths):
    return {str(p.resolve()): sha(p) for p in paths}


def freeze(a):
    run, overlay = a.run.resolve(), a.runtime_root.resolve()
    release = overlay / 'formal-runtime'
    manifest = verify_runtime(release)
    prep = read(release / 'preparation.json')
    admission = read(overlay / 'formal-protocol/admission.json')
    reload = read(overlay / 'gpu-reload.json')
    if not admission['passed'] or admission['runtime_manifest_sha256'] != sha(release / 'frozen-manifest.json'):
        raise ValueError('current corrected runtime lacks protocol admission')
    course = overlay / 'course'
    plan = read(course / 'plan.json')
    exam = overlay / 'independent-exam'
    exam_plan = read(exam / 'plan.json')
    validate_curriculum_binding(plan)
    validate_curriculum_binding(exam_plan)
    if not plan['overlay_event_reencoding']['all_rows_retained'] or not exam_plan['overlay_event_reencoding']['all_rows_retained']:
        raise ValueError('all source rows must survive corrected-runtime replay')
    if prep['diagnostic_sha256'] != plan['native_sha256'] or len(prep['decks']) != 4:
        raise ValueError('corrected normal/diagnostic runtime mismatch')
    endpoints = []
    for round_name in ('round1', 'round2'):
        for arm in ('control', 'repair'):
            folder = run / (round_name + '-training') / arm
            completion, _, _ = audit_endpoint(folder, 512)
            endpoints.append(dict(name=round_name + '-' + arm, checkpoint=str(folder / 'endpoint/model.flax_model'),
                checkpoint_sha256=completion['checkpoint_sha256']))
    checkpoint = run / 'round2-training/repair/endpoint/model.flax_model'
    if sha(checkpoint) != '28ec8cef58afc9cca24c1e1273e06ff4c5dc87b301a633702ee8905a9e8a033d':
        raise ValueError('the fixed round-two terminal parent changed')
    if not reload['passed'] or reload['checkpoint_sha256'] != sha(checkpoint) or reload['native_sha256'] != manifest['native_sha256']:
        raise ValueError('actual current-runtime GPU reload required')
    history = Path(read(run / 'protocol.json')['initial_checkpoint'])
    if sha(history) != '3103f344eb32f8745316ae06df3f8f9c118c80f7d601e11830796d3cfb2fd466':
        raise ValueError('historical opponent must be the original valid filtered-course parent')
    priority = set()
    for name in ('round1', 'round2'):
        priority.update(r['id'] for r in read(run / (name + '-training/comparison.json'))['original_success_regressions'] if r['split'] == 'fit')
    groups = defaultdict(list)
    for row in plan['rows']:
        if row['split'] == 'fit':
            groups[row['group']].append(row)
    selected = []
    for group, rows in sorted(groups.items()):
        order = sorted(rows, key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
        chosen = [r for r in order if r['id'] in priority]
        chosen += [r for r in order if r['id'] not in priority][:max(0, 4 - len(chosen))]
        selected.extend(chosen)
    if not priority <= {r['id'] for r in selected}:
        raise ValueError('fit regressions omitted from rehearsal')
    rows = []
    counts = Counter(r['group'] for r in selected)
    maximum = 0
    for item in selected:
        path = Path(item['path'])
        if sha(path) != item['record_sha256'] or sha(path.with_suffix('.npz')) != item['observations_sha256']:
            raise ValueError('corrected history changed')
        r = read(path)
        seat = r['definition']['controlled']
        own = [i for i, d in enumerate(r['decisions']) if d['player'] == seat]
        ce = [i for i in own if r['decisions'][i]['teaching'] and len(r['decisions'][i]['menu']) > 1]
        if not ce or r.get('full_legal_menu') is not False:
            raise ValueError('a selected fit history lacks valid teaching suffix')
        maximum = max(maximum, len(own))
        base = dict(path=str(path), record_sha256=item['record_sha256'], observations_sha256=item['observations_sha256'],
            controlled=seat, id=item['id'], group=item['group'], priority=item['id'] in priority)
        mass = 1. / (len(groups) * counts[item['group']])
        rows.append(dict(base, teaching_steps=ce, ce_coef=mass, kl_coef=0., kind='verified_fit_teaching_suffix_CE'))
        rows.append(dict(base, teaching_steps=own, ce_coef=0., kl_coef=.1 * mass, kind='complete_own_history_parent_KL'))
    if len(rows) != 64 or maximum != 138:
        raise ValueError('the predeclared 32-history inventory changed')
    output = run / 'ppo-study'
    output.mkdir(exist_ok=False)
    evidence = [course / 'plan.json', exam / 'plan.json', release / 'frozen-manifest.json',
        overlay / 'formal-protocol/admission.json', overlay / 'gpu-reload.json', checkpoint, history,
        checkpoint.with_name(checkpoint.name + '.metadata.json'), history.with_name(history.name + '.metadata.json'),
        run / 'round1-training/comparison.json', run / 'round2-training/comparison.json']
    config = dict(format_version=1, parent_checkpoint=str(checkpoint), parent_sha256=sha(checkpoint),
        sequence_steps=256, histories=len(rows), gradient_check_pairs=[0, len(selected)-1], learning_rate=1e-5,
        full_legal_menu=False, code_list_sha256=sha(release / 'code_list.txt'),
        semantic_metadata_sha256=sha(release / 'semantics/metadata.json'), protected=protected(evidence), rows=rows,
        scope='Fit-only deterministic group-balanced selection including every observed fit regression and all seven repair witnesses. CE only on verified teaching suffix; KL over every own decision. No truncation, no independent labels, fresh separate scene Adam after each complete PPO update.')
    for role in ('control', 'rehearse'):
        write(output / ('rehearsal-' + role + '.json'), dict(config, role=role))
    p = dict(format_version=1, id='reality-loop-filtered-ppo-v1', seed=2026102811,
        release=str(release), runtime_manifest_sha256=sha(release / 'frozen-manifest.json'),
        runtime_protocol=str(overlay / 'formal-protocol/admission.json'), course=str(course), independent_exam=str(exam),
        corrected_runtime_requires_new_pre_ppo_baseline=True,
        parent_checkpoint=str(checkpoint), parent_sha256=sha(checkpoint), history_checkpoint=str(history), history_sha256=sha(history),
        roles=['control', 'rehearse'], local_num_envs=8, local_env_threads=4, num_actor_threads=10, num_steps=128,
        num_minibatches=40, learning_rate=1e-5, update_epochs=1, total_timesteps_per_arm=1003520, smoke_timesteps=20480,
        batch=10240, formal_updates=98, smoke_updates=2, max_seconds_smoke=3600, max_seconds_per_arm=7200,
        train_opponent='mixed', mixed_actor_cycle=dict(self=5, history=3, native_greedy=2),
        deck_pool=prep['decks'], deck_sampling='independently sampled for each seat from the same four fit Crystron multisets',
        precision='highest', full_legal_menu=False, max_step_loss=2., q_training_mode='off',
        smoke_updates_discarded=True, both_arms_restart_same_parent=True, model_selection='fixed final update only',
        full_history_capacity=256, longest_own_history=maximum,
        rehearsal_configs=protected([output / ('rehearsal-' + r + '.json') for r in ('control', 'rehearse')]),
        protected=protected(evidence),
        source_sha256=protected([Path(__file__), PROJECT / 'scripts/run_elfnote_formal.py', PROJECT / 'ygoai/rl/scene_rehearsal.py', PROJECT / 'ygoai/rl/scene_optimizer.py']),
        pre_ppo_endpoints=[dict(name='parent', checkpoint=str(history), checkpoint_sha256=sha(history)), *endpoints],
        post_ppo_endpoints=['control-final', 'rehearse-final'],
        evaluation=dict(independent_all_rows=len(exam_plan['rows']), natural_duels_per_endpoint=256,
            decks=4, seeds_per_deck=32, seats=[0, 1], opponent_checkpoint=str(run / 'round2-training/control/endpoint/model.flax_model'),
            invalid_or_budget_is_unknown=True, independent_score_used_for_selection=False,
            development_before_and_after=True, all_seven_fixed_endpoints=True), promotion_eligible=False)
    write(run / 'ppo-protocol.json', p)
    ledger(run, 'ppo_protocol_frozen', [run / 'ppo-protocol.json', *[Path(p) for p in p['rehearsal_configs']]],
        claim='Fixed terminal parent, compatible corrected runtime, both ordinary PPO and complete-history rehearsal; no independent score opened')
    print('PPO_PROTOCOL_FROZEN', len(selected), maximum, flush=True)


def execute(a):
    run = a.run.resolve()
    path = run / 'ppo-protocol.json'
    p = read(path)
    if not all(sha(Path(f)) == value for f, value in {**p['protected'], **p['source_sha256'], **p['rehearsal_configs']}.items()):
        raise ValueError('frozen PPO input or source changed')
    release = Path(p['release'])
    verify_runtime(release)
    if sha(release / 'frozen-manifest.json') != p['runtime_manifest_sha256']:
        raise ValueError('PPO runtime changed')
    output = run / 'ppo-study'
    env = dict(os.environ, JAX_PLATFORMS='cuda', JAX_DEFAULT_MATMUL_PRECISION=p['precision'],
        PYTHONPATH=str(release / 'repo/ygoenv') + ':' + str(release / 'repo'),
        XLA_PYTHON_CLIENT_PREALLOCATE='false', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    if a.phase == 'baseline':
        folder = output / 'before'
        folder.mkdir(exist_ok=False)
        child([sys.executable, str(PROJECT / 'scripts/studies/crystron_curriculum/evaluate_supervision.py'),
            '--frozen', p['course'], '--checkpoint', p['parent_checkpoint'], '--matmul-precision', p['precision'],
            '--output', str(folder / 'evaluation')], folder, 'evaluation', env, 3600)
        summary = read(folder / 'evaluation/summary.json')
        if not all(r['audit']['passed'] and r['status'] != 'invalid' for r in summary['results']):
            raise ValueError('corrected-runtime baseline has an invalid or unaudited result')
        write(folder / 'verification.json', dict(passed=True, ppo_protocol_sha256=sha(path),
            checkpoint_sha256=sha(p['parent_checkpoint']), results=len(summary['results']),
            counts=dict(Counter(r['status'] for r in summary['results'])), summary_sha256=sha(folder / 'evaluation/summary.json')))
        ledger(run, 'corrected_runtime_pre_ppo_baseline', [folder / 'verification.json', folder / 'evaluation/summary.json'],
            claim='Predefined development baseline on the corrected runtime; fixed model and recipe, no unseen score selection')
        print('PPO_BASELINE_COMPLETED', len(summary['results']), flush=True)
        return
    if a.phase == 'train':
        baseline = read(output / 'before/verification.json')
        if not baseline['passed'] or baseline['ppo_protocol_sha256'] != sha(path):
            raise ValueError('corrected-runtime baseline required before formal optimization')
        for role in p['roles']:
            proof = read(output / ('smoke-' + role) / 'rehearsal-verification.json')
            if not proof['passed'] or proof['ppo_protocol_sha256'] != sha(path):
                raise ValueError('both same-protocol smokes must pass first')
    folder = output / (('smoke-' if a.phase == 'smoke' else '') + a.role)
    folder.mkdir(exist_ok=False)
    steps = p['smoke_timesteps'] if a.phase == 'smoke' else p['total_timesteps_per_arm']
    checkpoint = Path(p['parent_checkpoint'])
    argv = command(p, release, checkpoint, folder, steps)
    argv[argv.index('--deck') + 1] = str(release / 'decks')
    argv[argv.index('--historical-checkpoints') + 1] = p['history_checkpoint']
    config = output / ('rehearsal-' + a.role + '.json')
    argv += ['--scene-rehearsal-config', str(config), '--max-step-loss', str(p['max_step_loss'])]
    write(folder / 'launch-plan.json', dict(phase=a.phase, role=a.role, command=argv,
        ppo_protocol_sha256=sha(path), checkpoint_sha256=sha(checkpoint), runtime_manifest_sha256=p['runtime_manifest_sha256']))
    if a.phase == 'smoke':
        text = child(argv + ['--config-only'], folder, 'config-only', env, 300)
        if 'cuda' not in text.lower() or (folder / 'scene-rehearsal-admission.json').exists():
            raise ValueError('config-only failed or initialized an optimizer')
    child(argv, folder, 'train', env, p['max_seconds_smoke'] if a.phase == 'smoke' else p['max_seconds_per_arm'])
    result = verify_training(folder, checkpoint, steps, p['batch'])
    admission = read(folder / 'scene-rehearsal-admission.json')
    logs = [json.loads(line) for line in (folder / 'scene-rehearsal.jsonl').read_text().splitlines()]
    if not admission['admission']['passed'] or len(logs) != steps // p['batch']:
        raise ValueError('incomplete scene rehearsal qualification or updates')
    cc, kc = admission['admission']['ce_coeff'], admission['admission']['kl_coeff']
    for i, row in enumerate(logs, 1):
        if row['ppo_update'] != i or row['rehearsal_updates'] != i or row['role'] != a.role or not row['finite']:
            raise ValueError('invalid rehearsal update')
        values = [row['loss'], row['grad_norm'], row['parameter_delta_l2'], *row['ce'], *row['kl']]
        expected = sum(x*y for x,y in zip(cc,row['ce'])) + sum(x*y for x,y in zip(kc,row['kl']))
        if not all(math.isfinite(x) for x in values) or abs(expected-row['loss'])/max(1.,abs(expected),abs(row['loss'])) > 1e-6:
            raise ValueError('rehearsal objective or finiteness mismatch')
        if a.role == 'control' and (row['loss'] != 0 or row['grad_norm'] != 0 or row['parameter_delta_l2'] != 0):
            raise ValueError('ordinary PPO control received a scene update')
    if a.role == 'rehearse' and not any(r['parameter_delta_l2'] > 0 for r in logs):
        raise ValueError('rehearsal had no actual parameter update')
    verify_runtime(release)
    if not all(sha(Path(f)) == value for f, value in {**p['protected'], **p['rehearsal_configs']}.items()):
        raise ValueError('protected study evidence changed during training')
    write(folder / 'rehearsal-verification.json', dict(passed=True, ppo_protocol_sha256=sha(path), role=a.role,
        updates=len(logs), checkpoint_sha256=result['checkpoint_sha256'], extra_scene_seconds=sum(r['seconds'] for r in logs)))
    ledger(run, 'ppo_' + a.phase + '_completed', [folder / 'verification.json', folder / 'rehearsal-verification.json'],
        role=a.role, claim='Matched fixed-budget optimization completed; transfer and full natural-duel outcome still require independent exams')
    print('PPO_' + a.phase.upper() + '_COMPLETED', a.role, result['checkpoint_sha256'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze', 'execute'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path)
    p.add_argument('--phase', choices=['baseline', 'smoke', 'train'], default='smoke')
    p.add_argument('--role', choices=['control', 'rehearse'], default='control')
    a = p.parse_args()
    if a.mode == 'freeze' and a.runtime_root is None:
        p.error('--runtime-root required for freeze')
    {'freeze': freeze, 'execute': execute}[a.mode](a)


if __name__ == '__main__':
    main()
