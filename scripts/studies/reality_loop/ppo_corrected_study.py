"""Freeze and execute matched ordinary PPO/full-history rehearsal continuations.

The terminal round-two repair checkpoint is fixed before unseen inference.
Discard smoke updates; both formal arms restart from that exact checkpoint.
Fresh corrected-input teaching is a new experiment, not a runtime-only bridge.
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
from corrected_lineage import teaching_bindings, actor_path_bindings

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from scripts.run_elfnote_formal import child, command, verify_runtime, verify_training
from ygoai.rl.core_audit import validate_curriculum_binding


def protected(paths):
    return {str(p.resolve()): sha(p) for p in paths}


import ppo_study as legacy


def freeze(a):
    run, overlay = a.run.resolve(), a.runtime_root.resolve()
    release = overlay / 'formal-runtime'
    manifest = verify_runtime(release)
    prep = read(release / 'preparation.json')
    admission = read(overlay / 'formal-protocol/admission.json')
    reload = read(a.gpu_reload)
    if not admission['passed'] or admission['runtime_manifest_sha256'] != sha(release / 'frozen-manifest.json'):
        raise ValueError('current corrected runtime lacks protocol admission')
    course = run / 'round2-training/repair-inputs'
    plan = read(course / 'plan.json')
    exam = overlay / 'independent-exam'
    exam_plan = read(exam / 'plan.json')
    validate_curriculum_binding(plan)
    validate_curriculum_binding(exam_plan)
    initial = read(run / 'protocol.json')
    original = read(Path(initial['original']) / 'plan.json')
    if (len(original['rows']) != 235 or original['native_sha256'] != prep['diagnostic_sha256'] or
            not original['overlay_event_reencoding']['all_rows_retained'] or
            plan['native_sha256'] != original['native_sha256'] or
            exam_plan['native_sha256'] != original['native_sha256'] or
            prep.get('core_rule_revision') != 'explicit-nonlevel-material-v1'):
        raise ValueError('fresh two-round teaching must use the fully corrected runtime from the original 235 rows')
    if prep['diagnostic_sha256'] != plan['native_sha256'] or len(prep['decks']) != 4:
        raise ValueError('corrected normal/diagnostic runtime mismatch')
    fresh_bindings = teaching_bindings(run, prep['diagnostic_sha256'])
    fresh_bindings.update(actor_path_bindings(run, course, a.path_proof, prep['diagnostic_sha256']))
    endpoints = []
    for round_name in ('round1', 'round2'):
        for arm in ('control', 'repair'):
            folder = run / (round_name + '-training') / arm
            completion, _, _ = audit_endpoint(folder, 512)
            endpoints.append(dict(name=round_name + '-' + arm, checkpoint=str(folder / 'endpoint/model.flax_model'),
                checkpoint_sha256=completion['checkpoint_sha256']))
    checkpoint = run / 'round2-training/repair/endpoint/model.flax_model'
    if sha(checkpoint) != endpoints[-1]['checkpoint_sha256']:
        raise ValueError('PPO parent must be the audited terminal round-two repair endpoint')
    if not reload['passed'] or reload['checkpoint_sha256'] != sha(checkpoint) or reload['native_sha256'] != manifest['native_sha256']:
        raise ValueError('actual current-runtime GPU reload required')
    history = Path(read(run / 'protocol.json')['initial_checkpoint'])
    if sha(history) != initial['initial_checkpoint_sha256']:
        raise ValueError('historical opponent must be this experiment initial parent')
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
    if not 2 <= len(selected) <= 64 or maximum > 256:
        raise ValueError('full own-history must fit the fixed 256-step capacity without truncation')
    protocol_path = run / (a.name + '-protocol.json') if a.name != 'ppo-study' else run / 'ppo-protocol.json'
    if protocol_path.exists():
        raise FileExistsError(protocol_path)
    output = run / a.name
    output.mkdir(exist_ok=False)
    evidence = [course / 'plan.json', exam / 'plan.json', release / 'frozen-manifest.json',
        overlay / 'formal-protocol/admission.json', a.gpu_reload, checkpoint, history,
        run / 'protocol.json', run / 'round2-context.json', Path(initial['original']) / 'plan.json',
        a.duel_opponent.resolve(), a.duel_opponent.resolve().with_name(a.duel_opponent.name + '.metadata.json'),
        checkpoint.with_name(checkpoint.name + '.metadata.json'), history.with_name(history.name + '.metadata.json'),
        run / 'round1-training/comparison.json', run / 'round2-training/comparison.json']
    config = dict(format_version=1, parent_checkpoint=str(checkpoint), parent_sha256=sha(checkpoint),
        sequence_steps=256, histories=len(rows), gradient_check_pairs=[0, len(selected)-1], learning_rate=1e-5,
        full_legal_menu=False, code_list_sha256=sha(release / 'code_list.txt'),
        semantic_metadata_sha256=sha(release / 'semantics/metadata.json'), protected={**protected(evidence), **fresh_bindings}, rows=rows,
        scope='Fit-only deterministic group-balanced selection including every observed fit regression and four records per group or every priority record when larger. The group inventory includes freshly certified repair groups. CE only on verified teaching suffix; KL over every own decision. No truncation, no independent labels, fresh separate scene Adam after each complete PPO update.')
    for role in ('control', 'rehearse'):
        write(output / ('rehearsal-' + role + '.json'), dict(config, role=role))
    p = dict(format_version=1, id='reality-loop-corrected-input-ppo-v1', seed=2026102811, study_output=str(output),
        release=str(release), runtime_manifest_sha256=sha(release / 'frozen-manifest.json'),
        runtime_protocol=str(overlay / 'formal-protocol/admission.json'), course=str(course), independent_exam=str(exam),
        corrected_runtime_requires_new_pre_ppo_baseline=True,
        parent_checkpoint=str(checkpoint), parent_sha256=sha(checkpoint), history_checkpoint=str(history), history_sha256=sha(history),
        roles=['control', 'rehearse'], local_num_envs=8, local_env_threads=4, num_actor_threads=10, num_steps=128,
        num_minibatches=40, learning_rate=1e-5, update_epochs=1, total_timesteps_per_arm=1003520, smoke_timesteps=20480,
        batch=10240, formal_updates=98, smoke_updates=2, max_seconds_smoke=3600, max_seconds_per_arm=7200,
        train_opponent='mixed', mixed_actor_cycle=dict(self=5, history=3, native_greedy=2),
        deck_pool=prep['decks'], deck_sampling='independently sampled for each seat from the same four fit Crystron multisets',
        precision='highest', jax_precision_environment='float32', full_legal_menu=False, max_step_loss=2., q_training_mode='off',
        precision_environment_scope='This installed JAX accepts float32 at import, the full-precision equivalent of the highest runtime setting. No reduced precision.',
        smoke_updates_discarded=True, both_arms_restart_same_parent=True, model_selection='fixed final update only',
        full_history_capacity=256, longest_own_history=maximum,
        rehearsal_configs=protected([output / ('rehearsal-' + r + '.json') for r in ('control', 'rehearse')]),
        protected={**protected(evidence), **fresh_bindings},
        source_sha256=protected([Path(__file__), Path(legacy.__file__), Path(__file__).with_name('corrected_lineage.py'), PROJECT / 'scripts/run_elfnote_formal.py', PROJECT / 'ygoai/rl/scene_rehearsal.py', PROJECT / 'ygoai/rl/scene_optimizer.py']),
        pre_ppo_endpoints=[dict(name='parent', checkpoint=str(history), checkpoint_sha256=sha(history)), *endpoints],
        post_ppo_endpoints=['control-final', 'rehearse-final'],
        evaluation=dict(independent_all_rows=len(exam_plan['rows']), natural_duels_per_endpoint=256,
            decks=4, seeds_per_deck=32, seats=[0, 1], opponent_checkpoint=str(a.duel_opponent.resolve()),
            invalid_or_budget_is_unknown=True, independent_score_used_for_selection=False,
            development_before_and_after=True, all_seven_fixed_endpoints=True), promotion_eligible=False)
    write(protocol_path, p)
    ledger(run, 'ppo_protocol_frozen', [protocol_path, *[Path(p) for p in p['rehearsal_configs']]],
        claim='Fixed terminal parent, compatible corrected runtime, both ordinary PPO and complete-history rehearsal; no independent score opened')
    print('PPO_PROTOCOL_FROZEN', len(selected), maximum, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze', 'execute'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path)
    p.add_argument('--gpu-reload', type=Path)
    p.add_argument('--path-proof', type=Path)
    p.add_argument('--duel-opponent', type=Path)
    p.add_argument('--name', default='ppo-study')
    p.add_argument('--protocol', type=Path)
    p.add_argument('--phase', choices=['baseline', 'smoke', 'train'], default='smoke')
    p.add_argument('--role', choices=['control', 'rehearse'], default='control')
    a = p.parse_args()
    if Path(a.name).name != a.name or a.name in ('', '.', '..'):
        p.error('--name must be a single directory name')
    if a.mode == 'freeze' and any(v is None for v in (a.runtime_root, a.gpu_reload, a.path_proof, a.duel_opponent)):
        p.error('freeze requires runtime, GPU reload, complete path proof and fixed duel opponent')
    {'freeze': freeze, 'execute': legacy.execute}[a.mode](a)


if __name__ == '__main__':
    main()
