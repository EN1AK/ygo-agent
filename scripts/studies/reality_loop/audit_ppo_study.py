"""Finalize the full study only after every fixed endpoint exam is complete.

Report retention failures and unknown duel outcomes directly. Completion of an
experiment is distinct from a promotion, and no passing subset is substituted
for the six independent rows or the 256 planned natural duels per endpoint.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

from contracts import ledger, read, sha, write, round_protocol
from compare import audit_endpoint
from duel_exam import completed_case

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from scripts.run_elfnote_formal import finite_training_metrics, verify_runtime


def score(row):
    if row['status'] != 'natural_terminal':
        return None
    return {'win': 1., 'draw': .5, 'loss': 0.}[row['candidate_result']]


def round_evidence(run, name):
    """Bind actual current-policy roots, certified teachers and measured cost."""
    protocol = round_protocol(run, name)
    candidates = read(run / name / 'candidates.json')
    if len(candidates) > protocol['roots_per_round'] or any(r['checkpoint_sha256'] != protocol['initial_checkpoint_sha256'] for r in candidates):
        raise ValueError('search roots do not come from the fixed current round policy')
    if name == 'round2' and protocol['initial_checkpoint_sha256'] != sha(run / 'round1-training/repair/endpoint/model.flax_model'):
        raise ValueError('round two did not observe the updated repair policy')
    recovery = run / (name + '-recovery')
    summary = read(recovery / 'summary.json')
    reports = [read(path) for path in sorted(recovery.glob('*-search.json'))]
    if summary['cases'] != len(candidates) or not reports:
        raise ValueError('search denominator missing')
    for item in summary['admitted']:
        path = Path(item['path'])
        if sha(path) != item['record_sha256'] or sha(path.with_suffix('.npz')) != item['observations_sha256']:
            raise ValueError('certified teacher changed')
        record = read(path)
        admission = record['admission']
        if (not admission['same_actual_prefix'] or not admission['repeated_adapter'] or admission['split'] != 'fit' or
                not record['independent_endpoint_audit']['passed'] or not record['opponent_policy_audit']['passed'] or
                any(d.get('teaching', True) for d in record['decisions'][:item['root_step']])):
            raise ValueError('teacher lacks actual-prefix/independent/opponent/teaching-mask evidence')
    costs = {}
    artifacts = {str(path): sha(path) for path in sorted(recovery.glob('*-search.json'))}
    for arm in ('control', 'repair'):
        path = run / (name + '-training') / arm / 'endpoint/metrics.jsonl'
        metrics = [json.loads(line) for line in path.read_text().splitlines()]
        if len(metrics) != 512 or not all(math.isfinite(r['seconds']) for r in metrics):
            raise ValueError('SFT cost metrics incomplete')
        costs[arm] = dict(optimizer_updates=512, optimizer_phase_seconds=metrics[-1]['seconds'],
            scope='elapsed optimizer phase including compilation; excludes preflight and before/after exams')
        artifacts[str(path)] = sha(path)
    return dict(current_policy_sha256=protocol['initial_checkpoint_sha256'], failure_roots=len(candidates),
        positive_witnesses=len(summary['admitted']), unknown_search_reports=len(summary['unknown']),
        search_attempts=sum(len(r['attempts']) for r in reports), search_seconds=sum(r['seconds'] for r in reports),
        search_status_counts=dict(Counter(r['status'] for r in reports)), sft_cost=costs, artifacts=artifacts,
        information_mode='offline_oracle_exact_state', measured_flops=None,
        unmeasured_costs=['observation wall time', 'human corpus acquisition cost', 'preflight/evaluation wall time', 'hardware FLOPs'])


def ppo_evidence(folder, protocol, role):
    admission_path = folder / 'scene-rehearsal-admission.json'
    admission = read(admission_path)
    config_path = folder.parent / ('rehearsal-' + role + '.json')
    if admission['config_sha256'] != sha(config_path) or admission['config'] != read(config_path):
        raise ValueError('actual scene optimizer config changed')
    proof = admission['admission']
    if (not proof['passed'] or proof['gradient_reference_l2'] <= 1e-8 or proof['gradient_relative_error'] >= .002 or
            proof['histories'] != 64 or admission['config']['sequence_steps'] != 256):
        raise ValueError('full-history GPU arithmetic qualification missing')
    metrics_path = folder / 'scene-rehearsal.jsonl'
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    if len(rows) != 98 or not all(r['finite'] and r['ppo_update'] == r['rehearsal_updates'] == i for i,r in enumerate(rows,1)):
        raise ValueError('PPO scene updates incomplete')
    if role == 'control' and any(r['loss'] != 0 or r['grad_norm'] != 0 or r['parameter_delta_l2'] != 0 for r in rows):
        raise ValueError('ordinary PPO control received scene updates')
    if role == 'rehearse' and not any(r['parameter_delta_l2'] > 0 for r in rows):
        raise ValueError('full-history retention made no update')
    finite_training_metrics((folder / 'train.log').read_text(), 98)
    return dict(fitted_env_steps=protocol['total_timesteps_per_arm'], ppo_updates=98,
        complete_history_rehearsal_seconds=sum(r['seconds'] for r in rows),
        gradient_relative_error=proof['gradient_relative_error'], gradient_reference_l2=proof['gradient_reference_l2'],
        maximum_sequential_batched_logit_error=proof['max_logit_error'],
        cost_scope='scene update timings exclude optimizer qualification, PPO collection and PPO updates; control computes the same histories with zero coefficients',
        artifacts={str(p):sha(p) for p in (admission_path,metrics_path,folder/'train.log')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'exam', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    protocol = read(a.exam / 'protocol.json')
    ppo = read(protocol['ppo_protocol'])
    if sha(protocol['ppo_protocol']) != protocol['ppo_protocol_sha256']:
        raise ValueError('PPO protocol changed')
    if not all(sha(Path(f)) == value for f, value in {**ppo['protected'], **protocol['protected']}.items()):
        raise ValueError('fixed study evidence changed')
    verify_runtime(Path(ppo['release']))
    rounds = {}
    costs = dict(rounds={}, ppo={})
    for name in ('round1', 'round2'):
        for arm in ('control', 'repair'):
            audit_endpoint(a.run / (name + '-training') / arm, 512)
        rounds[name] = read(a.run / (name + '-training/comparison.json'))
        costs['rounds'][name] = round_evidence(a.run, name)
    endpoints = {}
    full_rows = {}
    template = None
    independent_ids = {r['id'] for r in read(Path(protocol['independent_exam']) / 'plan.json')['rows']}
    for endpoint in protocol['endpoints']:
        name = endpoint['name']
        if sha(endpoint['checkpoint']) != endpoint['checkpoint_sha256']:
            raise ValueError('fixed endpoint changed')
        counts = {}
        for phase in ('independent', 'natural'):
            proof = read(a.exam / (name + '-' + phase + '-completion.json'))
            if not proof['completed'] or proof['protocol_sha256'] != sha(a.exam / 'protocol.json') or sha(proof['summary']) != proof['summary_sha256']:
                raise ValueError('endpoint exam completion changed')
            if not all(sha(Path(f)) == value for f, value in proof['record_artifacts'].items()):
                raise ValueError('exam raw records changed')
            results = read(proof['summary'])['results']
            expected = 6 if phase == 'independent' else 256
            if len(results) != expected or len({r['id'] for r in results}) != expected:
                raise ValueError('endpoint denominator or identities incomplete')
            counts[phase] = dict(Counter(r['status'] for r in results))
            if phase == 'independent':
                if {r['id'] for r in results} != independent_ids or not all(r['audit']['passed'] for r in results):
                    raise ValueError('independent record lacks engine audit')
            else:
                duel_path = a.exam / (name + '-duel-plan/plan.json')
                duel = read(duel_path)
                cases = duel['cases']
                current = [(c['id'], c['seed'], c['candidate_seat'], c['definition']) for c in cases]
                if template is None:
                    template = current
                elif current != template:
                    raise ValueError('unpaired natural duel matrix')
                rows = []
                for case in cases:
                    row = completed_case(a.exam / (name + '-natural'), case, duel, sha(duel_path))
                    if row is None:
                        raise ValueError('natural duel lacks a bound completed attempt')
                    rows.append(row)
                if rows != results:
                    raise ValueError('natural summary differs from completed attempts')
                full_rows[name] = rows
                natural = [score(r) for r in rows]
                unknown = sum(v is None for v in natural)
                points = sum(v for v in natural if v is not None)
                counts['natural_results'] = dict(Counter(r['candidate_result'] for r in rows if r['status'] == 'natural_terminal'))
                counts['unknown'] = unknown
                counts['score_fraction_bounds_all_256'] = [points / 256., (points + unknown) / 256.]
        endpoints[name] = dict(checkpoint_sha256=endpoint['checkpoint_sha256'], **counts)
        coverage = protocol['independent_intake_coverage']
        unknown = coverage['conversion_unknown_cases'] + counts['independent'].get('budget_exhausted', 0)
        success = counts['independent'].get('success', 0)
        endpoints[name]['independent_intake_cases'] = coverage['intake_cases']
        endpoints[name]['independent_conversion_unknown'] = coverage['conversion_unknown_cases']
        endpoints[name]['independent_success_fraction_bounds_all_intake'] = [success / coverage['intake_cases'],
            (success + unknown) / coverage['intake_cases']]
    before = read(Path(ppo['study_output']) / 'before/evaluation/summary.json')['results']
    key = lambda r: (r['id'], r['seed'])
    baseline = {key(r): r for r in before}
    original_completion = read(a.exam / 'parent-development-completion.json')
    if sha(original_completion['summary']) != original_completion['summary_sha256'] or not all(sha(Path(f)) == value for f,value in original_completion['record_artifacts'].items()):
        raise ValueError('corrected-runtime initial-model baseline changed')
    original_results = read(original_completion['summary'])['results']
    original = {key(r):r for r in original_results}
    if original.keys() != baseline.keys() or not all(r['audit']['passed'] and r['status'] != 'invalid' for r in original.values()):
        raise ValueError('initial model needs the identical corrected-runtime development matrix')
    retention = {}
    for role in ppo['roles']:
        costs['ppo'][role] = ppo_evidence(Path(ppo['study_output']) / role, ppo, role)
        proof = read(Path(ppo['study_output']) / role / 'verification.json')
        replay = read(Path(ppo['study_output']) / role / 'rehearsal-verification.json')
        if not proof['passed'] or proof['updates'] != 98 or not replay['passed'] or replay['ppo_protocol_sha256'] != protocol['ppo_protocol_sha256']:
            raise ValueError('PPO producer did not complete all fixed updates')
        completion = read(a.exam / (role + '-final-development-completion.json'))
        if sha(completion['summary']) != completion['summary_sha256'] or not all(sha(Path(f)) == value for f, value in completion['record_artifacts'].items()):
            raise ValueError('post-PPO development evidence changed')
        after = {key(r): r for r in read(completion['summary'])['results']}
        if after.keys() != baseline.keys() or not all(r['audit']['passed'] and r['status'] != 'invalid' for r in after.values()):
            raise ValueError('development retention cases changed or lack audit')
        regressions = [dict(id=k[0], seed=k[1], after=after[k]['status']) for k, r in baseline.items() if r['status'] == 'success' and after[k]['status'] != 'success']
        original_regressions = [dict(id=k[0],seed=k[1],after=after[k]['status']) for k,r in original.items() if r['status']=='success' and after[k]['status']!='success']
        retention[role] = dict(before=dict(Counter(r['status'] for r in before)),
            after=dict(Counter(r['status'] for r in after.values())), regressions=regressions, passed=not regressions,
            initial_model_counts=dict(Counter(r['status'] for r in original_results)),
            initial_model_regressions=original_regressions, initial_model_retention_passed=not original_regressions)
    paired = []
    known_delta = 0.
    unknown_pairs = 0
    for left, right in zip(full_rows['control-final'], full_rows['rehearse-final']):
        x, y = score(left), score(right)
        if x is None or y is None:
            unknown_pairs += 1
        else:
            known_delta += y - x
        paired.append(dict(id=left['id'], control=x, rehearse=y))
    # Unknown pairs contribute any delta in [-1,1]; report the full planned
    # denominator rather than presenting the valid-only subset as a fair winrate.
    result = dict(experiment_complete=True, objective_scope='two audited feedback/SFT rounds, independent transfer, full natural duels, and matched PPO retention',
        exam_protocol_sha256=sha(a.exam / 'protocol.json'), ppo_protocol_sha256=protocol['ppo_protocol_sha256'],
        rounds={name: dict(counts=r['counts'], local_retention_passed=r['local_retention_passed']) for name, r in rounds.items()},
        endpoints=endpoints, post_ppo_retention=retention, measured_costs_and_limits=costs,
        independent_intake_coverage=protocol['independent_intake_coverage'],
        sampling_uncertainty='Descriptive fixed-component/four-fit-deck evaluation with correlated paired seats. Unknown-outcome bounds are not statistical confidence intervals or human-ladder guarantees.',
        paired_ppo=dict(planned=256, unknown_pairs=unknown_pairs, observed_paired_point_delta=known_delta,
            point_fraction_delta_bounds=[(known_delta - unknown_pairs) / 256., (known_delta + unknown_pairs) / 256.], rows=paired),
        promotion_eligible=False, no_independent_score_model_selection=True,
        interpretation='Engineering completion does not imply superior play. Old-skill regressions, failed transfer and unknown/invalid duels remain reported; no default model is promoted.')
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output / 'report.json', result)
    lines = ['# 过滤教学与正式PPO研究验收', '', '全部固定端点和预定分母完成；完成实验不代表可以晋级默认模型。', '',
        '| 端点 | 独立题成功 / 6 | 自然胜 | 自然负 | 和局 | 未知 / 256 |', '|---|---:|---:|---:|---:|---:|']
    for name, r in endpoints.items():
        n = r['natural_results']
        lines.append(f"| {name} | {r['independent'].get('success', 0)} | {n.get('win', 0)} | {n.get('loss', 0)} | {n.get('draw', 0)} | {r['unknown']} |")
    coverage = protocol['independent_intake_coverage']
    lines += ['', f"独立题原始接收{coverage['intake_cases']}条，其中{coverage['conversion_unknown_cases']}条转换失败保留为未知；表中列出可评测的6条，report.json另报原始接收分母上的成功比例边界。",
        '独立题从固定人类前缀进入己方主阶段，之后对手被动；完整策略对手的双方决策另由自然对局检验。', '',
        '无效和预算耗尽均保留为未知，未折算成胜负。', '',
        f"相对PPO起点的能力保持：普通组{'通过' if retention['control']['passed'] else '未通过'}，完整历史组{'通过' if retention['rehearse']['passed'] else '未通过'}。",
        f"相对最初模型的旧能力保持（在同一修订环境重测）：普通组{'通过' if retention['control']['initial_model_retention_passed'] else '未通过'}，完整历史组{'通过' if retention['rehearse']['initial_model_retention_passed'] else '未通过'}。", '',
        '比例边界仅反映未知结果，不是统计置信区间；固定组件、四副训练牌表及配对座位存在相关性，不能直接外推人类天梯胜率。', '',
        '两轮教学本身的旧能力回归、完整对局配对边界及所有输入/产物哈希见同目录 report.json。默认模型不自动晋级。']
    (a.output / 'report.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    ledger(a.run, 'complete_study_audit', [a.output / 'report.json', a.output / 'report.zh.md'],
        claim='All predeclared endpoint denominators completed and artifacts verified. Experimental completion only; no automatic model promotion.')
    print('FULL_STUDY_AUDITED', len(endpoints), flush=True)


if __name__ == '__main__':
    main()
