"""Finalize the full study only after every fixed endpoint exam is complete.

Report retention failures and unknown duel outcomes directly. Completion of an
experiment is distinct from a promotion, and no passing subset is substituted
for the six independent rows or the 256 planned natural duels per endpoint.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

from contracts import ledger, read, sha, write
from compare import audit_endpoint
from duel_exam import completed_case

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from scripts.run_elfnote_formal import verify_runtime


def score(row):
    if row['status'] != 'natural_terminal':
        return None
    return {'win': 1., 'draw': .5, 'loss': 0.}[row['candidate_result']]


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
    for name in ('round1', 'round2'):
        for arm in ('control', 'repair'):
            audit_endpoint(a.run / (name + '-training') / arm, 512)
        rounds[name] = read(a.run / (name + '-training/comparison.json'))
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
    before = read(Path(ppo['study_output']) / 'before/evaluation/summary.json')['results']
    key = lambda r: (r['id'], r['seed'])
    baseline = {key(r): r for r in before}
    retention = {}
    for role in ppo['roles']:
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
        retention[role] = dict(before=dict(Counter(r['status'] for r in before)),
            after=dict(Counter(r['status'] for r in after.values())), regressions=regressions, passed=not regressions)
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
        endpoints=endpoints, post_ppo_retention=retention,
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
    lines += ['', '无效和预算耗尽均保留为未知，未折算成胜负。', '',
        f"普通PPO的旧能力保持：{'通过' if retention['control']['passed'] else '未通过'}；完整历史保持组：{'通过' if retention['rehearse']['passed'] else '未通过'}。", '',
        '两轮教学本身的旧能力回归、完整对局配对边界及所有输入/产物哈希见同目录 report.json。默认模型不自动晋级。']
    (a.output / 'report.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    ledger(a.run, 'complete_study_audit', [a.output / 'report.json', a.output / 'report.zh.md'],
        claim='All predeclared endpoint denominators completed and artifacts verified. Experimental completion only; no automatic model promotion.')
    print('FULL_STUDY_AUDITED', len(endpoints), flush=True)


if __name__ == '__main__':
    main()
