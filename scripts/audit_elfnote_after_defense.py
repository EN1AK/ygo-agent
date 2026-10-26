"""Recount the fixed four-endpoint defense/PPO study without model inference."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','training','assessment','output']:parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();a.root=a.root.resolve();a.training=a.training.resolve();a.assessment=a.assessment.resolve()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from scripts.audit_elfnote_rehearsal_results import recount,paired_delta,verify_old_proof
    from scripts.run_elfnote_formal import finite_training_metrics
    from scripts.run_elfnote_formal_evaluation import paired_summary
    protected={}
    def evidence(path):
        protected[str(Path(path).resolve())]=sha(path);return read(path)
    for path in [Path(str(a.training)+'.formal.exit'),Path(str(a.assessment)+'.exit')]:
        assert path.read_text().strip()=='0';protected[str(path)]=sha(path)
    final=evidence(a.assessment/'summary.json');assert final['passed'] and not final['production_default_replaced']
    inputs=evidence(a.assessment/'frozen-inputs.json');assert inputs['frozen_before_evaluation']
    for name in ['elfnote-formal-after-defense-source-20261027','elfnote-after-defense-eval-source-20261027']:
        source=a.training.parent/name;manifest=evidence(source/'source-manifest.json')
        assert all(hashlib.sha256((source/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for path,value in manifest['all_files'].items())
        if name.startswith('elfnote-after-defense-eval'):assert manifest==inputs['source']
    planpath=a.training.parent/'elfnote-formal-after-defense-plan-20261027.json';plan=evidence(planpath)
    assert sha(planpath)==inputs['formal_plan_sha256']
    release=a.root/plan['runtime_release'];runtime=evidence(release/'frozen-manifest.json')
    assert runtime['native_sha256']==plan['native_sha256'] and all(sha(release/r['path'])==r['sha256'] for r in runtime['artifacts'])
    trained=evidence(a.training/'after_defense/verification.json');smoke=evidence(a.training/'preflight-after_defense/verification.json')
    assert trained['passed'] and trained['process_exit_zero'] and trained['zero_nonfinite_updates'] and trained['updates']==98 and trained['fitted_env_steps']==1003520
    assert smoke['passed'] and smoke['process_exit_zero'] and smoke['zero_nonfinite_updates'] and smoke['fitted_env_steps']==20480
    assert trained==final['formal'] and sha(a.training/'after_defense/verification.json')==inputs['formal_verification_sha256']
    for role,budget,updates in [('preflight-after_defense',20480,2),('after_defense',1003520,98)]:
        launch=evidence(a.training/role/'launch-plan.json');assert launch['checkpoint_sha256']==plan['arms']['after_defense']['checkpoint_sha256']
        assert launch['runtime_manifest_sha256']==sha(release/'frozen-manifest.json') and launch['plan_sha256']==sha(planpath)
        assert evidence(a.training/role/'train.exit.json')['returncode']==0
        log=(a.training/role/'train.log').read_text();finite_training_metrics(log,updates)
        protected[str(a.training/role/'train.log')]=sha(a.training/role/'train.log')
        assert (smoke if role.startswith('preflight') else trained)['fitted_env_steps']==budget
    teacher=a.root/plan['arms']['after_defense']['study'];teaching=evidence(teacher/'final-verification.json')
    assert teaching['passed'] and teaching['defensive_policy_gate'] and sha(teacher/'final-verification.json')==plan['teaching_verification_sha256']
    import flax.serialization
    import numpy as np
    def leaves(tree):
        if isinstance(tree,dict):return [v for k in sorted(tree) for v in leaves(tree[k])]
        return [np.asarray(tree)]
    before=leaves(flax.serialization.msgpack_restore((teacher/'teach/model.flax_model').read_bytes())['params'])
    after=leaves(flax.serialization.msgpack_restore(Path(trained['checkpoint']).read_bytes())['params'])
    assert len(before)==len(after) and all(x.shape==y.shape and np.isfinite(y).all() for x,y in zip(before,after))
    delta=math.sqrt(sum(float(np.sum((x.astype(np.float64)-y.astype(np.float64))**2)) for x,y in zip(before,after)))
    assert delta>0 and math.isclose(delta,trained['parameter_delta_l2'],rel_tol=1e-12)
    models=dict(initial=plan['course_checkpoint_sha256'],plain=plan['plain_comparison_checkpoint_sha256'],defense_initial=plan['arms']['after_defense']['checkpoint_sha256'],after_defense=trained['checkpoint_sha256'])
    assert sha(Path(trained['checkpoint']))==models['after_defense']
    games=a.assessment/'games';manifest=evidence(games/'manifest.json');summary=evidence(games/'summary.json')
    assert manifest['plan']==plan and summary['passed']
    rows={m:{op:[] for op in ['history','greedy']} for m in models};seen=set()
    expected={(m,op,seed,seat) for m in models for op in ['history','greedy'] for seed in plan['evaluation']['seeds'] for seat in plan['evaluation']['seats']}
    for i,job in enumerate(manifest['jobs']):
        key=job['model'],job['opponent'],job['seed'],job['seat'];assert key in expected and key not in seen;seen.add(key)
        folder=games/f'{i:02d}-{key[0]}-{key[1]}-{key[2]}-seat{key[3]}';assert evidence(folder/'exit.json')['returncode']==0
        batch=evidence(folder/'result.json');assert batch['checkpoint_sha256']==models[key[0]] and batch['opponent']==key[1]
        assert batch['seed']==key[2] and batch['candidate_seat']==key[3] and batch['native_sha256']==plan['native_sha256']
        assert batch['runtime_manifest_sha256']==sha(release/'frozen-manifest.json') and batch['deck_sha256']==plan['deck_sha256']
        assert batch['opponent_checkpoint_sha256']==(plan['history_checkpoint_sha256'] if key[1]=='history' else None)
        assert len(batch['episodes'])==32 and {r['environment_index'] for r in batch['episodes']}==set(range(32))
        assert all(r['candidate_seat']==key[3] for r in batch['episodes'])
        assert all(batch[k]==v for k,v in recount(batch['episodes']).items());rows[key[0]][key[1]].append(batch)
    assert seen==expected and len(seen)==64
    recounted={m:{op:recount([r for b in batches for r in b['episodes']]) for op,batches in opponents.items()} for m,opponents in rows.items()}
    for m,opponents in recounted.items():
        for op,value in opponents.items():assert value['attempts']==256 and all(summary['summary'][m][op][k]==v for k,v in value.items())
    for name,left,right in [('plain_vs_initial','initial','plain'),('defense_initial_vs_initial','initial','defense_initial'),('after_defense_vs_defense_initial','defense_initial','after_defense'),('after_defense_vs_plain','plain','after_defense'),('after_defense_vs_initial','initial','after_defense')]:
        for op in ['history','greedy']:
            value=summary['paired_differences'][name][op]
            assert value==paired_summary(rows[left][op],rows[right][op])
            assert math.isclose(value['win_per_attempt_delta'],paired_delta(rows[left][op],rows[right][op]),abs_tol=1e-12)
    folder=a.assessment/'retention';proof=evidence(folder/'final-verification.json');m=evidence(folder/'manifest.json')
    assert proof['passed'] and proof['checkpoint_sha256']==m['checkpoint_sha256']==models['after_defense']
    normal=evidence(folder/'normal/summary.json');functional=evidence(folder/'functional/summary.json');counts=Counter();failures=Counter();stable=0
    assert len(normal['cases'])==len(m['normal_cases'])==53
    for i,(case,row) in enumerate(zip(m['normal_cases'],normal['cases'])):
        record=evidence(folder/'normal'/f'case-{i:02}.json');assert record['case']==case['name']==row['name'] and record['status']==row['status']
        cards=[c for c in record['final']['cards'] if c['player']==0]
        monsters=Counter(c['code'] for c in cards if c['location']==4 and c['position']&5);spells=Counter(c['code'] for c in cards if c['location']==8 and c['position']&5);hand=Counter(c['code'] for c in cards if c['location']==2)
        checks=[dict(monsters=not(Counter(g['monsters'])-monsters),spells=not(Counter(g['spells'])-spells),wind_in_hand=hand[56651978]>=1,hand_resources=sum(hand.values())>=g['hand']) for g in case['goals']]
        if record['status'] in ['success','goal_not_reached']:
            stable+=1;assert checks==[r['checks'] for r in record['outcomes']] and (record['status']=='success')==any(all(r.values()) for r in checks)
        if record['status']=='success':counts[case['split']]+=1
        if record['status']=='goal_not_reached':failures[','.join(k for k,v in max(checks,key=lambda r:sum(r.values())).items() if not v)]+=1
    assert len(proof['old_audits'])==len(normal['old'])==5
    for item in proof['old_audits']:verify_old_proof(item['path'],item['proof'])
    counts['old']=sum(r['success'] for r in normal['old']);assert all(counts[k]==v for k,v in proof['normal'].items())
    scores={s:{mode:sum(r['status']=='success' for r in functional['cases'] if r['split']==s and r['mode']==mode) for mode in ['scripted','policy']} for s in functional['scores']}
    assert len(functional['cases'])==56 and scores==functional['scores']==proof['functional']
    probe=evidence(a.assessment/'terminal/final-verification.json');assert probe['passed'] and probe['independent_core_audits']==48
    audits=list(folder.glob('normal/*.audit.json'))+list(folder.glob('functional/*.audit.json'))+list((a.assessment/'terminal').glob('*/*.audit.json'))
    assert len(audits)==157
    native=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014/exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    native_hash=sha(native);assert native_hash=='52c5c91cfd9017732e0a4aca928c902050a37a1a41104c036a5d3f25e1c026e2'
    for path in audits:
        audit=evidence(path);record=path.with_name(path.name.replace('.audit.json','.json'))
        assert audit['passed'] and audit['record_sha256']==sha(record) and audit['observations_sha256']==sha(record.with_suffix('.npz')) and audit['native_sha256']==native_hash
    assert all(sha(Path(path))==value for path,value in protected.items())
    result=dict(passed=True,models=models,attempts=2048,games=recounted,paired_differences=summary['paired_differences'],normal=proof['normal'],functional=scores,terminal=probe['scores'],new_core_audits=157,old_core_audits=5,stable_normal_goals_checked=stable,normal_failure_signatures=dict(failures),protected=protected,auditor_sha256=sha(Path(__file__)),production_default_replaced=False,scope='Independent raw rewards, actual paired environment seeds, objective/exit/start bindings, strict terminal card recount and artifact consistency. Relies on separately executed native core proofs; audit pass is not playing-strength, retention or promotion success. One training seed and exposed exercise families only.')
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='protected'}),flush=True)


if __name__=='__main__':main()
