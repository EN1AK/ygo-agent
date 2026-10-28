"""Freeze corrected-rule exams without dropping earlier fixed PPO endpoints.

Keep the five SFT endpoints and both new terminal PPO models. Earlier PPO
models also receive the complete matrix unless their exact weight hash already
names a model in that matrix; such aliases are explicit and audited. No score
is read to select a checkpoint or change a training parameter.
"""
import argparse
from pathlib import Path
import os
import sys

from contracts import read,sha,write,ledger
from ppo_exams import intake_coverage

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
from scripts.run_elfnote_formal import child,verify_runtime
from ygoai.rl.core_audit import validate_curriculum_binding
from ygoai.rl.checkpoint_compat import load_checkpoint_metadata,metadata_path,REQUIRED_STRUCTURED_FIELDS


def terminal_endpoints(path):
    p=read(path);study=Path(p['study_output']);rows=[]
    for role in p['roles']:
        proof=read(study/role/'verification.json')
        replay=read(study/role/'rehearsal-verification.json')
        if (not proof['passed'] or proof['updates']!=p['formal_updates'] or not replay['passed'] or
                replay['ppo_protocol_sha256']!=sha(path) or replay['checkpoint_sha256']!=proof['checkpoint_sha256'] or
                sha(proof['checkpoint'])!=proof['checkpoint_sha256']):
            raise ValueError('fixed terminal PPO producer incomplete')
        rows.append(dict(name=role+'-final',checkpoint=proof['checkpoint'],checkpoint_sha256=proof['checkpoint_sha256']))
    return rows


def retain_historical(current,previous):
    """Preserve every previous weight; deduplicate only byte-identical models."""
    result=[dict(e) for e in current];aliases=[]
    if len({e['name'] for e in result})!=len(result):
        raise ValueError('duplicate current endpoint name')
    for old in previous:
        name='historical-'+old['name']
        if any(e['name']==name for e in result):
            raise ValueError('historical endpoint name collision')
        same=next((e for e in result if e['checkpoint_sha256']==old['checkpoint_sha256']),None)
        if same:
            aliases.append(dict(name=name,checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
                equivalent_endpoint=same['name'],reason='identical serialized model weights; identical corrected runtime and exam matrix'))
        else:
            result.append(dict(old,name=name))
    if {e['checkpoint_sha256'] for e in previous}-{e['checkpoint_sha256'] for e in result}:
        raise ValueError('a historical fixed model was omitted')
    return result,aliases


def verify_recipe(old,new):
    keys=['seed','parent_sha256','history_sha256','roles','local_num_envs','local_env_threads','num_actor_threads',
        'num_steps','num_minibatches','learning_rate','update_epochs','total_timesteps_per_arm','smoke_timesteps',
        'batch','formal_updates','train_opponent','mixed_actor_cycle','precision','jax_precision_environment',
        'full_legal_menu','max_step_loss','q_training_mode','full_history_capacity','longest_own_history']
    if any(old[k]!=new[k] for k in keys):
        raise ValueError('training recipe changed after the original exam; this is not a runtime-only revision')
    for role in old['roles']:
        left=read(Path(old['study_output'])/('rehearsal-'+role+'.json'))
        right=read(Path(new['study_output'])/('rehearsal-'+role+'.json'))
        fields=['id','group','controlled','teaching_steps','ce_coef','kl_coef','kind','priority']
        extract=lambda config:[[row[k] for k in fields] for row in config['rows']]
        if extract(left)!=extract(right):
            raise ValueError('rehearsal samples, masks or coefficients changed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','ppo-protocol','historical-ppo-protocol','previous-exam','historical-bridge','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();new=read(a.ppo_protocol);old=read(a.historical_ppo_protocol)
    previous=read(a.previous_exam/'protocol.json');verify_recipe(old,new)
    if previous['ppo_protocol_sha256']!=sha(a.historical_ppo_protocol):
        raise ValueError('previous fixed endpoints do not belong to the historical PPO protocol')
    release=Path(new['release']);verify_runtime(release)
    prep=read(release/'preparation.json')
    if prep.get('core_rule_revision')!='explicit-nonlevel-material-v1':
        raise ValueError('corrected-rule runtime required')
    rule_report_path=release.parent/'real-card-qualification/report.json'
    rule_report=read(rule_report_path)
    if (not rule_report['passed'] or rule_report['cases']!=48 or rule_report['positive_link1_cases']!=12 or
            rule_report['fixed_native_sha256']!=prep['diagnostic_sha256']):
        raise ValueError('actual real-card positive/negative qualification missing')
    course=read(Path(new['course'])/'plan.json');exam=read(Path(new['independent_exam'])/'plan.json')
    bridge=read(a.historical_bridge)
    if (not bridge['passed'] or bridge['native_sha256']!=prep['diagnostic_sha256'] or
            bridge['course_plan_sha256']!=sha(Path(new['course'])/'plan.json') or
            bridge['source_sha256']!=sha(PROJECT/'scripts/studies/reality_loop/audit_historical_actor_paths.py') or
            len(bridge['records'])!=bridge['unique_records'] or not all(r['passed'] for r in bridge['records'])):
        raise ValueError('complete historical SFT path/input bridge required')
    expected_summaries=[]
    for name in ('round1','round2'):
        expected_summaries.append(run/name/'evaluation/summary.json')
        for role in ('control','repair'):
            expected_summaries += [run/(name+'-training')/role/'before/summary.json',run/(name+'-training')/role/'endpoint/evaluation/summary.json']
    if set(bridge['summaries'])!={str(path) for path in expected_summaries} or not all(sha(path)==value for path,value in bridge['summaries'].items()):
        raise ValueError('historical SFT summary denominator changed')
    expected_paths=sorted(str(path.parent/row['path']) for path in expected_summaries for row in read(path)['results'])
    actual_paths=sorted(path for row in bridge['records'] for path in row['paths'])
    if actual_paths!=expected_paths or len(expected_paths)!=bridge['source_rows']:
        raise ValueError('a historical SFT path was omitted')
    for plan in (course,exam):
        validate_curriculum_binding(plan)
        if not plan['core_rule_reencoding']['all_known_inputs_identical'] or not plan['core_rule_reencoding']['all_rows_retained']:
            raise ValueError('known teaching inputs or denominator changed')
    if len(exam['rows'])!=6 or any(r['split']!='holdout' for r in exam['rows']):
        raise ValueError('all six admitted independent rows required')
    initial=lambda values:[(r['name'],r['checkpoint_sha256']) for r in values]
    if initial(new['pre_ppo_endpoints'])!=initial(old['pre_ppo_endpoints']):
        raise ValueError('a fixed SFT endpoint changed')
    historical=terminal_endpoints(a.historical_ppo_protocol)
    if initial(previous['endpoints'])!=initial([*old['pre_ppo_endpoints'],*historical]):
        raise ValueError('original seven-endpoint obligation changed')
    endpoints,aliases=retain_historical([*new['pre_ppo_endpoints'],*terminal_endpoints(a.ppo_protocol)],historical)
    for endpoint in [*endpoints,*aliases]:
        if sha(endpoint['checkpoint'])!=endpoint['checkpoint_sha256']:
            raise ValueError('a fixed checkpoint changed')
    metadata={endpoint['name']:load_checkpoint_metadata(endpoint['checkpoint']) for endpoint in [*endpoints,*aliases]}
    reference=metadata['parent']
    for name,value in metadata.items():
        if any(value[k]!=reference[k] for k in REQUIRED_STRUCTURED_FIELDS):
            raise ValueError('fixed model or alias has incompatible inference metadata: '+name)
    if reference['code_list_hash']!=course['code_list_sha256'] or reference['semantic_table_hash']!=course['semantics_sha256']:
        raise ValueError('model vocabulary or semantic inputs differ from the corrected course')
    coverage=intake_coverage(run,course,exam)
    paths=[a.ppo_protocol,a.historical_ppo_protocol,a.previous_exam/'protocol.json',Path(new['course'])/'plan.json',
        Path(new['independent_exam'])/'plan.json',release/'frozen-manifest.json',Path(__file__),
        PROJECT/'scripts/studies/reality_loop/ppo_exams.py',PROJECT/'scripts/studies/reality_loop/duel_exam.py',
        PROJECT/'scripts/studies/crystron_curriculum/evaluate_supervision.py',PROJECT/'scripts/studies/crystron_curriculum/supervision_runtime.py',
        rule_report_path,a.historical_bridge,PROJECT/'scripts/studies/reality_loop/audit_historical_actor_paths.py',
        *[metadata_path(e['checkpoint']) for e in [*endpoints,*aliases]]]
    protected={str(path.resolve()):sha(path) for path in paths}
    for config in (old,new):
        protected.update(config['protected'])
        for role in config['roles']:
            for filename in ('verification.json','rehearsal-verification.json'):
                path=Path(config['study_output'])/role/filename;protected[str(path.resolve())]=sha(path)
    protected.update(coverage['protected'])
    protected.update(bridge['summaries'])
    protected.update({path:row['source_record_sha256'] for row in bridge['records'] for path in row['paths']})
    if not all(sha(Path(path))==value for path,value in protected.items()):
        raise ValueError('protected prior evidence changed')
    development=['parent','control-final','rehearse-final']+[e['name'] for e in endpoints if e['name'].startswith('historical-')]
    protocol=dict(format_version=1,ppo_protocol=str(a.ppo_protocol.resolve()),ppo_protocol_sha256=sha(a.ppo_protocol),
        historical_ppo_protocol=str(a.historical_ppo_protocol.resolve()),historical_ppo_protocol_sha256=sha(a.historical_ppo_protocol),
        previous_exam_protocol=str((a.previous_exam/'protocol.json').resolve()),previous_exam_protocol_sha256=sha(a.previous_exam/'protocol.json'),
        historical_actor_bridge=str(a.historical_bridge.resolve()),historical_actor_bridge_sha256=sha(a.historical_bridge),
        endpoints=endpoints,checkpoint_aliases=aliases,all_original_fixed_models_retained=True,
        independent_exam=new['independent_exam'],independent_all_rows=6,independent_intake_coverage=coverage,
        development_course=new['course'],release=new['release'],protected=protected,development_endpoints=development,
        natural_duels=dict(planned_per_endpoint=256,decks=4,seeds=32,seats=[0,1],
            opponent=new['evaluation']['opponent_checkpoint'],qualification_prefix=4),
        phases=['development','qualification','independent','natural'],precision=new['precision'],
        jax_precision_environment=new['jax_precision_environment'],invalid_and_budget_are_unknown=True,
        independent_score_checkpoint_selection=False,promotion_eligible=False,
        exposure_scope='Original independent components; the initial-model old-runtime attempt was already observed and failed its audit. This is a registered rules correction with unchanged model starts, training recipe and fixed endpoints, not a newly untouched holdout. No labels from this set enter training.')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'protocol.json',protocol)
    env=dict(os.environ,JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    original_matrix=read(a.previous_exam/'parent-duel-plan/plan.json')
    matrix=None
    for endpoint in endpoints:
        name=endpoint['name'];target=a.output/(name+'-duel-plan')
        child([sys.executable,str(PROJECT/'scripts/studies/reality_loop/duel_exam.py'),'freeze',
            '--frozen',new['course'],'--candidate',endpoint['checkpoint'],'--opponent',protocol['natural_duels']['opponent'],
            '--output',str(target),'--decks','4','--seeds','32'],a.output,name+'-freeze',env)
        duel=read(target/'plan.json');cases=duel['cases']
        if len(cases)!=256 or len({c['id'] for c in cases})!=256:
            raise ValueError('full duel denominator missing')
        if matrix is None:matrix=cases
        elif cases!=matrix:raise ValueError('endpoint matrices are not exactly paired')
        if any(duel[k]!=original_matrix[k] for k in ('cases','budget','precision','opponent_sha256')):
            raise ValueError('original duel matrix, opponent or budget changed')
    ledger(run,'corrected_rule_complete_exam_protocol',[a.output/'protocol.json'],
        unique_endpoints=len(endpoints),explicit_aliases=len(aliases),
        claim='All original fixed SFT/PPO weights and both corrected-rule final PPO weights retained; all independent rows and 256 paired both-seat natural games per unique weight. No selection by scores.')
    print('CORRECTED_RULE_EXAMS_FROZEN',len(endpoints),'unique',len(aliases),'aliases',flush=True)


if __name__=='__main__':main()
