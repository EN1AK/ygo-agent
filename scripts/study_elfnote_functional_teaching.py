"""Evidence-gated functional curriculum, frozen variants, matched sequence teaching."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
from scripts.collect_teaching_exercise import load_native
from scripts.eval_capability_exercises import FrozenActor
from scripts.study_elfnote_coverage import read,write
from scripts.study_elfnote_recovery import checked
from scripts.study_elfnote_search import evaluate as normal_evaluate
from scripts.teach_elfnote_combo import load_row
from ygoai.rl.elfnote_curriculum import make_cases,correction_row
from ygoai.rl.elfnote_recovery import run
from ygoai.rl.elfnote_search import discover
from ygoai.rl.exercise_core import sha


def functional_evaluate(a,native,actor,cases,plan,folder,*,audit=False):
    folder.mkdir();results=[]
    for case in cases:
        for mode in ['scripted','policy']:
            name=case['name']+'-'+mode
            if audit:record,proof=checked(a,native,actor,case,plan,folder,name,probe_mode=mode)
            else:
                record,_=run(native,a.semantics,actor,case,plan,probe_mode=mode)
                proof={}
            proof={k:v for k,v in proof.items() if k!='status'}
            results.append(dict(name=case['name'],split=case['split'],mode=mode,status=record['status'],root_step=record['root_step'],probe_start_step=record['probe_start_step'],**proof))
    result=dict(cases=results,scores={split:{mode:sum(r['status']=='success' for r in results if r['split']==split and r['mode']==mode)
        for mode in ['scripted','policy']} for split in sorted({c['split'] for c in cases})})
    result['availability']={split:{mode:dict(total=sum(r['split']==split and r['mode']==mode for r in results),response_observed=sum(r['split']==split and r['mode']==mode and r['root_step'] is not None for r in results)) for mode in ['scripted','policy']} for split in result['scores']}
    write(folder/'summary.json',result);print(json.dumps(dict(stage=str(folder),scores=result['scores'])),flush=True)
    return result


def collect(a,native,actor,plan,cases,normal_cases,normal_plan,old):
    started=time.monotonic()
    baseline=functional_evaluate(a,native,actor,cases,plan,a.output/'before',audit=True)
    normal=normal_evaluate(a,native,actor,normal_cases,normal_plan,old,a.output/'normal-before',audit=True)
    folder=a.output/'lessons';folder.mkdir();results=[]
    for case in [c for c in cases if c['split']=='fit']:
        assert time.monotonic()-started<plan['max_collection_seconds']
        baseline_record=read(a.output/'before'/(case['name']+'-scripted.json'))
        if baseline_record['root_step'] is None:
            result=dict(case=case['name'],status='intervention_not_observed',trials=[],decisions=0,seconds=0)
        elif baseline_record['status']=='success':
            result=dict(case=case['name'],status='already_successful',trials=[],decisions=0,seconds=0)
        else:
            with (folder/(case['name']+'.trials.jsonl')).open('x') as log:
                def callback(row):log.write(json.dumps(row)+'\n');log.flush()
                found,result=discover(native,a.semantics,actor,case,plan,baseline_record,'search',callback,runner=run)
            if found:
                record,proof=checked(a,native,actor,case,plan,folder,case['name'],prefix=found['decisions'],baseline=baseline_record)
                _,steps=correction_row(record,{})
                control,control_proof=checked(a,native,actor,case,plan,folder,case['name']+'-pass',prefix=record['decisions'][:record['probe_start_step']],baseline=baseline_record,probe_mode='pass')
                assert control['status']=='goal_not_reached' and not control['functional_evidence']['success']
                first=steps[0];b,w=baseline_record['decisions'][first],record['decisions'][first]
                assert baseline_record['decisions'][:first]==record['decisions'][:first]
                assert all(b[k]==w[k] for k in ['menu','observation_sha256','logits','rnn_sha256'])
                result.update(proof,correction_steps=steps,negative_control=control_proof)
        results.append(result);write(folder/(case['name']+'.summary.json'),result)
        print(json.dumps(dict(stage='functional_lesson',case=case['name'],status=result['status'],steps=result.get('correction_steps'),trials=len(result['trials']))),flush=True)
    required={r['case'] for r in read(a.root/plan['intake']/'summary.json')['discoveries'] if 'path' in r}
    accepted={r['case'] for r in results if 'path' in r}
    write(a.output/'collection.json',dict(admitted=required<=accepted,required=sorted(required),results=results,before=baseline,normal_before=normal,seconds=time.monotonic()-started))


def training_rows(a,plan):
    previous=read(a.parent_study/'training-rows.json')['rows'];assert len(previous)==60
    root_lessons=read(a.parent_study/'collection.json')['results'];rows=[];provenance=[]
    for i,item in enumerate(previous):
        path=Path(item['path']);assert sha(path)==item['record_sha256'] and sha(path.with_suffix('.npz'))==item['observations_sha256']
        record,obs=load_row(path);record=copy.deepcopy(record)
        if i==35:record['definition']=dict(controlled=0)
        for step,d in enumerate(record['decisions']):
            d['teaching']=step in root_lessons[i-24]['correction_steps'] if 24<=i<26 else d['player']==record['definition']['controlled']
        rows.append((record,obs));provenance.append(dict(**item,role='prior_course' if i<26 else 'original_old_KL' if i<36 else 'successful_route_KL'))
    accepted=[r for r in read(a.output/'collection.json')['results'] if 'path' in r];assert 3<=len(accepted)<=4
    for item in accepted:
        path=Path(item['path']);record,obs=load_row(path)
        row,steps=correction_row(record,obs);assert steps==item['correction_steps']
        rows.append(row);provenance.append(dict(path=str(path),record_sha256=sha(path),observations_sha256=sha(path.with_suffix('.npz')),role='functional_correction',steps=steps))
    if len(rows)==63:
        rows.append(copy.deepcopy(rows[60]));provenance.append(dict(provenance[60],role='zero_weight_padding'))
    assert len(rows)==64
    write(a.output/'training-rows.json',dict(rows=provenance,accepted_functional=len(accepted),histories=64))
    return rows,len(accepted)


def train(a,native,actor,plan,cases,normal_cases,normal_plan,old):
    import flax.serialization
    from ygoai.rl.scene_optimizer import make_optimizer
    from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
    from ygoai.rl.jax.agent import ModelArgs
    assert read(a.output/'collection.json')['admitted']
    rows,count=training_rows(a,plan);parent=actor.params;arms=[]
    fit=[c for c in cases if c['split']=='fit']
    normal_probe=[c for c in normal_cases if c['split'] in ['repair','original'] or c['name'] in a.bases]
    for arm in plan['arms']:
        started=time.monotonic();folder=a.output/arm['name'];folder.mkdir();actor.params=parent
        cc=[1/24]*24+[.5]*2+[0.]*34+[arm['correction_coef']/count]*count+[0.]*(4-count)
        kc=[0.]*26+[.1]*10+[1/24]*24+[0.]*4
        update,state,admission=make_optimizer(actor,parent,rows,plan['sequence_steps'],plan['learning_rate'],cc,kc,check_pairs=[30,13])
        write(folder/'admission.json',admission)
        with (folder/'metrics.jsonl').open('x') as log:
            for step in range(1,plan['updates_per_arm']+1):
                assert time.monotonic()-started<plan['max_train_seconds_per_arm']
                actor.params,state,loss,terms,norm,finite=update(actor.params,state)
                metric=dict(update=step,loss=float(loss),ce=np.asarray(terms[0]).tolist(),kl=np.asarray(terms[1]).tolist(),grad_norm=float(norm))
                assert bool(finite) and np.isfinite([metric['loss'],metric['grad_norm']]+metric['ce']+metric['kl']).all()
                expected=float(np.dot(cc,metric['ce'])+np.dot(kc,metric['kl']))
                assert abs(metric['loss']-expected)/max(1.,abs(expected),abs(metric['loss']))<=8*np.finfo(np.float32).eps
                log.write(json.dumps(metric)+'\n');log.flush()
                if step in plan['probe_updates']:
                    functional_evaluate(a,native,actor,fit,plan,folder/f'probe-{step}')
                    normal_evaluate(a,native,actor,normal_probe,normal_plan,old,folder/f'normal-probe-{step}')
        checkpoint=folder/'model.flax_model';checkpoint.write_bytes(flax.serialization.to_bytes(actor.params))
        write_checkpoint_metadata(checkpoint,observation_schema='structured-lite-v1',model_args=ModelArgs(observation_schema='structured-lite-v1'),code_list_hash=sha(a.code_list),semantic_table_hash=sha(a.semantics/'metadata.json'),
            capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),training_context=dict(experiment=plan['id'],parent_sha256=plan['parent_sha256'],arm=arm,updates=plan['updates_per_arm'],promotion_eligible=False))
        actor.params=flax.serialization.from_bytes(parent,checkpoint.read_bytes())
        result=functional_evaluate(a,native,actor,cases,plan,folder/'endpoint',audit=True)
        normal=normal_evaluate(a,native,actor,normal_cases,normal_plan,old,folder/'normal-endpoint',audit=True)
        restored=[];root_folder=folder/'restored';root_folder.mkdir()
        for item in read(a.output/'collection.json')['results']:
            if 'path' not in item:continue
            case=next(c for c in fit if c['name']==item['case']);reference=read(Path(item['path']));first=item['correction_steps'][0]
            record,proof=checked(a,native,actor,case,plan,root_folder,case['name'],prefix=reference['decisions'][:first])
            restored.append(dict(case=case['name'],**proof))
        write(root_folder/'summary.json',restored)
        arms.append(dict(arm=arm,checkpoint_sha256=sha(checkpoint),result=result,normal=normal,restored=restored,seconds=time.monotonic()-started))
    write(a.output/'summary.json',dict(completed=True,arms=arms,production_default_replaced=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=['collect','train'],required=True);a=p.parse_args()
    a.root,a.source,a.output=a.root.resolve(),a.source.resolve(),a.output.resolve()
    a.plan=a.source/'assets/exercises/teaching-v1/elfnote-functional-teaching.json';plan=read(a.plan)
    a.parent_study=a.root/plan['parent_study'];a.checkpoint=a.parent_study/plan['parent_arm']/'model.flax_model'
    assert read(a.parent_study/'final-verification.json')['passed'] and read(a.root/plan['intake']/'final-verification.json')['passed']
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008'
    a.native=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014/exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    a.database=release/'repo/assets/locale/zh/cards.cdb';a.scripts=release/'repo/scripts/script';a.code_list=release/'code_list.txt';a.semantics=release/'semantics'
    assert sha(a.checkpoint)==plan['parent_sha256'] and sha(a.native)==plan['native_sha256']
    source=read(a.source/'source-manifest.json');assert all(hashlib.sha256((a.source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in source['all_files'].items())
    runtime=read(a.root/'training-runs/elfnote-runtime-binding-20261014.json');assert all(sha(a.root/name)==value for name,value in runtime.items())
    prior=read(a.parent_study/'manifest.json');normal_plan=prior['plan'];a.bases=prior['bases']
    normal_cases=prior['cases']
    old_paths=read(a.parent_study/'training-rows.json')['rows'][26:36]
    original_rows=[load_row(Path(r['path'])) for r in old_paths[:5]];old=[load_row(Path(r['path'])) for r in old_paths[5:]]
    from ygoai.rl.elfnote_combo import GOALS
    from scripts.study_elfnote_root_repair import score
    release_plan=read(a.root/'training-runs/elfnote-selected-scope-intake-03-20261014/joint-calibration-plan.json')
    original=[dict(name=e['name'],split='original',definition=r[0]['definition'],intervention=None,goals=[GOALS[e['goal']]]) for e,r in zip(release_plan['demonstrations'],original_rows)]
    normal_cases=normal_cases+original
    base=read(a.root/plan['intake']/'manifest.json')['cases'];previous=read(a.root/plan['definition_source'])['cases']
    cases=make_cases(plan,base,previous);assert len(cases)==20
    protected={str(path):sha(path) for path in [a.plan,a.checkpoint,a.native,a.database,a.parent_study/'final-verification.json',a.root/plan['intake']/'final-verification.json']}
    if a.stage=='collect':
        a.output.mkdir(parents=True,exist_ok=False)
        write(a.output/'manifest.json',dict(plan=plan,source=source,runtime=runtime,protected=protected,cases=cases,normal_cases=normal_cases,frozen_before_inference=True))
    else:
        m=read(a.output/'manifest.json');assert m['source']==source and m['plan']==plan and m['cases']==cases and m['protected']==protected
    import jax
    jax.config.update('jax_default_matmul_precision','highest');assert all(d.platform=='gpu' for d in jax.devices())
    native=load_native(a,release/'opening-02/registration.ydk');actor=FrozenActor(a.checkpoint,a.code_list,a.semantics,{k:v[:1] for k,v in original_rows[0][1].items()})
    infer=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3]);actor.forward=lambda obs,state:infer(actor.params,obs,state)
    if a.stage=='collect':collect(a,native,actor,plan,cases,normal_cases,normal_plan,old)
    else:train(a,native,actor,plan,cases,normal_cases,normal_plan,old)
    assert all(sha(Path(path))==value for path,value in protected.items())
    print(json.dumps(dict(stage=a.stage,completed=True,protected_unchanged=True)),flush=True)


if __name__=='__main__':main()
