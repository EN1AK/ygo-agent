"""Current-checkpoint root correction, with independently reconstructed route retention."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scripts.collect_teaching_exercise import load_native
from scripts.eval_capability_exercises import FrozenActor
from scripts.study_elfnote_coverage import read,write,variant,checked,digest
from scripts.study_elfnote_search import evaluate
from scripts.teach_elfnote_combo import load_row
from ygoai.rl.elfnote_combo import GOALS
from ygoai.rl.elfnote_search import run,discover,action_signature
from ygoai.rl.exercise_core import sha


def correction_indices(record):
    return [i for i,d in enumerate(record['decisions']) if d['player']==record['definition']['controlled']
            and action_signature(d['menu'][d['action']])!=action_signature(d['menu'][int(np.argmax(d['logits']))])]


def cases_for(plan,previous):
    cases=copy.deepcopy(previous)
    for case in cases:
        case['split']='repair' if case['name'] in plan['repair_cases'] else 'rehearsal' if case['split']=='train' else 'exposed_transfer'
    for i,name in enumerate(plan['repair_cases']):
        source=next(c for c in cases if c['name']==name)
        for j,settings in enumerate(plan['fresh_variants']):
            cases.append(variant(source,settings,f'{name}-new-{j}','fresh_repair',i))
    for i,settings in enumerate(plan['triples']):
        case=copy.deepcopy(next(c for c in cases if c['name']==settings['source']))
        case.update(name=f'triple-new-{i}',split='fresh_triple',lineage='elfnote-exposed-development')
        deck=[x for x in case['definition']['initial'] if x[1:3]==[0,1]]
        before=sorted(x[0] for x in deck)
        add=next(x for x in deck[:-5] if x[0]==settings['starter'])
        remove=next(x for x in deck[-5:] if x[0]==59438930)
        add[0],remove[0]=remove[0],add[0]
        case['definition']['seed_sequence']=np.random.default_rng(plan['seed']+100+i).integers(0,2**32,8).tolist()
        case['transformation']=dict(kind='registered_third_starter',**settings)
        case['definition_sha256']=digest(case['definition'])
        assert before==sorted(x[0] for x in deck)
        cases.append(case)
    assert len(cases)==48 and len({digest(c['definition']) for c in cases})==48
    return cases


def score(result):
    return {**{s:sum(c['status']=='success' for c in result['cases'] if c['split']==s)
               for s in ['repair','rehearsal','exposed_transfer','fresh_repair','fresh_triple','original']},
            'old':sum(x['success'] for x in result['old'])}


def collect(a,native,actor,plan,cases,originals,old):
    started=time.monotonic()
    before=evaluate(a,native,actor,cases+originals,plan,old,a.output/'before',audit=True)
    lookup={c['name']:i for i,c in enumerate(cases)}
    base_score=score(before)
    assert base_score['rehearsal']==24 and base_score['original']==base_score['old']==5
    folder=a.output/'corrections';folder.mkdir();results=[]
    for name in plan['repair_cases']:
        assert time.monotonic()-started<plan['max_collection_seconds']
        case=cases[lookup[name]];baseline=read(a.output/f'before/case-{lookup[name]:02}.json')
        assert baseline['status']!='success','repair already passed; freeze a new protocol rather than invent a fault'
        with (folder/(name+'.trials.jsonl')).open('x') as f:
            def callback(row):f.write(json.dumps(row)+'\n');f.flush()
            found,result=discover(native,a.semantics,actor,case,plan,baseline,'search',callback)
        if found:
            record,obs=run(native,a.semantics,actor,case,plan,prefix=found['decisions'])
            result.update(checked(a,native,actor,case,plan,record,obs,folder,name))
            indices=correction_indices(record);assert indices,'no verified semantic correction target'
            result['correction_steps']=indices
            first=indices[0];b=baseline['decisions'][first];w=record['decisions'][first]
            assert baseline['decisions'][:first]==record['decisions'][:first]
            assert all(b[k]==w[k] for k in ['observation_sha256','menu','logits','rnn_sha256'])
            branches=[]
            for label,prefix in [('original',baseline['decisions'][:first+1]),('replacement',record['decisions'][:first+1])]:
                r,arr=run(native,a.semantics,actor,case,plan,prefix=prefix)
                proof=checked(a,native,actor,case,plan,r,arr,folder,name+'-'+label)
                branches.append(dict(label=label,status=r['status'],**proof))
            result['paired_root']=dict(step=first,branches=branches,original=b['menu'][b['action']],replacement=w['menu'][w['action']])
        results.append(result);write(folder/(name+'.summary.json'),result)
        print(json.dumps(dict(stage='discover_correction',case=name,status=result['status'],steps=result.get('correction_steps'),trials=len(result['trials']))),flush=True)
    write(a.output/'collection.json',dict(admitted=all('path' in r for r in results),results=results,before=before,seconds=time.monotonic()-started))


def train(a,native,actor,plan,cases,originals,refs,old):
    import jax
    import flax.serialization
    from ygoai.rl.scene_optimizer import make_optimizer
    from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
    from ygoai.rl.jax.agent import ModelArgs
    collection=read(a.output/'collection.json');assert collection['admitted']
    previous=read(a.previous/'collection.json');m=read(a.output/'manifest.json')
    assert m['plan']==plan and m['cases']==cases
    lessons=[load_row(Path(x['path'])) for x in previous['lessons']]
    corrections=[]
    for result in collection['results']:
        path=Path(result['path']);record,obs=load_row(path)
        assert record['status']=='success' and sha(path)==result['record_sha256']
        steps=correction_indices(record);assert steps==result['correction_steps']
        for i,d in enumerate(record['decisions']):d['teaching']=i in steps
        corrections.append((record,obs))
    retained=[]
    for i,case in enumerate(cases):
        if case['split']!='rehearsal':continue
        path=a.output/f'before/case-{i:02}.json';record,obs=load_row(path)
        assert record['status']=='success'
        for d in record['decisions']:d['teaching']=d['player']==record['definition']['controlled']
        retained.append((record,obs))
    assert len(lessons)==len(retained)==24 and len(corrections)==2
    rows=lessons+corrections+refs+old+retained;assert len(rows)==60
    row_paths=[Path(x['path']) for x in previous['lessons']]+[Path(x['path']) for x in collection['results']]
    row_paths += a.reference_paths
    row_paths += [a.output/f'before/case-{i:02}.json' for i,c in enumerate(cases) if c['split']=='rehearsal']
    for path in row_paths:
        if path.parent==a.previous/'lessons' or path.parent==a.output/'corrections' or path.parent==a.output/'before':
            proof=read(path.with_name(path.stem+'.audit.json'))
            assert proof['passed'] and proof['record_sha256']==sha(path) and proof['observations_sha256']==sha(path.with_suffix('.npz'))
    row_manifest=dict(rows=[dict(path=str(path),record_sha256=sha(path),observations_sha256=sha(path.with_suffix('.npz'))) for path in row_paths],
          roles=dict(lesson=[0,24],correction=[24,26],old_retention=[26,36],successful_route_retention=[36,60]))
    if (a.output/'training-rows.json').exists():assert read(a.output/'training-rows.json')==row_manifest
    else:write(a.output/'training-rows.json',row_manifest)
    parent=actor.params;arms=[]
    probe_cases=[c for c in cases if c['split']=='repair' or c['name'] in m['bases']]+originals
    for arm in plan['arms']:
        started=time.monotonic();folder=a.output/arm['name'];folder.mkdir();actor.params=parent
        cc=[1/24]*24+[arm['correction_coef']/2]*2+[0.]*34
        kc=[0.]*26+[1/10]*10+[arm['route_retention_coef']/24]*24
        update,state,admission=make_optimizer(actor,parent,rows,plan['sequence_steps'],plan['learning_rate'],cc,kc,check_pairs=(12,13))
        write(folder/'admission.json',admission)
        with (folder/'metrics.jsonl').open('x') as f:
            for step in range(1,plan['updates_per_arm']+1):
                assert time.monotonic()-started<plan['max_train_seconds_per_arm']
                actor.params,state,loss,terms,norm,finite=update(actor.params,state)
                metric=dict(update=step,loss=float(loss),ce=np.asarray(terms[0]).tolist(),kl=np.asarray(terms[1]).tolist(),grad_norm=float(norm))
                assert bool(finite) and np.isfinite([metric['loss'],metric['grad_norm']]+metric['ce']+metric['kl']).all()
                f.write(json.dumps(metric)+'\n');f.flush()
                if step in plan['probe_updates']:
                    result=evaluate(a,native,actor,probe_cases,plan,old,folder/f'probe-{step}')
                    print(json.dumps(dict(arm=arm['name'],update=step,score=score(result))),flush=True)
        path=folder/'model.flax_model';path.write_bytes(flax.serialization.to_bytes(actor.params))
        write_checkpoint_metadata(path,observation_schema='structured-lite-v1',model_args=ModelArgs(observation_schema='structured-lite-v1'),
            code_list_hash=sha(a.code_list),semantic_table_hash=sha(a.semantics/'metadata.json'),capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),
            training_context=dict(experiment=plan['id'],parent_sha256=sha(a.checkpoint),arm=arm,updates=plan['updates_per_arm'],promotion_eligible=False))
        actor.params=flax.serialization.from_bytes(parent,path.read_bytes())
        result=evaluate(a,native,actor,cases+originals,plan,old,folder/'endpoint',audit=True)
        root_results=[];root_folder=folder/'restored-roots';root_folder.mkdir()
        for item in collection['results']:
            case=next(c for c in cases if c['name']==item['case']);reference=read(Path(item['path']));first=item['correction_steps'][0]
            record,obs=run(native,a.semantics,actor,case,plan,prefix=reference['decisions'][:first])
            proof=checked(a,native,actor,case,plan,record,obs,root_folder,item['case'])
            decision=record['decisions'][first] if first<len(record['decisions']) else None
            target=reference['decisions'][first]['menu'][reference['decisions'][first]['action']]
            match=decision is not None and action_signature(decision['menu'][decision['action']])==action_signature(target)
            root_results.append(dict(case=item['case'],status=record['status'],first_correction_matches=match,**proof))
        write(root_folder/'summary.json',root_results)
        arms.append(dict(arm=arm,result=result,restored_roots=root_results,score=score(result),checkpoint_sha256=sha(path),seconds=time.monotonic()-started))
    write(a.output/'summary.json',dict(completed=True,arms=arms,production_eligible=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=['collect','train'],required=True)
    p.add_argument('--collection-source',type=Path);a=p.parse_args()
    a.root,a.source,a.output=a.root.resolve(),a.source.resolve(),a.output.resolve()
    a.plan=a.source/'assets/exercises/teaching-v1/elfnote-root-repair.json';plan=read(a.plan)
    a.previous=a.root/plan['previous_study'];assert read(a.previous/'final-verification.json')['passed']
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008';intake=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014'
    a.native=intake/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so';a.database=release/'repo/assets/locale/zh/cards.cdb';a.scripts=release/'repo/scripts/script'
    a.code_list,a.semantics=release/'code_list.txt',release/'semantics';a.checkpoint=a.previous/'varied/model.flax_model'
    assert sha(a.native)==plan['native_sha256'] and sha(a.checkpoint)==plan['parent_sha256']
    source=read(a.source/'source-manifest.json')
    assert all(hashlib.sha256((a.source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in source['all_files'].items())
    runtime=read(a.root/'training-runs/elfnote-runtime-binding-20261014.json');assert all(sha(a.root/name)==value for name,value in runtime.items())
    oldplan=read(intake/'joint-calibration-plan.json');old_parent=a.root/'training-runs/elfnote-scope-joint-calibration-20261014/teach_retain'
    a.reference_paths=[intake/'actor-v1'/(e['name']+'.json') for e in oldplan['demonstrations']]+[old_parent/f'endpoint/old/{i}.json' for i in range(5)]
    refs=[load_row(path) for path in a.reference_paths[:5]];old=[load_row(path) for path in a.reference_paths[5:]]
    for i,(r,_) in enumerate(old):
        if i==4:r['definition']=dict(controlled=0)
        for d in r['decisions']:d['teaching']=d['player']==r['definition']['controlled']
    protected={str(path):sha(path) for path in [a.plan,a.native,a.database,a.checkpoint,a.previous/'manifest.json',a.previous/'collection.json',a.previous/'final-verification.json']}
    for path in a.reference_paths+[Path(x['path']) for x in read(a.previous/'collection.json')['lessons']]:
        for dependency in [path,path.with_suffix('.npz')]:protected[str(dependency)]=sha(dependency)
    cases=cases_for(plan,read(a.previous/'manifest.json')['cases']);bases=read(a.previous/'manifest.json')['bases']
    originals=[dict(name=e['name'],split='original',definition=r[0]['definition'],intervention=None,goals=[GOALS[e['goal']]]) for e,r in zip(oldplan['demonstrations'],refs)]
    if a.stage=='collect':
        a.output.mkdir(parents=True,exist_ok=False)
        write(a.output/'manifest.json',dict(plan=plan,plan_sha256=sha(a.plan),source=source,runtime=runtime,protected=protected,cases=cases,bases=bases,frozen_before_inference=True))
    else:
        m=read(a.output/'manifest.json');assert m['cases']==cases and m['plan']==plan
        if a.collection_source:
            original=a.collection_source.resolve();assert read(original/'source-manifest.json')==m['source']
            assert all(hashlib.sha256((original/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for path,value in m['source']['all_files'].items())
            changed={path for path,value in m['source']['all_files'].items() if source['all_files'].get(path)!=value}
            assert changed=={'ygoai/rl/scene_optimizer.py','scripts/study_elfnote_root_repair.py','scripts/verify_elfnote_root_repair.py'}
            assert source['all_files'].keys()==m['source']['all_files'].keys()
            previous_protected={p:v for p,v in m['protected'].items() if p!=str(original/'assets/exercises/teaching-v1/elfnote-root-repair.json')}
            current_protected={p:v for p,v in protected.items() if p!=str(a.plan)}
            assert previous_protected==current_protected and sha(a.plan)==m['plan_sha256']
            assert all(sha(Path(path))==value for path,value in m['protected'].items())
            assert read(a.output/'numerical-gate-diagnostic-01.json')['updates_executed']==0
            assert not (a.output/'continue').exists() and not (a.output/'repair').exists() and not (a.output/'repair_retain').exists()
            write(a.output/'training-source.json',dict(source=source,collection_source=str(original),training_source=str(a.source),protected=protected,
                  reason='Numerical qualification uses actual non-greedy correction examples as stable gradient denominator, same tolerances; arm objectives/data/budgets unchanged',changed_files=sorted(changed)))
        else:assert m['source']==source and m['protected']==protected
    import jax
    jax.config.update('jax_default_matmul_precision','highest');assert all(d.platform=='gpu' for d in jax.devices())
    native=load_native(a,release/'opening-02/registration.ydk');actor=FrozenActor(a.checkpoint,a.code_list,a.semantics,{k:v[:1] for k,v in refs[0][1].items()})
    infer=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3]);actor.forward=lambda obs,state:infer(actor.params,obs,state)
    if a.stage=='collect':collect(a,native,actor,plan,cases,originals,old)
    else:train(a,native,actor,plan,cases,originals,refs,old)
    assert all(sha(Path(path))==value for path,value in protected.items())
    print(json.dumps(dict(stage=a.stage,completed=True,protected_unchanged=True)),flush=True)


if __name__=='__main__':main()
