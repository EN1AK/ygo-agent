"""No-fitting functional recovery intake, after root teaching has completely stopped."""
import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from scripts.collect_teaching_exercise import load_native
from scripts.eval_capability_exercises import FrozenActor
from scripts.study_elfnote_coverage import read,write
from scripts.teach_elfnote_combo import load_row,save,audit_actor
from ygoai.rl.elfnote_recovery import run,CHALLENGE,NEGATORS,challenge_evidence
from ygoai.rl.elfnote_search import discover
from ygoai.rl.exercise_core import sha


def cases_for(plan,previous):
    cases=[]
    for pair in plan['pairs']:
        for response,card in plan['responses'].items():
            case=copy.deepcopy(next(c for c in previous if c['name']==pair+'-'+response))
            case.update(name=pair+'-'+response+'-functional',split='development',lineage='elfnote-exposed-development')
            deck=[x for x in case['definition']['initial'] if x[1:3]==[1,1]]
            deck[-2][0]=CHALLENGE;deck[-1][0]=card
            assert max(Counter(x[0] for x in deck).values())<=3 and len(deck)==40
            case['goals']=[dict(monsters=[code],spells=[],hand=plan['minimum_hand']) for code in NEGATORS]
            cases.append(case)
    assert len(cases)==4
    return cases


def checked(a,native,actor,case,plan,folder,name,*,prefix=(),baseline=None,probe_mode='scripted'):
    record,obs=run(native,a.semantics,actor,case,plan,prefix=prefix,baseline=baseline,probe_mode=probe_mode)
    again,arr=run(native,a.semantics,actor,case,plan,prefix=prefix,baseline=baseline,probe_mode=probe_mode)
    assert record==again and all(np.array_equal(obs[k],arr[k]) for k in obs)
    path=save(folder,name,record,obs);proof=audit_actor(a,path);assert proof['passed']
    assert all(not d['teaching'] and not d['searchable'] for d in record['decisions'][:record['root_step'] or 0])
    if record['probe_start_step'] is not None:
        assert all(not d['teaching'] and not d['searchable'] for d in record['decisions'][record['probe_start_step']:])
    if record['status']=='success':
        assert record['intervention_attempted'] and record['root_step'] is not None and record['functional_evidence']['success']
        assert record['board_at_probe']['resources']['hand']>=plan['minimum_hand']
    return record,dict(path=str(path),record_sha256=sha(path),observations_sha256=sha(path.with_suffix('.npz')),status=record['status'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.root,a.source,a.output=a.root.resolve(),a.source.resolve(),a.output.resolve()
    a.plan=a.source/'assets/exercises/teaching-v1/elfnote-functional-recovery.json';plan=read(a.plan)
    study=a.root/plan['root_study'];assert read(study/'final-verification.json')['passed']
    assert Path(str(study)+'.verification.exit').read_text().strip()=='0'
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008';intake=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014'
    a.native=intake/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so';a.database=release/'repo/assets/locale/zh/cards.cdb';a.scripts=release/'repo/scripts/script'
    a.code_list,a.semantics=release/'code_list.txt',release/'semantics'
    parent=a.root/'training-runs/elfnote-coverage-study-20261023/varied/model.flax_model'
    assert sha(parent)==plan['parent_sha256'] and sha(a.native)==plan['native_sha256']
    assert (a.scripts/f'c{CHALLENGE}.lua').is_file() and str(CHALLENGE) in a.code_list.read_text().split()
    checkpoints={'parent':parent,**{mode:study/mode/'model.flax_model' for mode in plan['models'] if mode!='parent'}}
    model_proof=read(study/'summary.json')
    for arm in model_proof['arms']:assert sha(checkpoints[arm['arm']['name']])==arm['checkpoint_sha256']
    protected={str(path):sha(path) for path in [a.plan,a.native,a.database,*checkpoints.values()]}
    source=read(a.source/'source-manifest.json')
    assert all(hashlib.sha256((a.source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in source['all_files'].items())
    runtime=read(a.root/'training-runs/elfnote-runtime-binding-20261014.json');assert all(sha(a.root/name)==value for name,value in runtime.items())
    cases=cases_for(plan,read(a.root/plan['definition_source'])['cases'])
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'manifest.json',dict(plan=plan,source=source,protected=protected,runtime=runtime,cases=cases,frozen_before_inference=True,optimizer_used=False))
    import jax
    import flax.serialization
    jax.config.update('jax_default_matmul_precision','highest');assert all(d.platform=='gpu' for d in jax.devices())
    sample=load_row(Path(read(study/'training-rows.json')['rows'][0]['path']))[1]
    native=load_native(a,release/'opening-02/registration.ydk');actor=FrozenActor(parent,a.code_list,a.semantics,{k:v[:1] for k,v in sample.items()})
    infer=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3]);actor.forward=lambda obs,state:infer(actor.params,obs,state)
    parent_params=actor.params;started=time.monotonic();evaluations=[];baselines={}
    for mode in plan['models']:
        actor.params=flax.serialization.from_bytes(parent_params,checkpoints[mode].read_bytes())
        folder=a.output/mode;folder.mkdir()
        for case in cases:
            for probe in plan['probe_modes']:
                record,result=checked(a,native,actor,case,plan,folder,case['name']+'-'+probe,probe_mode=probe)
                evaluations.append(dict(model=mode,case=case['name'],probe=probe,**result))
                if mode=='parent' and probe=='scripted':baselines[case['name']]=record
        print(json.dumps(dict(stage='functional_evaluation',model=mode,scores={probe:sum(r['status']=='success' for r in evaluations if r['model']==mode and r['probe']==probe) for probe in plan['probe_modes']})),flush=True)
    actor.params=parent_params;folder=a.output/'discovery';folder.mkdir();discoveries=[]
    for case in cases:
        assert time.monotonic()-started<plan['max_collection_seconds']
        baseline=baselines[case['name']]
        if baseline['root_step'] is None:
            result=dict(case=case['name'],status='intervention_not_observed',trials=[],decisions=0,seconds=0)
        else:
            with (folder/(case['name']+'.trials.jsonl')).open('x') as f:
                def callback(row):f.write(json.dumps(row)+'\n');f.flush()
                found,result=discover(native,a.semantics,actor,case,plan,baseline,'search',callback,runner=run)
            if found:
                witness,proof=checked(a,native,actor,case,plan,folder,case['name']+'-witness',prefix=found['decisions'],baseline=baseline)
                assert witness['status']=='success';result.update(proof)
                negative,negative_proof=checked(a,native,actor,case,plan,folder,case['name']+'-pass-control',prefix=witness['decisions'][:witness['probe_start_step']],baseline=baseline,probe_mode='pass')
                assert negative['status']=='goal_not_reached' and not negative['functional_evidence']['success']
                result['negative_control']=negative_proof
                result['post_response_own_choices']=sum(d['teaching'] for d in witness['decisions'])
        discoveries.append(result);write(folder/(case['name']+'.summary.json'),result)
        print(json.dumps(dict(stage='functional_discovery',case=case['name'],status=result['status'],trials=len(result['trials']))),flush=True)
    assert all(sha(Path(path))==value for path,value in protected.items())
    write(a.output/'summary.json',dict(completed=True,evaluations=evaluations,discoveries=discoveries,seconds=time.monotonic()-started,
          optimizer_used=False,production_eligible=False,protected_unchanged=True))


if __name__=='__main__':main()
