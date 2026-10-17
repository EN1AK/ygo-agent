"""Frozen-parent, known-feasibility coverage gate; never imports answer routes."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scripts.collect_teaching_exercise import load_native
from scripts.eval_capability_exercises import FrozenActor
from scripts.teach_elfnote_combo import audit_actor, load_row, save
from ygoai.rl.elfnote_search import discover, run
from ygoai.rl.exercise_core import sha


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def write(path, value):
    with path.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, ensure_ascii=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--platform',choices=['cpu','gpu'],required=True)
    args=p.parse_args()
    root,source,output=(getattr(args,k).resolve() for k in ['root','source','output'])
    output.mkdir(parents=True,exist_ok=False)
    plan_path=source/'assets/exercises/teaching-v1/elfnote-diverse-search.json';plan=read(plan_path)
    source_manifest=read(source/'source-manifest.json')
    assert all(hashlib.sha256((source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value
               for name,value in source_manifest['all_files'].items())
    release=root/'dist/runtime-releases/exercise-study-e9994b4-20261008'
    intake=root/'training-runs/elfnote-selected-scope-intake-03-20261014'
    previous=root/'training-runs/elfnote-search-valid-scope-20261014'
    assert read(previous/'final-verification.json')['passed']
    parent=root/'training-runs/elfnote-scope-joint-calibration-20261014/teach_retain/model.flax_model'
    a=SimpleNamespace(native=intake/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so',
          database=release/'repo/assets/locale/zh/cards.cdb',scripts=release/'repo/scripts/script',
          code_list=release/'code_list.txt',semantics=release/'semantics')
    assert sha(parent)==plan['parent_sha256'] and sha(a.native)==plan['native_sha256']
    runtime=read(root/'training-runs/elfnote-runtime-binding-20261014.json')
    # Runtime binding may be a direct path/hash mapping or a structured sidecar.
    runtime_files=runtime.get('files',runtime)
    assert isinstance(runtime_files,dict)
    for name,value in runtime_files.items():
        assert sha(root/name)==value,name
    protected={str(path):sha(path) for path in [parent,a.native,a.database,plan_path]}
    # Definitions only: no previous trajectory, witness or action string is read.
    cases=[c for c in read(previous/'manifest.json')['cases'] if c['name'] in plan['gate_cases']]
    cases.sort(key=lambda c:plan['gate_cases'].index(c['name']))
    assert len(cases)==2 and all(not c['intervention'] for c in cases)
    original=read(intake/'joint-calibration-plan.json')['demonstrations'][0]['name']
    _,sample=load_row(intake/'actor-v1'/(original+'.json'))
    import jax
    jax.config.update('jax_default_matmul_precision','highest')
    assert all(d.platform==args.platform for d in jax.devices())
    started=time.monotonic()
    native=load_native(a,release/'opening-02/registration.ydk')
    actor=FrozenActor(parent,a.code_list,a.semantics,{k:v[:1] for k,v in sample.items()})
    write(output/'manifest.json',dict(plan=plan,protected=protected,source=source_manifest,
         platform=args.platform,cases=cases,scope='exposed feasibility gate, no answers imported, no optimizer'))
    results=[]
    for case in cases:
        baseline,obs=run(native,a.semantics,actor,case,plan)
        baseline_path=save(output,case['name']+'-baseline',baseline,obs)
        audit_actor(a,baseline_path)
        for strategy in plan['gate_strategies']:
            current=copy.deepcopy(plan);current['search_strategy']=strategy
            stem=case['name']+'-'+strategy
            with (output/(stem+'.trials.jsonl')).open('x') as f:
                def callback(row):f.write(json.dumps(row)+'\n');f.flush()
                witness,result=discover(native,a.semantics,actor,case,current,baseline,'search',callback)
            result['strategy']=strategy
            if witness:
                record,arrays=run(native,a.semantics,actor,case,plan,prefix=witness['decisions'],baseline=baseline)
                repeated,arrays2=run(native,a.semantics,actor,case,plan,prefix=witness['decisions'],baseline=baseline)
                assert record==repeated and all(np.array_equal(arrays[k],arrays2[k]) for k in arrays)
                assert record['status']=='success'
                path=save(output,stem,record,arrays);audit_actor(a,path)
                result.update(witness=str(path),record_sha256=sha(path),npz_sha256=sha(path.with_suffix('.npz')))
            write(output/(stem+'.summary.json'),result)
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='trials'}),flush=True)
    assert all(sha(Path(path))==value for path,value in protected.items())
    passed=all(any(r['case']==name and r['strategy']=='rank_progress_mix' and 'witness' in r
                   for r in results) for name in plan['gate_cases'])
    write(output/'summary.json',dict(completed=True,gate_passed=passed,results=results,
         protected_unchanged=True,seconds=time.monotonic()-started,platform=args.platform,
         next_step='fresh conditional discovery then matched teaching' if passed else 'stop before broad collection/teaching; inspect coverage'))
    print(json.dumps(dict(completed=True,gate_passed=passed)),flush=True)


if __name__=='__main__':main()
