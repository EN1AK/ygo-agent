"""Audit functional alternatives before deciding which old goals need rehearsal."""
import argparse,json,os,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','plan','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.root=a.root.resolve();a.output=a.output.resolve();a.source=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(a.source),str(a.source/'scripts')]
    import jax,numpy as np
    from scripts.study_elfnote_coverage import read,write
    from scripts.collect_teaching_exercise import load_native
    from scripts.eval_capability_exercises import FrozenActor
    from scripts.teach_elfnote_combo import load_row,save,audit_actor
    from ygoai.rl.terminal_probe import run
    from ygoai.rl.exercise_core import sha
    import sqlite3,hashlib
    source=read(a.source/'source-manifest.json')
    assert all(hashlib.sha256((a.source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in source['all_files'].items())
    plan=read(a.plan);assert all(d.platform=='gpu' for d in jax.devices()) and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='float32'
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008'
    a.native=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014/exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    a.database=release/'repo/assets/locale/zh/cards.cdb';a.scripts=release/'repo/scripts/script';a.code_list=release/'code_list.txt';a.semantics=release/'semantics'
    assert sha(a.native)==plan['native_sha256']
    runtime=read(a.root/'training-runs/elfnote-runtime-binding-20261014.json');assert all(sha(a.root/name)==value for name,value in runtime.items())
    codes={line.split()[0] for line in a.code_list.read_text().splitlines() if line.strip()};names={}
    with sqlite3.connect(a.database) as db:
        for c in plan['challenges']:
            assert str(c['code']) in codes and (a.scripts/f'c{c["code"]}.lua').is_file()
            names[c['name']]=db.execute('select name from texts where id=?',(c['code'],)).fetchone()[0]
    course=a.root/plan['course'];m=read(course/'manifest.json');cases=m['normal_cases']
    train=read(a.root/plan['training']/'plain/verification.json');assert train['passed']
    models=dict(initial=course/'full_context/model.flax_model',plain=Path(train['checkpoint']))
    assert all(sha(path)==plan[name+'_sha256'] for name,path in models.items())
    paths={};protected={str(a.plan):sha(a.plan),str(a.native):sha(a.native)}
    for model in models:
        parent=a.root/plan[model+'_records'];assert read(parent/'final-verification.json')['passed']
        for case in plan['cases']:
            index=next(i for i,c in enumerate(cases) if c['name']==case);path=parent/'normal'/f'case-{index:02}.json'
            proof=read(path.with_name(path.stem+'.audit.json'));assert proof['passed'] and proof['record_sha256']==sha(path) and proof['observations_sha256']==sha(path.with_suffix('.npz'))
            paths[(model,case)]=path;protected[str(path)]=sha(path);protected[str(path.with_suffix('.npz'))]=sha(path.with_suffix('.npz'))
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'manifest.json',dict(plan=plan,protected=protected,runtime=runtime,source=source,challenge_names=names,frozen_before_inference=True))
    native=load_native(a,release/'opening-02/registration.ydk');sample=load_row(next(iter(paths.values())))[1]
    actor=FrozenActor(models['initial'],a.code_list,a.semantics,{k:v[:1] for k,v in sample.items()})
    import flax.serialization
    template=actor.params;forward=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3]);actor.forward=lambda obs,state:forward(actor.params,obs,state)
    results=[];started=time.time()
    for model,checkpoint in models.items():
        actor.params=flax.serialization.from_bytes(template,checkpoint.read_bytes())
        for case in plan['cases']:
            original=read(paths[(model,case)])
            for challenge in plan['challenges']:
                folder=a.output/f'{model}-{case}-{challenge["name"]}';folder.mkdir()
                for mode in ['scripted','pass','policy']:
                    record,obs=run(native,a.semantics,actor,original,challenge,plan,mode)
                    again,arrays=run(native,a.semantics,actor,original,challenge,plan,mode)
                    assert record==again and all(np.array_equal(obs[k],arrays[k]) for k in obs)
                    path=save(folder,mode,record,obs);proof=audit_actor(a,path);assert proof['passed']
                    row=dict(model=model,case=case,challenge=challenge['name'],mode=mode,status=record['status'],prefix_own_verified=record['prefix_own_observations_verified'],error=record['error'],evidence=record['functional_evidence'])
                    results.append(row);print(json.dumps(row),flush=True)
    assert all(sha(Path(p))==value for p,value in protected.items()) and all(sha(a.root/p)==value for p,value in runtime.items())
    assert all(hashlib.sha256((a.source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in source['all_files'].items())
    scores={model:{c['name']:{mode:sum(r['model']==model and r['challenge']==c['name'] and r['mode']==mode and r['status']=='success' for r in results) for mode in ['scripted','pass','policy']} for c in plan['challenges']} for model in models}
    write(a.output/'summary.json',dict(passed=True,results=results,scores=scores,audits=len(results),seconds=time.time()-started,scope=plan['scope'],production_default_replaced=False))
    print(json.dumps(dict(passed=True,scores=scores)),flush=True)


if __name__=='__main__':main()
