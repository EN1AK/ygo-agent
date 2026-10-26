"""Compare the four frozen response roots with each endpoint's fresh own memory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','teaching','training','assessment','audit','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for key,value in vars(a).items():setattr(a,key,value.resolve())
    sys.path[:0]=[str(a.source),str(a.source/'scripts')]
    import jax,numpy as np,flax.serialization
    from scripts.teach_elfnote_combo import load_row
    from scripts.collect_teaching_exercise import load_native
    from scripts.eval_capability_exercises import FrozenActor
    from scripts.study_elfnote_recovery import checked
    assert all(d.platform=='gpu' for d in jax.devices()) and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='float32'
    assert Path(str(a.assessment)+'.exit').read_text().strip()==Path(str(a.training)+'.formal.exit').read_text().strip()==Path(str(a.teaching)+'.train.exit').read_text().strip()=='0'
    audit=read(a.audit);assert audit['passed'] and audit['attempts']==2048 and read(a.audit.with_name(a.audit.stem+'.exit.json'))['returncode']==0
    proof=read(a.teaching/'final-verification.json');trained=read(a.training/'after_defense/verification.json');assert proof['passed'] and trained['passed'] and trained['process_exit_zero']
    models=dict(defense_initial=a.teaching/'teach/model.flax_model',after_defense=Path(trained['checkpoint']))
    assert all(sha(path)==audit['models'][name] for name,path in models.items())
    assert sha(models['defense_initial'])==proof['checkpoints']['teach']
    source=read(a.source/'source-manifest.json');original=read(a.teaching/'training-source-manifest.json')['source']
    assert all(hashlib.sha256((a.source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in source['all_files'].items())
    for name in ['scripts/study_elfnote_recovery.py','scripts/eval_capability_exercises.py','ygoai/rl/elfnote_recovery.py','ygoai/rl/jax/agent.py']:
        assert source['all_files'][name]==original['all_files'][name],name
    manifest=read(a.teaching/'manifest.json');plan=manifest['plan'];collection=read(a.teaching/'collection.json');assert len(collection['results'])==4
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008'
    a.native=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014/exercise_actor_native.cpython-310-x86_64-linux-gnu.so';assert sha(a.native)==plan['native_sha256']
    a.database=release/'repo/assets/locale/zh/cards.cdb';a.scripts=release/'repo/scripts/script';a.code_list=release/'code_list.txt';a.semantics=release/'semantics'
    assert all(sha(a.root/name)==value for name,value in manifest['runtime'].items())
    protected={str(path):sha(path) for path in [a.audit,a.teaching/'manifest.json',a.teaching/'final-verification.json',a.teaching/'collection.json',a.training/'after_defense/verification.json',*models.values()]}
    for row in collection['results']:
        path=Path(row['path']);assert sha(path)==row['record_sha256'] and sha(path.with_suffix('.npz'))==row['observations_sha256']
        for item in [path,path.with_suffix('.npz')]:protected[str(item)]=sha(item)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'manifest.json',dict(source=source,models={k:sha(v) for k,v in models.items()},protected=protected,runtime=manifest['runtime'],roots=[dict(case=r['case'],root_step=r['root_step'],source_record_sha256=r['record_sha256']) for r in collection['results']],frozen_before_inference=True,max_seconds=900,scope='Separate post-training diagnostic of all four original roots. Old legal actions reproduce the context; each model recomputes its own recurrent state. No prior hidden state, source masks, main evaluation score or checkpoint selection is reused or changed.'))
    sample=load_row(Path(collection['results'][0]['path']))[1]
    native=load_native(a,release/'opening-02/registration.ydk');actor=FrozenActor(models['defense_initial'],a.code_list,a.semantics,{k:v[:1] for k,v in sample.items()})
    template=actor.params;forward=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3]);actor.forward=lambda obs,state:forward(actor.params,obs,state)
    results=[];started=time.monotonic()
    for name,checkpoint in models.items():
        actor.params=flax.serialization.from_bytes(template,checkpoint.read_bytes());folder=a.output/name;folder.mkdir()
        for row in collection['results']:
            assert time.monotonic()-started<900
            reference=read(Path(row['path']));index=row['root_step'];case=next(c for c in manifest['functional_cases'] if c['name']==row['case'])
            record,core=checked(a,native,actor,case,plan,folder,row['case'],prefix=reference['decisions'][:index],probe_mode='policy')
            root=record['decisions'][index];expected=reference['decisions'][index]
            assert root['player']==0 and root['message']==16 and root['observation_sha256']==expected['observation_sha256'] and root['menu']==expected['menu']
            assert root['action']==max(range(len(root['logits'])),key=lambda i:root['logits'][i])
            results.append(dict(model=name,case=row['case'],status=record['status'],reference_root_step=index,current_action=root['action'],current_menu=root['menu'][root['action']],current_rnn_sha256=root['rnn_sha256'],**{k:v for k,v in core.items() if k!='status'}))
    audits=list(a.output.glob('*/*.audit.json'));assert len(audits)==8
    for path in audits:
        core=read(path);record=path.with_name(path.name.replace('.audit.json','.json'));assert core['passed'] and core['record_sha256']==sha(record) and core['observations_sha256']==sha(record.with_suffix('.npz'))
    assert all(sha(Path(path))==value for path,value in protected.items()) and all(sha(a.root/name)==value for name,value in manifest['runtime'].items())
    scores={name:sum(r['model']==name and r['status']=='success' for r in results) for name in models}
    assert scores['defense_initial']==4,'Shared diagnostic entry must reproduce the frozen teaching reference before interpreting post-PPO loss'
    write(a.output/'summary.json',dict(passed=True,scores=scores,results=results,new_core_audits=8,seconds=time.monotonic()-started,training_started=False,production_default_replaced=False,scope='Fixed-context response retention only. This does not substitute for autonomous opening, full-game or transfer scores.'))
    print(json.dumps(dict(passed=True,scores=scores,new_core_audits=8)),flush=True)


if __name__=='__main__':main()
