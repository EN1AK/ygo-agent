"""Reconstruct and admit four response-only lessons under a frozen new protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.root=a.root.resolve();a.source=a.source.resolve();a.output=a.output.resolve()
    sys.path[:0]=[str(a.source),str(a.source/'scripts')]
    import jax,numpy as np
    from scripts.study_elfnote_coverage import read,write
    from scripts.collect_teaching_exercise import load_native
    from scripts.eval_capability_exercises import FrozenActor
    from scripts.teach_elfnote_combo import load_row,save,audit_actor
    from scripts.study_elfnote_recovery import checked
    from ygoai.rl.exercise_core import sha
    from ygoai.rl.exercise_actor import array_digest,extract_trace,hidden_identity_audit
    from ygoai.rl.observation_schema import tensor_contract
    from ygoai.rl.defense_lessons import admit,validate_protocol
    plan_path=a.source/'assets/exercises/teaching-v1/elfnote-defense-teaching.json';plan=read(plan_path)
    validate_protocol(plan)
    execution=read(a.source/'source-manifest.json')
    assert all(hashlib.sha256((a.source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in execution['all_files'].items())
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='float32' and all(d.platform=='gpu' for d in jax.devices())
    baseline=a.root/plan['baseline'];candidates=a.root/plan['candidate_intake']
    assert sha(baseline/'final-verification.json')==plan['baseline_proof_sha256']
    proof=read(baseline/'final-verification.json');prior=read(baseline/'manifest.json')
    assert proof['passed'] and proof['retention_gate'] and prior['checkpoint_sha256']==proof['checkpoint_sha256']==plan['parent_sha256']
    assert proof['functional']['fit']==dict(scripted=4,policy=0)
    assert sha(candidates/'manifest.json')==plan['candidate_manifest_sha256']
    assert read(candidates/'manifest.json')['parent_sha256']==plan['parent_sha256']
    assert sha(a.root/plan['previous_audit'])==plan['previous_audit_sha256'] and read(a.root/plan['previous_audit'])['passed']
    assert sha(a.root/plan['base_curriculum'])==plan['base_curriculum_sha256']
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008'
    a.native=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014/exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    a.database=release/'repo/assets/locale/zh/cards.cdb';a.scripts=release/'repo/scripts/script';a.code_list=release/'code_list.txt';a.semantics=release/'semantics'
    a.checkpoint=a.root/plan['parent_study']/plan['parent_arm']/'model.flax_model'
    assert sha(a.native)==plan['native_sha256'] and sha(a.checkpoint)==plan['parent_sha256']
    runtime=read(a.root/'training-runs/elfnote-runtime-binding-20261014.json')
    assert all(sha(a.root/name)==value for name,value in runtime.items())
    protected={str(path):sha(path) for path in [plan_path,a.checkpoint,a.native,a.database,baseline/'manifest.json',baseline/'final-verification.json',
        candidates/'manifest.json',a.root/plan['previous_audit'],a.root/plan['base_curriculum']]}
    cases=[]
    for declared in plan['cases']:
        case=next(c for c in prior['functional_cases'] if c['name']==declared['name']);assert case['split']=='fit';cases.append(case)
        path=candidates/(case['name']+'.json');candidate=read(path)
        assert candidate['root_step']==declared['root_step'] and candidate['status']=='needs_current_model_reconstruction' and not candidate['training_ready']
        protected[str(path)]=sha(path)
        for item in candidate['sources']:
            source=Path(item['path']);assert sha(source)==item['record_sha256'];protected[str(source)]=sha(source)
            for other in [source.with_suffix('.npz'),source.with_name(source.stem+'.audit.json')]:protected[str(other)]=sha(other)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'manifest.json',dict(plan=plan,plan_sha256=sha(plan_path),source=execution,runtime=runtime,protected=protected,cases=cases,
        normal_cases=prior['normal_cases'],functional_cases=prior['functional_cases'],frozen_before_collection=True,training_started=False))
    raw=a.output/'raw';raw.mkdir();lessons=a.output/'lessons';lessons.mkdir();roots=a.output/'roots';roots.mkdir()
    native=load_native(a,release/'opening-02/registration.ydk')
    sample=load_row(baseline/'normal/case-00.json')[1]
    actor=FrozenActor(a.checkpoint,a.code_list,a.semantics,{k:v[:1] for k,v in sample.items()})
    forward=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3]);actor.forward=lambda obs,state:forward(actor.params,obs,state)
    def root_context(record,index):
        d=record['definition'];env=native.ExerciseActor('',d['initial'],str(a.semantics),1,'structured-lite-v1',d['seed_sequence'],d['duel_flags'],plan['max_decisions']+2)
        actor.reset()
        try:
            env.reset()
            for i in range(index+1):
                snap=env.snapshot();events,state,responses=extract_trace(env,snap);assert not snap['invalid']
                obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                if snap['player']==0:hidden_identity_audit(obs)
                logits,value,rnn=actor.predict(obs,snap['player']);expected=record['decisions'][i]
                assert (snap['player'],snap['message'],snap['menu'],array_digest(obs),rnn)==(expected['player'],expected['message'],expected['menu'],expected['observation_sha256'],expected['rnn_sha256'])
                assert np.array_equal(logits[:len(snap['menu'])],expected['logits']) and value==expected['value']
                assert len(responses)==expected['core_response_count']
                if i==index:
                    return dict(player=snap['player'],turn_player=snap['turn_player'],message=snap['message'],menu=snap['menu'],
                        observation_sha256=array_digest(obs),rnn_sha256=rnn,core_response_count=len(responses),events=events,
                        event_start=record['probe_event_start'],current_checkpoint_sha256=sha(a.checkpoint))
                env.step(expected['action'])
        finally:env.close()
    started=time.monotonic();results=[]
    for case in cases:
        assert time.monotonic()-started<plan['max_collection_seconds']
        positive,pp=checked(a,native,actor,case,plan,raw,case['name']+'-scripted',probe_mode='scripted')
        negative,npf=checked(a,native,actor,case,plan,raw,case['name']+'-pass',probe_mode='pass')
        candidate_path=candidates/(case['name']+'.json');candidate=read(candidate_path)
        context=root_context(positive,candidate['root_step']);root_path=roots/(case['name']+'.json');write(root_path,context)
        lesson,receipt=admit(positive,negative,context,candidate,plan)
        arrays=load_row(Path(pp['path']))[1]
        for i,row in enumerate(lesson['decisions']):
            if row['player']==0:hidden_identity_audit({k:v[i:i+1] for k,v in arrays.items()})
        path=save(lessons,case['name'],lesson,arrays);audit=audit_actor(a,path);assert audit['passed']
        receipt.update(path=str(path),record_sha256=sha(path),observations_sha256=sha(path.with_suffix('.npz')),
            raw_positive=pp,raw_negative=npf,candidate_path=str(candidate_path),candidate_sha256=sha(candidate_path),
            root_context_path=str(root_path),root_context_sha256=sha(root_path),lesson_audit_sha256=sha(path.with_name(path.stem+'.audit.json')),
            parent_sha256=plan['parent_sha256'],protocol_sha256=sha(plan_path))
        results.append(receipt);print(json.dumps(dict(case=case['name'],root=receipt['root_step'],status='admitted_response_only')),flush=True)
    assert len(results)==len(cases)==4 and all(sha(Path(path))==value for path,value in protected.items())
    assert all(sha(a.root/name)==value for name,value in runtime.items())
    write(a.output/'collection.json',dict(admitted=True,results=results,seconds=time.monotonic()-started,training_started=False))
    subprocess.run([sys.executable,str(a.source/'scripts/verify_elfnote_defense_lessons.py'),'--root',str(a.root),'--source',str(a.source),'--study',str(a.output)],check=True)


if __name__=='__main__':main()
