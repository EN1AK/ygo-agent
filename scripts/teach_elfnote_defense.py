"""Run the fixed256-step response-only SFT comparison after independent enrollment."""
import argparse,copy,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','source','collection-source','study']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    for key,value in vars(a).items():setattr(a,key,value.resolve())
    sys.path[:0]=[str(a.source),str(a.source/'scripts')]
    import jax,numpy as np,flax.serialization
    from scripts.study_elfnote_coverage import read,write
    from scripts.teach_elfnote_combo import load_row,save,audit_actor
    from scripts.collect_teaching_exercise import load_native
    from scripts.eval_capability_exercises import FrozenActor
    from scripts.study_elfnote_functional_teaching import functional_evaluate
    from scripts.study_elfnote_search import evaluate as normal_evaluate
    from scripts.study_elfnote_recovery import checked
    from ygoai.rl.scene_rehearsal import load_rows
    from ygoai.rl.scene_optimizer import make_optimizer
    from ygoai.rl.defense_curriculum import descriptors,coefficients
    from ygoai.rl.defense_lessons import validate_protocol
    from ygoai.rl.exercise_core import sha
    from ygoai.rl.terminal_probe import run as terminal_run
    from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
    from ygoai.rl.jax.agent import ModelArgs
    m=read(a.study/'manifest.json');plan=validate_protocol(read(a.source/'assets/exercises/teaching-v1/elfnote-defense-teaching.json'))
    assert plan==m['plan'] and sha(a.source/'assets/exercises/teaching-v1/elfnote-defense-teaching.json')==m['plan_sha256']
    assert read(a.study/'collection-verification.json')['passed']
    copy_path=a.study/'collection-copy.json'
    if copy_path.exists():
        origin=read(copy_path);assert origin['no_model_updates_before_copy']
        assert sha(Path(origin['original_collect_exit']))==origin['original_collect_exit_sha256']
        assert Path(origin['original_collect_exit']).read_text().strip()=='0'
        assert sha(Path(origin['original_manifest']))==origin['original_manifest_sha256']==sha(a.study/'manifest.json')
        assert sha(Path(origin['original_proof']))==origin['original_proof_sha256']==sha(a.study/'collection-verification.json')
        assert sha(Path(origin['original_collection']))==origin['original_collection_sha256']
    else:assert Path(str(a.study)+'.collect.exit').read_text().strip()=='0'
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='float32' and all(d.platform=='gpu' for d in jax.devices())
    execution=read(a.source/'source-manifest.json')
    assert all(hashlib.sha256((a.source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in execution['all_files'].items())
    # The optimizer is part of this independently frozen training source, not
    # collection semantics; its no-update qualification has a separate audit.
    for name in ['ygoai/rl/defense_lessons.py','ygoai/rl/scene_batch.py','ygoai/rl/jax/agent.py','scripts/eval_capability_exercises.py','ygoai/rl/elfnote_recovery.py']:
        assert hashlib.sha256((a.source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==m['source']['all_files'][name]
    subprocess.run([sys.executable,str(a.source/'scripts/verify_elfnote_defense_lessons.py'),'--root',str(a.root),'--source',str(a.collection_source),
        '--study',str(a.study),'--output',str(a.study/'training-collection-recheck.json')],check=True)
    assert all(sha(Path(path))==value for path,value in m['protected'].items()) and all(sha(a.root/path)==value for path,value in m['runtime'].items())
    base=read(a.root/plan['base_curriculum']);assert sha(a.root/plan['base_curriculum'])==plan['base_curriculum_sha256']
    collection=read(a.study/'collection.json');items=descriptors(base,collection,plan);rows=load_rows(dict(rows=items))
    for item,(record,_) in zip(items,rows):
        if item['kind']=='public_response_CE':
            assert record['schema']=='ygo-public-response-lesson-v1' and record['training_ready'] and record['parent_sha256']==plan['parent_sha256']
            assert item['teaching_steps']==[record['lesson_root']]
        elif item['ce_coef']:
            for i in item['teaching_steps']:
                d=record['decisions'][i];assert d['action']==max(range(len(d['logits'])),key=lambda j:d['logits'][j]),'Common CE must be actual successful parent choices'
    assert len(rows)==plan['histories']==66
    write(a.study/'training-rows.json',dict(histories=len(rows),rows=items,scope=plan['common_curriculum']))
    write(a.study/'training-source-manifest.json',dict(source=execution,collection_source=str(a.collection_source),collection_proof_sha256=sha(a.study/'collection-verification.json'),
        collection_copy_sha256=sha(copy_path) if copy_path.exists() else None,plan_sha256=m['plan_sha256']))
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008'
    a.native=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014/exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    a.database=release/'repo/assets/locale/zh/cards.cdb';a.scripts=release/'repo/scripts/script';a.code_list=release/'code_list.txt';a.semantics=release/'semantics'
    checkpoint=a.root/plan['parent_study']/plan['parent_arm']/'model.flax_model';assert sha(checkpoint)==plan['parent_sha256'] and sha(a.native)==plan['native_sha256']
    course=read(a.root/plan['parent_study']/'manifest.json');old_parent=a.root/course['plan']['parent_study']
    normal_plan=read(old_parent/'manifest.json')['plan'];old=[load_row(Path(r['path'])) for r in read(old_parent/'training-rows.json')['rows'][31:36]]
    native=load_native(a,release/'opening-02/registration.ydk');actor=FrozenActor(checkpoint,a.code_list,a.semantics,{k:v[:1] for k,v in old[0][1].items()})
    forward=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3]);actor.forward=lambda obs,state:forward(actor.params,obs,state)
    parent=actor.params;fit=[c for c in m['functional_cases'] if c['split']=='fit']
    normal_probe=[c for c in m['normal_cases'] if c['split'] in ['repair','original'] or c['name'] in ['red-gate-train-0','blue-gate-train-0','gate-innocent-train-0']]
    tp=read(a.source/'assets/exercises/teaching-v1/elfnote-terminal-probe.json');arms=[]
    for arm in plan['arms']:
        folder=a.study/arm['name'];folder.mkdir(exist_ok=False);actor.params=parent;started=time.monotonic()
        cc,kc=coefficients(items,arm['response_coef'])
        update,state,admission=make_optimizer(actor,parent,rows,plan['sequence_steps'],plan['learning_rate'],cc,kc,check_pairs=plan['gradient_check_pairs'])
        assert admission['gradient_reference_l2']>1e-8 and not admission['qualification_hard_targets']
        admission['supervised_steps']=[r['teaching_steps'] for r in items];write(folder/'admission.json',admission)
        with (folder/'metrics.jsonl').open('x') as log:
            for step in range(1,plan['updates_per_arm']+1):
                assert time.monotonic()-started<plan['max_train_seconds_per_arm']
                actor.params,state,loss,terms,norm,finite=update(actor.params,state)
                metric=dict(update=step,loss=float(loss),ce=np.asarray(terms[0]).tolist(),kl=np.asarray(terms[1]).tolist(),grad_norm=float(norm),finite=bool(finite))
                assert metric['finite'] and np.isfinite([metric['loss'],metric['grad_norm']]+metric['ce']+metric['kl']).all()
                expected=float(np.dot(cc,metric['ce'])+np.dot(kc,metric['kl']))
                assert abs(expected-metric['loss'])/max(1.,abs(expected),abs(metric['loss']))<=8*np.finfo(np.float32).eps
                log.write(json.dumps(metric)+'\n');log.flush()
                if step in plan['probe_updates']:
                    functional_evaluate(a,native,actor,fit,plan,folder/f'probe-{step}')
                    normal_evaluate(a,native,actor,normal_probe,normal_plan,old,folder/f'normal-probe-{step}')
        model=folder/'model.flax_model';model.write_bytes(flax.serialization.to_bytes(actor.params))
        write_checkpoint_metadata(model,observation_schema='structured-lite-v1',model_args=ModelArgs(observation_schema='structured-lite-v1'),
            code_list_hash=sha(a.code_list),semantic_table_hash=sha(a.semantics/'metadata.json'),capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),
            training_context=dict(experiment=plan['id'],parent_sha256=plan['parent_sha256'],arm=arm,updates=plan['updates_per_arm'],promotion_eligible=False))
        actor.params=flax.serialization.from_bytes(parent,model.read_bytes())
        functional=functional_evaluate(a,native,actor,m['functional_cases'],plan,folder/'endpoint',audit=True)
        normal=normal_evaluate(a,native,actor,m['normal_cases'],normal_plan,old,folder/'normal-endpoint',audit=True)
        restored=[];root_folder=folder/'restored';root_folder.mkdir()
        for item in collection['results']:
            case=next(c for c in fit if c['name']==item['case']);reference=read(Path(item['path']))
            record,proof=checked(a,native,actor,case,plan,root_folder,case['name'],prefix=reference['decisions'][:item['root_step']],probe_mode='policy')
            restored.append(dict(case=case['name'],status=record['status'],**{k:v for k,v in proof.items() if k!='status'}))
        write(root_folder/'summary.json',restored)
        terminal=[];terminal_folder=folder/'terminal';terminal_folder.mkdir()
        for name in tp['cases']:
            index=next(i for i,c in enumerate(m['normal_cases']) if c['name']==name);original=read(folder/'normal-endpoint'/f'case-{index:02}.json')
            for challenge in tp['challenges']:
                sub=terminal_folder/f'{name}-{challenge["name"]}';sub.mkdir()
                for mode in ['scripted','pass','policy']:
                    record,obs=terminal_run(native,a.semantics,actor,original,challenge,tp,mode)
                    again,arrays=terminal_run(native,a.semantics,actor,original,challenge,tp,mode)
                    assert record==again and all(np.array_equal(obs[k],arrays[k]) for k in obs)
                    path=save(sub,mode,record,obs);assert audit_actor(a,path)['passed']
                    terminal.append(dict(case=name,challenge=challenge['name'],mode=mode,status=record['status'],path=str(path)))
        write(terminal_folder/'summary.json',dict(plan=tp,results=terminal))
        arms.append(dict(arm=arm,checkpoint_sha256=sha(model),functional=functional,normal=normal,restored=restored,seconds=time.monotonic()-started))
    assert all(sha(Path(path))==value for path,value in m['protected'].items()) and all(sha(a.root/path)==value for path,value in m['runtime'].items())
    write(a.study/'summary.json',dict(completed=True,arms=arms,production_default_replaced=False))
    subprocess.run([sys.executable,str(a.source/'scripts/verify_elfnote_defense_teaching.py'),'--root',str(a.root),'--source',str(a.source),
        '--collection-source',str(a.collection_source),'--study',str(a.study)],env=dict(os.environ,JAX_PLATFORMS='cpu'),check=True)


if __name__=='__main__':main()
