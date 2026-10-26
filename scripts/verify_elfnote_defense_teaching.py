"""Audit both fixed SFT endpoints, true root masks, objective arithmetic and cores."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','source','collection-source','study']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sys.path[:0]=[str(a.source),str(a.source/'scripts')]
    import numpy as np,flax.serialization
    from scripts.study_elfnote_root_repair import score
    from ygoai.rl.defense_curriculum import descriptors,coefficients
    from ygoai.rl.elfnote_recovery import challenge_evidence
    m=read(a.study/'manifest.json');plan=m['plan'];summary=read(a.study/'summary.json');source=read(a.study/'training-source-manifest.json')
    assert summary['completed'] and not summary['production_default_replaced']
    assert all(sha(Path(path))==value for path,value in m['protected'].items()) and all(sha(a.root/path)==value for path,value in m['runtime'].items())
    assert all(hashlib.sha256((a.source/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for path,value in source['source']['all_files'].items())
    assert source['collection_proof_sha256']==sha(a.study/'collection-verification.json')
    if source.get('collection_copy_sha256'):assert sha(a.study/'collection-copy.json')==source['collection_copy_sha256']
    assert read(a.study/'training-collection-recheck.json')['passed']
    base=read(a.root/plan['base_curriculum']);collection=read(a.study/'collection.json');rows=read(a.study/'training-rows.json')
    expected=descriptors(base,collection,plan);assert rows['rows']==expected and rows['histories']==len(expected)==66
    for row in expected:
        path=Path(row['path']);assert sha(path)==row['record_sha256'] and sha(path.with_suffix('.npz'))==row['observations_sha256']
        record=read(path);assert all(record['decisions'][i]['player']==row['controlled'] for i in row['teaching_steps'])
        if row['kind']=='public_response_CE':
            assert record['schema']=='ygo-public-response-lesson-v1' and record['training_ready']
            assert row['teaching_steps']==[record['lesson_root']]
    proofs=list(a.study.rglob('*.audit.json'));assert len(proofs)==286,len(proofs)
    for path in proofs:
        proof=read(path);record=path.with_name(path.name.replace('.audit.json','.json'))
        assert proof['passed'] and proof['native_sha256']==plan['native_sha256']
        assert proof['record_sha256']==sha(record) and proof['observations_sha256']==sha(record.with_suffix('.npz'))
    normal={};functional={};terminal={};restored={};errors={};checkpoints={};deltas={}
    parent=flax.serialization.msgpack_restore((a.root/plan['parent_study']/plan['parent_arm']/'model.flax_model').read_bytes())
    def leaves(tree):
        if isinstance(tree,dict):return [v for k in sorted(tree) for v in leaves(tree[k])]
        return [np.asarray(tree)]
    assert [r['arm'] for r in summary['arms']]==plan['arms']
    for arm in summary['arms']:
        name=arm['arm']['name'];folder=a.study/name;admission=read(folder/'admission.json')
        cc,kc=coefficients(expected,arm['arm']['response_coef'])
        assert admission['passed'] and admission['histories']==66 and admission['micro_batch_histories']==2
        assert admission['gradient_reference_l2']>1e-8 and admission['gradient_relative_error']<.002 and admission['max_logit_error']<.002
        assert not admission['qualification_hard_targets']
        q=admission['qualification_objective'];assert q['passed'] and q['scalar_difference']<=q['derived_difference_bound']
        assert q['full_same_input_relative_error']<=8*np.finfo(np.float32).eps and q['micro_same_input_relative_error']<=8*np.finfo(np.float32).eps
        assert admission['actual_micro_objective_residual']<=8*np.finfo(np.float32).eps
        np.testing.assert_allclose(admission['ce_coeff'],cc);np.testing.assert_allclose(admission['kl_coeff'],kc)
        assert admission['supervised_steps']==[r['teaching_steps'] for r in expected]
        metrics=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert [r['update'] for r in metrics]==list(range(1,plan['updates_per_arm']+1))
        residual=[]
        for r in metrics:
            assert r['finite'] and len(r['ce'])==len(r['kl'])==66
            assert np.isfinite([r['loss'],r['grad_norm']]+r['ce']+r['kl']).all()
            value=float(np.dot(cc,r['ce'])+np.dot(kc,r['kl']));residual.append(abs(value-r['loss'])/max(1.,abs(value),abs(r['loss'])))
        assert max(residual)<=8*np.finfo(np.float32).eps;errors[name]=max(residual)
        model=folder/'model.flax_model';assert sha(model)==arm['checkpoint_sha256']==read(model.with_name(model.name+'.metadata.json'))['checkpoint_sha256']
        variables=flax.serialization.msgpack_restore(model.read_bytes())
        def finite(tree):
            if isinstance(tree,dict):return all(finite(v) for v in tree.values())
            return bool(np.isfinite(np.asarray(tree)).all())
        assert finite(variables);checkpoints[name]=sha(model)
        before,after=leaves(parent),leaves(variables);assert len(before)==len(after) and all(x.shape==y.shape for x,y in zip(before,after))
        deltas[name]=sum(float(np.sum((x.astype(np.float64)-y.astype(np.float64))**2)) for x,y in zip(before,after))**.5
        if name=='teach':assert deltas[name]>0
        f=read(folder/'endpoint/summary.json');n=read(folder/'normal-endpoint/summary.json')
        assert f==arm['functional'] and n==arm['normal'] and len(n['cases'])==53 and len(f['cases'])==56
        assert [(r['name'],r['split']) for r in n['cases']]==[(c['name'],c['split']) for c in m['normal_cases']]
        assert [(r['name'],r['mode']) for r in f['cases']]==[(c['name'],mode) for c in m['functional_cases'] for mode in ['scripted','policy']]
        for i,row in enumerate(n['cases']):
            record=read(folder/'normal-endpoint'/f'case-{i:02}.json');assert record['status']==row['status'] and record['case']==row['name']
        for row in f['cases']:
            record=read(row['path']);assert record['status']==row['status'] and record['case']==row['name']
            if record['status']=='success':assert challenge_evidence(record['events'],record['probe_event_start'])==record['functional_evidence'] and record['functional_evidence']['success']
        scores={split:{mode:sum(r['status']=='success' for r in f['cases'] if r['split']==split and r['mode']==mode) for mode in ['scripted','policy']} for split in f['scores']}
        assert scores==f['scores'];functional[name]=scores;normal[name]=score(n)
        roots=read(folder/'restored/summary.json');assert roots==arm['restored'] and len(roots)==4
        restored[name]=sum(r['status']=='success' for r in roots)
        for row in roots:
            record=read(row['path']);assert record['status']==row['status'] and record['probe_mode']=='policy'
        t=read(folder/'terminal/summary.json');assert len(t['results'])==24
        terminal[name]={challenge['name']:{mode:sum(r['status']=='success' for r in t['results'] if r['challenge']==challenge['name'] and r['mode']==mode) for mode in ['scripted','pass','policy']} for challenge in t['plan']['challenges']}
        for row in t['results']:
            record=read(row['path']);assert record['status']==row['status'] and not record['training_ready']
            evidence=challenge_evidence(record['events'],record['probe_event_start'],challenge=record['challenge']['code'],negators=tuple(t['plan']['negators'])) if record['probe_event_start'] is not None else dict(completed=False,success=False)
            assert evidence==record['functional_evidence']
        for step in plan['probe_updates']:
            assert all(c['split']=='fit' for c in read(folder/f'probe-{step}/summary.json')['cases'])
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008';native=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014/exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    old_paths=sorted(a.study.glob('**/old/[0-4].json'));assert len(old_paths)==50,len(old_paths)
    env=dict(os.environ,PYTHONPATH=str(a.source),JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for path in old_paths:
        command=[sys.executable,'-m','scripts.verify_elfnote_study','--study',str(a.study),'--source',str(a.source),'--native',str(native),'--database',str(release/'repo/assets/locale/zh/cards.cdb'),'--scripts',str(release/'repo/scripts/script'),'--output',str(a.study/'unused.json'),'--old-record',str(path)]
        result=subprocess.run(command,env=env,cwd=a.source,capture_output=True,text=True,check=True)
        assert json.loads(result.stdout.strip().splitlines()[-1])['passed']
    candidate=plan['formal_candidate_arm'];n=normal[candidate];fit=functional[candidate]['fit']
    retained=sum(n[k] for k in ['repair','rehearsal','exposed_transfer','fresh_repair','fresh_triple','original'])==49 and n['repair']==2 and n['rehearsal']==24 and n['original']==n['old']==5
    gate=retained and fit['scripted']==4
    defensive=gate and fit['policy']==restored[candidate]==4
    result=dict(passed=True,source_commit=source['source']['commit'],parent_sha256=plan['parent_sha256'],accepted_lessons=4,new_audits=286,old_audits=50,
        checkpoints=checkpoints,parameter_delta_l2=deltas,normal=normal,endpoints=functional,restored_roots=restored,terminal=terminal,objective_errors=errors,
        formal_training_exercise_gate=gate,defensive_policy_gate=defensive,production_default_replaced=False,
        scope='Fixed256 response-teaching endpoints, exposed family only. Process exit must additionally be checked outside this producer before any formal PPO. No promotion or unseen-mechanism claim.')
    with (a.study/'final-verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
