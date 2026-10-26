"""Freeze proven parent histories and masks for alternating PPO rehearsal."""
import argparse,json,hashlib,copy
from pathlib import Path


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(root,role,output):
    parent_study=root/'training-runs/elfnote-recovery-context-20261023'
    initial=root/'training-runs/elfnote-formal-initial-retention-20261024'
    terminal=root/'training-runs/elfnote-terminal-probe-v3-20261026/final-verification.json'
    tp=read(terminal);assert tp['passed'] and tp['reconstruction_failures']==0
    proof=read(initial/'final-verification.json');assert proof['passed'] and proof['retention_gate']
    parent=parent_study/'full_context/model.flax_model';assert sha(parent)==proof['checkpoint_sha256']
    release=root/'dist/runtime-releases/elfnote-formal-preflight-v3-3878cdc-20261023'
    protected={str(p):sha(p) for p in [terminal,initial/'final-verification.json',initial/'normal/summary.json',initial/'manifest.json',
        parent_study/'final-verification.json',parent_study/'manifest.json',parent_study/'summary.json',parent]}
    rows=[]
    def add(path,controlled,steps,ce,kc,kind):
        record=read(path);arrays=path.with_suffix('.npz');assert arrays.is_file()
        assert steps and all(record['decisions'][i]['player']==controlled for i in steps)
        if ce:
            for i in steps:
                d=record['decisions'][i]
                assert d['action']==max(range(len(d['logits'])),key=lambda j:d['logits'][j]),'Do not teach a forced prefix choice'
        rows.append(dict(path=str(path),record_sha256=sha(path),observations_sha256=sha(arrays),controlled=controlled,
            teaching_steps=steps,ce_coef=ce,kl_coef=kc,kind=kind))
    normal=read(initial/'normal/summary.json')
    chosen={'red-gate-train-0','blue-gate-train-0','gate-innocent-train-0'}
    for i,c in enumerate(normal['cases']):
        if c['status']!='success':continue
        path=initial/'normal'/f'case-{i:02}.json';r=read(path);audit=read(path.with_name(path.stem+'.audit.json'))
        assert r['status']=='success' and r['definition']['controlled']==0
        assert audit['passed'] and audit['record_sha256']==sha(path) and audit['observations_sha256']==sha(path.with_suffix('.npz'))
        add(path,0,[j for j,d in enumerate(r['decisions']) if d['player']==0],1/3 if c['name'] in chosen else 0.,1/58,'successful_normal')
    assert len(rows)==49 and sum(x['ce_coef']>0 for x in rows)==3
    for i,item in enumerate(normal['old']):
        assert item['success'];path=initial/'normal/old'/f'{i}.json';r=read(path);audit=proof['old_audits'][i]['proof']
        assert audit['passed'] and audit.get('record_sha256',audit.get('sha256'))==sha(path) and audit['observations_sha256']==sha(path.with_suffix('.npz'))
        controlled=r.get('definition',{}).get('controlled',0)
        if i==4:assert r.get('case')=='combo';controlled=0
        # The fifth synthetic fixture has teacher-prepared prefix choices.
        # Keep its full-history KL, but never turn those choices into CE.
        add(path,controlled,[j for j,d in enumerate(r['decisions']) if d['player']==controlled],1/4 if i<4 else 0.,1/58,
            'successful_old_skill' if i<4 else 'synthetic_old_KL_only')
    functional=[]
    for c in read(parent_study/'manifest.json')['cases']:
        if c['split']!='fit':continue
        path=parent_study/'full_context/endpoint'/(c['name']+'-scripted.json');r=read(path);audit=read(path.with_name(path.stem+'.audit.json'))
        assert r['status']=='success' and r['intervention_evidence']['success'] and audit['passed'] and audit['record_sha256']==sha(path) and audit['observations_sha256']==sha(path.with_suffix('.npz'))
        end=r['probe_start_step'];start=r['root_step'];assert start is not None and end is not None
        steps=[j for j,d in enumerate(r['decisions']) if d['player']==0 and start<=j<end]
        add(path,0,steps,1/4,0.,'functional_recovery_CE');functional.append((path,r,end))
    assert len(rows)==58 and len(functional)==4
    for path,r,end in functional:add(path,0,[j for j,d in enumerate(r['decisions']) if d['player']==0 and j<end],0.,1/58,'functional_context_KL')
    assert len(rows)==62
    config=dict(format_version=1,role=role,parent_checkpoint=str(parent),parent_sha256=sha(parent),sequence_steps=128,histories=62,gradient_check_pairs=[27,0],learning_rate=1e-5,
        code_list_sha256=sha(release/'code_list.txt'),semantic_metadata_sha256=sha(release/'semantics/metadata.json'),protected=protected,rows=rows,
        scope='49 successful full parent normal histories KL; four successful normal legacy skills CE/KL and fifth synthetic fixture KL only; four functional recovery CE and four own pre-probe contexts KL. Four known-failing triples excluded; artificial end/challenge/protector responses excluded. Red-blue alternative has no correction CE. All exposed development; no new probe tensors used as supervision.')
    with output.open('x') as f:json.dump(config,f,indent=2)
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=['control','rehearse'],required=True);a=p.parse_args()
    c=prepare(a.root.resolve(),a.role,a.output);print(json.dumps(dict(role=a.role,histories=len(c['rows']),parent_sha256=c['parent_sha256'])))


if __name__=='__main__':main()
