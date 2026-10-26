"""Export unadmitted response candidates from a completed frozen SFT endpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import _repo_bootstrap
from ygoai.rl.defense_candidates import from_files


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def own_history_key(record,root,controlled):
    # Observation digests preserve all model-visible inputs, including card
    # order. This does not assert semantic independence between variants.
    own=[(d['observation_sha256'],d['action'] if i<root else None) for i,d in enumerate(record['decisions'][:root+1]) if d['player']==controlled]
    value=dict(schema='complete-own-observation-prefix-v1',controlled=controlled,history=own,menu=record['decisions'][root]['menu'])
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['study','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=['continue','teach'],default='teach')
    parser.add_argument('--splits',nargs='+',help='Explicit exposed split names; omitted exports all completed conditions.')
    a=parser.parse_args();a.study=a.study.resolve();a.output=a.output.resolve()
    proof=read(a.study/'final-verification.json');assert proof['passed'] and Path(str(a.study)+'.train.exit').read_text().strip()=='0'
    native_hash=read(a.study/'manifest.json')['plan']['native_sha256']
    folder=a.study/a.arm;model=folder/'model.flax_model';model_hash=sha(model)
    assert model_hash==proof['checkpoints'][a.arm]
    summary=read(folder/'endpoint/summary.json');assert summary['scores']==proof['endpoints'][a.arm]
    available={r['split'] for r in summary['cases']};splits=set(a.splits or available);assert splits and splits<=available
    protected={str(p):sha(p) for p in [a.study/'manifest.json',a.study/'final-verification.json',Path(str(a.study)+'.train.exit'),model,folder/'endpoint/summary.json']}
    a.output.mkdir(parents=True,exist_ok=False);rows=[];groups={}
    for case in sorted({r['name'] for r in summary['cases'] if r['split'] in splits}):
        assert Path(case).name==case and case not in ['.','..']
        records=[r for r in summary['cases'] if r['name']==case];assert len(records)==2
        positive=next(r for r in records if r['mode']=='scripted');negative=next(r for r in records if r['mode']=='policy');assert positive['split']==negative['split']
        if negative['status']=='success':
            rows.append(dict(case=case,split=positive['split'],status='already_successful',training_ready=False));continue
        if positive['status']!='success':
            rows.append(dict(case=case,split=positive['split'],status='no_verified_teacher_solution',training_ready=False));continue
        p=Path(positive['path']).resolve();n=Path(negative['path']).resolve()
        assert p.parent==n.parent==(folder/'endpoint').resolve()
        try:candidate=from_files(p,n)
        except (AssertionError,ValueError,KeyError) as exc:
            rows.append(dict(case=case,split=positive['split'],status='reconstruction_required',reason=str(exc),training_ready=False));continue
        assert candidate['case']==case and all(r['native_sha256']==native_hash for r in candidate['sources'])
        history=own_history_key(read(p),candidate['root_step'],candidate['controlled'])
        candidate.update(parent_sha256=model_hash,parent_arm=a.arm,source_study=str(a.study),source_final_proof_sha256=sha(a.study/'final-verification.json'),split=positive['split'],exposure='Already exposed fixed-endpoint development condition',own_history_key=history)
        for path in [p,n]:
            for item in [path,path.with_suffix('.npz'),path.with_name(path.stem+'.audit.json')]:protected[str(item)]=sha(item)
        path=a.output/(case+'.json')
        with path.open('x') as f:json.dump(candidate,f,indent=2)
        groups.setdefault(history,[]).append(case)
        rows.append(dict(case=case,split=positive['split'],status=candidate['status'],root_step=candidate['root_step'],path=str(path),candidate_sha256=sha(path),own_history_key=history,training_ready=False))
    assert all(sha(Path(path))==value for path,value in protected.items())
    result=dict(schema='ygo-response-transfer-intake-v1',parent_sha256=model_hash,parent_arm=a.arm,study=str(a.study),splits=sorted(splits),candidates=rows,proposed=sum(r['status']=='needs_current_model_reconstruction' for r in rows),unique_observed_histories=len(groups),duplicate_groups=[v for v in groups.values() if len(v)>1],protected=protected,training_started=False,training_ready=False,producer_sha256=sha(Path(__file__)),scope='Completed exposed SFT endpoint only. Native-audited same-prefix/current-RNN/argmax causal candidates, never new supervision. Exact observed-history grouping does not prove semantic independence. A fresh protocol and current-parent public-chain/RNN reconstruction are required before teaching; source masks remain unchanged.')
    with (a.output/'manifest.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['protected','candidates']}),flush=True)


if __name__=='__main__':main()
