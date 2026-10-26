"""Verify current-parent response enrollment separately from raw core proof."""
import argparse,hashlib,json,sys
from pathlib import Path


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','source','study']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--output',type=Path,help='Write a fresh recheck receipt without replacing the original collection proof.')
    a=p.parse_args();sys.path.insert(0,str(a.source))
    from ygoai.rl.defense_lessons import admit
    m=read(a.study/'manifest.json');plan=m['plan'];collection=read(a.study/'collection.json')
    assert collection['admitted'] and not collection['training_started'] and m['frozen_before_collection']
    assert all(sha(path)==value for path,value in m['protected'].items())
    assert all(sha(a.root/path)==value for path,value in m['runtime'].items())
    assert all(hashlib.sha256((a.source/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for path,value in m['source']['all_files'].items())
    results=collection['results'];assert len(results)==4 and {r['case'] for r in results}=={c['name'] for c in plan['cases']}
    for row in results:
        path=Path(row['path']);assert sha(path)==row['record_sha256'] and sha(path.with_suffix('.npz'))==row['observations_sha256']
        root=Path(row['root_context_path']);candidate=Path(row['candidate_path'])
        assert sha(root)==row['root_context_sha256'] and sha(candidate)==row['candidate_sha256']
        context=read(root);assert context['current_checkpoint_sha256']==plan['parent_sha256']==row['parent_sha256']
        assert row['protocol_sha256']==m['plan_sha256']
        positive=read(row['raw_positive']['path']);negative=read(row['raw_negative']['path'])
        lesson,receipt=admit(positive,negative,context,read(candidate),plan)
        assert lesson==read(path) and all(row[k]==v for k,v in receipt.items())
        assert [i for i,d in enumerate(lesson['decisions']) if d['teaching']]==[row['root_step']]
        assert sum(d['player']==0 for d in lesson['decisions'])<=plan['sequence_steps']
    proofs=list(a.study.glob('raw/*.audit.json'))+list(a.study.glob('lessons/*.audit.json'))
    assert len(proofs)==12
    for path in proofs:
        proof=read(path);record=path.with_name(path.name.replace('.audit.json','.json'))
        assert proof['passed'] and proof['record_sha256']==sha(record) and proof['observations_sha256']==sha(record.with_suffix('.npz'))
        assert proof['native_sha256']==plan['native_sha256']
    result=dict(passed=True,accepted_lessons=4,independent_core_audits=12,parent_sha256=plan['parent_sha256'],source_commit=m['source']['commit'],
        roots=[dict(case=r['case'],root_step=r['root_step'],record_sha256=r['record_sha256']) for r in results],
        training_started=False,scope='Four new response-only lessons; current-parent full prefix/RNN and public top verified. Old probes unchanged; local goal failure is not a global strategic negative label.')
    result['verifier_sha256']=sha(Path(__file__))
    with (a.output or a.study/'collection-verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
