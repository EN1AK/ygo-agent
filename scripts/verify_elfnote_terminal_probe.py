"""Independently verify terminal-probe provenance, matched controls and scoring."""
import argparse,hashlib,json,sys
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','study']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.root=a.root.resolve();a.source=a.source.resolve();a.study=a.study.resolve()
    sys.path.insert(0,str(a.source))
    from ygoai.rl.elfnote_recovery import challenge_evidence
    from ygoai.rl.terminal_probe import definition_for
    m=read(a.study/'manifest.json');plan=m['plan'];summary=read(a.study/'summary.json')
    assert summary['passed'] and not summary['production_default_replaced']
    assert Path(str(a.study)+'.exit').read_text().strip()=='0'
    assert all(sha(Path(p))==value for p,value in m['protected'].items())
    assert all(sha(a.root/p)==value for p,value in m['runtime'].items())
    assert all(hashlib.sha256((a.source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in m['source']['all_files'].items())
    normal=read(a.root/plan['course']/'manifest.json')['normal_cases'];checks=[];rows=[]
    for model in ['initial','plain']:
        for case in plan['cases']:
            index=next(i for i,c in enumerate(normal) if c['name']==case)
            original=read(a.root/plan[model+'_records']/'normal'/f'case-{index:02}.json')
            for challenge in plan['challenges']:
                folder=a.study/f'{model}-{case}-{challenge["name"]}';records={}
                for mode in ['scripted','pass','policy']:
                    path=folder/(mode+'.json');r=read(path);proof=read(folder/(mode+'.audit.json'))
                    assert proof['passed'] and proof['native_sha256']==plan['native_sha256'] and proof['record_sha256']==sha(path) and proof['observations_sha256']==sha(path.with_suffix('.npz'))
                    assert r['definition']==definition_for(original,challenge)[0] and r['source_prefix_steps']==len(original['decisions'])
                    assert all(not d['teaching'] and not d['searchable'] for d in r['decisions'])
                    if r['status']!='reconstruction_failed':
                        assert r['prefix_own_observations_verified']==sum(d['player']==0 for d in original['decisions'])
                        for x,y in zip(r['decisions'],original['decisions']):
                            assert (x['player'],x['message'],x['menu'],x['action'])==(y['player'],y['message'],y['menu'],y['action'])
                            if x['player']==0:assert (x['observation_sha256'],x['rnn_sha256'])==(y['observation_sha256'],y['rnn_sha256'])
                    start=r['probe_event_start'];e=challenge_evidence(r['events'],start,challenge=challenge['code'],negators=tuple(plan['negators'])) if start is not None else dict(completed=False,success=False)
                    assert e==r['functional_evidence']
                    if r['status']=='success':assert e['completed'] and e['success']
                    elif r['status']=='challenge_resolved':assert e['completed'] and not e['success']
                    records[mode]=r;rows.append(dict(model=model,case=case,challenge=challenge['name'],mode=mode,status=r['status']))
                positive,negative=records['scripted'],records['pass']
                admitted=positive['status']=='success' and negative['status']=='challenge_resolved'
                if admitted:
                    x,y=positive['decisions'],negative['decisions'];first=next(i for i,(u,v) in enumerate(zip(x,y)) if u!=v)
                    assert x[:first]==y[:first] and x[first]['player']==y[first]['player']==0 and x[first]['message']==y[first]['message']==16
                    assert x[first]['menu']==y[first]['menu'] and x[first]['observation_sha256']==y[first]['observation_sha256'] and x[first]['rnn_sha256']==y[first]['rnn_sha256']
                    assert x[first]['menu'][x[first]['action']]['act']=='Activate' and y[first]['menu'][y[first]['action']]['act']=='Cancel'
                checks.append(dict(model=model,case=case,challenge=challenge['name'],available_with_valid_negative=admitted,
                    policy_success=records['policy']['status']=='success',statuses={mode:r['status'] for mode,r in records.items()}))
    assert len(rows)==summary['audits']==48 and len(list(a.study.glob('*/*.audit.json')))==48
    scores={model:{c['name']:{mode:sum(r['model']==model and r['challenge']==c['name'] and r['mode']==mode and r['status']=='success' for r in rows) for mode in ['scripted','pass','policy']} for c in plan['challenges']} for model in ['initial','plain']}
    assert scores==summary['scores']
    result=dict(passed=True,source_commit=m['source']['commit'],independent_core_audits=48,scores=scores,paired_checks=checks,
        reconstruction_failures=sum(r['status']=='reconstruction_failed' for r in rows),
        unobserved_or_budget=sum(r['status'] in ['challenge_not_observed','budget_exhausted'] for r in rows),
        verifier_sha256=sha(Path(__file__)),scope=plan['scope'],training_started=False)
    with (a.study/'final-verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='paired_checks'}),flush=True)


if __name__=='__main__':main()
