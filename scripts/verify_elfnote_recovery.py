"""Validate conditional functional witnesses and their pass-response controls."""
import argparse
import hashlib
from pathlib import Path
from scripts.study_elfnote_coverage import read,write
from scripts.study_elfnote_recovery import cases_for
from ygoai.rl.elfnote_recovery import challenge_evidence,board_resources
from ygoai.rl.exercise_core import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','source','study']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();s=a.study;m=read(s/'manifest.json');plan=m['plan'];summary=read(s/'summary.json')
    assert summary['completed'] and not summary['optimizer_used'] and not m['optimizer_used'] and summary['protected_unchanged']
    assert not list(s.rglob('model.flax_model'))
    assert all(sha(Path(path))==value for path,value in m['protected'].items())
    assert all(sha(a.root/path)==value for path,value in m['runtime'].items())
    assert all(hashlib.sha256((a.source/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for path,value in m['source']['all_files'].items())
    assert cases_for(plan,read(a.root/plan['definition_source'])['cases'])==m['cases']
    evaluations=summary['evaluations'];assert len(evaluations)==32
    assert {(r['model'],r['case'],r['probe']) for r in evaluations}=={(model,c['name'],probe) for model in plan['models'] for c in m['cases'] for probe in plan['probe_modes']}
    accepted=[r for r in summary['discoveries'] if 'path' in r];proofs=list(s.rglob('*.audit.json'))
    assert len(proofs)==32+2*len(accepted)
    for path in proofs:
        proof=read(path);record_path=path.with_name(path.name.replace('.audit.json','.json'));record=read(record_path)
        assert proof['passed'] and proof['native_sha256']==plan['native_sha256']
        assert proof['record_sha256']==sha(record_path) and proof['observations_sha256']==sha(record_path.with_suffix('.npz'))
        root=record['root_step'];probe=record['probe_start_step']
        assert all(not d['teaching'] and not d['searchable'] for d in record['decisions'][:root or 0])
        if probe is not None:assert all(not d['teaching'] and not d['searchable'] for d in record['decisions'][probe:])
        if record['status']=='success':
            assert root is not None and probe is not None and record['intervention_attempted']
            evidence=challenge_evidence(record['events'],record['probe_event_start'])
            assert evidence==record['functional_evidence'] and evidence['success']
            assert record['final']['turn_player']==1 and board_resources(record['final'])['hand']>=plan['minimum_hand']
            assert record['board_at_probe']['resources']['negators'] and record['board_at_probe']['resources']['hand']>=plan['minimum_hand']
    for r in accepted:
        witness=read(Path(r['path']));control=read(Path(r['negative_control']['path']))
        assert witness['status']=='success' and control['status']=='goal_not_reached' and not control['functional_evidence']['success']
        assert witness['definition']==control['definition']
        step=witness['probe_start_step'];assert witness['decisions'][:step]==control['decisions'][:step]
    scores={mode:{probe:sum(r['status']=='success' for r in evaluations if r['model']==mode and r['probe']==probe) for probe in plan['probe_modes']} for mode in plan['models']}
    result=dict(passed=True,source_commit=m['source']['commit'],source_files=len(m['source']['all_files']),runtime_files=len(m['runtime']),
                core_audits=len(proofs),scores=scores,discovered=len(accepted),cases=4,
                trials=sum(len(r['trials']) for r in summary['discoveries']),actor_decisions=sum(r['decisions'] for r in summary['discoveries']),
                discovery_seconds=sum(r['seconds'] for r in summary['discoveries']),optimizer_used=False,production_eligible=False,
                scope='Conditional exposed functional scenes; scripted availability separated from policy response, not all-responses strength')
    write(s/'final-verification.json',result);print(__import__('json').dumps(result),flush=True)


if __name__=='__main__':main()
