"""Propose public-response lessons from audited causal pairs, never enroll them."""
import hashlib,json,math
from pathlib import Path


def propose(teacher,baseline,*,controlled=0):
    """Reject different contexts rather than manufacturing a counterfactual label."""
    assert teacher['schema']==baseline['schema']=='ygo-elfnote-functional-recovery-v1'
    assert teacher['definition']==baseline['definition'] and teacher['case']==baseline['case']
    assert teacher['core_seed']==baseline['core_seed']
    assert teacher['probe_mode']=='scripted' and baseline['probe_mode']=='policy'
    assert teacher['status']=='success' and baseline['status']=='goal_not_reached'
    assert teacher['functional_evidence']['success'] and not baseline['functional_evidence']['success']
    assert teacher['intervention_evidence']['success'] and baseline['intervention_evidence']['success']
    probe=teacher['probe_start_step'];assert probe is not None and probe==baseline['probe_start_step']
    x,y=teacher['decisions'],baseline['decisions']
    first=next((i for i,(a,b) in enumerate(zip(x,y)) if a!=b),None)
    assert first is not None and first>=probe
    a,b=x[first],y[first]
    # Equal visible input alone is insufficient: recurrence and the complete
    # earlier legal history must match at this same response window.
    assert x[:first]==y[:first]
    for key in ['step','player','message','menu','observation_sha256','rnn_sha256','core_response_count','logits']:
        assert a[key]==b[key],key
    assert a['player']==controlled and a['message']==16
    assert 0<=a['action']<len(a['menu']) and 0<=b['action']<len(b['menu'])
    assert len(a['logits'])==len(a['menu']) and all(math.isfinite(v) for v in a['logits'])
    assert b['action']==max(range(len(b['logits'])),key=lambda i:b['logits'][i]),'Baseline response must be its actual unassisted argmax'
    correct,chosen=a['menu'][a['action']],b['menu'][b['action']]
    assert correct!=chosen and correct['act']=='Activate' and chosen['act']=='Cancel'
    assert correct['code'] in [50954680,84815190]
    assert not a['teaching'] and not b['teaching'],'Old probes must retain their original masks'
    return dict(schema='ygo-defense-candidate-v1',case=teacher['case'],controlled=controlled,root_step=first,
        root=dict(observation_sha256=a['observation_sha256'],rnn_sha256=a['rnn_sha256'],
                  correct_action=a['action'],chosen_action=b['action'],correct=correct,chosen=chosen),
        status='needs_current_model_reconstruction',training_ready=False,
        lesson_scope='New public-response lesson proposed from an explicitly exposed old probe; old records and masks unchanged. Teacher end, opponent challenge and private setup are not labels.',
        required_admission=['Freeze new lesson protocol before fitting','Reconstruct current-model full history and recurrent state',
                            'Replay both same-root branches twice and through independent core',
                            'Confirm actual public challenge negation and cancel failure',
                            'Label only the verified own response; protect successful earlier context separately'])


def from_files(teacher_path,baseline_path):
    paths=[Path(teacher_path),Path(baseline_path)];records=[];sources=[]
    for path in paths:
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        proof_path=path.with_name(path.stem+'.audit.json');proof=json.loads(proof_path.read_text())
        arrays=path.with_suffix('.npz');assert arrays.is_file()
        assert proof['passed'] and proof['record_sha256']==digest
        assert proof['observations_sha256']==hashlib.sha256(arrays.read_bytes()).hexdigest()
        records.append(json.loads(data));sources.append(dict(path=str(path),record_sha256=digest,
            audit_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),native_sha256=proof['native_sha256']))
    assert sources[0]['native_sha256']==sources[1]['native_sha256']
    result=propose(*records);result['sources']=sources
    return result
