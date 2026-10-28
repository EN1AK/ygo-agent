"""Register updated-policy feedback while preserving earlier certified lessons."""
import argparse
from pathlib import Path

from compare import audit_endpoint
from contracts import audit_opponent_policy, ledger, read, round_protocol, sha, validate_splits, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source-round',default='round1')
    p.add_argument('--target-round',default='round2')
    a=p.parse_args();run=a.run.resolve()
    base=read(run/'protocol.json')
    source=round_protocol(run,a.source_round)
    if not (a.target_round.startswith('round') and a.target_round[5:].isdigit()):
        raise ValueError('invalid target round')
    if int(a.target_round[5:])!=int(a.source_round[5:])+1 or int(a.target_round[5:])>base['rounds']:
        raise ValueError('advance must follow the frozen round budget')
    before=run/(a.source_round+'-training')
    for arm in ('control','repair'):
        audit_endpoint(before/arm,source['training']['updates_per_arm'])
    comparison=read(before/'comparison.json')
    parent=before/'repair/endpoint/model.flax_model'
    original=before/'repair-inputs'
    plan=read(original/'plan.json')
    validate_splits(plan['rows'])
    if sha(parent)!=comparison['endpoint_sha256']['repair']:
        raise ValueError('not the terminal updated research candidate')
    path=run/(a.target_round+'-context.json')
    if path.exists():
        raise FileExistsError('round context already registered')
    # Retention references are real successful predecessor rollouts with the
    # same passive opponent, not unverified human suffixes. They propose routes;
    # the later search must still repeat and independently certify its witness.
    bank=run/(a.target_round+'-reference-bank');bank.mkdir(exist_ok=False)
    items={r['id']:r for r in plan['rows']}
    references={}
    for regression in comparison['original_success_regressions']:
        item=items[regression['id']]
        if item['split']!='fit' or regression['seed'] not in (None,1):
            continue
        trace_path=before/'control/before'/(item['id']+'-'+str(regression['seed'])+'.json')
        trace=read(trace_path)
        if (trace['status']!='success' or not trace['independent_audit']['passed'] or
                trace['checkpoint_sha256']!=source['initial_checkpoint_sha256'] or
                trace['goal']!=item['goal']):
            raise ValueError('retention reference lacks the declared successful source')
        audit_opponent_policy(trace,trace['root_step'])
        record=dict(trace,schema='reality-retention-reference-v1',
            source_sha256=item['source_sha256'],training_ready=False,
            reference_only=True,source_trace_sha256=sha(trace_path))
        if trace['seed'] is not None:
            record['definition']=dict(trace['definition'],seed=trace['seed'])
        target=bank/(item['id']+'.json');write(target,record)
        references[item['id']]=dict(path=str(target),sha256=sha(target),
            source_trace=str(trace_path),source_trace_sha256=sha(trace_path))
    write(bank/'index.json',references)
    context=dict(base_protocol_sha256=sha(run/'protocol.json'),
        original=str(original),original_plan_sha256=sha(original/'plan.json'),
        initial_checkpoint=str(parent),initial_checkpoint_sha256=sha(parent),
        source_round=a.source_round,target_round=a.target_round,
        prior_regressions=comparison['original_success_regressions'],
        reference_bank=str(bank/'index.json'),reference_bank_sha256=sha(bank/'index.json'),
        selection='predeclared terminal research candidate; not a production promotion',
        independent_exam_used=False,earlier_certified_repairs_carried=True)
    write(path,context)
    ledger(run,'round_context',[path,before/'comparison.json',bank/'index.json'],
        claim='collect fresh failures from updated parameters, retain prior admitted curriculum; no exam selection')
    print('REGISTERED',a.target_round,context['initial_checkpoint_sha256'],flush=True)


if __name__=='__main__':
    main()
