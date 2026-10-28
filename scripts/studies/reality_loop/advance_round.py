"""Register updated-policy feedback while preserving earlier certified lessons."""
import argparse
from pathlib import Path

from compare import audit_endpoint
from contracts import ledger, read, round_protocol, sha, validate_splits, write


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
    context=dict(base_protocol_sha256=sha(run/'protocol.json'),
        original=str(original),original_plan_sha256=sha(original/'plan.json'),
        initial_checkpoint=str(parent),initial_checkpoint_sha256=sha(parent),
        source_round=a.source_round,target_round=a.target_round,
        prior_regressions=comparison['original_success_regressions'],
        selection='predeclared terminal research candidate; not a production promotion',
        independent_exam_used=False,earlier_certified_repairs_carried=True)
    write(path,context)
    ledger(run,'round_context',[path,before/'comparison.json'],
        claim='collect fresh failures from updated parameters, retain prior admitted curriculum; no exam selection')
    print('REGISTERED',a.target_round,context['initial_checkpoint_sha256'],flush=True)


if __name__=='__main__':
    main()
