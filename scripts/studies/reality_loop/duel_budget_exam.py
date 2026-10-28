"""Same fixed duel protocol; certify known process-fuse cutoffs as unknown prefixes."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
from contracts import read, sha, write
from duel_exam import completed_case, stopping_status
PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT/'scripts/studies/crystron_curriculum'))


def evaluate(args):
    from budget_runtime import Runtime
    from supervision_runtime import jax,jnp,np,extract_trace,tensor_contract,array_digest,hidden_identity_audit
    from ygoai.rl.checkpoint_compat import validate_checkpoint_compatibility
    exam = read(args.exam/'plan.json')
    if sha(Path(exam['frozen'])/'plan.json')!=exam['frozen_plan_sha256']:
        raise ValueError('duel runtime plan changed')
    for role in ('candidate','opponent'):
        if sha(exam[role])!=exam[role+'_sha256']:
            raise ValueError('frozen duel model changed')
    resume = getattr(args, 'resume', False)
    args.output.mkdir(parents=True,exist_ok=resume)
    execution = dict(exam_sha256=sha(args.exam/'plan.json'), source_sha256=sha(__file__))
    execution_path = args.output / 'execution.json'
    if execution_path.exists():
        if not resume or read(execution_path) != execution:
            raise ValueError('resume requires the identical exam and evaluator source')
    else:
        if any(args.output.iterdir()):
            raise ValueError('cannot adopt an output directory without an execution binding')
        write(execution_path, execution)
    jax.config.update('jax_default_matmul_precision',exam['precision'])
    rt = Runtime(exam['frozen'])
    if rt.plan['native_sha256']!=exam['native_sha256']:
        raise ValueError('duel native contract mismatch')
    models = {}
    for role in ('candidate','opponent'):
        validate_checkpoint_compatibility(exam[role],observation_schema='structured-lite-v1',
            model_args=rt.modelargs,code_list_hash=rt.plan['code_list_sha256'],
            semantic_table_hash=rt.plan['semantics_sha256'],capacities=rt.capacities)
        models[role] = rt.load(exam[role])
    cases = exam['cases'][:args.max_cases] if args.max_cases else exam['cases']
    results = []
    for case in cases:
        previous = completed_case(args.output, case, exam, execution['exam_sha256'])
        if previous is not None:
            if not resume:
                raise ValueError('completed case exists without explicit resume')
            results.append(previous)
            print('RESUMED_COMPLETED_CASE', case['id'], previous['status'], flush=True)
            continue
        orphan = args.output / (case['id'] + '.json')
        if orphan.exists():
            # A power cut between record write and completion publication must
            # not silently replace its evidence or count it as a graded game.
            orphan.rename(args.output / (case['id'] + '.incomplete-' + str(time.time_ns()) + '.json'))
        write(args.output/'inflight.json',dict(case=case['id'],status='running',exam_sha256=sha(args.exam/'plan.json')))
        record = dict(id=case['id'],definition=case['definition'],seed=case['seed'],
            candidate_seat=case['candidate_seat'],candidate_sha256=exam['candidate_sha256'],
            opponent_sha256=exam['opponent_sha256'],status='unknown',decisions=[])
        env = None
        states = [rt.agent.init_rnn_state(1),rt.agent.init_rnn_state(1)]
        checked = set()
        started = time.monotonic()
        try:
            env = rt.env(record,case['seed'])
            env.reset()
            for step in range(exam['budget']['decisions']+1):
                snap = env.snapshot()
                status=stopping_status(snap,step,exam['budget'],time.monotonic()-started)
                if status:
                    record['status']=status
                    if status=='natural_terminal':record['winner']=snap['winner']
                    if status=='budget_exhausted':record['native_invalid_at_budget']=bool(snap['invalid'])
                    break
                seat=snap['player'];role='candidate' if seat==case['candidate_seat'] else 'opponent'
                obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                if seat not in checked:
                    hidden_identity_audit(obs);checked.add(seat)
                states[seat],logits,value=rt.forward(models[role],{k:jnp.asarray(v) for k,v in obs.items()},states[seat])
                scores=np.asarray(logits)[0,:len(snap['menu'])]
                v=float(np.asarray(value).reshape(-1)[0])
                if not len(scores) or not np.isfinite(scores).all() or not np.isfinite(v):
                    raise ValueError('nonfinite or empty duel policy')
                probabilities=np.exp(scores-scores.max());probabilities/=probabilities.sum()
                action=int(scores.argmax())
                events,_,responses=extract_trace(env,snap)
                record['decisions'].append(dict(step=step,player=seat,actor=role,message=snap['message'],
                    core_response_count=len(responses),chain_prompt=next((e for e in reversed(events) if e['op']==16),None) if snap['message']==16 else None,
                    phase=snap['phase'],menu=snap['menu'],action=action,observation_sha256=array_digest(obs),
                    policy_logits=scores.tolist(),policy_probabilities=probabilities.tolist(),value=v))
                env.step(action)
        except Exception as error:
            record.update(status='invalid',error=str(error))
        finally:
            try:
                if env is None:raise ValueError('native constructor failed; no trace available')
                snap=env.snapshot();events,state,responses=extract_trace(env,snap)
                record.update(events=events,final=state,core_responses=responses,core_seed=env.trace()['core_seed'])
            except Exception as error:
                record.update(status='invalid',capture_error=str(error))
            finally:
                if env is not None:env.close()
            record['seconds']=time.monotonic()-started
        if 'final' in record:
            try:
                record['independent_audit']=rt.independent(record)
            except Exception as error:
                record.update(independent_audit=dict(passed=False,error=str(error)),untrusted_status=record['status'],status='invalid')
        else:
            record['independent_audit']=dict(passed=False,error='native trace capture failed')
        if record['status']=='natural_terminal':
            winner=record['winner']
            if winner not in (0,1,2):
                raise ValueError('unknown natural winner code')
            record['candidate_result']='draw' if winner==2 else 'win' if winner==case['candidate_seat'] else 'loss'
        path=args.output/(case['id']+'.json');write(path,record)
        row=dict(id=case['id'],status=record['status'],candidate_seat=case['candidate_seat'],
            candidate_result=record.get('candidate_result'),path=path.name,record_sha256=sha(path),audit=record['independent_audit'])
        proof=args.output/(case['id']+'.completed.json')
        temporary=proof.with_suffix('.pending')
        write(temporary,dict(exam_sha256=execution['exam_sha256'],result=row))
        temporary.replace(proof)
        results.append(row)
        write(args.output/'progress.json',dict(results=results,planned=len(exam['cases']),requested=len(cases)))
        print(row,flush=True)
    summary=dict(completed_attempts=len(results),planned=len(exam['cases']),pending=len(exam['cases'])-len(results),
        qualification_only=len(cases)!=len(exam['cases']),results=results,
        status_counts=dict(Counter(r['status'] for r in results)),
        natural_results=dict(Counter(r['candidate_result'] for r in results if r['status']=='natural_terminal')),
        invalids_and_timeouts_are_not_losses=True,promotion_eligible=False)
    write(args.output/'summary.json',summary)
    write(args.output/'inflight.json',dict(status='finished',completed=len(results)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate'])
    for name in ('exam','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--max-cases',type=int)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error('positive case count required')
    evaluate(args)


if __name__=='__main__':main()
