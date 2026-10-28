"""Paired natural-opening mirror duels with two independently recurrent policies.

Unlike a fixed-prefix/passive-opponent exercise, both seats choose every exposed
decision. Incomplete or invalid games never become wins. The exact runtime,
decks, model hashes, seeds, case denominator and budget are frozen first.
"""
import argparse
from collections import Counter
from pathlib import Path
import random
import sys
import time

from contracts import digest, read, sha, write

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT/'scripts/studies/crystron_curriculum'))


def natural_definition(main, extra, flags, seed):
    words = [int(digest(['duel-rng', seed, i])[:8], 16) for i in range(8)]
    cards = []
    for seat in (0, 1):
        deck = list(main)
        random.Random(int(digest(['opening', seed, seat])[:16], 16)).shuffle(deck)
        cards += [[code, seat, 1, 0, 8] for code in deck]
        cards += [[code, seat, 64, 0, 8] for code in extra]
    return dict(initial=cards, controlled=0, seed_sequence=words, duel_flags=flags)


def freeze(args):
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.frozen.resolve()
    p = read(frozen/'plan.json')
    decks = {}
    for item in p['rows']:
        if item['split']!='fit':
            continue
        path = frozen/item['path']
        if sha(path)!=item['record_sha256']:
            raise ValueError('source deck provenance changed')
        d = read(path)['definition']
        if d.get('lua') or any(c[2] not in (1,64) for c in d['initial']):
            continue
        seat = d['controlled']
        main = [c[0] for c in d['initial'] if c[1:3]==[seat,1]]
        extra = [c[0] for c in d['initial'] if c[1:3]==[seat,64]]
        if not 40<=len(main)<=60 or len(extra)>15 or max(Counter(main+extra).values())>3:
            continue
        key = digest([sorted(main),sorted(extra),d['duel_flags']])
        decks.setdefault(key,dict(main=main,extra=extra,flags=d['duel_flags'],source_record_sha256=sha(path)))
    selected = sorted(decks)[:args.decks]
    if len(selected)!=args.decks:
        raise ValueError('not enough distinct valid fit-deck multisets')
    cases = []
    for key in selected:
        d = decks[key]
        for i in range(args.seeds):
            seed = int(digest(['mirror-duels-v1',key,i])[:8],16)
            definition = natural_definition(d['main'],d['extra'],d['flags'],seed)
            for seat in (0,1):
                cases.append(dict(id=key[:12]+'-'+str(i)+'-'+str(seat),deck_sha256=key,
                    seed=seed,candidate_seat=seat,definition=definition))
    exam = dict(schema='ygo-natural-mirror-duels-v1',frozen=str(frozen),
        frozen_plan_sha256=sha(frozen/'plan.json'),native_sha256=p['native_sha256'],
        input_behavior=p.get('input_behavior','legacy'),
        candidate=str(args.candidate.resolve()),candidate_sha256=sha(args.candidate),
        opponent=str(args.opponent.resolve()),opponent_sha256=sha(args.opponent),
        cases=cases,decks={k:decks[k] for k in selected},
        budget=dict(decisions=2048,seconds_per_game=120),precision='highest',
        grading='natural terminal only; invalid and budget exhaustion reported separately',
        scope='fit-deck mirrors against a compatible frozen policy; not a human ladder or unseen-deck exam',
        no_debug_injection=True,no_fixed_human_action_prefix=True,
        candidate_seat_is_not_a_first_turn_claim=True,promotion_eligible=False)
    write(args.output/'plan.json',exam)
    print('FROZEN_DUELS',len(cases),flush=True)


def evaluate(args):
    from supervision_runtime import Runtime,jax,jnp,np,extract_trace,tensor_contract,array_digest,hidden_identity_audit
    from ygoai.rl.checkpoint_compat import validate_checkpoint_compatibility
    exam = read(args.exam/'plan.json')
    if sha(Path(exam['frozen'])/'plan.json')!=exam['frozen_plan_sha256']:
        raise ValueError('duel runtime plan changed')
    for role in ('candidate','opponent'):
        if sha(exam[role])!=exam[role+'_sha256']:
            raise ValueError('frozen duel model changed')
    args.output.mkdir(parents=True,exist_ok=False)
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
        write(args.output/'inflight.json',dict(case=case['id'],status='running',exam_sha256=sha(args.exam/'plan.json')))
        record = dict(id=case['id'],definition=case['definition'],seed=case['seed'],
            candidate_seat=case['candidate_seat'],candidate_sha256=exam['candidate_sha256'],
            opponent_sha256=exam['opponent_sha256'],status='unknown',decisions=[])
        env = rt.env(record,case['seed'])
        states = [rt.agent.init_rnn_state(1),rt.agent.init_rnn_state(1)]
        checked = set()
        started = time.monotonic()
        try:
            env.reset()
            for step in range(exam['budget']['decisions']+1):
                snap = env.snapshot()
                if snap['invalid']:
                    record['status']='invalid';break
                if snap['done']:
                    record.update(status='natural_terminal',winner=snap['winner']);break
                if step==exam['budget']['decisions'] or time.monotonic()-started>exam['budget']['seconds_per_game']:
                    record['status']='budget_exhausted';break
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
                record['decisions'].append(dict(step=step,player=seat,actor=role,message=snap['message'],
                    phase=snap['phase'],menu=snap['menu'],action=action,observation_sha256=array_digest(obs),
                    policy_logits=scores.tolist(),policy_probabilities=probabilities.tolist(),value=v))
                env.step(action)
        except Exception as error:
            record.update(status='invalid',error=str(error))
        finally:
            snap=env.snapshot();events,state,responses=extract_trace(env,snap)
            record.update(events=events,final=state,core_responses=responses,core_seed=env.trace()['core_seed'],seconds=time.monotonic()-started)
            env.close()
        try:
            record['independent_audit']=rt.independent(record)
        except Exception as error:
            record.update(independent_audit=dict(passed=False,error=str(error)),untrusted_status=record['status'],status='invalid')
        if record['status']=='natural_terminal':
            winner=record['winner']
            if winner not in (0,1,2):
                raise ValueError('unknown natural winner code')
            record['candidate_result']='draw' if winner==2 else 'win' if winner==case['candidate_seat'] else 'loss'
        path=args.output/(case['id']+'.json');write(path,record)
        row=dict(id=case['id'],status=record['status'],candidate_seat=case['candidate_seat'],
            candidate_result=record.get('candidate_result'),path=path.name,record_sha256=sha(path),audit=record['independent_audit'])
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
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('freeze')
    for name in ('frozen','candidate','opponent','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--decks',type=int,default=4)
    p.add_argument('--seeds',type=int,default=32)
    p=commands.add_parser('evaluate')
    for name in ('exam','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-cases',type=int)
    args=parser.parse_args()
    if args.command=='freeze' and (args.decks<1 or args.seeds<1):
        parser.error('positive deck and seed counts required')
    if args.command=='evaluate' and args.max_cases is not None and args.max_cases<1:
        parser.error('positive qualification case count required')
    {'freeze':freeze,'evaluate':evaluate}[args.command](args)


if __name__=='__main__':
    main()
