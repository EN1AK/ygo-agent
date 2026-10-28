"""Predeclare endpoint exams, then execute after the paired producers terminate.

Endpoint selection is fixed by producer path and plan, before exam inference.
Six unseen components and natural-duel qualification remain separate exams.
The four-case qualification never replaces the frozen 256-attempt duel exam.
"""
import argparse
from pathlib import Path
import subprocess
import sys

from compare import audit_endpoint
from contracts import ledger, read, round_protocol, sha, write

PROJECT = Path(__file__).resolve().parents[3]


def freeze(args):
    run = args.run.resolve()
    protocol = read(run/'protocol.json')
    exam = protocol.get('independent_exam')
    if not exam or sha(Path(exam['path'])/'plan.json')!=exam['plan_sha256']:
        raise ValueError('independent exam was not frozen or has changed')
    output = run/args.name
    output.mkdir(exist_ok=False)
    endpoints = [dict(name='parent',checkpoint=protocol['initial_checkpoint'],
                      checkpoint_sha256=protocol['initial_checkpoint_sha256'])]
    for observation in args.rounds:
        current=round_protocol(run,observation)
        for arm in ('control','repair'):
            producer=run/(observation+'-training')/arm
            inputs=producer.parent/(arm+'-inputs')/'plan.json'
            plan=read(inputs)
            if (plan['parent_sha256']!=current['initial_checkpoint_sha256'] or
                    plan['stages'][0]['updates']!=current['training']['updates_per_arm']):
                raise ValueError('producer does not match the registered paired protocol')
            name=arm if len(args.rounds)==1 else observation+'-'+arm
            endpoints.append(dict(name=name,round=observation,producer_output=str(producer),
                producer=str(inputs),producer_plan_sha256=sha(inputs),
                checkpoint=str(producer/'endpoint/model.flax_model'),
                no_intermediate_checkpoint_selection=True))
    sources = ['scripts/studies/reality_loop/endpoint_exams.py',
        'scripts/studies/reality_loop/duel_exam.py',
        'scripts/studies/crystron_curriculum/evaluate_supervision.py',
        'scripts/studies/crystron_curriculum/supervision_runtime.py',
        'scripts/studies/reality_loop/contracts.py','scripts/studies/reality_loop/compare.py']
    p = dict(schema='reality-loop-endpoint-exams-v1',run=str(run),precision='highest',
        source_sha256={path:sha(PROJECT/path) for path in sources},
        research_protocol_sha256=sha(run/'protocol.json'),independent_exam=exam,
        endpoints=endpoints,rounds=args.rounds,natural_duels=dict(decks=4,seeds=32,planned_per_endpoint=256,
            qualification_prefix=4,opponent_endpoint='control' if len(args.rounds)==1 else args.rounds[-1]+'-control'),
        fixed_before_exam_inference=True,independent_score_checkpoint_selection=False,
        execute_training_concurrently=False,independent_exam_deferred_until_ppo_protocol_fixed=True,
        ppo_protocol_sha256=sha(run/'ppo-protocol.json') if (run/'ppo-protocol.json').exists() else None,
        promotion_eligible=False)
    write(output/'protocol.json',p)
    ledger(run,'endpoint_exam_protocol',[output/'protocol.json'],
        claim='all predefined terminal endpoints; no selection using unseen exam scores')


def command(argv,log):
    with log.open('x',encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable,*argv],stdout=stream,stderr=subprocess.STDOUT)
        write(log.with_suffix('.process.json'),dict(pid=process.pid,argv=argv))
        code = process.wait()
    log.with_suffix('.exit').write_text(str(code)+'\n',encoding='utf-8')
    if code:
        raise RuntimeError('exam producer failed: '+str(log))


def execute(args):
    run = args.run.resolve()
    folder = run/args.name
    protocol = read(folder/'protocol.json')
    if args.phase=='independent' and (not protocol.get('ppo_protocol_sha256') or
            not (run/'ppo-protocol.json').exists() or
            sha(run/'ppo-protocol.json')!=protocol['ppo_protocol_sha256']):
        raise ValueError('freeze the PPO protocol in an exam revision before opening the independent set')
    research = read(run/'protocol.json')
    for path, expected in protocol['source_sha256'].items():
        if sha(PROJECT/path)!=expected:
            raise ValueError('exam source changed; preserve protocol and register a revision')
    if sha(run/'protocol.json')!=protocol['research_protocol_sha256']:
        raise ValueError('research protocol changed')
    if sha(Path(protocol['independent_exam']['path'])/'plan.json')!=protocol['independent_exam']['plan_sha256']:
        raise ValueError('independent exam changed')
    # A written model is insufficient: require both terminal producer exits,
    # complete finite updates and independent evaluation before model exams.
    for e in protocol['endpoints']:
        if e['name']=='parent':
            if sha(e['checkpoint'])!=e['checkpoint_sha256']:
                raise ValueError('parent model changed')
        else:
            if sha(e['producer'])!=e['producer_plan_sha256']:
                raise ValueError('producer input plan changed')
            completion,_,_ = audit_endpoint(Path(e['producer_output']),research['training']['updates_per_arm'])
            if sha(e['checkpoint'])!=completion['checkpoint_sha256']:
                raise ValueError('exam is not using the registered terminal checkpoint')
    output = folder/('results-'+args.phase)
    output.mkdir(exist_ok=False)
    opponent = next(e['checkpoint'] for e in protocol['endpoints'] if e['name']==protocol['natural_duels']['opponent_endpoint'])
    for e in protocol['endpoints']:
        name = e['name']
        evidence=[]
        if args.phase=='independent':
            command([str(PROJECT/'scripts/studies/crystron_curriculum/evaluate_supervision.py'),
                '--frozen',protocol['independent_exam']['path'],'--checkpoint',e['checkpoint'],
                '--matmul-precision',protocol['precision'],
                '--output',str(output/(name+'-independent'))],output/(name+'-independent.log'))
            evidence.append(output/(name+'-independent/summary.json'))
        else:
            command([str(PROJECT/'scripts/studies/reality_loop/duel_exam.py'),'freeze',
                '--frozen',research['original'],'--candidate',e['checkpoint'],'--opponent',opponent,
                '--output',str(output/(name+'-duel-plan')),
                '--decks',str(protocol['natural_duels']['decks']),
                '--seeds',str(protocol['natural_duels']['seeds'])],output/(name+'-duel-freeze.log'))
            command([str(PROJECT/'scripts/studies/reality_loop/duel_exam.py'),'evaluate',
                '--exam',str(output/(name+'-duel-plan')),'--output',str(output/(name+'-duel-qualification')),
                '--max-cases',str(protocol['natural_duels']['qualification_prefix'])],output/(name+'-duel-qualification.log'))
            evidence.append(output/(name+'-duel-qualification/summary.json'))
        ledger(run,'endpoint_exam_observation',evidence,
            endpoint=name,checkpoint_sha256=sha(e['checkpoint']),
            phase=args.phase,claim='registered exam phase; full duel denominator and PPO retention remain pending')
    write(output/'completion.json',dict(completed=True,predeclared_endpoints=len(protocol['endpoints']),
        protocol_sha256=sha(folder/'protocol.json'),full_duel_exam_completed=False,
        ppo_retention_completed=False,promotion_eligible=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze','execute'])
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--name',default='endpoint-exams')
    p.add_argument('--rounds',nargs='+',default=['round1'])
    p.add_argument('--phase',choices=['qualification','independent'],default='qualification')
    args=p.parse_args()
    if Path(args.name).name!=args.name or args.name in ('','..','.'):
        p.error('exam protocol name must be one directory name')
    {'freeze':freeze,'execute':execute}[args.mode](args)


if __name__=='__main__':
    main()
