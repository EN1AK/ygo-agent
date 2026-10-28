"""Reproduce an old factual trace on the fixed core without querying a policy.

Every controlled-seat input and action index must remain exact. A changed
passive-opponent menu is allowed only when its frozen Cancel/first rule chooses
the identical semantic action. Require identical core responses and endpoint,
then independently replay the new complete event stream. This is diagnostic
evidence, never a new model score or permission to silently adopt old exams.
"""
import argparse
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

from contracts import read,sha,write

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))


def main():
    from scripts.collect_teaching_exercise import load_native
    from ygoai.rl.exercise_actor import array_digest,extract_trace,native_seed32
    from ygoai.rl.observation_schema import tensor_contract
    from ygoai.rl.core_audit import audit_record,validate_curriculum_binding
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('course','record','native','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    plan=read(a.course/'plan.json');validate_curriculum_binding(plan)
    runtime=Path(plan['runtime'])
    config=SimpleNamespace(native=a.native,database=runtime/'assets/cards.cdb',scripts=runtime/'assets/script',
        code_list=runtime/'code_list.txt',semantics=runtime/'semantics')
    native=load_native(config,runtime/'registration.ydk')
    if getattr(native,'core_synchro_level_ex_version',None)!='explicit-nonlevel-material-v1':
        raise ValueError('fixed Synchro-level core required')
    source=read(a.record);result=copy.deepcopy(source);d=source['definition'];seat=d['controlled']
    seed=source.get('seed')
    env=native.ExerciseActor(d.get('lua',''),d['initial'],str(config.semantics),
        native_seed32(d.get('seed',1) if seed is None else seed),'structured-lite-v1',d['seed_sequence'],d['duel_flags'],2048,False)
    changed=[];decisions=[]
    try:
        env.reset()
        for index,old in enumerate(source['decisions']):
            snap=env.snapshot();_,_,responses=extract_trace(env,snap)
            if (snap['message'],snap['player'],len(responses))!=(old['message'],old['player'],old['core_response_count']):
                raise ValueError('prompt or response boundary changed')
            obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
            digest=array_digest(obs)
            action=old['action']
            if snap['player']==seat:
                if snap['menu']!=old['menu'] or digest!=old['observation_sha256']:
                    raise ValueError('controlled policy input changed at step '+str(index))
            elif snap['menu']!=old['menu'] or digest!=old['observation_sha256']:
                if old['actor']!='passive_opponent':
                    raise ValueError('non-passive historical input changed')
                cancels=[i for i,x in enumerate(snap['menu']) if x['act']=='Cancel']
                action=cancels[0] if cancels else 0
                if snap['menu'][action]!=old['menu'][old['action']]:
                    raise ValueError('frozen passive opponent now chooses a different action')
                changed.append(index)
            decisions.append(dict(old,menu=snap['menu'],action=action,observation_sha256=digest))
            env.step(action)
        events,state,responses=extract_trace(env,env.snapshot())
        if responses!=source['core_responses'] or state!=source['final']:
            raise ValueError('factual responses or endpoint changed')
        result.update(decisions=decisions,events=events,final=state,core_responses=responses,core_seed=env.trace()['core_seed'],
            native_sha256=sha(a.native),full_legal_menu=False,diagnostic_replay_only=True)
    finally:
        env.close()
    proof=audit_record(result,plan['source_root'],config.database,config.scripts,plan['independent_library'])
    if not proof['passed']:
        raise ValueError('independent audit still rejects fixed-core trace')
    result['independent_audit']=proof
    write(a.output/'record.json',result)
    write(a.output/'report.json',dict(passed=True,source_record=str(a.record.resolve()),source_record_sha256=sha(a.record),
        native_sha256=sha(a.native),source_native_sha256=plan['native_sha256'],changed_passive_steps=changed,
        controlled_inputs_identical=True,core_responses_and_endpoint_identical=True,independent_audit=proof,
        record_sha256=sha(a.output/'record.json'),model_inference=False,official_model_exam=False))
    print('FIXED_CORE_FACTUAL_REPLAY_VERIFIED',len(changed),'passive-input revisions',flush=True)


if __name__=='__main__':
    main()
