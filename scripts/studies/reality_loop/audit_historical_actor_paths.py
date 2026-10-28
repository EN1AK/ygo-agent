"""Replay every recorded SFT observation/before/after path on a corrected core.

No policy inference, new labels or score selection. Preserve the full source
denominator, deduplicating only identical record bytes. Every decision input,
menu, response boundary, core response and endpoint must remain exact; only
the established adjacent CARD_HINT notification permutation is normalized.
"""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

from contracts import read,sha,write
from reencode_overlay_course import event_blocks

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))


def main():
    from scripts.collect_teaching_exercise import load_native
    from ygoai.rl.exercise_actor import array_digest,extract_trace,native_seed32
    from ygoai.rl.observation_schema import tensor_contract
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','course','native','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.run=a.run.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    plan=read(a.course/'plan.json');runtime=Path(plan['runtime'])
    config=SimpleNamespace(native=a.native,database=runtime/'assets/cards.cdb',scripts=runtime/'assets/script',
        code_list=runtime/'code_list.txt',semantics=runtime/'semantics')
    native=load_native(config,runtime/'registration.ydk')
    if getattr(native,'core_synchro_level_ex_version',None)!='explicit-nonlevel-material-v1':
        raise ValueError('corrected Synchro core required')
    summaries=[]
    for name in ('round1','round2'):
        summaries.append(a.run/name/'evaluation/summary.json')
        for role in ('control','repair'):
            summaries += [a.run/(name+'-training')/role/'before/summary.json',
                a.run/(name+'-training')/role/'endpoint/evaluation/summary.json']
    unique={};summary_bindings={};source_rows=0
    for path in summaries:
        summary_bindings[str(path)]=sha(path)
        summary=read(path)
        for row in summary['results']:
            source=path.parent/row['path'];digest=sha(source);source_rows+=1
            record=read(source)
            if (record['checkpoint_sha256']!=summary['checkpoint_sha256'] or record['id']!=row['id'] or
                    record['seed']!=row['seed'] or record['status']!=row['status']):
                raise ValueError('historical record is not the registered model/case/outcome')
            unique.setdefault(digest,[]).append(str(source))
    results=[]
    for digest,paths in unique.items():
        source=read(paths[0]);d=source['definition'];seed=source.get('seed');env=None;step=-1
        result=dict(source_record_sha256=digest,paths=paths,id=source['id'],seed=seed,
            checkpoint_sha256=source['checkpoint_sha256'],source_status=source['status'],passed=False)
        try:
            if not source['independent_audit']['passed'] or source['independent_audit']['independent_library_sha256']!=plan['independent_library']['sha256']:
                raise ValueError('historical raw-core proof is absent or from another library')
            env=native.ExerciseActor(d.get('lua',''),d['initial'],str(config.semantics),
                native_seed32(d.get('seed',1) if seed is None else seed),'structured-lite-v1',d['seed_sequence'],d['duel_flags'],2048,False)
            env.reset()
            for step,decision in enumerate(source['decisions']):
                snap=env.snapshot();_,_,responses=extract_trace(env,snap)
                if (snap['message'],snap['player'],len(responses))!=(decision['message'],decision['player'],decision['core_response_count']):
                    raise ValueError('historical prompt/actor/response boundary changed')
                obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                if snap['menu']!=decision['menu'] or array_digest(obs)!=decision['observation_sha256']:
                    raise ValueError('historical menu or complete policy input changed')
                env.step(decision['action'])
            events,state,responses=extract_trace(env,env.snapshot())
            if responses!=source['core_responses'] or state!=source['final'] or event_blocks(events)!=event_blocks(source['events']):
                raise ValueError('historical responses, events or endpoint changed')
            result.update(passed=True,decisions=len(source['decisions']),all_policy_inputs_identical=True,
                responses_and_endpoint_identical=True,independent_event_proof_transfers_by_exact_replay=True)
        except Exception as error:
            result.update(error=str(error),step=step)
        finally:
            if env is not None:env.close()
        if any(sha(path)!=digest for path in paths):
            raise ValueError('historical source changed during replay')
        results.append(result)
        write(a.output/'progress.json',dict(source_rows=source_rows,unique_records=len(unique),completed=len(results),passed=sum(r['passed'] for r in results)))
        print(result['id'],seed,'verified' if result['passed'] else result['error'],flush=True)
    if not all(sha(path)==value for path,value in summary_bindings.items()):
        raise ValueError('historical summaries changed')
    proof=dict(passed=all(r['passed'] for r in results),source_rows=source_rows,unique_records=len(unique),
        summaries=summary_bindings,native_sha256=sha(a.native),course_plan_sha256=sha(a.course/'plan.json'),
        source_sha256=sha(__file__),records=results,model_inference=False,official_new_exam=False,
        scope='Exact path/input bridge for all recorded observations and both 512-update before/after SFT arms. No changed record is silently adopted. Future corrected-runtime behavior and PPO still need actual evaluation.')
    write(a.output/'report.json',proof)
    if not proof['passed']:raise RuntimeError('historical SFT paths are not all equivalent on the corrected core')
    print('ALL_HISTORICAL_SFT_PATHS_VERIFIED',len(results),source_rows,flush=True)


if __name__=='__main__':main()
