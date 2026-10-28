"""Certify complete recorded prefixes at the known 4096-call native fuse.

Every matching independent-core chain branch must consume every recorded
response, reach the same fuse and reproduce every event and final field state.
No JAX/native actor imports, winner inference, new policy decisions or relaxed
ambiguity limits. This certifies a resource-bounded prefix, not a natural result.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ygoai.rl.chain_response import canonical_responses,chain_choice,chain_response_candidates
from ygoai.rl.exercise_core import PROMPTS
from ygoai.rl.human_replay import comparable_game_events
from engine_budget_audit import finish_prefix, eligible, trajectory_digest, PROCESS_LIMIT


def resolve_library(request,source):
    binding=request.get('library')
    if binding is None:return source/'resources/libocgcore-current-v2.so'
    library=Path(binding['path']);manifest=Path(binding['manifest'])
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(library)!=binding['sha256'] or digest(manifest)!=binding['manifest_sha256']:
        raise ValueError('independent library binding changed')
    proof=json.loads(manifest.read_text())
    if (proof['schema']!='independent-actor-random-select-v1' or
            proof['library_sha256']!=binding['sha256'] or proof['changed_files']!=['libgroup.cpp'] or
            proof['model_native_changed'] is not False or proof.get('load_passed') is not True):
        raise ValueError('independent compatibility provenance mismatch')
    return library


def cloned_branch(core,action,continuation,budget):
    if len(list(Path('/proc/self/task').iterdir()))!=1:
        raise ValueError('exact core cloning requires a single-threaded CPU worker')
    reader,writer=os.pipe();pid=os.fork()
    if pid==0:
        os.close(reader)
        try:
            action()
            result=continuation(budget)
        except BaseException as error:
            result=dict(passed=False,error=str(error))
        try:
            data=json.dumps(result).encode()
            while data:
                count=os.write(writer,data);data=data[count:]
        finally:
            os.close(writer);os._exit(0)
    os.close(writer);parts=[]
    try:
        while True:
            data=os.read(reader,65536)
            if not data:break
            parts.append(data)
    finally:os.close(reader)
    _,status=os.waitpid(pid,0)
    if status!=0 or not parts:raise ValueError('cloned core branch did not exit cleanly')
    result=json.loads(b''.join(parts))
    if not result.get('passed'):raise ValueError('independent clone rejected: '+result.get('error','unknown'))
    return result


def audit(request):
    limit=request.get('max_variants',8)
    if type(limit) is not int or not 1<=limit<=8:raise ValueError('audit branch limit must be 1..8')
    source=Path(request['source']);sys.path.insert(1,str(source))
    library=resolve_library(request,source)
    from modern_core import Core
    record=request['record'];definition=record['definition']
    if (not eligible(record) or record['status'] != 'budget_exhausted' or
            record.get('core_process_cutoff', {}).get('limit') != PROCESS_LIMIT or
            len(record.get('native_sha256', '')) != 64):
        raise ValueError('explicit captured native-budget prefix and runtime binding required')
    canonical_responses(record)
    prompts=[e for e in record['events'] if e['op'] in PROMPTS]
    if len(prompts) not in (len(record['core_responses']),len(record['core_responses'])+1):
        raise ValueError('source core prompts cannot be bound to responses')
    chains={}
    for index,response in enumerate(record['core_responses']):
        if prompts[index]['op']==16:
            raw=bytes.fromhex(response['data'])
            if response['kind']!='int' or len(raw)!=4:raise ValueError('invalid source chain response encoding')
            chains[index]=(prompts[index]['player'],chain_choice(prompts[index],int.from_bytes(raw,'little',signed=True)))
    core=Core(library,Path(request['database']),Path(request['scripts']),
        seed=record.get('core_seed',1),seed_sequence=definition['seed_sequence'] or None)
    try:
        if definition.get('lua'):
            core.lp=[8000,8000];content=definition['lua'].encode();name=b'__crystron_resource_fixture.lua'
            core.buffers[name.decode()]=(ctypes.create_string_buffer(content),len(content))
            core.assets[name.decode()]=hashlib.sha256(content).hexdigest()
            if not core.preload_script(core.d,name,len(name)):raise ValueError('independent fixture preload rejected')
            core._check();core.start_duel(core.d,definition['duel_flags']|2)
        else:core.start_standard(definition['initial'],duel_flags=definition['duel_flags'])

        def replay(start,budget,remaps=0):
            for index in range(start,len(record['core_responses'])):
                prompt=core.next_prompt()
                if prompt is None:raise ValueError('independent core ended early')
                data=bytes.fromhex(record['core_responses'][index]['data'])
                candidates=[data]
                if index in chains:
                    if prompt['op']!=16 or prompt['player']!=chains[index][0]:
                        raise ValueError('independent chain prompt/actor drift')
                    candidates=chain_response_candidates(prompt,chains[index][1])
                if len(candidates)>1:
                    if len(candidates)>budget:raise ValueError('independent ambiguity branch budget exhausted')
                    outcomes=[]
                    for candidate in candidates:
                        outcomes.append(cloned_branch(core,lambda value=candidate:core.respond(prompt,value),
                            lambda remaining,value=candidate:replay(index+1,remaining,remaps+(value!=data)),budget//len(candidates)))
                    return dict(passed=True,validated_variants=sum(x['validated_variants'] for x in outcomes),
                        ambiguity_nodes=1+sum(x['ambiguity_nodes'] for x in outcomes),
                        semantic_chain_index_remaps=max(x['semantic_chain_index_remaps'] for x in outcomes),
                        events=max(x['events'] for x in outcomes), recorded_events=len(record['events']), core_process_calls=PROCESS_LIMIT,
                        exact_recorded_prefix=all(x['exact_recorded_prefix'] for x in outcomes),
                        natural_outcome_certified=False)
                remaps+=candidates[0]!=data
                core.respond(prompt,candidates[0])
            proof = finish_prefix(core, record)
            proof['semantic_chain_index_remaps'] = remaps
            return proof

        result=replay(0,limit)
        result.update(kind='engine-process-budget-prefix-v1', trajectory_sha256=trajectory_digest(record), responses=len(record['core_responses']),clone_mode='fork-at-ambiguous-core-prompt',branch_limit=limit,
            independent_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
        return result
    finally:core.close()


if __name__=='__main__':
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    try:result=audit(json.load(sys.stdin))
    except BaseException as error:result=dict(passed=False,error=str(error))
    print('AUDIT_RESULT '+json.dumps(result),flush=True)
