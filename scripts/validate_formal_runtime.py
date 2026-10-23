"""CPU release admission with isolated over-capacity rejection and real schemas.

This never starts an optimizer or claims GPU/model/live-match qualification.
"""
import argparse,hashlib,importlib.metadata,json,os,re,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2026102301)
    parser.add_argument('--attempt',default='cpu-v2')
    parser.add_argument('--overflow-child',action='store_true')
    args=parser.parse_args();release=args.release.resolve();repo=release/'repo'
    if args.overflow_child and os.name=='posix':
        import resource
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if hasattr(os,'sched_getaffinity'):
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[8:12] if len(allowed)>=12 else allowed[-4:])
    prep=json.loads((release/'preparation.json').read_text());source=json.loads((release/'source-manifest.json').read_text())
    assert source['commit']==prep['source']
    assert all(hashlib.sha256((repo/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in source['all_files'].items())
    sys.path[:0]=[str(repo),str(repo/'ygoenv'),str(repo/'scripts')]
    import ygoenv.ygopro.ygopro_ygoenv as native
    assert Path(native.__file__).resolve()==repo/'ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so'
    assert sha(Path(native.__file__))==prep['native_sha256'] and native.exercise_training_header_sha256==prep['header_sha256']
    assert native.chain_event_provenance_version==prep['chain_feature_version']
    from scripts import validate_structured_environment as validator
    from ygoai.rl.observation_schema import LEGACY_SCHEMA,STRUCTURED_LITE_SCHEMA,tensor_contract,manifest
    from ygoai.utils import init_ygopro
    import numpy as np
    deck=release/'decks/replay-elfnote.ydk';semantics=release/'semantics';codes=release/'code_list.txt'
    assert sha(deck)==prep['deck_sha256'] and sha(codes)==prep['code_list_sha256'] and sha(semantics/'metadata.json')==prep['semantic_metadata_sha256']
    name=init_ygopro('YGOPro-v1','chinese',str(deck),str(codes))
    env_args=SimpleNamespace(seed=args.seed,deck_name=name,semantic_assets=semantics)
    if args.overflow_child:
        result=validator.overflow_probe(env_args)
        print(json.dumps(dict(mode='explicit_overflow_flag',result=result)),flush=True)
        return
    assert re.fullmatch(r'[a-z0-9-]+',args.attempt)
    output=release/'validation'/args.attempt;output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),output/'validator.py')
    original=validator.make_env
    def make_env(env_args,schema,max_options=128):return original(env_args,schema,max_options)
    validator.make_env=make_env
    reports={};shapes={}
    for schema in [LEGACY_SCHEMA,STRUCTURED_LITE_SCHEMA]:
        env=validator.make_env(env_args,schema);obs=validator.unpack_reset(env.reset());contract=tensor_contract(schema,max_options=128)
        try:
            assert set(obs)==set(contract)
            for key,spec in contract.items():assert np.asarray(obs[key]).shape==(1,)+spec.shape and str(np.asarray(obs[key]).dtype)==spec.dtype,(schema,key)
        finally:env.close()
        shapes[schema]={key:list(np.asarray(obs[key]).shape) for key in contract}
        first,second=validator.run_duel(env_args,schema),validator.run_duel(env_args,schema)
        assert first==second,'Fixed-seed complete schema replay changed'
        if schema==STRUCTURED_LITE_SCHEMA:assert not first['hidden_semantic_violations'] and not first['unsafe_event_references'] and not first['structured_identity_mismatches']
        reports[schema]=first
        (output/(schema+'.schema.json')).write_text(json.dumps(manifest(schema,max_options=128),indent=2))
    child=subprocess.run([sys.executable,str(output/'validator.py'),'--release',str(release),'--seed',str(args.seed),'--overflow-child'],capture_output=True,text=True,timeout=90,cwd=release)
    if child.returncode==0:
        overflow=json.loads(child.stdout.strip().splitlines()[-1]);assert overflow['result']['overflow_observed']
    else:
        assert child.returncode in [-6,134,1] and re.search(r'\[protocol boundary\] \d+ legal actions exceed model capacity 2',child.stderr),(child.returncode,child.stderr)
        overflow=dict(mode='rejects_over_capacity',capacity=2,returncode=child.returncode,expected_boundary_rejection=True)
    (output/'overflow-child.json').write_text(json.dumps(dict(returncode=child.returncode,stdout=child.stdout,stderr=child.stderr,interpretation=overflow),indent=2))
    after=validator.run_duel(env_args,STRUCTURED_LITE_SCHEMA);assert after==reports[STRUCTURED_LITE_SCHEMA]
    result=dict(passed=True,source_commit=source['commit'],source_files=len(source['all_files']),native_sha256=prep['native_sha256'],loaded_native=str(native.__file__),
        capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),chain_feature_version=native.chain_event_provenance_version,
        reports=reports,actual_shapes=shapes,overflow=overflow,parent_healthy_after_negative_control=True,validator_sha256=sha(output/'validator.py'),python=sys.version,
        dependencies=sorted({str(d.metadata['Name']):d.version for d in importlib.metadata.distributions()}.items()),
        scope='CPU real-engine schema, deterministic replay, visibility, actual shapes and isolated negative control only; no GPU optimization, model reload or live-strength acceptance')
    (output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['dependencies','actual_shapes','reports']}),flush=True)


if __name__=='__main__':main()
