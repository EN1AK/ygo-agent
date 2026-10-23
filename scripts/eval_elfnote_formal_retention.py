"""Audit the unchanged exposed exercise suite after ordinary formal PPO."""
import argparse,json,os,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','training','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--initial-course',action='store_true',help='Posthoc same-pipeline check of the fixed course parent, without optimization.')
    a=p.parse_args();a.root=a.root.resolve();a.output=a.output.resolve();a.source=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(a.source),str(a.source/'scripts')]
    from scripts.study_elfnote_coverage import read,write
    from scripts.study_elfnote_functional_teaching import functional_evaluate
    from scripts.study_elfnote_search import evaluate as normal_evaluate
    from scripts.study_elfnote_root_repair import score
    from scripts.collect_teaching_exercise import load_native
    from scripts.eval_capability_exercises import FrozenActor
    from scripts.teach_elfnote_combo import load_row,audit_actor
    from ygoai.rl.exercise_core import sha
    import jax
    assert all(d.platform=='gpu' for d in jax.devices())
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='float32'
    study=a.root/'training-runs/elfnote-recovery-context-20261023';m=read(study/'manifest.json');plan=m['plan']
    assert read(study/'final-verification.json')['passed']
    proof=read(a.training/'plain/verification.json');assert proof['passed'] and proof['process_exit_zero']
    checkpoint=Path(proof['checkpoint']);expected=proof['checkpoint_sha256']
    if a.initial_course:
        checkpoint=study/'full_context/model.flax_model'
        expected=next(x['checkpoint_sha256'] for x in read(study/'summary.json')['arms'] if x['arm']['name']=='full_context')
    assert sha(checkpoint)==expected
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008'
    a.native=a.root/'training-runs/elfnote-selected-scope-intake-03-20261014/exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    assert sha(a.native)==plan['native_sha256']
    a.database=release/'repo/assets/locale/zh/cards.cdb';a.scripts=release/'repo/scripts/script';a.code_list=release/'code_list.txt';a.semantics=release/'semantics'
    runtime=read(a.root/'training-runs/elfnote-runtime-binding-20261014.json');assert all(sha(a.root/name)==value for name,value in runtime.items())
    parent=a.root/plan['parent_study'];normal_plan=read(parent/'manifest.json')['plan']
    old=[load_row(Path(row['path'])) for row in read(parent/'training-rows.json')['rows'][31:36]]
    native=load_native(a,release/'opening-02/registration.ydk')
    actor=FrozenActor(checkpoint,a.code_list,a.semantics,{k:v[:1] for k,v in old[0][1].items()})
    forward=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3]);actor.forward=lambda obs,state:forward(actor.params,obs,state)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'manifest.json',dict(checkpoint_sha256=sha(checkpoint),role='initial_course' if a.initial_course else 'plain',course_manifest_sha256=sha(study/'manifest.json'),
        evaluator_sha256=sha(Path(__file__)),runtime=runtime,normal_cases=m['normal_cases'],functional_cases=m['cases'],
        scope='Same exposed exercises and diagnostic52 native; full-game runtime13aa assessed separately. No optimization, selection, or new unseen-family claim.'))
    started=time.time()
    functional=functional_evaluate(a,native,actor,m['cases'],plan,a.output/'functional',audit=True)
    normal=normal_evaluate(a,native,actor,m['normal_cases'],normal_plan,old,a.output/'normal',audit=True)
    # Normal-opening and legacy/synthetic fixtures have different core starts;
    # the independent verifier dispatches each old fixture to its existing path.
    import subprocess
    subprocess.run([sys.executable,str(Path(__file__).with_name('verify_elfnote_formal_retention.py')),
        '--root',str(a.root),'--training',str(a.training),'--study',str(a.output)],check=True)


if __name__=='__main__':main()
