"""Evaluate compatible checkpoints on the fixed normal and synthetic roots."""
import argparse
import json
import os
from pathlib import Path

from scripts.collect_teaching_exercise import load_native
from scripts.eval_capability_exercises import FrozenActor
from scripts.train_exercise_demonstration import evaluate
from ygoai.rl.observation_schema import tensor_contract
from ygoai.rl.exercise_core import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('native', 'database', 'code-list', 'scripts', 'semantics', 'combo', 'battle', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, action='append', required=True)
    a = p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path): setattr(a,k,v.resolve())
    a.checkpoint = [path.resolve() for path in a.checkpoint]
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import jax
    jax.config.update('jax_default_matmul_precision','highest')
    native = load_native(a, a.combo/'registration.ydk')
    definition=json.loads((a.combo/'v0-reference-0.json').read_text())['definition']
    env=native.ExerciseActor('',definition['initial'],str(a.semantics))
    env.reset()
    sample={k:v for k,v in env.snapshot()['observation'].items() if k in tensor_contract('structured-lite-v1')}
    env.close()
    results=[]
    for i,checkpoint in enumerate(a.checkpoint):
        before=sha(checkpoint)
        actor=FrozenActor(checkpoint,a.code_list,a.semantics,sample)
        rows=evaluate(a,native,actor,f'checkpoint-{i}')
        assert sha(checkpoint)==before
        results.append(dict(checkpoint=str(checkpoint),sha256=before,results=rows))
    (a.output/'summary.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
