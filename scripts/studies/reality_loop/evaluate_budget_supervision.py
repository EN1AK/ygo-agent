"""Independent fixed-checkpoint evaluation, without training or model selection."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'crystron_curriculum'))
from budget_runtime import Runtime
from supervision_runtime import sha,write,jax
from ygoai.rl.checkpoint_compat import validate_checkpoint_compatibility


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--matmul-precision',choices=['default','high','highest'],default='highest')
    a=p.parse_args();a.checkpoint=a.checkpoint.resolve();a.output=a.output.resolve()
    jax.config.update('jax_default_matmul_precision',a.matmul_precision)
    rt=Runtime(a.frozen)
    validate_checkpoint_compatibility(a.checkpoint,observation_schema='structured-lite-v1',
        model_args=rt.modelargs,code_list_hash=rt.plan['code_list_sha256'],
        semantic_table_hash=rt.plan['semantics_sha256'],capacities=rt.capacities)
    results=rt.evaluate(rt.load(a.checkpoint),sha(a.checkpoint),a.output)
    write(a.output/'inference-contract.json',dict(matmul_precision=a.matmul_precision,
        devices=[str(d) for d in jax.devices()],checkpoint_sha256=sha(a.checkpoint),
        native_sha256=rt.plan['native_sha256'],source_sha256=sha(__file__)))
    if not all(x['audit']['passed'] for x in results):raise RuntimeError('evaluation replay audit failed')


if __name__=='__main__':main()
