"""Independent fixed-checkpoint evaluation, without training or model selection."""
import argparse
from pathlib import Path
from supervision_runtime import Runtime,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.checkpoint=a.checkpoint.resolve();a.output=a.output.resolve()
    rt=Runtime(a.frozen)
    results=rt.evaluate(rt.load(a.checkpoint),sha(a.checkpoint),a.output)
    if not all(x['audit']['passed'] for x in results):raise RuntimeError('evaluation replay audit failed')


if __name__=='__main__':main()
