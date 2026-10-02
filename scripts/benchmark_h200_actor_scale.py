"""Run sequential bounded actor-count pilots from one verified checkpoint."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--parent-steps', type=int, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--actors', type=int, nargs='+', default=[30, 45, 60])
    parser.add_argument('--updates', type=int, default=40)
    parser.add_argument('--envs-per-actor', type=int, default=16)
    parser.add_argument('--env-threads', type=int, default=11)
    parser.add_argument('--sampler-repartition', action='store_true')
    args = parser.parse_args()
    if (args.updates < 20 or args.envs_per_actor < 2 or args.envs_per_actor % 2
            or args.env_threads < 1 or any(n <= 0 or n*args.envs_per_actor % 80 for n in args.actors)):
        parser.error('Use >=20 updates, positive thread counts, even env width, and total envs divisible by 80')
    args.output.mkdir(parents=True, exist_ok=False)
    sha = hashlib.sha256(args.parent.read_bytes()).hexdigest()
    meta = json.loads(Path(str(args.parent)+'.metadata.json').read_text())
    if meta['checkpoint_sha256'] != sha:
        raise ValueError('Parent checkpoint checksum mismatch')
    results = []
    for actors in args.actors:
        run = args.output / f'actors-{actors}'
        steps = actors * args.envs_per_actor * 64 * args.updates
        env = dict(os.environ, SOURCE_COMMIT=args.source_commit,
                   RUN_DIR=str(run), RUN_NAME=f'actor-scale-{actors}__02102026',
                   PARENT_CHECKPOINT=str(args.parent), PARENT_SHA256=sha,
                   PARENT_STEPS=str(args.parent_steps), CONTINUATION_STEPS=str(steps),
                   TARGET_STEPS=str(args.parent_steps+steps), ACTOR_THREADS=str(actors),
                   LOCAL_NUM_ENVS=str(args.envs_per_actor), LOCAL_ENV_THREADS=str(args.env_threads),
                   SAMPLER_REPARTITION='1' if args.sampler_repartition else '0',
                   NUM_MINIBATCHES='80', SAVE_INTERVAL=str(args.updates), LOG_FREQUENCY='5',
                   NATIVE_COMMIT='92154ab7a165c5749cead7d10f5f862aeb63f3bf',
                   NATIVE_SHA256='5962816be0362007c751c43a515d18a50abcd7716c00be579aac0aba1e6705a8')
        start = time.monotonic()
        with (args.output / f'actors-{actors}-launcher.log').open('wb') as stream:
            completed = subprocess.run(['bash', str(args.repo/'scripts/launch_h200_45m_to_100m_20261002.sh')],
                                       env=env, stdout=stream, stderr=stream)
        log = (run/'train.log').read_text(errors='replace') if (run/'train.log').exists() else ''
        sps = [int(x) for x in re.findall(r'SPS:\s*(\d+)', log)]
        finite = re.findall(r'optimizer_finite=\[\s*(True|False)\s*\].*total_notfinite=\[\s*(\d+)\s*\]', log)
        healthy = bool(finite) and all(a=='True' and b=='0' for a,b in finite)
        success = completed.returncode == 0 and (run/'completed.txt').exists() and len(sps)>=4 and healthy
        row = {'actors':actors, 'envs_per_actor':args.envs_per_actor,
               'env_threads':args.env_threads, 'sampler_repartition':args.sampler_repartition,
               'envs':actors*args.envs_per_actor, 'batch_size':actors*args.envs_per_actor*64,
               'minibatch_size':actors*args.envs_per_actor*64//80, 'steps':steps, 'run':str(run),
               'parent_sha256':sha, 'seconds':time.monotonic()-start,
               'exit_code':completed.returncode, 'finite':healthy, 'passed':success,
               'steady_sps':sum(sps[-4:])/4 if len(sps)>=4 else None, 'logged_sps':sps}
        results.append(row)
        (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if not success:
            raise RuntimeError('Pilot failed; inspect evidence before continuing')
    (args.output/'completed.txt').write_text('all actor pilots completed\n')


if __name__ == '__main__':
    main()
