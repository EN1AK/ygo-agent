"""Compare batched teaching with sequential current-parameter inference."""
import argparse
from pathlib import Path
import sys

from contracts import read, sha, write

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT/'scripts/studies/crystron_curriculum'))


def main():
    from supervision_runtime import Runtime, jax, jnp, np
    from ygoai.rl.scene_batch import pack_scenes
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen', 'checkpoint', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--group', default='repair')
    parser.add_argument('--rollouts', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    jax.config.update('jax_default_matmul_precision', 'highest')
    rt = Runtime(args.frozen)
    params = rt.load(args.checkpoint)
    reports = []
    for item in rt.plan['rows']:
        if item['group'] != args.group or item['split'] != 'fit':
            continue
        record, obs = rt.records[item['id']], rt.observations[item['id']]
        seat = record['definition']['controlled']
        own = [(i, d) for i, d in enumerate(record['decisions']) if d['player'] == seat]
        size = next(n for n in (64, 128, 256, 512, 1024, 2048) if len(own) <= n)
        packed = pack_scenes([(record, obs), (record, obs)], size)
        batch = {k: jnp.asarray(v) for k, v in packed[0].items()}
        apply = jax.jit(lambda p, x: rt.agent.apply(p, x, rt.agent.init_rnn_state(2), jnp.zeros(size*2, bool), None)[1])
        logits = np.asarray(apply(params, batch)).reshape(size, 2, -1)
        state = rt.agent.init_rnn_state(1)
        steps = []
        for t, (i, d) in enumerate(own):
            state, scores, value = rt.forward(params, {k:jnp.asarray(v[i:i+1]) for k,v in obs.items()}, state)
            count = len(d['menu'])
            sequential = np.asarray(scores)[0, :count]
            training = logits[t, 0, :count]
            steps.append(dict(step=i, teaching=d.get('teaching', True), target=d['action'],
                sequential_argmax=int(sequential.argmax()), batched_argmax=int(training.argmax()),
                max_legal_logit_error=float(np.max(np.abs(sequential-training))),
                target_margin=float(sequential[d['action']] - np.max(np.delete(sequential, d['action']))) if count > 1 else None))
        taught = [d for d in steps if d['teaching']]
        row = dict(id=item['id'], decisions=len(own), teaching=len(taught), steps=steps,
            sequential_correct=sum(d['sequential_argmax']==d['target'] for d in taught),
            batched_correct=sum(d['batched_argmax']==d['target'] for d in taught),
            argmax_agreement=sum(d['sequential_argmax']==d['batched_argmax'] for d in steps),
            max_legal_logit_error=max(d['max_legal_logit_error'] for d in steps))
        if args.rollouts:
            actual = read(args.rollouts/(item['id']+'-None.json'))
            first = None
            for i, d in enumerate(actual['decisions']):
                if i >= len(record['decisions']):
                    break
                teacher = record['decisions'][i]
                if (d['menu'], d['observation_sha256']) != (teacher['menu'], teacher['observation_sha256']):
                    break
                if d['actor']=='model' and d['action'] != teacher['action']:
                    first = dict(step=i, selected=d['action'], teacher=teacher['action'], menu=d['menu'])
                    break
            row.update(autonomous_status=actual['status'], first_same_root_deviation=first)
        reports.append(row)
        print({k:v for k,v in row.items() if k not in ('steps','first_same_root_deviation')}, flush=True)
    if not reports:
        raise ValueError('no fit rows for requested diagnostic group')
    write(args.output, dict(checkpoint_sha256=sha(args.checkpoint), plan_sha256=sha(args.frozen/'plan.json'),
        precision='highest', devices=[str(d) for d in jax.devices()], reports=reports,
        claim='teacher-history numerical and argmax agreement; autonomous result reported separately'))


if __name__ == '__main__':
    main()
