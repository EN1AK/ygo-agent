"""Direct on-policy REINFORCE of Laya weights using simulator tool observations.

This is a development-only, fixed-mechanism experiment, not a duel policy or
official Jev training. The Laya encoder and existing decision head are updated;
there is no secondary actor, critic, supervised answer target, or cached reward.
"""
import argparse
import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path

from ygoai.rl.jev_experiment import BranchEnvironment, SCENARIOS, returns_to_go, leave_one_out_baselines


class LayaPolicy:
    def __init__(self, model_dir, device, max_len=4096, init_weights=None):
        import torch
        import laya
        from safetensors.torch import load_file
        if importlib.metadata.version('laya') != '0.4.0':
            raise ValueError('experiment requires the reviewed laya==0.4.0 interface')
        self.torch = torch
        self.agent = laya.load(str(model_dir), device=device, fast=False, compile=False)
        self.model, self.tok = self.agent.model, self.agent.tok
        if init_weights:
            self.model.load_state_dict(load_file(str(init_weights)), strict=True)
        self.model.eval()  # Disable dropout; eval does not disable autograd.
        for name, param in self.model.named_parameters():
            # The upstream confidence head is unused, not a newly trained actor.
            param.requires_grad_(not name.startswith('act_head.'))
        self.device, self.max_len = device, max_len

    def encode(self, request):
        from laya.common import build_sequence, render_options, QTYPES
        q = dict(t='choice', ins='Choose the next operation to achieve the stated goal. Simulation is hypothetical; commit executes it.',
                 crit=request['criteria'])
        option_lengths = [len(self.tok(' '+s.replace(self.tok.mask_token, ' '),
                                      add_special_tokens=False)['input_ids'])
                          for s in render_options(q)]
        if any(n > 48 for n in option_lengths):
            raise ValueError('upstream 48-token option cap would truncate an action')
        head = sum(n+1 for n in option_lengths) + len(self.tok('choice question: '+q['ins'], add_special_tokens=False)['input_ids']) + 16
        ids, markers, stats, trunc = build_sequence(self.tok, request['state'], q,
             max_len=self.max_len, head_max_len=head, return_stats=True, return_truncation_stats=True)
        if trunc['truncated'] or len(markers) != len(request['criteria']) or stats['options_distinct'] != len(markers):
            raise ValueError('state/action truncation or collapsed choices')
        return dict(ids=ids, markers=markers, qtype=QTYPES['choice'])

    def distribution(self, encoded):
        t = self.torch
        args = dict(input_ids=t.tensor([encoded['ids']], device=self.device),
                    attention_mask=t.ones((1, len(encoded['ids'])), dtype=t.long, device=self.device),
                    marker_pos=t.tensor([encoded['markers']], device=self.device),
                    marker_mask=t.ones((1, len(encoded['markers'])), dtype=t.bool, device=self.device),
                    qtype=t.tensor([encoded['qtype']], device=self.device))
        logits, _ = self.model(**args)
        if not t.isfinite(logits).all():
            raise ValueError('nonfinite model logits')
        return t.distributions.Categorical(logits=logits[0])


def episode(policy, env, case, rng, greedy=False):
    env.reset(case)
    records = []
    while not env.done:
        request = env.request(rng)
        encoded = policy.encode(request)
        with policy.torch.no_grad():
            dist = policy.distribution(encoded)
            action = dist.probs.argmax() if greedy else dist.sample()
            index = int(action.item())
            log_prob = float(dist.log_prob(action).item())
            probabilities = dist.probs.cpu().tolist()
        operation = request['operations'][index]
        reward = env.step(operation)
        records.append(dict(request=request, encoded=encoded, selected=index,
                            operation=operation, old_log_prob=log_prob,
                            probabilities=probabilities, reward=reward))
    return dict(case=case, steps=records, total=sum(r['reward'] for r in records),
                success=env.success, evidence=env.evidence)


def train_group(policy, optimizer, episodes):
    t = policy.torch
    baselines = leave_one_out_baselines([e['total'] for e in episodes])
    optimizer.zero_grad(set_to_none=True)
    losses, max_drift = [], 0.
    for e, baseline in zip(episodes, baselines):
        for step, future_return in zip(e['steps'], returns_to_go([r['reward'] for r in e['steps']])):
            dist = policy.distribution(step['encoded'])
            lp = dist.log_prob(t.tensor(step['selected'], device=policy.device))
            drift = abs(float(lp.detach())-step['old_log_prob'])
            max_drift = max(max_drift, drift)
            if drift > 1e-4:
                raise ValueError('on-policy log probability changed before update')
            loss = -lp * (future_return-baseline) / len(episodes)
            if not t.isfinite(loss):
                raise ValueError('nonfinite policy loss')
            loss.backward()
            losses.append(float(loss.detach()))
    parameters = [p for p in policy.model.parameters() if p.requires_grad]
    norm = t.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    encoder_grad = any(p.grad is not None and bool(t.count_nonzero(p.grad))
                       for name, p in policy.model.named_parameters() if name.startswith('encoder.'))
    optimizer.step()
    return dict(loss=sum(losses), grad_norm=float(norm), encoder_nonzero_gradient=encoder_grad,
                max_log_prob_drift=max_drift)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir', 'core', 'database', 'scripts', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--init-weights', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--updates', type=int, default=3)
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--max-probes', type=int, default=2)
    p.add_argument('--probe-cost', type=float, default=.02)
    p.add_argument('--learning-rate', type=float, default=1e-6)
    p.add_argument('--seed', type=int, default=8102026)
    p.add_argument('--max-len', type=int, default=4096)
    p.add_argument('--model-revision', default='1720e3e3357cfe1e281542e223f8273b0890ca34')
    args = p.parse_args()
    if args.group_size < 2 or args.updates < 0 or args.learning_rate <= 0 or not 0 < args.max_len <= 8192:
        p.error('invalid training budget')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    policy = LayaPolicy(args.model_dir, args.device, args.max_len, args.init_weights)
    optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],
                                 lr=args.learning_rate, weight_decay=0.)
    env = BranchEnvironment(args.core, args.database, args.scripts,
                           max_probes=args.max_probes, probe_cost=args.probe_cost)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(model_id='convaiinnovations/laya-multilingual',
                  input_weight_sha256=sha(args.init_weights or args.model_dir/'model.safetensors'),
                  algorithm='on-policy REINFORCE, leave-one-out baseline, no critic',
                  trainable_parameters=sum(p.numel() for p in policy.model.parameters() if p.requires_grad),
                  scope='development-only known synthetic macro-action mechanism tasks',
                  full_duel_policy=False, versions={k: importlib.metadata.version(k) for k in ('laya','torch','transformers')})
    config['source_sha256'] = {str(path): sha(path) for path in (
        Path(__file__), Path('scripts/run_jev_branch.py'), Path('ygoai/rl/jev_experiment.py'),
        Path('ygoai/rl/exercise_core.py'), Path('ygoai/rl/exercise_starters.py'))}
    (args.output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    before = {name: param.detach().cpu().clone() for name, param in policy.model.named_parameters()
              if name in ('encoder.embeddings.tok_embeddings.weight', 'scorer.3.weight')}
    # Actual encoder names vary; always track one backbone parameter separately.
    first_name, first_param = next((n,p) for n,p in policy.model.named_parameters() if n.startswith('encoder.'))
    before[first_name] = first_param.detach().cpu().clone()
    start = time.monotonic()
    metrics = []
    with (args.output/'episodes.jsonl').open('x', encoding='utf-8') as log:
        def evaluate(label):
            results = []
            for i, case in enumerate(SCENARIOS):
                e = episode(policy, env, case, random.Random(args.seed+100+i), greedy=True)
                log.write(json.dumps(dict(stage=label, **e), ensure_ascii=False)+'\n'); log.flush()
                results.append(dict(case=case, success=e['success'], total=e['total'], steps=len(e['steps'])))
            return results
        initial = evaluate('before')
        for update in range(args.updates):
            case = list(SCENARIOS)[update % len(SCENARIOS)]
            group = []
            for _ in range(args.group_size):
                e = episode(policy, env, case, rng)
                log.write(json.dumps(dict(stage='train', update=update, **e), ensure_ascii=False)+'\n')
                log.flush()
                group.append(e)
            stat = train_group(policy, optimizer, group)
            stat.update(update=update, case=case, successes=sum(e['success'] for e in group),
                        episodes=len(group), elapsed_seconds=time.monotonic()-start)
            metrics.append(stat)
            print(json.dumps(stat), flush=True)
        final = evaluate('after')
    deltas = {n: float((dict(policy.model.named_parameters())[n].detach().cpu()-v).abs().max())
              for n,v in before.items()}
    weights = args.output/'model.safetensors'
    save_file({n:t.detach().cpu().contiguous() for n,t in policy.model.state_dict().items()}, str(weights))
    torch.save(dict(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    python_rng=rng.getstate()), args.output/'optimizer.pt')
    summary = dict(before=initial, after=final, updates=metrics, parameter_max_abs_delta=deltas,
                   output_weight_sha256=sha(weights), episodes_sha256=sha(args.output/'episodes.jsonl'),
                   elapsed_seconds=time.monotonic()-start, generalization_claim=False,
                   success=all(torch.isfinite(p).all().item() for p in policy.model.parameters()) and
                           (args.updates == 0 or any(v > 0 for v in deltas.values())))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    if not summary['success']:
        raise SystemExit('No verified parameter change or nonfinite model')


if __name__ == '__main__':
    main()
