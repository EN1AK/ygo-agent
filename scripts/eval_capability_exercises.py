"""Frozen actor baseline on synthetic exercises, using the native actor writer.

Teacher prefix is replayed from the declared synthetic episode boundary.
Each acting seat's recurrent state is recomputed, never loaded from a fixture.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from ygoai.rl.exercise_actor import (ActorReference, actor_grade, array_digest,
                                   extract_trace, hidden_identity_audit, initial_cards, settled)
from ygoai.rl.exercise_starters import CASES, fixture
from ygoai.rl.exercise_core import sha


class FrozenActor:
    def __init__(self, checkpoint, code_list, semantics, sample):
        import jax
        import jax.numpy as jnp
        import flax.serialization
        from ygoai.rl.jax.agent import RNNAgent, ModelArgs
        from ygoai.rl.checkpoint_compat import validate_checkpoint_compatibility
        self.jax = jax
        model_args = ModelArgs(observation_schema='structured-lite-v1')
        validate_checkpoint_compatibility(checkpoint, observation_schema=model_args.observation_schema,
            model_args=model_args, code_list_hash=sha(code_list), semantic_table_hash=sha(semantics / 'metadata.json'),
            capacities=dict(max_cards=80, max_options=128, history_actions=32, public_events=32, group_references=8))
        self.agent = RNNAgent(**asdict(model_args), embedding_shape=sum(bool(s.strip()) for s in code_list.read_text().splitlines()))
        template = jax.jit(self.agent.init)(jax.random.PRNGKey(0), sample, self.agent.init_rnn_state(1))
        self.params = flax.serialization.from_bytes(template, checkpoint.read_bytes())
        self.forward = jax.jit(lambda obs, state: self.agent.apply(self.params, obs, state)[:3])
        self.reset()

    def reset(self):
        self.states = [self.agent.init_rnn_state(1), self.agent.init_rnn_state(1)]

    def predict(self, obs, player):
        leaves = self.jax.tree_util.tree_leaves(self.states[player])
        before = array_digest({str(i): x for i, x in enumerate(leaves)})
        self.states[player], logits, value = self.forward(obs, self.states[player])
        return np.asarray(logits)[0], float(np.asarray(value).reshape(-1)[0]), before


def run(args, native, actor, case, branch, mode):
    from ygoai.rl.observation_schema import tensor_contract
    lua, _ = fixture(case)
    schema = getattr(actor, 'observation_schema', 'structured-lite-v1')
    env_args = (lua, initial_cards(lua), str(args.semantics))
    env = native.ExerciseActor(*env_args) if schema == 'structured-lite-v1' else native.ExerciseActor(*env_args, 1, schema)
    policy = ActorReference(case, branch)
    controlled = 1 if case == 'interaction' else 0
    actor.reset() if actor else None
    decisions, observations, root, root_event, error = [], [], None, None, None
    status, stable = 'unknown', False
    core_seed = None
    private_rows = 0
    try:
        env.reset()
        for step in range(160):
            snapshot = env.snapshot()
            obs = {k: snapshot['observation'][k] for k in sorted(tensor_contract(schema))}
            events, state, responses = extract_trace(env, snapshot)
            if snapshot['invalid']:
                raise ValueError('native invalid game')
            if settled(case, snapshot, events, root_event):
                stable = True
                status = actor_grade(case, state, events, stable)
                break
            if snapshot['done']:
                raise ValueError('terminated before exercise root')
            at_root = root is None and policy.is_root(snapshot, events)
            if step == 0 or at_root:
                private_rows += hidden_identity_audit(obs)
            logits, value, rnn_hash = actor.predict(obs, snapshot['player']) if actor else (None, None, None)
            # A reference may be undefined after model divergence. Never invent
            # a correct-action label in those later menus.
            try:
                reference = policy.choose(snapshot, events)
            except ValueError:
                if mode != 'policy' or root is None or snapshot['player'] != controlled:
                    raise
                reference = None
            if at_root:
                root_event = len(events) - 1
                root = dict(step=step, event_index=root_event, observation_sha256=array_digest(obs),
                            rnn_sha256=rnn_hash, player=snapshot['player'], menu=snapshot['menu'])
            use_actor = mode == 'policy' and root is not None and snapshot['player'] == controlled
            count = len(snapshot['menu'])
            if not count:
                raise ValueError('empty actor menu')
            action = int(np.argmax(logits[:count])) if use_actor else reference
            if action is None or not 0 <= action < count:
                raise ValueError('no legal action')
            row = dict(step=step, player=snapshot['player'], message=snapshot['message'],
                       menu=snapshot['menu'], action=action, reference_action=reference,
                       source='actor' if use_actor else 'reference', observation_sha256=array_digest(obs),
                       rnn_sha256=rnn_hash, policy_logits=logits[:count].tolist() if actor else None,
                       state_value=value, core_response_count=len(responses))
            if actor:
                weights = np.exp(logits[:count] - logits[:count].max())
                row['policy_probabilities'] = (weights / weights.sum()).tolist()
                if not np.isfinite(logits[:count]).all() or not np.isfinite(value):
                    raise ValueError('nonfinite policy output')
            decisions.append(row)
            observations.append(obs)
            env.step(action)
        else:
            status = 'budget_exhausted'
        snapshot = env.snapshot()
        events, state, responses = extract_trace(env, snapshot)
        core_seed = env.trace()['core_seed']
    except (ValueError, RuntimeError) as exc:
        status, error = 'invalid_fixture', str(exc)
        events, state, responses = [], None, []
    finally:
        env.close()
    tensors = {key: np.concatenate([obs[key] for obs in observations], axis=0)
               for key in observations[0]} if observations else {}
    return dict(case=case, branch=branch, mode=mode, status=status, error=error,
                root=root, decisions=decisions, events=events, final_state=state,
                observation_archive_digest=array_digest(tensors),
                core_responses=responses, stable_boundary=stable, private_rows_checked=private_rows,
                core_seed=core_seed,
                source_kind='synthetic_mechanism_fixture', history_origin='declared synthetic episode start',
                full_standard_duel_history_verified=False, training_ready=False), tensors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--code-list', type=Path, required=True)
    p.add_argument('--semantics', type=Path, required=True)
    p.add_argument('--scripts', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case', choices=list(CASES))
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.chdir(args.scripts.parent) # actual init_module resolves ./script assets
    spec = importlib.util.spec_from_file_location('exercise_actor_native', args.native)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    # init_module loads the existing code list unchanged; the fixture's deck
    # composition comes from its declared initial cards, not this registration.
    deck = args.output / 'registration.ydk'
    deck.write_text('#main\n' + '46986414\n' * 40 + '#extra\n!side\n')
    native.init_module(str(args.database), str(args.code_list), {'_exercise': str(deck)})
    inputs = [args.native, args.database, args.code_list]
    inputs += sorted(args.semantics.rglob('*'))
    inputs += sorted(args.scripts.rglob('*.lua'))
    source_root = Path(__file__).resolve().parents[1]
    inputs += [source_root / name for name in (
        'scripts/eval_capability_exercises.py', 'ygoai/rl/exercise_actor.py',
        'ygoai/rl/exercise_core.py', 'ygoai/rl/exercise_starters.py',
        'ygoai/rl/jax/agent.py', 'ygoai/rl/observation_schema.py')]
    if args.checkpoint:
        inputs += [args.checkpoint, Path(str(args.checkpoint) + '.metadata.json')]
    protected = {str(path): sha(path) for path in inputs if path.is_file()}
    actor = None
    if args.checkpoint:
        from ygoai.rl.observation_schema import tensor_contract
        lua, _ = fixture('battle')
        sample_env = native.ExerciseActor(lua, initial_cards(lua), str(args.semantics))
        sample_env.reset()
        sample = sample_env.snapshot()['observation']
        sample_env.close()
        sample = {k: sample[k] for k in tensor_contract('structured-lite-v1')}
        actor = FrozenActor(args.checkpoint, args.code_list, args.semantics, sample)
    results = []
    reference_roots = {}
    for case, definition in CASES.items():
        if args.case and case != args.case:
            continue
        # Audit every known reference including bad branches before policy runs.
        for branch, expected in definition['branches'].items():
            modes = ['reference']
            if actor and branch == next(iter(definition['branches'])):
                modes += ['policy']
            for mode in modes:
                runs = [run(args, native, actor, case, branch, mode) for _ in range(2)]
                records = [record for record, _ in runs]
                same = records[0] == records[1]
                wanted = 'success' if expected else 'verified_failure'
                passed = same and (records[0]['status'] == wanted if mode == 'reference'
                                   else records[0]['status'] in ('success', 'verified_failure', 'budget_exhausted'))
                if mode == 'reference' and branch == next(iter(definition['branches'])):
                    reference_roots[case] = records[0]['root']
                root_matches = mode != 'policy' or records[0]['root'] == reference_roots.get(case)
                passed = passed and root_matches
                names = []
                for i, (record, tensors) in enumerate(runs):
                    name = f'{case}-{branch}-{mode}-{i}.json'
                    path = args.output / name
                    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
                    names.append(dict(path=name, sha256=sha(path)))
                    archive = path.with_suffix('.npz')
                    np.savez_compressed(archive, **tensors)
                    names.append(dict(path=archive.name, sha256=sha(archive)))
                result = dict(case=case, branch=branch, mode=mode, status=records[0]['status'],
                              error=records[0]['error'], passed=passed, identical_replays=same, evidence=names)
                result['root_matches_reference'] = root_matches
                results.append(result)
                print(json.dumps(result), flush=True)
    after = {path: sha(path) for path in protected}
    summary = dict(schema='exercise-actor-baseline-v1', results=results,
                   passed=bool(results) and all(r['passed'] for r in results) and protected == after,
                   artifacts_before=protected, artifacts_after=after, actor_evaluated=actor is not None,
                   parameters_updated=False, training_ready=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    raise SystemExit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
