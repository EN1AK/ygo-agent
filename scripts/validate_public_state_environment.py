"""Bounded paired native check: identical menus/old tensors, public-only new state.

Run on an isolated compiled runtime. No policy updates or production installs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import _repo_bootstrap  # noqa: F401
import numpy as np
import ygoenv
from ygoai.rl.env import VersionedObservation
from ygoai.rl.effect_semantics import validate_assets
from ygoai.rl.observation_schema import tensor_contract
from ygoai.utils import init_ygopro


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deck', type=Path, required=True)
    p.add_argument('--code-list', type=Path, required=True)
    p.add_argument('--semantic-assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1092026)
    p.add_argument('--games', type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.games <= 16:
        raise ValueError('bounded validation allows 1..16 games')
    sem = validate_assets(a.semantic_assets)
    if hashlib.sha256(a.code_list.read_bytes()).hexdigest() != sem['source_hashes']['code_list_sha256']:
        raise ValueError('semantic code-list mismatch')
    deck = init_ygopro('YGOPro-v1', 'chinese', str(a.deck.resolve()), str(a.code_list.resolve()))
    common = tuple(tensor_contract('structured-lite-v1'))
    summary = []
    for game in range(a.games):
        envs = []
        try:
            for schema in ('structured-lite-v1', 'structured-state-v2'):
                env = ygoenv.make('YGOPro-v1', env_type='gymnasium', num_envs=1, num_threads=1,
                                 seed=a.seed+game, deck1=deck, deck2=deck, player=0, play_mode='self',
                                 max_options=192, max_steps=1000, greedy_reward=False,
                                 n_history_actions=32, observation_schema=schema,
                                 semantic_asset_dir=str(a.semantic_assets.resolve()))
                env.num_envs = 1
                envs.append(VersionedObservation(env, schema))
            left, right = [e.reset()[0] for e in envs]
            rng = np.random.default_rng(a.seed+game)
            chain_rows = usage_rows = bound_actions = steps = 0
            completed = False
            for step in range(1000):
                for key in common:
                    np.testing.assert_array_equal(left[key], right[key], err_msg=key)
                for key, spec in tensor_contract('structured-state-v2', max_options=192).items():
                    assert right[key].shape[1:] == spec.shape and str(right[key].dtype) == spec.dtype, key
                chain = np.asarray(right['public_chain_'])[0]
                usage = np.asarray(right['public_turn_effects_'])[0]
                # Every exposed ID must have occurred in an earlier public activation;
                # the ledger cannot discover new cards by inspecting a hidden zone.
                events = np.asarray(right['public_events_'])[0]
                ids = events[:, 10].astype(np.int32) * 256 + events[:, 11]
                if step == 0:
                    disclosed = set()
                disclosed.update(int(i) for i in ids[(events[:, 0] == 1) & (ids != 0)])
                assert set(map(int, chain[chain[:,0]>0,1])) <= disclosed
                assert set(map(int, usage[usage[:,0]>0,1])) <= disclosed
                chain_rows += int((chain[:,0]>0).sum())
                usage_rows += int((usage[:,0]>0).sum())
                bound_actions += int(np.asarray(right['action_effect_semantics_'])[0,:,0].sum())
                count = int((np.asarray(left['actions_'])[0,:,3] != 0).sum())
                action = np.array([rng.integers(max(count,1))], dtype=np.int32)
                l, r = [e.step(action) for e in envs]
                for i in (1,2,3):
                    np.testing.assert_array_equal(l[i],r[i])
                for key in ('num_options', 'to_play', 'invalid_game'):
                    if key in l[4] or key in r[4]:
                        np.testing.assert_array_equal(l[4][key], r[4][key], err_msg=key)
                left, right = l[0], r[0]
                steps = step+1
                if bool(np.asarray(l[2] | l[3])[0]):
                    completed = True
                    break
            summary.append(dict(game=game, seed=a.seed+game, steps=steps,
                                completed=completed, chain_rows=chain_rows, usage_rows=usage_rows, bound_actions=bound_actions,
                                invalid_game=int(np.asarray(l[4].get('invalid_game', [0]))[0])))
        finally:
            for e in envs:
                e.close()
    if not all(sum(g[k] for g in summary) > 0 for k in ('chain_rows','usage_rows','bound_actions')):
        raise ValueError('validation did not exercise each new input')
    if any(g['invalid_game'] for g in summary):
        raise ValueError('paired validation encountered an invalid duel')
    if not all(g['completed'] for g in summary):
        raise ValueError('paired validation reached its bound before termination')
    result = dict(schema='public-state-native-validation-v1', old_tensor_parity=True,
                  games=summary, semantic_metadata_sha256=hashlib.sha256((a.semantic_assets/'metadata.json').read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
