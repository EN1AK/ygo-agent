"""Explicit v1 -> v2 copy with zero residuals; no optimizer or training run."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path

try:
    import _repo_bootstrap  # noqa: F401
except ModuleNotFoundError:
    import scripts._repo_bootstrap  # noqa: F401
import flax
import jax
import jax.numpy as jnp
import numpy as np
from scripts.migrate_structured_checkpoint import migrate
from ygoai.rl.checkpoint_compat import load_checkpoint_metadata, sha256_file, write_checkpoint_metadata
from ygoai.rl.jax.agent import ModelArgs, RNNAgent
from ygoai.rl.observation_schema import tensor_contract


def observation(schema, capacities, batch=1):
    return {k: jnp.zeros((batch,) + v.shape, v.dtype)
            for k, v in tensor_contract(schema, **capacities).items()}


def migrate_model(source, old_args, embeddings, capacities):
    if old_args.observation_schema != 'structured-lite-v1' or old_args.structured_variant != 'full':
        raise ValueError('migration requires full structured-lite-v1 source')
    new_args = ModelArgs(**(asdict(old_args) | {'observation_schema': 'structured-state-v2'}))
    old = RNNAgent(embedding_shape=embeddings, **asdict(old_args))
    new = RNNAgent(embedding_shape=embeddings, **asdict(new_args))
    obs = observation('structured-state-v2', capacities)
    obs['actions_'] = obs['actions_'].at[:, :2, 3].set(1)
    obs['action_features_'] = obs['action_features_'].at[:, :2, 0].set(1)
    state = old.init_rnn_state(1)
    template = new.init(jax.random.PRNGKey(91), obs, state)
    migrated, report = migrate(source, template)
    if report['shape_incompatible'] or report['intentionally_excluded']:
        raise ValueError('migration would change or drop an existing parameter')
    for nonempty in (False, True):
        fixture = dict(obs)
        if nonempty:
            fixture['action_effect_semantics_'] = fixture['action_effect_semantics_'].at[:, 0, :].set(1)
            fixture['public_chain_'] = fixture['public_chain_'].at[:, 0, :].set(jnp.array([1,1,1,1,13,1,1], dtype=jnp.uint16))
            fixture['public_turn_effects_'] = fixture['public_turn_effects_'].at[:, 0, :].set(jnp.array([1,1,1,1,3,2,1,0,1], dtype=jnp.uint16))
        left = old.apply(source, fixture, state)
        right = new.apply(migrated, fixture, state)
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
            np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
            assert np.isfinite(np.asarray(b)).all()
    report['zero_residual_parity'] = True
    return migrated, new_args, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--semantic-metadata', type=Path, required=True)
    p.add_argument('--source-semantic-metadata', type=Path, required=True)
    p.add_argument('--code-list', type=Path, required=True)
    p.add_argument('--num-embeddings', type=int, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    metadata = load_checkpoint_metadata(a.source)
    if metadata['code_list_hash'] != sha256_file(a.code_list):
        raise ValueError('source code-list mismatch')
    sem = json.loads(a.semantic_metadata.read_text())
    if sha256_file(a.source_semantic_metadata) != metadata['semantic_table_hash']:
        raise ValueError('source semantic metadata does not match checkpoint')
    old_sem = json.loads(a.source_semantic_metadata.read_text())
    if any(old_sem['table_hashes'][k] != sem['table_hashes'][k] for k in ('static', 'tags', 'confidence')):
        raise ValueError('v2 migration must preserve all existing semantic table bytes')
    if sem.get('effect_descriptions', {}).get('version') != 'effect-description-declarations-v1':
        raise ValueError('missing versioned effect-description table')
    for key, filename in {'static':'card-semantics.u8', 'tags':'effect-tags.u8',
                          'confidence':'effect-tag-confidence.u8', 'effect_descriptions':'effect-descriptions.u8'}.items():
        if sha256_file(a.semantic_metadata.parent / filename) != sem['table_hashes'][key]:
            raise ValueError('semantic table hash mismatch: ' + key)
    if sem['source_hashes']['code_list_sha256'] != sha256_file(a.code_list):
        raise ValueError('semantic code-list mismatch')
    old_args = ModelArgs(**metadata['model_architecture'])
    old = RNNAgent(embedding_shape=a.num_embeddings, **asdict(old_args))
    obs = observation('structured-lite-v1', metadata['capacities'])
    template = old.init(jax.random.PRNGKey(0), obs, old.init_rnn_state(1))
    source = flax.serialization.from_bytes(template, a.source.read_bytes())
    migrated, new_args, report = migrate_model(source, old_args, a.num_embeddings, metadata['capacities'])
    a.output_dir.mkdir(parents=True, exist_ok=False)
    output = a.output_dir / 'public-state-v2.flax_model'
    output.write_bytes(flax.serialization.to_bytes(migrated))
    write_checkpoint_metadata(output, observation_schema='structured-state-v2', model_args=new_args,
                              semantic_table_hash=sha256_file(a.semantic_metadata),
                              code_list_hash=sha256_file(a.code_list), capacities=metadata['capacities'],
                              migrated_from=metadata['checkpoint_sha256'],
                              training_context={'promotion_eligible': False, 'migration': 'zero-residual-v1'})
    report.update(source_sha256=metadata['checkpoint_sha256'], output_sha256=sha256_file(output))
    (a.output_dir / 'migration-report.json').write_text(json.dumps(report, indent=2) + '\n')
    if sha256_file(a.source) != metadata['checkpoint_sha256']:
        raise ValueError('source changed during migration')
    print(json.dumps({'output': str(output), 'zero_residual_parity': True}))


if __name__ == '__main__':
    main()
