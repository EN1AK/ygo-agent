"""Fail-closed, portable artifact gate for the first bounded shadow pilot."""
import hashlib
import json
from pathlib import Path


def validate_shadow_baseline(manifest_path, checkpoint, requested_steps=None, *, frozen_actor=False,
                             resume_checkpoint=None, cumulative_offset=0):
    if not manifest_path or not checkpoint:
        raise ValueError('shadow requires a baseline manifest and explicit PPO actor import')
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    continuation = manifest.get('frozen_critic_continuation')
    if continuation is not None:
        if not frozen_actor or not resume_checkpoint:
            raise ValueError('critic continuation requires frozen actor and Q resume')
        if continuation.get('authorization') != 'user-2m-20261004':
            raise ValueError('critic continuation authorization mismatch')
        if continuation.get('start_step') != cumulative_offset or not 0 <= cumulative_offset < 2000000:
            raise ValueError('critic continuation step mismatch')
        if requested_steps is None or not 0 < requested_steps or cumulative_offset + requested_steps > 2007040:
            raise ValueError('critic continuation exceeds batch-rounded 2M budget')
        if hashlib.sha256(Path(resume_checkpoint).read_bytes()).hexdigest() != continuation.get('q_checkpoint_sha256'):
            raise ValueError('critic continuation checkpoint mismatch')
    elif frozen_actor and requested_steps is not None and requested_steps > 262144:
        raise ValueError('frozen actor pilot cap is 262144 transitions')
    if manifest.get('gate') != 'bounded-shadow-start-v1' or manifest.get('status') != 'passed':
        raise ValueError('bounded shadow baseline gate is not passed')
    if requested_steps is not None and not 0 < requested_steps <= manifest.get('max_new_steps', 1013760):
        raise ValueError('bounded shadow gate authorizes at most 1,013,760 new steps')
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() not in manifest['allowed_actor_sha256']:
        raise ValueError('actor checkpoint is not an approved baseline/resume')
    if not manifest.get('artifacts') or not manifest.get('strategy_metrics'):
        raise ValueError('baseline evidence or strategy metrics missing')
    if manifest.get('actor_estimator') != ('frozen' if frozen_actor else 'gae') or manifest.get('promotion_allowed') is not False:
        raise ValueError('manifest actor update mode does not match the requested shadow gate')
    for item in manifest['artifacts']:
        path = Path(item['path'])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'baseline evidence hash mismatch: {path}')
    return manifest
