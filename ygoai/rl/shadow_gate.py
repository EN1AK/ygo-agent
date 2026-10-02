"""Fail-closed, portable artifact gate for the first bounded shadow pilot."""
import hashlib
import json
from pathlib import Path


def validate_shadow_baseline(manifest_path, checkpoint, requested_steps=None):
    if not manifest_path or not checkpoint:
        raise ValueError('shadow requires a baseline manifest and explicit PPO actor import')
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('gate') != 'bounded-shadow-start-v1' or manifest.get('status') != 'passed':
        raise ValueError('bounded shadow baseline gate is not passed')
    if requested_steps is not None and not 0 < requested_steps <= manifest.get('max_new_steps', 1013760):
        raise ValueError('bounded shadow gate authorizes at most 1,013,760 new steps')
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() not in manifest['allowed_actor_sha256']:
        raise ValueError('actor checkpoint is not an approved baseline/resume')
    if not manifest.get('artifacts') or not manifest.get('strategy_metrics'):
        raise ValueError('baseline evidence or strategy metrics missing')
    if manifest.get('actor_estimator') != 'gae' or manifest.get('promotion_allowed') is not False:
        raise ValueError('manifest must authorize only the bounded GAE-actor shadow gate')
    for item in manifest['artifacts']:
        path = Path(item['path'])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'baseline evidence hash mismatch: {path}')
    return manifest
