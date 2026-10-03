"""Actor-visible decision fixtures; no engine/private-state snapshot is implied."""
import hashlib
import json
from pathlib import Path

import numpy as np


def save_decision_fixture(directory, *, step, observation, recurrent_leaves,
                          dones, logits, probabilities, value, selected_action,
                          num_options, metadata):
    if not 0 <= selected_action < num_options:
        raise ValueError('selected action outside recorded menu')
    arrays = {f'obs__{key}': np.asarray(item).copy() for key, item in observation.items()}
    arrays.update({f'rnn__{i}': np.asarray(item).copy() for i, item in enumerate(recurrent_leaves)})
    arrays.update(dones=np.asarray(dones).copy(), logits=np.asarray(logits).copy(),
                  probabilities=np.asarray(probabilities).copy(), value=np.asarray(value).copy())
    if any(x.dtype.hasobject or not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError('nonfinite or object array in decision fixture')
    if arrays['obs__actions_'].shape[0] != 1:
        raise ValueError('decision fixtures require a single environment')
    path = Path(directory) / f'step-{step:06d}.npz'
    # Exclusive create prevents silently changing the meaning of an old fixture.
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    record = dict(metadata, schema='actor-visible-decision-v1', step=step,
                  selected_action=selected_action, num_options=num_options,
                  recurrent_leaf_count=len(recurrent_leaves),
                  observation_keys=sorted(observation),
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  engine_snapshot=False, counterfactual_labels_validated=False)
    with path.with_suffix('.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    return record
