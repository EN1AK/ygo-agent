"""Complete controlled-seat scene sequences with independent teaching masks."""
import numpy as np


def pack_scenes(rows, steps=64):
    observations, targets, weights, legal = [], [], [], []
    for record, obs in rows:
        indices = [i for i,d in enumerate(record['decisions']) if d['player'] == record['definition']['controlled']]
        if not 0 < len(indices) <= steps:
            raise ValueError('complete scene exceeds sequence budget')
        selected = [record['decisions'][i] for i in indices]
        capacity = obs['actions_'].shape[1]
        support = np.zeros((steps, capacity), bool)
        for t,d in enumerate(selected):
            if not 0 <= d['action'] < len(d['menu']) <= capacity:
                raise ValueError('demonstration outside native legal menu')
            support[t,:len(d['menu'])] = True
        support[len(indices):,0] = True
        padded = indices + [indices[-1]]*(steps-len(indices))
        observations.append({k:v[padded] for k,v in obs.items()})
        targets.append([d['action'] for d in selected]+[0]*(steps-len(indices)))
        weights.append([1/len(indices)]*len(indices)+[0.]*(steps-len(indices)))
        legal.append(support)
    tensors = {k:np.stack([o[k] for o in observations],axis=1).reshape((-1,)+observations[0][k].shape[1:]) for k in observations[0]}
    return (tensors, np.asarray(targets,np.int32).T.reshape(-1),
            np.asarray(weights,np.float32).T, np.stack(legal,axis=1).reshape(-1,legal[0].shape[-1]))
