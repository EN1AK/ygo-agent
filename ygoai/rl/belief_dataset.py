"""Validated offline self-play supervision; labels are never model input features.

The producer must export only legal-view encodings and public slot descriptors.
This consumer validates shapes, constraints and split isolation; it cannot
certify an arbitrary producer's information boundary. Native collection and
hidden-state application remain independent promotion gates.
"""
import hashlib
import math

import numpy as np


FIELDS={'public_tokens','public_mask','hidden_slot_features','hidden_tokens',
        'slot_mask','remaining_counts','duel_ids'}


def validate_dataset(data):
    if set(data)!=FIELDS:
        raise ValueError(f'Belief dataset fields differ: {set(data)^FIELDS}')
    d={key:np.asarray(value) for key,value in data.items()}
    x,m,s,y,sm,c,ids=(d[k] for k in ('public_tokens','public_mask','hidden_slot_features',
                                    'hidden_tokens','slot_mask','remaining_counts','duel_ids'))
    if x.ndim!=3 or s.ndim!=3 or y.ndim!=2 or c.ndim!=2 or ids.ndim!=1:
        raise ValueError('Invalid belief tensor ranks')
    n,slots=y.shape
    if (n<1 or slots<1 or x.shape[1]<1 or x.shape[0]!=n or s.shape[:2]!=y.shape
            or m.shape!=x.shape[:2] or sm.shape!=y.shape or c.shape[0]!=n or ids.shape!=(n,)):
        raise ValueError('Belief tensor dimensions do not align')
    if m.dtype!=np.bool_ or sm.dtype!=np.bool_ or y.dtype.kind not in 'iu' or c.dtype.kind not in 'iu':
        raise ValueError('Masks must be bool; labels/counts must be integers')
    if x.dtype.kind not in 'fiu' or s.dtype.kind not in 'fiu' or not np.isfinite(x).all() or not np.isfinite(s).all():
        raise ValueError('Nonfinite or nonnumeric legal-view features')
    if ids.dtype.kind not in 'US' or any(not str(v) for v in ids):
        raise ValueError('Explicit stable duel identities are required for group splits')
    if c.shape[1]<2 or (c<0).any() or c[:,0].any():
        raise ValueError('Invalid remaining counts; card 0 is padding')
    if (y<0).any() or (y>=c.shape[1]).any() or (y[sm]==0).any() or (y[~sm]!=0).any():
        raise ValueError('Invalid supervised hidden identities or padding')
    if (sm[:,1:] & ~sm[:,:-1]).any() or not sm.any(axis=1).all():
        raise ValueError('Hidden slots must be a nonempty prefix followed by padding')
    if x[~m].any() or s[~sm].any():
        raise ValueError('Padding features must be zero, not privileged residual data')
    for row in range(n):
        used=np.bincount(y[row,sm[row]],minlength=c.shape[1])
        if (used>c[row]).any():
            raise ValueError('Labels violate remaining-card constraints')
    return d


def split_by_duel(data, *, holdout_fraction=.2, seed=0):
    data=validate_dataset(data)
    if not 0<holdout_fraction<1:
        raise ValueError('Holdout fraction must be between zero and one')
    ids=sorted(set(data['duel_ids'].tolist()))
    if len(ids)<2:
        raise ValueError('At least two independent duels are required')
    ids.sort(key=lambda identity:hashlib.sha256(f'{seed}:{identity}'.encode()).digest())
    held=set(ids[:max(1,min(len(ids)-1,math.ceil(len(ids)*holdout_fraction)))])
    mask=np.array([identity in held for identity in data['duel_ids']])
    return ({k:v[~mask] for k,v in data.items()}, {k:v[mask] for k,v in data.items()})


def calibration_metrics(logits, targets, mask, *, bins=10):
    scores=np.asarray(logits,dtype=np.float64)[mask]
    target=np.asarray(targets)[mask]
    if not len(target) or not np.isfinite(scores).all():
        raise ValueError('Calibration requires finite held-out predictions')
    scores-=scores.max(axis=-1,keepdims=True)
    p=np.exp(scores)
    p/=p.sum(axis=-1,keepdims=True)
    correct=p.argmax(axis=-1)==target
    confidence=p.max(axis=-1)
    buckets=np.minimum((confidence*bins).astype(int),bins-1)
    ece=0.
    rows=[]
    for bucket in range(bins):
        selected=buckets==bucket
        if selected.any():
            accuracy=float(correct[selected].mean())
            conf=float(confidence[selected].mean())
            ece+=selected.mean()*abs(accuracy-conf)
            rows.append({'bin':bucket,'count':int(selected.sum()),'accuracy':accuracy,'confidence':conf})
    return {'tokens':len(target),'nll':float(-np.log(np.maximum(p[np.arange(len(target)),target],1e-300)).mean()),
            'top1_accuracy':float(correct.mean()),'ece':float(ece),'bins':rows,
            'conditioning':'teacher-forced-prefix; not joint-particle calibration'}
