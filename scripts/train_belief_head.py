"""Offline belief-head pilot; separate params, optimizer and data from PPO.

Consumes a validated NPZ with the fields in ygoai.rl.belief_dataset.FIELDS.
Requires a producer manifest declaring a legal-view, known-deck self-play
fixture. That declaration is recorded, not treated as a passed leakage gate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import _repo_bootstrap  # noqa: F401
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ygoai.rl.belief_dataset import validate_dataset, split_by_duel, calibration_metrics
from ygoai.rl.jax.belief_head import BeliefHead, belief_nll, sample_belief


def inputs(data):
    return tuple(jnp.asarray(data[k]) for k in ('public_tokens','public_mask','hidden_slot_features',
                                              'hidden_tokens','remaining_counts'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--producer-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=1000)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--width',type=int,default=64)
    parser.add_argument('--heads',type=int,default=4)
    parser.add_argument('--layers',type=int,default=2)
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--seed',type=int,default=1)
    args=parser.parse_args()
    if args.updates<1 or args.batch_size<1 or args.heads<1 or args.width<1 or args.layers<1 or args.width%args.heads:
        parser.error('Positive update/batch/width/head counts and divisible width required')
    if not np.isfinite(args.learning_rate) or args.learning_rate<=0:
        parser.error('Positive finite learning rate required')
    producer=json.loads(args.producer_manifest.read_text())
    data_hash=hashlib.sha256(args.data.read_bytes()).hexdigest()
    if (producer.get('schema')!='ygo-belief-dataset-v1' or producer.get('data_sha256')!=data_hash
            or producer.get('information_assumption')!='known-deck-self-play'
            or producer.get('input_semantics')!='acting-player-legal-view'
            or not producer.get('runtime_manifest_sha256')):
        raise ValueError('Missing or incompatible belief data provenance')
    with np.load(args.data,allow_pickle=False) as archive:
        data=validate_dataset(dict(archive))
    train,held=split_by_duel(data,seed=args.seed)
    model=BeliefHead(data['remaining_counts'].shape[1],width=args.width,heads=args.heads,layers=args.layers)
    rng=jax.random.PRNGKey(args.seed)
    rng,init_key=jax.random.split(rng)
    variables=model.init(init_key,*inputs({k:v[:1] for k,v in train.items()}))
    optimizer=optax.adam(args.learning_rate)
    state=optimizer.init(variables)

    @jax.jit
    def update(variables,state,batch,key):
        def objective(params):
            logits,valid=model.apply(params,*batch[:-1],deterministic=False,rngs={'dropout':key})
            return belief_nll(logits,valid,batch[3],batch[-1])
        (loss,invalid),grads=jax.value_and_grad(objective,has_aux=True)(variables)
        delta,next_state=optimizer.update(grads,state,variables)
        candidate=optax.apply_updates(variables,delta)
        finite=jnp.isfinite(loss)&(invalid==0)
        for leaf in jax.tree_util.tree_leaves((grads,candidate,next_state)):
            finite=finite & jnp.all(jnp.isfinite(leaf))
        return candidate,next_state,loss,invalid,finite

    args.output.mkdir(parents=True,exist_ok=False)
    selector=np.random.default_rng(args.seed)
    with (args.output/'train.jsonl').open('w') as stream:
        for step in range(args.updates):
            index=selector.integers(0,len(train['duel_ids']),size=args.batch_size)
            batch={k:v[index] for k,v in train.items()}
            rng,key=jax.random.split(rng)
            variables,state,loss,invalid,finite=update(variables,state,
                inputs(batch)+(jnp.asarray(batch['slot_mask']),),key)
            if not bool(finite):
                raise RuntimeError(f'Invalid belief update {step}: loss={loss}, invalid_targets={invalid}')
            if step%10==0 or step==args.updates-1:
                row={'update':step+1,'loss':float(loss),'invalid_targets':int(invalid),'finite':True}
                stream.write(json.dumps(row)+'\n'); stream.flush()
                print(json.dumps(row),flush=True)
    # Bounded batches avoid turning an offline held-out set into a GPU OOM.
    outputs=[]
    valid_particles=0
    for offset in range(0,len(held['duel_ids']),args.batch_size):
        chunk={k:v[offset:offset+args.batch_size] for k,v in held.items()}
        logits,valid=model.apply(variables,*inputs(chunk))
        _,invalid=belief_nll(logits,valid,jnp.asarray(chunk['hidden_tokens']),jnp.asarray(chunk['slot_mask']))
        if int(invalid): raise RuntimeError('Invalid held-out target')
        outputs.append(np.asarray(logits))
        rng,key=jax.random.split(rng)
        tokens,ok=sample_belief(model,variables,*inputs(chunk)[:3],jnp.asarray(chunk['remaining_counts']),
                               key,slot_mask=jnp.asarray(chunk['slot_mask']))
        valid_particles+=int(np.asarray(ok).sum())
    report=calibration_metrics(np.concatenate(outputs),held['hidden_tokens'],held['slot_mask'])
    checkpoint=args.output/'belief.flax_model'
    checkpoint.write_bytes(flax.serialization.to_bytes(variables))
    report.update(schema='ygo-belief-pilot-v1',data_sha256=data_hash,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),producer=producer,
        model={'vocabulary_size':model.vocabulary_size,'width':args.width,'heads':args.heads,'layers':args.layers,
               'max_hidden_slots':data['hidden_tokens'].shape[1]},
        updates=args.updates,seed=args.seed,heldout_duels=sorted(set(held['duel_ids'].tolist())),
        train_duels=sorted(set(train['duel_ids'].tolist())),heldout_decks_validated=False,
        constraint_valid_particle_rate=valid_particles/len(held['duel_ids']),
        constraint_scope='remaining-card counts and padding only; native/public-slot application unvalidated',
        fair_play_promoted=False,limits=['producer leakage gate not certified by this loader',
            'native particle application not validated','known-deck self-play only'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'completed.txt').write_text('offline belief pilot completed; no fair-play promotion\n')


if __name__=='__main__': main()
