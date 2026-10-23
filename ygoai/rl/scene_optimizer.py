"""Exact weighted complete-history optimization with bounded activation memory."""
import numpy as np
from ygoai.rl.scene_batch import pack_scenes


def make_optimizer(actor, parent, rows, steps, learning_rate, ce_coeff, kl_coeff):
    import jax
    import jax.numpy as jnp
    import optax
    from ygoai.rl.exercise_retention import parent_kl_rows
    n=len(rows);width=2
    assert n%width==0 and len(ce_coeff)==len(kl_coeff)==n
    tensors,targets,weights,legal=pack_scenes(rows,steps)
    def split(value):
        return value.reshape((steps,n//width,width)+value.shape[1:]).swapaxes(0,1).reshape((n//width,steps*width)+value.shape[1:])
    batch=jax.tree.map(lambda v:jnp.asarray(split(v)),tensors)
    targets=jnp.asarray(split(targets));weights=jnp.asarray(split(weights.reshape(-1))).reshape(n//width,steps,width)
    legal=jnp.asarray(split(legal));cc=jnp.asarray(ce_coeff,dtype=jnp.float32).reshape(-1,width);kc=jnp.asarray(kl_coeff,dtype=jnp.float32).reshape(-1,width)
    @jax.jit
    def sequence(params,obs):
        count=obs['actions_'].shape[0]//steps
        return actor.agent.apply(params,obs,actor.agent.init_rnn_state(count),jnp.zeros(steps*count,bool),None)[1]
    anchor=jnp.stack([sequence(parent,{k:v[i] for k,v in batch.items()}) for i in range(n//width)])
    errors=[];actor.params=parent
    for b,(record,obs) in enumerate(rows):
        actor.reset();own=0
        for t,d in enumerate(record['decisions']):
            logits,_,_=actor.predict({k:v[t:t+1] for k,v in obs.items()},d['player'])
            if d['player']==record['definition']['controlled']:
                count=len(d['menu']);errors.append(float(np.max(np.abs(logits[:count]-np.asarray(anchor)[b//width,width*own+b%width,:count]))));own+=1
    assert max(errors)<.002,max(errors)
    def objective(params,obs,targets,weights,legal,anchor,cc,kc):
        logits=sequence(params,obs);count=logits.shape[0]//steps
        ces=(weights*optax.softmax_cross_entropy_with_integer_labels(logits,targets).reshape(-1,count)).sum(0)
        kls=(weights*parent_kl_rows(logits,anchor,legal).reshape(-1,count)).sum(0)
        return (cc*ces+kc*kls).sum(),(ces,kls)
    value_grad=jax.jit(jax.value_and_grad(objective,has_aux=True))
    # Include a CE-only and a KL-only pair in a real-model accumulation check.
    selected=[next(i for i in range(n//2) if np.any(np.asarray(cc[i])>0)),
              next(i for i in range(n//2) if np.any(np.asarray(kc[i])>0))]
    def merge(value,weight=False):
        if weight:return value[jnp.asarray(selected)].swapaxes(0,1).reshape(steps,4)
        v=value[jnp.asarray(selected)]
        return v.reshape((2,steps,width)+v.shape[2:]).swapaxes(0,1).reshape((steps*4,)+v.shape[2:])
    four_batch=jax.tree.map(merge,batch);four_anchor=sequence(parent,four_batch)
    leaves,tree=jax.tree.flatten(parent);leaves[0]=leaves[0]+jnp.asarray(1e-4,leaves[0].dtype);candidate=jax.tree.unflatten(tree,leaves)
    (full_value,_),full_grad=value_grad(candidate,four_batch,merge(targets),merge(weights,True),merge(legal),four_anchor,cc[jnp.asarray(selected)].reshape(-1),kc[jnp.asarray(selected)].reshape(-1))
    total=jax.tree.map(jnp.zeros_like,parent);values=[]
    for i in selected:
        (value,_),grad=value_grad(candidate,{k:v[i] for k,v in batch.items()},targets[i],weights[i],legal[i],anchor[i],cc[i],kc[i])
        total=jax.tree.map(lambda x,y:x+y,total,grad);values.append(float(value))
    relative=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,full_grad,total))/jnp.maximum(optax.global_norm(full_grad),1e-8))
    value_error=abs(float(full_value)-sum(values));assert relative<.002 and value_error<1e-5,(relative,value_error)
    admission=dict(passed=True,histories=n,micro_batch_histories=2,max_logit_error=max(errors),gradient_relative_error=relative,gradient_objective_error=value_error,ce_coeff=list(ce_coeff),kl_coeff=list(kl_coeff))
    del full_grad,total,four_batch,four_anchor,candidate
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(learning_rate))
    @jax.jit
    def update(params,state,batch,targets,weights,legal,anchor,cc,kc):
        def accumulate(total,data):
            (value,terms),grad=jax.value_and_grad(objective,has_aux=True)(params,*data)
            return jax.tree.map(lambda x,y:x+y,total,grad),(value,terms)
        grad,(values,terms)=jax.lax.scan(accumulate,jax.tree.map(jnp.zeros_like,params),(batch,targets,weights,legal,anchor,cc,kc))
        changes,state=tx.update(grad,state,params);params=optax.apply_updates(params,changes)
        finite=jnp.all(jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(params)]))
        return params,state,values.sum(),tuple(x.reshape(-1) for x in terms),optax.global_norm(grad),finite
    def step(params,state):return update(params,state,batch,targets,weights,legal,anchor,cc,kc)
    return step,tx.init(parent),admission
