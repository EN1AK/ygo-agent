"""Exact weighted complete-history optimization with bounded activation memory."""
import numpy as np
from ygoai.rl.scene_batch import pack_scenes


def make_optimizer(actor, parent, rows, steps, learning_rate, ce_coeff, kl_coeff, *, check_pairs=None, qualification_hard_targets=False):
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
    selected=list(check_pairs) if check_pairs is not None else [next(i for i in range(n//2) if np.any(np.asarray(cc[i])>0)),
              next(i for i in range(n//2) if np.any(np.asarray(kc[i])>0))]
    assert len(selected)==2 and len(set(selected))==2 and all(0<=i<n//2 for i in selected)
    # Qualification uses the same CE/KL function and hard real examples in all
    # arms. A nearly-zero already-fitted control gradient is a poor denominator
    # for relative-error tests. Arm coefficients below remain completely intact.
    check_cc=jnp.asarray([.5,.5,0.,0.]);check_kc=jnp.asarray([0.,0.,.5,.5])
    def merge(value,weight=False):
        if weight:return value[jnp.asarray(selected)].swapaxes(0,1).reshape(steps,4)
        v=value[jnp.asarray(selected)]
        return v.reshape((2,steps,width)+v.shape[2:]).swapaxes(0,1).reshape((steps*4,)+v.shape[2:])
    four_batch=jax.tree.map(merge,batch);four_anchor=sequence(parent,four_batch)
    four_targets=merge(targets)
    if qualification_hard_targets:
        # An already fitted teacher can have exactly zero CE gradient. Use a
        # rotated native-legal index only in this no-update arithmetic check.
        # Actual training targets, masks, coefficients and losses stay intact.
        four_targets=(four_targets+1)%merge(legal).sum(-1)
    leaves,tree=jax.tree.flatten(parent);leaves[0]=leaves[0]+jnp.asarray(1e-4,leaves[0].dtype);candidate=jax.tree.unflatten(tree,leaves)
    (full_value,_),full_grad=value_grad(candidate,four_batch,four_targets,merge(weights,True),merge(legal),four_anchor,check_cc,check_kc)
    total=jax.tree.map(jnp.zeros_like,parent);values=[]
    for j,i in enumerate(selected):
        numerical_targets=(targets[i]+1)%legal[i].sum(-1) if qualification_hard_targets else targets[i]
        (value,_),grad=value_grad(candidate,{k:v[i] for k,v in batch.items()},numerical_targets,weights[i],legal[i],anchor[i],check_cc[j*2:j*2+2],check_kc[j*2:j*2+2])
        total=jax.tree.map(lambda x,y:x+y,total,grad);values.append(float(value))
    relative=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,full_grad,total))/jnp.maximum(optax.global_norm(full_grad),1e-8))
    value_error=abs(float(full_value)-sum(values))
    value_scale=max(1.,abs(float(full_value)),abs(sum(values)))
    normalized_value_error=value_error/value_scale
    # Float32 has absolute spacing >1e-5 for losses above 128. Bound the
    # reduction error in float32 units, retaining raw and normalized evidence.
    value_tolerance=8*np.finfo(np.float32).eps
    assert relative<.002 and normalized_value_error<=value_tolerance,(relative,value_error,normalized_value_error)
    admission=dict(passed=True,histories=n,micro_batch_histories=2,max_logit_error=max(errors),gradient_relative_error=relative,gradient_objective_error=value_error,
                   gradient_objective_normalized_error=normalized_value_error,gradient_objective_scale=value_scale,objective_reduction_relative_tolerance=float(value_tolerance),
                   gradient_check_pairs=selected,gradient_check_ce_coeff=np.asarray(check_cc).tolist(),gradient_check_kl_coeff=np.asarray(check_kc).tolist(),
                   gradient_reference_l2=float(optax.global_norm(full_grad)),gradient_absolute_error_l2=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,full_grad,total))),
                   gradient_check_scope=('Native-legal no-update numerical CE and retention KL qualification; actual teaching objective independently reconstructed each update' if qualification_hard_targets else 'Fixed equal-weight real correction CE and retention KL qualification; actual arm objective independently reconstructed each update'),
                   qualification_hard_targets=qualification_hard_targets,
                   qualification_target_scope='Rotated native-legal targets only in the no-update arithmetic check; never supervision' if qualification_hard_targets else 'Original demonstration targets',
                   ce_coeff=list(ce_coeff),kl_coeff=list(kl_coeff))
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
