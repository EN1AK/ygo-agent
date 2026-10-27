"""Complete-history recurrent curriculum, with gradient masks on suffixes."""
import argparse,time,copy
import optax
from learning_common import *
from ygoai.rl.scene_batch import pack_scenes
from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args()
position=next(i for i,s in enumerate(plan['stages']) if s['name']==a.stage);stage=plan['stages'][position]
folder=run/a.stage;folder.mkdir(exist_ok=False)
parent=root/plan['parent'] if position==0 else run/plan['stages'][position-1]['name']/'model.flax_model'
if position==0:assert sha(parent)==plan['parent_sha256']
params=load_params(parent);initial=params
assert any(d.platform=='gpu' for d in jax.devices())
jax.config.update('jax_default_matmul_precision','highest')
def batch(split,suffix):
    rows=[]
    for item in plan['selected']:
        if item['split']!=split:continue
        r=copy.deepcopy(records[item['sha']]);own=[i for i,d in enumerate(r['decisions']) if d['player']==0];active=set(own[-suffix:] if suffix else own)
        for i,d in enumerate(r['decisions']):d['teaching']=i in active and len(d['menu'])>1
        rows.append((r,observations[item['sha']]))
    return tuple(jax.tree.map(jnp.asarray,x) for x in pack_scenes(rows,plan['sequence_steps']))
fit=batch('fit',stage['own_suffix']);all_fit=batch('fit',None);held=batch('holdout',None)
@jax.jit
def objective(params,obs,targets,weights,legal):
    logits=agent.apply(params,obs,agent.init_rnn_state(4),jnp.zeros(plan['sequence_steps']*4,bool),None)[1]
    ce=optax.softmax_cross_entropy_with_integer_labels(logits,targets).reshape(-1,4)
    loss=(weights*ce).sum(0);acc=(weights*(jnp.argmax(logits,axis=-1)==targets).reshape(-1,4)).sum(0)
    return loss.mean(),(loss,acc)
def metrics(p):
    out={}
    for name,b in [('fit_full',all_fit),('holdout_full',held),('fit_curriculum',fit)]:
        loss,(ce,acc)=objective(p,*b);out[name]=dict(loss=float(loss),accuracy=float(jnp.mean(acc)),per_history_accuracy=np.asarray(acc).tolist())
    return out
before=metrics(params);write(folder/'before.json',before)
tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(plan['learning_rate']));optim=tx.init(params)
if position:
    optim=flax.serialization.from_bytes(optim,(run/plan['stages'][position-1]['name']/'optimizer.msgpack').read_bytes())
@jax.jit
def update(params,state,batch):
    (loss,aux),gradient=jax.value_and_grad(objective,has_aux=True)(params,*batch)
    delta,state=tx.update(gradient,state,params);params=optax.apply_updates(params,delta)
    finite=jnp.all(jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(params)]))
    return params,state,loss,aux,optax.global_norm(gradient),finite
write(folder/'manifest.json',dict(plan_sha256=sha(root/'learning-plan-v1.json'),stage=stage,parent_sha256=sha(parent),native_sha256=sha(args.native),source_sha256=sha(__file__),devices=[str(d) for d in jax.devices()],loss_scope='controlled-seat nontrivial actions in suffix; all preceding controlled-seat observations update recurrent state',holdout_used_for_gradient=False))
started=time.monotonic()
with (folder/'metrics.jsonl').open('w') as f:
    for step in range(1,stage['updates']+1):
        params,optim,loss,aux,norm,finite=update(params,optim,fit)
        entry=dict(update=step,loss=float(loss),per_history_ce=np.asarray(aux[0]).tolist(),per_history_accuracy=np.asarray(aux[1]).tolist(),grad_norm=float(norm),finite=bool(finite),seconds=time.monotonic()-started)
        assert entry['finite'] and np.isfinite([entry['loss'],entry['grad_norm']]).all()
        f.write(json.dumps(entry)+'\n');f.flush()
        if step%64==0:print(json.dumps(entry),flush=True)
checkpoint=folder/'model.flax_model';checkpoint.write_bytes(flax.serialization.to_bytes(params));(folder/'optimizer.msgpack').write_bytes(flax.serialization.to_bytes(optim))
write_checkpoint_metadata(checkpoint,observation_schema='structured-lite-v1',model_args=modelargs,code_list_hash=sha(args.code_list),semantic_table_hash=sha(args.semantics/'metadata.json'),capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),training_context=dict(experiment=plan['id'],stage=a.stage,parent_sha256=sha(parent),updates=stage['updates'],full_legal_menu=True,promotion_eligible=False))
after=metrics(params);write(folder/'after.json',after)
reloaded=load_params(checkpoint)
delta=float(sum(np.sum((np.asarray(x).astype(np.float64)-np.asarray(y).astype(np.float64))**2) for x,y in zip(jax.tree.leaves(reloaded),jax.tree.leaves(initial)))**.5)
assert delta>0
write(folder/'summary.json',dict(completed=True,updates=stage['updates'],parent_sha256=sha(parent),checkpoint_sha256=sha(checkpoint),parameter_delta_l2=delta,before=before,after=after,autonomous_evaluation='separate process'))
print(json.dumps(dict(stage=a.stage,completed=True,after=after,parameter_delta_l2=delta)),flush=True)
