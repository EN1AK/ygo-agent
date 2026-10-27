"""Rehearse full demonstrations while teaching certified off-reference suffixes."""
import copy,time,optax
from learning_common import *
from ygoai.rl.scene_batch import pack_scenes
from ygoai.rl.checkpoint_compat import write_checkpoint_metadata
ap=read(root/'recovery-learning-plan-v1.json');assert sha(root/'learning-plan-v1.json')==ap['base_plan_sha256']
assert sha(root/'recovery-search-v2/independent-certification.json')==ap['certification_sha256']
parent=root/ap['parent'];full=read(run/'full/summary.json');assert full['completed'] and sha(parent)==full['checkpoint_sha256']
assert (root/'learning-launch.exit').read_text().strip()=='0'
folder=run/'recovery-augmented';folder.mkdir(exist_ok=False)
params=load_params(parent);initial=params
assert any(d.platform=='gpu' for d in jax.devices())
jax.config.update('jax_default_matmul_precision','highest')
rows=[]
for item in plan['selected']:
    if item['split']!='fit':continue
    r=copy.deepcopy(records[item['sha']])
    for d in r['decisions']:d['teaching']=d['player']==0 and len(d['menu'])>1
    rows.append((r,observations[item['sha']]))
n_original=len(rows);assert n_original==4
for item in ap['rows']:
    path=root/item['path'];assert sha(path)==item['record_sha256'] and sha(path.with_suffix('.npz'))==item['observations_sha256']
    r=read(path);assert r['training_ready'] and r['source_sha256'] not in {x['sha'] for x in plan['selected'] if x['split']=='holdout'}
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as f:obs=dict(f)
    rows.append((r,obs))
batch=tuple(jax.tree.map(jnp.asarray,x) for x in pack_scenes(rows,ap['sequence_steps']));B=len(rows)
@jax.jit
def objective(p,obs,targets,weights,legal):
    logits=agent.apply(p,obs,agent.init_rnn_state(B),jnp.zeros(ap['sequence_steps']*B,bool),None)[1]
    ce=optax.softmax_cross_entropy_with_integer_labels(logits,targets).reshape(-1,B)
    loss=(weights*ce).sum(0);acc=(weights*(jnp.argmax(logits,-1)==targets).reshape(-1,B)).sum(0)
    return .5*(loss[:n_original].mean()+loss[n_original:].mean()),(loss,acc)
def metrics(p):
    loss,(ce,acc)=objective(p,*batch)
    return dict(loss=float(loss),original_accuracy=float(acc[:n_original].mean()),recovery_accuracy=float(acc[n_original:].mean()),per_history_accuracy=np.asarray(acc).tolist(),per_history_ce=np.asarray(ce).tolist())
before=metrics(params)
tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(ap['learning_rate']));optim=tx.init(params)
@jax.jit
def update(p,s):
    (loss,aux),g=jax.value_and_grad(objective,has_aux=True)(p,*batch)
    updates,s=tx.update(g,s,p);p=optax.apply_updates(p,updates)
    finite=jnp.all(jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(p)]))
    return p,s,loss,aux,optax.global_norm(g),finite
write(folder/'manifest.json',dict(plan_sha256=sha(root/'recovery-learning-plan-v1.json'),parent_sha256=sha(parent),native_sha256=sha(args.native),script_sha256=sha(__file__),original_histories=n_original,recovery_histories=B-n_original,holdout_gradient=False))
start=time.monotonic()
with (folder/'metrics.jsonl').open('w') as f:
    for i in range(1,ap['updates']+1):
        params,optim,loss,aux,norm,finite=update(params,optim)
        entry=dict(update=i,loss=float(loss),grad_norm=float(norm),finite=bool(finite),per_history_accuracy=np.asarray(aux[1]).tolist(),seconds=time.monotonic()-start)
        assert entry['finite'] and np.isfinite([entry['loss'],entry['grad_norm']]).all()
        f.write(json.dumps(entry)+'\n');f.flush()
        if i%64==0:print(json.dumps(entry),flush=True)
checkpoint=folder/'model.flax_model';checkpoint.write_bytes(flax.serialization.to_bytes(params));(folder/'optimizer.msgpack').write_bytes(flax.serialization.to_bytes(optim))
write_checkpoint_metadata(checkpoint,observation_schema='structured-lite-v1',model_args=modelargs,code_list_hash=sha(args.code_list),semantic_table_hash=sha(args.semantics/'metadata.json'),capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8),training_context=dict(experiment=ap['id'],parent_sha256=sha(parent),updates=ap['updates'],full_legal_menu=True,promotion_eligible=False))
reloaded=load_params(checkpoint);delta=float(sum(np.sum((np.asarray(x).astype(np.float64)-np.asarray(y).astype(np.float64))**2) for x,y in zip(jax.tree.leaves(reloaded),jax.tree.leaves(initial)))**.5)
assert delta>0
write(folder/'summary.json',dict(completed=True,updates=ap['updates'],parent_sha256=sha(parent),checkpoint_sha256=sha(checkpoint),parameter_delta_l2=delta,before=before,after=metrics(reloaded)))
print(json.dumps(read(folder/'summary.json')),flush=True)
