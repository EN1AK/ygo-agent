"""Frozen three-stage recurrent SFT with endpoint audits and autonomous tests."""
import argparse
import copy
import json
from pathlib import Path
import time
import optax
from supervision_runtime import Runtime,sha,read,write,jax,jnp,np,flax
from ygoai.rl.scene_batch import pack_scenes
from ygoai.rl.checkpoint_compat import write_checkpoint_metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();a.frozen=a.frozen.resolve();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    rt=Runtime(a.frozen);plan=rt.plan
    assert any(d.platform=='gpu' for d in jax.devices()),'GPU required'
    jax.config.update('jax_default_matmul_precision','highest')
    verification=[]
    for item in plan['rows']:
        proof=rt.verify(item)
        verification.append(dict(id=item['id'],**proof))
        print(json.dumps(dict(preflight=item['id'],**proof)),flush=True)
    write(a.output/'preflight.json',dict(passed=True,rows=verification,plan_sha256=sha(a.frozen/'plan.json'),devices=[str(d) for d in jax.devices()]))
    if a.preflight_only:return
    params=rt.load(plan['parent']);initial=params
    before=rt.evaluate(params,plan['parent_sha256'],a.output/'before')
    assert all(x['audit']['passed'] for x in before),'baseline replay audit failed'
    tx=optax.chain(optax.clip_by_global_norm(plan['gradient_clip']),optax.adam(plan['learning_rate']))
    optimizer=tx.init(params)
    rng=np.random.default_rng(plan['seed'])
    fit={g:[x for x in plan['rows'] if x['split']=='fit' and x['group']==g] for stage in plan['stages'] for g in stage['mix']}
    assert all(fit.values()),'all declared curriculum phases require data'
    compiled={}
    def batch(items):
        steps=max(x['own_steps'] for x in items)
        size=next((n for n in (64,128,256,512,1024,2048) if steps<=n),None)
        if size is None or size>plan['max_sequence_steps']:raise ValueError('full recurrent history exceeds budget')
        rows=[(rt.records[x['id']],rt.observations[x['id']]) for x in items]
        return size,tuple(jax.tree.map(jnp.asarray,v) for v in pack_scenes(rows,size))
    def functions(size):
        if size in compiled:return compiled[size]
        @jax.jit
        def objective(params,obs,targets,weights,legal):
            logits=rt.agent.apply(params,obs,rt.agent.init_rnn_state(2),jnp.zeros(size*2,bool),None)[1]
            logits=jnp.where(legal,logits,jnp.finfo(logits.dtype).min)
            ce=optax.softmax_cross_entropy_with_integer_labels(logits,targets).reshape(-1,2)
            accuracy=(weights*(jnp.argmax(logits,axis=-1)==targets).reshape(-1,2)).sum(0).mean()
            return (weights*ce).sum(0).mean(),accuracy
        @jax.jit
        def update(params,state,inputs):
            (loss,accuracy),grad=jax.value_and_grad(objective,has_aux=True)(params,*inputs)
            delta,state=tx.update(grad,state,params);params=optax.apply_updates(params,delta)
            finite=jnp.all(jnp.stack([jnp.isfinite(v).all() for v in jax.tree.leaves(params)]))
            return params,state,loss,accuracy,optax.global_norm(grad),finite
        compiled[size]=(objective,update)
        return compiled[size]
    parent_hash=plan['parent_sha256'];total=0;started=time.monotonic()
    for stage in plan['stages']:
        folder=a.output/stage['name'];folder.mkdir()
        groups=list(stage['mix']);probs=np.array(list(stage['mix'].values()),dtype=float);probs/=probs.sum()
        manifest=dict(plan_sha256=sha(a.frozen/'plan.json'),stage=stage,parent_sha256=parent_hash,source_sha256=sha(__file__),runtime_source_sha256=sha(Path(__file__).with_name('supervision_runtime.py')),native_sha256=plan['native_sha256'],holdout_gradient=False,full_recurrent_history=True)
        write(folder/'manifest.json',manifest)
        with (folder/'metrics.jsonl').open('w',encoding='utf-8') as f:
            for update in range(1,stage['updates']+1):
                items=[]
                for _ in range(2):
                    group=str(rng.choice(groups,p=probs));items.append(fit[group][int(rng.integers(len(fit[group])))])
                size,inputs=batch(items);_,step=functions(size)
                params,optimizer,loss,accuracy,norm,finite=step(params,optimizer,inputs)
                row=dict(update=update,total_update=total+update,examples=[x['id'] for x in items],sequence_steps=size,loss=float(loss),accuracy=float(accuracy),gradient_norm=float(norm),finite=bool(finite),seconds=time.monotonic()-started)
                assert row['finite'] and np.isfinite([row['loss'],row['gradient_norm']]).all()
                f.write(json.dumps(row)+'\n');f.flush()
                if update%32==0:print(json.dumps(row),flush=True)
        total+=stage['updates']
        checkpoint=folder/'model.flax_model';checkpoint.write_bytes(flax.serialization.to_bytes(params))
        (folder/'optimizer.msgpack').write_bytes(flax.serialization.to_bytes(optimizer))
        write_checkpoint_metadata(checkpoint,observation_schema='structured-lite-v1',model_args=rt.modelargs,code_list_hash=plan['code_list_sha256'],semantic_table_hash=plan['semantics_sha256'],capacities=rt.capacities,training_context=dict(experiment=plan['id'],stage=stage['name'],parent_sha256=parent_hash,updates=stage['updates'],full_legal_menu=True,promotion_eligible=False))
        reloaded=rt.load(checkpoint)
        delta=float(sum(np.sum((np.asarray(x).astype(np.float64)-np.asarray(y).astype(np.float64))**2) for x,y in zip(jax.tree.leaves(reloaded),jax.tree.leaves(initial)))**.5)
        assert delta>0
        parent_hash=sha(checkpoint)
        result=rt.evaluate(reloaded,parent_hash,folder/'evaluation')
        write(folder/'summary.json',dict(completed=True,updates=stage['updates'],checkpoint_sha256=parent_hash,parameter_delta_from_initial=delta,independent_evaluation_passed=all(x['audit']['passed'] for x in result),results=result))
        assert all(x['audit']['passed'] for x in result),'evaluation replay audit failed'
    write(a.output/'completion.json',dict(completed=True,updates=total,checkpoint_sha256=parent_hash,plan_sha256=sha(a.frozen/'plan.json'),promotion_eligible=False))


if __name__=='__main__':main()
