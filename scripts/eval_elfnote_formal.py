"""Validity-aware complete matches on an admitted frozen elfnote runtime."""
import argparse,json,os,random,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['release','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--opponent-checkpoint',type=Path)
    p.add_argument('--opponent',choices=['history','greedy'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--seat',type=int,choices=[0,1],required=True)
    p.add_argument('--episodes',type=int,default=32);a=p.parse_args()
    assert a.episodes>0 and not a.output.exists()
    release=a.release.resolve();repo=release/'repo'
    sys.path[:0]=[str(repo/'ygoenv'),str(repo),str(repo/'scripts')]
    import jax,jax.numpy as jnp,numpy as np,ygoenv
    from dataclasses import asdict
    from scripts.eval_mixed_match import _load,Args
    from ygoai.rl.jax.agent import RNNAgent,ModelArgs
    from ygoai.rl.env import VersionedObservation
    from ygoai.rl.utils import EnvPreprocess,RecordEpisodeStatistics
    from ygoai.rl.match_report import battle_report
    from ygoai.rl.checkpoint_compat import sha256_file
    from ygoai.utils import init_ygopro
    import ygoenv.ygopro.ygopro_ygoenv as native
    assert Path(native.__file__).resolve()==repo/'ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so'
    frozen=json.loads((release/'frozen-manifest.json').read_text())
    assert sha256_file(native.__file__)==frozen['native_sha256']
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='float32'
    assert all(d.platform=='gpu' for d in jax.devices())
    deck=init_ygopro('YGOPro-v1','chinese',str(release/'decks/replay-elfnote.ydk'),str(release/'code_list.txt'))
    seed=random.Random(a.seed+100000).randint(0,int(1e8))
    raw=ygoenv.make(task_id='YGOPro-v1',env_type='gymnasium',num_envs=a.episodes,num_threads=min(4,a.episodes),seed=seed,
        deck1=deck,deck2=deck,max_options=128,max_steps=1000,n_history_actions=32,async_reset=False,greedy_reward=True,
        play_mode='bot' if a.opponent=='greedy' else 'self',player=a.seat,timeout=600,oppo_info=False,
        observation_schema='structured-lite-v1',semantic_asset_dir=str(release/'semantics'),n_public_events=32,max_group_references=8)
    raw.num_envs=a.episodes;env=VersionedObservation(raw,'structured-lite-v1')
    sample=jax.tree.map(lambda x:jnp.array([x]),env.observation_space.sample())
    env=RecordEpisodeStatistics(EnvPreprocess(env,skip_mask=True))
    model=ModelArgs(observation_schema='structured-lite-v1');count=sum(bool(s.strip()) for s in (release/'code_list.txt').read_text().splitlines())
    agent=RNNAgent(embedding_shape=count,**asdict(model))
    args=Args(checkpoint=str(a.checkpoint),opponent_checkpoint=str(a.opponent_checkpoint or a.checkpoint),
        observation_schema='structured-lite-v1',max_options=128,semantic_asset_dir=str(release/'semantics'),code_list_file=str(release/'code_list.txt'))
    one=_load(a.checkpoint,agent,jax.random.PRNGKey(seed),sample,model,'structured-lite-v1',args)
    two=one
    if a.opponent=='history':
        assert a.opponent_checkpoint is not None
        two=_load(a.opponent_checkpoint,agent,jax.random.PRNGKey(seed+1),sample,model,'structured-lite-v1',args)
    @jax.jit
    def predict(params1,params2,obs,state1,state2,main,done):
        next1,logits1=agent.apply(params1,obs,state1)[:2]
        if a.opponent=='history':
            next2,logits2=agent.apply(params2,obs,state2)[:2]
            logits=jnp.where(main[:,None],logits1,logits2)
            state2=jax.tree.map(lambda new,old:jnp.where(main[:,None],old,new),next2,state2)
        else:logits=logits1
        state1=jax.tree.map(lambda new,old:jnp.where(main[:,None],new,old),next1,state1)
        state1,state2=jax.tree.map(lambda x:jnp.where(done[:,None],0,x),(state1,state2))
        # Illegal logits may deliberately be -inf; NaN/+inf are never accepted.
        finite=jnp.all(~jnp.isnan(logits)&~jnp.isposinf(logits))
        return state1,state2,logits.argmax(axis=1),finite
    def policy(obs,state1,state2,main,done):
        # A terminal native observation may carry its default to_play=0. Its
        # next action is unused; only live bot environments must be our turn.
        if a.opponent=='greedy':assert np.all(np.logical_or(main,done)),'Native bot returned a live opponent decision'
        state1,state2,actions,finite=predict(one,two,obs,state1,state2,main,done)
        assert bool(finite),'Nonfinite model policy'
        return state1,state2,actions
    start=time.time()
    try:
        result=battle_report(env,a.episodes,policy,agent.init_rnn_state(a.episodes),agent.init_rnn_state(a.episodes),candidate_seat=a.seat)
    finally:env.close()
    result.update(checkpoint_sha256=sha256_file(a.checkpoint),opponent=a.opponent,
        opponent_checkpoint_sha256=sha256_file(a.opponent_checkpoint) if a.opponent=='history' else None,
        native_sha256=frozen['native_sha256'],runtime_manifest_sha256=sha256_file(release/'frozen-manifest.json'),
        deck_sha256=sha256_file(release/'decks/replay-elfnote.ydk'),seed=a.seed,environment_seed=seed,
        candidate_seat=a.seat,collection='first_episode_per_environment; no replacement of invalid attempts',
        elapsed_seconds=time.time()-start)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['episodes','by_seat']}),flush=True)


if __name__=='__main__':main()
