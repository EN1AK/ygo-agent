"""Actual GPU checkpoint reload on a frozen full environment; no optimization."""
import argparse,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    release=a.release.resolve();repo=release/'repo';sys.path[:0]=[str(repo/'ygoenv'),str(repo),str(repo/'scripts')]
    import jax,numpy as np,ygoenv
    from ygoai.rl.env import VersionedObservation
    from ygoai.utils import init_ygopro
    from scripts.eval_capability_exercises import FrozenActor
    from ygoai.rl.exercise_core import sha
    import ygoenv.ygopro.ygopro_ygoenv as native
    assert Path(native.__file__).resolve()==repo/'ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so'
    jax.config.update('jax_default_matmul_precision','highest');assert all(d.platform=='gpu' for d in jax.devices())
    deck=init_ygopro('YGOPro-v1','chinese',str(release/'decks/replay-elfnote.ydk'),str(release/'code_list.txt'))
    raw=ygoenv.make(task_id='YGOPro-v1',env_type='gymnasium',num_envs=1,num_threads=1,seed=2026102308,deck1=deck,deck2=deck,max_options=128,n_history_actions=32,play_mode='self',oppo_info=False,observation_schema='structured-lite-v1',semantic_asset_dir=str(release/'semantics'),n_public_events=32,max_group_references=8)
    raw.num_envs=1;env=VersionedObservation(raw,'structured-lite-v1');obs,info=env.reset()
    actor=FrozenActor(a.checkpoint,release/'code_list.txt',release/'semantics',obs)
    forward=jax.jit(lambda params,obs,state:actor.agent.apply(params,obs,state)[:3]);actor.forward=lambda obs,state:forward(actor.params,obs,state)
    decisions=[]
    try:
        for step in range(32):
            n=int(info['num_options'][0]);player=int(info['to_play'][0]);assert n>0
            logits,value,_=actor.predict(obs,player);assert np.isfinite(logits[:n]).all() and np.isfinite(value)
            action=int(np.argmax(logits[:n]));decisions.append(dict(step=step,player=player,legal_actions=n,action=action,value=value))
            obs,reward,terminated,truncated,info=env.step(np.array([action],np.int32))
            assert np.isfinite(reward).all()
            if bool(terminated[0] or truncated[0]):break
    finally:env.close()
    result=dict(passed=True,devices=[str(d) for d in jax.devices()],checkpoint_sha256=sha(a.checkpoint),native_sha256=sha(Path(native.__file__)),decisions=decisions,scope='GPU reload and actual full-adapter decisions; no optimizer or live-strength claim')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='decisions'}),flush=True)


if __name__=='__main__':main()
