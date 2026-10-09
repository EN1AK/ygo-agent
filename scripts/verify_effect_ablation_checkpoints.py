"""Reload each saved candidate and reproduce its recorded endpoint predictions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--code-list',type=Path,required=True)
    a=p.parse_args()
    import jax
    import flax.serialization
    from ygoai.rl.jax.agent import RNNAgent,ModelArgs
    from ygoai.rl.checkpoint_compat import load_checkpoint_metadata,sha256_file
    from scripts.study_effect_public_state import masked
    jax.config.update('jax_default_matmul_precision','highest')
    result=json.loads((a.run/'summary.json').read_text())
    meta=load_checkpoint_metadata(a.run/'migrated.flax_model')
    assert meta['code_list_hash']==sha256_file(a.code_list)
    agent=RNNAgent(embedding_shape=sum(bool(s.strip()) for s in a.code_list.read_text().splitlines()),
                   **asdict(ModelArgs(**meta['model_architecture'])))
    path=next((a.run/'before').glob('*-0.npz'))
    with np.load(path,allow_pickle=False) as f: sample={k:v[:1] for k,v in f.items()}
    template=jax.jit(agent.init)(jax.random.PRNGKey(0),sample,agent.init_rnn_state(1))
    infer=jax.jit(lambda params,obs,state:agent.apply(params,obs,state)[:3])
    reports=[]
    for summary in result['arms']:
        arm=summary['arm']; folder=a.run/arm['name']
        checkpoint=folder/'model.flax_model'
        current=load_checkpoint_metadata(checkpoint)
        assert current['checkpoint_sha256']==summary['checkpoint_sha256']
        assert current['model_architecture']==meta['model_architecture']
        params=flax.serialization.from_bytes(template,checkpoint.read_bytes())
        error=0.; decisions=0; trajectories=0
        for path in sorted((folder/'endpoint').glob('*-0.json')):
            record=json.loads(path.read_text())
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as f: arrays=dict(f)
            states=[agent.init_rnn_state(1),agent.init_rnn_state(1)]
            for i,d in enumerate(record['decisions']):
                player=d['player']; obs=masked({k:v[i:i+1] for k,v in arrays.items()},arm)
                states[player],logits,value=infer(params,obs,states[player])
                expected=d.get('logits',d.get('policy_logits'))
                expected_value=d.get('value',d.get('state_value'))
                n=len(d['menu'])
                delta=max(float(np.abs(np.asarray(logits)[0,:n]-expected).max()),
                          abs(float(np.asarray(value).reshape(-1)[0])-expected_value))
                assert np.isfinite(delta) and delta<.002,(arm['name'],path.name,i,delta)
                error=max(error,delta); decisions+=1
            trajectories+=1
        assert trajectories==31
        reports.append(dict(arm=arm['name'],trajectories=trajectories,decisions=decisions,max_error=error,passed=True))
    output=a.run/'saved-checkpoint-verification.json'
    with output.open('x') as f: json.dump(dict(passed=True,results=reports),f,indent=2)
    print(json.dumps(reports),flush=True)


if __name__=='__main__': main()
