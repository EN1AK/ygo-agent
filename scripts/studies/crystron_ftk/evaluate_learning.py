"""Autonomous continuation with a complete recurrent prefix at frozen roots."""
import argparse
from learning_common import *
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args()
checkpoint=root/a.checkpoint;params=load_params(checkpoint);out=run/'evaluation'/a.label;out.mkdir(parents=True,exist_ok=True);results=[]
def rollout(item,root_spec):
    reference=records[item['sha']];env=new_env(reference);root_step=root_spec['step']
    states=[agent.init_rnn_state(1),agent.init_rnn_state(1)];decisions=[]
    result=dict(source_sha256=item['sha'],split=item['split'],mode=root_spec['mode'],root_step=root_step,checkpoint_sha256=sha(checkpoint),full_legal_menu=True,root_reached=False,status='unknown')
    try:
        env.reset()
        for step in range(root_step+384):
            snap=env.snapshot();assert snap['full_legal_menu'];events,state,responses=extract_trace(env,snap)
            turns=sum(e['op']==40 for e in events)
            if snap['invalid']:raise ValueError('invalid native state')
            if snap['done'] or turns>1:
                result['status']='success' if state['winner']==0 and state['lp'][1]<=0 and turns==1 else 'goal_not_reached';break
            obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')};player=snap['player'];n=len(snap['menu'])
            if step==0:hidden_identity_audit(obs)
            rnn_hash=array_digest({str(i):x for i,x in enumerate(jax.tree.leaves(states[player]))})
            states[player],logits,value=forward(params,{k:jnp.asarray(v) for k,v in obs.items()},states[player]);scores=np.asarray(logits)[0,:n];v=float(np.asarray(value).reshape(-1)[0])
            assert np.isfinite(scores).all() and np.isfinite(v)
            probs=np.exp(scores-scores.max());probs/=probs.sum()
            if step<=root_step:
                expected=reference['decisions'][step]
                assert (snap['message'],player,snap['menu'],array_digest(obs))==(expected['message'],expected['player'],expected['menu'],expected['observation_sha256']),('prefix drift',step)
                assert responses==reference['core_responses'][:len(responses)],'prefix response drift'
            if step<root_step:action=reference['decisions'][step]['action'];source='prefix'
            elif player==0:action=int(scores.argmax());source='model';result['root_reached']=True
            else:
                options=[i for i,m in enumerate(snap['menu']) if m['act']=='Cancel'];action=options[0] if options else 0;source='opponent_pass'
            decisions.append(dict(step=step,player=player,phase=state['phase'],turn=turns,message=snap['message'],legal_actions=snap['menu'],selected_action=action,source=source,policy_logits=scores.tolist(),policy_probabilities=probs.tolist(),state_value=v,rnn_sha256=rnn_hash,observation_sha256=array_digest(obs),core_response_count=len(responses)))
            env.step(action)
        else:result['status']='decision_budget_exhausted'
    except Exception as e:result.update(status='invalid',error=str(e))
    finally:
        snap=env.snapshot();events,state,responses=extract_trace(env,snap);env.close()
        result.update(decisions=decisions,events=events,final=state,core_responses=responses,terminal_reward=1 if result['status']=='success' else 0 if result['status']=='goal_not_reached' else None)
    return result
for item in plan['selected']:
    for root_spec in plan['roots'][item['sha']]:
        path=out/(item['sha'][:12]+'-'+root_spec['mode']+'.json')
        if path.exists():
            result=read(path);assert result['checkpoint_sha256']==sha(checkpoint) and result['root_step']==root_spec['step']
        else:result=rollout(item,root_spec);write(path,result)
        entry=dict(label=a.label,sha=item['sha'],split=item['split'],mode=root_spec['mode'],status=result['status'],root_reached=result['root_reached'],path=str(path),sha256=sha(path),error=result.get('error'))
        results.append(entry);print(json.dumps(entry),flush=True)
write(out/'summary.json',dict(completed=True,plan_sha256=sha(root/'learning-plan-v1.json'),checkpoint_sha256=sha(checkpoint),results=results))
