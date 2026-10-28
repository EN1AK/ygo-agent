"""Shared full-history reconstruction and autonomous curriculum evaluation."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
PROJECT=Path(os.environ.get('YGO_PROJECT_ROOT',Path(__file__).resolve().parents[3]))
sys.path.insert(0,str(PROJECT))
import jax
import jax.numpy as jnp
import flax.serialization
from scripts.collect_teaching_exercise import load_native
from ygoai.rl.exercise_actor import array_digest,extract_trace,hidden_identity_audit
from ygoai.rl.human_replay import comparable_game_events
from ygoai.rl.jax.agent import RNNAgent,ModelArgs
from ygoai.rl.observation_schema import tensor_contract
from ygoai.rl.checkpoint_compat import validate_checkpoint_compatibility
from collect_single import independent_start
from goals import achieved
from ygoai.rl.chain_response import canonical_responses,chosen_chain_action,chain_response


def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Runtime:
    def __init__(self,folder):
        self.folder=Path(folder).resolve();self.plan=read(self.folder/'plan.json');p=self.plan
        # Missing means immutable historical full-menu data. New filtered
        # courses must declare False and reconstruct their stored histories.
        self.full_legal_menu=p.get('full_legal_menu',True)
        assert type(self.full_legal_menu) is bool
        self.source=Path(p.get('source_root',Path(p['parent']).parents[2]))
        sys.path.insert(0,str(self.source))
        from modern_core import Core
        self.Core=Core
        rt=Path(p['runtime'])
        self.args=SimpleNamespace(native=rt/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so',database=rt/'assets/cards.cdb',scripts=rt/'assets/script',code_list=rt/'code_list.txt',semantics=rt/'semantics')
        assert sha(self.args.native)==p['native_sha256']
        assert sha(self.args.code_list)==p['code_list_sha256']
        assert sha(self.args.semantics/'metadata.json')==p['semantics_sha256']
        self.native=load_native(self.args,rt/'registration.ydk')
        self.records={};self.observations={}
        for item in p['rows']:
            path=self.folder/item['path']
            assert sha(path)==item['record_sha256'] and sha(path.with_suffix('.npz'))==item['observations_sha256']
            self.records[item['id']]=read(path)
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as f:self.observations[item['id']]=dict(f)
        self.modelargs=ModelArgs(observation_schema='structured-lite-v1')
        self.capacities=dict(max_cards=80,max_options=128,history_actions=32,public_events=32,group_references=8)
        validate_checkpoint_compatibility(p['parent'],observation_schema='structured-lite-v1',model_args=self.modelargs,code_list_hash=p['code_list_sha256'],semantic_table_hash=p['semantics_sha256'],capacities=self.capacities)
        assert sha(p['parent'])==p['parent_sha256']
        self.agent=RNNAgent(**asdict(self.modelargs),embedding_shape=len(self.args.code_list.read_text().splitlines()))
        sample={k:jnp.asarray(v[:1]) for k,v in next(iter(self.observations.values())).items()}
        self.template=jax.jit(self.agent.init)(jax.random.PRNGKey(p['seed']),sample,self.agent.init_rnn_state(1))
        self.forward=jax.jit(lambda params,obs,state:self.agent.apply(params,obs,state)[:3])

    def load(self,path):return flax.serialization.from_bytes(self.template,Path(path).read_bytes())

    def env(self,record,seed=None):
        d=record['definition']
        return self.native.ExerciseActor(d.get('lua',''),d['initial'],str(self.args.semantics),d.get('seed',1) if seed is None else seed,'structured-lite-v1',d['seed_sequence'],d['duel_flags'],2048,self.full_legal_menu)

    def independent(self,record):
        d=record['definition']
        c=self.Core(self.source/'resources/libocgcore-current-v2.so',self.args.database,self.args.scripts,seed=record.get('core_seed',1),seed_sequence=d['seed_sequence'] or None)
        try:
            if d.get('lua'):independent_start(c,d['lua'],d['duel_flags'])
            else:c.start_standard(d['initial'],duel_flags=d['duel_flags'])
            canonical_responses(record)  # verify each native raw index first
            chains={d['core_response_count']:d for d in record['decisions'] if d['message']==16 and d.get('chain_prompt') is not None}
            remaps=0
            for index,response in enumerate(record['core_responses']):
                prompt=c.next_prompt()
                if prompt is None:raise ValueError('independent core ended early')
                data=bytes.fromhex(response['data'])
                if index in chains:
                    if prompt['op']!=16 or prompt['player']!=chains[index]['player']:
                        raise ValueError('independent chain prompt/actor drift')
                    actual=chain_response(prompt,chosen_chain_action(chains[index]))
                    remaps+=actual!=data;data=actual
                c.respond(prompt,data)
            c.next_prompt()
            assert comparable_game_events(c.events)==comparable_game_events(record['events']),'independent event mismatch'
            assert c.state()==record['final'],'independent full state mismatch'
            return dict(passed=True,responses=len(record['core_responses']),events=len(c.events),semantic_chain_index_remaps=remaps)
        finally:c.close()

    def verify(self,item):
        r=self.records[item['id']];obs=self.observations[item['id']]
        if not self.full_legal_menu and r.get('full_legal_menu') is not False:
            raise ValueError('filtered curriculum requires reconstructed filtered records')
        env=self.env(r)
        try:
            env.reset()
            checked_decisions=[]
            for i,d in enumerate(r['decisions']):
                snap=env.snapshot()
                assert not snap['invalid'] and not snap['done']
                actual={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                assert (snap['message'],snap['player'],snap['menu'],array_digest(actual))==(d['message'],d['player'],d['menu'],d['observation_sha256']),('actor mismatch',item['id'],i)
                assert all(np.array_equal(v[0],obs[k][i]) for k,v in actual.items()),('stored tensor mismatch',item['id'],i)
                if i==0:hidden_identity_audit(actual)
                events,_,responses=extract_trace(env,snap)
                checked_decisions.append(dict(d,core_response_count=len(responses),
                    chain_prompt=next((e for e in reversed(events) if e['op']==16),None) if snap['message']==16 else None))
                env.step(d['action'])
            snap=env.snapshot();events,state,responses=extract_trace(env,snap)
            checked=dict(r,decisions=checked_decisions,core_responses=responses,events=events,final=state,core_seed=env.trace()['core_seed'])
            assert state==r['final'] and canonical_responses(checked)==canonical_responses(r),'endpoint mismatch'
            assert comparable_game_events(events)==comparable_game_events(r['events'])
        finally:env.close()
        return self.independent(checked)

    def achieved(self,item,state,snapshot,events):
        seat=self.records[item['id']]['definition']['controlled']
        return achieved(item['goal'],state,snapshot,events,seat)

    def evaluate(self,params,checkpoint_hash,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);results=[]
        selected=[x for x in self.plan['rows'] if x['group'] in ('foundation','derivation','natural')]
        for group in sorted({x['group'] for x in self.plan['rows']}-{'foundation','derivation','natural','practice'}):
            for split in ('fit','holdout'):
                options=sorted([x for x in self.plan['rows'] if x['group']==group and x['split']==split],key=lambda x:hashlib.sha256(x['id'].encode()).hexdigest())
                selected+=options[:self.plan['evaluation'].get('local_'+split+'_count',4)]
        for split in ('fit','holdout'):
            practice=sorted([x for x in self.plan['rows'] if x['group']=='practice' and x['split']==split],key=lambda x:hashlib.sha256(x['id'].encode()).hexdigest())
            selected+=practice[:self.plan['evaluation'][split+'_practice_count']]
        for item in selected:
            seeds=self.plan['evaluation']['resource_seeds'] if item['group'] in ('foundation','derivation') else [None]
            for seed in seeds:
                reference=self.records[item['id']];seat=reference['definition']['controlled'];root=0
                if item['group']=='practice':
                    root=next(i for i,d in enumerate(reference['decisions']) if d['player']==seat and d['message']==11 and d['turn_player']==seat)
                root=item.get('root_step',root)
                budget=self.plan['evaluation']['practice_decision_budget' if item['group']=='practice' else 'resource_decision_budget']
                env=self.env(reference,seed);rnn=self.agent.init_rnn_state(1);decisions=[]
                result=dict(id=item['id'],group=item['group'],split=item['split'],definition=reference['definition'],seed=seed,root_step=root,checkpoint_sha256=checkpoint_hash,status='unknown',goal=item['goal'],scoring_version='first-turn-resource-v2')
                try:
                    env.reset()
                    for step in range(min(2048,root+budget)):
                        snap=env.snapshot();events,state,responses=extract_trace(env,snap)
                        if snap['invalid']:raise ValueError('invalid duel')
                        if 'max_turn' in item['goal'] or item['goal']['kind']=='settled_resource_inventory':
                            if sum(e['op']==40 for e in events)>item['goal'].get('max_turn',1):
                                result['status']='goal_not_reached';result['boundary']='target_turn_ended';break
                        if step>=root and self.achieved(item,state,snap,events):result['status']='success';break
                        if snap['done']:result['status']='goal_not_reached';break
                        obs={k:snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                        if step==0:hidden_identity_audit(obs)
                        scores=None;value=None;probs=None
                        if snap['player']==seat:
                            rnn,logits,v=self.forward(params,{k:jnp.asarray(x) for k,x in obs.items()},rnn)
                            scores=np.asarray(logits)[0,:len(snap['menu'])];value=float(np.asarray(v).reshape(-1)[0])
                            assert np.isfinite(scores).all() and np.isfinite(value)
                            probs=np.exp(scores-scores.max());probs/=probs.sum()
                        if step<root:
                            d=reference['decisions'][step]
                            assert (snap['message'],snap['player'],snap['menu'],array_digest(obs))==(d['message'],d['player'],d['menu'],d['observation_sha256']),'prefix drift'
                            action=d['action'];actor='reference_prefix'
                        elif snap['player']==seat:action=int(scores.argmax());actor='model'
                        else:
                            choices=[i for i,x in enumerate(snap['menu']) if x['act']=='Cancel'];action=choices[0] if choices else 0;actor='passive_opponent'
                        decisions.append(dict(step=step,player=snap['player'],message=snap['message'],phase=snap['phase'],menu=snap['menu'],action=action,actor=actor,core_response_count=len(responses),chain_prompt=next((e for e in reversed(events) if e['op']==16),None) if snap['message']==16 else None,policy_logits=scores.tolist() if scores is not None else None,policy_probabilities=probs.tolist() if probs is not None else None,value=value,observation_sha256=array_digest(obs)))
                        env.step(action)
                    else:result['status']='budget_exhausted'
                except Exception as exc:result.update(status='invalid',error=str(exc))
                finally:
                    snap=env.snapshot();events,state,responses=extract_trace(env,snap)
                    result.update(decisions=decisions,events=events,final=state,core_responses=responses,core_seed=env.trace()['core_seed'])
                    env.close()
                try:result['independent_audit']=self.independent(result)
                except Exception as exc:result['independent_audit']=dict(passed=False,error=str(exc))
                name=item['id']+'-'+str(seed)
                write(folder/(name+'.json'),result)
                row=dict(id=item['id'],group=item['group'],split=item['split'],seed=seed,status=result['status'],audit=result['independent_audit'],path=name+'.json')
                results.append(row);print(json.dumps(row),flush=True)
        write(folder/'summary.json',dict(checkpoint_sha256=checkpoint_hash,results=results))
        return results
