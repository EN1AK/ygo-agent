"""Opt-in alternating scene rehearsal between complete ordinary PPO updates."""
import copy,hashlib,json,math
from pathlib import Path


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_training_mode(args,config=None):
    if args.distributed or len(args.learner_device_ids)!=1 or args.bfloat16 or getattr(args,'q_training_mode','off')!='off':
        raise ValueError('Scene rehearsal requires one float32 ordinary PPO learner')
    if config is not None and 'full_legal_menu' in config and getattr(args,'full_legal_menu',False)!=config['full_legal_menu']:
        raise ValueError('PPO and rehearsal menu contracts differ')


def validate_header(config):
    if config.get('format_version')!=1 or config.get('role') not in ['control','rehearse']:
        raise ValueError('Explicit rehearsal format and role required')
    rate=config.get('learning_rate')
    if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or not 0<rate<=1e-4:
        raise ValueError('Invalid rehearsal learning rate')
    n=config.get('histories')
    steps=config.get('sequence_steps')
    if type(n)!=int or not 4<=n<=128 or n%2 or type(steps)!=int or steps not in (128,256,512,1024,2048) or len(config.get('rows',[]))!=n:
        raise ValueError('Declare bounded even complete histories and a sequence capacity')
    if 'full_legal_menu' in config and type(config['full_legal_menu']) is not bool:
        raise ValueError('Explicit boolean menu contract required')
    pairs=config.get('gradient_check_pairs',[])
    if len(pairs)!=2 or len(set(pairs))!=2 or any(type(i)!=int or not 0<=i<n//2 for i in pairs):
        raise ValueError('Declare two distinct gradient qualification pairs')
    for row in config['rows']:
        for key in ['ce_coef','kl_coef']:
            value=row[key]
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
                raise ValueError('Invalid scene coefficient')
        steps=row['teaching_steps']
        if not steps or steps!=sorted(set(steps)) or any(type(x)!=int or x<0 for x in steps):
            raise ValueError('Explicit unique ordered scene mask required')
    return config


def load_rows(config):
    import numpy as np
    rows=[]
    for item in config['rows']:
        path=Path(item['path']);arrays=path.with_suffix('.npz')
        assert sha(path)==item['record_sha256'] and sha(arrays)==item['observations_sha256']
        record=copy.deepcopy(read(path));controlled=item['controlled']
        if 'full_legal_menu' in config and record.get('full_legal_menu') is not config['full_legal_menu']:
            raise ValueError('rehearsal record menu contract differs')
        record.setdefault('definition',{})['controlled']=controlled
        if sum(d['player']==controlled for d in record['decisions'])>config['sequence_steps']:
            raise ValueError('complete rehearsal history exceeds declared capacity; never truncate')
        assert all(i<len(record['decisions']) and record['decisions'][i]['player']==controlled for i in item['teaching_steps'])
        active=set(item['teaching_steps'])
        for i,d in enumerate(record['decisions']):d['teaching']=i in active
        with np.load(arrays,allow_pickle=False) as f:obs=dict(f)
        assert all(len(v)==len(record['decisions']) for v in obs.values())
        rows.append((record,obs))
    return rows


class SceneRehearsal:
    """Separate fresh Adam; PPO moments/step remain untouched and are not reset."""
    def __init__(self,config_path,checkpoint,agent,parent,code_list,semantics,output):
        import jax
        from scripts.eval_capability_exercises import FrozenActor
        from ygoai.rl.scene_optimizer import make_optimizer
        self.path=Path(config_path);self.config=validate_header(read(self.path));m=self.config
        assert sha(checkpoint)==m['parent_sha256']==sha(m['parent_checkpoint'])
        assert sha(code_list)==m['code_list_sha256'] and sha(Path(semantics)/'metadata.json')==m['semantic_metadata_sha256']
        assert all(sha(path)==digest for path,digest in m['protected'].items())
        rows=load_rows(m);sample={k:v[:1] for k,v in rows[0][1].items()}
        assert not parent.get('batch_stats',{}),'This replay study requires the current stateless-normalization model'
        actor=FrozenActor(Path(m['parent_checkpoint']),Path(code_list),Path(semantics),sample)
        actor.agent=agent;actor.params=parent
        infer=jax.jit(lambda params,obs,state:agent.apply(params,obs,state)[:3]);actor.forward=lambda obs,state:infer(actor.params,obs,state)
        active=m['role']=='rehearse'
        self.cc=[r['ce_coef'] if active else 0. for r in m['rows']];self.kc=[r['kl_coef'] if active else 0. for r in m['rows']]
        self.update,self.state,admission=make_optimizer(actor,parent,rows,m['sequence_steps'],m['learning_rate'],self.cc,self.kc,
            check_pairs=m['gradient_check_pairs'],qualification_hard_targets=True)
        assert admission['gradient_reference_l2']>1e-8,'Gradient qualification must be nonvacuous'
        self.log=Path(output)/'scene-rehearsal.jsonl';assert not self.log.exists()
        proof=Path(output)/'scene-rehearsal-admission.json';assert not proof.exists()
        proof.write_text(json.dumps(dict(config=m,config_sha256=sha(self.path),admission=admission,
            scope='One separate full-scene Adam step after each complete PPO batch. Control computes same declared histories with zero coefficients; PPO optimizer state unchanged.'),indent=2))
        self.count=0
        @jax.jit
        def difference(x,y):
            import jax.numpy as jnp
            leaves=jax.tree.leaves(jax.tree.map(lambda a,b:jnp.sum((a-b)**2),x,y))
            return jnp.sqrt(sum(leaves))
        self.difference=difference

    def apply(self,variables,ppo_update):
        import time,numpy as np
        started=time.time();new,state,loss,terms,norm,finite=self.update(variables,self.state)
        loss=float(loss);ce=np.asarray(terms[0]);kl=np.asarray(terms[1]);expected=float(np.dot(self.cc,ce)+np.dot(self.kc,kl))
        assert bool(finite) and np.isfinite([loss,float(norm)]).all() and np.isfinite(ce).all() and np.isfinite(kl).all()
        assert abs(loss-expected)/max(1.,abs(loss),abs(expected))<=8*np.finfo(np.float32).eps
        delta=float(self.difference(new,variables))
        if self.config['role']=='control':assert loss==float(norm)==delta==0.
        self.count+=1;assert self.count==ppo_update
        self.state=state
        with self.log.open('a') as f:f.write(json.dumps(dict(ppo_update=ppo_update,rehearsal_updates=self.count,role=self.config['role'],loss=loss,
            ce=ce.tolist(),kl=kl.tolist(),grad_norm=float(norm),finite=bool(finite),parameter_delta_l2=delta,seconds=time.time()-started))+'\n')
        return new
