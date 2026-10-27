import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import hashlib,json,pathlib,sys
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
root=pathlib.Path(__file__).resolve().parent;deployment=root.parents[1];project=deployment/'ygo-agent'
sys.path[:0]=[str(project),str(project/'scripts')]
import jax,jax.numpy as jnp,flax.serialization
from ygoai.rl.jax.agent import RNNAgent,ModelArgs
from ygoai.rl.observation_schema import tensor_contract
from ygoai.rl.exercise_actor import array_digest,extract_trace,hidden_identity_audit
from scripts.collect_teaching_exercise import load_native
import exercise_core,ygoai.rl.exercise_actor as trace_module
trace_module.messages=exercise_core.messages
def read(p):return json.loads(pathlib.Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf-8')
plan=read(root/'learning-plan-v1.json');run=root/'learning-v1';run.mkdir(exist_ok=True)
runtime=root/plan['runtime'];args=SimpleNamespace(native=runtime/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so',database=runtime/'assets/cards.cdb',scripts=runtime/'assets/script',code_list=runtime/'code_list.txt',semantics=runtime/'semantics')
native=load_native(args,runtime/'registration.ydk')
records={};observations={}
for item in plan['selected']:
    path=root/plan['trajectories']/(item['sha']+'.json')
    assert sha(path)==item['record_sha256'] and sha(path.with_suffix('.npz'))==item['observations_sha256']
    records[item['sha']]=read(path)
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as data:observations[item['sha']]=dict(data)
modelargs=ModelArgs(observation_schema='structured-lite-v1')
agent=RNNAgent(**asdict(modelargs),embedding_shape=len(args.code_list.read_text().splitlines()))
sample={k:jnp.asarray(v[:1]) for k,v in next(iter(observations.values())).items()}
template=jax.jit(agent.init)(jax.random.PRNGKey(plan['seed']),sample,agent.init_rnn_state(1))
forward=jax.jit(lambda p,o,s:agent.apply(p,o,s)[:3])
def load_params(checkpoint):return flax.serialization.from_bytes(template,pathlib.Path(checkpoint).read_bytes())
def new_env(record):
    d=record['definition']
    return native.ExerciseActor('',d['initial'],str(args.semantics),1,'structured-lite-v1',d['seed_sequence'],d['duel_flags'],2048,True)
