"""Bounded home PPO continuation: 5M saves, 20M serial WindBot screens."""
import os,sys,json,time,math,hashlib,re,signal,subprocess,shutil
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
import flax.serialization
import numpy as np
BASE=Path('/home/ygo/ygo-agent')
ROOT=BASE/'training-runs/skystriker-121m-eight-hours-20261004'
RELEASE=BASE/'dist/runtime-releases/skystriker-place-forward-505b855-20261003'
REPO=RELEASE/'source'
EVAL=BASE/'training-runs/q2m-source-65443f7'
OLD=BASE/'training-runs/skystriker-place-forward-107m-to121m-fast-4x32-20261003'
PARENT=OLD/'checkpoints/31032028_step_000121208832.flax_model'
PARENT_SHA='cde8e3f6a06000319cbe3fe59f6112b2ffa37f5ece03b2f05b1ef06f16a03649'
NATIVE='a2dbd2604fec0e01ad722d887c639c00ed4c5aa74c912bd8f37d69f57a815486'
BATCH=8192; SAVE_UPDATES=math.ceil(5_000_000/BATCH); BLOCK=BATCH*SAVE_UPDATES*4
# Eight hours from the authorization/setup window, including evaluation.
DEADLINE=1791083239
ROOT.mkdir(exist_ok=False)
shutil.copy2(__file__,ROOT/'launcher.py')
state={'status':'preflight','started':time.time(),'deadline_epoch':DEADLINE,
 'start_step':121208832,'source_commit':'505b855d9e1e0a2363a3d81cf49d6097f7539555',
 'eval_source':'65443f76ba39a738aac79db924ba14defcc16cf2','parent_sha256':PARENT_SHA,
 'native_sha256':NATIVE,'save_steps':BATCH*SAVE_UPDATES,'eval_every_steps':BLOCK,
 'actor_optimizer':'fresh Adam on each segment, not restored','opponents':['self','self','history100m','bot'],
 'q_search_belief':'off','evaluations':{},'segments':[],'checkpoints':[]}
def write(p,x):
 temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(x,indent=2));temp.replace(p)
def save():write(ROOT/'schedule.json',state)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def leaves(x):
 if isinstance(x,dict):
  for v in x.values():yield from leaves(v)
 elif isinstance(x,(list,tuple)):
  for v in x:yield from leaves(v)
 else:yield np.asarray(x)
def verify(p):
 digest=sha(p);assert json.loads(Path(str(p)+'.metadata.json').read_text())['checkpoint_sha256']==digest
 arr=list(leaves(flax.serialization.msgpack_restore(p.read_bytes())))
 assert len(arr)==165 and all(np.isfinite(x).all() for x in arr)
 return {'checkpoint':str(p),'sha256':digest,'arrays':165,'finite':True,'step':int(p.stem.split('_')[-1])}
env=dict(os.environ,JAX_PLATFORMS='cuda',CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',LD_PRELOAD=str(RELEASE/'libcompat_glibc.so'),PYTHONFAULTHANDLER='1')
class Deadline(Exception):pass
def execute(cmd,repo,log,limit=900,checkpoints=None):
 env['PYTHONPATH']=f'{repo}/ygoenv:{repo}'
 with log.open('x') as stream:
  p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  state['active']={'pid':p.pid,'command':cmd,'log':str(log)};save()
  started=time.time()
  try:
   while p.poll() is None:
    time.sleep(3)
    if time.time()>=DEADLINE:raise Deadline('authorized eight-hour window ended')
    if time.time()-max(started,log.stat().st_mtime)>limit:raise RuntimeError('log stalled; evidence retained')
    tail=log.read_text(errors='replace')[-24000:]
    if re.search(r'optimizer_finite=\[\s*False|total_notfinite=\[\s*[1-9]',tail):raise RuntimeError('nonfinite optimizer')
    if checkpoints:
     seen={x['checkpoint'] for x in state['checkpoints']}
     for cp in sorted(checkpoints.glob('*.flax_model')):
      if str(cp) not in seen and Path(str(cp)+'.metadata.json').exists():
       proof=verify(cp);state['checkpoints'].append(proof);state['latest_checkpoint']=proof;save()
   return p.returncode
  except BaseException:
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGINT)
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:
     os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
  finally:
   state['last_process_exit']=p.returncode;state['active']=None;save()
def training_command(run,parent,offset,steps,seed,config=False):
 cmd=list(json.loads((OLD/'run-manifest.json').read_text())['command'])
 def setarg(k,v):cmd[cmd.index(k)+1]=str(v)
 for k,v in {'--seed':seed,'--ckpt-dir':run/'checkpoints','--run-name':run.name+'__'+str(seed),'--checkpoint':parent,'--tb-offset':offset,'--total-timesteps':steps,'--save-interval':SAVE_UPDATES,'--max-checkpoints':100}.items():setarg(k,v)
 if config:cmd.append('--config-only')
 return cmd
def evaluate(name,cp):
 run=ROOT/name;windbot=Path('/home/ygo/windbot-b0a2355')
 cmd=[sys.executable,'-u',str(EVAL/'scripts/eval_windbot.py'),'--games','64','--workers','4','--seed','2026104100','--player','0','--alternate-seats','--learner-deck','SkyStriker','--opponent-deck','SkyStriker','--timeout','120','--decision-logs','--isolate-replays','--output',str(run),'--repo-root',str(EVAL),'--eval-script','scripts/eval_structured.py','--','--deck',str(RELEASE/'SkyStriker.ydk'),'--code-list-file',str(RELEASE/'code_list.crlf.txt'),'--checkpoint',str(cp),'--observation-schema','structured-lite-v1','--semantic-asset-dir',str(RELEASE/'semantics'),'--max-options','128','--n-history-actions','32','--n-public-events','32','--max-group-references','8','--max-steps','1000','--windbot-executable',str(windbot/'WindBot.exe'),'--windbot-workdir',str(windbot),'--windbot-deck','SkyStriker','--windbot-mono','/usr/bin/mono','--windbot-source-revision','b0a2355f00bd59491add14ff09efa9914a3b47c6','--record','--no-cycle-guard']
 state['status']='evaluating';state['evaluations'][name]={'status':'running','checkpoint':str(cp),'sha256':sha(cp),'protocol':'64 attempts, 32 paired seeds; no replacement; invalid separate'};save()
 rc=execute(cmd,EVAL,ROOT/(name+'.log'),limit=1800)
 summary=json.loads((run/'summary.json').read_text());assert rc in (0,1) and summary['attempted']==64
 state['evaluations'][name].update(status='completed',summary=summary)
 assert not summary.get('interrupted') and summary.get('cleanup_complete'),summary
 save()
 if summary['invalid']>6:raise RuntimeError('more than 10 percent invalid evaluation attempts; investigate before continuation')
try:
 assert sha(PARENT)==PARENT_SHA;state['parent']=verify(PARENT)
 context=json.loads((RELEASE/'context.json').read_text())
 assert sha(REPO/'ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so')==NATIVE
 assert sha(EVAL/'ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so')==NATIVE
 assert sha(REPO/'scripts/script/procedure.lua')==context['procedure_sha256']
 assert sha(RELEASE/'SkyStriker.ydk')==context['input_deck_sha256']
 assert sha(RELEASE/'code_list.crlf.txt')==context['code_list_sha256']
 assert (Path('/home/ygo/windbot-b0a2355')/'WindBot.exe').exists()
 for proc in Path('/proc').glob('[0-9]*'):
  try:a=(proc/'cmdline').read_bytes().split(b'\0')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if a and b'python' in a[0] and any(Path(x.decode(errors='replace')).name in ('cleanba.py','eval_windbot.py','eval_structured.py') for x in a[1:]):raise RuntimeError('another GPU task active '+proc.name)
 save()
 config=ROOT/'config';config.mkdir();assert execute(training_command(config,PARENT,121208832,BATCH*2,104202601,True),REPO,config/'train.log')==0
 smoke=ROOT/'smoke';smoke.mkdir();assert execute(training_command(smoke,PARENT,121208832,BATCH*2,104202601),REPO,smoke/'train.log')==0
 smoke_cp=next((smoke/'checkpoints').glob('*.flax_model'));state['smoke']=verify(smoke_cp)
 log=(smoke/'train.log').read_text();assert 'optimizer_finite=[ True]' in log and 'total_notfinite=[0]' in log
 for actor,mode in enumerate(('self','self','history','bot')):assert f'actor {actor}: opponent_mode={mode}' in log
 state['smoke_weights_promoted']=False;save()
 evaluate('baseline121m',PARENT)
 parent=PARENT;offset=121208832;index=0
 while time.time()<DEADLINE-300:
  index+=1;run=ROOT/f'segment-{index:02d}';run.mkdir();seed=104202610+index
  cmd=training_command(run,parent,offset,BLOCK,seed)
  record={'run':str(run),'parent':verify(parent),'start_step':offset,'target_step':offset+BLOCK,'optimizer':'fresh Adam','status':'running','seed':seed}
  state['segments'].append(record);state['status']='training';save();write(run/'run-manifest.json',record)
  rc=execute(cmd,REPO,run/'train.log',checkpoints=run/'checkpoints');assert rc==0,rc
  cp=run/'checkpoints'/f'{seed}_step_{offset+BLOCK:012d}.flax_model';proof=verify(cp)
  log=(run/'train.log').read_text();finite=re.findall(r'optimizer_finite=\[([^]]+)\], notfinite_count=\[([^]]+)\], total_notfinite=\[([^]]+)\]',log)
  assert len(finite)==BLOCK//BATCH and all(a.strip()=='True' and b.strip()=='0' and c.strip()=='0' for a,b,c in finite)
  record.update(status='completed',endpoint=proof,optimizer_updates=len(finite));write(run/'run-manifest.json',record);(run/'completed.txt').write_text('completed');state['latest_checkpoint']=proof;save()
  evaluate(f'eval-{index:02d}',cp)
  parent=cp;offset+=BLOCK
 state['status']='window_completed';(ROOT/'completed.txt').write_text('authorized window completed')
except Deadline as error:
 state.update(status='window_completed',stop_reason=str(error));(ROOT/'completed.txt').write_text(str(error))
except BaseException as error:
 state.update(status='failed',error=repr(error));(ROOT/'failed.txt').write_text(repr(error));raise
finally:
 state['finished']=time.time();save()
