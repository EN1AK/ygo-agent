"""One native duel per process; JSON RPC keeps native callbacks and branches isolated."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from ygoai.rl.jev_duel_observation import Catalog, decode_fields, observation


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    wire=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1,encoding='utf-8')
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())  # native diagnostics cannot corrupt RPC
    env=None
    try:
        for line in sys.stdin:
            q=json.loads(line)
            try:
                cmd=q['cmd']
                if cmd=='init':
                    if env is not None: raise ValueError('worker already initialized')
                    cfg=q['config']
                    os.chdir(str(Path(cfg['scripts']).parent))
                    spec=importlib.util.spec_from_file_location('jev_duel_native',cfg['native'])
                    native=importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
                    native.init_module(cfg['database'],cfg['code_list'],{'_exercise':cfg['deck1']})
                    catalog=Catalog(cfg['database'])
                    env=native.Duel('',q['initial'],cfg['semantics'],cfg['seed'])
                    env.reset()
                    for action in q.get('prefix',[]):
                        if env.snapshot()['done']: raise ValueError('replay prefix beyond terminal')
                        env.step(action)
                elif cmd=='step':
                    snap=env.snapshot()
                    if snap['done'] or not 0<=q['action']<len(snap['menu']): raise ValueError('illegal native action')
                    env.step(q['action'])
                elif cmd=='close':
                    wire.write(json.dumps({'ok':True})+'\n'); break
                elif cmd not in ('snapshot','audit','observe'):
                    raise ValueError('unknown RPC operation')
                snap=env.snapshot(); trace=env.trace()
                responses=[dict(kind=r['kind'],data=bytes(r['data']).hex()) for r in trace['responses']]
                frames=[bytes(f).hex() for f in trace['frames']]
                fields=[bytes(f).hex() for f in trace['fields']]
                if cmd=='audit':
                    result=dict(responses=responses,frames=frames,fields=fields,
                                packets=[bytes(f).hex() for f in trace['packets']],core_seed=trace['core_seed'])
                else:
                    state,menu=observation(snap,trace,catalog,q.get('viewer'),recent_limit=cfg.get('recent_events',20))
                    result=dict(state=state,menu=menu,done=snap['done'],invalid=snap['invalid'],
                                termination_reason=snap['termination_reason'],winner=snap['winner'],
                                player=snap['player'],message=snap['message'],
                                response_count=len(responses),frame_count=len(frames),
                                fingerprint=digest(dict(fields=fields,frames=frames,responses=responses,menu=snap['menu'])))
                wire.write(json.dumps(dict(ok=True,result=result),ensure_ascii=False)+'\n')
            except Exception as exc:
                import traceback
                traceback.print_exc(file=sys.stderr)
                wire.write(json.dumps(dict(ok=False,error=repr(exc)))+'\n')
                break
    finally:
        if env is not None: env.close()
        wire.close()


if __name__=='__main__':
    main()
