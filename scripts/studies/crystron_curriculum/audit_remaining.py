"""Replay previously unaudited deck sizes, isolated per recording."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sha')
    a=p.parse_args();a.source=a.source.resolve();a.catalog=a.catalog.resolve();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((a.catalog/'catalog.json').read_text('utf-8'))
    if a.sha:
        sys.path.insert(0,str(a.source))
        from modern_core import Core
        replay=json.loads((a.catalog/'parsed.json').read_text('utf-8'))[a.sha]
        source=next(r for r in json.loads((a.catalog/'candidates.json').read_text('utf-8')) if r['source_sha256']==a.sha)
        result=dict(source=source,training_ready=False)
        c=None
        try:
            c=Core(a.source/'resources/libocgcore-current-v2.so',a.source/'actor-runtime-v3/assets/cards.cdb',a.source/'actor-runtime-v3/assets/script',seed_sequence=replay['seed_sequence'])
            initial=[[code,s,loc,0,8] for s in (0,1) for loc,deck in ((1,replay['main_decks'][s]),(64,replay['extra_decks'][s])) for code in deck]
            c.start_standard(initial,duel_flags=replay['duel_flags'],lp=replay['initial_lp'],hand=replay['initial_hand'],draw=replay['draw_count'])
            for response in replay['responses']:
                prompt=c.next_prompt()
                if prompt is None:raise ValueError('engine ended before source responses')
                c.respond(prompt,bytes.fromhex(response))
            prompt=c.next_prompt()
            result.update(status='complete' if c.winner is not None else 'prefix_only',next_prompt=prompt)
        except Exception as exc:result.update(status='rejected',error=str(exc))
        finally:
            if c:
                result.update(runtime=c.runtime,assets=c.assets,events=c.events,decisions=c.decisions,final=c.state())
                c.close()
        (a.output/(a.sha+'.json')).write_text(json.dumps(result,ensure_ascii=False),'utf-8')
        print(json.dumps(dict(sha=a.sha,status=result['status'],error=result.get('error'))),flush=True)
        return
    for row in rows:
        if row['audit_status']!='not_audited':continue
        path=a.output/(row['source_sha256']+'.json')
        if path.exists():continue
        command=[sys.executable,__file__,'--source',str(a.source),'--catalog',str(a.catalog),'--output',str(a.output),'--sha',row['source_sha256']]
        try:
            process=subprocess.run(command,capture_output=True,text=True,timeout=45)
            print(process.stdout.strip(),flush=True)
            error=dict(status='runtime_failure',exit_code=process.returncode,error=process.stderr[-2000:])
        except subprocess.TimeoutExpired:error=dict(status='runtime_failure',error='45s process timeout; unknown outcome')
        if not path.exists():path.write_text(json.dumps(dict(source_sha256=row['source_sha256'],training_ready=False,**error)),'utf-8')


if __name__=='__main__':main()
