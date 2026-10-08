"""Download a pinned public Qwen3-4B snapshot, validating every large artifact."""
import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

REVISION='1cfa9a7208912126459214e8b04321603b3df60c'
MODEL_ID='Qwen/Qwen3-4B'


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',choices=('huggingface','modelscope'),default='huggingface')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    api=f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true'
    manifest=a.output/'download-manifest.json'
    meta=json.loads(manifest.read_text()) if manifest.exists() else json.load(urllib.request.urlopen(api,timeout=30))
    if meta['sha']!=REVISION: raise ValueError('revision mismatch')
    (a.output/'download-manifest.json').write_text(json.dumps(meta,indent=2))
    def ranged(row):
        name=row['rfilename']; chunk_size=4*1024*1024
        pieces=[(i,start,min(start+chunk_size,row['size'])-1) for i,start in enumerate(range(0,row['size'],chunk_size))]
        folder=a.output/(name+'.chunks4'); folder.mkdir(exist_ok=True)
        def chunk(piece):
            i,start,end=piece; path=folder/str(i)
            if path.exists() and path.stat().st_size==end-start+1: return path
            for attempt in range(8):
                try:
                    url=(f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}' if a.source=='huggingface'
                         else f'https://modelscope.cn/models/{MODEL_ID}/resolve/master/{name}')+f'?download=true&part4={i}&attempt={attempt}'
                    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(req,timeout=45) as r:
                        if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{row["size"]}':
                            raise ValueError('server did not honor exact range')
                        with path.open('wb') as stream:
                            for data in iter(lambda:r.read(1024*1024),b''): stream.write(data)
                    if path.stat().st_size!=end-start+1: raise ValueError('partial chunk')
                    if i%50==0: print(f'{name}: chunk {i}/{len(pieces)}',flush=True)
                    return path
                except Exception:
                    if attempt==7: raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            paths=list(pool.map(chunk,pieces))
        part=a.output/(name+'.part')
        with part.open('wb') as out:
            for path in paths:
                with path.open('rb') as stream:
                    for data in iter(lambda:stream.read(4*1024*1024),b''): out.write(data)
        part.replace(a.output/name)
    def fetch(row):
        name=row['rfilename']
        if name.startswith('.') or '/' in name: return
        target=a.output/name; part=target.with_suffix(target.suffix+'.part')
        expected=row.get('lfs',{}).get('sha256')
        def valid():
            if not target.exists() or target.stat().st_size!=row['size']: return False
            if not expected: return True
            h=hashlib.sha256()
            with target.open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
            return h.hexdigest()==expected
        if valid(): print('verified existing '+name,flush=True); return
        if row['size']>512*1024*1024:
            ranged(row)
            if not valid(): raise ValueError('hash mismatch '+name)
            print('verified '+name,flush=True); return
        for attempt in range(4):
            try:
                offset=part.stat().st_size if part.exists() else 0
                req=urllib.request.Request(f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}?download=true&t={int(time.time())}',
                                           headers={'Range':f'bytes={offset}-'} if offset else {})
                with urllib.request.urlopen(req,timeout=90) as response:
                    append=offset>0 and response.status==206
                    if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise ValueError('invalid range response')
                    with part.open('ab' if append else 'wb') as stream:
                        for block in iter(lambda:response.read(4*1024*1024),b''): stream.write(block)
                if part.stat().st_size!=row['size']: raise ValueError('size mismatch '+name)
                part.replace(target)
                if not valid(): raise ValueError('hash mismatch '+name)
                print('verified '+name,flush=True); return
            except Exception as exc:
                print(f'retry {name}: {exc}',flush=True)
                if attempt==3: raise
                time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch,meta['siblings']))
    print('snapshot verified '+REVISION,flush=True)


if __name__=='__main__': main()
