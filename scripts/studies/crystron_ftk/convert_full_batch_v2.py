import collections,json,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).parent
manifest=json.loads((root/'audit-repaired-v7-summary.json').read_text())
out=root/'actor-converted-v4';out.mkdir(exist_ok=True)
for sha in manifest['ftk']:
    target=out/(sha+'.json')
    try:
        r=subprocess.run([sys.executable,str(root/'convert_full_menu_v2.py'),'--sha',sha,'--audit','audit-repaired-v7'],text=True,capture_output=True,timeout=40)
        if not target.exists():target.write_text(json.dumps(dict(source_sha256=sha,status='runtime_failure',error=r.stderr[-1000:],exit_code=r.returncode)))
    except subprocess.TimeoutExpired:target.write_text(json.dumps(dict(source_sha256=sha,status='timeout')))
    data=json.loads(target.read_text());print(json.dumps(dict(sha=sha,status=data['status'],error=data.get('error'))),flush=True)
rows=[json.loads((out/(sha+'.json')).read_text()) for sha in manifest['ftk']]
summary=dict(status=dict(collections.Counter(r['status'] for r in rows)),converted=[r['source_sha256'] for r in rows if r['status']=='converted'],rejected=[dict(sha=r['source_sha256'],error=r.get('error')) for r in rows if r['status']!='converted'])
(root/'conversion-summary-v4.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(dict(status=summary['status'])),flush=True)
