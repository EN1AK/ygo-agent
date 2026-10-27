"""Build a local, inspectable Chinese course viewer from verified records."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import sqlite3


def read(path):
    return json.loads(path.read_text('utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--database',type=Path,required=True)
    p.add_argument('--local-goals',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    courses=[]
    def load(folder):
        plan=read(folder/'plan.json')
        for row in plan['rows']:
            if row['group']=='practice':continue
            if any(c['id']==row['id'] for c in courses):continue
            path=folder/row['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['record_sha256'],('record hash mismatch',row['id'])
            r=read(path);seat=r['definition']['controlled']
            root=row.get('root_step',0)
            start=r.get('lesson_start',{}).get('state')
            if start is None:
                start=dict(cards=[dict(code=c[0],player=c[1],location=c[2],position=c[4]) for c in r['definition']['initial']],lp=[8000,8000])
            decisions=[]
            for i,d in enumerate(r['decisions'][root:],root):
                selected=d['menu'][d['action']]
                decisions.append(dict(step=i+1,player=d['player'],message=d['message'],action=selected,choices=len(d['menu'])))
            stages=[]
            for b in r.get('boundaries',[]):
                if b['step']<=len(r['decisions']):stages.append(dict(step=b['step'],state=b['state']))
            courses.append(dict(id=row['id'],group=row['group'],split=row['split'],seat=seat,ordinal=1+sum(c['group']==row['group'] for c in courses),
                source=row['source_sha256'],goal=row['goal'],start=start,end=r['final'],
                actions=decisions,stages=stages,root=root,verified=bool(r.get('independent_endpoint_audit',{}).get('passed') or r.get('independent_audit',{}).get('passed'))))
    load(a.study/'frozen-supervision-v1')
    if a.local_goals:load(a.local_goals)
    codes=set()
    for c in courses:
        for state in [c['start'],c['end']]+[s['state'] for s in c['stages']]:
            codes.update(x['code'] for x in state['cards'] if x['location'] in (2,4,8,16,32))
        codes.update(d['action']['code'] for d in c['actions'])
    with sqlite3.connect(a.database) as db:
        cards={str(code):dict(name=name,text=text) for code,name,text in db.execute('select id,name,desc from texts') if code in codes}
    data=dict(courses=courses,cards=cards,counts=dict(Counter(c['group'] for c in courses)))
    template=Path(__file__).with_name('course_browser.html').read_text('utf-8')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(template.replace('__COURSE_DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),'utf-8')
    print(json.dumps(dict(output=str(a.output),courses=len(courses),counts=data['counts'])))


if __name__=='__main__':main()
