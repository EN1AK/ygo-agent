"""Export readable local-goal questions without hidden opponent card identities.

The recurrent learner uses the separately frozen tensors and full histories;
these JSONL questions are a browsing/export interface, not a substitute for them.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


def read(path):return json.loads(path.read_text('utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def visible_state(state,seat,names):
    cards=[]
    for c in state['cards']:
        own=c['player']==seat
        if c['location'] not in (2,4,8,16,32,64,128):continue
        if not own and c['location'] in (2,64):continue
        card=dict(c)
        if not own and c['location'] in (4,8,32) and not c['position']&5:
            card['code']=0;card['name']='未公开卡片'
        else:card['name']=names.get(c['code'],str(c['code']))
        cards.append(card)
    return dict(cards=cards,lp=state['lp'],phase=state.get('phase'),turn_player=state.get('turn_player'))


def prompt(goal,names):
    kind=goal['kind']
    if kind=='settled_resource_inventory':return '本回合内，形成目标清单所列的资源分布。'
    if kind=='remove_public_threat':return f"本回合内，使对手场上的{name(goal['code'],names)}不多于{goal['max_remaining']}张。"
    if kind in ('attack_and_lp_reduction','battle_damage'):return f"本回合内，至少完成一次攻击，并将对手生命值压到{goal['opponent_lp_at_most']}或以下。"
    if kind=='postbattle_resources':return f"完成攻击后进入主要阶段二，手牌与表侧怪兽合计至少保留{goal['min_resources']}张。"
    if kind=='recovery_board':return '从对手响应后的给定局面接续展开，形成目标怪兽资源。'
    raise ValueError(kind)


def name(code,names):return names.get(code,str(code))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--database',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan=read(a.frozen/'plan.json')
    with sqlite3.connect(a.database) as db:names=dict(db.execute('select id,name from texts'))
    questions=[];artifacts=[]
    for row in plan['rows']:
        path=a.frozen/row['path'];assert sha(path)==row['record_sha256']
        r=read(path);assert r['independent_endpoint_audit']['passed']
        seat=r['definition']['controlled'];root=row.get('root_step',0)
        if 'lesson_start' in r:state=r['lesson_start']['state']
        else:state=dict(cards=[dict(code=x[0],player=x[1],location=x[2],position=x[4]) for x in r['definition']['initial']],lp=[8000,8000])
        answer=[]
        for i,d in enumerate(r['decisions'][root:],root):
            action=dict(d['menu'][d['action']]);action['name']=name(action['code'],names) if action['code'] else ''
            answer.append(dict(step=i,player=d['player'],action=action))
        q=dict(id=row['id'],group=row['group'],split=row['split'],source_sha256=row['source_sha256'],family=row['family'],
            controlled=seat,root_step=root,question=prompt(row['goal'],names),goal=row['goal'],
            observation=visible_state(state,seat,names),answer=dict(kind='one verified witness; not asserted unique or optimal',actions=answer,endpoint=visible_state(r['final'],seat,names)),
            terminal_win_claim=False,record_sha256=sha(path),card_database_sha256=sha(a.database),
            scope='fixed recorded context; model evaluation is separate from this witness')
        questions.append(q)
        if r['definition'].get('lua'):
            folder=a.output/row['id'];folder.mkdir()
            (folder/'puzzle.lua').write_text(r['definition']['lua'],'utf-8')
            zones={location:[x[0] for x in r['definition']['initial'] if x[1]==seat and x[2]==location] for location in (1,2,64)}
            # The designated hand belongs to the original main deck.
            deck=['#created by verified curriculum','#main',*map(str,zones[1]+zones[2]),'#extra',*map(str,zones[64]),'!side']
            (folder/'deck.ydk').write_text('\n'.join(deck)+'\n','utf-8')
            artifacts.append(dict(id=row['id'],lua_sha256=sha(folder/'puzzle.lua'),deck_sha256=sha(folder/'deck.ydk')))
    with (a.output/'questions.jsonl').open('w',encoding='utf-8') as f:
        for q in questions:f.write(json.dumps(q,ensure_ascii=False)+'\n')
    manifest=dict(questions=len(questions),counts=dict(Counter(q['group'] for q in questions)),sources=len({q['source_sha256'] for q in questions}),plan_sha256=sha(a.frozen/'plan.json'),questions_sha256=sha(a.output/'questions.jsonl'),card_database_sha256=sha(a.database),puzzles=artifacts,
        limitations=['Same replay may supply several correlated questions.','Lua initializes the scene; goal grading is performed by the curriculum runtime.','Lua puzzles require the pinned card scripts and duel rules.','Reference opponent actions are part of the witness, not a model response guarantee.','No hidden opponent hand or deck identities are exported in the question observation.'])
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
