"""Playable independently certified search routes; no fabricated policy values."""
import hashlib,json,lzma,pathlib,sqlite3,struct,sys
root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(root.parents[1]/'ygo-agent'))
from ygoai.rl.demonstrations import parse_yrp2
read=lambda p:json.loads(p.read_text('utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parsed=read(root/'parsed.json');proof=read(root/'recovery-search-v2/independent-certification.json')
names=dict(sqlite3.connect(root/'resources/repair-v6/cards.cdb').execute('select id,name from texts'))
out=root/'recovery-search-v2/teacher-replays';out.mkdir(exist_ok=True);results=[]
for item in proof['results']:
    if not item['passed']:continue
    r=read(root/'recovery-search-v2/admitted'/(item['id']+'.json'));assert r['training_ready'] and r['independent_proof']==item
    source=parsed[item['source_sha256']];path=out/(item['id']+'.yrp')
    payload=b''.join(n.encode('utf-16le').ljust(40,b'\0') for n in ['search-teacher','recorded-opponent'])+struct.pack('<4I',source['initial_lp'],source['initial_hand'],source['draw_count'],source['duel_flags'])
    for s in (0,1):
        for deck in (source['main_decks'][s],source['extra_decks'][s]):payload+=struct.pack('<I',len(deck))+struct.pack('<'+'I'*len(deck),*deck)
    for response in r['core_responses']:
        b=bytes.fromhex(response['data']);payload+=bytes([len(b)])+b
    packed=lzma.compress(payload,format=lzma.FORMAT_ALONE)
    header=struct.pack('<6I',846230137,4962,17,0,len(payload),source['start_time'])+packed[:5]+b'\0'*3+struct.pack('<8I',*source['seed_sequence'])+struct.pack('<4I',1,0,0,0)
    path.write_bytes(header+packed[13:]);y=parse_yrp2(path)
    assert y.responses==[v['data'] for v in r['core_responses']] and list(y.seed_sequence)==source['seed_sequence']
    lines=[];chains=[];damage=[]
    for i,e in enumerate(r['events']):
        b=bytes.fromhex(e['raw']);detail=''
        if e['op'] in (50,60,62,70):
            code=int.from_bytes(b[1:5],'little');detail=names.get(code,str(code))
            if e['op']==70:chains.append(detail)
        if e['op']==91:detail=f'玩家{b[1]}受到{int.from_bytes(b[2:6],"little")}伤害';damage.append(detail)
        lines.append(f'{i:04d} opcode={e["op"]} {detail}\nraw={e["raw"]}')
    with path.with_suffix('.decisions.jsonl').open('w',encoding='utf-8') as f:
        for d in r['decisions']:
            chosen=d['menu'][d['action']];text=f'步骤{d["step"]} 玩家{d["player"]}：{names.get(chosen["code"],str(chosen["code"]))} '+json.dumps(chosen,ensure_ascii=False)
            lines.append(text)
            f.write(json.dumps(dict(**d,source='search_teacher' if d['step']>=item['recovery_root'] else 'frozen_actual_prefix',policy_logits=None,policy_probabilities=None,state_value=None,checkpoint_sha256=None),ensure_ascii=False)+'\n')
        f.write(json.dumps(dict(terminal=True,first_turn_ftk=True,final=r['final'],independent_proof=item),ensure_ascii=False)+'\n')
    path.with_suffix('.log').write_text('\n'.join(lines),'utf-8')
    desc=[f'来源：{item["source_sha256"]}。',f'保留模型实际偏离及此前全部动作，从步骤{item["recovery_root"]}之后恢复。搜索增加{item["search_branch_steps"]}步，再执行完整参考后续；所有动作均由引擎执行验证。',f'独立核心确认第一回合获胜，最终LP：{r["final"]["lp"]}。该路线提供{item["teacher_decisions"]}个恢复教学决策，前缀错误/偏离动作不计算教学损失。','连锁与展开：'+' → '.join(chains),'伤害收尾：'+'；'.join(damage),'这是引擎搜索教师路线，没有策略网络输出，因此 logits、概率和 V 均记为空；不伪造模型检查点或Q值。','完整前缀和后续动作、原始核心事件见同名日志。']
    path.with_suffix('.说明.txt').write_text('\n\n'.join(desc),'utf-8')
    files=[path,path.with_suffix('.decisions.jsonl'),path.with_suffix('.log'),path.with_suffix('.说明.txt')]
    results.append(dict(id=item['id'],passed=True,artifacts={p.name:sha(p) for p in files}))
(out/'manifest.json').write_text(json.dumps(dict(passed=True,count=len(results),results=results),ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(dict(passed=True,count=len(results))))
