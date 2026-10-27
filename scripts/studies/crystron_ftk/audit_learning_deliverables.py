"""Independent engine audits and complete replay bundles for completed evaluations."""
import argparse,collections,hashlib,json,lzma,math,pathlib,sqlite3,struct,sys
root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(root.parents[1]/'ygo-agent'))
from modern_core import Core
from ygoai.rl.human_replay import comparable_game_events
from ygoai.rl.demonstrations import parse_yrp2
read=lambda p:json.loads(p.read_text('utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf-8')
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--recovery',action='store_true');a=p.parse_args()
out=root/'learning-v1'/('recovery-evaluation' if a.recovery else 'evaluation')/a.label
summary=read(out/'summary.json');assert summary['completed']
parsed=read(root/'parsed.json')
names=dict(sqlite3.connect(root/'resources/repair-v6/cards.cdb').execute('select id,name from texts'))
opnames={2:'提示',5:'胜负结果',11:'主阶段操作',12:'效果发动询问',13:'是否询问',14:'选择选项',15:'选卡',16:'连锁选择',18:'选区域',19:'选表示形式',20:'选解放',22:'选计数器',23:'等级合计选择',24:'选禁用区域',25:'排序',26:'选择或取消卡片',30:'确认卡组顶',31:'确认卡片',32:'洗卡组',33:'洗手牌',39:'洗额外',40:'新回合',41:'新阶段',50:'卡片移动',53:'表示形式变化',60:'通常召唤开始',61:'通常召唤成功',62:'特殊召唤开始',63:'特殊召唤成功',70:'连锁发动',71:'连锁入链',72:'处理连锁',73:'连锁处理完毕',74:'连锁结束',75:'发动无效',76:'效果无效',83:'选择对象',90:'抽卡',91:'伤害',92:'回复',94:'LP更新',100:'支付LP',140:'宣言种族',141:'宣言属性',142:'宣言卡名',143:'宣言数字',160:'卡片提示'}
results=[]
for item in summary['results']:
    path=out/pathlib.Path(item['path']).name;r=read(path);source=parsed[item['sha']]
    assert sha(path)==item['sha256'] and r['checkpoint_sha256']==summary['checkpoint_sha256']
    c=Core(root/'resources/libocgcore-current-v2.so',root/'resources/repair-v6/cards.cdb',root/'resources/repair-v6/script',seed_sequence=source['seed_sequence'])
    proof=dict(path=str(path),record_sha256=sha(path),status=r['status'],passed=False)
    try:
        c.start_standard([[code,s,loc,0,8] for s in (0,1) for loc,deck in ((1,source['main_decks'][s]),(64,source['extra_decks'][s])) for code in deck],duel_flags=source['duel_flags'])
        for response in r['core_responses']:
            prompt=c.next_prompt();assert prompt is not None;c.respond(prompt,bytes.fromhex(response['data']))
        c.next_prompt()
        assert comparable_game_events(c.events)==comparable_game_events(r['events']),'engine event mismatch'
        assert c.state()==r['final'],'final state mismatch'
        if r['status']=='success':assert c.winner==0 and c.lp[1]<=0 and sum(e['op']==40 for e in c.events)==1
        proof.update(passed=True,event_sha256=digest(c.events),final_sha256=digest(c.state()))
    except Exception as ex:proof['error']=str(ex)
    finally:c.close()
    for d in r['decisions']:
        assert len(d['legal_actions'])==len(d['policy_logits'])==len(d['policy_probabilities'])
        assert 0<=d['selected_action']<len(d['legal_actions']) and abs(sum(d['policy_probabilities'])-1)<1e-5
        assert all(math.isfinite(v) for v in d['policy_logits']+[d['state_value']])
    payload=b''.join(n.encode('utf-16le')[:38].ljust(40,b'\0') for n in ['crystron-'+a.label,'pass'])+struct.pack('<4I',source['initial_lp'],source['initial_hand'],source['draw_count'],source['duel_flags'])
    for s in (0,1):
        for deck in (source['main_decks'][s],source['extra_decks'][s]):payload+=struct.pack('<I',len(deck))+struct.pack('<'+'I'*len(deck),*deck)
    for response in r['core_responses']:
        b=bytes.fromhex(response['data']);payload+=bytes([len(b)])+b
    packed=lzma.compress(payload,format=lzma.FORMAT_ALONE)
    header=struct.pack('<6I',846230137,4962,17,0,len(payload),source['start_time'])+packed[:5]+b'\0'*3+struct.pack('<8I',*source['seed_sequence'])+struct.pack('<4I',1,0,0,0)
    replay=path.with_suffix('.yrp');replay.write_bytes(header+packed[13:]);y=parse_yrp2(replay)
    assert y.responses==[v['data'] for v in r['core_responses']]
    for k in ('seed_sequence','main_decks','extra_decks'):assert list(getattr(y,k))==source[k]
    assert y.duel_flags==source['duel_flags'];proof['yrp_parse_passed']=True
    with path.with_suffix('.decisions.jsonl').open('w',encoding='utf-8') as f:
        for d in r['decisions']:f.write(json.dumps(dict(**d,checkpoint_sha256=r['checkpoint_sha256']),ensure_ascii=False)+'\n')
        f.write(json.dumps(dict(evaluation_end=True,duel_terminal=r['final']['winner'] is not None,status=r['status'],reward=r['terminal_reward'],final=r['final'],checkpoint_sha256=r['checkpoint_sha256'],opponent='optional pass; otherwise first legal',opponent_checkpoint=None),ensure_ascii=False)+'\n')
    lines=[];chains=[];damages=[]
    for i,e in enumerate(r['events']):
        b=bytes.fromhex(e['raw']);op=e['op'];detail=''
        if op in (50,60,62,70):
            code=int.from_bytes(b[1:5],'little');detail=f'{names.get(code,str(code))} ({code})'
            if op==70:chains.append(f'玩家{b[9]}：'+names.get(code,str(code)))
        if op in (91,92,94,100):
            detail=f'玩家{b[1]} 数值{int.from_bytes(b[2:6],"little")}'
            if op==91:damages.append(detail)
        if op==40:detail=f'行动方：玩家{b[1]}'
        if op==5:detail=f'玩家{b[1]}获胜，原因码{b[2]}'
        lines.append(f'{i:04d} [{op} {opnames.get(op,"引擎通知")}] {detail}\n  raw={e["raw"]}')
    actions=[];selected=[]
    for d in r['decisions']:
        chosen=d['legal_actions'][d['selected_action']];code=chosen.get('code',0)
        actions.append(f'步骤{d["step"]}，回合{d["turn"]}，阶段{d["phase"]}，玩家{d["player"]}，来源{d["source"]}：{names.get(code,str(code))} '+json.dumps(chosen,ensure_ascii=False))
        if d['source']=='model':selected.append(chosen)
    path.with_suffix('.log').write_text('\n'.join(lines+['全部操作选择：']+actions)+'\n','utf-8')
    if a.recovery:
        recovery=read(root/'recovery-learning-plan-v1.json');case=next(x for x in recovery['rows'] if x['id']==item['id']);reference=read(root/case['path'])
    else:reference=read(root/'actor-converted-v4'/(item['sha']+'.json'))
    divergence=None
    for d in r['decisions']:
        i=d['step']
        if d['source']!='model':continue
        if i>=len(reference['decisions']):break
        expected=reference['decisions'][i]
        if d['observation_sha256']!=expected['observation_sha256']:break
        if d['selected_action']!=expected['action']:
            divergence=dict(step=i,chosen=d['legal_actions'][d['selected_action']],reference=expected['menu'][expected['action']]);break
    cancel=sum(x['act']=='Cancel' or x.get('unselect',False) for x in selected)
    desc=[f'来源：{item["sha"]}；检查点：{r["checkpoint_sha256"]}',f'模式：{r["mode"]}；完整重放前缀至步骤{r["root_step"]}，再由模型自主控制先攻方。'+('前缀包含已实际发生的偏离动作，恢复从偏离后开始。' if a.recovery else ''),f'结果：{r["status"]}；停止时LP：{r["final"]["lp"]}；独立核心复核：{proof["passed"]}。','未达目标或步数耗尽只表示本次第一回合FTK任务未完成，不补写整场败北。','连锁与互动：'+' → '.join(chains),'伤害与收尾：'+'；'.join(damages),'与参考路线首次不同：'+json.dumps(divergence,ensure_ascii=False),'首次不同的动作不必然是错误；需后续引擎分支验证。',f'模型共选择{len(selected)}步，其中取消/撤选{cancel}次。反复选择和取消可能导致预算耗尽，详见完整操作日志。','前缀策略输出只用于诊断；对手后续让过可选发动，其余选首个合法项，无对手模型检查点。V(s)是状态价值，不是各动作Q值。','全部模型选择：']
    desc.extend(x for d,x in zip(r['decisions'],actions) if d['source']=='model')
    path.with_suffix('.说明.txt').write_text('\n\n'.join(desc),'utf-8')
    proof['artifacts']={f.name:sha(f) for f in [replay,path.with_suffix('.decisions.jsonl'),path.with_suffix('.log'),path.with_suffix('.说明.txt')]}
    results.append(proof)
    print(json.dumps(dict(path=path.name,passed=proof['passed'],status=r['status'],error=proof.get('error'))),flush=True)
write(out/'independent-audit.json',dict(passed=all(x['passed'] for x in results),count=len(results),summary_sha256=sha(out/'summary.json'),results=results))
assert all(x['passed'] for x in results)
