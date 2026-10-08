"""Player-visible text observations from native queries and processed engine events."""
import json
import sqlite3
from pathlib import Path

from ygoai.rl.exercise_core import Reader

ZONES = {1:'deck',2:'hand',4:'monster',8:'spell/trap',16:'grave',32:'banished',64:'extra'}
PHASES = {1:'draw',2:'standby',4:'main1',8:'battle start',16:'battle step',32:'damage',
          64:'damage calculation',128:'battle',256:'main2',512:'end'}
POSITIONS = {1:'face-up attack',2:'face-down attack',4:'face-up defense',8:'face-down defense'}
PROMPTS = {10:'battle action',11:'main phase action',12:'activate this effect?',13:'yes/no',
           14:'choose effect option',15:'select cards',16:'chain response',18:'select zones',
           19:'select position',20:'select tributes',22:'allocate counters',23:'select sum/materials',
           24:'select disabled zones',25:'order cards',26:'select/unselect card',
           132:'rock paper scissors',140:'announce race',141:'announce attribute',
           142:'announce card',143:'announce number'}


def location(raw):
    return dict(player=raw[0], location=raw[1], sequence=raw[2], position=raw[3])


def identity_visible(card, viewer):
    loc, pos = card['location'], card.get('position', 0)
    if loc == 1:
        return False
    if card['player'] == viewer:
        return loc in (2,4,8,16,32,64) or bool(loc & 128)
    return loc == 16 or (loc in (4,8,32,64) and bool(pos & 5)) or bool(loc & 128)


def card_ref(card, viewer, code=None):
    ref = dict(side='self' if card['player']==viewer else 'opponent',
               zone=ZONES.get(card['location'], 'overlay' if card['location'] & 128 else str(card['location'])),
               slot=card['sequence'], position=POSITIONS.get(card.get('position',0), 'unspecified'))
    if code and identity_visible(card, viewer):
        ref['code'] = code
    return ref


def decode_fields(fields):
    """Decode exactly the query flags emitted by the isolated Jev bridge."""
    cards = []
    scalars = {1:'code',8:'type',16:'level',32:'rank',64:'attribute',128:'race',
               256:'attack',512:'defense',0x80000:'status',0x200000:'lscale',0x400000:'rscale'}
    for block in fields:
        r = Reader(bytes(block))
        while r.i < len(r.data):
            size = r.u32()
            if size == 4:
                continue
            b = Reader(r.take(size-4))
            flags = b.u32()
            if flags & ~0xebc3fb:
                raise ValueError('unexpected dynamic query flags')
            card = {}
            for bit in range(24):
                flag = 1 << bit
                if not flags & flag:
                    continue
                if flag == 2:
                    card.update(location(b.take(4)))
                elif flag in scalars:
                    value = b.u32()
                    if flag in (256,512) and value >= 2**31:
                        value -= 2**32
                    card[scalars[flag]] = value
                elif flag == 0x4000:
                    card['equipped_to'] = location(b.take(4))
                elif flag in (0x8000,0x10000,0x20000):
                    name = {0x8000:'targets',0x10000:'materials',0x20000:'counters'}[flag]
                    card[name] = [location(b.take(4)) if flag == 0x8000 else b.u32() for _ in range(b.u32())]
                elif flag == 0x800000:
                    card['link'], card['link_markers'] = b.u32(), b.u32()
                else:
                    raise ValueError('unsupported query flag')
            if b.i != len(b.data) or not all(k in card for k in ('code','player','location','sequence','position')):
                raise ValueError('incomplete dynamic card query')
            cards.append(card)
    return cards


def visible_cards(cards, viewer):
    result, counts = [], {}
    for c in cards:
        side = 'self' if c['player']==viewer else 'opponent'
        key = side+'/'+ZONES[c['location']]
        counts[key] = counts.get(key,0)+1
        if c['location']==1 or (c['player']!=viewer and c['location'] in (2,64) and not identity_visible(c,viewer)):
            continue
        row = card_ref(c,viewer,c['code'])
        if 'code' in row:
            for key in ('type','level','rank','attribute','race','attack','defense','link','link_markers','lscale','rscale','counters','materials'):
                if key in c:
                    row[key] = c[key]
            # Core QUERY_STATUS exposes effective disabled state. Do not feed the entire private bitfield.
            row['disabled'] = bool(c.get('status',0) & 1)
            if c.get('equipped_to'):
                row['equipped_to'] = card_ref(c['equipped_to'], viewer)
            if c.get('targets'):
                row['targets'] = [card_ref(t,viewer) for t in c['targets']]
        result.append(row)
    return dict(cards=result, zone_counts=counts)


def visible_event(frame, viewer):
    """Only whitelisted public/own-private facts enter policy context; never raw bytes."""
    raw = bytes(frame)
    if not raw:
        return None
    op = raw[0]
    b = Reader(raw[1:])
    if op == 50:
        code, src, dst, reason = b.u32(), location(b.take(4)), location(b.take(4)), b.u32()
        e = dict(event='move', source=card_ref(src,viewer), destination=card_ref(dst,viewer), reason=reason)
        if identity_visible(src,viewer) or identity_visible(dst,viewer): e['code']=code
        return e
    if op == 53:
        code = b.u32(); old=location(b.take(4)); new=dict(old,position=b.u8())
        return dict(event='position change', before=card_ref(old,viewer,code), after=card_ref(new,viewer,code))
    if op in (54,60,62,64):
        code, loc = b.u32(), location(b.take(4))
        return dict(event={54:'set',60:'summoning',62:'special summoning',64:'flip summoning'}[op], card=card_ref(loc,viewer,code))
    if op == 70:
        code, loc = b.u32(), location(b.take(4))
        b.take(3)  # triggering location, not effect target
        description, chain = b.u32(), b.u8()
        return dict(event='effect activated', code=code, source=card_ref(loc,viewer), description_id=description, chain=chain)
    if op in (71,72,73,75,76):
        return dict(event={71:'chain built',72:'resolving effect',73:'effect resolved',75:'activation negated',76:'effect negated'}[op],chain=b.u8())
    if op in (61,63,65,74,112,113,114):
        return dict(event={61:'summon completed',63:'special summon completed',65:'flip summon completed',74:'chain ended',112:'attack disabled',113:'damage step begins',114:'damage step ends'}[op])
    if op in (91,92,94,100):
        return dict(event={91:'damage',92:'recover',94:'LP update',100:'pay LP cost'}[op], player=b.u8(), amount=b.u32())
    if op == 90:
        player,count=b.u8(),b.u8()
        e=dict(event='draw',player=player,count=count)
        codes=[b.u32() & 0x7fffffff for _ in range(count)]
        if player==viewer: e['codes']=codes
        return e
    if op in (30,31,42):
        recipient=b.u8(); rows=[]
        for _ in range(b.u8()):
            code=b.u32(); c=dict(player=b.u8(),location=b.u8(),sequence=b.u8(),position=0)
            ref=card_ref(c,viewer)
            if recipient==viewer or identity_visible(c,viewer): ref['code']=code
            rows.append(ref)
        return dict(event='reveal cards', recipient=recipient, cards=rows)
    if op in (32,33,35,39):
        return dict(event={32:'shuffle deck',33:'shuffle hand',35:'swap grave/deck',39:'shuffle extra'}[op],player=b.u8())
    if op == 40: return dict(event='new turn',player=b.u8())
    if op == 41: return dict(event='new phase',phase=PHASES.get(int.from_bytes(b.take(2),'little'),'other'))
    if op == 5: return dict(event='duel ended',winner=b.u8(),reason=b.u8())
    if op == 110: return dict(event='attack',attacker=card_ref(location(b.take(4)),viewer),target=card_ref(location(b.take(4)),viewer))
    if op in (93,96,97):
        return dict(event={93:'equip',96:'target relation',97:'target relation removed'}[op],source=card_ref(location(b.take(4)),viewer),target=card_ref(location(b.take(4)),viewer))
    if op == 83: return dict(event='become target',cards=[card_ref(location(b.take(4)),viewer) for _ in range(b.u8())])
    # Uninterpreted packets remain in the audit trace, explicitly not presented as decoded events.
    return None


class Catalog:
    def __init__(self, database):
        self.db = sqlite3.connect(f'file:{Path(database).resolve().as_posix()}?mode=ro',uri=True)
        self.db.row_factory = sqlite3.Row
        self.cache = {}
        self.system = {}
        strings=Path(database).parent/'strings.conf'
        if strings.exists():
            for line in strings.read_text(encoding='utf-8-sig').splitlines():
                if line.startswith('!system '):
                    parts=line.split(maxsplit=2)
                    if len(parts)==3: self.system[int(parts[1])]=parts[2]

    def card(self, code):
        if code not in self.cache:
            r=self.db.execute('select * from texts where id=?',(code,)).fetchone()
            if r is None: raise ValueError('missing card text '+str(code))
            self.cache[code]=dict(r)
        return self.cache[code]

    def describe(self, code):
        r=self.card(code)
        return dict(code=code,name=r['name'],effect=r['desc'])

    def effect(self, code, effect):
        if code and 10010 <= effect < 10026:
            return self.card(code).get('str'+str(effect-10009),'')
        return ''

    def description(self, value):
        if value < 10000: return self.system.get(value,'description '+str(value))
        return self.card(value >> 4).get('str'+str((value & 15)+1),'')


def menu_view(snap, catalog):
    """Native menu is authoritative; keep stable indices and distinct descriptions."""
    rows=[]
    for i,a in enumerate(snap['menu']):
        code=a['code']; spec=a['spec']
        # Unknown opponent hand/facedown choices must remain anonymous even if the audit knows their code.
        if spec.startswith(('oh','ox')): code=0
        if spec.startswith(('om','os','or')):
            code=0  # conservative: field identity remains separately available in visible state
        parts=[]
        if a['act'] != 'None': parts.append(a['act'])
        if a['phase'] != 'None': parts.append('Enter '+a['phase'])
        if a['finish']: parts.append('Shuffle hand' if snap['message']==11 else 'Finish selection')
        if a['unselect']: parts.append('Unselect')
        if code: parts.append(str(code))
        if spec: parts.append('at '+spec)
        if a['position']: parts.append(POSITIONS.get(a['position'],str(a['position'])))
        if a.get('place'): parts.append('zone '+a['place_name'])
        if a.get('number'): parts.append('number '+str(a['number']))
        if a.get('race'): parts.append('race mask '+str(a['race']))
        if a.get('attribute'): parts.append('attribute mask '+str(a['attribute']))
        if a['effect'] >= 0: parts.append('effect '+str(a['effect']))
        if not parts: parts.append('Select')
        row=dict(index=i,description=' '.join(parts),code=code)
        if code: row['effect_description']=catalog.effect(code,a['effect'])
        rows.append(row)
    return rows


def observation(snap, trace, catalog, viewer=None, recent_limit=20):
    viewer=snap['player'] if viewer is None else viewer
    state=visible_cards(decode_fields(trace['fields']),viewer)
    events=[]; chain=[]; skipped=set()
    for f in trace['frames']:
        e=visible_event(f,viewer)
        if e:
            events.append(e)
            if e['event']=='effect activated': chain.append(e)
            elif e['event']=='chain ended': chain=[]
        elif f and f[0] not in PROMPTS: skipped.add(f[0])
    menu=menu_view(snap,catalog) if viewer==snap['player'] and not snap['done'] else []
    codes={c['code'] for c in state['cards'] if c.get('code')}
    codes.update(m['code'] for m in menu if m['code'])
    state.update(player=viewer,acting_player=snap['player'],lp=snap['lp'],turn=snap['turn_count'],
                 turn_player=snap['turn_player'],phase=PHASES.get(snap['phase'],str(snap['phase'])),
                 winner=snap['winner'] if snap['done'] and not snap['invalid'] else None,
                 current_chain=chain,recent_events=events[-recent_limit:],
                 omitted_older_events=max(0,len(events)-recent_limit),undecoded_event_types=sorted(skipped),
                 prompt=PROMPTS.get(snap['message'],str(snap['message'])),
                 selection=snap['selection'] if viewer==snap['player'] else None,
                 card_texts=[catalog.describe(c) for c in sorted(codes)],
                 effect_options=[dict(index=m['index'],text=m['effect_description']) for m in menu if m.get('effect_description')])
    if viewer==snap['player'] and snap['message'] in (12,13,14):
        raw=next((bytes(f) for f in reversed(trace['frames']) if f[0]==snap['message']),None)
        if raw:
            if raw[0] in (12,13):
                state['question']=catalog.description(int.from_bytes(raw[-4:],'little'))
            else:
                state['question_options']=[catalog.description(int.from_bytes(raw[i:i+4],'little'))
                                           for i in range(3,len(raw),4)]
    return state,menu
