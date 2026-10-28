"""Real Yummy-script positive/negative fixtures on old, fixed and modern cores.

Verify all three real Synchros, both seats and both Extra Deck orders. Only an
explicitly allowed Link-1 gains level/material permission; Link-2 and Xyz do
not. Ordinary levelled materials retain their prior behavior. The first real
idle menu must expose exactly the corresponding three legal Synchro summons.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from ygoai.rl.exercise_core import Core as ActorCore
from scripts.studies.crystron_ftk.modern_core import Core as ModernCore

YUMMY = [31603289, 93192592, 67098897]
MATERIALS = [('link1',41999284,0,False), ('ordinary',94145021,1,True),
    ('link2',29301450,0,False), ('xyz',90448279,0,False)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(kind, library, database, scripts, seat, reverse, material, target):
    core = kind(library, database, scripts, seed=31)
    extra = [*YUMMY, 50091196]
    if reverse:
        extra.reverse()
    name, code, _, _ = material
    lua = '\n'.join(['Debug.ReloadFieldBegin(0)', 'Debug.SetPlayerInfo(0,8000,0,0)',
        'Debug.SetPlayerInfo(1,8000,0,0)',
        f'Debug.AddCard({code},{seat},{seat},4,0,1)',
        f'Debug.AddCard(94145021,{seat},{seat},4,1,1)',
        *[('local sc=' if c==target else '')+f'Debug.AddCard({c},{seat},{seat},64,{i},8)' for i,c in enumerate(extra)],
        'Debug.ReloadFieldEnd()',
        f'local m=Duel.GetFieldCard({seat},4,0)',
        f'assert(sc:IsCode({target}))',
        f'Duel.Hint(HINT_NUMBER,{seat},m:GetSynchroLevel(sc))',
        f'Duel.Hint(HINT_NUMBER,{seat},m:IsCanBeSynchroMaterial(sc) and 1 or 0)',
        f'Duel.Hint(HINT_NUMBER,{seat},5852465)'])
    try:
        key = b'__synchro_level_ex_qualification.lua'
        data = lua.encode()
        core.buffers[key.decode()] = (ctypes.create_string_buffer(data),len(data))
        if not core.preload_script(core.d,key,len(key)):
            raise ValueError('real-card fixture preload rejected')
        core._check()
        buffer = ctypes.create_string_buffer(65536)
        length = core.get_message(core.d,buffer)
        tail = buffer.raw[:length][-21:]
        if len(tail) != 21 or any(tail[i] != 2 for i in (0,7,14)):
            raise ValueError('fixture did not emit three final real-core hints')
        values = [int.from_bytes(tail[i+3:i+7],'little') for i in (0,7,14)]
        if values[-1] != 5852465:
            raise ValueError('fixture sentinel mismatch')
        core.start_duel(core.d,(5<<16)|2)
        for _ in range(16):
            prompt = core.next_prompt()
            if prompt is None:
                raise ValueError('fixture ended before its material-choice idle menu')
            if prompt['op'] == 11 and prompt['player'] == seat:
                offered = sorted(c['code'] for c in prompt['special'] if c['code'] in YUMMY)
                return dict(level=values[0],material_allowed=bool(values[1]),yummy_summons=offered)
            if prompt['op'] == 11 and prompt['end']:
                core.respond(prompt,7)
            elif prompt['op'] == 16 and not prompt['forced']:
                core.respond(prompt,-1)
            else:
                raise ValueError('unexpected real-card fixture prompt: '+str(prompt))
        raise ValueError('fixture did not reach the requested seat')
    finally:
        core.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('old-native','fixed-native','modern-library','database','scripts','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    rows = []
    for seat in (0,1):
        for reverse in (False,True):
            for material in MATERIALS:
                for target in YUMMY:
                    old = fixture(ActorCore,a.old_native,a.database,a.scripts,seat,reverse,material,target)
                    fixed = fixture(ActorCore,a.fixed_native,a.database,a.scripts,seat,reverse,material,target)
                    modern = fixture(ModernCore,a.modern_library,a.database,a.scripts,seat,reverse,material,target)
                    positive = material[0]=='link1'
                    expected = dict(level=1 if positive else material[2],material_allowed=positive or material[3],
                        yummy_summons=sorted(YUMMY) if positive else [])
                    legacy = dict(level=material[2],material_allowed=material[3],yummy_summons=[])
                    if old != legacy or fixed != expected or modern != expected:
                        raise ValueError(str(dict(seat=seat,reverse=reverse,material=material[0],target=target,
                            old=old,fixed=fixed,modern=modern,expected=expected)))
                    rows.append(dict(seat=seat,reverse=reverse,material=material[0],target=target,old=old,fixed=fixed,modern=modern))
                    print('VERIFIED',seat,reverse,material[0],target,flush=True)
    proof = dict(passed=True,cases=len(rows),positive_link1_cases=sum(r['material']=='link1' for r in rows),
        source_sha256=sha(Path(__file__)),old_native_sha256=sha(a.old_native),fixed_native_sha256=sha(a.fixed_native),
        modern_library_sha256=sha(a.modern_library),database_sha256=sha(a.database),
        actual_card_scripts={str(c):sha(a.scripts/f'c{c}.lua') for c in [*YUMMY,50091196,*[m[1] for m in MATERIALS]]},
        rows=rows,policy_model_inference=False,production_replaced=False,
        scope='Real scripts, effect queries and actual idle menus; old core is a failing negative control only for explicitly enabled Link-1 material. Model-route replay and new-runtime examinations remain required.')
    (a.output/'report.json').write_text(json.dumps(proof,indent=2))
    print('SYNCHRO_LEVEL_EX_QUALIFIED',len(rows),flush=True)


if __name__=='__main__':
    main()
