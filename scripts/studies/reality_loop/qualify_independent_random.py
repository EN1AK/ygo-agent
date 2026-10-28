"""Exercise actual Lua RandomSelect in independent and actor-core libraries.

Synthetic normal cards and empty effect scripts isolate sampling from card
effects. This checks seeded outcomes across multiple consecutive samples;
it does not replace complete recorded-game replay or runtime qualification.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
from ygoai.rl.exercise_core import Core as ActorCore
from scripts.studies.crystron_ftk.modern_core import Core as ModernCore


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample(kind,library,database,scripts,seed,count):
    words=[int.from_bytes(hashlib.sha256(f'random-select:{seed}:{i}'.encode()).digest()[:4],'little') for i in range(8)]
    core=kind(library,database,scripts,seed_sequence=words)
    try:
        lua='\n'.join(['Debug.ReloadFieldBegin(0)','Debug.SetPlayerInfo(0,8000,0,0)',
            'Debug.SetPlayerInfo(1,8000,0,0)',
            *[f'Debug.AddCard({1000+i},0,0,2,{i},10)' for i in range(20)],
            # No ReloadFieldEnd: it emits a version-specific display packet.
            # The raw field already contains the cards needed by RandomSelect.
            'local g=Duel.GetFieldGroup(0,2,0)',
            *[f'g:RandomSelect(0,{n})' for n in (count,5,7,1)]])
        name=b'__random_select_qualification.lua';data=lua.encode()
        core.buffers[name.decode()]=(ctypes.create_string_buffer(data),len(data))
        if not core.preload_script(core.d,name,len(name)):raise ValueError('sampling fixture preload rejected')
        core._check()
        buffer=ctypes.create_string_buffer(65536)
        size=core.get_message(core.d,buffer)
        raw=bytes(buffer.raw[:size])
        # Every observed event must actually be a random-selection packet.
        from ygoai.rl.exercise_core import messages
        events=list(messages(raw))
        if not events or any(e['op']!=81 for e in events):
            raise ValueError('fixture did not isolate real RandomSelect packets')
        return dict(events=events,state=core.state())
    finally:core.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('actor-bridge','modern-library','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--upstream-library',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[];negative_differences=0
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);database=root/'cards.cdb';scripts=root/'script';scripts.mkdir()
        with sqlite3.connect(database) as db:
            db.execute('create table datas (id integer, alias integer, setcode integer, type integer, atk integer, def integer, level integer, race integer, attribute integer)')
            db.executemany('insert into datas values (?,?,?,?,?,?,?,?,?)',[(1000+i,0,0,0x11,1000,1000,4,1,1) for i in range(20)])
        for name in ['constant.lua','utility.lua','procedure.lua',*[f'c{1000+i}.lua' for i in range(20)]]:
            (scripts/name).write_text('-- Deliberately empty: no card effects in this sampling fixture.\n')
        for seed in range(16):
            for count in (0,1,5,19,20,25):
                actual=sample(ActorCore,a.actor_bridge,database,scripts,seed,count)
                expected=sample(ModernCore,a.modern_library,database,scripts,seed,count)
                if actual!=expected:raise ValueError(f'core sampling mismatch seed={seed} count={count}')
                if a.upstream_library:
                    upstream=sample(ModernCore,a.upstream_library,database,scripts,seed,count)
                    negative_differences+=upstream!=actual
                rows.append(dict(seed=seed,count=count,passed=True,events=len(actual['events'])))
    if a.upstream_library and not negative_differences:
        raise ValueError('unpatched-core negative control did not distinguish sampling algorithms')
    report=dict(passed=True,cases=len(rows),consecutive_samples_per_case=4,rows=rows,
        actor_bridge_sha256=sha(a.actor_bridge),modern_library_sha256=sha(a.modern_library),
        upstream_library_sha256=sha(a.upstream_library) if a.upstream_library else None,
        unpatched_differing_cases=negative_differences,qualifier_sha256=sha(__file__),
        scope='actual Lua/core seeded sampling and unchanged synthetic field only; complete factual replay remains required')
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
