"""Backport explicit non-level Synchro materials in an isolated core cache.

Yummy scripts register EFFECT_SYNCHRO_LEVEL_EX (373). The actor core rejects
Link/Xyz materials before consulting it. Add only this explicit exception;
retain existing levelled-material semantics and all unrelated objects.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('missing or ambiguous core anchor: ' + old)
    return text.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'modern-source', 'lua-prefix', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    core = a.cache / 'core'
    before = {p.name: sha(p) for p in sorted(core.iterdir()) if p.is_file() and p.suffix in ('.h', '.cpp')}
    objects = {p.name: sha(p) for p in sorted((a.cache / 'objects').glob('*.o'))}
    if 'card.cpp.o' not in objects:
        raise ValueError('actual card object missing')
    modern = (a.modern_source / 'card.cpp').read_text()
    if not re.search(r'#define EFFECT_SYNCHRO_LEVEL_EX\s+373\b', (a.modern_source / 'effect.h').read_text()):
        raise ValueError('upstream explicit Synchro level effect changed')
    if 'if((data.type & (TYPE_XYZ | TYPE_LINK)) && !get_synchro_level(scard))' not in modern:
        raise ValueError('upstream material permission changed')
    shutil.copytree(a.cache, a.output, ignore=lambda directory,names:
        ['ygopro-core'] if Path(directory)==a.cache else [])
    # Both adapter include spellings must resolve to the same patched headers.
    (a.output / 'ygopro-core').symlink_to('core', target_is_directory=True)
    target = a.output / 'core'
    path = target / 'card.cpp'
    text = path.read_text()
    start = text.index('uint32 card::get_synchro_level(card* pcard) {')
    end = text.index('uint32 card::get_ritual_level(card* pcard) {', start)
    function = replace_once(text[start:end], '''\tif((data.type & (TYPE_XYZ | TYPE_LINK)) || (status & STATUS_NO_LEVEL))
\t\treturn 0;''', '''\tif((data.type & (TYPE_XYZ | TYPE_LINK)) || (status & STATUS_NO_LEVEL)) {
\t\t// Explicit scripted level for this particular Synchro monster.
\t\teffect_set allowed;
\t\tfilter_effect(EFFECT_SYNCHRO_LEVEL_EX, &allowed);
\t\tfor(int32 i = 0; i < allowed.size(); ++i) {
\t\t\tuint32 level = allowed[i]->get_value(pcard);
\t\t\tif(level) return level;
\t\t}
\t\treturn 0;
\t}''')
    text = text[:start] + function + text[end:]
    start = text.index('int32 card::is_can_be_synchro_material(card* scard, card* tuner) {')
    end = text.index('int32 card::is_can_be_ritual_material(card* scard) {', start)
    function = replace_once(text[start:end], '\tif(data.type & (TYPE_XYZ | TYPE_LINK))',
        '\tif((data.type & (TYPE_XYZ | TYPE_LINK)) && !get_synchro_level(scard))')
    text = text[:start] + function + text[end:]
    path.write_text(text)
    path = target / 'effect.h'
    text = path.read_text()
    if 'EFFECT_SYNCHRO_LEVEL_EX' in text or re.search(r'#define\s+\w+\s+373\b', text):
        raise ValueError('effect number already defined; inspect instead of overwriting')
    text, count = re.subn(r'(#define EFFECT_SYNCHRO_LEVEL[\t ]+240\n)',
        r'\1#define EFFECT_SYNCHRO_LEVEL_EX 373\n#define YGO_SYNCHRO_LEVEL_EX_CORE_REVISION 1\n', text)
    if count != 1:
        raise ValueError('effect definition anchor changed')
    path.write_text(text)
    command = ['g++', '-std=c++17', '-O2', '-DNDEBUG', '-fPIC', '-include', 'lua.hpp',
        '-I' + str(target), '-I' + str(a.lua_prefix / 'include/lua'), '-c', str(target / 'card.cpp'),
        '-o', str(a.output / 'objects/card.cpp.o')]
    subprocess.run(command, check=True)
    after = {name: sha(target / name) for name in before}
    changed = sorted(name for name in before if before[name] != after[name])
    if changed != ['card.cpp', 'effect.h']:
        raise ValueError('unexpected core source changes')
    linked = {name: sha(a.output / 'objects' / name) for name in objects}
    if sorted(name for name in objects if objects[name] != linked[name]) != ['card.cpp.o']:
        raise ValueError('unrelated core/build objects changed')
    if any(sha(core / name) != value for name, value in before.items()) or any(sha(a.cache / 'objects' / name) != value for name, value in objects.items()):
        raise ValueError('original cache mutated')
    proof = dict(schema='actor-synchro-level-ex-core-v1', passed=True, revision='explicit-nonlevel-material-v1',
        original_source=before, patched_source=after, changed_core_files=changed, original_objects=objects,
        linked_objects=linked, changed_objects=['card.cpp.o'], compiler_command=command,
        upstream_reference={name:sha(a.modern_source/name) for name in ('card.cpp','effect.h')},
        patcher_sha256=sha(Path(__file__)), original_cache_unchanged=True, production_replaced=False,
        scope='Explicit EFFECT_SYNCHRO_LEVEL_EX for non-level materials only. Existing levelled-material behavior retained. Real-card negative/positive fixtures and replay qualification required.')
    (a.output / 'core-rule-binding.json').write_text(json.dumps(proof, indent=2))
    print('ISOLATED_SYNCHRO_CORE_PREPARED', changed, flush=True)


if __name__ == '__main__':
    main()
