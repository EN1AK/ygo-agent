"""Build an isolated modern-ABI core with the actor's seeded RandomSelect.

The modern core samples without replacement using a partial shuffle; the
actor's pinned core samples indices repeatedly, rejecting duplicates. Both
are uniform, but their seeded outcomes and subsequent RNG states differ.
Only that sampling loop is replaced; model/production libraries are untouched.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','actor-core','lua-prefix','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    target=a.output/'source';target.mkdir()
    originals={}
    for path in sorted(a.source.iterdir()):
        if path.is_file() and path.suffix in ('.h','.cpp'):
            originals[path.name]=sha(path);shutil.copy2(path,target/path.name)
    current=target/'libgroup.cpp';text=current.read_text()
    start=text.index('int32_t scriptlib::group_random_select(')
    end=text.index('int32_t scriptlib::group_cancelable_select(',start)
    function=text[start:end]
    before='''\t\tcard_vector cv(pgroup->container.begin(), pgroup->container.end());
\t\tint32_t back = (int32_t)cv.size() - 1;
\t\tfor (int32_t i = 0; i < count; ++i) {
\t\t\tint32_t r = pduel->get_next_integer(i, back);
\t\t\tstd::swap(cv[i], cv[r]);
\t\t}
\t\tnewgroup->container.insert(cv.begin(), cv.begin() + count);'''
    after='''\t\twhile(newgroup->container.size() < (size_t)count) {
\t\t\tint32_t i = pduel->get_next_integer(0, (int32_t)pgroup->container.size() - 1);
\t\t\tauto cit = pgroup->container.begin();
\t\t\tstd::advance(cit, i);
\t\t\tnewgroup->container.insert(*cit);
\t\t}'''
    if function.count(before)!=1:raise ValueError('modern RandomSelect source changed')
    actor=(a.actor_core/'libgroup.cpp').read_text()
    if not all(value in actor for value in ('while(newgroup->container.size() < count)',
            'pduel->get_next_integer(0, (int32)pgroup->container.size() - 1)',
            'std::advance(cit, i);','newgroup->container.insert(*cit);')):
        raise ValueError('actor sampling implementation changed')
    rng=(a.actor_core/'mtrandom.h').read_text()
    if not all(value in rng for value in ('if (replay_v2)', 'uint32_t bound = -range % range;',
            'std::seed_seq q(seq, seq + 8);')):
        raise ValueError('actor replay RNG implementation changed')
    current.write_text(text[:start]+function.replace(before,after)+text[end:])
    library=a.output/'libocgcore-actor-random-select-v1.so'
    command=['g++','-std=c++17','-O2','-fPIC','-shared','-I'+str(a.lua_prefix/'include/lua'),'-include','lua.hpp',
        *[str(path) for path in sorted(target.glob('*.cpp'))],
        str(a.lua_prefix/'lib/liblua.a'),'-ldl','-lm','-o',str(library)]
    subprocess.run(command,check=True)
    ctypes.CDLL(str(library.resolve()))
    manifest=dict(schema='independent-actor-random-select-v1',library=library.name,
        library_sha256=sha(library),builder_sha256=sha(Path(__file__)),command=command,
        original_source=originals,patched_source={path.name:sha(path) for path in sorted(target.iterdir())},
        actor_reference={name:sha(a.actor_core/name) for name in ('libgroup.cpp','mtrandom.h')},
        changed_files=['libgroup.cpp'],model_native_changed=False,load_passed=True,
        scope='seeded RandomSelect compatibility only; actual replay validation required')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(library=str(library),sha256=sha(library))),flush=True)


if __name__=='__main__':main()
