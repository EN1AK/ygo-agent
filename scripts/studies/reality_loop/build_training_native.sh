#!/usr/bin/env bash
# Inside the pinned portable image; /src and /cache are read-only, /out is new.
set -euo pipefail
work=/reality-training-native-v1
test ! -e "$work"
mkdir "$work"
cp -a /src/ygoenv "$work/ygoenv"
revision=()
if [[ "${OVERLAY_EVENT_FIX:-0}" = 1 ]]; then revision+=(--overlay-event-fix); fi
python3 /src/scripts/studies/reality_loop/prepare_training_native.py "$work" /out "${revision[@]}"
cd "$work"
lua=/root/.xmake/packages/l/lua/v5.3.6/e36ca01661cc4da38a40db5fbb826426
includes=(-I ygoenv -I /cache -I /cache/core -I /cache/SQLiteCpp-3.2.1/include -I /usr/include/python3.10 -I "$lua/include/lua")
for path in /root/.xmake/packages/{f/fmt,g/glog,g/gflags,c/concurrentqueue,u/unordered_dense,p/pybind11,l/lua}/*/*/include; do includes+=(-I "$path");done
includes+=(-I /root/.xmake/packages/s/sqlite3/3.43.0+200/95fcc1aa9fa6461d859b22bc16ee0811/include)
g++ -std=c++17 -O2 -fPIC -DNDEBUG -shared "${includes[@]}" ygoenv/ygoenv/ygopro/ygopro.cpp /cache/objects/*.o \
 /root/.xmake/packages/f/fmt/10.2.1/*/lib/libfmt.a /root/.xmake/packages/g/glog/v0.6.0/*/lib/libglog.a /root/.xmake/packages/g/gflags/v2.3.1/*/lib/libgflags.a /root/.xmake/packages/s/sqlite3/3.43.0+200/*/lib/libsqlite3.a "$lua/lib/liblua.a" -ldl -lm -lpthread -o /out/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so
cp ygoenv/ygoenv/ygopro/ygopro.cpp /out/compiled-binding.cpp
cp ygoenv/ygoenv/ygopro/ygopro.h /out/compiled-adapter.h
python3 - <<'PY'
import importlib.util,json
path='/out/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so'
spec=importlib.util.spec_from_file_location('ygopro_ygoenv',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
results=[]
if hasattr(m,'overlay_event_position_version'):
 cases=[]
 for location in [0x80,0x84]:
  for index in range(256):
   result=m._reality_public_event_position_fixture(location,index)
   assert result==(0,index,location),result
   cases.append(dict(location=location,index=index,result=result))
 for position in [1,2,4,8]:
  result=m._reality_public_event_position_fixture(4,position)
  assert result[0]>0 and result[1:]==(position,4),result
  cases.append(dict(location=4,position=position,result=result))
 open('/out/overlay-event-admission.json','w').write(json.dumps(dict(passed=True,cases=cases,scope='actual public event writer; overlay index retained; ordinary position encoded'),indent=2))
navigation=[]
chain=[]
for count in [2,3,5]:
 for forced in [False,True]:
  for action in range(count):
   frame,n,response,accepted=m._reality_chain_permutation_fixture(count,forced,action)
   assert response==count-1-action and accepted and n==count+(not forced),(count,action,response)
   chain.append(dict(count=count,forced=forced,policy_index=action,raw_index=response,core_accepted=accepted))
open('/out/chain-order-admission.json','w').write(json.dumps(dict(passed=True,cases=chain),indent=2))
for message in [15,20,26]:
 for pending in [False,True]:
  result=m._reality_navigation_filter(message,pending)
  assert result==(2,1,False,True) if pending else result==(2,2,True,False),result
  navigation.append(dict(message=message,summon_navigation=pending,result=result))
open('/out/navigation-admission.json','w').write(json.dumps(dict(passed=True,cases=navigation),indent=2))
for greedy in [False,True]:
 for seat in [0,1]:
  for winner in [0,1,2]:
   result=m._reality_terminal_fixture(winner,seat,greedy,True)
   assert result[0]==winner and result[2] and not result[3],result
   reward=result[1]
   assert reward==0 if winner==2 else (reward>0 if winner==seat else reward<0),result
   results.append(dict(winner=winner,seat=seat,greedy=greedy,result=result))
open('/out/terminal-admission.json','w').write(json.dumps(dict(passed=True,cases=results),indent=2))
print('real parser/step terminal cases passed:',len(results),flush=True)
PY
python3 /src/scripts/exercises/prepare_actor_header.py /out/compiled-adapter.h /out/exercise_ygopro.h
g++ -std=c++17 -O2 -fPIC -DNDEBUG -DEXERCISE_REPLAY_RNG -shared -I /out "${includes[@]}" \
 /src/scripts/exercises/actor_bridge.cpp /src/scripts/exercises/core_bridge.cpp /cache/objects/*.o \
 /root/.xmake/packages/f/fmt/10.2.1/*/lib/libfmt.a /root/.xmake/packages/g/glog/v0.6.0/*/lib/libglog.a /root/.xmake/packages/g/gflags/v2.3.1/*/lib/libgflags.a /root/.xmake/packages/s/sqlite3/3.43.0+200/*/lib/libsqlite3.a "$lua/lib/liblua.a" -ldl -lm -lpthread -o /out/exercise_actor_native.cpython-310-x86_64-linux-gnu.so
if test -e /cache/core-rule-binding.json; then
 cp /cache/core-rule-binding.json /out/core-rule-binding.json
 python3 - <<'PY'
import hashlib,importlib.util,json
from pathlib import Path
for name in ['ygopro_ygoenv','exercise_actor_native']:
 path=Path('/out')/(name+'.cpython-310-x86_64-linux-gnu.so')
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 assert module.core_synchro_level_ex_version=='explicit-nonlevel-material-v1'
path=Path('/out/source-binding.json');binding=json.loads(path.read_text())
binding['core_rule_binding_sha256']=hashlib.sha256(Path('/out/core-rule-binding.json').read_bytes()).hexdigest()
binding['core_rule_revision']='explicit-nonlevel-material-v1'
path.write_text(json.dumps(binding,indent=2))
PY
fi
sha256sum /out/*.so /out/compiled-* /out/source-binding.json > /out/artifact.sha256
