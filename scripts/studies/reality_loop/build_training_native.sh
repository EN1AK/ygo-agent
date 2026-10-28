#!/usr/bin/env bash
# Inside the pinned portable image; /src and /cache are read-only, /out is new.
set -euo pipefail
work=/reality-training-native-v1
test ! -e "$work"
mkdir "$work"
cp -a /src/ygoenv "$work/ygoenv"
python3 /src/scripts/studies/reality_loop/prepare_training_native.py "$work" /out
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
sha256sum /out/*.so /out/compiled-* > /out/artifact.sha256
