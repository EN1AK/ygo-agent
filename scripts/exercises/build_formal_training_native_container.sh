#!/usr/bin/env bash
set -euo pipefail
work=/elfnote-formal-selected-scope-20261023
test ! -e "$work"
mkdir "$work"
cp -a /src/ygoenv "$work/ygoenv"
cd "$work"
python3 - <<'PY'
from pathlib import Path
import hashlib,json
h=Path('ygoenv/ygoenv/ygopro/ygopro.h')
assert hashlib.sha256(h.read_bytes()).hexdigest()=='ef3c004a46dd5acf48ff004ae3c31eaa8cc3bb4917e1b76d7c4a34f2499b1827'
p=Path('ygoenv/ygoenv/ygopro/ygopro.cpp');text=p.read_text();anchor='PYBIND11_MODULE(ygopro_ygoenv, m) {'
assert text.count(anchor)==1
text=text.replace(anchor,anchor+'\n  m.attr("exercise_training_scope_version") = "selected-card-message-scope-v3";\n  m.attr("exercise_training_header_sha256") = "ef3c004a46dd5acf48ff004ae3c31eaa8cc3bb4917e1b76d7c4a34f2499b1827";')
p.write_text(text)
Path('/out/source-binding.json').write_text(json.dumps(dict(header_sha256=hashlib.sha256(h.read_bytes()).hexdigest(),binding_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='Exact 52c5 original adapter header; normal factory, no diagnostic hooks; no chain feature macro change; only two module provenance attributes added'),indent=2))
PY
lua=/root/.xmake/packages/l/lua/v5.3.6/e36ca01661cc4da38a40db5fbb826426
includes=(-I ygoenv -I /cache -I /cache/core -I /cache/SQLiteCpp-3.2.1/include -I /usr/include/python3.10 -I "$lua/include/lua")
for path in /root/.xmake/packages/{f/fmt,g/glog,g/gflags,c/concurrentqueue,u/unordered_dense,p/pybind11,l/lua}/*/*/include; do includes+=(-I "$path");done
includes+=(-I /root/.xmake/packages/s/sqlite3/3.43.0+200/95fcc1aa9fa6461d859b22bc16ee0811/include)
g++ -std=c++17 -O2 -fPIC -DNDEBUG -shared "${includes[@]}" ygoenv/ygoenv/ygopro/ygopro.cpp /cache/objects/*.o \
 /root/.xmake/packages/f/fmt/10.2.1/*/lib/libfmt.a /root/.xmake/packages/g/glog/v0.6.0/*/lib/libglog.a /root/.xmake/packages/g/gflags/v2.3.1/*/lib/libgflags.a /root/.xmake/packages/s/sqlite3/3.43.0+200/*/lib/libsqlite3.a "$lua/lib/liblua.a" -ldl -lm -lpthread -o /out/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so
cp ygoenv/ygoenv/ygopro/ygopro.cpp /out/compiled-binding.cpp
cp ygoenv/ygoenv/ygopro/ygopro.h /out/compiled-adapter.h
sha256sum /out/*.so /out/compiled-*.{cpp,h}
python3 - <<'PY'
import importlib.util,json
p='/out/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so'
s=importlib.util.spec_from_file_location('ygopro_ygoenv',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert m.exercise_training_scope_version=='selected-card-message-scope-v3'
print(json.dumps(dict(import_passed=True,scope=m.exercise_training_scope_version,header=m.exercise_training_header_sha256,chain_version=m.chain_event_provenance_version)),flush=True)
PY
