import pathlib,hashlib,json,shutil
root=pathlib.Path(__file__).resolve().parent
out=root/'actor-runtime-v3';out.mkdir(exist_ok=False)
for name in ('assets','semantics','code_list.txt','registration.ydk'):
    (out/name).symlink_to((root/'actor-runtime-v2'/name).resolve())
native=out/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
shutil.copy2(root/'exercise_actor_full_v1.cpython-310-x86_64-linux-gnu.so',native)
assert hashlib.sha256(native.read_bytes()).hexdigest()=='567e0707cf8430947c35da4347ec1ff7895dcb0f0d4e7b1be9035387f16f9a18'
(out/'manifest.json').write_text(json.dumps(dict(native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),source_commit='236515aff7a2989ffc712dbab2aaddb0800e794b',full_legal_menu=True,assets_inherit='actor-runtime-v2',production_replaced=False),indent=2))
