"""Install a scoped dispatcher into home's isolated evaluation runtime."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/home/ygo/ygo-agent/training-runs/skystriker-generations-20m-v2-20261004')
EVAL = Path('/home/ygo/ygo-agent/training-runs/q2m-source-65443f7/scripts')
EXPECTED = '74a83d95b61345681181eb2d8850ff6cba95f6149f123044c3a5fdedd1559445'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    state = json.loads((ROOT / 'schedule.json').read_text())
    assert state['status'] == 'training', 'install only while training, not during evaluation'
    original = EVAL / 'eval_mixed_match.py'
    backup = EVAL / 'eval_mixed_match_before_resident.py'
    helper = EVAL / 'eval_generation_resident.py'
    assert sha(original) == EXPECTED, 'unexpected evaluation source; do not overwrite'
    assert not backup.exists() and not helper.exists()
    source = Path(__file__).with_name('eval_generation_resident.py')
    compile(source.read_text(), str(source), 'exec')
    shim = f'''import importlib.util
import sys
from pathlib import Path

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

helper = load("generation_resident_eval", Path(__file__).with_name("eval_generation_resident.py"))
req = helper.request(sys.argv[1:], {str(ROOT)!r})
if req is None:
    load("legacy_mixed_evaluation", Path(__file__).with_name("eval_mixed_match_before_resident.py")).main()
elif not helper.cached(req):
    single = load("legacy_mixed_evaluation", Path(__file__).with_name("eval_mixed_match_before_resident.py"))
    args = single.tyro.cli(single.Args)
    helper.run_matrix(single, args, req)
'''
    compile(shim, str(original), 'exec')
    shutil.copy2(original, backup)
    shutil.copy2(source, helper)
    assert sha(backup) == EXPECTED
    temporary = original.with_suffix('.resident.tmp')
    temporary.write_text(shim)
    temporary.replace(original)
    record = dict(original_sha256=EXPECTED, dispatcher_sha256=sha(original),
                  helper_sha256=sha(helper), backup=str(backup), root=str(ROOT),
                  runtime_validation='pending first scheduled evaluation; no training interruption',
                  timeout='existing caller 1800-second bound now applies to first full matrix call')
    (ROOT / 'resident-eval-deployment.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
