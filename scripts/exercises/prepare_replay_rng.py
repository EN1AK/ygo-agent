"""Add opt-in YRP2 seed-sequence RNG to an isolated copy of the pinned core.

The default single-seed path is unchanged. Never patch a production core tree.
Reference: Fluorohydride/ygopro-core mtrandom.h, create_duel_v2 in ocgapi.cpp.
"""
from pathlib import Path
import argparse
import hashlib
import json


def prepare(path):
    path = Path(path)
    before = path.read_bytes()
    source = before.decode().replace('\r\n', '\n')
    changes = {
        '\tvoid reset(uint_fast32_t seed) {\n\t\trng.seed(seed);':
        '\tvoid reset(uint_fast32_t seed) {\n\t\treplay_v2 = false;\n\t\trng.seed(seed);',
        '\tuint_fast32_t rand() {':
        '\tvoid reset_replay(const uint32_t* seq) {\n\t\tstd::seed_seq q(seq, seq + 8);\n\t\trng.seed(q);\n\t\treplay_v2 = true;\n\t}\n\tuint_fast32_t rand() {',
        '\tint get_random_integer(int l, int h) {':
        '\tint get_random_integer(int l, int h) {\n\t\tif (replay_v2) {\n\t\t\tuint32_t range = uint32_t(h - l + 1);\n\t\t\tuint32_t bound = -range % range;\n\t\t\tauto x = rng();\n\t\t\twhile (x < bound) x = rng();\n\t\t\treturn l + int(x % range);\n\t\t}',
        '\tstd::mt19937 rng;': '\tstd::mt19937 rng;\n\tbool replay_v2 = false;',
    }
    for old, new in changes.items():
        if source.count(old) != 1:
            raise ValueError('pinned RNG anchor missing or ambiguous: ' + old)
        source = source.replace(old, new)
    path.write_text(source, encoding='utf-8', newline='\n')
    path.with_suffix('.replay.json').write_text(json.dumps(dict(
        original_sha256=hashlib.sha256(before).hexdigest(),
        patched_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        default_rng_changed=False, replay_rng='std::seed_seq[8] + v2 rejection sampling'
    ), indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('isolated_header', type=Path)
    prepare(p.parse_args().isolated_header)
