"""Generate a diagnostic header without editing the production adapter.

Input must come from a clean Git archive. Only factory dispatch and transparent
core tracing change; step/next/WriteState, masks and history remain identical.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(source, output):
    text = Path(source).read_text(encoding='utf-8')
    changes = {
        '  MDuel new_duel(uint32_t seed) {': '  virtual MDuel new_duel(uint32_t seed) {',
        '    return get_message(pduel, buf);': '    return exercise_get_message(pduel, buf);',
        '    return process(pduel);': '    return exercise_process(pduel);',
        '    set_responsei(pduel, value);': '    exercise_responsei(pduel, value);',
        '    set_responseb(pduel, buf);': '    exercise_responseb(pduel, buf, msg_, n_places_);',
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise ValueError('adapter anchor missing or ambiguous: ' + old)
        text = text.replace(old, new)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    manifest = {'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                'generated_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'changes': changes, 'actor_encoding_changed': False}
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    prepare(a.source, a.output)
