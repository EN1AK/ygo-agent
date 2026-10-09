"""Append effect declarations while preserving a checkpoint's exact old tables."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base','cards-db','code-list','scripts','output-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    a = p.parse_args()
    old = json.loads((a.base/'metadata.json').read_text())
    assert digest(a.code_list) == old['source_hashes']['code_list_sha256']
    assert digest(a.cards_db) == old['source_hashes']['cards_db_sha256']
    assert not a.output_dir.exists()
    subprocess.run([sys.executable,str(Path(__file__).with_name('build_structured_semantics.py')),
        '--cards-db',str(a.cards_db),'--code-list',str(a.code_list),'--scripts',str(a.scripts),
        '--output-dir',str(a.output_dir),'--effect-descriptions'],check=True,stdout=subprocess.DEVNULL)
    current = json.loads((a.output_dir/'metadata.json').read_text())
    assert current['rows'] == old['rows']
    for key,name in {'static':'card-semantics.u8','tags':'effect-tags.u8','confidence':'effect-tag-confidence.u8'}.items():
        source,target = a.base/name,a.output_dir/name
        assert digest(source) == old['table_hashes'][key] and source.stat().st_size == target.stat().st_size
        target.write_bytes(source.read_bytes())
        current['table_hashes'][key] = old['table_hashes'][key]
    current['preserved_v1_tables'] = dict(metadata_sha256=digest(a.base/'metadata.json'),
        source_hashes=old['source_hashes'],table_hashes=old['table_hashes'])
    current['source_hash_scope'] = 'source_hashes describes newly built effect declarations; preserved_v1_tables describes inherited old tensors'
    (a.output_dir/'metadata.json').write_text(json.dumps(current,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(output=str(a.output_dir),preserved_v1=True,effect_table_sha256=current['table_hashes']['effect_descriptions'])))


if __name__ == '__main__': main()
