import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from audit_core_replay import resolve_library


class AuditLibraryTests(unittest.TestCase):
    def test_historical_default_is_preserved(self):
        self.assertEqual(resolve_library({},Path('/source')),Path('/source/resources/libocgcore-current-v2.so'))

    def test_changed_library_or_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);library=root/'core.so';manifest=root/'manifest.json'
            library.write_bytes(b'isolated-library')
            sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(dict(schema='independent-actor-random-select-v1',
                library_sha256=sha(library),changed_files=['libgroup.cpp'],model_native_changed=False)))
            binding=dict(path=str(library),sha256=sha(library),manifest=str(manifest),manifest_sha256=sha(manifest))
            self.assertEqual(resolve_library(dict(library=binding),root),library)
            library.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'binding changed'):resolve_library(dict(library=binding),root)
            library.write_bytes(b'isolated-library');manifest.write_text('{}')
            with self.assertRaisesRegex(ValueError,'binding changed'):resolve_library(dict(library=binding),root)


if __name__=='__main__':unittest.main()
