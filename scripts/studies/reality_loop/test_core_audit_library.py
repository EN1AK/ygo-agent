import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from audit_core_replay import resolve_library
from ygoai.rl.core_audit import validate_curriculum_binding,derive_curriculum_binding


class AuditLibraryTests(unittest.TestCase):
    def test_added_repair_gets_a_fresh_proof_without_dropping_parent_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'initial.json';source.write_text('{}')
            sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            rows=[]
            for name in ('old','repair'):
                record=root/(name+'.json');record.write_text('{}');arrays=record.with_suffix('.npz');arrays.write_bytes(b'inputs')
                rows.append(dict(id=name,path=str(record),record_sha256=sha(record),observations_sha256=sha(arrays)))
            audit=dict(passed=True,independent_library_sha256='core-sha')
            parent=dict(independent_library=dict(path='core',sha256='core-sha'),native_sha256='native',
                rows=rows[:1],runtime=str(root),source_root=str(root))
            proof=root/'parent-admission.json'
            proof.write_text(json.dumps(dict(passed=True,source_plan=str(source),source_plan_sha256=sha(source),
                independent_library=parent['independent_library'],native_sha256='native',teacher_data_changed=False,
                rows=[dict(id='old',record_sha256=rows[0]['record_sha256'],audit=audit)])))
            parent['independent_compatibility_admission']=dict(path=str(proof),sha256=sha(proof))
            parent_path=root/'parent.json';parent_path.write_text(json.dumps(parent))
            child=dict(parent,rows=rows);output=root/'derived.json'
            with patch('ygoai.rl.core_audit.audit_record',return_value=audit) as replay:
                derive_curriculum_binding(child,parent_path,output)
                self.assertEqual(replay.call_count,1)
            validate_curriculum_binding(child)
            result=json.loads(output.read_text())
            self.assertEqual((result['inherited_rows'],result['newly_replayed_rows']),(1,1))
            self.assertEqual([r['id'] for r in result['rows']],['old','repair'])
            Path(rows[1]['path']).write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'input changed'):
                derive_curriculum_binding(dict(parent,rows=rows),parent_path,root/'rejected.json')

    def test_backend_revision_requires_all_frozen_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.json';proof=root/'admission.json'
            source.write_text('{}')
            sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            row=dict(id='a',record_sha256='record')
            plan=dict(independent_library=dict(path='core',sha256='core-sha'),native_sha256='native',rows=[row])
            with self.assertRaisesRegex(ValueError,'lacks curriculum admission'):validate_curriculum_binding(plan)
            admission=dict(passed=True,source_plan=str(source),source_plan_sha256=sha(source),
                independent_library=plan['independent_library'],native_sha256='native',teacher_data_changed=False,
                rows=[dict(row,audit=dict(passed=True,independent_library_sha256='core-sha'))])
            proof.write_text(json.dumps(admission))
            plan['independent_compatibility_admission']=dict(path=str(proof),sha256=sha(proof))
            validate_curriculum_binding(plan)
            plan['rows'].append(dict(id='b',record_sha256='other'))
            with self.assertRaisesRegex(ValueError,'does not cover'):validate_curriculum_binding(plan)

    def test_historical_default_is_preserved(self):
        self.assertEqual(resolve_library({},Path('/source')),Path('/source/resources/libocgcore-current-v2.so'))

    def test_changed_library_or_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);library=root/'core.so';manifest=root/'manifest.json'
            library.write_bytes(b'isolated-library')
            sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(dict(schema='independent-actor-random-select-v1',
                library_sha256=sha(library),changed_files=['libgroup.cpp'],model_native_changed=False,load_passed=True)))
            binding=dict(path=str(library),sha256=sha(library),manifest=str(manifest),manifest_sha256=sha(manifest))
            self.assertEqual(resolve_library(dict(library=binding),root),library)
            library.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'binding changed'):resolve_library(dict(library=binding),root)
            library.write_bytes(b'isolated-library');manifest.write_text('{}')
            with self.assertRaisesRegex(ValueError,'binding changed'):resolve_library(dict(library=binding),root)


if __name__=='__main__':unittest.main()
