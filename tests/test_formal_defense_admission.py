import json
from pathlib import Path
import tempfile
import unittest
from scripts.run_elfnote_formal import resolve_start,verify_preflight,sha


class DefenseAdmissionTests(unittest.TestCase):
    def test_admitted_endpoint_requires_exact_hash_and_real_producer_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';(study/'teach').mkdir(parents=True)
            checkpoint=study/'teach/model.flax_model';checkpoint.write_bytes(b'admitted')
            plan=dict(course_checkpoint_sha256='parent',arms=dict(after_defense=dict(study='study',arm='teach',checkpoint_sha256=sha(checkpoint))))
            (study/'manifest.json').write_text(json.dumps(dict(plan=dict(parent_sha256='parent'))))
            (study/'final-verification.json').write_text(json.dumps(dict(passed=True,defensive_policy_gate=True,checkpoints=dict(teach=sha(checkpoint)))))
            exitfile=Path(str(study)+'.train.exit');exitfile.write_text('1')
            with self.assertRaises(AssertionError):resolve_start(root,plan,root/'release','after_defense')
            exitfile.write_text('0');self.assertEqual(resolve_start(root,plan,root/'release','after_defense'),checkpoint)
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(AssertionError):resolve_start(root,plan,root/'release','after_defense')

    def test_smoke_is_bound_to_endpoint_arm_plan_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'smoke';folder.mkdir()
            plan=root/'plan.json';plan.write_text(json.dumps(dict(smoke_timesteps=20480)))
            runtime=root/'runtime.json';runtime.write_text('{}');checkpoint=root/'model';checkpoint.write_bytes(b'endpoint')
            proof=dict(passed=True,process_exit_zero=True,zero_nonfinite_updates=True,fitted_env_steps=20480)
            launch=dict(phase='preflight',arm='after_defense',plan_sha256=sha(plan),runtime_manifest_sha256=sha(runtime),checkpoint_sha256=sha(checkpoint))
            (folder/'verification.json').write_text(json.dumps(proof));path=folder/'launch-plan.json';path.write_text(json.dumps(launch))
            verify_preflight(folder,plan,runtime,checkpoint,'after_defense')
            for key in ['arm','plan_sha256','runtime_manifest_sha256','checkpoint_sha256']:
                broken=dict(launch);broken[key]='wrong';path.write_text(json.dumps(broken))
                with self.assertRaises(AssertionError):verify_preflight(folder,plan,runtime,checkpoint,'after_defense')


if __name__=='__main__':unittest.main()
