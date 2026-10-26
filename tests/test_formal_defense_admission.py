import json
from pathlib import Path
import tempfile
import unittest
from scripts.run_elfnote_formal import resolve_start,verify_preflight,sha
from scripts.run_elfnote_formal_evaluation import verified_endpoint


class DefenseAdmissionTests(unittest.TestCase):
    def test_comparison_requires_same_runtime_budget_and_exit_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'plain';folder.mkdir();checkpoint=root/'model';checkpoint.write_bytes(b'ppo')
            keys=['seed','total_timesteps_per_arm','local_num_envs','num_actor_threads','local_env_threads','num_steps','num_minibatches','learning_rate','update_epochs']
            plan={k:1 for k in keys};proof=dict(passed=True,process_exit_zero=True,zero_nonfinite_updates=True,fitted_env_steps=1,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
            launch=dict(phase='train',arm='plain',runtime_manifest_sha256='runtime',plan=plan)
            (folder/'verification.json').write_text(json.dumps(proof));(folder/'launch-plan.json').write_text(json.dumps(launch));(folder/'train.exit.json').write_text('{"returncode":0}')
            self.assertEqual(verified_endpoint(root,'plain',plan,'runtime')[0],checkpoint)
            with self.assertRaises(AssertionError):verified_endpoint(root,'plain',plan,'other-runtime')
            different=dict(plan);different['seed']=2
            with self.assertRaises(AssertionError):verified_endpoint(root,'plain',different,'runtime')
            (folder/'train.exit.json').write_text('{"returncode":1}')
            with self.assertRaises(AssertionError):verified_endpoint(root,'plain',plan,'runtime')

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
