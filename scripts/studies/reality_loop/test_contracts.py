"""Safety checks for evidence admission rather than model-score snapshots."""
import json
from pathlib import Path
import tempfile
import unittest
from contracts import diagnose, first_deviation, ledger, validate_splits
from contracts import read, sha, write
from contracts import audit_opponent_policy,round_protocol
from train_pair import prepare
from upgrade_inputs import assert_allowed_difference, expected_counts
from duel_exam import natural_definition,stopping_status
from ygoai.rl.exercise_actor import native_seed32
from endpoint_exams import freeze as freeze_endpoint_exams
from endpoint_exams import execute as execute_endpoint_exams
from types import SimpleNamespace


class ContractsTest(unittest.TestCase):
    def test_full_uint32_seed_range_crosses_signed_actor_abi_without_changing_bits(self):
        for seed,expected in [(0,0),(2147483647,2147483647),(2147483648,-2147483648),
                              (2479374650,-1815592646),(4294967295,-1)]:
            self.assertEqual(native_seed32(seed),expected)
        for seed in [True,1.5,4294967296,-2147483649]:
            with self.assertRaises(ValueError):native_seed32(seed)

    def test_native_step_cutoff_is_unknown_and_real_terminal_stays_terminal(self):
        budget=dict(decisions=2048,seconds_per_game=120)
        self.assertEqual(stopping_status(dict(done=True,invalid=True),2048,budget,30),'budget_exhausted')
        self.assertEqual(stopping_status(dict(done=True,invalid=False),2048,budget,30),'natural_terminal')
        self.assertEqual(stopping_status(dict(done=True,invalid=True),20,budget,30),'invalid')
        self.assertIsNone(stopping_status(dict(done=False,invalid=False),20,budget,30))

    def test_endpoint_protocol_predeclares_terminal_models_without_exam_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root, checkpoint = self.pair_fixture(folder)
            prepare(root,'round1',checkpoint)
            independent=root/'untouched-exam';independent.mkdir()
            write(independent/'plan.json',dict(rows=[dict(split='holdout',family='unseen')]))
            protocol=read(root/'protocol.json')
            protocol.update(initial_checkpoint=str(checkpoint),initial_checkpoint_sha256=sha(checkpoint),
                independent_exam=dict(path=str(independent),plan_sha256=sha(independent/'plan.json')))
            write(root/'protocol.json',protocol)
            freeze_endpoint_exams(SimpleNamespace(run=root,name='endpoint-exams',rounds=['round1']))
            p=read(root/'endpoint-exams/protocol.json')
            self.assertEqual([r['name'] for r in p['endpoints']],['parent','control','repair'])
            self.assertTrue(p['fixed_before_exam_inference'])
            self.assertFalse(p['independent_score_checkpoint_selection'])
            self.assertEqual(p['natural_duels']['planned_per_endpoint'],256)
            self.assertTrue(all(not Path(r['checkpoint']).exists() for r in p['endpoints'][1:]))
            with self.assertRaisesRegex(ValueError,'PPO protocol'):
                execute_endpoint_exams(SimpleNamespace(run=root,name='endpoint-exams',phase='independent'))

    def test_mirror_openings_preserve_decks_and_reproduce_paired_seeds(self):
        main = list(range(40))
        extra = [100,101]
        first = natural_definition(main,extra,327680,123)
        self.assertEqual(first,natural_definition(main,extra,327680,123))
        self.assertNotEqual(first,natural_definition(main,extra,327680,124))
        for seat in (0,1):
            self.assertEqual(sorted(c[0] for c in first['initial'] if c[1:3]==[seat,1]),main)
            self.assertEqual([c[0] for c in first['initial'] if c[1:3]==[seat,64]],extra)
        self.assertTrue(all(c[2] in (1,64) for c in first['initial']))
        self.assertEqual(len(first['seed_sequence']),8)

    def test_legal_opponent_activation_is_not_a_passive_teacher(self):
        record = dict(definition=dict(controlled=1), decisions=[dict(player=0, action=0,
            menu=[dict(act='Activate'), dict(act='Cancel')])])
        with self.assertRaisesRegex(ValueError, 'opponent-policy violation'):
            audit_opponent_policy(record, 0)
        record['decisions'][0]['action'] = 1
        self.assertTrue(audit_opponent_policy(record, 0)['passed'])
    def test_raw_select_unselect_counts_include_selected_pool(self):
        # player=0, finish=1, cancel=0, min=1, max=3; two selectable
        # cards and one selected card. Each protocol card entry is 8 bytes.
        raw = bytes([26, 0, 1, 0, 1, 3, 2]) + bytes(16) + bytes([1]) + bytes(8)
        counts = expected_counts([dict(op=26, raw=raw.hex())])
        self.assertEqual(counts, dict(minimum=1, maximum=3, selected_count=1, must_continue=0))

    def test_input_upgrade_only_changes_declared_public_count_fields(self):
        import numpy as np
        original = dict(selection_=np.zeros((1, 16), np.uint8), cards_=np.zeros((1, 2), np.uint8))
        fixed = {k: v.copy() for k, v in original.items()}
        fixed['selection_'][0, [5, 6, 8, 11]] = [1, 3, 2, 1]
        assert_allowed_difference(original, fixed, 26)
        with self.assertRaisesRegex(ValueError, 'unexpected input change'):
            assert_allowed_difference(original, fixed, 15)
        fixed['cards_'][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'unexpected input change'):
            assert_allowed_difference(original, fixed, 26)

    def test_connected_family_cannot_cross_split(self):
        with self.assertRaises(ValueError):
            validate_splits([dict(split='fit', family='p', source_sha256='a'),
                             dict(split='holdout', family='p', source_sha256='b')])

    def test_failure_prefix_keeps_executed_deviation(self):
        decisions = [dict(player=0, menu=['a', 'b'], observation_sha256='x', action=0)]
        reference = dict(decisions=decisions)
        trace = dict(decisions=[dict(decisions[0], action=1, actor='model')])
        self.assertEqual(first_deviation(trace, reference), 1)
        trace['decisions'][0]['observation_sha256'] = 'different root'
        self.assertIsNone(first_deviation(trace, reference))

    def test_cycle_diagnostic_does_not_create_negative_labels(self):
        decisions = []
        for i in range(6):
            decisions.append(dict(step=i, actor='model', message=26,
                observation_sha256=str(i % 2), action=0,
                menu=[dict(code=1, spec='m1', unselect=bool(i % 2))]))
        result = diagnose(dict(status='budget_exhausted', decisions=decisions))
        self.assertEqual(result['classification'], 'selection_cycle_candidate')
        self.assertFalse(result['wrong_action_label'])

    def test_ledger_rejects_tampered_prior_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = Path(folder) / 'evidence.json'
            evidence.write_text('{}', encoding='utf-8')
            ledger(folder, 'observation', [evidence], count=1)
            path = Path(folder) / 'ledger.jsonl'
            row = json.loads(path.read_text(encoding='utf-8'))
            row['count'] = 99
            path.write_text(json.dumps(row) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                ledger(folder, 'finding', [evidence])

    def pair_fixture(self, folder, bad_prefix=False):
        root = Path(folder)
        original = root / 'original'
        original.mkdir()
        plan = dict(rows=[dict(id='original', path='original.json', split='fit', family='f', source_sha256='s'),
                          dict(id='exam', path='exam.json', split='holdout', family='h', source_sha256='t')],
                    stages=[dict(mix={'foundation': 1})], evaluation={})
        write(original / 'plan.json', plan)
        checkpoint = root / 'parent'
        checkpoint.write_bytes(b'weights')
        write(root / 'protocol.json', dict(original=str(original), original_plan_sha256=sha(original / 'plan.json'),
              initial_checkpoint=str(checkpoint),initial_checkpoint_sha256=sha(checkpoint),rounds=2,
              training=dict(repair_mass=.5, updates_per_arm=512, learning_rate=2e-5)))
        observation = root / 'round1/evaluation'
        observation.mkdir(parents=True)
        write(observation / 'summary.json', dict(checkpoint_sha256=sha(checkpoint)))
        recovery = root / 'round1-recovery'
        recovery.mkdir()
        path = recovery / 'repair.json'
        write(path, dict(training_ready=True, independent_endpoint_audit=dict(passed=True),
                         definition=dict(controlled=1),
                         decisions=[dict(teaching=bad_prefix, player=1), dict(teaching=True, player=1)]))
        path.with_suffix('.npz').write_bytes(b'tensor evidence')
        repair = dict(id='x', path=str(path), split='fit', family='f', source_sha256='s',
                      root_step=1, record_sha256=sha(path), observations_sha256=sha(path.with_suffix('.npz')))
        write(recovery / 'summary.json', dict(admitted=[repair]))
        return root, checkpoint

    def test_control_has_identical_exam_but_no_repair_gradients(self):
        with tempfile.TemporaryDirectory() as folder:
            root, checkpoint = self.pair_fixture(folder)
            prepared = prepare(root, 'round1', checkpoint)
            control = read(prepared / 'control-inputs/plan.json')
            candidate = read(prepared / 'repair-inputs/plan.json')
            self.assertEqual(control['rows'], candidate['rows'])
            self.assertNotIn('repair', control['stages'][0]['mix'])
            self.assertEqual(candidate['stages'][0]['mix']['repair'], .5)
            self.assertEqual(control['parent_sha256'], candidate['parent_sha256'])

    def test_failed_prefix_cannot_be_relabelled_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root, checkpoint = self.pair_fixture(folder, bad_prefix=True)
            with self.assertRaisesRegex(ValueError, 'failed prefix'):
                prepare(root, 'round1', checkpoint)

    def test_next_round_keeps_old_repairs_out_of_new_treatment_group(self):
        with tempfile.TemporaryDirectory() as folder:
            root, checkpoint = self.pair_fixture(folder)
            original = root / 'original/plan.json'
            plan = read(original)
            plan['stages'][0]['mix'] = dict(foundation=.5, repair=.5)
            write(original, plan)
            protocol = read(root / 'protocol.json')
            protocol['original_plan_sha256'] = sha(original)
            write(root / 'protocol.json', protocol)
            prepared = prepare(root, 'round1', checkpoint)
            control = read(prepared / 'control-inputs/plan.json')['stages'][0]['mix']
            candidate = read(prepared / 'repair-inputs/plan.json')['stages'][0]['mix']
            self.assertEqual(control, dict(foundation=.5, repair=.5))
            self.assertEqual(candidate, dict(foundation=.25, repair=.25, **{'repair-round1': .5}))

    def test_second_round_cannot_silently_reuse_initial_policy_context(self):
        with tempfile.TemporaryDirectory() as folder:
            root,checkpoint=self.pair_fixture(folder)
            with self.assertRaisesRegex(ValueError,'updated-policy context'):
                round_protocol(root,'round2')
            write(root/'round2-context.json',dict(base_protocol_sha256=sha(root/'protocol.json'),
                original=str(root/'original'),original_plan_sha256=sha(root/'original/plan.json'),
                initial_checkpoint=str(checkpoint),initial_checkpoint_sha256=sha(checkpoint)))
            self.assertEqual(round_protocol(root,'round2')['initial_checkpoint_sha256'],sha(checkpoint))
            checkpoint.write_bytes(b'changed policy')
            with self.assertRaisesRegex(ValueError,'round context changed'):
                round_protocol(root,'round2')


if __name__ == '__main__':
    unittest.main()
