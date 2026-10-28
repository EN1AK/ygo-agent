"""Safety checks for evidence admission rather than model-score snapshots."""
import json
from pathlib import Path
import tempfile
import unittest
from contracts import diagnose, first_deviation, ledger, validate_splits


class ContractsTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
