import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ygoai.rl.decision_fixture import save_decision_fixture


class FixtureTests(unittest.TestCase):
    def test_roundtrip_and_exclusive_write(self):
        with tempfile.TemporaryDirectory() as folder:
            obs = {'actions_': np.zeros((1, 3, 12), dtype=np.uint8)}
            kw = dict(step=5, observation=obs, recurrent_leaves=[np.ones((1, 4))],
                      dones=np.array([False]), logits=np.ones((1, 3)),
                      probabilities=np.full((1, 3), 1/3), value=np.zeros((1, 1)),
                      selected_action=1, num_options=3, metadata={'checkpoint_sha256': 'fixture'})
            record = save_decision_fixture(folder, **kw)
            self.assertFalse(record['engine_snapshot'])
            self.assertFalse(record['counterfactual_labels_validated'])
            with np.load(Path(folder)/'step-000005.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['obs__actions_'], obs['actions_'])
                np.testing.assert_array_equal(saved['rnn__0'], np.ones((1, 4)))
            self.assertEqual(json.loads((Path(folder)/'step-000005.json').read_text())['num_options'], 3)
            with self.assertRaises(FileExistsError):
                save_decision_fixture(folder, **kw)
            kw['step'] = 6
            kw['value'] = np.array([[np.nan]])
            with self.assertRaises(ValueError):
                save_decision_fixture(folder, **kw)
            self.assertFalse((Path(folder)/'step-000006.npz').exists())
