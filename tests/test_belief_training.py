"""Synthetic pipeline regression only; not evidence of learned duel strength."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class BeliefTrainingTest(unittest.TestCase):
    def test_offline_updates_checkpoint_and_heldout_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            data=root/'fixture.npz'
            np.savez(data,public_tokens=np.ones((4,2,3),dtype=np.float32),
                public_mask=np.ones((4,2),dtype=bool),hidden_slot_features=np.ones((4,2,2),dtype=np.float32),
                hidden_tokens=np.array([[1,2],[2,1],[1,2],[2,1]],dtype=np.int32),
                slot_mask=np.ones((4,2),dtype=bool),remaining_counts=np.tile([0,1,1],(4,1)),
                duel_ids=np.array(['duel1','duel1','duel2','duel2']))
            manifest=root/'producer.json'
            manifest.write_text(json.dumps(dict(schema='ygo-belief-dataset-v1',
                data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                information_assumption='known-deck-self-play',input_semantics='acting-player-legal-view',
                runtime_manifest_sha256='synthetic-test-not-engine-evidence')))
            subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/train_belief_head.py'),
                '--data',str(data),'--producer-manifest',str(manifest),'--output',str(root/'output'),
                '--updates','3','--batch-size','2','--width','8','--heads','2','--layers','1'],
                check=True,timeout=180,env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES=''))
            report=json.loads((root/'output/report.json').read_text())
            self.assertEqual(report['updates'],3)
            self.assertTrue(np.isfinite(report['nll']))
            self.assertEqual(report['constraint_valid_particle_rate'],1.)
            self.assertFalse(report['fair_play_promoted'])
            self.assertFalse(set(report['heldout_duels'])&set(report['train_duels']))
            self.assertEqual(report['checkpoint_sha256'],hashlib.sha256((root/'output/belief.flax_model').read_bytes()).hexdigest())


if __name__=='__main__': unittest.main()
