import hashlib,json,tempfile,unittest
from pathlib import Path
from ygoai.rl.shadow_gate import validate_shadow_baseline

class ContinuationGateTest(unittest.TestCase):
    def test_explicit_continuation_and_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);a=p/'actor';q=p/'q';m=p/'manifest'
            a.write_bytes(b'actor');q.write_bytes(b'q')
            sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
            data=dict(gate='bounded-shadow-start-v1',status='passed',actor_estimator='frozen',
                promotion_allowed=False,max_new_steps=1744896,allowed_actor_sha256=[sha(a)],
                strategy_metrics={'scope':'diagnostic'},artifacts=[{'path':str(q),'sha256':sha(q)}])
            m.write_text(json.dumps(data))
            with self.assertRaises(ValueError):validate_shadow_baseline(m,a,1744896,frozen_actor=True)
            data['frozen_critic_continuation']=dict(authorization='user-2m-20261004',start_step=262144,q_checkpoint_sha256=sha(q))
            m.write_text(json.dumps(data))
            validate_shadow_baseline(m,a,1744896,frozen_actor=True,resume_checkpoint=q,cumulative_offset=262144)
            for steps,offset in [(1744897,262144),(1744896,0)]:
                with self.assertRaises(ValueError):validate_shadow_baseline(m,a,steps,frozen_actor=True,resume_checkpoint=q,cumulative_offset=offset)
            q.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_shadow_baseline(m,a,1744896,frozen_actor=True,resume_checkpoint=q,cumulative_offset=262144)
