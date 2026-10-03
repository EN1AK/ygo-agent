"""Fail closed if any actor parameter, optimizer field or step changes."""
import hashlib
import flax.serialization


class FrozenActorGuard:
    def __init__(self, state):
        self.reference = flax.serialization.to_bytes(state)
        self.sha256 = hashlib.sha256(self.reference).hexdigest()

    def verify(self, state):
        actual = flax.serialization.to_bytes(state)
        if actual != self.reference:
            raise AssertionError('frozen actor parameters/optimizer/step changed')
        return {'exact': True, 'actor_state_sha256': self.sha256,
                'scope': 'serialized actor parameters, batch stats, optimizer and step'}
