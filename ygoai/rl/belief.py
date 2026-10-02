"""Conservative, versioned known-deck information sets and belief particles.

This is a self-play fixture contract, not an unknown-deck opponent model. The
caller must construct it from legal history. No engine or privileged-state
handle is accepted by the sampler. Applying particles to live Lua/core state
requires a separate validated native backend and is deliberately not emulated.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import random


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class HiddenSlot:
    slot_id: str
    owner: int
    zone: str
    # A pool partitions main/extra decks and constraints from legal public history.
    pool: str
    allowed_cards: tuple[int,...] = ()


@dataclass(frozen=True)
class InformationSet:
    player: int
    observation_digest: str
    public_history_digest: str
    deck_prior_id: str
    hidden_slots: tuple[HiddenSlot,...]
    remaining_pools: tuple[tuple[str,tuple[int,...]],...]
    known_cards: tuple[tuple[str,int],...]
    assumption: str = 'known-deck-self-play'
    schema: str = 'ygo-information-set-v1'

    def validate(self):
        if self.schema != 'ygo-information-set-v1' or self.assumption != 'known-deck-self-play':
            raise ValueError('Only declared known-deck self-play fixtures are supported')
        if self.player not in (0,1) or not all((self.observation_digest,
                self.public_history_digest,self.deck_prior_id)):
            raise ValueError('Missing legal-view provenance')
        pools=dict(self.remaining_pools)
        known=dict(self.known_cards)
        slots=[s.slot_id for s in self.hidden_slots]
        if len(pools)!=len(self.remaining_pools) or len(known)!=len(self.known_cards):
            raise ValueError('Duplicate pool or known-card identity')
        if len(set(slots))!=len(slots) or set(slots)&set(known):
            raise ValueError('Hidden slots overlap or would overwrite a known card')
        if any(not isinstance(c,int) or isinstance(c,bool) or c<=0
               for c in list(known.values())+[c for cards in pools.values() for c in cards]):
            raise ValueError('Invalid card identifier')
        if any(s.owner not in (0,1) or s.zone not in ('hand','deck','extra','facedown')
               or not s.slot_id or s.pool not in pools for s in self.hidden_slots):
            raise ValueError('Invalid hidden slot')
        for name,cards in pools.items():
            if sum(s.pool==name for s in self.hidden_slots)!=len(cards):
                raise ValueError('Pool must account for every remaining hidden card')
        return self

    @property
    def identity(self):
        self.validate()
        return digest(asdict(self))


@dataclass(frozen=True)
class BeliefParticle:
    information_set_digest: str
    seed: int
    assignments: tuple[tuple[str,int],...]
    sampler: str = 'uniform-permutation-rejection-v1'
    schema: str = 'ygo-belief-particle-v1'

    @property
    def identity(self):
        return digest(asdict(self))

    def validate(self, view):
        view.validate()
        if self.schema!='ygo-belief-particle-v1' or self.information_set_digest!=view.identity:
            raise ValueError('Particle belongs to another information set')
        assigned=dict(self.assignments)
        if len(assigned)!=len(self.assignments) or set(assigned)!={s.slot_id for s in view.hidden_slots}:
            raise ValueError('Particle must assign exactly the unknown slots')
        for slot in view.hidden_slots:
            if slot.allowed_cards and assigned[slot.slot_id] not in slot.allowed_cards:
                raise ValueError('Particle violates a public slot constraint')
        for name,cards in view.remaining_pools:
            if Counter(cards)!=Counter(assigned[s.slot_id] for s in view.hidden_slots if s.pool==name):
                raise ValueError('Particle violates remaining card multiset')
        return self


def sample_uniform_particle(view, seed, *, max_attempts=1000):
    """Rejection over complete uniform permutations, not biased greedy placement."""
    view.validate()
    if max_attempts<1:
        raise ValueError('Positive sampling budget required')
    rng=random.Random(seed)
    for _ in range(max_attempts):
        assignments={}
        for name,cards in view.remaining_pools:
            shuffled=list(cards)
            rng.shuffle(shuffled)
            assignments.update(zip((s.slot_id for s in view.hidden_slots if s.pool==name),shuffled))
        if all(not s.allowed_cards or assignments[s.slot_id] in s.allowed_cards for s in view.hidden_slots):
            return BeliefParticle(view.identity,seed,tuple((s.slot_id,assignments[s.slot_id])
                                  for s in view.hidden_slots)).validate(view)
    raise ValueError('belief_constraint_sampling_budget')


class ParticleQueryGate:
    """A branch adapter must acknowledge application before any evaluator query.

    This enforces call order, not the correctness of a native implementation;
    only native differential/leakage fixtures can validate actual application.
    """
    def __init__(self, view, particle):
        particle.validate(view)
        self.view,self.particle=view,particle
        self.state='unapplied'

    def applied(self, *, particle_digest, observation_digest, known_cards):
        if self.state!='unapplied':
            raise RuntimeError('particle_application_out_of_order')
        if (particle_digest!=self.particle.identity or observation_digest!=self.view.observation_digest
                or tuple(known_cards)!=self.view.known_cards):
            raise RuntimeError('particle_application_identity_mismatch')
        self.state='applied'

    def require_query(self):
        if self.state!='applied':
            raise RuntimeError('query_before_particle_application_or_after_close')

    def close(self):
        self.state='closed'
