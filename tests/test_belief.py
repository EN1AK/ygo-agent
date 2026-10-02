from dataclasses import replace
import unittest

from ygoai.rl.belief import (BeliefParticle, HiddenSlot, InformationSet,
                            ParticleQueryGate, sample_uniform_particle)


class BeliefTest(unittest.TestCase):
    def view(self):
        return InformationSet(0,'legal-view','public-events','fixture-public-decklist',
            (HiddenSlot('opp-hand-0',1,'hand','main',(1,)),
             HiddenSlot('opp-deck-0',1,'deck','main'),
             HiddenSlot('opp-deck-1',1,'deck','main')),
            (('main',(1,2,2)),),(('own-hand-0',42),('revealed-field-0',51)))

    def test_seeded_sampling_constraints_and_reproducible_hash(self):
        view=self.view()
        a=sample_uniform_particle(view,91)
        b=sample_uniform_particle(view,91)
        self.assertEqual(a.identity,b.identity)
        self.assertEqual(dict(a.assignments)['opp-hand-0'],1)
        self.assertEqual(sorted(dict(a.assignments).values()),[1,2,2])
        self.assertNotIn('own-hand-0',dict(a.assignments))

    def test_invalid_information_set_and_particle_rejected(self):
        view=self.view()
        for invalid in (replace(view,assumption='unknown-deck'),
                replace(view,known_cards=(('opp-hand-0',1),)),
                replace(view,remaining_pools=(('main',(1,2)),))):
            with self.assertRaises(ValueError): invalid.validate()
        a=sample_uniform_particle(view,91)
        with self.assertRaises(ValueError):
            replace(a,assignments=(('opp-hand-0',2),('opp-deck-0',1),('opp-deck-1',2))).validate(view)
        with self.assertRaises(ValueError):
            a.validate(replace(view,public_history_digest='other-history'))

    def test_unsatisfiable_constraint_fails_boundedly(self):
        view=self.view()
        impossible=replace(view,hidden_slots=(replace(view.hidden_slots[0],allowed_cards=(99,)),)+view.hidden_slots[1:])
        with self.assertRaisesRegex(ValueError,'budget'):
            sample_uniform_particle(impossible,1,max_attempts=3)

    def test_query_guard_rejects_unapplied_and_closed_branch(self):
        view=self.view()
        p=sample_uniform_particle(view,91)
        gate=ParticleQueryGate(view,p)
        with self.assertRaises(RuntimeError): gate.require_query()
        with self.assertRaises(RuntimeError):
            gate.applied(particle_digest='wrong',observation_digest=view.observation_digest,known_cards=view.known_cards)
        gate.applied(particle_digest=p.identity,observation_digest=view.observation_digest,known_cards=view.known_cards)
        gate.require_query()
        with self.assertRaises(RuntimeError):
            gate.applied(particle_digest=p.identity,observation_digest=view.observation_digest,known_cards=view.known_cards)
        gate.close()
        with self.assertRaises(RuntimeError): gate.require_query()


if __name__=='__main__': unittest.main()
