import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ygoai.rl.jax.belief_head import BeliefHead, belief_nll, sample_belief


class BeliefHeadTest(unittest.TestCase):
    def setUp(self):
        self.model=BeliefHead(5,width=16,heads=2,layers=1,dropout_rate=0.)
        self.public=jnp.ones((1,3,4))
        self.mask=jnp.array([[True,True,False]])
        self.slots=jnp.ones((1,3,2))
        self.labels=jnp.array([[1,2,3]])
        self.counts=jnp.array([[0,1,1,1,0]])
        self.variables=self.model.init(jax.random.PRNGKey(1),self.public,self.mask,
                                       self.slots,self.labels,self.counts)

    def forward(self,labels=None,public=None):
        return self.model.apply(self.variables,self.public if public is None else public,
                                self.mask,self.slots,self.labels if labels is None else labels,self.counts)

    def test_future_labels_and_masked_public_tokens_cannot_leak(self):
        original,valid=self.forward()
        changed,changed_valid=self.forward(labels=jnp.array([[1,3,2]]))
        np.testing.assert_array_equal(original[:,:2],changed[:,:2])
        np.testing.assert_array_equal(valid[:,:2],changed_valid[:,:2])
        masked,_=self.forward(public=self.public.at[:,2,:].set(900.))
        np.testing.assert_allclose(original,masked,rtol=0,atol=0)

    def test_count_masks_loss_and_gradient(self):
        logits,valid=self.forward()
        self.assertFalse(bool(valid[0,1,1]))
        self.assertFalse(bool(valid[0,2,2]))
        loss,invalid=belief_nll(logits,valid,self.labels,jnp.ones_like(self.labels,dtype=bool))
        self.assertEqual(int(invalid),0)
        self.assertTrue(np.isfinite(float(loss)))
        def objective(variables):
            values,mask=self.model.apply(variables,self.public,self.mask,self.slots,self.labels,self.counts)
            return belief_nll(values,mask,self.labels,jnp.ones_like(self.labels,dtype=bool))[0]
        grads=jax.grad(objective)(self.variables)
        self.assertTrue(all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(grads)))
        _,invalid=belief_nll(logits,valid,jnp.array([[1,1,99]]),jnp.ones_like(self.labels,dtype=bool))
        self.assertEqual(int(invalid),2)

    def test_seeded_sampling_preserves_multiset_and_reports_exhaustion(self):
        def sample(counts):
            return sample_belief(self.model,self.variables,self.public,self.mask,self.slots,
                                 counts,jax.random.PRNGKey(2))
        tokens,valid=sample(self.counts)
        self.assertTrue(bool(valid[0]))
        self.assertEqual(sorted(np.asarray(tokens[0]).tolist()),[1,2,3])
        np.testing.assert_array_equal(tokens,sample(self.counts)[0])
        _,valid=sample(jnp.array([[0,1,0,0,0]]))
        self.assertFalse(bool(valid[0]))
        tokens,valid=sample_belief(self.model,self.variables,self.public,self.mask,self.slots,
            jnp.array([[0,1,0,0,0]]),jax.random.PRNGKey(2),slot_mask=jnp.array([[True,False,False]]))
        self.assertTrue(bool(valid[0]))
        np.testing.assert_array_equal(tokens,[[1,0,0]])


if __name__ == '__main__':
    unittest.main()
