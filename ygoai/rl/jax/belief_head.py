"""Separate autoregressive hidden-card head; never attached to actor weights.

Inputs are a caller-validated legal public/private information-set encoding and
declared remaining-card counts. Hidden identities are teacher labels only.
This module does not authorize using an opponent's actual private deck list.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp


class BeliefHead(nn.Module):
    vocabulary_size: int
    width: int = 64
    heads: int = 4
    layers: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, public_tokens, public_mask, hidden_slot_features,
                 hidden_tokens, remaining_counts, *, deterministic=True):
        """Token 0 is padding; teacher labels affect only later hidden slots."""
        if public_tokens.ndim != 3 or hidden_tokens.ndim != 2:
            raise ValueError('Expected batched public tokens and hidden labels')
        if public_tokens.shape[1] < 1 or hidden_tokens.shape[1] < 1:
            raise ValueError('Use a masked placeholder for an empty input sequence')
        if hidden_slot_features.shape[:2] != hidden_tokens.shape:
            raise ValueError('Hidden slot features and labels must align')
        if remaining_counts.shape != (hidden_tokens.shape[0], self.vocabulary_size):
            raise ValueError('Count vocabulary must match the belief head')
        context = nn.Dense(self.width, name='public_projection')(public_tokens)
        # A legal constant token prevents all-masked cross-attention in empty views.
        context = jnp.concatenate([jnp.zeros_like(context[:, :1]), context], axis=1)
        public_mask = jnp.concatenate([jnp.ones((public_mask.shape[0],1),dtype=bool), public_mask],axis=1)
        shifted = jnp.concatenate([jnp.zeros_like(hidden_tokens[:,:1]), hidden_tokens[:,:-1]],axis=1)
        x = nn.Embed(self.vocabulary_size,self.width,name='previous_cards')(shifted)
        x += nn.Dense(self.width,name='slot_projection')(hidden_slot_features)
        positions = self.param('position',nn.initializers.normal(.02),(hidden_tokens.shape[1],self.width))
        x += positions[None]
        causal = nn.make_causal_mask(hidden_tokens)
        cross_mask = nn.make_attention_mask(jnp.ones_like(hidden_tokens,dtype=bool),public_mask)
        for index in range(self.layers):
            norm = nn.LayerNorm(name=f'self_norm_{index}')(x)
            x += nn.MultiHeadDotProductAttention(self.heads,dropout_rate=self.dropout_rate,
                     name=f'causal_{index}')(norm,norm,mask=causal,deterministic=deterministic)
            norm = nn.LayerNorm(name=f'cross_norm_{index}')(x)
            x += nn.MultiHeadDotProductAttention(self.heads,dropout_rate=self.dropout_rate,
                     name=f'public_{index}')(norm,context,mask=cross_mask,deterministic=deterministic)
            norm = nn.LayerNorm(name=f'mlp_norm_{index}')(x)
            hidden = nn.gelu(nn.Dense(4*self.width,name=f'expand_{index}')(norm))
            hidden = nn.Dropout(self.dropout_rate)(hidden,deterministic=deterministic)
            x += nn.Dense(self.width,name=f'contract_{index}')(hidden)
        logits = nn.Dense(self.vocabulary_size,name='card_logits')(nn.LayerNorm()(x))
        # Only already emitted cards reduce availability; current/future labels
        # cannot change a prediction or its mask.
        used = jnp.cumsum(jax.nn.one_hot(shifted,self.vocabulary_size,dtype=jnp.int32),axis=1)
        available = remaining_counts[:,None,:] - used
        valid = (available > 0) & (jnp.arange(self.vocabulary_size)[None,None,:] != 0)
        return jnp.where(valid,logits,-1e9), valid


def belief_nll(logits, valid, targets, slot_mask):
    """Supervised loss with separate invalid-target count for fail-closed gates."""
    in_range = (targets >= 0) & (targets < logits.shape[-1])
    safe_targets = jnp.clip(targets,0,logits.shape[-1]-1)
    chosen_valid = jnp.take_along_axis(valid,safe_targets[...,None],axis=-1)[...,0] & in_range
    logp = jax.nn.log_softmax(logits,axis=-1)
    selected = jnp.take_along_axis(logp,safe_targets[...,None],axis=-1)[...,0]
    loss = -jnp.sum(jnp.where(slot_mask,selected,0.0))/jnp.maximum(jnp.sum(slot_mask),1)
    invalid = jnp.sum(slot_mask & ~chosen_valid)
    return loss, invalid


def sample_belief(model, variables, public_tokens, public_mask, hidden_slot_features,
                  remaining_counts, key, *, slot_mask=None):
    """Autoregressively sample without replacement; exhausted rows fail closed."""
    batch, slots = hidden_slot_features.shape[:2]
    tokens = jnp.zeros((batch,slots),dtype=jnp.int32)
    valid_rows = jnp.ones(batch,dtype=bool)
    if slot_mask is None:
        slot_mask = jnp.ones((batch,slots),dtype=bool)

    def draw(index, carry):
        tokens, valid_rows, key = carry
        key, draw_key = jax.random.split(key)
        logits, valid = model.apply(variables,public_tokens,public_mask,
                                    hidden_slot_features,tokens,remaining_counts)
        available = jnp.any(valid[:,index,:],axis=-1)
        chosen = jax.random.categorical(draw_key,logits[:,index,:],axis=-1)
        active = slot_mask[:,index]
        chosen = jnp.where(available & valid_rows & active,chosen,0)
        tokens = tokens.at[:,index].set(chosen)
        return tokens, valid_rows & (available | ~active), key

    tokens, valid_rows, _ = jax.lax.fori_loop(0,slots,draw,(tokens,valid_rows,key))
    return tokens, valid_rows
