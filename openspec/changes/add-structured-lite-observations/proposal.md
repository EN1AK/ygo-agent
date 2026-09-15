## Why

The current agent represents cards and legal actions, but it weakly represents the causal relationships between an active chain effect, its source card, candidate responses, targets, costs, and resulting public events. This makes semantically different actions look too similar and contributes to errors such as spending Effect Veiler on a monster while failing to interrupt the continuous-spell effect that actually drives the opponent's play.

## What Changes

- Add a versioned Structured-lite observation path containing selection context, structured legal-action fields, role-aware card references, and bounded public chain/event history.
- Add deterministic CDB-derived card attributes and a small audited set of rule/API effect tags without requiring online text models or full Lua semantic execution.
- Update the model encoder so each legal action reads its referenced cards and current public chain context before the existing FiLM policy head scores it.
- Preserve partial-observation boundaries and separate per-player recurrent memory.
- Introduce an explicit observation/model version boundary and checkpoint migration policy; existing checkpoints remain runnable on the legacy schema but are not silently loaded into incompatible Structured-lite shapes.
- Add feature coverage, information-leakage, snapshot regression, combo, throughput, and head-to-head promotion evaluations.
- Keep terminal game return as the value target. Local interaction outcomes are diagnostic features and regression assertions, not replacement rewards.

## Capabilities

### New Capabilities

- `structured-lite-observations`: Versioned environment and model interfaces for relationship-aware state, action, and public-event features, including compatibility and evaluation requirements.

### Modified Capabilities

None.

## Impact

- Environment observation construction in `ygoenv/ygoenv/ygopro/ygopro.h` and its Python-facing state schema.
- JAX model encoding in `ygoai/rl/jax/agent.py`, model arguments, training launch configuration, and checkpoint metadata/loading.
- Training and evaluation scripts that construct environments or models.
- New offline semantic-table builder artifacts derived from the existing card database and card scripts.
- New diagnostic reports and regression fixtures, beginning with the validated Effect Veiler versus Ghost Ogre snapshot.
- Increased observation bandwidth and encoder compute; the design must measure GPU throughput and memory before promotion.
