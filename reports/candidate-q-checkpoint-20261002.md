# Candidate-Q checkpoint boundary validation

This is preparation for the post-100M experiment, **not a VRPO training run**.
The active PPO training path, native module and Lua assets are unchanged by
this change. The old 40M starting-point plan has been superseded by the user's
100M-then-VRPO order; the baseline gate is reopened until the endpoint,
evaluation, combo and interruption evidence actually exists.

## Implemented boundary

- Version-2 Q envelopes retain actor variables, actor optimizer, separate
  critic variables and critic optimizer. Every context field and file hash
  must match before deserialization. Malformed/missing JSON, a partial resume
  template, and mismatched array shape/dtype fail closed.
- Old PPO import remains explicit, importing actor weights only; it does not
  claim to restore experimental optimizer or Q state.
- `export_ppo_actor_checkpoint` validates the entire Q envelope and writes only
  actor variables with the existing PPO metadata format. It refuses existing
  targets. No Q state, privileged critic input or sampler state is exported.
- The normal evaluator and default trainer remain unchanged. Non-off Q modes
  still refuse training until their actual collection/update paths are wired.

## Verified evidence

Home WSL isolated directory `/home/ygo/vrpo-checkpoint-20261002-v1`, CPU only,
JAX 0.6.2 / Flax 0.10.7:

- **21 tests passed in 34.620 seconds**: checkpoint import/export and rejection,
  Adam moments/counter restore with identical next updates, unchanged unflagged
  inference/GAE/checkpoint bytes, separate critic update, menu alignment and
  deterministic two-seat Q target fixtures.
- Three Q modes export exactly the original actor bytes in fixtures.
- Real checkpoint `2102026_step_000053002240.flax_model` completed explicit PPO
  import -> synthetic Q envelope -> complete state restore -> actor-only export
  -> unchanged legacy reader. Input and exported bytes have identical SHA-256:
  `efaa6448201e49778b1604f56efa294bd05c1518861aed30bdf55051078c6fda`.
- The actual-weight validator uses a **synthetic serialization-only critic**,
  not a trained Q network. Its report explicitly records this limitation and
  `q_training_performed=false`. It does not advance the shadow calibration or
  centralized leakage gates, nor change the designated 100M experiment start.
- The installed CUDA plugin prints `CUDA_ERROR_NO_DEVICE` during discovery on
  CPU-only home; tests continue on the explicitly selected CPU backend and pass.

Local evidence bundle: `training-runs/q-checkpoint-evidence-20261002.tar.gz`.
It contains the complete test log, hashes of the tested helper/test/validator,
and the real-weight round-trip report. The validation context labels its
source honestly as `92f225c-with-checkpoint-helper-patch`; the tested file hashes
identify the patch before its final commit.

Reproduce with the five Candidate-Q unittest modules and
`scripts/validate_candidate_q_checkpoint.py --help`. Use an isolated CPU
environment, a checksum-valid PPO sidecar, and a fresh output directory.

## Still pending

The 100M endpoint and complete baseline evaluation, real shadow collection,
held-out Q calibration, recurrent learner integration, the learner-only
centralized state channel and matched GAE/Q pilots remain incomplete.
Checkpoint serialization does not establish any of those capabilities.
