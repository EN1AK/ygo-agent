# Reality-anchored research execution

Read `docs/reality-anchored-learning.md` for the full ongoing objective and
the distinction between completing an experiment and improving game strength.

Freeze before inference (use an absent output directory):

```sh
python scripts/studies/reality_loop/run.py freeze \
  --frozen /data/frozen-local-goals-v2 --checkpoint /data/parent.flax_model \
  --output /data/reality-loop-20261028
python scripts/studies/reality_loop/run.py observe \
  --run /data/reality-loop-20261028 --checkpoint /data/parent.flax_model \
  --name round1
JAX_PLATFORMS=cpu python scripts/studies/reality_loop/recover.py \
  --run /data/reality-loop-20261028 --observation round1
```

Observation audits every autonomous rollout in an independent core, then selects
fit-only actual divergent prefixes. Recovery never drops the divergent action;
it searches and repeats complete goal-reaching continuations. In-process native
duels are sequential. Every certified route includes tensors and its original
history; only controlled-seat post-root choices are supervised. A successful
reference suffix is a witness, not proof of optimality. Budget exhaustion is
unknown. The old holdout is regression only and never contributes gradients.

The protocol and append-only evidence ledger live outside source in the new
run directory. Preserve failed attempts and production modules. Training,
repeated feedback, independent new transfer and full-duel/PPO gates remain
required by the project objective; the freeze/observe/search commands alone do
not claim that those gates have passed.
