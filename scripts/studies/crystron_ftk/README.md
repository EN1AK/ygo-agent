# Crystron replay curriculum study

These are the scripts used by the isolated 2026-10-27 study. Copy them into
`<deployment>/training-runs/crystron-ftk-20261026/` before running; they resolve
`<deployment>/ygo-agent` as their source snapshot. Do not run them in place in
this source directory. Runtime assets, demonstrations, proof files, checkpoints,
and frozen experiment plans live in the study data/evidence bundles, not Git.

The input is the independently audited 110-replay `actor-converted-v4` corpus
and the `actor-runtime-v3` full-legal-menu adapter. Freeze the original
four-fit/four-holdout curriculum with `freeze_learning.py`; run
`launch_learning.sh` once. It serializes three training stages and their
fixed-root autonomous evaluations. It does not promote a production model.

`search_recovery_v2.py` searches actual failed baseline prefixes. Each result
must pass `certify_recoveries_v2.py` before becoming teaching data. Search
budgets that find no solution remain unknown. `search_recovery_v3.py` can also
propose joins across own-deck alias differences, but requires complete actual
engine execution and independent certification; a state-key match is never
proof of equivalence. The observed full-curriculum fit suite has no failed
roots, so its v3 search plan is empty.

After the original launcher exits successfully, freeze the admitted recovery
mix with `freeze_recovery_learning.py`, then run
`launch_recovery_learning.sh`. It tests the identical five off-reference roots
before/after teaching, and re-evaluates the original 32 roots. Deviating prefix
actions update recurrent history but do not receive teaching loss. No holdout
source supplies gradients or recovery demonstrations.

`audit_learning_deliverables.py` independently replays evaluated responses,
checks game events and full final state, and exports YRP2, readable engine and
action logs, decision JSONL with logits/probabilities/V, and Chinese notes.
`export_recovery_teachers.py` exports independently certified search routes;
these have no policy logits or value estimates and explicitly record nulls.

Preserve completed folders and frozen plans. Do not restart a live process or
silently overwrite evidence. Native callbacks are process-global: only one
live duel may exist in a process. Read each snapshot before its action.
