# Reality-anchored learning

User-authorized direction, 2026-10-28: implement the Build, Don't Play research
loop (arXiv:2610.33845v1), carrying it through execution and evaluation. This is
an ongoing project objective, not completed by creating this document.

## End state and evidence

1. Freeze environment, policy, opponents, curriculum, independent evaluation,
   budgets and promotion criteria before each experiment. Refuse incompatible
   inputs; preserve full histories, original menus and hidden-information rules.
2. Collect autonomous failures from the current policy. Distinguish invalid
   runtime, incomplete observation, repeated reversible selections, strategic
   failure and unknown/budget exhaustion. A difference from a demonstration is
   a candidate, never automatically a wrong-action label.
3. Restore actual failed prefixes; search useful continuations. Only complete
   goal-reaching routes reproduced by the adapter and independently replayed
   through the raw core become teaching data. Unknowns remain unknown.
4. Compare continuing the original curriculum with adding admitted repairs,
   from the same parent and with equal optimizer-update budgets. Prefix actions
   reconstruct the current model's recurrent history and receive no repair CE.
5. Repeat collection using the updated policy, rather than recycling only the
   initial model's errors. Append typed findings, corrections, run records and
   lessons with evidence hashes. Process completion and scientific improvement
   are separate claims.
6. Validate autonomous transfer on newly frozen player/deck/match components,
   full duels against compatible frozen opponents, both seats and paired seeds.
   Previously inspected holdouts are regression data. Do not select checkpoints
   on the independent exam. Report invalids, uncertainty and search/teaching
   costs in addition to win rate.
7. Run the compatible formal PPO continuation with ordinary-PPO control and
   verified full-history retention, then repeat skills/transfer/full-duel exams.
   Local-course success cannot promote a production default.

## Initial cycle

Start from the completed Crystron local-skills checkpoint 831a78a04c15 and the
235-row frozen-local-goals-v2 corpus. These are existing development resources,
not new independent tests. The initial diagnosis is repeated select/unselect
cycles in MSG_SELECT_UNSELECT_CARD (26); verify it per decision before drawing
a policy or interface conclusion. Existing independent engine audits establish
execution fidelity, not strategy quality or transfer.

Run two bounded repair rounds, each observing its own starting checkpoint.
Search at most eight deterministic fit-only failure roots per round, with
bounded continuation attempts/time. Compare 512-update control and repair arms
per round, preserving foundation/derivation/natural and earlier repairs.
The exact machine-readable plan is frozen in a new run directory before model
inference. Failed searches and launches are kept; never overwrite a result or
restart a process because an observation timed out.

## Search information boundary

Exact historical replay is an offline oracle used to certify local teaching
witnesses. It does not demonstrate fair-play online search or globally optimal
actions. Core RNG is replayed for fidelity; an eventual online branch backend
must separately replace hidden states with legal particles and resample future
randomness. Search-tree visit counts and policy logits are not ground-truth Q.

## Status

The research objective remains active until the implemented loop has actual
audited runs, repeated feedback, transfer/full-duel validation and post-PPO
retention evidence. Negative results trigger a new diagnosed intervention;
they do not authorize claiming success or silently weakening the exam.
