# Structured-lite capacity selection (2026-09-21)

## Decision

Freeze `structured-lite-v1` at 32 public events and 8 references per
set-valued action role for the elfnote A/B/C experiment. The resulting
observation is 26,095 bytes. This is a semantic schema decision, not a limit
derived from the current GPU; a future compute upgrade does not by itself
change these capacities.

## Measurement

The accepted audit ran 512 deterministic random-vs-random elfnote mirror games
with seed 63003 and observed 113,714 decisions. To measure demand before
truncation, the audit environment used 8,192 public-event slots, 64 group
references, and 128 action slots. None of these measurement capacities
overflowed.

| Demand | p50 | p95 | p99 | p99.9 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complete public-event history | 329 | 773 | 937 | 1,081 | 1,302 |
| Group references | 1 | 1 | 2 | 2 | 2 |
| Legal actions | 3 | 10 | 14 | 18 | 67 |

Public-event overflow means the bounded tensor no longer contains the complete
duel history. It does not mean that its newest entries are corrupt: the writer
deterministically drops the oldest event. The useful comparison is therefore
how much recent decision context each window retains.

| Event capacity | Full-history truncation | Retained decisions p50 | Retained decisions p95 | Observation bytes |
| --- | ---: | ---: | ---: | ---: |
| 8 | 98.30% | 5 | 9 | 25,231 |
| 16 | 97.14% | 8 | 15 | 25,519 |
| **32** | **94.75%** | **14** | **24** | **26,095** |
| 64 | 90.08% | 26 | 42 | 27,247 |
| 128 | 80.34% | 47 | 77 | 29,551 |

Capacity 32 keeps substantially more immediate context than 8 or 16. There is
no moderate candidate that approaches complete-history retention—even 128
truncates old history in 80.3% of decisions—so increasing the sequence solely
to reduce the overflow counter has no natural stopping point and would add
event-encoder cost. Revisit this choice only if snapshot or combo diagnostics
show that the newest 32 events omit required causal context.

Group-reference capacity 1 overflowed in 1.84% of decisions and 70.7% of
episodes. Capacities 2, 4, and 8 had zero measured overflow. Eight is retained
because it gives four times the observed maximum and preserves margin for the
multi-deck audit that is currently blocked by unsupported native messages.

## Rejected and invalid attempts

The first broad-deck-pool run aborted on the unsupported native message
`field_disabled`; it produced no report and contributes no rate estimates. An
elfnote run using the production action capacity 24 was also rejected because
action overflow occurred. The accepted audit raised only the measurement
action envelope to 128. Production `max_options=24` remains an explicit open
item under the general overflow-validation task.

## Contract verification

Regenerating the Structured-lite manifest produced JSON content identical to
the committed manifest; byte hashes differ only because the generated WSL file
uses LF and the committed Windows file uses CRLF. Python defaults, manifest
shapes, native defaults, training/evaluation arguments, and the full benchmark
checkpoint metadata all encode public events 32 and group references 8.

## Compute-upgrade trail

The accompanying compute manifest records the RTX 5070 Ti identity and UUID,
16,303 MiB VRAM, driver 616.92, compute capability 12.0, CPU and memory data,
kernel/WSL/Python details, JAX backend and package versions, repository dirty
state, tracked-diff hash, and hashes of every important audit input.

When hardware or parallelism changes, keep decision ID
`structured-lite-v1-capacities-20260921`, capture a new compute manifest, and
rerun the matched legacy/relationship-only/full resource benchmark. Rerun the
capacity audit only when the deck pool, native event writer, or semantic
requirements change. If capacities must change, create a new schema and
checkpoint version rather than mutating `structured-lite-v1` in place.

Machine-readable evidence is in
`training-runs/structured-lite-local-validation-20260921`:

- `capacity-audit-elfnote-512.json`
- `capacity-selection.json`
- `capacity-audit-attempts.json`
- `compute-manifest-capacity-audit.json`

