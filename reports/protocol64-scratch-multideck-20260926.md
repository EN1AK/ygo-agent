# Protocol-64 from-scratch multi-deck training, 2026-09-26

## Decision

The initial multi-deck policy uses random initialization, not a continuation of
the 40M elfnote specialist. The specialist and its verified migration remain
immutable historical/reference artifacts. The pilot freezes Structured-lite-v1
features, a 128-slot legal-action capacity, the clustered corpus, equal
non-anchor cluster sampling, and a 25% per-seat elfnote reserve. A later change
in compute is a linked run with a new compute manifest and exact parent hash.

## Protocol and feature gate

- Pinned ygopro-core revision: `f96929650ff8685b82fd48670126eae406366734`.
- Protocol contract internal SHA-256: `d97532683c4675f012e4b5ab24a82bafbc62631856586cc0e4ed8d4d5dc2627c`.
- Protocol contract file SHA-256: `d0b00736ed915483f2457ef4cb1b8ff16af24ec4841469c7e5f332d06761a852`.
- Current core/adapter coverage: 64 of 64 normal interactive branches; no
  uncovered branch. The source-derived and same-core response-oracle suite has
  96 passing tests.
- Real menu sizes exceeded the previous 24 slots, so the pilot uses 128. Large
  announce-card spaces are narrowed in bounded stages, not materialized as all
  possible selections or permutations. Unknown card code 0 is represented by
  the schema sentinel rather than sent to an embedding out of range. Automatic
  counter selections advance the engine before the next policy decision.

## Completed from-scratch smoke

GPU server result directory:
`/root/ygo-agent-gpu-20260910/training-runs/multideck-scratch-smoke100k-v4-20260926-h200`.

- Initialization file says `random-from-scratch`, seed 1; no parent checkpoint.
- 100,352 learner steps; 190 completed valid games; zero invalid games.
- Optimizer stayed finite with zero non-finite counts. Action/logit/value checks
  were finite and deterministic; exact save/reload and cross-manifest rejection
  passed (`checkpoint-verification.json`).
- Final weights SHA-256:
  `4926fe1325d1b2c558c0be563dec68eb49c7a47d0f89b80a2c931830b8e9cdaf`.
  Final metadata SHA-256:
  `43a215704e5d55b9fe3302c93ad64c10b5baebe0ccc83548c949b413134a73e6`.
- Realized elfnote seat frequency: 96 of 380 seat samples, 25.26%.
- Earlier failed diagnostic attempts remain in separate result directories; none
  was silently resumed as the successful smoke.

## Fixed 5M pilot

Launched as an independent random-initialized run on the H200:
`/root/ygo-agent-gpu-20260910/training-runs/multideck-scratch-pilot5m-20260926-h200`.
The outer launcher log has the same basename with `.outer.log`. The run directory
holds `initialization.txt`, `launch-command.txt`, `launch.sh`,
`protocol-contract.txt`, `compute-manifest.json`, `train.log`, checkpoints, and
eventual checkpoint hashes and completion/failure marker. Its training target is
5,000,192 steps. This report does not assert completion until the marker and
post-run verification exist.

### Pilot result

The run did not complete. It terminated with native exit code 139 at roughly
1,740,800 learner steps. The last durable checkpoint is step 1,579,008. The
optimizer remained finite and the Python log contained no exception before the
segmentation fault, so this is treated as a native environment/adapter stability
failure rather than a successful protocol gate. Task 8.7 remains incomplete and
the checkpoint is diagnostic only.

### Post-run protocol-contract drift

The protocol asset originally recorded adapter SHA-256
`2a3677b28482f783e4189a7e7deaf8b23161e7f263cff30608fba4308edbf334`
and internal contract SHA-256
`d97532683c4675f012e4b5ab24a82bafbc62631856586cc0e4ed8d4d5dc2627c`.
The adapter used by the final native build was subsequently changed. Its
canonical source SHA-256 is
`adb01710371bdcb3f2a0208d646898ff2ed9e1e1c7d3b0156a7b63739257b9a1`.
The audit now normalizes CRLF and bare CR line endings to LF before hashing
source files, so Windows, WSL, Git archives, and Linux deployments produce the
same contract. Regenerating the deterministic inventory produces internal
contract SHA-256
`ed4259cbad49c1b740d8b6e74d05ffaad1d54344bfdca611e5b70f3d137a01fe`.
The inventory still contains 82 messages, 17 policy-interactive messages,
3 automatic-interactive messages, and 79 messages with core writers. The source
asset is updated to the regenerated contract, while the run-local
`protocol-contract.txt` is preserved unchanged as historical evidence of what
the failed launcher claimed. A rerun must bind the regenerated contract and
record the native module hash before it can satisfy the gate.

## Promotion interpretation

The fresh 5M policy is not expected to match the mature 40M elfnote specialist
yet. The latter is a descriptive baseline only. Functional promotion requires
zero protocol/domain violations and invalid games, finite inference/optimizer,
working checkpoint reload and sampler resume, realized exposure of all ready
clusters, and a complete both-seat held-out/elfnote evaluation with uncertainty
intervals. Long-run budget and actor topology are selected only after the pilot
resource and learning-curve report is recorded.
