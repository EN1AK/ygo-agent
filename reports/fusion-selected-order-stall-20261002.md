# Fusion selected-material search: stall diagnosis and bounded recovery

The 5-actor × 48-env continuation reached 58,762,240 logged steps, then
made no logged progress for about twelve minutes. The environment timeout
replaced env 44 and progress briefly resumed at 58,915,840. The old worker
remained CPU-active. A deliberate diagnostic SIGINT at 20:35 CST terminated
the process under GDB; exit 255 is **not a spontaneous crash**.

Evidence: `training-runs/stall-58762240-gdb-20261002.log`; H200 run
`multideck-fastest-56m-to100m-913b40d-20261002`, including
`stall-interrupt-requested.txt` and `train-before-stall-interrupt.log`.
The timeout was `elfnote` versus `deck-0db5f537194f9bfc` (Fluffal/Frightfur),
step 212, turn 6, phase 4, LP 8000/1800, select-unselect-card prompt.

## What is established, and what is not

Thread 324285 repeatedly called Lua -> `Group.IsExists` ->
`interpreter::check_filter` -> Lua. This is within-engine computation, not
repeated policy cancellation or a PPO gradient update.

A same-source, unstripped portable module was built for diagnosis only
(SHA256 `67146e27016f23069f6b0826ebcef4710de9a1483d9e3d3e71a2e400c809e755`).
Its full `.text` differs because core functions were reordered at link time;
directly applying its symbols at the old offsets is invalid. Unique instruction
windows instead matched original ELF return PCs 0x345ec1/0x345f9b to
`scriptlib::group_is_exists`, and 0x329793 to `interpreter::check_filter`.
The intervening Lua frames match 48-byte windows at the same offsets.
The rebuilt binary was **not deployed**.

The deployed `procedure.lua` has a reproducible factorial path in
`FCheckMixRepSelectedCond`: it permutes already-selected cards while assigning
material roles, although every selected card must be assigned exactly once.
The patch uses a fixed card order while retaining all role/substitute choices,
quantity checks, remaining-material search and field constraints. Temporary
clones preserve the caller's group membership and iteration state. This does
not cap legal combinations or hide more model actions.

The original stalled duel had no action-prefix/Lua-stack capture, so the exact
Lua helper in that duel is not independently proven. The confirmed native call
family and a reproduced compatible pathological fusion path support this fix;
they do not prove all engine stalls are eliminated. Long-run confirmation and
stronger future Lua/prefix diagnostics remain necessary.

## Validation and asset identity

- 6,000 deterministic actual-Lua old/new helper fixtures: zero disagreements,
  including role assignment, one substitute, material counts, mandatory cards,
  field-space constraints, group restoration and iterator preservation.
- Nine selected repeated-role cards with an impossible mandatory material:
  old helper 1,349,291 `IsExists` calls; patched helper 3; both return false.
- H200 production native module + frozen 57,994,240 model: 12 complete duels,
  both deck orders, deterministic and seeded-sampling policies; all 2,424
  decision transitions, observations, menus, logits, values, actions and final
  outcomes match. Evidence: `fusion-selected-order-preflight-20261002`.
  These are differential regressions, not a replay of the original stall.
- Upstream procedure SHA256:
  `64e510643688d192e1b2ad4c979e131df93401a519fbcf6493bdf5d42b8ebd3c`.
- Patched procedure SHA256:
  `c2742f65f750092ef4298889391bc718c52ece5fad8d07c5a6e2ed69e0fe1322`.
- `scripts/patch_fusion_material_search.py` refuses unknown inputs and existing
  output files. Preserve the original procedure and record the patch as an
  explicit runtime-asset overlay, not as an unchanged upstream asset snapshot.
- Native module stays
  `5962816be0362007c751c43a515d18a50abcd7716c00be579aac0aba1e6705a8`.

## Recovery gate

Recover from cumulative 57,994,240 checkpoint SHA256
`ac5dcc0c8ab533f5341f89d78f6fed365568563afa123bb45a766eb3ca560fee`,
decoded with 165 finite arrays and matching metadata. Target remains
100,003,840. Keep 5×48, 11 env threads/actor, batch 15,360, 80 minibatches,
learning rate 1e-4, Q/search/belief off. Restore sampler counters without a new
repartition; weights plus fresh optimizer remain the established continuation
convention. Require config-only, asset hashes, a new run directory and a newly
validated saved checkpoint before reporting healthy recovery. Do not call the
interrupted run complete or restart using its earlier throughput as live status.

The baseline report parser was also corrected to exclude non-natural terminal
reasons and non-finite rewards, retain draws and per-episode records, and report
valid-game Wilson intervals. Seven focused Python tests passed. This changes
evaluation reporting, not PPO training behavior.
