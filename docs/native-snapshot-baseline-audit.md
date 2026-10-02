# Native snapshot baseline audit (2026-10-02)

This is a pre-integration inventory, not a passed snapshot/rollback gate. Do not
enable native snapshots on the running PPO module based on this document.

## Exact baseline

- Core package: `ygopro-core 0.0.4`, upstream
  `f96929650ff8685b82fd48670126eae406366734`.
- Package recipe: `repo/packages/y/ygopro-core/xmake.lua`; static core and
  `lua 5.3.6`. The recipe also inserts `<cstring>` and C linkage around Lua headers.
- Empty-deck patch SHA-256:
  `08a7e03f883657b60da21f918be3c717db229b335b43b7c2f5388fd105acdd70`.
- Disable-check guards patch SHA-256:
  `8f385649ef81c06ed28dc8ae2fbcfd4b114a4d0fe99e96a661f9b4a057fafe9e`.
- Portable native build source:
  `92154ab7a165c5749cead7d10f5f862aeb63f3bf`.
- Production module SHA-256:
  `5962816be0362007c751c43a515d18a50abcd7716c00be579aac0aba1e6705a8`.
- Build image `ygo-build-cache:replay-fix`, ID
  `sha256:be59db1ab77bfada7a5b111e873520ae2fbe96b70cf754d2e1c656fe1781f96a`;
  Ubuntu 22.04, g++ 11.4.0, Python 3.10.12, xmake 3.1.1+20260910,
  native optimization disabled. Build job on the documented build host:
  `/root/ygo-build/jobs/disable-check-guard-92154ab`.
- The rebuild passed 62 existing protocol tests and two deterministic 117-step
  duels per observation schema. These establish baseline behavior, not snapshot
  behavior or absence of all pointer-lifetime defects.

H200 `ldd` inspection showed libpython3.10, libstdc++, libm, libgcc_s, libc,
libexpat and libz, with no separate shared Lua dependency. This agrees with the
static Lua/core recipe; it does not prove individual allocator interception.
Dynamic symbols include `PyInit_ygopro_ygoenv`, but no exported `create_duel`,
`process`, `lua_newstate`, `snapshot` or `rollback` API was found. The normal
symbol table is stripped (`nm: no symbols`). A snapshot binding therefore
requires a new isolated build; it cannot be bolted onto this binary with ctypes.
Retain debug symbols for the new diagnostic build.

## Mutable state and ownership inventory

| Area | Current storage | Snapshot requirement |
| --- | --- | --- |
| Core duel graph | `duel` creates `interpreter`, `field`, cards, groups and effects with ordinary `new`; owning sets/vectors allocate separately | Route the entire graph and all container allocations, not only the root object |
| Lua | `interpreter.cpp` calls `luaL_newstate`; `lua_state`, `current_state`, registry and coroutine map persist across prompts | Use duel-scoped Lua allocator; keep all live coroutine pointers at fixed addresses |
| Core RNG and buffers | `duel::random`, message buffer, response fields and field processor queues | Restore bytes and allocation metadata together |
| Callback registry | `ocgapi.cpp` process-global `sreader`, `creader`, `mhandler` | Immutable after initialization; validate identity, do not roll back another duel's registry |
| Default script loader | Shared `buffer[0x20000]` in `ocgapi.cpp` | Prohibit that mutable shared fallback or give it explicit lifetime/thread ownership |
| Live duel registry | Process-global `std::set<duel*> duel_set` | Keep outside arena; synchronize lifecycle and validate owner/generation; never restore another duel's membership |
| Lua library registration | `static const luaL_Reg` tables | Immutable, excluded from arena |
| Adapter caches | Card data, card scripts, semantic/database/deck tables | Keep pinned and immutable outside duel snapshots; audit cache population/lazy writes separately |
| Adapter mutable state | Message/response buffers and cursors, staged selection, options, public events, action history, RNGs, terminal/reward state | Separate explicit adapter snapshot is required even with a perfect core arena |
| Model state | Independent recurrent state for each player | Capture with branch ownership; never share mutated recurrent arrays across branches |
| External bot/network state | Remote process and socket state | Unsupported; reject search eligibility rather than pretend core rollback restores it |

No process-wide `operator new` hook has been installed. Such a hook could capture
unrelated Python/JAX/worker allocations and would need a complete ownership and
concurrency proof. Likewise, shallow copying `duel` leaves STL/Lua pointers
pointing at live mutable storage and is not an acceptable snapshot.

Next native gate: implement the pinned side-by-side arena variant, inventory and
test every allocation/free/realloc path, preserve debug symbols, and compare raw
engine frames plus adapter/model state against deterministic replay under ASan.
The production module remains unchanged until those gates pass.
