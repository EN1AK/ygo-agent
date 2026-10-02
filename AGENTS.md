# Workspace instructions

## YGO Agent servers

Use the following SSH targets for work on `ygoai/ygo-agent`. The username is
part of the SSH gateway target; quote the complete target in PowerShell.

### GPU training server (default)

```powershell
ssh -CAXY -p 2202 "ws-a46c23cbbc247a98.yingtianyi+root.fdj-infra.ws@ssh.gate.yicloud.com.cn"
```

- Primary server for GPU training and evaluation.
- Deployment root: `/root/ygo-agent-gpu-20260910`
- Repository directory: `/root/ygo-agent-gpu-20260910/ygo-agent`
- This server may not always have usable public-network access. Prefer packaging
  files locally and uploading them with `scp -P 2202` when a pull or dependency
  download fails.
- NVIDIA userspace library path: `/usr/local/nvidia/lib64`
- CUDA library path: `/usr/local/cuda-12.9/targets/x86_64-linux/lib`
- cuDNN library path:
  `/root/ygo-agent-gpu-20260910/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib`

### Internet-connected build server

```powershell
ssh -CAXY -p 2202 "ws-d1fd808734bb0029.yingtianyi+root.fdj-infra.ws@ssh.gate.yicloud.com.cn"
```

- Use for downloading dependencies and producing portable Linux/Python 3.10
  native builds.
- Build portable release artifacts by default. Do not enable
  `--native_optimization=y` when the result will be copied to another server.

### Offline server

```powershell
ssh -CAXY -p 2202 "ws-f96506c843eb41ef.yingtianyi+root.fdj-infra.ws@ssh.gate.yicloud.com.cn"
```

- This server cannot access the public internet.
- Transfer source code, assets, dependencies, and build outputs from the local
  machine or the internet-connected build server.

## Deployment convention

- Local repository: `C:\Users\Mortis\Desktop\Workspace\ygoai\ygo-agent`
- Local distribution files: `C:\Users\Mortis\Desktop\Workspace\ygoai\dist`
- When the user says "the GPU server" without further qualification, use the
  `ws-a46c23cbbc247a98` target above.
- Never assume a remote server can pull from GitHub. If remote network access
  fails, create a Git archive locally, upload it, and extract it over the
  existing deployment directory.

## Four-way synchronization

Keep these four copies aligned whenever source code or operational scripts
change:

1. Local checkout:
   `C:\Users\Mortis\Desktop\Workspace\ygoai\ygo-agent`
2. GitHub branch `experiment/counterfactual-search`
3. Home checkout: `/d/workspace/ygo-agent`
4. H200 snapshot: `/root/ygo-agent-gpu-20260910/ygo-agent`

Use this order: commit locally, push GitHub, fast-forward home with
`git pull --ff-only`, then deploy that exact commit to H200. Preserve untracked
`assets/deck-corpus/`, `training-runs/`, native modules, checkpoints, and other
machine-local resources. H200 is a source snapshot rather than a Git checkout;
write a deployment marker under
`/root/ygo-agent-gpu-20260910/training-runs/source-deploy-<short-commit>.txt`
containing the full commit, transport archive hash, and any line-ending
normalization performed. Verify all four source revisions after deployment.

## PowerShell, SSH, and SCP pitfalls

- The default local shell is PowerShell 7. When an SSH alias is needed, pass
  `-F C:\Users\Mortis\.ssh\config` explicitly.
- Quote the complete gateway target. `ssh` uses lowercase `-p 2202`; `scp` uses
  uppercase `-P 2202`.
- Do not put remote `$variable`, `$!`, or `$(command)` expressions inside a
  PowerShell double-quoted command string. PowerShell may expand them locally;
  backslash is not a PowerShell escape for `$`. Prefer a checked-in or temporary
  shell script uploaded with `scp`. For short inline commands, avoid remote
  interpolation or use PowerShell-safe single quoting.
- On home, the interactive MSYS shell path is `/d/workspace/ygo-agent`, but the
  SFTP/SCP path is `D:/workspace/ygo-agent`. A path that works through `ssh`
  may therefore fail through `scp` with "No such file or directory".
- The `home` SSH alias lands in a Windows/MSYS session as `Administrator`; it
  is not the Linux build shell. Invoke Linux tools through
  `wsl.exe -d Ubuntu-22.04 -- bash -lc '<command>'`. Inside WSL the canonical
  checkout is `/mnt/d/workspace/ygo-agent`, xmake is
  `/home/ygo/.local/bin/xmake`, and the reusable Python 3.10 environment is
  `/home/ygo/ygo-agent/.venv-wsl`. The separate `/home/ygo/ygo-agent` tree
  contains machine-local runtime resources and may have experimental changes;
  do not treat it as the four-way synchronized source checkout or overwrite it.
- MSYS can rewrite a direct Linux path passed to `wsl.exe` into a Git-for-Windows
  path (for example `/home/ygo/...` becomes `D:/Program Files/Git/home/ygo/...`).
  In the remote MSYS shell, prefix WSL calls with `MSYS_NO_PATHCONV=1`, including
  calls that pass an absolute Linux executable or script path after `--`.
- Do not trust a printed `pid=$!` unless the value is numeric. Confirm detached
  jobs with `pgrep -af` and their run-directory logs.

## Linux snapshot and line-ending pitfalls

- A Git archive produced from the Windows checkout may contain CRLF text even
  though H200 requires LF shell scripts. After extraction on H200, normalize
  every deployed `*.sh` to LF and run `bash -n` on the launch script before
  starting a job. Never continue after a `syntax error ... $'\r'` or a carriage
  return shown at the end of a shell line.
- Treat the archive SHA-256 as a transport check only. Do not use a raw text-file
  SHA-256 as a cross-platform semantic gate when CRLF/LF conversion is allowed.
  Protocol gates must use the canonical internal `contract_sha256`; record the
  raw file hash separately for provenance.
- Uploading a tracked-source archive does not upload runtime-only assets. Check
  that deck corpora, sampling manifests, semantic tables, databases, scripts,
  native modules, and prior evidence under `training-runs/` still exist before
  launching training.

## Protocol-gate artifacts

- `protocol-coverage-current.json` is generated evidence, not timeless source.
  Rebuild it whenever the protocol contract's internal hash, adapter, or
  boundary-test file changes; never bypass a failed startup assertion.
- The coverage build also needs the original runtime evidence files
  `runtime-smoke-50-v4.json` and `structured-environment.json`. Their durable
  copies are in the home checkout at
  `/d/workspace/ygo-agent/training-runs/protocol-gate-v1-20260923/` (use the
  corresponding `D:/...` path with `scp`). Preserve and verify their hashes
  before rebuilding coverage on H200.
- Before a long pilot, run the launch script once with `CONFIG_ONLY=1` and a new
  run directory. Require zero uncovered branches, matching canonical contract,
  adapter, and boundary-test hashes, successful CUDA/JAX discovery, and the
  intended observation/action capacities.

## Native-module diagnostics and cleanup

- Do not run `xmake f -c` with `/root/.local/bin/xmake` directly on H200. Its
  2.9.9 executable is paired with newer program scripts that call
  `on_prepare_file`, so configuration fails before project compilation. Build
  portable CPython 3.10 modules on the internet-connected build server in the
  `ygo-build-cache:replay-fix` Ubuntu 22.04 image, then verify the SHA-256 and
  transfer the artifact to H200. If that image must fetch SQLiteCpp 3.2.1 and
  GitHub is unreachable, inject the official tag archive (known SHA-256
  `70c67d5680c47460f82a7abf8e6b0329bf2fb10795a982a6d8abc06adb42d693`)
  into the image's xmake cache rather than changing H200's global toolchain.
- Never overwrite the production native module without first recording its
  SHA-256 and making a byte-for-byte backup. Use an `EXIT INT TERM` cleanup trap
  and do not start another job until the production hash is restored.
- Cached native-build images may already contain a stale `/work/ygo-agent`.
  `cp -a /src/ygo-agent /work/ygo-agent` then nests the requested source below
  the stale tree and can silently compile the old checkout. Always copy into a
  unique, asserted-absent container path and verify a commit-specific source
  marker or symbol before invoking xmake.
- Sending `TERM` only to a wrapper shell that is waiting for Python may defer its
  cleanup trap. Terminate the child process normally, wait for the wrapper to
  run the trap, then verify both the restoration marker and module SHA-256.
- ASan and CUDA/JAX may conflict in virtual address space and report
  `cuInit CUDA_ERROR_OUT_OF_MEMORY` even when GPU memory is free. Use ASan for a
  bounded CPU/native-environment diagnostic, and use the production or
  debug-symbol module with JAX GPU for throughput and long training. Do not let
  a slow ASan run block the GPU pilot after it has covered the target failure
  window without an error.
- `nvidia-smi` can be absent or can fail with an NVML driver/library mismatch on
  this container. The JAX backend/device report and an actual config-only run
  are the authoritative GPU availability checks.

## Deferred protocol optimization: canonical unweighted multi-select

- This is a possible sample-efficiency optimization, not a current correctness
  blocker. The active `YGOPro-v1` adapter stages ordinary `MSG_SELECT_CARD` and
  unweighted `MSG_SELECT_TRIBUTE` decisions and fails loudly on action-capacity
  overflow; it does not use the legacy EDOPro combination expansion and silent
  truncation path.
- The remaining mode-0 selection path removes cards after they are selected but
  still permits every ordering of the same final set. Selecting `k` cards can
  therefore expose up to `k!` equivalent micro-action trajectories. A future
  change may require monotonically increasing original candidate indices so
  each unordered result set has one policy path.
- Preserve `min`/`max`, finish, cancel, forced-selection, and response-index
  semantics when canonicalizing. Never apply unordered-set canonicalization to
  `MSG_SORT_CARD` or another prompt whose response order is meaningful.
- Weighted `MSG_SELECT_SUM` and tribute paths already use monotonic staged
  choices with completion checks. Consider memoization or dynamic programming
  only if measurement shows their bounded DFS search is material. Likewise,
  consider an explicit decision-group identifier or more than eight selected
  card references only after observing an actual ambiguity or overflow rate.
- Treat canonicalization as an environment/protocol behavior change even if
  tensor shapes remain compatible. Keep the current 5M run as the baseline and
  use a fresh, otherwise identical A/B run when evaluating this optimization;
  do not attribute a warm-start continuation delta solely to canonicalization.

## Training workflow

- Follow `ygoai/ygo-agent/TRAINING_PLAN.md` for subsequent model-training work.
- Execute its stages in order unless the user explicitly changes the plan.
- Preserve experiment configurations, logs, checkpoint hashes, and evaluation
  reports so results remain comparable across conversations.

## Replay deliverables

When the user asks for a duel replay or match video/recording, deliver all of
the following together for every requested duel:

1. The playable YGOPro `.yrp` replay file.
2. The complete human-readable engine/action `.log` file.
3. A structured decision log (prefer JSONL) containing, for every model
   decision, the acting player/model, step and phase context, legal actions,
   selected action, raw policy logits, normalized action probabilities, and
   critic state value `V(s)`. Also include terminal reward/result and checkpoint
   identifiers. Do not label `V(s)` or policy logits as per-action Q-values;
   record Q-values only if a real Q estimator was used.
4. A natural-language Chinese description of the duel, covering the opening,
   important interactions, turning point, finishing sequence, and any
   suspicious or strategically weak decisions.

Keep these files in one clearly named result directory, preserve seeds and both
players' checkpoint hashes, and download or link the complete bundle for the
user. A `.yrp` file alone does not satisfy a replay request.
