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
