# YGO Agent training plan

This is the default execution plan for the K9VS training work. Follow the
stages in order and record configuration, checkpoints, metrics, and conclusions
for every experiment. Do not scale model size until the baseline is trustworthy.

## Stage 1: Reliable baseline

Before the run, change checkpoint names to include the exact global step so
checkpoints in the same million-step range cannot overwrite one another.

Run a medium-scale 10,000,000-step baseline with approximately:

- Deck: K9VS
- Language: Chinese
- External pretrained embedding: disabled; use the correctly sized learned
  embedding derived from `code_list.txt`
- `max_options=24`
- 128 parallel environments
- 112 environment threads
- 128 rollout steps
- 64 minibatches
- FP32
- Learning rate `1e-4` with linear annealing
- Advantage normalization enabled
- Fixed and recorded random seed
- Save every 50 updates
- Run the larger evaluation every 100 updates

Continuously verify that:

- `optimizer_finite=True`
- `notfinite_count=0`
- Policy loss, value loss, entropy, KL, model outputs, and gradients remain
  finite
- Checkpoint hashes and/or parameter deltas change after optimizer updates

Expected throughput is around 10,000 SPS on the current H200 setup. Preserve
TensorBoard events and the complete console log.

## Stage 2: Trustworthy evaluation

Evaluate both the final checkpoint and the best intermediate checkpoint. Use at
least 256 games per matchup and report win-rate confidence intervals.

Maintain these evaluation matchups:

1. Fixed greedy bot for an absolute baseline
2. Initial random model to verify acquisition of basic play
3. Historical checkpoints to detect regressions and strategy cycling

Report:

- Training and evaluation SPS
- Wall-clock duration
- GPU utilization, memory usage, and power
- Win rate, confidence interval, average reward, and average episode length
- Loss, entropy, KL, and learning-rate trends
- Final and best checkpoint paths and SHA256 values

## Stage 3: Historical opponent pool

After the baseline and evaluation pipeline are stable, extend opponent sampling
approximately as follows:

- 50% current self-play policy
- 30% randomly sampled historical checkpoint
- 20% fixed greedy bot

Add checkpoints to the historical pool periodically. Confirm that model loading,
architecture metadata, code-list identity, and learned-embedding dimensions
match before a checkpoint enters the pool.

## Stage 4: Pretrained card embeddings

Only after establishing the learned-embedding baseline, compare it with card
text or card-attribute pretrained embeddings:

1. Randomly initialized, trainable learned embedding baseline
2. Pretrained embedding initially frozen
3. Pretrained embedding with later gradual unfreezing/fine-tuning

Keep all other training settings and evaluation seeds constant so the embedding
comparison is attributable.

## Stage 5: Larger models and further scaling

Consider wider/deeper models, longer context, bfloat16, more environments, or
longer runs only after Stages 1-4 are stable. Prefer controlled ablations over
changing several variables simultaneously.

## Current infrastructure

Use the GPU server and deployment paths documented in the workspace-level
`AGENTS.md`. If a server cannot access the public internet, package the relevant
Git revision locally, upload it, and extract it over the deployment directory.

