# Effect Veiler vs Ghost Ogre snapshot validation

Date: 2026-09-15

## Case identity

- Source replay seed argument: `62`; effective environment seed: `71900845`.
- Decision: `battle-62-seat1:14` (zero-based model decision 14).
- Legal actions: `0 = Effect Veiler`, `1 = Ghost Ogre`, `2 = cancel`.
- Recorded model choice: Effect Veiler (`action 0`).
- Original policy probabilities: Veiler `0.554231`, Ogre `0.390856`, cancel `0.054914`.
- Snapshot restoration reproduces the observation, acting player, and legal-action count exactly.

## Immediate causal branch result

Both branches restore the same action prefix and differ only in the forced root
action.

### Effect Veiler (`action 0`)

The engine log shows that Effect Veiler is activated, sent from hand to the
graveyard, and targets the opponent's `耀圣诗之狱神精`. It does not negate or
remove the effect source, the continuous spell `耀圣之诗～回乡之平行体～`.
The spell therefore finishes resolving and special summons
`耀圣之月诗 福尔图娜` from the deck.

### Ghost Ogre (`action 1`)

The engine log shows that Ghost Ogre is activated and sent from hand to the
graveyard. During chain resolution, `耀圣之诗～回乡之平行体～` is explicitly
reported as destroyed. The chain ends without the corresponding deck search
and special summon of `耀圣之月诗 福尔图娜`.

This establishes the local tactical error directly: Effect Veiler spends a
card without stopping the activated continuous-spell effect, while Ghost Ogre
removes the face-up continuous spell and prevents that effect from resolving
successfully.

## Rollout evidence

One paired checkpoint-policy seed, used chiefly to retain a complete readable
branch trace, returned `-4.0` for Veiler and `+0.8333333` for Ogre. This single
sample is illustrative, not an estimate.

The 32-seed paired terminal-return experiment produced:

| Root action | Mean return | Standard error |
|---|---:|---:|
| Effect Veiler | -5.21875 | 0.37764 |
| Ghost Ogre | -5.18750 | 0.43865 |
| Cancel | -5.37500 | 0.40598 |

Ghost Ogre has the best mean, but its estimated advantage over the recorded
Veiler action is only `0.03125`; paired regret standard error is `0.56882` and
the lower 95% bound is zero. Full-duel stochastic terminal return is therefore
too noisy to certify this local interaction by itself.

## Decision for the training pipeline

Treat this case as a validated snapshot regression and a positive example of
`Ghost Ogre > Effect Veiler`. The causal engine outcome is the acceptance
criterion. Long-horizon rollout Q remains auxiliary evidence and must not veto
an unambiguous local rules interaction when its confidence interval is this
wide.

For automated mining, add short-horizon features/rewards for interruption
success, including whether the source continuous spell remains on the field
and whether the threatened search/special summon occurs. Use the full-duel Q
estimate afterward as a strategic filter, not as the sole label generator.

## Preserved artifacts

Server directory:
`/root/ygo-agent-gpu-20260910/training-runs/veiler-ogre-forced-logs-20260915T1725Z`

Local bundle:
`ygoai/dist/veiler-ogre-snapshot-validation-20260915`

The bundle contains both complete engine logs, the paired single-seed rollout
records, and the 32-seed soft-target/statistics record.
