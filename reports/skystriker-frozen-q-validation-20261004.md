# Frozen SkyStriker Q: bounded validation, no promotion

The 262,144-transition critic-only run completed 128 finite updates in 439.4s.
Actor/optimizer/step remained unchanged; checkpoint SHA-256 is
`a903bc2b7e0ba7004ccbea15de044b1bae0834256a75157e923de9c63a048567`.
This follow-up made no parameter updates and did not start a 5M continuation.

## Fresh seed return calibration

Protocol declared before sampling: seeds61042001/61042002, 32 first episodes
per environment batch, stochastic frozen121M self-play, both seats, gamma1,
greedy_reward=False, max1000 steps. Natural terminations require
termination_reason1 and invalid_game0. Do not replace unfinished/invalid games.
All64 ended naturally, giving22,653 decisions. Menu mismatch/nonfinite counts0.
The Q prediction uses the acting-seat channel and the actual chosen legal slot.
Terminal reward is signed into each earlier acting seat's perspective.

| Predictor | Chosen-action realized-return RMSE |
| --- | ---: |
| Independent Q | 0.953886 |
| Constant zero | 1.000000 |
| Frozen actor V | 0.901986 |

Equal-game-weight bootstrap of Q MSE minus zero-baseline MSE: 95% interval
[-0.235954,-0.046805]. These are game clusters, not independent decision rows;
the point RMSE table is decision-weighted. Single realized outcomes contain
irreducible rollout noise, so RMSE alone is not Q approximation error.
This is held-out seeds on one deck, not unseen-deck validation. Actor V is an
additional state-value reference, not a per-action ranking model.

Declared preliminary gates (natural count>=48, improvement over zero with
upper95% delta<0, warm Q batch latency<50ms, finite/menu errors0) passed.
Q batch32 median latency4.360ms; measured total decision throughput393.7/s
includes compilation, synchronization and logging. Peak CPU RSS3,387,879,424B;
GPU memory and prompt-stratified metrics are in report.json.

## Selected diagnostic roots: transfer still insufficient

The five frozen roots were never fitted. Three loss/loss attack alternatives
have no established long-term preference label. The two resource roots share
one duel and are not independent examples:

- Step189: battle over Area Zero, Q margin+0.045188, agrees with the single-seed
  loss-to-win intervention.
- Step190: target Solemn Warning over Bomber, Q margin-0.002741, disagrees.

Near ties are not reliable counterfactual value estimates. These roots use
argmax versus WindBot, unlike stochastic self-play fitting/calibration. Do not
interpret one correct root as a searched-policy strength gain. The small
training TD residual (~0.131 final) did not establish Monte Carlo calibration.

No promotion: keep task2.4 open for broader ranking/held-out-deck evidence and
keep Q-actor integration/matched5M gates closed. Prefer additional critic/ranking
investigation with actor frozen before letting Q affect policy learning.

## Evidence and startup failures

Home: `training-runs/skystriker-q-validation-v4-20261004` under
`/home/ygo/ygo-agent`. Package `skystriker-q-validation-20261004.tar.gz`, SHA256
`730cb5320a297a85cdfd35e45e0b9faf8265a33761ea8df98875b4e391780ab9`.
Includes protocol, diagnostic scores, all64 game records, per-decision Q/menu
hash/logits/V/return, Chinese explanation, verification, scripts and logs.
This numerical calibration bundle is not advertised as a replay bundle; the
original diagnostic YRP/engine evidence remains in the prior new64-reviewed pack.

Three startup attempts were retained: native does not accept max_step_loss
(training handles it); EnvPreprocess has an optional None field; Gymnasium
step returns five values without the statistics wrapper. No complete first
episodes were produced in those attempts. Same planned seeds were retained.
Independent delivery audit rechecked all64 unique first episodes, contiguous
decision steps, selected Q slots, natural terminal rewards+/-1, return signs,
RMSE and unchanged native hash. Source/runtime production files were untouched.
