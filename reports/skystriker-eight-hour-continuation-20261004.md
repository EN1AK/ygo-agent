# User-authorized eight-hour SkyStriker PPO continuation

Window: 2026-10-04 approximately03:07 through11:07:19 Asia/Shanghai, deadline epoch1791083239. Includes startup and serial evaluations; no automatic extension. Parent121208832, SHA256 `cde8e3f6a06000319cbe3fe59f6112b2ffa37f5ece03b2f05b1ef06f16a03649`. User supersedes the suggested10M budget and requests5M saves/20M evaluation for eight hours.

Run: `/home/ygo/ygo-agent/training-runs/skystriker-121m-eight-hours-20261004`; initial launcher PID149890. Schedule and active child identity: `schedule.json`. Script `training-runs/run-skystriker-eight-hours-20261004.py`. Original immutable PPO source505b855d9e1e0a2363a3d81cf49d6097f7539555; native a2dbd2604fec0e01ad722d887c639c00ed4c5aa74c912bd8f37d69f57a815486. Evaluation source isolated65443f7. Parent sidecar and165 finite arrays verified before launch, deck/code-list/procedure/native hashes checked.

PPO4actors x32env,4envthreads/actor, horizon64, batch8192,64minibatches128, LR3e-5; two self-play actors, one frozen100M history actor, one in-process bot. Both decks use pinned SkyStriker. Qoff, UPGOoff, concurrencyoff, no search or belief. Each segment initializes fresh Adam, as existing PPO files contain actor weights rather than optimizer state; this is explicitly declared, not exact optimizer continuation.

Saves every611updates=5,005,312 steps. Four intervals give20,021,248 steps per evaluation segment (batch rounding). The first endpoint is141,230,080; subsequent endpoints add20,021,248. Checkpoints are validated against sidecar and decoded for165 finite arrays. Retain all saves. No benchmark/smoke weights are promoted.

Preflight: config-only request followed by two real updates from original121M parent; verify mixed-opponent modes, finite metrics and checkpoint. Baseline evaluation then64attempts versusSkyStriker WindBot,32seeds2026104100..2026104131 both seats. Each20M endpoint repeats the exact seeds and seats; rawargmax, no cycle guard, maxsteps1000,120sec perattempt,4isolatedworkers. Preserve invalid attempts without replacement and all replay/engine/decision evidence. No simultaneous GPU training/evaluation. WindBot sourceb0a2355f00bd59491add14ff09efa9914a3b47c6.

Failure, nonfinite optimizer, logstall or >10% invalid evaluation stops the schedule for investigation. No blind retries. At deadline stop the owned active process group and retain the newest verified save; unsaved progress since the last5M save can be lost and must not be reported as retained training. A deadline-interrupted evaluation remains partial. No new segment starts in the final five minutes. Intermediate evaluations are monitoring data, not a fresh final test or proof the last checkpoint is best.

Heartbeat `monitor-skystriker-home-20m` repurposed for this newly authorized run; do not restart its earlier interrupted256-game evaluation. Normal progress quiet; report20M results, failures/stalls/nonfinite and final window completion. H200 remains unavailable/pending synchronization. User BO3/TRAINING_PLAN edits are untouched. Existing Q2M backup monitor is independent and must not launch GPU work.

## First 20M evaluation (2026-10-04 05:41 Beijing)

Segment01 completed 2444 finite optimizer updates with exit0, reaching141230080. Endpoint `segment-01/checkpoints/104202611_step_000141230080.flax_model` SHA256 `31896f9ad28de8c5dbe1988310badb1ca1769b9dd8f7c2faacd76252bddd8127`; independent sidecar/hash and165 finite-array verification passed. All four5M interval saves are retained.

Same64 seed/seat attempts: baseline121M50W14L (78.125%), endpoint141M43W21L (67.1875%); both0 invalid,64 unique seed/seat pairs, cleanup complete and64 replay files per arm. Baseline seat0/seat1 each25/32 wins; endpoint23/32 and20/32. Paired transitions:36 both win,7 loss-to-win,14 win-to-loss,7 both lose. Seed-cluster paired difference is -10.9375 percentage points;20000 bootstrap samples give95% interval[-25,+3.125] points. This monitoring sample shows no improvement evidence but does not establish general regression; it is not an independent final test. Full replay/decision audit remains pending.

Controller continued segment02 unchanged, as authorized: trainer169752, start141230080, target161251328. At05:41, learner141434880,25 finite updates, SPS2541, cumulative nonfinite0. No automatic rollback, opponent/config change, Q promotion or extension of the eight-hour deadline.

## Second 20M evaluation (2026-10-04 08:02 Beijing)

Segment02 completed2444 finite optimizer updates, reaching161251328. Endpoint `segment-02/checkpoints/104202612_step_000161251328.flax_model` SHA256 `12b8211532ecea3194913d256bf8148abdc2d9f514cd079d279ef1f9f7b5676b`; independent sidecar/hash and165 finite-array checks passed. Eight5M interval saves now retained.

Eval02:40W24L/64 (62.5%),0 invalid, cleanup complete,64 unique seed/seat attempts and64 replay files. Seat0 is19/32 and seat1 is21/32. Against the same121M baseline50W14L:32 both win,8 loss-to-win,18 win-to-loss,6 both lose. Seed-cluster paired difference -15.625 percentage points;20000 bootstrap resamples (seed20261004) give95% interval[-29.6875,-3.125] points. This is evidence of poorer performance on this monitored seed block, not a fresh independent test, nor proof of its cause. Full replay strategy audit remains pending.

Segment03 started automatically with unchanged configuration, trainer179588,161251328 to181272576. At08:01 learner161447936/SPS2504; all38 optimizer rows checked at08:02 finite with zero nonfinite. Continue the authorized window without automatic rollback; retain the121M reference and all intermediate candidates.

## Third 20M evaluation (2026-10-04 10:22 Beijing)

Segment03 completed2444 finite optimizer updates, reaching181272576. Endpoint `segment-03/checkpoints/104202613_step_000181272576.flax_model` SHA256 `12a09d51245d68c7a5f80bb208f7a50e50d14d6451f4cab509decc916985b65e`; independent sidecar/hash and165 finite-array checks passed. Twelve5M saves retained.

Eval03:49W15L/64 (76.5625%),0 invalid, cleanup complete,64 unique seed/seat attempts and64 replay files. Seat0 is26/32 and seat1 is23/32. Against baseline121M50W14L:40 both win,9 loss-to-win,10 win-to-loss,5 both lose. Seed-cluster paired difference -1.5625 percentage points;20000 bootstrap resamples (seed20261004) give95% interval[-15.625,+12.5] points. Recovery relative to161M is observed on this repeated monitoring block, but no improvement over121M is established; do not claim general superiority or an independent final test.

Segment04 automatically started unchanged, trainer189017,181272576 to201293824 nominal target, still bounded by11:07:19 Beijing. At10:21 learner181542912/SPS2514,33 finite updates with zero nonfinite. The remaining window is insufficient for another full20M at current throughput; retain the last5M save and explicitly distinguish deadline-discarded updates from saved progress. No extension or new evaluation is inferred.

## Window closure (2026-10-04 11:11 Beijing)

Deadline interruption occurred11:07:19; supervisor finished11:07:50 after shutdown escalation (exit -15), not a spontaneous training failure. `completed.txt` says `authorized eight-hour window ended`; root status `window_completed`, active process null, launcher149890 and trainer189017 absent. The nested segment04 status remains stale `running`; it is not evidence of a live process.

Last completed learner update188522496; saved endpoint186277888, so2244608 completed updates' steps were not retained. Saved continuation adds65069056 steps to121208832. Endpoint `segment-04/checkpoints/104202614_step_000186277888.flax_model`, SHA256 `dc29c11254c0418a4b43cc29083db3d05e17b5c8e3dacd5c27b17d63c8670721`. All13 interval checkpoints independently rehashed against schedule and sidecars and decoded to165 finite arrays each. All8217 optimizer log rows (2444+2444+2444+885) finite, cumulative nonfinite0. Final sustained console SPS2537-2538.

Home artifact root: `/home/ygo/ygo-agent/training-runs/skystriker-121m-eight-hours-20261004`. Training logs: `segment-01/train.log` through `segment-04/train.log`; schedule: `schedule.json`. Evaluation evidence: `baseline121m/`, `eval-01/`, `eval-02/`, `eval-03/`, each64 playable replays and raw engine/decision logs (65 JSONL and192 log files counted per arm). These are remote artifacts, not a downloaded/commentated delivery bundle; full per-duel Chinese analysis remains pending.

Monitoring results:121M50/64,141M43/64,161M40/64,181M49/64; all256 attempts valid. Original121M is the highest observed point estimate, not statistically proven best.181M is the strongest evaluated continuation by point estimate.186M was not evaluated and must not be promoted as best. No further training or evaluation launched after the window.
