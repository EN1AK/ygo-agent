# Max-step loop training signal (2026-10-01)

The frozen 40M WindBot baseline contains 18 max-step terminations in 40 attempts. In one inspected Kashtira decision trace, the model repeatedly selected two cards and then unselected them, although a finish action was legal. This is a policy choice, not an unknown action or response-decoding failure.

Before this change, `YGOProEnv` returned reward zero with `invalid_game=1` and `termination_reason=2` at the 1000-step limit. `scripts/cleanba.py` then masked that terminal transition out of PPO. Consequently a loop that prevented a likely loss could be preferable to losing, and the final loop action received no corrective gradient.

Training now assigns a configurable `max_step_loss` (default 2.0) as a loss to the player acting when the limit is reached and keeps that terminal transition trainable. Other invalid-game reasons remain excluded. `max_step_loss=0` restores the old masking behavior. Evaluation rewards, invalid-game reporting, and the native action protocol are unchanged; unselect remains legal because it is needed for valid corrections. This changes future training only and does not repair the already-frozen 40M checkpoint.

The next comparison must use the same initialization, decks, compute budget, opponent mix, and evaluation seeds for legacy, relationship-only, and full features. Record both valid-game results and the max-step/invalid fraction; a lower timeout rate alone is not evidence that the features improve play. Re-evaluate a newly trained checkpoint against WindBot and Stage3, with decision logs for any residual loops.
