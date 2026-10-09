# Effect semantics and public state candidate — 2026-10-09

Status: local implementation, native compilation and real-core paired validation
passed. Not promoted into an active training runtime. No strength improvement
claimed.

## Scope

- Isolated branch `codex/effect-public-state-20261009`, based on `b2cb4c4`.
- Opt-in `structured-state-v2`: description-bound action features, public chain,
  and per-turn public activation/status counts.
- Zero-initialized model residuals and explicit checkpoint migration.
- Teaching code and the original checkout's native edits are untouched.
- See [contract and usage](../docs/effect-public-state-v2.md) for limitations.

## Completed evidence

Command: `python -m pytest tests/test_effect_semantics.py tests/test_public_state_model.py -q`.
Result: **9 passed in 51.22 seconds** on the isolated Windows CPU environment
(JAX/JAXLIB 0.4.35, Flax 0.10.2).

Covered: conservative description binding, ambiguity rejection, clone override,
deterministic assets, asset tamper detection, unchanged v1 tensor contracts,
zero-residual migration output parity (policy/value/recurrent state), finite
nonzero residual gradients, action permutation equivariance, and sensitivity
to the new inputs when residuals are enabled. Migration parity used synthetic
model parameters and fixtures; no production checkpoint was migrated.

Built and validated real local semantic assets from the existing Chinese card
database, code list and Lua scripts. Of **13,492** listed cards, **6,570** have at
least one bound effect; **9,394** effect descriptions are bound. Coverage is
syntax/declaration coverage, not a proof of effect behavior. Unbound descriptions
retain the whole-card representation.

| Asset/input | SHA-256 |
| --- | --- |
| Code list | `f54528c6927d7ddd92d411b1423560700fc07c008eba86235e7edb963c45e437` |
| Cards database | `5f13245de4e665450858f66ef2f736a3d2f48cc7f0036961373a96a650cb6797` |
| Lua script corpus | `bf2fa7979d40a45438ab55a418be130a481200ec7732164ef185e13672d83271` |
| New effect descriptions | `2d56806a9145081022522774036b7bde0092dd68bf57897ff12a62d9f4b4b148` |
| Existing static table | `c1bb0a92be94d003a97a56833fa9735edcec123fd6d2b40834f4dc9459ae8858` |
| Existing effect tags | `dcfae0a76bd8bb5bb2819222ace06f927b6f14951a80d155343cb9d244ffebbd` |
| Existing tag confidence | `1dcecdad4fd970c42c80dadb06d6e5a1cd664a11ed8afe7140872918ea3b025a` |

## Native verification completed after upload approval

The user approved the candidate source, assets and build-script upload. The
independent `ygo-build-cache:replay-fix` container completed the standalone
C++17 ledger test and portable CPython 3.10 native build, with native optimization
disabled. Native SHA-256:
`81c653e0a94bcef354e8b7754f4b8b802eb2436b424ab8276f3fbc89ffbf80d0`.
The compiled source is `4becb50`; rebasing onto `8ad0ab0` did not change any
native or model implementation files.

Four paired Sky Striker self-play duels, seeds 1092026..1092029, completed
**1,624 decisions**. Every old tensor, reward, termination flag, option count
and acting seat matched between v1 and v2. All games reported invalid_game=0.
Observed cumulative row counts: chain=420, turn ledger=2,827, bound legal
action effects=425. These count row observations across decisions, not distinct
activations. Public ledger identities passed the observed-disclosure check.
See [raw paired results](effect-public-state-native-20261009.json).

The native binary and build/validation logs are preserved locally under
`ygoai/dist/effect-public-state-4becb50/`. Existing production modules and
checkpoints were not replaced. This establishes bounded interface correctness;
it does not establish playing strength or exhaustive card-rule coverage.

## Remaining synchronization

The isolated branch includes the latest completed teaching commits. Preserve
the primary checkout's outstanding BO3 native edits when integrating. Home SSH
timed out at 100.67.112.51:22 during this run; the home fast-forward and subsequent
H200 source deployment remain pending under the required synchronization order.

The remote source upload was rejected by automatic approval review because
authorization of the source payload and build-server destination was not
explicit. No upload from that rejected command occurred. The later explicit
approval enabled the isolated build and real-core validation recorded above.
