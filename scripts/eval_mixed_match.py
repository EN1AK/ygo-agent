import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import _repo_bootstrap  # noqa: F401
import flax
import jax
import jax.numpy as jnp
import numpy as np
import tyro
import ygoenv

from ygoai.rl.checkpoint_compat import sha256_file, validate_checkpoint_compatibility
from ygoai.rl.env import VersionedObservation
from ygoai.rl.jax.agent import ModelArgs, RNNAgent
from ygoai.rl.match_report import battle_report
from ygoai.rl.observation_schema import (
    DEFAULT_GROUP_REFERENCES,
    DEFAULT_PUBLIC_EVENTS,
    LEGACY_SCHEMA,
)
from ygoai.rl.utils import EnvPreprocess, RecordEpisodeStatistics
from ygoai.utils import init_ygopro


STRUCTURED_ONLY_FIELDS = frozenset({
    "visible_card_ids_", "card_semantics_", "effect_tags_",
    "effect_tag_confidence_", "selection_", "action_features_",
    "action_single_refs_", "action_group_refs_", "action_group_mask_",
    "public_events_", "public_event_refs_", "structured_diagnostics_",
})


@dataclass
class Args:
    checkpoint: str
    opponent_checkpoint: str
    output: Optional[str] = None
    seed: int = 1
    num_episodes: int = 128
    env_threads: Optional[int] = None
    env_id: str = "YGOPro-v1"
    deck: str = "../assets/deck"
    deck1: Optional[str] = None
    deck2: Optional[str] = None
    code_list_file: str = "code_list.txt"
    lang: str = "chinese"
    max_options: int = 24
    max_steps: int = 1000
    n_history_actions: int = 32
    observation_schema: str = LEGACY_SCHEMA
    checkpoint_schema: str = LEGACY_SCHEMA
    opponent_schema: str = LEGACY_SCHEMA
    checkpoint_variant: str = "full"
    opponent_variant: str = "full"
    ablate_structured: bool = False
    """Zero every structured-only tensor while preserving legacy inputs."""
    semantic_asset_dir: str = ""
    n_public_events: int = DEFAULT_PUBLIC_EVENTS
    max_group_references: int = DEFAULT_GROUP_REFERENCES
    num_embeddings: Optional[int] = None
    xla_device: Optional[str] = None
    m1: ModelArgs = field(default_factory=ModelArgs)
    m2: ModelArgs = field(default_factory=ModelArgs)


def _semantic_hash(asset_dir: str) -> Optional[str]:
    if not asset_dir:
        return None
    metadata = Path(asset_dir) / "metadata.json"
    return sha256_file(metadata) if metadata.exists() else None


def _load(
    checkpoint: str,
    agent: RNNAgent,
    key,
    sample_obs,
    model_args,
    checkpoint_schema: str,
    args: Args,
):
    rstate = agent.init_rnn_state(1)
    variables = jax.jit(agent.init)(key, sample_obs, rstate)
    validate_checkpoint_compatibility(
        checkpoint,
        observation_schema=checkpoint_schema,
        model_args=model_args,
        semantic_table_hash=(
            _semantic_hash(args.semantic_asset_dir)
            if checkpoint_schema == args.observation_schema else None
        ),
        code_list_hash=sha256_file(args.code_list_file),
        capacities={
            "max_cards": 80,
            "max_options": args.max_options,
            "history_actions": args.n_history_actions,
            "public_events": args.n_public_events,
            "group_references": args.max_group_references,
        } if checkpoint_schema == args.observation_schema else None,
    )
    with open(checkpoint, "rb") as stream:
        return flax.serialization.from_bytes(variables, stream.read())


def main() -> None:
    args = tyro.cli(Args)
    if args.num_episodes < 2:
        raise ValueError("num_episodes must be at least 2 so both seats are tested")
    if args.xla_device:
        os.environ.setdefault("JAX_PLATFORMS", args.xla_device)

    args.m1.observation_schema = args.checkpoint_schema
    args.m1.structured_variant = args.checkpoint_variant
    args.m2.observation_schema = args.opponent_schema
    args.m2.structured_variant = args.opponent_variant
    if args.num_embeddings is None:
        with open(args.code_list_file, "r", encoding="utf-8-sig") as stream:
            args.num_embeddings = sum(1 for line in stream if line.strip())

    deck, _ = init_ygopro(
        args.env_id, args.lang, args.deck, args.code_list_file, return_deck_names=True
    )
    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck
    random.seed(args.seed + 100000)
    env_seed = random.randint(0, int(1e8))
    np.random.seed(env_seed)
    env_threads = min(args.env_threads or args.num_episodes, args.num_episodes)

    envs = ygoenv.make(
        task_id=args.env_id,
        env_type="gymnasium",
        num_envs=args.num_episodes,
        num_threads=env_threads,
        seed=env_seed,
        deck1=args.deck1,
        deck2=args.deck2,
        max_options=args.max_options,
        max_steps=args.max_steps,
        n_history_actions=args.n_history_actions,
        async_reset=False,
        greedy_reward=True,
        play_mode="self",
        timeout=600,
        oppo_info=False,
        observation_schema=args.observation_schema,
        semantic_asset_dir=args.semantic_asset_dir,
        n_public_events=args.n_public_events,
        max_group_references=args.max_group_references,
    )
    envs.num_envs = args.num_episodes
    envs = VersionedObservation(envs, args.observation_schema)
    sample_obs = jax.tree.map(lambda x: jnp.array([x]), envs.observation_space.sample())
    envs = EnvPreprocess(envs, skip_mask=True)
    envs = RecordEpisodeStatistics(envs)

    agent1 = RNNAgent(embedding_shape=args.num_embeddings, **asdict(args.m1))
    agent2 = RNNAgent(embedding_shape=args.num_embeddings, **asdict(args.m2))
    key1, key2 = jax.random.split(jax.random.PRNGKey(env_seed))
    params1 = jax.device_put(
        _load(
            args.checkpoint, agent1, key1, sample_obs,
            args.m1, args.checkpoint_schema, args,
        )
    )
    params2 = jax.device_put(
        _load(
            args.opponent_checkpoint, agent2, key2, sample_obs,
            args.m2, args.opponent_schema, args,
        )
    )
    rstate1 = jax.device_put(agent1.init_rnn_state(args.num_episodes))
    rstate2 = jax.device_put(agent2.init_rnn_state(args.num_episodes))

    @jax.jit
    def predict(p1, p2, obs, state1, state2, main, done):
        if args.ablate_structured:
            obs = {
                name: jnp.zeros_like(value) if name in STRUCTURED_ONLY_FIELDS else value
                for name, value in obs.items()
            }
        next_state1, logits1 = agent1.apply(p1, obs, state1)[:2]
        next_state2, logits2 = agent2.apply(p2, obs, state2)[:2]
        logits = jnp.where(main[:, None], logits1, logits2)
        state1 = jax.tree.map(
            lambda new, old: jnp.where(main[:, None], new, old), next_state1, state1
        )
        state2 = jax.tree.map(
            lambda new, old: jnp.where(main[:, None], old, new), next_state2, state2
        )
        state1, state2 = jax.tree.map(
            lambda x: jnp.where(done[:, None], 0, x), (state1, state2)
        )
        return state1, state2, logits.argmax(axis=1)

    start = time.time()
    predict_fn = lambda *values: predict(params1, params2, *values)
    match_report = battle_report(
        envs, args.num_episodes, predict_fn, rstate1, rstate2
    )
    elapsed = time.time() - start
    envs.close()

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "opponent_checkpoint": str(Path(args.opponent_checkpoint).resolve()),
        "opponent_checkpoint_sha256": sha256_file(args.opponent_checkpoint),
        "observation_schema": args.observation_schema,
        "checkpoint_schema": args.checkpoint_schema,
        "opponent_schema": args.opponent_schema,
        "checkpoint_variant": args.checkpoint_variant,
        "opponent_variant": args.opponent_variant,
        "ablate_structured": args.ablate_structured,
        "semantic_table_hash": _semantic_hash(args.semantic_asset_dir),
        "code_list_hash": sha256_file(args.code_list_file),
        "seed": args.seed,
        "environment_seed": env_seed,
        "num_episodes": args.num_episodes,
        "candidate_first_player_games": args.num_episodes // 2,
        "candidate_second_player_games": args.num_episodes - args.num_episodes // 2,
        "max_steps": args.max_steps,
        **match_report,
        "elapsed_seconds": elapsed,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
