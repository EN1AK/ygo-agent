import sys
import time
import os
import random
import hashlib
import json
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict, replace
import atexit

import _repo_bootstrap  # noqa: F401
import ygoenv
import numpy as np

import tyro

from ygoai.utils import init_ygopro
from ygoai.rl.utils import RecordEpisodeStatistics
from ygoai.rl.checkpoint_compat import (
    sha256_file, validate_checkpoint_compatibility,
)
from ygoai.rl.counterfactual import observation_digest
from ygoai.rl.cycle_guard import PolicyCycleGuard, public_state_digest
from ygoai.rl.env import VersionedObservation
from ygoai.rl.match_report import FirstEpisodeCollector
from ygoai.rl.observation_schema import (
    DEFAULT_GROUP_REFERENCES, DEFAULT_PUBLIC_EVENTS, LEGACY_SCHEMA,
)
from ygoai.rl.jax.agent import RNNAgent, ModelArgs
from ygoai.windbot import WindBotConfig, WindBotProcess, allocate_port, validate_config, write_metadata
from ygoai.windbot_protocol import LegacyChainProxy


@dataclass
class Args:
    seed: int = 1
    """the random seed"""

    env_id: str = "YGOPro-v1"
    """the id of the environment"""
    deck: str = "../assets/deck"
    """the deck file to use"""
    deck1: Optional[str] = None
    """the deck name for the first player, for example, `Hero`"""
    deck2: Optional[str] = None
    """the deck name for the second player, for example, `CyberDragon`"""
    code_list_file: str = "code_list.txt"
    """the code list file for card embeddings"""
    lang: str = "chinese"
    """the language to use"""
    max_options: int = 24
    """the maximum number of options"""
    n_history_actions: int = 32
    """the number of history actions to use"""
    max_steps: int = 1000
    """maximum external policy decisions before an invalid forced terminal"""
    observation_schema: str = LEGACY_SCHEMA
    """versioned observation schema"""
    semantic_asset_dir: str = ""
    """generated Structured-lite semantic asset directory"""
    n_public_events: int = DEFAULT_PUBLIC_EVENTS
    """bounded public event history"""
    max_group_references: int = DEFAULT_GROUP_REFERENCES
    """maximum references in each set-valued action role"""
    num_embeddings: Optional[int] = None
    """the number of embeddings of the agent"""

    player: int = -1
    """the player to play as, -1 means random, 0 is the first player, 1 is the second player"""
    play: bool = False
    """whether to play the game"""
    verbose: bool = False
    """whether to print debug information"""
    record: bool = False
    """whether to record the game as YGOPro replays"""
    decision_log: Optional[str] = None
    """optional JSONL path for model decisions and the terminal result"""
    diagnostic_dir: Optional[str] = None
    """new directory for selected actor-visible observation/RNN fixtures (not engine snapshots)"""
    diagnostic_steps: tuple[int, ...] = ()
    """zero-based model decision steps to capture; no effect on action selection"""
    cycle_log: Optional[str] = None
    """optional JSONL path for detected deterministic policy cycles"""
    cycle_guard: bool = False
    """opt-in diagnostic: replace a repeated argmax with the next-ranked legal action"""
    cycle_max_period: int = 8
    """maximum decision distance considered a policy cycle"""

    num_episodes: int = 1024
    """the number of episodes to run""" 
    num_envs: int = 64
    """the number of parallel game environments"""
    first_episode_per_env: bool = False
    """Count each initial duel once, not the fastest N completed games; requires episodes=envs."""

    bot_type: Literal["random", "greedy", "first", "windbot"] = "greedy"
    """the type of bot to use"""
    strategy: Literal["random", "greedy"] = "greedy"
    """the strategy to use if agent is not used"""

    m: ModelArgs = field(default_factory=lambda: ModelArgs())
    """the model arguments for the agent1"""

    checkpoint: Optional[str] = None
    """the checkpoint to load, must be a `flax_model` file"""

    xla_device: Optional[str] = None
    """the XLA device to use, `cpu` for forcing running on CPU"""

    env_threads: Optional[int] = None
    """the number of threads to use for envpool, defaults to `num_envs`"""

    windbot_executable: Optional[str] = None
    """path to WindBot.exe or compatible launcher"""
    windbot_workdir: Optional[str] = None
    """WindBot working directory, defaults to the executable directory"""
    windbot_deck: str = "AI_Default"
    """WindBot deck name or .ydk path"""
    windbot_name: str = "WindBot"
    """WindBot player name"""
    windbot_host: str = "127.0.0.1"
    """host WindBot should connect to"""
    windbot_port: int = 0
    """port WindBot should connect to, 0 means allocate a free local port for validation"""
    windbot_host_info: str = ""
    """WindBot HostInfo argument"""
    windbot_password: str = ""
    """WindBot Password argument"""
    windbot_dialog: Optional[str] = None
    """WindBot Dialog argument"""
    windbot_timeout: float = 30.0
    """WindBot connection/setup timeout in seconds"""
    windbot_legacy_chain: bool = True
    """Adapt legacy chain and confirm-card messages to the deployed WindBot"""
    windbot_log_dir: str = "logs/windbot"
    """directory for WindBot logs"""
    windbot_mono: Optional[str] = None
    """optional mono executable for running WindBot.exe on non-Windows systems"""
    windbot_source_revision: Optional[str] = None
    """optional WindBot source revision or build identifier for run metadata"""
    windbot_metadata: Optional[str] = "logs/windbot/eval-windbot-metadata.json"
    """path to write WindBot run metadata"""
    windbot_server_mode: bool = False
    """launch WindBot in srvpro-style HTTP server mode before adding a bot"""
    windbot_server_host: str = "127.0.0.1"
    """WindBot server-mode HTTP host"""
    windbot_server_port: int = 2399
    """WindBot server-mode HTTP port"""


def create_agent(args):
    return RNNAgent(
        **asdict(args.m),
        embedding_shape=args.num_embeddings,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.diagnostic_dir or args.diagnostic_steps:
        if (not args.diagnostic_dir or not args.diagnostic_steps or not args.checkpoint
                or args.num_envs != 1 or args.num_episodes != 1
                or min(args.diagnostic_steps) < 0):
            raise ValueError('diagnostics require directory, nonnegative steps, checkpoint, and one duel/env')
        Path(args.diagnostic_dir).mkdir(parents=True, exist_ok=False)
    if (args.cycle_guard or args.cycle_log) and not args.checkpoint:
        raise ValueError("cycle detection requires --checkpoint")
    args.m.observation_schema = args.observation_schema
    windbot_config = None
    if args.num_embeddings is None:
        with open(args.code_list_file, "r", encoding="utf-8-sig") as f:
            args.num_embeddings = sum(1 for line in f if line.strip())
    if args.bot_type == "windbot":
        if not args.windbot_executable:
            raise ValueError("--windbot-executable is required when --bot_type windbot")
        windbot_port = args.windbot_port or allocate_port(args.windbot_host)
        windbot_config = WindBotConfig(
            executable=args.windbot_executable,
            workdir=args.windbot_workdir,
            deck=args.windbot_deck,
            name=args.windbot_name,
            host=args.windbot_host,
            port=windbot_port,
            host_info=args.windbot_host_info,
            password=args.windbot_password,
            dialog=args.windbot_dialog,
            timeout=args.windbot_timeout,
            log_dir=args.windbot_log_dir,
            mono=args.windbot_mono,
            source_revision=args.windbot_source_revision,
            server_mode=args.windbot_server_mode,
            server_host=args.windbot_server_host,
            server_port=args.windbot_server_port,
        )
        info = validate_config(windbot_config, check_port=True)
        if args.windbot_metadata:
            write_metadata(windbot_config, args.windbot_metadata)
        print(f"Validated WindBot setup: {info}")
        if args.num_envs != 1 or args.num_episodes != 1:
            raise ValueError("WindBot evaluation currently runs one complete duel per invocation; use num_envs=1 and num_episodes=1")
    if args.play or args.record:
        args.num_envs = 1
        args.verbose = True
        print("Set num_envs=1 and verbose=True for recording or playing the game")
        if args.record and not os.path.exists("replay"):
            os.makedirs("replay")

    args.env_threads = min(args.env_threads or args.num_envs, args.num_envs)
    if args.first_episode_per_env and args.num_episodes != args.num_envs:
        raise ValueError("first_episode_per_env requires num_episodes == num_envs")
    if args.decision_log and (args.num_envs != 1 or args.num_episodes != 1):
        raise ValueError("A complete decision log requires one environment and one episode")

    deck, deck_names = init_ygopro(args.env_id, args.lang, args.deck, args.code_list_file, return_deck_names=True)

    args.deck1 = args.deck1 or deck
    args.deck2 = args.deck2 or deck

    seed = args.seed + 100000
    random.seed(seed)
    seed = random.randint(0, int(1e8))
    random.seed(seed)
    np.random.seed(seed)

    if args.xla_device is not None:
        os.environ.setdefault("JAX_PLATFORMS", args.xla_device)

    num_envs = args.num_envs

    envs = ygoenv.make(
        task_id=args.env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        num_threads=args.env_threads,
        seed=seed,
        deck1=args.deck1,
        deck2=args.deck2,
        player=args.player,
        max_options=args.max_options,
        n_history_actions=args.n_history_actions,
        max_steps=args.max_steps,
        play_mode='human' if args.play else args.bot_type.replace("greedy", "bot"),
        windbot_host=args.windbot_host,
        windbot_port=windbot_config.port if windbot_config else 0,
        windbot_timeout=int(args.windbot_timeout),
        async_reset=False,
        verbose=args.verbose,
        record=args.record,
        observation_schema=args.observation_schema,
        semantic_asset_dir=args.semantic_asset_dir,
        n_public_events=args.n_public_events,
        max_group_references=args.max_group_references,
    )
    envs.num_envs = num_envs
    envs = VersionedObservation(envs, args.observation_schema)
    obs_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)

    if args.checkpoint:
        import jax
        import jax.numpy as jnp
        import flax
        from jax.experimental.compilation_cache import compilation_cache as cc
        cc.set_cache_dir(os.path.expanduser("~/.cache/jax"))

        agent = create_agent(args)
        key = jax.random.PRNGKey(seed)
        sample_obs = jax.tree.map(lambda x: jnp.array([x]), obs_space.sample())

        rstate = agent.init_rnn_state(1)
        params = jax.jit(agent.init)(key, sample_obs, rstate)

        semantic_hash = None
        if args.semantic_asset_dir:
            semantic_metadata = Path(args.semantic_asset_dir) / "metadata.json"
            if semantic_metadata.exists():
                semantic_hash = sha256_file(semantic_metadata)
        validate_checkpoint_compatibility(
            args.checkpoint,
            observation_schema=args.observation_schema,
            model_args=args.m,
            semantic_table_hash=semantic_hash,
            code_list_hash=sha256_file(args.code_list_file),
            capacities={
                "max_cards": 80,
                "max_options": args.max_options,
                "history_actions": args.n_history_actions,
                "public_events": args.n_public_events,
                "group_references": args.max_group_references,
            },
        )

        with open(args.checkpoint, "rb") as f:
            params = flax.serialization.from_bytes(params, f.read())
        checkpoint_sha256 = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()

        params = jax.device_put(params)
        rstate = agent.init_rnn_state(num_envs)

        @jax.jit
        def get_probs_and_value(params, rstate, obs, done):
            next_rstate, logits, value = agent.apply(params, obs, rstate)[:3]
            probs = jax.nn.softmax(logits, axis=-1)
            next_rstate = jax.tree.map(
                lambda x: jnp.where(done[:, None], 0, x), next_rstate)
            return next_rstate, logits, probs, value

        def predict_fn(rstate, obs, done):
            rstate, logits, probs, value = get_probs_and_value(params, rstate, obs, done)
            return rstate, np.array(logits), np.array(probs), np.array(value)

        print(f"loaded checkpoint from {args.checkpoint}")


    windbot_process = None
    windbot_proxy = None
    if windbot_config:
        if args.windbot_legacy_chain:
            windbot_proxy = LegacyChainProxy(windbot_config.host, windbot_config.port, windbot_config.timeout)
            windbot_proxy.start()
            atexit.register(windbot_proxy.stop)
            windbot_config = replace(windbot_config, port=windbot_proxy.port)
        windbot_process = WindBotProcess(windbot_config)
        windbot_process.start()
        if windbot_config.server_mode:
            windbot_process.add_bot()
    try:
        obs, infos = envs.reset()
    except Exception:
        if windbot_process:
            windbot_process.stop()
        raise
    next_to_play = infos['to_play']
    dones = np.zeros(num_envs, dtype=np.bool_)

    episode_rewards = []
    episode_lengths = []
    win_rates = []
    win_reasons = []
    raw_win_reasons = []
    invalid_games = []
    termination_reasons = []
    first_episodes = FirstEpisodeCollector(num_envs) if args.first_episode_per_env else None

    step = 0
    decision_stream = None
    if args.decision_log:
        decision_path = Path(args.decision_log).expanduser().resolve()
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_stream = decision_path.open("w", encoding="utf-8")
    cycle_stream = None
    if args.cycle_log:
        cycle_path = Path(args.cycle_log).expanduser().resolve()
        cycle_path.parent.mkdir(parents=True, exist_ok=True)
        cycle_stream = cycle_path.open("w", encoding="utf-8")
    cycle_guard = PolicyCycleGuard(
        enabled=args.cycle_guard, max_cycle_period=args.cycle_max_period)
    action_prefixes = [[] for _ in range(num_envs)]
    cycle_detections = 0
    cycle_interventions = 0
    start = time.time()
    start_step = step

    deck_names = sorted(deck_names)
    deck_times = {name: 0 for name in deck_names}
    deck_time_count = {name: 0 for name in deck_names}

    model_time = env_time = 0
    captured_steps = set()
    while True:
        if start_step == 0 and len(episode_lengths) > int(args.num_episodes * 0.1):
            start = time.time()
            start_step = step
            model_time = env_time = 0

        if args.checkpoint:
            _start = time.time()
            capture = step in args.diagnostic_steps
            if capture:
                input_rstate = [np.asarray(x).copy() for x in jax.tree.leaves(rstate)]
            rstate, logits, probs, value = predict_fn(rstate, obs, dones)
            if args.verbose:
                print(f"probs: {[f'{p:.4f}' for p in probs[probs != 0].tolist()]}")
                print(f"value: {value[0][0]}")
            raw_actions = probs.argmax(axis=1)
            actions = raw_actions.copy()
            for env_index in range(num_envs):
                count = int(infos['num_options'][env_index])
                player = int(next_to_play[env_index])
                fingerprint = public_state_digest(
                    obs, env_index=env_index, num_options=count, player=player)
                cycle = cycle_guard.select(
                    env_index=env_index, step=step,
                    state_fingerprint=fingerprint,
                    policy_logits=logits[env_index], num_options=count)
                actions[env_index] = cycle.selected_action
                if cycle.detected:
                    cycle_detections += 1
                    cycle_interventions += int(cycle.intervened)
                    cycle_record = cycle.to_json()
                    cycle_record.update({
                        "record_type": "policy_cycle",
                        "mode": "next_ranked_guard" if args.cycle_guard else "raw",
                        "player": player,
                        "checkpoint": str(Path(args.checkpoint).resolve()),
                        "checkpoint_sha256": checkpoint_sha256,
                        "state_value": float(value[env_index][0]),
                        "observation_digest": observation_digest({
                            key: np.asarray(item[env_index:env_index + 1])
                            for key, item in obs.items()
                        }),
                        "snapshot": {
                            "seed": seed,
                            "actions": list(action_prefixes[env_index]),
                            "player": player,
                            "play_mode": args.bot_type.replace("greedy", "bot"),
                            "controlled_player": player,
                            "replayable": (
                                num_envs == 1 and len(episode_lengths) == 0),
                            "deck1": args.deck1,
                            "deck2": args.deck2,
                            "max_options": args.max_options,
                            "n_history_actions": args.n_history_actions,
                            "observation_schema": args.observation_schema,
                        },
                        "decision_id": f"eval-structured:{env_index}:{step}",
                    })
                    if cycle_stream:
                        cycle_stream.write(json.dumps(
                            cycle_record, ensure_ascii=False) + "\n")
            if decision_stream:
                count = int(infos['num_options'][0])
                global_features = np.asarray(obs['global_'][0])
                action_features = np.asarray(obs['actions_'][0, :count])
                phase_names = {
                    0: "draw", 1: "standby", 2: "main1",
                    3: "battle_start", 4: "battle_step", 5: "damage",
                    6: "damage_calculation", 7: "battle", 8: "main2", 9: "end",
                }
                legal_actions = []
                for index in range(count):
                    item = {"index": index, "legacy_features": action_features[index].tolist()}
                    if 'action_features_' in obs:
                        item.update(
                            structured_features=np.asarray(obs['action_features_'][0, index]).tolist(),
                            single_references=np.asarray(obs['action_single_refs_'][0, index]).tolist(),
                            group_references=np.asarray(obs['action_group_refs_'][0, index]).tolist(),
                            group_reference_mask=np.asarray(obs['action_group_mask_'][0, index]).tolist(),
                        )
                    legal_actions.append(item)
                decision_stream.write(json.dumps({
                    "record_type": "decision", "step": step,
                    "player": int(next_to_play[0]), "model": "checkpoint",
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    "turn": int(global_features[4]),
                    "phase_id": int(global_features[5]),
                    "phase": phase_names.get(int(global_features[5]), "unknown"),
                    "legal_actions": legal_actions,
                    "raw_selected_action": int(raw_actions[0]),
                    "selected_action": int(actions[0]),
                    "cycle_guard_intervened": bool(
                        int(raw_actions[0]) != int(actions[0])),
                    "policy_logits": logits[0, :count].tolist(),
                    "policy_probabilities": probs[0, :count].tolist(),
                    "state_value": float(value[0][0]),
                }, ensure_ascii=False) + "\n")
            if capture:
                from ygoai.rl.decision_fixture import save_decision_fixture
                save_decision_fixture(
                    args.diagnostic_dir, step=step, observation=obs,
                    recurrent_leaves=input_rstate, dones=dones, logits=logits,
                    probabilities=probs, value=value, selected_action=int(actions[0]),
                    num_options=int(infos['num_options'][0]),
                    metadata={'checkpoint_sha256': checkpoint_sha256,
                              'seed': seed, 'requested_seed': args.seed,
                              'player': int(next_to_play[0]),
                              'observation_schema': args.observation_schema,
                              'deck1': args.deck1, 'deck2': args.deck2,
                              'purpose': 'diagnostic_only_not_training_labels'})
                captured_steps.add(step)
            model_time += time.time() - _start
        else:
            if args.strategy == "random":
                actions = np.random.randint(infos['num_options'])
            else:
                actions = np.zeros(num_envs, dtype=np.int32)

        to_play = next_to_play

        _start = time.time()
        obs, rewards, dones, infos = envs.step(actions)
        for env_index, action in enumerate(actions):
            action_prefixes[env_index].append(int(action))
        next_to_play = infos['to_play']
        env_time += time.time() - _start

        step += 1

        finished_indices = (first_episodes.take(dones) if first_episodes is not None
                            else np.flatnonzero(dones))
        for idx in finished_indices:
            if len(episode_lengths) >= args.num_episodes:
                break
            win_reason = infos['win_reason'][idx]
            episode_length = infos['l'][idx]
            episode_reward = infos['r'][idx]
            win = int(episode_reward > 0)
            invalid = int(infos.get('invalid_game', np.zeros(num_envs, dtype=np.int32))[idx])
            termination_reason = int(
                infos.get('termination_reason', np.zeros(num_envs, dtype=np.int32))[idx])
            episode_steps = int(
                infos.get('episode_steps', np.zeros(num_envs, dtype=np.int32))[idx])
            turn_count = int(
                infos.get('turn_count', np.zeros(num_envs, dtype=np.int32))[idx])

            episode_lengths.append(episode_length)
            episode_rewards.append(episode_reward)
            win_rates.append(win)
            win_reasons.append(1 if win_reason == 1 else 0)
            raw_win_reasons.append(int(win_reason))
            invalid_games.append(invalid)
            termination_reasons.append(termination_reason)
            cycle_guard.reset(idx)
            action_prefixes[idx].clear()
            sys.stderr.write(
                f"Episode {len(episode_lengths)}: length={episode_length}, "
                f"reward={episode_reward}, win={win}, win_reason={win_reason}, "
                f"invalid_game={invalid}, termination_reason={termination_reason}, "
                f"episode_steps={episode_steps}, turn_count={turn_count}, "
                f"environment_index={idx}\n")
        if len(episode_lengths) >= args.num_episodes:
            break

    if decision_stream:
        decision_stream.write(json.dumps({
            "record_type": "terminal", "steps": step,
            "terminal_reward": float(episode_rewards[-1]),
            "win": int(win_rates[-1]), "win_reason": raw_win_reasons[-1],
            "invalid_game": int(invalid_games[-1]),
            "termination_reason": int(termination_reasons[-1]),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_sha256,
        }, ensure_ascii=False) + "\n")
        decision_stream.close()
    if cycle_stream:
        cycle_stream.close()

    natural = np.logical_not(np.asarray(invalid_games, dtype=np.bool_))
    natural_wins = np.asarray(win_rates, dtype=np.float64)[natural]
    natural_reasons = np.asarray(win_reasons, dtype=np.float64)[natural]
    natural_win_rate = float(np.mean(natural_wins)) if natural_wins.size else float('nan')
    natural_win_reason = float(np.mean(natural_reasons)) if natural_reasons.size else float('nan')
    termination_counts = {
        int(reason): int(count) for reason, count in zip(
            *np.unique(np.asarray(termination_reasons, dtype=np.int32), return_counts=True))
    }
    print(
        f"len={np.mean(episode_lengths):.4f}, reward={np.mean(episode_rewards):.4f}, "
        f"win_rate={natural_win_rate:.4f}, win_reason={natural_win_reason:.4f}, "
        f"natural_games={int(natural.sum())}, invalid_games={int(np.sum(invalid_games))}, "
        f"invalid_rate={np.mean(invalid_games):.4f}, "
        f"termination_counts={json.dumps(termination_counts, sort_keys=True)}, "
        f"cycle_mode={'next_ranked_guard' if args.cycle_guard else 'raw'}, "
        f"cycle_detections={cycle_detections}, "
        f"cycle_interventions={cycle_interventions}")
    if not args.play:
        total_time = time.time() - start
        total_steps = (step - start_step) * num_envs
        print(f"SPS: {total_steps / total_time:.0f}, total_steps: {total_steps}")
        print(f"total: {total_time:.4f}, model: {model_time:.4f}, env: {env_time:.4f}")
    if windbot_process:
        windbot_process.stop()
    if windbot_proxy:
        windbot_proxy.stop()
        if windbot_proxy.error:
            raise RuntimeError(windbot_proxy.error)
        print(f'WindBot legacy protocol messages translated: {windbot_proxy.converted}')
    if set(args.diagnostic_steps) != captured_steps:
        raise RuntimeError(f'requested diagnostic steps not reached: {set(args.diagnostic_steps) - captured_steps}')

