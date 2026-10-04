"""Resident model matrix for the existing generation evaluator's serial calls.

Imported by an opt-in shim in the isolated evaluation runtime. No trainer reload
is needed: the first call produces the matrix, later calls read verified results.
"""
import hashlib
import json
from pathlib import Path
import random
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def signature(argv):
    # All other CLI arguments, including model paths and environment capacities,
    # must be identical across requests for one cached matrix.
    result = []
    index = 0
    while index < len(argv):
        if argv[index] in ('--output', '--seed', '--candidate-seat'):
            index += 2
        else:
            result.append(argv[index])
            index += 1
    return result


def request(argv, root):
    def option(name):
        return argv[argv.index(name) + 1]
    if '--output' not in argv:
        return None
    output = Path(option('--output')).resolve()
    try:
        relative = output.relative_to(Path(root).resolve())
    except ValueError:
        return None
    if len(relative.parts) != 3 or relative.parts[1] not in ('screen', 'confirmation'):
        raise ValueError('unexpected generation evaluation output')
    generation, kind, _ = relative.parts
    index = int(generation.removeprefix('generation-'))
    assert generation == f'generation-{index:03d}' and index > 0
    start = 61050000 + index * 1000 + (100 if kind == 'confirmation' else 0)
    count = 16 if kind == 'confirmation' else 8
    seed, seat = int(option('--seed')), int(option('--candidate-seat'))
    assert start <= seed < start + count and seat in (0, 1)
    assert output.name == f'{seed}-seat{seat}.json'
    return dict(output=output, seed_start=start, seed_count=count, seed=seed,
                seat=seat, signature=signature(argv),
                candidate=Path(option('--checkpoint')).resolve(),
                opponent=Path(option('--opponent-checkpoint')).resolve())


def cached(req):
    marker = req['output'].parent / 'resident-matrix.json'
    if not marker.exists():
        return False
    manifest = json.loads(marker.read_text())
    assert manifest['status'] == 'completed'
    assert manifest['signature'] == req['signature']
    assert manifest['candidate_sha256'] == digest(req['candidate'])
    assert manifest['opponent_sha256'] == digest(req['opponent'])
    output = req['output']
    assert manifest['reports'][output.name] == digest(output)
    report = json.loads(output.read_text())
    assert report['seed'] == req['seed'] and report['candidate_seat'] == req['seat']
    print(json.dumps({'resident_cache_hit': str(output), 'attempts': report['attempts']}), flush=True)
    return True


def run_matrix(single, args, req):
    jax, jnp, np = single.jax, single.jnp, single.np
    directory = req['output'].parent
    assert req['seed'] == req['seed_start'] and req['seat'] == 0, 'matrix must start at its first scheduled request'
    assert args.num_episodes == 32 and not args.ablate_structured
    assert not list(directory.glob('*-seat*.json')), 'refuse partial result overwrite'
    started = time.monotonic()
    progress = dict(status='running', signature=req['signature'],
                    candidate_sha256=digest(req['candidate']), opponent_sha256=digest(req['opponent']),
                    seed_start=req['seed_start'], seed_count=req['seed_count'], reports={}, batches=[])
    atomic(directory / 'resident-progress.json', progress)
    args.m1.observation_schema, args.m1.structured_variant = args.checkpoint_schema, args.checkpoint_variant
    args.m2.observation_schema, args.m2.structured_variant = args.opponent_schema, args.opponent_variant
    if args.num_embeddings is None:
        args.num_embeddings = sum(bool(line.strip()) for line in Path(args.code_list_file).read_text(encoding='utf-8-sig').splitlines())
    deck, _ = single.init_ygopro(args.env_id, args.lang, args.deck, args.code_list_file, return_deck_names=True)
    args.deck1, args.deck2 = args.deck1 or deck, args.deck2 or deck
    agent1 = single.RNNAgent(embedding_shape=args.num_embeddings, **single.asdict(args.m1))
    agent2 = single.RNNAgent(embedding_shape=args.num_embeddings, **single.asdict(args.m2))
    params1 = params2 = None
    traces = []

    @jax.jit
    def predict(p1, p2, obs, state1, state2, main, done):
        traces.append(1)  # Runs at tracing time, not on each compiled invocation.
        next_state1, logits1 = agent1.apply(p1, obs, state1)[:2]
        next_state2, logits2 = agent2.apply(p2, obs, state2)[:2]
        logits = jnp.where(main[:, None], logits1, logits2)
        state1 = jax.tree.map(lambda new, old: jnp.where(main[:, None], new, old), next_state1, state1)
        state2 = jax.tree.map(lambda new, old: jnp.where(main[:, None], old, new), next_state2, state2)
        state1, state2 = jax.tree.map(lambda x: jnp.where(done[:, None], 0, x), (state1, state2))
        return state1, state2, logits.argmax(axis=1)

    try:
        for seed in range(req['seed_start'], req['seed_start'] + req['seed_count']):
            for seat in (0, 1):
                batch_start = time.monotonic()
                random.seed(seed + 100000)
                env_seed = random.randint(0, int(1e8))
                np.random.seed(env_seed)
                envs = single.ygoenv.make(
                    task_id=args.env_id, env_type='gymnasium', num_envs=args.num_episodes,
                    num_threads=min(args.env_threads or args.num_episodes, args.num_episodes),
                    seed=env_seed, deck1=args.deck1, deck2=args.deck2,
                    max_options=args.max_options, max_steps=args.max_steps,
                    n_history_actions=args.n_history_actions, async_reset=False,
                    greedy_reward=True, play_mode='self', timeout=600, oppo_info=False,
                    observation_schema=args.observation_schema, semantic_asset_dir=args.semantic_asset_dir,
                    n_public_events=args.n_public_events, max_group_references=args.max_group_references)
                try:
                    envs.num_envs = args.num_episodes
                    envs = single.VersionedObservation(envs, args.observation_schema)
                    if params1 is None:
                        sample = jax.tree.map(lambda x: jnp.array([x]), envs.observation_space.sample())
                        key1, key2 = jax.random.split(jax.random.PRNGKey(env_seed))
                        params1 = jax.device_put(single._load(args.checkpoint, agent1, key1, sample, args.m1, args.checkpoint_schema, args))
                        params2 = jax.device_put(single._load(args.opponent_checkpoint, agent2, key2, sample, args.m2, args.opponent_schema, args))
                    envs = single.RecordEpisodeStatistics(single.EnvPreprocess(envs, skip_mask=True))
                    state1 = jax.device_put(agent1.init_rnn_state(args.num_episodes))
                    state2 = jax.device_put(agent2.init_rnn_state(args.num_episodes))
                    duel_start = time.monotonic()
                    match = single.battle_report(envs, args.num_episodes,
                        lambda *values: predict(params1, params2, *values), state1, state2, candidate_seat=seat)
                    elapsed = time.monotonic() - duel_start
                finally:
                    envs.close()
                report = dict(checkpoint=str(req['candidate']), checkpoint_sha256=progress['candidate_sha256'],
                    opponent_checkpoint=str(req['opponent']), opponent_checkpoint_sha256=progress['opponent_sha256'],
                    observation_schema=args.observation_schema, checkpoint_schema=args.checkpoint_schema,
                    opponent_schema=args.opponent_schema, checkpoint_variant=args.checkpoint_variant,
                    opponent_variant=args.opponent_variant, ablate_structured=False,
                    semantic_table_hash=single._semantic_hash(args.semantic_asset_dir),
                    code_list_hash=digest(args.code_list_file), seed=seed, environment_seed=env_seed,
                    num_episodes=args.num_episodes, candidate_seat=seat, collection='first_episode_per_environment',
                    candidate_first_player_games=match['by_seat']['0']['attempts'],
                    candidate_second_player_games=match['by_seat']['1']['attempts'],
                    max_steps=args.max_steps, **match, elapsed_seconds=elapsed,
                    resident_batch_wall_seconds=time.monotonic() - batch_start)
                output = directory / f'{seed}-seat{seat}.json'
                atomic(output, report)
                progress['reports'][output.name] = digest(output)
                progress['batches'].append(dict(seed=seed, seat=seat,
                    wall_seconds=report['resident_batch_wall_seconds'], predict_traces=len(traces)))
                atomic(directory / 'resident-progress.json', progress)
                print(json.dumps(progress['batches'][-1]), flush=True)
                if match['invalid_games']:
                    raise RuntimeError('invalid games; halt without filling or replacing attempts')
        progress.update(status='completed', wall_seconds=time.monotonic() - started,
                        model_loads=2, predict_traces=len(traces))
        atomic(directory / 'resident-matrix.json', progress)
    except BaseException as error:
        progress.update(status='failed', error=repr(error), wall_seconds=time.monotonic() - started)
        atomic(directory / 'resident-progress.json', progress)
        raise
