"""Resolved multi-deck run identity, telemetry, and linked compute manifests."""

from __future__ import annotations

import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

from ygoai.deck_corpus import CorpusError, sha256_file


def resume_sampler_counters(actors, local_envs, actor_count, *, repartition=False):
    """Restore counters; explicit repartition preserves their flat logical order.

    Repartition is distributional continuation, not exact RNG restoration:
    envpool seeds also depend on the actor topology. Total env count must stay
    fixed when changing per-actor width. Never silently drop or duplicate state.
    """
    if not actors:
        return None
    if local_envs < 1 or actor_count < 1:
        raise ValueError('Sampler topology counts must be positive')
    if set(actors) != {str(i) for i in range(len(actors))}:
        raise ValueError('Sampler actor IDs must be contiguous')
    rows = [actors[str(i)] for i in range(len(actors))]
    widths = {len(row) for row in rows}
    if len(widths) != 1 or 0 in widths:
        raise ValueError('Inconsistent sampler counter widths')
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for row in rows for x in row):
        raise ValueError('Invalid sampler counter')
    if widths != {local_envs}:
        if not repartition:
            raise ValueError('Per-actor env count changed; explicit sampler repartition required')
        flat = [x for row in rows for x in row]
        if len(flat) != local_envs * actor_count:
            raise ValueError('Sampler repartition must preserve total environment count')
        rows = [flat[i:i+local_envs] for i in range(0,len(flat),local_envs)]
    return [','.join(map(str,row)) for row in rows]


def resolve_training_context(
    corpus_manifest: str, cluster_manifest: str, curriculum_manifest: str,
    elfnote_reserve: float,
) -> dict[str, Any]:
    paths = [Path(value) for value in (corpus_manifest, cluster_manifest, curriculum_manifest)]
    if not all(path.is_file() for path in paths):
        raise CorpusError("corpus, cluster, and curriculum manifests must all exist")
    curriculum = json.loads(paths[2].read_text(encoding="utf-8"))
    declared = curriculum["anchor"]["reserve_ppm_per_seat"] / 1_000_000
    if abs(declared - elfnote_reserve) > 1e-12:
        raise CorpusError(f"elfnote reserve {elfnote_reserve} != curriculum {declared}")
    return {
        "corpus_revision": json.loads(paths[0].read_text(encoding="utf-8"))["revision"],
        "cluster_revision": json.loads(paths[1].read_text(encoding="utf-8"))["revision"],
        "curriculum_revision": curriculum["revision"],
        "corpus_hash": sha256_file(paths[0]), "cluster_hash": sha256_file(paths[1]),
        "curriculum_hash": sha256_file(paths[2]), "elfnote_reserve_per_seat": declared,
    }


def linked_compute_manifest(args, devices: list[str], parent_checkpoint_hash: str | None) -> dict[str, Any]:
    actor_count = len(args.actor_device_ids) * args.num_actor_threads
    return {
        "schema_version": 1, "run_name": args.run_name,
        "parent_checkpoint_sha256": parent_checkpoint_hash,
        "continuation_semantics": "linked-distributional-not-bit-identical-across-topology",
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "devices": devices,
        "topology": {
            "world_size": args.world_size, "local_num_envs_per_actor": args.local_num_envs,
            "actor_threads": args.num_actor_threads, "actor_devices": args.actor_device_ids,
            "learner_devices": args.learner_device_ids, "actor_count": actor_count,
            "total_envs": args.num_envs, "local_batch_size": args.local_batch_size,
        },
    }


class SamplingTelemetry:
    def __init__(self):
        self.completed_games = 0
        self.invalid_games = 0
        self.invalid_terminations = Counter()
        self.learner_decisions = 0
        self.decks = Counter()
        self.families = Counter()
        self.clusters = Counter()
        self.seats = Counter()
        self.matchups = Counter()
        self.decision_decks = Counter()
        self.decision_families = Counter()
        self.decision_clusters = Counter()

    def decision(self, seat: int, info: dict[str, Any] | None = None, index: int | None = None) -> None:
        self.learner_decisions += 1
        self.seats[str(int(seat))] += 1
        if info is not None and index is not None and "deck_cluster" in info:
            cluster = int(info["deck_cluster"][index][seat])
            family = f"{cluster}:{int(info['deck_family'][index][seat])}"
            deck = f"{family}:{int(info['deck_member'][index][seat])}"
            self.decision_clusters[str(cluster)] += 1
            self.decision_families[family] += 1
            self.decision_decks[deck] += 1

    def game(self, info: dict[str, Any], index: int) -> None:
        if int(info.get("invalid_game", [0])[index]):
            self.invalid_games += 1
            reason = int(info.get("termination_reason", [0])[index])
            self.invalid_terminations[str(reason)] += 1
            return
        self.completed_games += 1
        clusters = [int(value) for value in info["deck_cluster"][index]]
        families = [f"{clusters[seat]}:{int(info['deck_family'][index][seat])}" for seat in (0, 1)]
        decks = [f"{families[seat]}:{int(info['deck_member'][index][seat])}" for seat in (0, 1)]
        for seat in (0, 1):
            self.clusters[str(clusters[seat])] += 1
            self.families[families[seat]] += 1
            self.decks[decks[seat]] += 1
        self.matchups["|".join(decks)] += 1

    def report(self) -> dict[str, Any]:
        if sum(self.decks.values()) != 2 * self.completed_games:
            raise RuntimeError("deck telemetry does not reconcile to completed games")
        if self.decision_decks and sum(self.decision_decks.values()) != self.learner_decisions:
            raise RuntimeError("deck decision telemetry does not reconcile to learner decisions")
        return {
            "completed_games": self.completed_games, "invalid_games": self.invalid_games,
            "invalid_termination_counts": dict(sorted(self.invalid_terminations.items())),
            "learner_decisions": self.learner_decisions,
            "deck_counts": dict(sorted(self.decks.items())),
            "family_counts": dict(sorted(self.families.items())),
            "cluster_counts": dict(sorted(self.clusters.items())),
            "seat_decision_counts": dict(sorted(self.seats.items())),
            "deck_decision_counts": dict(sorted(self.decision_decks.items())),
            "family_decision_counts": dict(sorted(self.decision_families.items())),
            "cluster_decision_counts": dict(sorted(self.decision_clusters.items())),
            "matchup_counts": dict(sorted(self.matchups.items())),
        }
