"""Frozen hierarchical deck-sampling manifests and reference sampler."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from ygoai.deck_corpus import CorpusError, parse_ydk, sha256_bytes, sha256_file, stable_json


MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def build_sampling_manifest(
    clusters_path: Path,
    split_path: Path,
    canonical_path: Path,
    anchor_deck_path: Path,
    reserve_ppm: int = 250_000,
) -> dict[str, Any]:
    if not 0 < reserve_ppm < 1_000_000:
        raise CorpusError("anchor reserve_ppm must be between zero and one million")
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    training = set(split["train_decks"])
    anchor_hash = parse_ydk(anchor_deck_path).canonical_hash()
    anchor_canonical_ids = sorted(
        item["canonical_id"] for item in canonical["decks"]
        if item["canonical_hash"] == anchor_hash
    )
    training.difference_update(anchor_canonical_ids)
    grouped: dict[str, dict[str, list[str]]] = {}
    assignments = clusters["assignments"]
    for deck_id in sorted(training):
        item = assignments[deck_id]
        grouped.setdefault(item["cluster_id"], {}).setdefault(item["family_id"], []).append(deck_id)
    cluster_rows = []
    for cluster_index, cluster_id in enumerate(sorted(grouped)):
        families = []
        for family_index, family_id in enumerate(sorted(grouped[cluster_id])):
            decks = sorted(grouped[cluster_id][family_id])
            families.append({
                "family_id": family_id, "family_index": family_index,
                "decks": [{"deck_id": deck_id, "deck_index": index}
                          for index, deck_id in enumerate(decks)],
            })
        cluster_rows.append({"cluster_id": cluster_id, "cluster_index": cluster_index,
                             "families": families})
    if not cluster_rows:
        raise CorpusError("sampling manifest has no non-anchor training clusters")
    manifest = {
        "schema_version": 1, "revision": "deck-curriculum-v1",
        "selection": "anchor-or-equal-cluster-equal-family-equal-deck",
        "anchor": {
            "deck_id": "elfnote", "reserve_ppm_per_seat": reserve_ppm,
            "source_sha256": sha256_file(anchor_deck_path),
            "canonical_ids_excluded_from_non_anchor": anchor_canonical_ids,
        },
        "non_anchor_probability_ppm": 1_000_000 - reserve_ppm,
        "clusters": cluster_rows,
        "inputs": {
            "clusters_sha256": sha256_file(clusters_path),
            "split_sha256": sha256_file(split_path),
            "canonical_sha256": sha256_file(canonical_path),
        },
    }
    validate_sampling_manifest(manifest)
    manifest["manifest_hash"] = sha256_bytes(stable_json(manifest).encode("utf-8"))
    return manifest


def validate_sampling_manifest(manifest: dict[str, Any]) -> None:
    reserve = int(manifest["anchor"]["reserve_ppm_per_seat"])
    if reserve + int(manifest["non_anchor_probability_ppm"]) != 1_000_000:
        raise CorpusError("anchor and non-anchor probabilities do not normalize")
    clusters = manifest["clusters"]
    if not clusters:
        raise CorpusError("sampling manifest has no clusters")
    seen_decks = set()
    cluster_probability = (1_000_000 - reserve) / 1_000_000 / len(clusters)
    if not cluster_probability > 0:
        raise CorpusError("sampling cluster floor is zero")
    for expected_cluster, cluster in enumerate(clusters):
        if cluster["cluster_index"] != expected_cluster or not cluster["families"]:
            raise CorpusError("cluster indices must be dense and families nonempty")
        for expected_family, family in enumerate(cluster["families"]):
            if family["family_index"] != expected_family or not family["decks"]:
                raise CorpusError("family indices must be dense and decks nonempty")
            for expected_deck, deck in enumerate(family["decks"]):
                if deck["deck_index"] != expected_deck or deck["deck_id"] in seen_decks:
                    raise CorpusError("deck indices must be dense and deck ids unique")
                seen_decks.add(deck["deck_id"])


def write_sampling_manifest(manifest: dict[str, Any], json_path: Path, native_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["deck-sampler-v1", f"reserve_ppm\t{manifest['anchor']['reserve_ppm_per_seat']}",
             f"anchor\t{manifest['anchor']['deck_id']}"]
    for cluster in manifest["clusters"]:
        for family in cluster["families"]:
            for deck in family["decks"]:
                lines.append("\t".join([
                    "deck", str(cluster["cluster_index"]), str(family["family_index"]),
                    str(deck["deck_index"]), cluster["cluster_id"], family["family_id"],
                    deck["deck_id"],
                ]))
    native_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class HierarchicalDeckSampler:
    def __init__(self, manifest: dict[str, Any], seed: int, counter: int = 0):
        validate_sampling_manifest(manifest)
        self.manifest = manifest
        self.seed = int(seed) & MASK64
        self.counter = int(counter)
        if self.counter < 0:
            raise CorpusError("sampler counter cannot be negative")

    def _random(self) -> int:
        value = splitmix64((self.seed + self.counter) & MASK64)
        self.counter += 1
        return value

    def draw(self) -> dict[str, Any]:
        reserve = self.manifest["anchor"]["reserve_ppm_per_seat"]
        if self._random() % 1_000_000 < reserve:
            return {"is_anchor": True, "deck_id": self.manifest["anchor"]["deck_id"],
                    "cluster_id": "anchor", "family_id": "anchor",
                    "cluster_index": -1, "family_index": -1, "deck_index": -1}
        clusters = self.manifest["clusters"]
        cluster = clusters[self._random() % len(clusters)]
        family = cluster["families"][self._random() % len(cluster["families"])]
        deck = family["decks"][self._random() % len(family["decks"])]
        return {"is_anchor": False, "deck_id": deck["deck_id"],
                "cluster_id": cluster["cluster_id"], "family_id": family["family_id"],
                "cluster_index": cluster["cluster_index"],
                "family_index": family["family_index"], "deck_index": deck["deck_index"]}

    def draw_game(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.draw(), self.draw()

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "counter": self.counter,
                "manifest_hash": self.manifest.get("manifest_hash")}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("manifest_hash") != self.manifest.get("manifest_hash"):
            raise CorpusError("sampler manifest hash mismatch")
        if int(state["seed"]) != self.seed:
            raise CorpusError("sampler seed mismatch")
        self.counter = int(state["counter"])


def audit_sampler(manifest: dict[str, Any], seed: int, draws: int) -> dict[str, Any]:
    if draws < 1:
        raise CorpusError("draw count must be positive")
    sampler = HierarchicalDeckSampler(manifest, seed)
    anchors = Counter()
    clusters = Counter()
    families = Counter()
    decks = Counter()
    seats = Counter()
    for game in range((draws + 1) // 2):
        for seat, sample in enumerate(sampler.draw_game()):
            if sum(seats.values()) >= draws:
                break
            seats[seat] += 1
            anchors[sample["is_anchor"]] += 1
            clusters[sample["cluster_id"]] += 1
            families[sample["family_id"]] += 1
            decks[sample["deck_id"]] += 1
    expected_anchor = manifest["anchor"]["reserve_ppm_per_seat"] / 1_000_000
    observed_anchor = anchors[True] / draws
    standard_error = math.sqrt(expected_anchor * (1 - expected_anchor) / draws)
    nonanchor_clusters = [row["cluster_id"] for row in manifest["clusters"]]
    expected_cluster = (1 - expected_anchor) / len(nonanchor_clusters)
    cluster_tolerance = max(5 * math.sqrt(expected_cluster * (1 - expected_cluster) / draws),
                            0.0025)
    violations = []
    if abs(observed_anchor - expected_anchor) > 5 * standard_error:
        violations.append("anchor_frequency")
    for cluster_id in nonanchor_clusters:
        if abs(clusters[cluster_id] / draws - expected_cluster) > cluster_tolerance:
            violations.append(f"cluster_frequency:{cluster_id}")
    if any(clusters[cluster_id] == 0 for cluster_id in nonanchor_clusters):
        violations.append("cluster_coverage")
    return {
        "schema_version": 1, "draws": draws, "seed": seed,
        "manifest_hash": manifest.get("manifest_hash"),
        "expected_anchor_fraction": expected_anchor,
        "observed_anchor_fraction": observed_anchor,
        "anchor_five_sigma_tolerance": 5 * standard_error,
        "expected_non_anchor_cluster_fraction": expected_cluster,
        "cluster_tolerance": cluster_tolerance,
        "seat_counts": dict(sorted(seats.items())),
        "cluster_counts": dict(sorted(clusters.items())),
        "family_counts": dict(sorted(families.items())),
        "deck_counts": dict(sorted(decks.items())),
        "violations": violations, "passed": not violations,
        "final_state": sampler.state_dict(),
    }
