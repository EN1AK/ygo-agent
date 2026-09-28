"""Deterministic composition features, clustering, and family-safe splits."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix, save_npz

from ygoai.deck_corpus import CorpusError, sha256_bytes, sha256_file, stable_json, validate_artifact


def _matrix_hash(matrix: csr_matrix, row_ids: list[str], column_ids: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(stable_json({"rows": row_ids, "columns": column_ids,
                               "shape": list(matrix.shape)}).encode("utf-8"))
    for value in (matrix.indptr.astype("<i8"), matrix.indices.astype("<i8"),
                  matrix.data.astype("<f8")):
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_tfidf_features(
    ready_document: dict[str, Any], main_weight: float = 1.0, extra_weight: float = 1.5
) -> tuple[csr_matrix, dict[str, Any]]:
    if main_weight <= 0 or extra_weight <= 0:
        raise CorpusError("feature zone weights must be positive")
    decks = sorted((row["deck"] for row in ready_document["decks"]),
                   key=lambda item: item["canonical_id"])
    row_ids = [deck["canonical_id"] for deck in decks]
    columns = sorted({
        (zone, int(code))
        for deck in decks for zone in ("main", "extra")
        for code in deck["zones"].get(zone, {})
    }, key=lambda item: (0 if item[0] == "main" else 1, item[1]))
    column_ids = [f"{zone}:{code}" for zone, code in columns]
    position = {column: index for index, column in enumerate(columns)}
    document_frequency = Counter()
    for deck in decks:
        present = {
            (zone, int(code)) for zone in ("main", "extra")
            for code in deck["zones"].get(zone, {})
        }
        document_frequency.update(present)
    n = len(decks)
    idf = {column: np.log((1.0 + n) / (1.0 + document_frequency[column])) + 1.0
           for column in columns}
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row_index, deck in enumerate(decks):
        weighted = []
        for zone, zone_weight in (("main", main_weight), ("extra", extra_weight)):
            for raw_code, count in deck["zones"].get(zone, {}).items():
                column = (zone, int(raw_code))
                weighted.append((position[column], float(count) * idf[column] * zone_weight))
        norm = np.sqrt(sum(value * value for _, value in weighted))
        for column_index, value in sorted(weighted):
            rows.append(row_index)
            cols.append(column_index)
            values.append(value / norm if norm else 0.0)
    matrix = csr_matrix((np.asarray(values, dtype=np.float64), (rows, cols)),
                        shape=(n, len(columns)), dtype=np.float64)
    artifact = {
        "schema_version": 1, "feature_schema": "zone-tfidf-v1",
        "matrix_hash": _matrix_hash(matrix, row_ids, column_ids),
        "row_ids": row_ids, "column_ids": column_ids,
        "shape": list(matrix.shape), "nonzero": int(matrix.nnz),
        "weights": {"main": main_weight, "extra": extra_weight, "side": 0.0},
        "term_frequency": "zone-card-multiplicity",
        "inverse_document_frequency": "ln((1+n)/(1+df))+1",
        "normalization": "row-l2",
        "idf": [float(idf[column]) for column in columns],
    }
    validate_artifact("feature", artifact)
    return matrix, artifact


def silhouette_from_distances(distances: np.ndarray, labels: np.ndarray) -> float:
    values = []
    unique = sorted(set(int(value) for value in labels))
    if len(unique) < 2:
        return 0.0
    for index, label in enumerate(labels):
        same = np.flatnonzero(labels == label)
        same = same[same != index]
        if len(same) == 0:
            values.append(0.0)
            continue
        a = float(distances[index, same].mean())
        b = min(float(distances[index, np.flatnonzero(labels == other)].mean())
                for other in unique if other != label)
        values.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(values))


def _stable_cluster_id(members: Iterable[str]) -> str:
    digest = sha256_bytes("\n".join(sorted(members)).encode("ascii"))
    return f"cluster-{digest[:12]}"


def cluster_ready_families(
    matrix: csr_matrix,
    features: dict[str, Any],
    ready_document: dict[str, Any],
    family_document: dict[str, Any],
    candidate_k: Iterable[int] = range(32, 65, 4),
    min_families: int = 1,
    max_family_fraction: float = 0.25,
    max_deck_fraction: float = 0.30,
    cluster_penalty: float = 0.0015,
) -> dict[str, Any]:
    row_position = {deck_id: index for index, deck_id in enumerate(features["row_ids"])}
    ready_ids = set(row_position)
    family_members = {
        family["family_id"]: sorted(ready_ids.intersection(family["members"]))
        for family in family_document["families"]
    }
    family_members = {key: value for key, value in family_members.items() if value}
    family_ids = sorted(family_members)
    representative_ids = [family_members[family_id][0] for family_id in family_ids]
    representative_rows = np.asarray([row_position[deck_id] for deck_id in representative_ids])
    family_matrix = matrix[representative_rows]
    similarities = (family_matrix @ family_matrix.T).toarray()
    distances = np.clip(1.0 - similarities, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    hierarchy = linkage(squareform(distances, checks=False), method="average",
                        optimal_ordering=False)
    candidates = []
    labels_by_k = {}
    total_decks = len(ready_ids)
    for requested_k in sorted(set(int(value) for value in candidate_k)):
        if requested_k < 2 or requested_k >= len(family_ids):
            continue
        labels = fcluster(hierarchy, requested_k, criterion="maxclust").astype(np.int32)
        labels_by_k[requested_k] = labels
        family_sizes = Counter(int(value) for value in labels)
        deck_sizes = Counter()
        for family_id, label in zip(family_ids, labels):
            deck_sizes[int(label)] += len(family_members[family_id])
        violations = []
        if min(family_sizes.values()) < min_families:
            violations.append("minimum_family_count")
        if max(family_sizes.values()) / len(family_ids) > max_family_fraction:
            violations.append("dominant_family_fraction")
        if max(deck_sizes.values()) / total_decks > max_deck_fraction:
            violations.append("dominant_deck_fraction")
        silhouette = silhouette_from_distances(distances, labels)
        candidates.append({
            "requested_clusters": requested_k,
            "actual_clusters": len(family_sizes),
            "silhouette": silhouette,
            "selection_score": silhouette - cluster_penalty * len(family_sizes),
            "family_sizes": sorted(family_sizes.values()),
            "deck_sizes": sorted(deck_sizes.values()),
            "accepted": not violations, "violations": violations,
        })
    accepted = [item for item in candidates if item["accepted"]]
    if not accepted:
        raise CorpusError("all cluster candidates violate configured size constraints")
    selected = max(accepted, key=lambda item: (item["selection_score"], item["silhouette"],
                                                -item["actual_clusters"],
                                                -item["requested_clusters"]))
    labels = labels_by_k[selected["requested_clusters"]]
    groups: dict[int, list[str]] = defaultdict(list)
    for family_id, label in zip(family_ids, labels):
        groups[int(label)].append(family_id)
    clusters = []
    assignment = {}
    for raw_label, members_families in sorted(groups.items()):
        family_indices = [family_ids.index(family_id) for family_id in members_families]
        subdistance = distances[np.ix_(family_indices, family_indices)]
        medoid_local = int(np.argmin(subdistance.mean(axis=1)))
        medoid_family = members_families[medoid_local]
        member_decks = sorted(deck_id for family_id in members_families
                              for deck_id in family_members[family_id])
        cluster_id = _stable_cluster_id(member_decks)
        for family_id in members_families:
            for deck_id in family_members[family_id]:
                assignment[deck_id] = {"cluster_id": cluster_id, "family_id": family_id}
        cluster = {
            "cluster_id": cluster_id,
            "representative": family_members[medoid_family][0],
            "representative_family": medoid_family,
            "members": member_decks, "families": sorted(members_families),
            "metrics": {
                "deck_count": len(member_decks), "family_count": len(members_families),
                "mean_intra_family_rep_cosine_distance": float(
                    subdistance[np.triu_indices(len(family_indices), 1)].mean()
                    if len(family_indices) > 1 else 0.0
                ),
            },
        }
        validate_artifact("cluster", cluster)
        clusters.append(cluster)
    clusters.sort(key=lambda item: item["cluster_id"])
    return {
        "schema_version": 1, "algorithm": "average-linkage",
        "distance": "cosine-on-zone-tfidf-v1", "feature_matrix_hash": features["matrix_hash"],
        "configuration": {
            "candidate_k": sorted(set(int(value) for value in candidate_k)),
            "min_families": min_families,
            "max_family_fraction": max_family_fraction,
            "max_deck_fraction": max_deck_fraction,
            "cluster_penalty": cluster_penalty,
            "selection_score": "silhouette-cluster_penalty*actual_clusters",
            "tie_break": "selection-score-desc-then-silhouette-desc-then-fewer-clusters",
        },
        "selected": selected,
        "candidates": candidates,
        "linkage": hierarchy.tolist(),
        "family_order": family_ids,
        "family_representatives": representative_ids,
        "clusters": clusters,
        "assignments": assignment,
    }


def build_family_split(clusters: dict[str, Any], holdout_fraction: float = 0.10) -> dict[str, Any]:
    if not 0 < holdout_fraction < 1:
        raise CorpusError("holdout fraction must be between zero and one")
    train_families: list[str] = []
    held_out_families: list[str] = []
    for cluster in clusters["clusters"]:
        families = cluster["families"]
        ordered = sorted(families, key=lambda family: (
            sha256_bytes(f"deck-corpus-v1-split\n{family}".encode("utf-8")), family))
        holdout_count = min(len(ordered) - 1, max(1, round(len(ordered) * holdout_fraction)))
        held_out_families.extend(ordered[:holdout_count])
        train_families.extend(ordered[holdout_count:])
    train = set(train_families)
    held = set(held_out_families)
    if train & held:
        raise CorpusError("family split overlap")
    assignment = clusters["assignments"]
    train_decks = sorted(deck for deck, item in assignment.items() if item["family_id"] in train)
    held_decks = sorted(deck for deck, item in assignment.items() if item["family_id"] in held)
    split = {
        "schema_version": 1, "revision": "deck-corpus-v1-family-split-v1",
        "method": "per-cluster-stable-sha256", "holdout_fraction": holdout_fraction,
        "train_families": sorted(train), "held_out_families": sorted(held),
        "train_decks": train_decks, "held_out_decks": held_decks,
    }
    validate_artifact("split", split)
    return split


def write_cluster_artifacts(
    ready_path: Path, families_path: Path, output_dir: Path,
    main_weight: float = 1.0, extra_weight: float = 1.5,
    candidate_k: Iterable[int] = range(32, 65, 4), min_families: int = 1,
    max_family_fraction: float = 0.25, max_deck_fraction: float = 0.30,
    holdout_fraction: float = 0.10,
) -> dict[str, Any]:
    if output_dir.exists():
        raise CorpusError(f"cluster output already exists: {output_dir}")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    families = json.loads(families_path.read_text(encoding="utf-8"))
    matrix, features = build_tfidf_features(ready, main_weight, extra_weight)
    cluster_doc = cluster_ready_families(
        matrix, features, ready, families, candidate_k, min_families,
        max_family_fraction, max_deck_fraction,
    )
    split = build_family_split(cluster_doc, holdout_fraction)
    output_dir.mkdir(parents=True)
    save_npz(output_dir / "features.csr.npz", matrix, compressed=False)
    (output_dir / "features.json").write_text(
        json.dumps(features, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "clusters.json").write_text(
        json.dumps(cluster_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "split.json").write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "representative-audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cluster_id", "representative_deck", "representative_family",
                         "deck_count", "family_count", "manual_notes", "approved"])
        for cluster in cluster_doc["clusters"]:
            writer.writerow([cluster["cluster_id"], cluster["representative"],
                             cluster["representative_family"], cluster["metrics"]["deck_count"],
                             cluster["metrics"]["family_count"], "", ""])
    manifest = {
        "schema_version": 1, "revision": "deck-clusters-v1",
        "ready_table_hash": sha256_file(ready_path), "families_hash": sha256_file(families_path),
        "feature_matrix_hash": features["matrix_hash"],
        "features_artifact_hash": sha256_file(output_dir / "features.json"),
        "clusters_hash": sha256_file(output_dir / "clusters.json"),
        "split_hash": sha256_file(output_dir / "split.json"),
        "ready_decks": len(features["row_ids"]), "clusters": len(cluster_doc["clusters"]),
        "families": len(cluster_doc["family_order"]),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
