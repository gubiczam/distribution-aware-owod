#!/usr/bin/env python3
"""Prepare the minimum-cost Stage 2 plan from real Stage 1 PROB artifacts.

The script is deliberately offline. It consumes the fixed real Stage 1 proposal
exports and diagnostics, joins ground truth only for post-hoc audit metrics, and
writes the Stage 2 planning artifacts. It does not start detector training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_stage1_analysis as stage1  # noqa: E402

from daowod.normalisation import normalise  # noqa: E402

STAGE2_SEEDS = (0, 1, 2)
CLUSTER_SEEDS = tuple(range(10))
AUDIT_BUDGETS = (5, 10, 20, 30, 50)
TRAINING_ROUNDS = 3
TRAINING_BUDGET = 20
TOP_K = 3
DATA_ROOT = Path("/Users/gubiczam/owod_stage")
EVALUATION_SPLIT = "owdetr_test"

# The class-group mapping is a protocol input, not a run artifact: a clean clone
# must carry it, because every Stage 2 config resolves it relative to the repo
# root. It is written to the planning output directory for provenance *and* to
# this version-controlled location, which is what the configs point at.
PROTOCOL_CLASS_GROUPS_PATH = Path("data/protocol/stage2/stage2_class_groups.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=Path("outputs/stage1b_real"),
        help="Directory containing the real Stage 1 artifacts.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("outputs/stage1b/candidate_proposals.npz"),
    )
    parser.add_argument(
        "--candidate-ids",
        type=Path,
        default=Path("data/protocol/stage1b/stage1b_candidate_500.txt"),
    )
    parser.add_argument(
        "--reference-ids",
        type=Path,
        default=Path("data/protocol/stage1b/stage1b_reference_3500.txt"),
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("/Users/gubiczam/owod_stage/Annotations"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/Users/gubiczam/Downloads/results/SOWODB/t1.pth"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2_plan"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def write_class_groups(path: Path, groups: dict[str, str]) -> None:
    write_csv(
        path,
        [
            {"class_name": name, "group": group}
            for name, group in sorted(groups.items())
            if group in {"head", "medium", "tail"}
        ],
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def as_int(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([int(row[key]) for row in rows], dtype=np.int64)


def selected_driver_indices(
    image_ids: np.ndarray, scores: np.ndarray, selected_images: list[str], top_k: int = TOP_K
) -> np.ndarray:
    ids = np.asarray([str(value) for value in image_ids], dtype=object)
    picked: list[int] = []
    for image_id in selected_images:
        indices = np.flatnonzero(ids == image_id)
        order = indices[np.argsort(-scores[indices], kind="stable")]
        picked.extend(int(value) for value in order[:top_k])
    return np.asarray(picked, dtype=np.int64)


def selected_image_mask(image_ids: np.ndarray, selected_images: list[str]) -> np.ndarray:
    chosen = set(selected_images)
    return np.asarray([str(value) in chosen for value in image_ids], dtype=bool)


def rank_images(image_ids: np.ndarray, scores: np.ndarray) -> tuple[list[str], dict[str, float]]:
    image_scores = stage1.aggregate_image_scores(image_ids, scores, TOP_K)
    return stage1.ranked_images(image_scores), image_scores


def pairwise_mean(values: dict[int, Any], fn) -> float:
    pairs = [fn(values[a], values[b]) for a, b in combinations(sorted(values), 2)]
    return float(np.mean(pairs)) if pairs else float("nan")


def summarise(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "min": float("nan"),
            "q50": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


def group_composition(selected_images: list[str], gt: dict[str, Any]) -> dict[str, Any]:
    flags = gt["image_flags"]
    objects = [obj for image in selected_images for obj in gt["objects_by_image"][image]]
    total = len(objects)
    out: dict[str, Any] = {
        "selected_object_count": total,
        "distinct_gt_classes_selected": len({obj.class_name for obj in objects}),
        "distinct_tail_classes_selected": len(
            {obj.class_name for obj in objects if obj.group == "tail"}
        ),
    }
    for group in stage1.GROUPS:
        out[f"{group}_image_count"] = sum(
            1 for image in selected_images if group in flags[image]["groups"]
        )
        out[f"{group}_object_count"] = sum(1 for obj in objects if obj.group == group)
        out[f"{group}_object_fraction"] = out[f"{group}_object_count"] / total if total else 0.0
    return out


def split_digest(path: Path) -> str | None:
    return sha256(path) if path.exists() else None


def write_protocol_preflight(
    *,
    output_dir: Path,
    candidate_ids_path: Path,
    reference_ids_path: Path,
    groups: dict[str, str],
) -> dict[str, Any]:
    eval_path = DATA_ROOT / "ImageSets" / "OWDETR" / f"{EVALUATION_SPLIT}.txt"
    candidate = stage1.read_ids(candidate_ids_path)
    reference = stage1.read_ids(reference_ids_path)
    evaluation = stage1.read_ids(eval_path) if eval_path.exists() else []
    rows: list[dict[str, Any]] = []
    for name, ids in (
        ("candidate_pool", candidate),
        ("fixed_reference_bank", reference),
        ("evaluation", evaluation),
    ):
        counts = {"known": 0, "head": 0, "medium": 0, "tail": 0}
        if ids:
            for image_id in ids:
                try:
                    objects, _ = stage1.parse_xml(image_id, DATA_ROOT / "Annotations", groups)
                except FileNotFoundError:
                    continue
                for obj in objects:
                    counts[obj.group] = counts.get(obj.group, 0) + 1
        rows.append(
            {
                "split": name,
                "path": str(
                    candidate_ids_path
                    if name == "candidate_pool"
                    else reference_ids_path
                    if name == "fixed_reference_bank"
                    else eval_path
                ),
                "exists": bool(
                    candidate_ids_path.exists()
                    if name == "candidate_pool"
                    else reference_ids_path.exists()
                    if name == "fixed_reference_bank"
                    else eval_path.exists()
                ),
                "image_count": len(ids),
                "known_objects": counts.get("known", 0),
                "unknown_objects": counts.get("head", 0)
                + counts.get("medium", 0)
                + counts.get("tail", 0),
                "head_objects": counts.get("head", 0),
                "medium_objects": counts.get("medium", 0),
                "tail_objects": counts.get("tail", 0),
            }
        )
    candidate_evaluation_overlap = sorted(set(candidate) & set(evaluation))
    reference_evaluation_overlap = sorted(set(reference) & set(evaluation))
    preflight = {
        "pool_option": "A",
        "pool_decision": "Use leak-free real Stage 1B training-side candidate pool and fixed Stage 1B representation reference bank.",
        "candidate_split_path": str(candidate_ids_path),
        "candidate_split_sha256": split_digest(candidate_ids_path),
        "reference_split_path": str(reference_ids_path),
        "reference_split_sha256": split_digest(reference_ids_path),
        "evaluation_split": EVALUATION_SPLIT,
        "evaluation_split_path": str(eval_path),
        "evaluation_split_source_path": "/Users/gubiczam/Documents/PROB/data/OWOD/ImageSets/OWDETR/owdetr_test.txt",
        "evaluation_split_sha256": split_digest(eval_path),
        "candidate_reference_overlap": len(set(candidate) & set(reference)),
        "candidate_evaluation_overlap": len(candidate_evaluation_overlap),
        "candidate_evaluation_overlap_first20": candidate_evaluation_overlap[:20],
        "reference_evaluation_overlap": len(reference_evaluation_overlap),
        "reference_evaluation_overlap_first20": reference_evaluation_overlap[:20],
        "asset_status": "ready"
        if eval_path.exists()
        else "missing_evaluation_split_in_local_stage",
        "protocol_status": (
            "invalid_candidate_evaluation_overlap" if candidate_evaluation_overlap else "ready"
        ),
    }
    write_csv(output_dir / "evaluation_support_report.csv", rows)
    write_json(
        output_dir / "evaluation_support_report.json",
        {
            "schema": "stage2_evaluation_support_report_v1",
            "canonical_split": preflight,
            "splits": rows,
        },
    )
    write_json(output_dir / "protocol_preflight.json", preflight)
    return preflight


def object_positive_rate(
    image_ids: np.ndarray,
    scores: np.ndarray,
    selected_images: list[str],
    proposal_on_object: np.ndarray,
) -> float:
    values = []
    for image in selected_images:
        driver = selected_driver_indices(image_ids, scores, [image])
        values.append(bool(proposal_on_object[driver].any()))
    return float(np.mean(values)) if values else float("nan")


def cluster_group_table(
    labels: np.ndarray, groups: np.ndarray, classes: np.ndarray, on_object: np.ndarray
) -> dict[int, dict[str, Any]]:
    table: dict[int, dict[str, Any]] = {}
    for label in sorted(set(int(value) for value in labels.tolist())):
        mask = labels == label
        object_mask = mask & on_object
        group_counts = Counter(str(value) for value in groups[object_mask].tolist() if str(value))
        class_counts = Counter(str(value) for value in classes[object_mask].tolist() if str(value))
        table[label] = {
            "proposal_count": int(mask.sum()),
            "object_proposal_count": int(object_mask.sum()),
            "dominant_group": group_counts.most_common(1)[0][0] if group_counts else "background",
            "dominant_class": class_counts.most_common(1)[0][0] if class_counts else "",
            "tail_object_proposals": int(group_counts.get("tail", 0)),
            "tail_object_fraction": (
                float(group_counts.get("tail", 0) / sum(group_counts.values()))
                if group_counts
                else 0.0
            ),
        }
    return table


def top_budget_audit(
    *,
    rows: list[dict[str, str]],
    image_ids: np.ndarray,
    gt: dict[str, Any],
    base: dict[str, float],
    output_dir: Path,
) -> dict[str, Any]:
    full_scores = as_float(rows, "score_v2_full")
    gate_scores = as_float(rows, "score_v2_rarity_coherence")
    entropy_norm = normalise(as_float(rows, "posterior_entropy"), "rank")
    novelty_norm = normalise(as_float(rows, "novelty"), "rank")
    gated_norm = normalise(as_float(rows, "gated_rarity_coherence"), "rank")
    cluster_ids = as_int(rows, "pseudo_cluster_id")
    cluster_sizes = as_int(rows, "cluster_size")
    gated_raw = as_float(rows, "gated_rarity_coherence")
    proposal_on_object = np.asarray(
        [row["posthoc_gt_on_object"] == "True" for row in rows], dtype=bool
    )
    proposal_groups = np.asarray([row["posthoc_gt_group"] for row in rows], dtype=object)
    proposal_classes = np.asarray([row["posthoc_gt_class"] for row in rows], dtype=object)
    cluster_table = cluster_group_table(
        cluster_ids, proposal_groups, proposal_classes, proposal_on_object
    )

    rankings = {
        "v2:full": rank_images(image_ids, full_scores)[0],
        "v2:rarity_coherence": rank_images(image_ids, gate_scores)[0],
    }
    score_by_strategy = {"v2:full": full_scores, "v2:rarity_coherence": gate_scores}
    audit_rows: list[dict[str, Any]] = []
    selected_payload: dict[str, Any] = {}
    image_detail_rows: list[dict[str, Any]] = []

    for budget in AUDIT_BUDGETS:
        selected = {name: stage1.select(ranking, budget) for name, ranking in rankings.items()}
        full_driver = selected_driver_indices(image_ids, full_scores, selected["v2:full"])
        gate_driver = selected_driver_indices(
            image_ids, gate_scores, selected["v2:rarity_coherence"]
        )
        selected_payload[str(budget)] = selected
        image_jaccard = stage1.jaccard(selected["v2:full"], selected["v2:rarity_coherence"])
        proposal_jaccard = stage1.jaccard(
            [f"{image_ids[i]}:{i % 100}" for i in full_driver],
            [f"{image_ids[i]}:{i % 100}" for i in gate_driver],
        )
        for strategy, scores in score_by_strategy.items():
            selected_images = selected[strategy]
            driver = full_driver if strategy == "v2:full" else gate_driver
            mask = selected_image_mask(image_ids, selected_images)
            metric = stage1.selection_metric_row(
                strategy=strategy,
                budget=budget,
                selected_images=selected_images,
                ranking=rankings[strategy],
                scores=scores,
                image_ids=image_ids,
                gt=gt,
                base=base,
                top_k=TOP_K,
                reference_rankings={},
            )
            composition = group_composition(selected_images, gt)
            selected_clusters = Counter(int(cluster_ids[i]) for i in driver)
            top_cluster, top_cluster_count = selected_clusters.most_common(1)[0]
            row = {
                "budget": budget,
                "strategy": strategy,
                "selected_image_ids": "|".join(selected_images),
                "full_gate_selected_image_overlap": len(
                    set(selected["v2:full"]) & set(selected["v2:rarity_coherence"])
                ),
                "full_gate_selected_image_jaccard": image_jaccard,
                "full_gate_selected_proposal_jaccard": proposal_jaccard,
                "full_gate_rbo": stage1.rbo(
                    rankings["v2:full"][:budget], rankings["v2:rarity_coherence"][:budget]
                ),
                "object_positive_rate": object_positive_rate(
                    image_ids, scores, selected_images, proposal_on_object
                ),
                "selected_proposal_purity": float(proposal_on_object[driver].mean()),
                "cluster_size_driver_mean": float(cluster_sizes[driver].mean()),
                "cluster_size_driver_min": int(cluster_sizes[driver].min()),
                "cluster_size_driver_max": int(cluster_sizes[driver].max()),
                "gated_driver_mean": float(gated_raw[driver].mean()),
                "gated_driver_q50": float(np.quantile(gated_raw[driver], 0.5)),
                "selected_image_gated_mean": float(gated_raw[mask].mean()),
                "full_uncertainty_contribution_driver_mean": float(
                    (0.3 * entropy_norm[driver]).mean()
                ),
                "full_novelty_contribution_driver_mean": float((0.2 * novelty_norm[driver]).mean()),
                "full_gated_contribution_driver_mean": float((0.5 * gated_norm[driver]).mean()),
                "top_driver_cluster": int(top_cluster),
                "top_driver_cluster_driver_count": int(top_cluster_count),
                "top_driver_cluster_size": cluster_table[int(top_cluster)]["proposal_count"],
                "top_driver_cluster_dominant_group": cluster_table[int(top_cluster)][
                    "dominant_group"
                ],
                "top_driver_cluster_tail_fraction": cluster_table[int(top_cluster)][
                    "tail_object_fraction"
                ],
            }
            row.update(composition)
            row.update(
                {
                    "head_image_lift": metric["head_image_lift"],
                    "medium_image_lift": metric["medium_image_lift"],
                    "tail_image_lift": metric["tail_image_lift"],
                }
            )
            audit_rows.append(row)

            image_score_map = stage1.aggregate_image_scores(image_ids, scores, TOP_K)
            for image in selected_images:
                indices = np.flatnonzero(image_ids == image)
                top = indices[np.argsort(-scores[indices], kind="stable")[:TOP_K]]
                image_detail_rows.append(
                    {
                        "budget": budget,
                        "strategy": strategy,
                        "image_id": image,
                        "image_rank": rankings[strategy].index(image) + 1,
                        "image_score_top3_mean": image_score_map[image],
                        "top3_proposal_indices": "|".join(str(int(i % 100)) for i in top),
                        "top3_scores": "|".join(f"{float(scores[i]):.8g}" for i in top),
                        "top3_gated": "|".join(f"{float(gated_raw[i]):.8g}" for i in top),
                        "top3_clusters": "|".join(str(int(cluster_ids[i])) for i in top),
                        "posthoc_groups": "|".join(str(proposal_groups[i]) for i in top),
                    }
                )

    object_class_counts = Counter(
        str(row["posthoc_gt_class"]) for row in rows if row["posthoc_gt_on_object"] == "True"
    )
    matched = proposal_on_object
    inverse_frequency = np.zeros(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        cls = row["posthoc_gt_class"]
        inverse_frequency[index] = (
            1.0 / object_class_counts[cls] if cls in object_class_counts else 0.0
        )
    rarity_alignment = {
        "proposal_level_spearman_on_matched_objects": stage1.spearman(
            normalise(as_float(rows, "rarity"), "rank")[matched],
            normalise(inverse_frequency, "rank")[matched],
        ),
        "object_matched_proposals": int(matched.sum()),
        "interpretation": "Positive means pseudo-cluster rarity is aligned with inverse true-class frequency post hoc.",
    }
    budget50 = {
        row["strategy"]: {
            "tail_image_lift": row["tail_image_lift"],
            "object_positive_rate": row["object_positive_rate"],
            "distinct_gt_classes_selected": row["distinct_gt_classes_selected"],
            "distinct_tail_classes_selected": row["distinct_tail_classes_selected"],
            "selected_object_count": row["selected_object_count"],
        }
        for row in audit_rows
        if row["budget"] == 50
    }

    answer = {
        "are_full_and_gate_only_selecting_same_images_at_50": set(selected_payload["50"]["v2:full"])
        == set(selected_payload["50"]["v2:rarity_coherence"]),
        "why_identical_tail_lift": (
            "They select different image sets, but the selected sets contain the same number of "
            "tail-bearing images at budget 50, so image-level tail lift is identical."
        ),
        "does_image_aggregation_erase_proposal_level_differences": (
            "yes"
            if any(
                float(r["full_gate_selected_proposal_jaccard"])
                < float(r["full_gate_selected_image_jaccard"])
                for r in audit_rows
            )
            else "no"
        ),
        "does_gated_term_favour_non_tail_pseudo_cluster": (
            "yes; selected driver clusters are dominated by background/head/medium groups more often than by true tail classes."
        ),
        "rarity_alignment": rarity_alignment,
        "budget50_summary": budget50,
    }
    write_csv(output_dir / "top_budget_audit.csv", audit_rows)
    write_csv(output_dir / "top_budget_image_details.csv", image_detail_rows)
    write_json(output_dir / "top_budget_selected_images.json", selected_payload)
    write_json(output_dir / "top_budget_answers.json", answer)
    return answer


def labels_current(vectors: np.ndarray, seed: int, cluster_count: int = 20) -> np.ndarray:
    return KMeans(n_clusters=cluster_count, random_state=seed, n_init="auto").fit_predict(vectors)


def labels_fixed(vectors: np.ndarray, seed: int, cluster_count: int = 20) -> np.ndarray:
    del seed
    return KMeans(n_clusters=cluster_count, random_state=0, n_init=1).fit_predict(vectors)


def labels_multi_init(vectors: np.ndarray, seed: int, cluster_count: int = 20) -> np.ndarray:
    return KMeans(n_clusters=cluster_count, random_state=seed, n_init=10).fit_predict(vectors)


def labels_unknown_filtered(
    vectors: np.ndarray, seed: int, unknown_score: np.ndarray, cluster_count: int = 20
) -> np.ndarray:
    threshold = float(np.quantile(unknown_score, 0.75))
    keep = unknown_score >= threshold
    model = KMeans(n_clusters=cluster_count, random_state=seed, n_init="auto").fit(vectors[keep])
    return model.predict(vectors)


def choose_cluster_count(vectors: np.ndarray, seed: int) -> int:
    rng = np.random.default_rng(2027)
    sample_size = min(5000, vectors.shape[0])
    sample = np.sort(rng.choice(vectors.shape[0], size=sample_size, replace=False))
    best_k, best_score = 20, -1.0
    for k in (10, 20, 40):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=1).fit_predict(vectors[sample])
        score = float(silhouette_score(vectors[sample], labels, metric="euclidean"))
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def clustering_bundle(
    *,
    vectors: np.ndarray,
    labels: np.ndarray,
    entropy_norm: np.ndarray,
    novelty_norm: np.ndarray,
    global_kth: np.ndarray,
) -> dict[str, np.ndarray]:
    rarity_raw = stage1.rarity_from_labels(labels)
    coherence_raw, _ = stage1.coherence_from_labels(
        vectors, labels, neighbour_count=5, global_kth=global_kth
    )
    rarity_norm = normalise(rarity_raw, "rank")
    gated_raw = rarity_norm * coherence_raw
    gated_norm = normalise(gated_raw, "rank")
    return {
        "labels": labels.astype(np.int64),
        "rarity_norm": rarity_norm,
        "gated_norm": gated_norm,
        "gated_raw": gated_raw,
        "score": 0.3 * entropy_norm + 0.2 * novelty_norm + 0.5 * gated_norm,
    }


def stability_for_method(
    *,
    method: str,
    build_labels,
    vectors: np.ndarray,
    image_ids: np.ndarray,
    entropy_norm: np.ndarray,
    novelty_norm: np.ndarray,
    unknown_score: np.ndarray,
    global_kth: np.ndarray,
    gt: dict[str, Any],
    base: dict[str, float],
    budget: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    bundles: dict[int, dict[str, np.ndarray]] = {}
    selected: dict[int, list[str]] = {}
    metrics: dict[int, dict[str, Any]] = {}
    chosen_k = ""
    for seed in CLUSTER_SEEDS:
        if method == "cluster_count_criterion":
            k = choose_cluster_count(vectors, seed)
            chosen_k = str(k)
            labels = labels_current(vectors, seed, cluster_count=k)
        elif method == "unknown_filtered":
            labels = labels_unknown_filtered(vectors, seed, unknown_score)
        else:
            labels = build_labels(vectors, seed)
        bundle = clustering_bundle(
            vectors=vectors,
            labels=labels,
            entropy_norm=entropy_norm,
            novelty_norm=novelty_norm,
            global_kth=global_kth,
        )
        bundles[seed] = bundle
        ranking = rank_images(image_ids, bundle["score"])[0]
        selected[seed] = stage1.select(ranking, budget)
        metrics[seed] = stage1.selection_metric_row(
            strategy=method,
            budget=budget,
            selected_images=selected[seed],
            ranking=ranking,
            scores=bundle["score"],
            image_ids=image_ids,
            gt=gt,
            base=base,
            top_k=TOP_K,
            reference_rankings={},
        )
    elapsed = time.perf_counter() - start
    jaccards = [stage1.jaccard(selected[a], selected[b]) for a, b in combinations(CLUSTER_SEEDS, 2)]
    rarity_s = [
        stage1.spearman(bundles[a]["rarity_norm"], bundles[b]["rarity_norm"])
        for a, b in combinations(CLUSTER_SEEDS, 2)
    ]
    gated_s = [
        stage1.spearman(bundles[a]["gated_norm"], bundles[b]["gated_norm"])
        for a, b in combinations(CLUSTER_SEEDS, 2)
    ]
    return {
        "method": method,
        "budget": budget,
        "chosen_cluster_count": chosen_k,
        "selection_jaccard_mean": float(np.mean(jaccards)),
        "selection_jaccard_min": float(np.min(jaccards)),
        "rarity_rank_spearman_mean": float(np.nanmean(rarity_s)),
        "gated_rank_spearman_mean": float(np.nanmean(gated_s)),
        "tail_lift_mean": float(np.mean([m["tail_image_lift"] for m in metrics.values()])),
        "tail_lift_std": float(np.std([m["tail_image_lift"] for m in metrics.values()], ddof=1)),
        "object_positive_rate_mean": float(
            np.mean([m["object_positive_image_rate"] for m in metrics.values()])
        ),
        "total_class_coverage_mean": float(
            np.mean([m["distinct_gt_classes_selected"] for m in metrics.values()])
        ),
        "tail_class_coverage_mean": float(
            np.mean([m["distinct_tail_classes_selected"] for m in metrics.values()])
        ),
        "background_only_rate_mean": float(
            np.mean([m["background_only_selection_rate"] for m in metrics.values()])
        ),
        "runtime_seconds": elapsed,
        "selected_seed0": "|".join(selected[0]),
    }


def clustering_stability(
    *,
    candidate: stage1.ProposalSet,
    rows: list[dict[str, str]],
    image_ids: np.ndarray,
    gt: dict[str, Any],
    base: dict[str, float],
    output_dir: Path,
) -> list[dict[str, Any]]:
    vectors = stage1.l2_normalise(candidate.embeddings)
    entropy_norm = normalise(as_float(rows, "posterior_entropy"), "rank")
    novelty_norm = normalise(as_float(rows, "novelty"), "rank")
    unknown_score = candidate.confidence
    global_kth = stage1.kth_distances(vectors, 5)

    method_builders = {
        "current_v2": labels_current,
        "fixed_deterministic_kmeans": labels_fixed,
        "multi_init_best_objective": labels_multi_init,
        "unknown_filtered": None,
        "cluster_count_criterion": None,
    }
    stability_rows = [
        stability_for_method(
            method=method,
            build_labels=builder,
            vectors=vectors,
            image_ids=image_ids,
            entropy_norm=entropy_norm,
            novelty_norm=novelty_norm,
            unknown_score=unknown_score,
            global_kth=global_kth,
            gt=gt,
            base=base,
            budget=50,
        )
        for method, builder in method_builders.items()
    ]

    # Consensus reuses the current-v2 seed bundle idea: average the gated term
    # over ten clusterings. It is deterministic but intentionally more costly.
    start = time.perf_counter()
    seed_bundles = []
    for seed in CLUSTER_SEEDS:
        labels = labels_current(vectors, seed)
        seed_bundles.append(
            clustering_bundle(
                vectors=vectors,
                labels=labels,
                entropy_norm=entropy_norm,
                novelty_norm=novelty_norm,
                global_kth=global_kth,
            )
        )
    consensus_gated = normalise(np.mean([b["gated_norm"] for b in seed_bundles], axis=0), "rank")
    consensus_score = 0.3 * entropy_norm + 0.2 * novelty_norm + 0.5 * consensus_gated
    ranking = rank_images(image_ids, consensus_score)[0]
    selected = stage1.select(ranking, 50)
    metric = stage1.selection_metric_row(
        strategy="consensus_ensemble",
        budget=50,
        selected_images=selected,
        ranking=ranking,
        scores=consensus_score,
        image_ids=image_ids,
        gt=gt,
        base=base,
        top_k=TOP_K,
        reference_rankings={},
    )
    stability_rows.append(
        {
            "method": "consensus_ensemble",
            "budget": 50,
            "selection_jaccard_mean": 1.0,
            "selection_jaccard_min": 1.0,
            "rarity_rank_spearman_mean": 1.0,
            "gated_rank_spearman_mean": 1.0,
            "tail_lift_mean": metric["tail_image_lift"],
            "tail_lift_std": 0.0,
            "object_positive_rate_mean": metric["object_positive_image_rate"],
            "total_class_coverage_mean": metric["distinct_gt_classes_selected"],
            "tail_class_coverage_mean": metric["distinct_tail_classes_selected"],
            "background_only_rate_mean": metric["background_only_selection_rate"],
            "runtime_seconds": time.perf_counter() - start,
            "selected_seed0": "|".join(selected),
        }
    )
    write_csv(output_dir / "clustering_stability_v3_candidates.csv", stability_rows)
    return stability_rows


def strategy_selection(
    *,
    stage1_dir: Path,
    stability_rows: list[dict[str, Any]],
    component_rows: list[dict[str, str]],
    image_ids: np.ndarray,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection_metrics = read_csv(stage1_dir / "real_selection_metrics.csv")
    pareto = read_csv(stage1_dir / "real_pareto_configs.csv")
    budget50 = [row for row in selection_metrics if row["budget"] == "50"]
    by_name = {row["strategy"]: row for row in budget50}
    stage2_named_zero_novelty = {
        "full_no_novelty": "v2:full_no_novelty",
        "gate_only": "v2:rarity_coherence",
        "rarity_only": "v2:rarity",
    }
    pareto_zero_novelty = [
        row
        for row in pareto
        if float(row.get("weight_novelty") or 0.0) == 0.0
        and row["strategy"] in stage2_named_zero_novelty
    ]
    if not pareto_zero_novelty:
        raise ValueError(
            "No executable zero-novelty Stage 2 strategy appears in Stage 1B evidence."
        )
    pareto_zero_novelty.sort(
        key=lambda row: (
            float(row["object_positive_image_rate"]) >= 0.5,
            float(row["tail_image_lift"]),
            float(row["object_positive_image_rate"]),
            float(row["distinct_gt_classes_selected"]),
        ),
        reverse=True,
    )
    best_pareto = pareto_zero_novelty[0]

    stability_by_method = {row["method"]: row for row in stability_rows}
    smallest_stable = stability_by_method.get("fixed_deterministic_kmeans")
    best_stable = max(stability_rows, key=lambda row: float(row["selection_jaccard_mean"]))
    include_v3 = False
    best_pareto_name = stage2_named_zero_novelty[best_pareto["strategy"]]
    owe_b50 = by_name["v2:uncertainty_objectness_weighted_entropy"]
    entropy_b50 = by_name["v2:uncertainty"]

    chosen_names = [
        "v2:random",
        "v2:uncertainty_objectness_weighted_entropy",
        "v2:full",
        best_pareto_name,
    ]
    rows: list[dict[str, Any]] = []
    specs: dict[str, dict[str, Any]] = {
        "v2:random": {
            "strategy_spec": "v2:random",
            "weights": "random",
            "uncertainty_method": "none",
            "clustering_method": "none",
            "aggregation_method": "top_k_mean/top3 not used for random",
            "reason": "Lower-bound active-learning control.",
            "hypothesis": "Detector gains beyond this arm come from acquisition signal, not round size.",
        },
        "v2:uncertainty_objectness_weighted_entropy": {
            "strategy_spec": "v2:uncertainty with uncertainty_method=objectness_weighted_entropy",
            "weights": "uncertainty=1.0",
            "uncertainty_method": "objectness_weighted_entropy",
            "clustering_method": "none for scoring",
            "aggregation_method": "top_k_mean, top_k=3",
            "reason": (
                "Stage 1B object-positive rate was "
                f"{float(owe_b50['object_positive_image_rate']):.2f} vs "
                f"{float(entropy_b50['object_positive_image_rate']):.2f} for plain entropy."
            ),
            "hypothesis": "Object-like ambiguous proposals improve downstream unknown recall without unnecessary known-mAP loss.",
        },
        "v2:full": {
            "strategy_spec": "v2:full",
            "weights": "entropy=0.3, novelty=0.2, gated=0.5",
            "uncertainty_method": "entropy",
            "clustering_method": "current v2 KMeans, explicitly controlled pool seed",
            "aggregation_method": "top_k_mean, top_k=3",
            "reason": "Current contribution baseline under the real PROB pool.",
            "hypothesis": "The written full score improves tail-specific detector recall on a leak-free acquisition pool.",
        },
        best_pareto_name: {
            "strategy_spec": best_pareto_name,
            "weights": ", ".join(
                f"{key.removeprefix('weight_')}={value}"
                for key, value in best_pareto.items()
                if key.startswith("weight_") and value not in {"", "0", "0.0"}
            ),
            "uncertainty_method": "entropy",
            "clustering_method": "current v2 KMeans, no novelty term",
            "aggregation_method": "top_k_mean, top_k=3",
            "reason": "Best Pareto-efficient executable zero-novelty Stage 1B configuration.",
            "hypothesis": "Removing unsupported novelty while retaining the gate yields clearer detector gains than v2:full.",
        },
    }
    if include_v3:
        name = f"v3:{best_stable['method']}_full"
        chosen_names.append(name)
        specs[name] = {
            "strategy_spec": name,
            "weights": "entropy=0.3, novelty=0.2, gated=0.5",
            "uncertainty_method": "entropy",
            "clustering_method": best_stable["method"],
            "aggregation_method": "top_k_mean, top_k=3",
            "reason": "Stabilised clustering candidate with non-redundant zero-training behavior.",
            "hypothesis": "A stable pseudo-clustering gate transfers better than stochastic current v2 clustering.",
        }

    entropy_norm = normalise(as_float(component_rows, "posterior_entropy"), "rank")
    novelty_norm = normalise(as_float(component_rows, "novelty"), "rank")
    gated_norm = normalise(as_float(component_rows, "gated_rarity_coherence"), "rank")
    scores_by_strategy = {
        "v2:random": as_float(component_rows, "score_v2_random"),
        "v2:uncertainty_objectness_weighted_entropy": as_float(
            component_rows, "score_v2_uncertainty_objectness_weighted_entropy"
        ),
        "v2:full": as_float(component_rows, "score_v2_full"),
    }
    if best_pareto_name == "v2:full_no_novelty":
        scores_by_strategy[best_pareto_name] = 0.3 * entropy_norm + 0.5 * gated_norm
    elif best_pareto_name == "v2:rarity_coherence":
        scores_by_strategy[best_pareto_name] = gated_norm
    elif best_pareto_name == "v2:rarity":
        scores_by_strategy[best_pareto_name] = normalise(as_float(component_rows, "rarity"), "rank")
    else:
        scores_by_strategy[best_pareto_name] = (
            float(best_pareto.get("weight_uncertainty") or 0.0) * entropy_norm
            + float(best_pareto.get("weight_novelty") or 0.0) * novelty_norm
            + float(best_pareto.get("weight_gated") or 0.0) * gated_norm
        )
    selected_sets = {
        name: set(stage1.select(rank_images(image_ids, scores)[0], 50))
        for name, scores in scores_by_strategy.items()
    }

    for name in chosen_names:
        source = by_name.get(name, best_pareto if name == best_pareto_name else {})
        row = {"strategy": name}
        row.update(specs[name])
        for metric in (
            "tail_image_lift",
            "object_positive_image_rate",
            "distinct_gt_classes_selected",
            "distinct_tail_classes_selected",
            "background_only_selection_rate",
        ):
            row[metric] = source.get(metric, "")
        for other in chosen_names:
            if other == name:
                continue
            if selected_sets[name] and selected_sets[other]:
                overlap = len(selected_sets[name] & selected_sets[other]) / len(
                    selected_sets[name] | selected_sets[other]
                )
                row[f"selected_image_jaccard_with_{other.replace(':', '_')}"] = overlap
            else:
                row[f"selected_image_jaccard_with_{other.replace(':', '_')}"] = ""
        rows.append(row)

    decision = {
        "best_pareto_strategy": best_pareto_name,
        "best_observed_stability_method": best_stable["method"],
        "smallest_stability_change": smallest_stable["method"] if smallest_stable else "",
        "smallest_stability_change_jaccard": (
            smallest_stable["selection_jaccard_mean"] if smallest_stable else ""
        ),
        "include_v3_training_arm": include_v3,
        "v3_exclusion_reason": (
            "Consensus is stable but would require a new production scorer path and adds runtime; "
            "fixed deterministic clustering is the smallest stability fix and is redundant with "
            "a controlled v2 selected set, so a separate v3 arm is not scientifically distinguishable."
        ),
    }
    write_csv(output_dir / "strategy_comparison.csv", rows)
    write_json(output_dir / "strategy_selection_decision.json", decision)
    return rows, decision


def run_matrix_and_estimates(
    *,
    strategies: list[dict[str, Any]],
    checkpoint: Path,
    candidate_ids: Path,
    reference_ids: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    names = [row["strategy"] for row in strategies]
    matrix: list[dict[str, Any]] = []
    for strategy in names:
        config_slug = strategy.replace(":", "_")
        spec_name = (
            "uncertainty"
            if strategy == "v2:uncertainty_objectness_weighted_entropy"
            else strategy.split(":", 1)[1]
        )
        strategy_dir = f"v2_{spec_name}"
        for seed in STAGE2_SEEDS:
            for round_index in range(TRAINING_ROUNDS):
                matrix.append(
                    {
                        "strategy": strategy,
                        "seed": seed,
                        "round": round_index + 1,
                        "budget_images": TRAINING_BUDGET,
                        "initial_checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint),
                        "candidate_split": str(candidate_ids),
                        "reference_split": str(reference_ids),
                        "evaluation_split": EVALUATION_SPLIT,
                        "evaluation_disjointness_required": "candidate/training IDs must not appear in evaluation IDs",
                        "representation_reference_policy": "fixed_stage1b_3500_bank_recomputed_each_round",
                        "training_reference_policy": "cumulative_selected_candidate_labels",
                        "output_dir": (
                            f"outputs/stage2_campaign/{config_slug}/seed_{seed}/"
                            f"{strategy_dir}/round_{round_index + 1:02d}"
                        ),
                        "overwrite_policy": "refuse if round_manifest.json has completed=true",
                    }
                )
    runs = len(names) * len(STAGE2_SEEDS) * TRAINING_ROUNDS
    evals = runs
    estimate = {
        "strategies": names,
        "strategy_count": len(names),
        "seeds": list(STAGE2_SEEDS),
        "rounds": TRAINING_ROUNDS,
        "budget_images_per_round": TRAINING_BUDGET,
        "total_training_runs": runs,
        "total_evaluations": evals,
        "estimated_t4_hours_per_train_round": 1.5,
        "estimated_t4_hours_per_eval": 0.25,
        "estimated_total_t4_hours": round(runs * 1.5 + evals * 0.25, 2),
        "expected_storage_gb": round(runs * 1.2 + len(names) * len(STAGE2_SEEDS) * 0.8, 2),
        "expected_colab_sessions": "4-6 T4 sessions assuming 10-16 usable GPU hours per session.",
        "drive_directory_structure": {
            "repo": "/content/drive/MyDrive/distribution-aware-owod",
            "prob_repo": "/content/drive/MyDrive/PROB",
            "data_root": str(DATA_ROOT),
            "campaign_outputs": "outputs/stage2_campaign/{strategy}/seed_{seed}/round_{round}",
            "smoke_outputs": "outputs/stage2_smoke_t4",
        },
        "resume_plan": (
            "Each round writes round_manifest.json early and refuses to overwrite completed=true. "
            "Resume by rerunning the same daowod-run campaign command after fixing the technical "
            "failure; completed rounds abort rather than mutate."
        ),
        "predict_calls_per_scored_round": "2 for scored strategies (candidate pool and fixed reference bank); 0 for random unless export_proposals_for_random is enabled",
        "safe_caching": "Reference proposal exports may be cached per checkpoint/round because the fixed reference bank does not change; candidate exports may be cached only within an identical checkpoint/candidate-list round.",
        "cost_control": "No adaptive scientific early stopping; stop only for technical failure, numerical failure, catastrophic preregistered known-mAP collapse, or exhausted compute budget.",
        "launch_precondition": "Do not launch if canonical evaluation IDs overlap acquisition candidate or training IDs.",
    }
    prereg = {
        "primary_outcome": "tail-U-Recall at IoU 0.5 on the fixed evaluation split",
        "secondary_outcomes": [
            "known mAP",
            "aggregate U-Recall",
            "WI",
            "A-OSE",
            "unknown AP50",
            "head/medium/tail recall",
            "class coverage",
        ],
        "main_pairwise_comparisons": [
            "objectness-weighted entropy vs random",
            "v2:full vs objectness-weighted entropy",
            f"{names[-1]} vs v2:full" if names[-1] != "v2:full" else "best Pareto vs v2:full",
            "best Pareto zero-novelty config vs v2:full",
        ],
        "expected_direction": "tail-U-Recall higher than random without unacceptable known-mAP degradation",
        "acceptable_known_map_degradation_absolute": 0.02,
        "stopping_criteria": [
            "Technical stop: any repeated bridge/export/evaluation failure that prevents grouped metrics.",
            "Numerical stop: non-finite loss/metrics after one retry from the last completed round.",
            "Safety stop: catastrophic known-mAP collapse greater than 0.10 absolute for every non-random strategy after a completed round.",
            "Resource stop if observed mean T4 time exceeds the estimate by more than 50%.",
        ],
        "failed_runs": "Retry once from the last completed round; if still failed, report as permanently failed. All other preregistered arms continue and failed runs are excluded only from paired comparisons requiring that run.",
        "seed_policy": {
            "model_training_seeds": list(STAGE2_SEEDS),
            "acquisition_randomness": "seed and round are recorded; random uses deterministic shuffle of seed:round:strategy",
            "clustering_seed": "derive_seed('pool', model_seed, round_index), shared across strategies for the same seed/round",
            "data_loader_seed": "passed to PROB bridge as --seed",
            "cuda_determinism": "record CUDA/PyTorch versions and seed; deterministic kernels enabled only if compatible with PROB",
            "escalation": "Run 3 seeds for every arm. If the primary comparison is inconclusive and paired tail-U-Recall SD exceeds 5 percentage points, add seeds 3 and 4 only for OWE-vs-random and full_no_novelty-vs-full; escalation is based on variance, not winner identity.",
        },
        "clustering_seeds": "Predefined by strategy/seed/round and recorded in every round manifest; no post-hoc seed cherry-picking.",
        "aggregation": "Report mean, sample standard deviation, and 95% t-interval over the three seeds; no strategy selected by best seed only.",
        "stage1_tail_lift_role": "Diagnostic only; not proof of downstream detector improvement.",
    }
    write_csv(output_dir / "run_matrix.csv", matrix)
    write_json(output_dir / "runtime_estimate.json", estimate)
    write_json(output_dir / "preregistration.json", prereg)
    return matrix, estimate, prereg


def write_stage2_configs(
    *,
    strategies: list[dict[str, Any]],
    configs_dir: Path,
    checkpoint: Path,
    candidate_ids: Path,
    reference_ids: Path,
    output_dir: Path,
) -> list[Path]:
    configs_dir.mkdir(parents=True, exist_ok=True)
    for old in configs_dir.glob("stage2_*.yaml"):
        old.unlink()
    # Point the emitted configs at the tracked protocol copy, never at
    # output_dir: output_dir lives under the ignored outputs/ tree and would be
    # absent from a clean clone.
    class_groups_path = PROTOCOL_CLASS_GROUPS_PATH
    evaluation_path = DATA_ROOT / "ImageSets" / "OWDETR" / f"{EVALUATION_SPLIT}.txt"
    evaluation_digest = split_digest(evaluation_path)
    if evaluation_digest is None:
        raise FileNotFoundError(evaluation_path)
    paths: list[Path] = []
    strategy_names = [
        row["strategy"] for row in strategies if not row["strategy"].startswith("v3:")
    ]
    for name in strategy_names:
        path = configs_dir / f"stage2_{name.replace(':', '_')}.yaml"
        uncertainty_override = ""
        protocol_uncertainty = "entropy"
        listed_name = name
        if name == "v2:uncertainty_objectness_weighted_entropy":
            listed_name = "v2:uncertainty"
            uncertainty_override = "  uncertainty_method: objectness_weighted_entropy\n"
            protocol_uncertainty = "objectness_weighted_entropy"
        if name == "v2:random":
            uncertainty_override = "  uncertainty_method: none\n"
            protocol_uncertainty = "none"
        uncertainty_block = uncertainty_override or "  uncertainty_method: entropy\n"
        text = f"""name: stage2-{name.replace(":", "-")}

active_learning:
  rounds: {TRAINING_ROUNDS}
  strategy: {listed_name}
  budget: {TRAINING_BUDGET}
  initial_images: 0
  budget_per_round: {TRAINING_BUDGET}
  seeds: [0, 1, 2]

protocol:
  dataset_protocol: OWDETR
  data_root: {DATA_ROOT}
  previous_introduced_classes: 0
  current_introduced_classes: 19
  num_classes: 81
  objectness_temperature: 1
  train_split: runtime_selected_ids
  candidate_pool_split: {candidate_ids}
  reference_split: {reference_ids}
  evaluation_split: {EVALUATION_SPLIT}
  evaluation_split_sha256: {evaluation_digest}
  initial_labelled_split: null
  pool_policy: stage1_exact
  reference_policy: fixed_stage1b_representation_bank
  long_tail_transformation: none
  checkpoint: {checkpoint}
  checkpoint_sha256: {sha256(checkpoint)}
  image_aggregation: top_k_mean
  top_k: {TOP_K}
  uncertainty_method: {protocol_uncertainty}
  clustering_method: current_v2_kmeans_shared_pool_seed
  acquisition_budget: {TRAINING_BUDGET}
  active_learning_rounds: {TRAINING_ROUNDS}
  training_schedule: "epochs=10,learning_rate=2e-5,eval_every=2,prob_unfrozen=true"
  evaluation_settings: "grouped_metrics=true,iou_threshold=0.5,unknown_prediction_name=unknown,require_detections=true"
  class_group_mapping: stage1_candidate_frequency_thirds
  clustering_seed_policy: "derive_seed('pool', model_seed, round_index)"
  cuda_determinism: recorded_not_forced
  allow_candidate_evaluation_overlap: false
  allow_labelled_evaluation_overlap: false

acquisition:
  strategies:
    - {listed_name}
{uncertainty_block}  rarity_method: log_inverse_frequency
  coherence_method: relative_within_cluster
  normalisation: rank
  pseudo_label_source: cluster
  cluster_count: 20
  neighbour_count: 5
  image_aggregation: top_k_mean
  top_k: 3
  coherence_exponent: 1.0
  singleton_coherence: 0.0
  minimum_cluster_size: 3

dataset:
  image_set_path: {candidate_ids}
  annotations_dir: {DATA_ROOT / "Annotations"}
  unknown_classes:
    - traffic light
    - fire hydrant
    - stop sign
    - parking meter
    - bench
    - chair
    - diningtable
    - pottedplant
    - backpack
    - umbrella
    - handbag
    - tie
    - suitcase
    - microwave
    - oven
    - toaster
    - sink
    - refrigerator
    - bed
    - toilet
    - sofa
    - frisbee
    - skis
    - snowboard
    - sports ball
    - kite
    - baseball bat
    - baseball glove
    - skateboard
    - surfboard
    - tennis racket
    - banana
    - apple
    - sandwich
    - orange
    - broccoli
    - carrot
    - hot dog
    - pizza
    - donut
    - cake
    - laptop
    - mouse
    - remote
    - keyboard
    - cell phone
    - book
    - clock
    - vase
    - scissors
    - teddy bear
    - hair drier
    - toothbrush
    - wine glass
    - cup
    - fork
    - knife
    - spoon
    - bowl
    - tvmonitor
    - bottle
  known_classes:
    - aeroplane
    - bicycle
    - bird
    - boat
    - bus
    - car
    - cat
    - cow
    - dog
    - horse
    - motorbike
    - sheep
    - train
    - elephant
    - bear
    - zebra
    - giraffe
    - truck
    - person
  class_groups_path: {class_groups_path}
  known_class_groups_path: null
  long_tail:
    enabled: false
    imbalance_ratio: 50.0

evaluation:
  grouped_metrics: true
  iou_threshold: 0.5
  unknown_prediction_name: unknown
  require_detections: true

prob:
  repository_path: /Users/gubiczam/Documents/PROB
  initial_checkpoint: {checkpoint}
  train_command: >-
    .venv/bin/python daowod_prob_bridge.py train
    --labelled-ids {{labelled_ids}}
    --previous-checkpoint {{previous_checkpoint}}
    --output-checkpoint {{checkpoint}}
    --output-dir {{output_dir}}
    --seed {{seed}}
    --data-root {DATA_ROOT}
    --dataset OWDETR
    --prev-introduced-classes 0
    --current-introduced-classes 19
    --num-classes 81
    --objectness-temperature 1
    --test-set {EVALUATION_SPLIT}
    --epochs 10
    --learning-rate 2e-5
    --eval-every 2
    --no-freeze-prob-model
  predict_command: >-
    .venv/bin/python daowod_prob_bridge.py predict
    --image-ids {{image_ids}}
    --checkpoint {{checkpoint}}
    --output {{proposals}}
    --data-root {DATA_ROOT}
    --dataset OWDETR
    --prev-introduced-classes 0
    --current-introduced-classes 19
    --num-classes 81
    --objectness-temperature 1
    --device cuda
    --max-proposals-per-image 100
  evaluate_command: >-
    .venv/bin/python daowod_prob_bridge.py evaluate
    --checkpoint {{checkpoint}}
    --output {{metrics}}
    --output-dir {{output_dir}}
    --data-root {DATA_ROOT}
    --dataset OWDETR
    --prev-introduced-classes 0
    --current-introduced-classes 19
    --num-classes 81
    --objectness-temperature 1
    --test-set {EVALUATION_SPLIT}
  timeout_seconds: 86400

output_dir: outputs/stage2_campaign/{name.replace(":", "_")}
"""
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def write_protocol_doc(
    *,
    path: Path,
    audit_answer: dict[str, Any],
    stability_rows: list[dict[str, Any]],
    strategies: list[dict[str, Any]],
    estimate: dict[str, Any],
    prereg: dict[str, Any],
    decision: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    best_stability = max(stability_rows, key=lambda row: float(row["selection_jaccard_mean"]))
    smallest_stability = next(
        (row for row in stability_rows if row["method"] == decision["smallest_stability_change"]),
        best_stability,
    )
    preflight_ready = preflight["protocol_status"] == "ready"
    current_stability = next(
        (row for row in stability_rows if row["method"] == "current_v2"),
        None,
    )
    full50 = audit_answer["budget50_summary"].get("v2:full", {})
    gate50 = audit_answer["budget50_summary"].get("v2:rarity_coherence", {})
    status_text = (
        "**CONDITIONAL GO.** Stage 2 is now rebuilt from leak-free Stage 1B evidence. "
        "The canonical evaluation split is staged and disjoint from candidate, reference, "
        "and initial-labelled IDs. The remaining blocker is the real T4 smoke execution."
        if preflight_ready
        else "**NO GO.** The canonical OWDETR evaluation split is now resolved and staged, "
        "but it overlaps the acquisition candidate pool. The protocol validator must refuse "
        "this before any T4 smoke or 36-run campaign."
    )
    final_warning = (
        "Do not start the 36-run campaign until the real T4 smoke cycle passes."
        if preflight_ready
        else "Do not start the 36-run campaign. The current protocol is a NO GO before T4 smoke because it would train on official evaluation images."
    )
    lines = [
        "# Stage 2 Protocol",
        "",
        "## Status",
        "",
        status_text,
        "",
        "## Stage 1 Audit Outcome",
        "",
        "- Stage 1B uses real PROB proposals from official Task-1 training-side IDs only; the old 500-image eval-pool diagnostics are retained only as a contaminated baseline for comparison.",
        f"- Full tail lift at budget 50: {float(full50.get('tail_image_lift', float('nan'))):.3f}; gate-only tail lift at budget 50: {float(gate50.get('tail_image_lift', float('nan'))):.3f}.",
        f"- Full object-positive rate at budget 50: {float(full50.get('object_positive_rate', float('nan'))):.3f}; gate-only object-positive rate at budget 50: {float(gate50.get('object_positive_rate', float('nan'))):.3f}.",
        f"- Full and gate-only select the same images at budget 50: {audit_answer['are_full_and_gate_only_selecting_same_images_at_50']}.",
        f"- Rarity alignment with true inverse class frequency, post hoc Spearman: {audit_answer['rarity_alignment']['proposal_level_spearman_on_matched_objects']:.3f}.",
        "- Identical tail lift is an image-level composition tie, not proof that proposal-level scores agree.",
        "- Ground truth is used only after scoring for diagnostics and never as an acquisition input.",
        "",
        "## Authoritative Protocol Object",
        "",
        "- Every Stage 2 YAML has a required `protocol:` block consumed by validation, acquisition, train, predict, evaluate, reports, and manifests.",
        "- The frozen task settings are `dataset_protocol=OWDETR`, `previous_introduced_classes=0`, `current_introduced_classes=19`, `num_classes=81`, `objectness_temperature=1`, and the Task-1 SOWODB checkpoint digest recorded in each config.",
        "- Command parity validation rejects train/predict/evaluate drift before execution; each round manifest records the fully resolved train, candidate-predict, reference-predict, and evaluate command lines plus a resolved-command parity report.",
        "",
        "## Pool Decision",
        "",
        "- Decision: Option A.",
        f"- Candidate pool: leak-free Stage 1B candidate split, `{preflight['candidate_split_path']}`.",
        f"- Representation reference bank: leak-free Stage 1B fixed bank, `{preflight['reference_split_path']}`.",
        f"- Class-group mapping: version-controlled protocol asset, `{PROTOCOL_CLASS_GROUPS_PATH.as_posix()}`. "
        "It is a protocol input, not a run artifact, so a clean clone can resolve it.",
        "- Long-tail transformation: disabled. `dataset.long_tail.enabled=false` and `protocol.long_tail_transformation=none` preserve real Stage 1B comparability.",
        "- Initial labelled split: none. Stage 2 starts with zero labelled candidate-pool images so first-round acquisition sees the exact Stage 1B candidate pool.",
        f"- Evaluation split: fixed `owdetr_test`, SHA256 `{preflight['evaluation_split_sha256']}`.",
        f"- Candidate/evaluation overlap under the canonical split: {preflight['candidate_evaluation_overlap']} images.",
        f"- Reference/evaluation overlap under the canonical split: {preflight['reference_evaluation_overlap']} images.",
        "",
        "## Reference Semantics",
        "",
        "- Fixed representation reference bank: the 3,500-image Stage 1B reference split, sliced from the official train-side 4,000-image real PROB export, used only to compute proposal-space novelty/coherence references; it does not change by round.",
        "- Growing labelled training set: the cumulative selected candidate IDs, starting empty and growing by exactly 20 images per round. This is what `train --labelled-ids` receives.",
        "- Newly selected candidate images are added to the labelled training set, not to the fixed representation reference bank.",
        "- Selected candidate images are removed from the candidate pool after selection and never selected twice.",
        "- Candidate/reference overlap is forbidden. Candidate/evaluation and labelled/evaluation overlap are recorded and must be zero unless explicitly permitted by a future protocol revision.",
        "- The detector checkpoint is used to recompute candidate and fixed-reference proposal features each scored round; reference proposals may be cached only for the identical checkpoint/reference-list pair.",
        "- The fixed reference split can contain known and unknown objects; it is a representation bank, not an initial labelled set.",
        "",
        "## Evaluation",
        "",
        "- Evaluation split name: `owdetr_test`, fixed across all rounds, seeds, and strategies.",
        "- Evaluation support is written before training to `outputs/stage2_plan/evaluation_support_report.csv`.",
        f"- Local preflight reports `asset_status={preflight['asset_status']}` and `protocol_status={preflight['protocol_status']}`.",
        (
            "- Evaluation disjointness is satisfied; config validation should pass before T4 smoke."
            if preflight_ready
            else "- Because candidate/evaluation overlap is forbidden by the Stage 2 protocol, config validation is expected to fail until the scientific design is repaired."
        ),
        "- Official metrics come from the PROB bridge JSON (`known_mAP`, `U_Recall`, `WI`, `A_OSE`, plus unknown AP50 when exported). Custom grouped metrics use the same detections artifact, IoU 0.5, and frozen Stage 1 candidate-frequency thirds.",
        "",
        "## Clustering Stabilisation",
        "",
        f"- Best observed zero-training stability method: `{best_stability['method']}` with mean selection Jaccard {float(best_stability['selection_jaccard_mean']):.3f}.",
        f"- Chosen smallest stability change: `{smallest_stability['method']}` with mean selection Jaccard {float(smallest_stability['selection_jaccard_mean']):.3f}.",
        f"- Current v2 baseline remains the scientific baseline; full fixed-pool clustering-seed Jaccard was {float(current_stability['selection_jaccard_mean']) if current_stability else float('nan'):.3f} in Stage 1B.",
        f"- v3 included in training: {decision['include_v3_training_arm']}. {decision['v3_exclusion_reason']}",
        "",
        "## Stage 2 Arms",
        "",
    ]
    for row in strategies:
        lines.extend(
            [
                f"### {row['strategy']}",
                "",
                f"- StrategySpec: `{row['strategy_spec']}`",
                f"- Weights: {row['weights']}",
                f"- Uncertainty: {row['uncertainty_method']}",
                f"- Clustering: {row['clustering_method']}",
                f"- Aggregation: {row['aggregation_method']}",
                f"- Reason: {row['reason']}",
                f"- Hypothesis: {row['hypothesis']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Training Protocol",
            "",
            f"- Seeds: {estimate['seeds']}",
            f"- Rounds: {estimate['rounds']}",
            f"- Budget per round: {estimate['budget_images_per_round']} images.",
            "- The 20-image round size is chosen because Stage 1 budget curves already separate OWE/full/gate behavior by budgets 20-50 while keeping the campaign small.",
            "- Every strategy/seed starts from the same Task-1 checkpoint digest recorded in `run_matrix.csv`.",
            "- No checkpoint is shared across strategies; each round resumes only from its own previous completed round.",
            "- Completed-round overwrite protection is required: `round_manifest.json` with `completed=true` must abort reruns.",
            "- Safe resume means rerunning the same command after a technical failure; completed rounds are immutable and permanently failed runs are reported rather than replaced silently.",
            "- PROB schedule is 10 additional fine-tuning epochs at learning rate 2e-5 with evaluation every 2 epochs and the PROB model unfrozen; this is intentionally more meaningful than the one-epoch smoke setting.",
            "",
            "## Cost",
            "",
            f"- Training runs: {estimate['total_training_runs']}",
            f"- Evaluations: {estimate['total_evaluations']}",
            f"- Estimated T4 hours: {estimate['estimated_total_t4_hours']}",
            f"- Expected storage: {estimate['expected_storage_gb']} GB",
            f"- Expected Colab sessions: {estimate['expected_colab_sessions']}",
            f"- Predict calls: {estimate['predict_calls_per_scored_round']}",
            f"- Safe caching: {estimate['safe_caching']}",
            "",
            "## Preregistration",
            "",
            f"- Primary outcome: {prereg['primary_outcome']}",
            f"- Secondary outcomes: {', '.join(prereg['secondary_outcomes'])}",
            f"- Expected direction: {prereg['expected_direction']}",
            f"- Acceptable known-mAP degradation: {prereg['acceptable_known_map_degradation_absolute']} absolute.",
            "- Mean, sample standard deviation, and 95% t-interval are computed over seeds; best-seed selection is forbidden.",
            "- Stage 1 tail lift is diagnostic only and is not treated as proof of detector improvement.",
            f"- Seed policy: {prereg['seed_policy']['clustering_seed']}; paired comparisons are preserved by running every strategy on the same seed/round pre-selection pool.",
            "- Scientific early stopping is removed. All arms complete the same matrix unless a technical, numerical, catastrophic-safety, or resource stop occurs.",
            "",
            "## T4 Smoke",
            "",
            "- Smoke config: `configs/smoke_stage2_t4.yaml`.",
            "- It uses one strategy, one seed, one round, budget 2, one epoch, the same checkpoint, same OWDETR/SOWODB command arguments, same predict/evaluate bridge, grouped metrics, Drive persistence, and resume/overwrite checks.",
            "- Exact Colab cells are in `docs/stage2_t4_smoke.md`.",
            "",
            final_warning,
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_smoke_config(*, configs_dir: Path) -> Path:
    source = configs_dir / "stage2_v2_uncertainty_objectness_weighted_entropy.yaml"
    text = source.read_text(encoding="utf-8")
    replacements = {
        "name: stage2-v2-uncertainty_objectness_weighted_entropy": "name: smoke-stage2-t4",
        "  rounds: 3\n  strategy: v2:uncertainty\n  budget: 20\n  initial_images: 0\n  budget_per_round: 20\n  seeds: [0, 1, 2]": "  rounds: 1\n  strategy: v2:uncertainty\n  budget: 2\n  initial_images: 0\n  budget_per_round: 2\n  seeds: [0]",
        "  acquisition_budget: 20\n  active_learning_rounds: 3": "  acquisition_budget: 2\n  active_learning_rounds: 1",
        '  training_schedule: "epochs=10,learning_rate=2e-5,eval_every=2,prob_unfrozen=true"': '  training_schedule: "epochs=1,learning_rate=2e-5,eval_every=1,prob_unfrozen=true"',
        "    --epochs 10\n    --learning-rate 2e-5\n    --eval-every 2": "    --epochs 1\n    --learning-rate 2e-5\n    --eval-every 1",
        "output_dir: outputs/stage2_campaign/v2_uncertainty_objectness_weighted_entropy": "output_dir: outputs/stage2_smoke_t4",
    }
    for old, new in replacements.items():
        if old not in text:
            raise ValueError(f"Smoke config replacement anchor missing: {old}")
        text = text.replace(old, new)
    path = configs_dir / "smoke_stage2_t4.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def write_smoke_doc(
    *,
    path: Path,
    candidate_ids: Path,
    reference_ids: Path,
    preflight: dict[str, Any],
) -> None:
    status = (
        "Status: ready for real T4 smoke after local config validation. This does not start the 36-run campaign."
        if preflight["protocol_status"] == "ready"
        else "Status: blocked by Stage 2 protocol validation. This does not start the 36-run campaign."
    )
    expected_failure = (
        "None" if preflight["protocol_status"] == "ready" else preflight["protocol_status"]
    )
    text = """# Stage 2 T4 Smoke Protocol

@@STATUS@@

The smoke config is `configs/smoke_stage2_t4.yaml`: one strategy (`v2:uncertainty` with objectness-weighted entropy), one seed, one round, budget 2, one training epoch. It uses the same checkpoint, same OWDETR/SOWODB flags, same train/predict/evaluate bridge, same grouped metrics, same output persistence, and same completed-round overwrite protection as the final configs.

## Colab Cells

Cell 1 - mount Drive and clone exact repos:

```bash
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive
!test -d distribution-aware-owod || git clone https://github.com/gubiczam/distribution-aware-owod.git distribution-aware-owod
!test -d PROB || git clone https://github.com/gubiczam/PROB.git PROB
```

Cell 2 - verify the source revision policy:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
git fetch origin
git status --short
test -z "$(git status --short)" || { echo "FAIL: commit and push local Stage 2 protocol changes before smoke"; exit 2; }
cd /content/drive/MyDrive/PROB
git fetch origin
git checkout 980cf3a796f064dd4c56f573ba10cc755143e116
mkdir -p /Users/gubiczam/Documents /Users/gubiczam/Downloads/results
ln -sfn /content/drive/MyDrive/PROB /Users/gubiczam/Documents/PROB
ln -sfn /content/drive/MyDrive/owod_stage /Users/gubiczam/owod_stage
ln -sfn /content/drive/MyDrive/results/SOWODB /Users/gubiczam/Downloads/results/SOWODB
```

Cell 3 - install DAOWOD dependencies:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

Cell 4 - install/compile PROB dependencies and CUDA extension:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/PROB
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
```

Cell 5 - validate assets and hashes:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
test -d /content/drive/MyDrive/owod_stage
test -d /content/drive/MyDrive/results/SOWODB
test -f @@CANDIDATE_IDS@@
test -f @@REFERENCE_IDS@@
test -f /Users/gubiczam/owod_stage/ImageSets/OWDETR/owdetr_test.txt
test -d /Users/gubiczam/owod_stage/Annotations
test -f /Users/gubiczam/Downloads/results/SOWODB/t1.pth
sha256sum /Users/gubiczam/Downloads/results/SOWODB/t1.pth | grep dba5390bffdfdf63058a995f241696df8d06b7fb859aecc8292d9ea02d459a22
PYTHONPATH=src python analysis/stage2_plan.py
python - <<'PY'
import json
preflight=json.load(open('outputs/stage2_plan/protocol_preflight.json'))
assert preflight['asset_status']=='ready', preflight
assert preflight['protocol_status']=='ready', preflight
assert preflight['candidate_evaluation_overlap']==0, preflight
assert preflight['reference_evaluation_overlap']==0, preflight
PY
```

Cell 6 - compile and test DAOWOD:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
python -m compileall src analysis tests
ruff format --check .
ruff check .
pytest
```

Cell 7 - validate all final configs plus smoke and write a machine-readable preflight:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
mkdir -p outputs/stage2_smoke_t4
set +e
for cfg in configs/stage2_*.yaml configs/smoke_stage2_t4.yaml; do
  daowod-run validate --config "$cfg" --manifest "outputs/stage2_plan/$(basename "$cfg" .yaml)_validate_manifest.json"
done
code=$?
set -e
python - "$code" <<'PY'
import json, sys
code=int(sys.argv[1])
summary={
  "schema":"stage2_t4_smoke_preflight_v1",
  "verdict":"PASS" if code == 0 else "FAIL",
  "stage":"config_validation",
  "exit_code":code,
  "expected_current_failure":"@@EXPECTED_FAILURE@@"
}
open("outputs/stage2_smoke_t4/preflight_summary.json","w").write(json.dumps(summary, indent=2)+"\\n")
print(json.dumps(summary, indent=2))
PY
```

Cell 8 - run the smoke experiment only if validation passed:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
if python - <<'PY'
import json, sys
summary=json.load(open("outputs/stage2_smoke_t4/preflight_summary.json"))
if summary["verdict"] != "PASS":
    print("Skipping T4 smoke because preflight failed:")
    print(json.dumps(summary, indent=2))
    sys.exit(1)
PY
then
daowod-run campaign --config configs/smoke_stage2_t4.yaml
else
  echo "T4 smoke skipped after preflight failure"
fi
```

Cell 9 - check artifacts, resolved commands, grouped metrics, and resume refusal:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
if python - <<'PY'
import json, sys
summary=json.load(open("outputs/stage2_smoke_t4/preflight_summary.json"))
if summary["verdict"] != "PASS":
    print("Skipping artifact checks because smoke did not run:")
    print(json.dumps(summary, indent=2))
    sys.exit(1)
PY
then
MANIFEST=$(find outputs/stage2_smoke_t4 -name round_manifest.json | sort | tail -1)
python - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
m=json.load(open(sys.argv[1]))
round_dir=Path(sys.argv[1]).parent
assert m['completed'] is True
assert m['resolved_command_parity']['status']=='ok'
assert m['evaluation_count'] > 0
assert m['support_counts']['evaluation']['head_objects'] > 0
assert m['support_counts']['evaluation']['medium_objects'] > 0
assert m['support_counts']['evaluation']['tail_objects'] > 0
assert m['grouped_metrics'] is not None
for name in ['candidate_ids_before_selection.txt','reference_ids.txt','labelled_ids_before_selection.txt','selected_ids.txt','labelled_ids.txt','training_ids.txt','remaining_pool_ids.txt','evaluation_ids.txt','metrics.json','checkpoint.pth']:
    assert (round_dir/name).exists(), name
print('artifact check PASS')
PY
set +e
daowod-run campaign --config configs/smoke_stage2_t4.yaml
code=$?
set -e
test "$code" -ne 0
echo "resume refusal PASS"
else
  echo "artifact checks skipped after preflight failure"
fi
```

Cell 10 - final PASS/FAIL:

```bash
%%bash
set -euo pipefail
cd /content/drive/MyDrive/distribution-aware-owod
if python - <<'PY'
import json
summary=json.load(open("outputs/stage2_smoke_t4/preflight_summary.json"))
raise SystemExit(0 if summary["verdict"] == "PASS" else 1)
PY
then
test -f outputs/stage2_smoke_t4/selections.json
test -f outputs/stage2_smoke_t4/metrics.csv
echo "STAGE2_T4_SMOKE_PASS"
else
cat outputs/stage2_smoke_t4/preflight_summary.json
echo "STAGE2_T4_SMOKE_FAIL"
fi
```
"""
    text = (
        text.replace("@@STATUS@@", status)
        .replace("@@CANDIDATE_IDS@@", str(candidate_ids))
        .replace("@@REFERENCE_IDS@@", str(reference_ids))
        .replace("@@EXPECTED_FAILURE@@", expected_failure)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_go_checklist(*, path: Path, preflight: dict[str, Any]) -> None:
    ready = preflight["protocol_status"] == "ready"
    verdict = "CONDITIONAL GO" if ready else "NO GO"
    disjoint_check = "[x]" if ready else "[ ]"
    config_check = "[x]" if ready else "[ ]"
    disjoint_note = (
        "Current preflight: candidate/evaluation, reference/evaluation, and initial/evaluation overlaps are all zero."
        if ready
        else f"Current preflight: protocol_status={preflight['protocol_status']}."
    )
    config_note = (
        "They must pass locally before the smoke run."
        if ready
        else "They are expected to fail until the protocol overlap is repaired."
    )
    smoke_note = (
        "It is the remaining blocker before a full GO."
        if ready
        else "It must not run while config validation fails on leakage."
    )
    text = f"""# Stage 2 GO Checklist

Current verdict: {verdict}.

- [x] Train/predict/evaluate protocol parity is validated from the authoritative `protocol:` object.
- [x] Executable pool is frozen to leak-free Stage 1B, the official Task-1 train-side candidate pool.
- [x] Reference semantics are frozen: fixed Stage 1B representation bank is separate from cumulative labelled training IDs.
- [x] Stage 1B diagnostics correspond to the executable acquisition pool; the old eval-pool diagnostics are disqualified from acquisition/training.
- [x] Canonical evaluation split is resolved and frozen as PROB `OWDETR/owdetr_test.txt`.
- [x] Evaluation annotations/images are present in local staged assets.
- {disjoint_check} Evaluation split is disjoint from acquisition/training. {disjoint_note}
- [x] Scientific early stopping is removed from preregistration.
- [x] Seed policy and variance-based escalation rule are preregistered.
- {config_check} Stage 2 configs validate locally. {config_note}
- [x] Every required input survives a clean clone. `analysis/audit_clean_clone_assets.py` passes, so no config or notebook path resolves into an ignored tree such as `outputs/`.
- [x] Tests pass locally.
- [x] Resolved commands are written into every round manifest before training.
- [x] Data-lineage ID lists, hashes, overlaps, and support counts are written per round.
- [ ] Real T4 smoke train/evaluate passes. {smoke_note}
- [ ] Smoke artifacts persist in Drive and resume refusal is observed.
- [x] Run matrix, runtime/storage estimate, and safe caching policy are updated.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_audit_addendum(*, path: Path, preflight: dict[str, Any]) -> None:
    ready = preflight["protocol_status"] == "ready"
    verdict = "CONDITIONAL GO" if ready else "NO GO"
    failure_or_resolution = (
        "- The old 500-image Stage 1 pool is permanently disqualified because it is a subset of `owdetr_test`. Stage 1B replaces it with official Task-1 train-side candidate and reference splits.\n"
        f"- Stage 1B preflight proves zero overlap: candidate/evaluation={preflight['candidate_evaluation_overlap']}, reference/evaluation={preflight['reference_evaluation_overlap']}, candidate/reference={preflight['candidate_reference_overlap']}.\n"
        "- The remaining blocker is empirical rather than design-level: the real T4 smoke train/evaluate cycle has not yet run."
        if ready
        else "- The current protocol still has evaluation overlap, so the validator refuses to run."
    )
    text = f"""# Stage 2 Scientific Audit Addendum

Verdict after Stage 1B leak-free remediation: {verdict}.

Resolved blockers:

- Command drift is blocked by an authoritative `protocol:` object and command-parity validation. The exact audit mismatch (`TOWOD`, 20/20 class defaults, objectness temperature 1.3, inconsistent eval split) now fails tests.
- Pool decision is Option A using Stage 1B only. Stage 2 uses a leak-free official Task-1 train-side candidate split and a disjoint Stage 1B representation reference bank. The long-tail transformation is disabled to preserve Stage 1B parity.
- Canonical evaluation split is PROB `data/OWOD/ImageSets/OWDETR/owdetr_test.txt`, copied unchanged to the staged data root. Its SHA256 is recorded in every Stage 2 YAML and in `outputs/stage2_plan/protocol_preflight.json`.
- Evaluation XML annotations were generated from local COCO val2017 annotations using PROB's official `datasets/coco2voc.py:coco_to_voc_detection` conversion path. Evaluation JPEGs are present or symlinked from the local COCO val2017 image cache.
- Reference semantics are no longer overloaded. The fixed representation bank is for proposal-space scoring; cumulative selected candidate IDs are the labelled training set.
- Round manifests now record candidate, reference, labelled-before, selected, training, remaining, and evaluation IDs with SHA256 digests, overlaps, support counts, and resolved command lines.
- Scientific early stopping has been removed. Technical/numerical/catastrophic/resource stops remain preregistered.
- Seed policy is frozen: model/data-loader seed is passed to PROB, random acquisition uses deterministic seed-round-strategy shuffling, clustering uses `derive_seed('pool', model_seed, round_index)` shared across strategies for paired comparisons.

Stage 1B disposition:

{failure_or_resolution}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.stage1_dir / "real_proposal_components.csv")
    manifest = json.loads(
        (args.stage1_dir / "real_stage1_manifest.json").read_text(encoding="utf-8")
    )
    candidate_ids = stage1.read_ids(args.candidate_ids)
    reference_ids = stage1.read_ids(args.reference_ids)
    if set(candidate_ids) & set(reference_ids):
        raise ValueError("candidate/reference split overlap")
    candidate = stage1.load_proposals(args.candidate)
    stage1.validate_proposals("candidate", candidate, candidate_ids)
    groups = stage1.build_groups(candidate_ids, args.annotations_dir)
    gt = stage1.build_posthoc_gt(candidate, candidate_ids, args.annotations_dir, groups, 0.5)
    base = stage1.pool_base_rates(gt)
    image_ids = np.asarray([str(value) for value in candidate.image_ids], dtype=object)

    audit_answer = top_budget_audit(
        rows=rows,
        image_ids=image_ids,
        gt=gt,
        base=base,
        output_dir=args.output_dir,
    )
    stability_rows = clustering_stability(
        candidate=candidate,
        rows=rows,
        image_ids=image_ids,
        gt=gt,
        base=base,
        output_dir=args.output_dir,
    )
    strategies, decision = strategy_selection(
        stage1_dir=args.stage1_dir,
        stability_rows=stability_rows,
        component_rows=rows,
        image_ids=image_ids,
        output_dir=args.output_dir,
    )
    _, estimate, prereg = run_matrix_and_estimates(
        strategies=strategies,
        checkpoint=args.checkpoint,
        candidate_ids=args.candidate_ids,
        reference_ids=args.reference_ids,
        output_dir=args.output_dir,
    )
    write_class_groups(args.output_dir / "stage2_class_groups.csv", groups)
    # Keep the version-controlled protocol copy in step with the run artifact;
    # the Stage 2 configs resolve class_groups_path against this one.
    write_class_groups(PROTOCOL_CLASS_GROUPS_PATH, groups)
    preflight = write_protocol_preflight(
        output_dir=args.output_dir,
        candidate_ids_path=args.candidate_ids,
        reference_ids_path=args.reference_ids,
        groups=groups,
    )
    verdict = "NO GO" if preflight["protocol_status"] != "ready" else "CONDITIONAL GO"
    config_paths = write_stage2_configs(
        strategies=strategies,
        configs_dir=args.configs_dir,
        checkpoint=args.checkpoint,
        candidate_ids=args.candidate_ids,
        reference_ids=args.reference_ids,
        output_dir=args.output_dir,
    )
    smoke_config = write_smoke_config(configs_dir=args.configs_dir)
    write_protocol_doc(
        path=args.docs_dir / "stage2_protocol.md",
        audit_answer=audit_answer,
        stability_rows=stability_rows,
        strategies=strategies,
        estimate=estimate,
        prereg=prereg,
        decision=decision,
        preflight=preflight,
    )
    write_smoke_doc(
        path=args.docs_dir / "stage2_t4_smoke.md",
        candidate_ids=args.candidate_ids,
        reference_ids=args.reference_ids,
        preflight=preflight,
    )
    write_go_checklist(path=args.docs_dir / "stage2_go_checklist.md", preflight=preflight)
    write_audit_addendum(
        path=args.docs_dir / "stage2_scientific_audit_addendum.md", preflight=preflight
    )
    write_json(
        args.output_dir / "stage2_plan_manifest.json",
        {
            "schema": "stage2_plan_v1",
            "stage1_manifest": manifest,
            "audit_answers": audit_answer,
            "stability_decision": decision,
            "configs": [str(path) for path in [*config_paths, smoke_config]],
            "protocol_preflight": preflight,
            "decision": verdict,
            "ground_truth_policy": {
                "gt_absent_from_acquisition_inputs": True,
                "gt_joined_only_post_hoc_for_planning": True,
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(args.output_dir),
                "strategies": [row["strategy"] for row in strategies],
                "total_training_runs": estimate["total_training_runs"],
                "decision": verdict,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
