#!/usr/bin/env python3
"""Stage 1 real-PROB proposal diagnostics.

This script is deliberately offline: it consumes fixed PROB proposal exports and
VOC-style XML annotations, joins ground truth only for post-hoc metrics, and
writes the artifacts requested for Stage 1. It does not train or mutate PROB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from daowod.normalisation import average_ranks, normalise

T1_CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bus",
    "car",
    "cat",
    "cow",
    "dog",
    "horse",
    "motorbike",
    "sheep",
    "train",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "truck",
    "person",
]
T2_CLASS_NAMES = [
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "chair",
    "diningtable",
    "pottedplant",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "bed",
    "toilet",
    "sofa",
]
T3_CLASS_NAMES = [
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
]
T4_CLASS_NAMES = [
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "tvmonitor",
    "bottle",
]
UNKNOWN_CLASS = ["unknown"]
OWDETR_CLASS_NAMES = tuple(
    T1_CLASS_NAMES + T2_CLASS_NAMES + T3_CLASS_NAMES + T4_CLASS_NAMES + UNKNOWN_CLASS
)
KNOWN_CLASSES = set(T1_CLASS_NAMES)
UNKNOWN_CLASSES = tuple(T2_CLASS_NAMES + T3_CLASS_NAMES + T4_CLASS_NAMES)
GROUPS = ("head", "medium", "tail")
CLASS_ALIASES = {
    "airplane": "aeroplane",
    "motorcycle": "motorbike",
    "couch": "sofa",
    "dining table": "diningtable",
    "potted plant": "pottedplant",
    "tv": "tvmonitor",
}


@dataclass(frozen=True)
class ProposalSet:
    path: Path
    image_ids: np.ndarray
    confidence: np.ndarray
    embeddings: np.ndarray
    posterior: np.ndarray
    predicted_labels: np.ndarray
    boxes: np.ndarray
    objectness: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GTObject:
    image_id: str
    class_name: str
    group: str
    box_xyxy: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate-ids", required=True, type=Path)
    parser.add_argument("--reference-ids", required=True, type=Path)
    parser.add_argument("--annotations-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prob-repo", required=True, type=Path)
    parser.add_argument("--daowod-repo", default=Path.cwd(), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cluster-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--cluster-counts", default="10,20,40")
    parser.add_argument("--neighbour-counts", default="3,5,10")
    parser.add_argument("--budget", default="5,10,20,30,50")
    parser.add_argument("--baseline-clusters", type=int, default=20)
    parser.add_argument("--baseline-neighbours", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--novelty-candidate-chunk", type=int, default=512)
    return parser.parse_args()


def ints(csv_text: str) -> list[int]:
    return [int(part) for part in csv_text.split(",") if part.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> dict[str, Any]:
    def run(*parts: str) -> str:
        return subprocess.check_output(parts, cwd=path, text=True).strip()

    try:
        return {
            "path": str(path),
            "commit": run("git", "rev-parse", "HEAD"),
            "status_short": run("git", "status", "--short"),
        }
    except Exception as error:  # pragma: no cover - defensive manifest metadata
        return {"path": str(path), "error": str(error)}


def read_ids(path: Path) -> list[str]:
    values = [
        line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"{path}: duplicate image IDs")
    return values


def canonical_class_name(class_name: str) -> str:
    return CLASS_ALIASES.get(class_name, class_name)


def load_proposals(path: Path) -> ProposalSet:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        required = {
            "image_ids",
            "confidence",
            "embeddings",
            "posterior",
            "predicted_labels",
            "boxes",
            "objectness",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        image_ids = np.asarray(data["image_ids"], dtype=object)
        confidence = np.asarray(data["confidence"], dtype=np.float64)
        embeddings = np.asarray(data["embeddings"], dtype=np.float64)
        posterior = np.asarray(data["posterior"], dtype=np.float64)
        predicted_labels = np.asarray(data["predicted_labels"], dtype=np.int64)
        boxes = np.asarray(data["boxes"], dtype=np.float64)
        objectness = np.asarray(data["objectness"], dtype=np.float64)
    count = image_ids.shape[0]
    parallel = {
        "confidence": confidence.shape,
        "embeddings": embeddings.shape[:1],
        "posterior": posterior.shape[:1],
        "predicted_labels": predicted_labels.shape,
        "boxes": boxes.shape[:1],
        "objectness": objectness.shape,
    }
    if any(shape != (count,) for shape in parallel.values()):
        raise ValueError(f"{path}: non-parallel proposal arrays {parallel}, image_ids={(count,)}")
    if embeddings.ndim != 2 or posterior.ndim != 2 or boxes.shape != (count, 4):
        raise ValueError(f"{path}: invalid dimensions")
    for name, array in {
        "confidence": confidence,
        "embeddings": embeddings,
        "posterior": posterior,
        "boxes": boxes,
        "objectness": objectness,
    }.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: {name} contains non-finite values")
    meta_path = path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return ProposalSet(
        path,
        image_ids,
        confidence,
        embeddings,
        posterior,
        predicted_labels,
        boxes,
        objectness,
        metadata,
    )


def validate_proposals(
    name: str, proposals: ProposalSet, expected_ids: list[str]
) -> dict[str, Any]:
    ids = np.asarray([str(value) for value in proposals.image_ids], dtype=object)
    unique = list(dict.fromkeys(ids.tolist()))
    if unique != expected_ids:
        missing = sorted(set(expected_ids) - set(unique))
        extra = sorted(set(unique) - set(expected_ids))
        raise ValueError(
            f"{name}: image-ID coverage mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    counts = Counter(ids.tolist())
    if len(set(counts.values())) != 1:
        raise ValueError(f"{name}: proposals per image are not constant: {counts.most_common(5)}")
    if proposals.posterior.shape[1] != 20:
        raise ValueError(
            f"{name}: posterior dim {proposals.posterior.shape[1]} != 20 for OWDETR T1 + unknown"
        )
    if proposals.embeddings.shape[1] != 256:
        raise ValueError(f"{name}: embedding dim {proposals.embeddings.shape[1]} != 256")
    mass = proposals.posterior.sum(axis=1)
    if not np.allclose(mass, 1.0, atol=1e-5):
        raise ValueError(f"{name}: posterior rows are not normalised")
    allowed_labels = set(range(19)) | {80}
    bad = sorted(set(map(int, proposals.predicted_labels.tolist())) - allowed_labels)
    if bad:
        raise ValueError(f"{name}: predicted labels outside introduced/unknown set: {bad}")
    return {
        "image_count": len(unique),
        "proposal_count": int(ids.size),
        "proposals_per_image": int(next(iter(counts.values()))),
        "posterior_dimensions": int(proposals.posterior.shape[1]),
        "embedding_dimensions": int(proposals.embeddings.shape[1]),
    }


def parse_xml(
    image_id: str, annotations_dir: Path, groups: dict[str, str]
) -> tuple[list[GTObject], tuple[int, int]]:
    path = annotations_dir / f"{image_id}.xml"
    if not path.exists():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    size = root.find("size")
    width = int(size.findtext("width")) if size is not None else 0
    height = int(size.findtext("height")) if size is not None else 0
    objects: list[GTObject] = []
    for node in root.findall("object"):
        class_name = canonical_class_name((node.findtext("name") or "").strip())
        if class_name not in OWDETR_CLASS_NAMES:
            raise ValueError(f"{path}: class {class_name!r} not in OWDETR mapping")
        box = node.find("bndbox")
        if box is None:
            continue
        xyxy = tuple(float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax"))
        objects.append(GTObject(image_id, class_name, groups.get(class_name, "known"), xyxy))
    return objects, (width, height)


def build_groups(candidate_ids: list[str], annotations_dir: Path) -> dict[str, str]:
    counts: Counter[str] = Counter({name: 0 for name in UNKNOWN_CLASSES})
    for image_id in candidate_ids:
        root = ET.parse(annotations_dir / f"{image_id}.xml").getroot()
        for node in root.findall("object"):
            name = canonical_class_name((node.findtext("name") or "").strip())
            if name in counts:
                counts[name] += 1
    ordered = sorted(UNKNOWN_CLASSES, key=lambda name: (-counts[name], name))
    head_end = (len(ordered) + 2) // 3
    medium_end = (2 * len(ordered) + 2) // 3
    groups: dict[str, str] = {name: "known" for name in KNOWN_CLASSES}
    for index, name in enumerate(ordered):
        groups[name] = "head" if index < head_end else "medium" if index < medium_end else "tail"
    groups["unknown"] = "unknown_prediction"
    return groups


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def posterior_entropy(posterior: np.ndarray) -> np.ndarray:
    probs = posterior / np.maximum(posterior.sum(axis=1, keepdims=True), 1e-12)
    return np.clip(
        -(probs * np.log(probs + 1e-12)).sum(axis=1) / math.log(probs.shape[1]), 0.0, 1.0
    )


def margin_uncertainty(posterior: np.ndarray) -> np.ndarray:
    probs = posterior / np.maximum(posterior.sum(axis=1, keepdims=True), 1e-12)
    ordered = np.sort(probs, axis=1)
    return 1.0 - (ordered[:, -1] - ordered[:, -2])


def chunked_novelty(candidates: np.ndarray, references: np.ndarray, chunk: int) -> np.ndarray:
    cand = l2_normalise(candidates)
    ref = l2_normalise(references)
    out = np.empty(cand.shape[0], dtype=np.float64)
    ref_t = ref.T
    for start in range(0, cand.shape[0], chunk):
        stop = min(start + chunk, cand.shape[0])
        out[start:stop] = 1.0 - (cand[start:stop] @ ref_t).max(axis=1)
    return out


def cluster_sizes(labels: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return counts[inverse].astype(np.int64)


def rarity_from_labels(labels: np.ndarray) -> np.ndarray:
    sizes = cluster_sizes(labels).astype(np.float64)
    return -np.log(sizes / float(sizes.size))


def kth_distances(vectors: np.ndarray, neighbour_count: int) -> np.ndarray:
    k = min(neighbour_count, vectors.shape[0] - 1)
    if k < 1:
        return np.zeros(vectors.shape[0], dtype=np.float64)
    distances, _ = (
        NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(vectors).kneighbors(vectors)
    )
    return distances[:, k].astype(np.float64)


def coherence_from_labels(
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    neighbour_count: int,
    global_kth: np.ndarray,
    minimum_cluster_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    sizes = cluster_sizes(labels)
    singletons = sizes == 1
    within = np.zeros(labels.size, dtype=np.float64)
    for label in np.unique(labels):
        member = np.flatnonzero(labels == label)
        if member.size > 1:
            within[member] = kth_distances(vectors[member], neighbour_count)
    measurable = ~singletons
    pooled = within[measurable]
    pooled_scale = max(float(np.median(pooled)) if pooled.size else 0.0, 1e-12)
    coherence = np.zeros(labels.size, dtype=np.float64)
    for label in np.unique(labels):
        member = np.flatnonzero(labels == label)
        if member.size < 2:
            continue
        scale = (
            max(float(np.median(within[member])), 1e-12)
            if member.size >= minimum_cluster_size
            else pooled_scale
        )
        coherence[member] = 1.0 / (1.0 + within[member] / scale)
    isolated = singletons | (global_kth > np.quantile(global_kth, 0.9))
    return np.clip(coherence, 0.0, 1.0), isolated


def density_coherence(global_kth: np.ndarray) -> np.ndarray:
    scale = max(float(np.median(global_kth)), 1e-12)
    return np.clip(1.0 / (1.0 + global_kth / scale), 0.0, 1.0)


def aggregate_image_scores(
    image_ids: np.ndarray, scores: np.ndarray, top_k: int
) -> dict[str, float]:
    result: dict[str, float] = {}
    ids = np.asarray([str(value) for value in image_ids], dtype=object)
    for image_id in np.unique(ids):
        values = np.sort(scores[ids == image_id])[::-1]
        result[str(image_id)] = float(values[:top_k].mean())
    return result


def ranked_images(image_scores: dict[str, float]) -> list[str]:
    return [
        str(key) for key in sorted(image_scores, key=lambda item: (-image_scores[item], str(item)))
    ]


def select(ranking: list[str], budget: int) -> list[str]:
    if budget > len(ranking):
        raise ValueError(f"budget {budget} exceeds pool size {len(ranking)}")
    return ranking[:budget]


def jaccard(first: list[str], second: list[str]) -> float:
    left, right = set(first), set(second)
    return len(left & right) / len(left | right) if left or right else float("nan")


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2:
        return float("nan")
    a, b = average_ranks(first), average_ranks(second)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rbo(left: list[str], right: list[str], p: float = 0.9) -> float:
    depth = max(len(left), len(right))
    if depth == 0:
        return float("nan")
    left_seen: set[str] = set()
    right_seen: set[str] = set()
    score = 0.0
    for d in range(1, depth + 1):
        if d <= len(left):
            left_seen.add(left[d - 1])
        if d <= len(right):
            right_seen.add(right[d - 1])
        score += (len(left_seen & right_seen) / d) * (p ** (d - 1))
    return float((1 - p) * score)


def box_iou_xyxy(
    box: tuple[float, float, float, float], boxes: list[tuple[float, float, float, float]]
) -> np.ndarray:
    if not boxes:
        return np.zeros(0, dtype=np.float64)
    ax1, ay1, ax2, ay2 = box
    arr = np.asarray(boxes, dtype=np.float64)
    ix1 = np.maximum(ax1, arr[:, 0])
    iy1 = np.maximum(ay1, arr[:, 1])
    ix2 = np.minimum(ax2, arr[:, 2])
    iy2 = np.minimum(ay2, arr[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, arr[:, 2] - arr[:, 0]) * np.maximum(0.0, arr[:, 3] - arr[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def proposal_xyxy(
    norm_cxcywh: np.ndarray, width: int, height: int
) -> tuple[float, float, float, float]:
    cx, cy, w, h = map(float, norm_cxcywh)
    x1 = (cx - w / 2.0) * width
    y1 = (cy - h / 2.0) * height
    x2 = (cx + w / 2.0) * width
    y2 = (cy + h / 2.0) * height
    return x1, y1, x2, y2


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


def summarise(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()) if arr.size else float("nan"),
        "q05": float(np.quantile(arr, 0.05)) if arr.size else float("nan"),
        "q25": float(np.quantile(arr, 0.25)) if arr.size else float("nan"),
        "q50": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
        "q75": float(np.quantile(arr, 0.75)) if arr.size else float("nan"),
        "q95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
    }


def component_bundle(
    candidate: ProposalSet,
    reference: ProposalSet,
    *,
    cluster_seed: int,
    cluster_count: int,
    neighbour_count: int,
    novelty_chunk: int,
    global_kth_cache: dict[int, np.ndarray],
    novelty_cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    vectors = l2_normalise(candidate.embeddings)
    if "novelty" not in novelty_cache:
        novelty_cache["novelty"] = chunked_novelty(
            candidate.embeddings, reference.embeddings, novelty_chunk
        )
    if neighbour_count not in global_kth_cache:
        global_kth_cache[neighbour_count] = kth_distances(vectors, neighbour_count)
    labels = KMeans(
        n_clusters=min(cluster_count, vectors.shape[0]), random_state=cluster_seed, n_init="auto"
    ).fit_predict(vectors)
    sizes = cluster_sizes(labels)
    rarity_raw = rarity_from_labels(labels)
    coherence_raw, isolated = coherence_from_labels(
        vectors,
        labels,
        neighbour_count=neighbour_count,
        global_kth=global_kth_cache[neighbour_count],
    )
    entropy = posterior_entropy(candidate.posterior)
    margin = margin_uncertainty(candidate.posterior)
    owe_raw = np.sqrt(normalise(entropy, "rank") * normalise(candidate.objectness, "rank"))
    legacy_unc = 1.0 - np.abs(2.0 * candidate.confidence - 1.0)
    rarity_norm = normalise(rarity_raw, "rank")
    coherence_norm = normalise(coherence_raw, "rank")
    gated_raw = rarity_norm * coherence_raw
    density = density_coherence(global_kth_cache[neighbour_count])
    legacy_rarity = normalise(1.0 / np.maximum(sizes.astype(np.float64), 1.0), "minmax")
    legacy_novelty = normalise(novelty_cache["novelty"], "minmax")
    return {
        "labels": labels.astype(np.int64),
        "cluster_sizes": sizes,
        "isolated": isolated,
        "global_kth": global_kth_cache[neighbour_count],
        "raw": {
            "unknown_score": candidate.confidence,
            "objectness": candidate.objectness,
            "posterior_entropy": entropy,
            "objectness_weighted_entropy": owe_raw,
            "margin_uncertainty": margin,
            "novelty": novelty_cache["novelty"],
            "rarity": rarity_raw,
            "coherence": coherence_raw,
            "gated": gated_raw,
            "legacy_uncertainty": legacy_unc,
            "legacy_density_coherence": density,
        },
        "norm": {
            "entropy": normalise(entropy, "rank"),
            "objectness_weighted_entropy": normalise(owe_raw, "rank"),
            "margin": normalise(margin, "rank"),
            "novelty": normalise(novelty_cache["novelty"], "rank"),
            "rarity": rarity_norm,
            "coherence": coherence_norm,
            "gated": normalise(gated_raw, "rank"),
            "legacy_novelty": legacy_novelty,
            "legacy_rarity": legacy_rarity,
            "legacy_density": density,
        },
    }


def strategy_scores(
    bundle: dict[str, Any], image_ids: np.ndarray, *, random_seed: int = 0
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    per_image_random = {image_id: rng.random() for image_id in np.unique(image_ids.astype(str))}
    random_scores = np.asarray(
        [per_image_random[str(image_id)] for image_id in image_ids], dtype=np.float64
    )
    norm = bundle["norm"]
    legacy_gated_p1 = norm["legacy_rarity"] * norm["legacy_density"]
    legacy_gated_p05 = norm["legacy_rarity"] * np.sqrt(norm["legacy_density"])
    legacy_unc = bundle["raw"]["legacy_uncertainty"]
    return {
        "v2:random": random_scores,
        "v2:uncertainty": norm["entropy"],
        "v2:uncertainty_objectness_weighted_entropy": norm["objectness_weighted_entropy"],
        "v2:novelty": norm["novelty"],
        "v2:rarity": norm["rarity"],
        "v2:coherence": norm["coherence"],
        "v2:rarity_coherence": norm["gated"],
        "v2:full": 0.3 * norm["entropy"] + 0.2 * norm["novelty"] + 0.5 * norm["gated"],
        "v2:full_no_coherence": 0.3 * norm["entropy"]
        + 0.2 * norm["novelty"]
        + 0.5 * norm["rarity"],
        "v1:rarity_no_coherence": 0.3 * legacy_unc
        + 0.2 * norm["legacy_novelty"]
        + 0.5 * norm["legacy_rarity"],
        "v1:full_p1": 0.3 * legacy_unc + 0.2 * norm["legacy_novelty"] + 0.5 * legacy_gated_p1,
        "v1:full_p05": 0.3 * legacy_unc + 0.2 * norm["legacy_novelty"] + 0.5 * legacy_gated_p05,
    }


def weight_score(bundle: dict[str, Any], weights: dict[str, float]) -> np.ndarray:
    norm = bundle["norm"]
    return (
        weights.get("uncertainty", 0.0) * norm["entropy"]
        + weights.get("novelty", 0.0) * norm["novelty"]
        + weights.get("rarity", 0.0) * norm["rarity"]
        + weights.get("gated", 0.0) * norm["gated"]
    )


def build_posthoc_gt(
    candidate: ProposalSet,
    candidate_ids: list[str],
    annotations_dir: Path,
    groups: dict[str, str],
    iou_threshold: float,
) -> dict[str, Any]:
    objects_by_image: dict[str, list[GTObject]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    all_objects: list[GTObject] = []
    for image_id in candidate_ids:
        objects, size = parse_xml(image_id, annotations_dir, groups)
        objects_by_image[image_id] = objects
        sizes[image_id] = size
        all_objects.extend(objects)
    matched_group: list[str] = []
    matched_class: list[str] = []
    max_iou: list[float] = []
    on_object: list[bool] = []
    for index, image_id_obj in enumerate(candidate.image_ids):
        image_id = str(image_id_obj)
        width, height = sizes[image_id]
        pbox = proposal_xyxy(candidate.boxes[index], width, height)
        objects = objects_by_image[image_id]
        ious = box_iou_xyxy(pbox, [item.box_xyxy for item in objects])
        if ious.size and float(ious.max()) >= iou_threshold:
            best = int(ious.argmax())
            matched_group.append(objects[best].group)
            matched_class.append(objects[best].class_name)
            max_iou.append(float(ious[best]))
            on_object.append(True)
        else:
            matched_group.append("background")
            matched_class.append("")
            max_iou.append(float(ious.max()) if ious.size else 0.0)
            on_object.append(False)
    image_flags: dict[str, dict[str, Any]] = {}
    for image_id, objects in objects_by_image.items():
        cls = {item.class_name for item in objects}
        grp = {item.group for item in objects}
        image_flags[image_id] = {
            "classes": cls,
            "groups": grp,
            "object_count": len(objects),
            "tail_object_count": sum(1 for item in objects if item.group == "tail"),
            "medium_object_count": sum(1 for item in objects if item.group == "medium"),
            "head_object_count": sum(1 for item in objects if item.group == "head"),
        }
    return {
        "objects_by_image": objects_by_image,
        "image_flags": image_flags,
        "proposal_on_object": np.asarray(on_object, dtype=bool),
        "proposal_gt_group": np.asarray(matched_group, dtype=object),
        "proposal_gt_class": np.asarray(matched_class, dtype=object),
        "proposal_max_iou": np.asarray(max_iou, dtype=np.float64),
    }


def pool_base_rates(gt: dict[str, Any]) -> dict[str, float]:
    flags = gt["image_flags"]
    total_images = len(flags)
    total_objects = sum(item["object_count"] for item in flags.values())
    return {
        f"{group}_image_rate": sum(1 for item in flags.values() if group in item["groups"])
        / total_images
        for group in GROUPS
    } | {
        f"{group}_object_rate": (
            sum(item[f"{group}_object_count"] for item in flags.values()) / total_objects
            if total_objects
            else 0.0
        )
        for group in GROUPS
    }


def selected_driver_mask(
    image_ids: np.ndarray, scores: np.ndarray, selected_images: list[str], top_k: int
) -> np.ndarray:
    ids = np.asarray([str(value) for value in image_ids], dtype=object)
    mask = np.zeros(ids.size, dtype=bool)
    for image_id in selected_images:
        indices = np.flatnonzero(ids == image_id)
        order = indices[np.argsort(-scores[indices], kind="stable")]
        mask[order[:top_k]] = True
    return mask


def selection_metric_row(
    *,
    strategy: str,
    budget: int,
    selected_images: list[str],
    ranking: list[str],
    scores: np.ndarray,
    image_ids: np.ndarray,
    gt: dict[str, Any],
    base: dict[str, float],
    top_k: int,
    reference_rankings: dict[str, list[str]],
) -> dict[str, Any]:
    flags = gt["image_flags"]
    selected_flags = [flags[item] for item in selected_images]
    selected_object_count = sum(item["object_count"] for item in selected_flags)
    selected_tail_classes = {
        obj.class_name
        for image in selected_images
        for obj in gt["objects_by_image"][image]
        if obj.group == "tail"
    }
    driver = selected_driver_mask(image_ids, scores, selected_images, top_k)
    on_object = gt["proposal_on_object"][driver]
    proposal_groups = gt["proposal_gt_group"][driver]
    row: dict[str, Any] = {
        "strategy": strategy,
        "budget": budget,
        "selected_images": len(selected_images),
        "selected_object_count": selected_object_count,
        "distinct_gt_classes_selected": len(
            set().union(*(item["classes"] for item in selected_flags))
        )
        if selected_flags
        else 0,
        "distinct_tail_classes_selected": len(selected_tail_classes),
        "selected_proposal_purity": float(on_object.mean()) if on_object.size else float("nan"),
        "object_positive_image_rate": float(
            np.mean(
                [
                    gt["proposal_on_object"][
                        selected_driver_mask(image_ids, scores, [image], top_k)
                    ].any()
                    for image in selected_images
                ]
            )
        )
        if selected_images
        else float("nan"),
    }
    row["background_only_selection_rate"] = 1.0 - row["object_positive_image_rate"]
    for group in GROUPS:
        image_rate = sum(1 for item in selected_flags if group in item["groups"]) / budget
        object_rate = (
            (sum(item[f"{group}_object_count"] for item in selected_flags) / selected_object_count)
            if selected_object_count
            else 0.0
        )
        proposal_rate = float((proposal_groups == group).mean()) if proposal_groups.size else 0.0
        pool_prop = (
            float((gt["proposal_gt_group"][gt["proposal_on_object"]] == group).mean())
            if gt["proposal_on_object"].any()
            else 0.0
        )
        row[f"{group}_image_coverage"] = image_rate
        row[f"{group}_image_lift"] = (
            image_rate / base[f"{group}_image_rate"]
            if base[f"{group}_image_rate"]
            else float("nan")
        )
        row[f"{group}_object_count_weighted_coverage"] = object_rate
        row[f"{group}_object_count_weighted_lift"] = (
            object_rate / base[f"{group}_object_rate"]
            if base[f"{group}_object_rate"]
            else float("nan")
        )
        row[f"{group}_proposal_coverage"] = proposal_rate
        row[f"{group}_proposal_lift"] = proposal_rate / pool_prop if pool_prop else float("nan")
    for ref_name, ref_ranking in reference_rankings.items():
        ref_sel = select(ref_ranking, budget)
        prefix = ref_name.replace(":", "_").replace("v2_", "").replace("v1_", "")
        row[f"top_budget_overlap_with_{prefix}"] = len(set(selected_images) & set(ref_sel))
        row[f"jaccard_with_{prefix}"] = jaccard(selected_images, ref_sel)
        row[f"rbo_with_{prefix}"] = rbo(ranking[:budget], ref_ranking[:budget])
    return row


def image_score_rows(
    scores_by_strategy: dict[str, np.ndarray], image_ids: np.ndarray, top_k: int, budgets: list[int]
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    rankings: dict[str, list[str]] = {}
    image_scores_all: dict[str, dict[str, float]] = {}
    budget_set = set(budgets)
    for strategy, scores in scores_by_strategy.items():
        image_scores = aggregate_image_scores(image_ids, scores, top_k)
        image_scores_all[strategy] = image_scores
        ranking = ranked_images(image_scores)
        rankings[strategy] = ranking
        rank_by_image = {image_id: index + 1 for index, image_id in enumerate(ranking)}
        for image_id in ranking:
            row = {
                "strategy": strategy,
                "image_id": image_id,
                "image_score": image_scores[image_id],
                "rank": rank_by_image[image_id],
            }
            for budget in budgets:
                row[f"selected_at_{budget}"] = (
                    rank_by_image[image_id] <= budget if budget in budget_set else False
                )
            rows.append(row)
    return rows, rankings, image_scores_all


def metric_rows_for_scores(
    scores_by_strategy: dict[str, np.ndarray],
    image_ids: np.ndarray,
    gt: dict[str, Any],
    base: dict[str, float],
    budgets: list[int],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    image_rows, rankings, _ = image_score_rows(scores_by_strategy, image_ids, top_k, budgets)
    refs = {name: rankings[name] for name in ("v2:full", "v2:random") if name in rankings}
    rows: list[dict[str, Any]] = []
    for strategy, scores in scores_by_strategy.items():
        for budget in budgets:
            ranking = rankings[strategy]
            rows.append(
                selection_metric_row(
                    strategy=strategy,
                    budget=budget,
                    selected_images=select(ranking, budget),
                    ranking=ranking,
                    scores=scores,
                    image_ids=image_ids,
                    gt=gt,
                    base=base,
                    top_k=top_k,
                    reference_rankings=refs,
                )
            )
    return rows, image_rows, rankings


def cluster_stability(
    bundles: dict[int, dict[str, Any]],
    scores_by_seed: dict[int, dict[str, np.ndarray]],
    image_ids: np.ndarray,
    gt: dict[str, Any],
    base: dict[str, float],
    budget: int,
    top_k: int,
) -> list[dict[str, Any]]:
    seeds = sorted(bundles)
    base_seed = seeds[0]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        if seed == base_seed:
            continue
        rows.append(
            {
                "scope": "cluster_seed",
                "strategy": "__cluster_assignment__",
                "seed": seed,
                "budget": budget,
                "cluster_assignment_ari_vs_seed0": adjusted_rand_score(
                    bundles[base_seed]["labels"], bundles[seed]["labels"]
                ),
                "rarity_rank_spearman_vs_seed0": spearman(
                    bundles[base_seed]["norm"]["rarity"], bundles[seed]["norm"]["rarity"]
                ),
                "gated_rank_spearman_vs_seed0": spearman(
                    bundles[base_seed]["norm"]["gated"], bundles[seed]["norm"]["gated"]
                ),
            }
        )
    for strategy in scores_by_seed[base_seed]:
        selections: dict[int, list[str]] = {}
        metrics: list[dict[str, Any]] = []
        for seed in seeds:
            image_scores = aggregate_image_scores(image_ids, scores_by_seed[seed][strategy], top_k)
            ranking = ranked_images(image_scores)
            selected = select(ranking, budget)
            selections[seed] = selected
            metrics.append(
                selection_metric_row(
                    strategy=strategy,
                    budget=budget,
                    selected_images=selected,
                    ranking=ranking,
                    scores=scores_by_seed[seed][strategy],
                    image_ids=image_ids,
                    gt=gt,
                    base=base,
                    top_k=top_k,
                    reference_rankings={},
                )
            )
        pairwise = [jaccard(selections[a], selections[b]) for a, b in combinations(seeds, 2)]
        rows.append(
            {
                "scope": "cluster_seed",
                "strategy": strategy,
                "seed": "all",
                "budget": budget,
                "selection_jaccard_mean": float(np.mean(pairwise)),
                "selection_jaccard_std": float(np.std(pairwise, ddof=1))
                if len(pairwise) > 1
                else 0.0,
                "selection_jaccard_min": float(np.min(pairwise)),
                "tail_lift_mean": float(np.mean([m["tail_image_lift"] for m in metrics])),
                "tail_lift_std": float(np.std([m["tail_image_lift"] for m in metrics], ddof=1)),
                "distinct_class_mean": float(
                    np.mean([m["distinct_gt_classes_selected"] for m in metrics])
                ),
                "distinct_tail_class_mean": float(
                    np.mean([m["distinct_tail_classes_selected"] for m in metrics])
                ),
                "top_budget_stability": float(np.mean(pairwise)),
            }
        )
    return rows


def weight_sweep(
    bundles: dict[int, dict[str, Any]],
    image_ids: np.ndarray,
    gt: dict[str, Any],
    base: dict[str, float],
    baseline_rankings: dict[str, list[str]],
    budget: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configs: list[tuple[str, dict[str, float]]] = [
        ("gate_only", {"gated": 1.0}),
        ("rarity_only", {"rarity": 1.0}),
        ("uncertainty_only", {"uncertainty": 1.0}),
        ("uncertainty_plus_gate", {"uncertainty": 0.3, "gated": 0.7}),
        ("uncertainty_rarity_gate", {"uncertainty": 0.2, "rarity": 0.3, "gated": 0.8}),
        ("original_full", {"uncertainty": 0.3, "novelty": 0.2, "gated": 0.5}),
        ("full_no_novelty", {"uncertainty": 0.3, "gated": 0.5}),
        ("reduced_uncertainty_high_gate", {"uncertainty": 0.1, "gated": 1.0}),
    ]
    for u in (0.0, 0.1, 0.3):
        for n in (0.0, 0.2):
            for r in (0.0, 0.3, 0.5):
                for g in (0.0, 0.5, 1.0, 1.5):
                    if u + n + r + g > 0:
                        configs.append(
                            (
                                f"grid_u{u}_n{n}_r{r}_g{g}",
                                {"uncertainty": u, "novelty": n, "rarity": r, "gated": g},
                            )
                        )
    seen: set[tuple[tuple[str, float], ...]] = set()
    unique: list[tuple[str, dict[str, float]]] = []
    for name, weights in configs:
        key = tuple(sorted(weights.items()))
        if key not in seen:
            unique.append((name, weights))
            seen.add(key)

    base_seed = sorted(bundles)[0]
    rows: list[dict[str, Any]] = []
    selection_by_config: dict[str, list[str]] = {}
    for name, weights in unique:
        scores = weight_score(bundles[base_seed], weights)
        image_scores = aggregate_image_scores(image_ids, scores, top_k)
        ranking = ranked_images(image_scores)
        selected = select(ranking, budget)
        selection_by_config[name] = selected
        row = selection_metric_row(
            strategy=name,
            budget=budget,
            selected_images=selected,
            ranking=ranking,
            scores=scores,
            image_ids=image_ids,
            gt=gt,
            base=base,
            top_k=top_k,
            reference_rankings={
                k: v for k, v in baseline_rankings.items() if k in ("v2:random", "v2:full")
            },
        )
        row.update({f"weight_{key}": value for key, value in weights.items()})
        seed_selections = []
        for _seed, bundle in bundles.items():
            seed_ranking = ranked_images(
                aggregate_image_scores(image_ids, weight_score(bundle, weights), top_k)
            )
            seed_selections.append(select(seed_ranking, budget))
        pairwise = [jaccard(a, b) for a, b in combinations(seed_selections, 2)]
        row["clustering_stability"] = float(np.mean(pairwise)) if pairwise else float("nan")
        row["selection_diversity"] = row["distinct_gt_classes_selected"]
        row["overlap_with_random"] = len(
            set(selected) & set(select(baseline_rankings["v2:random"], budget))
        )
        row["overlap_with_original_full"] = len(
            set(selected) & set(select(baseline_rankings["v2:full"], budget))
        )
        rows.append(row)

    objectives = [
        ("tail_image_lift", 1),
        ("distinct_tail_classes_selected", 1),
        ("distinct_gt_classes_selected", 1),
        ("object_positive_image_rate", 1),
        ("clustering_stability", 1),
        ("background_only_selection_rate", -1),
        ("overlap_with_random", -1),
    ]
    pareto: list[dict[str, Any]] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            better_or_equal = True
            strictly_better = False
            for key, direction in objectives:
                a = direction * float(row.get(key, float("nan")))
                b = direction * float(other.get(key, float("nan")))
                if math.isnan(a) or math.isnan(b):
                    continue
                if b < a - 1e-12:
                    better_or_equal = False
                    break
                if b > a + 1e-12:
                    strictly_better = True
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            out = dict(row)
            out["pareto_efficient"] = True
            pareto.append(out)
    return rows, pareto


def write_report(
    path: Path,
    manifest: dict[str, Any],
    metrics: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    sweep: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
    uncertainty: dict[str, Any],
) -> None:
    by_strategy_budget = {(row["strategy"], row["budget"]): row for row in metrics}
    budget = max(int(row["budget"]) for row in metrics)

    def m(strategy: str, key: str) -> float:
        return float(by_strategy_budget.get((strategy, budget), {}).get(key, float("nan")))

    gate_lift = m("v2:rarity_coherence", "tail_image_lift")
    rarity_lift = m("v2:rarity", "tail_image_lift")
    full_lift = m("v2:full", "tail_image_lift")
    no_coh_lift = m("v2:full_no_coherence", "tail_image_lift")
    novelty_lift = m("v2:novelty", "tail_image_lift")
    entropy_object_rate = m("v2:uncertainty", "object_positive_image_rate")
    owe_object_rate = m("v2:uncertainty_objectness_weighted_entropy", "object_positive_image_rate")
    stability_rows = [
        row for row in stability if row.get("scope") == "cluster_seed" and row.get("seed") == "all"
    ]
    full_stability = next((row for row in stability_rows if row["strategy"] == "v2:full"), {})
    best = sorted(
        pareto,
        key=lambda row: (-float(row["tail_image_lift"]), -float(row["object_positive_image_rate"])),
    )[:5]

    decision_rows = [
        (
            "Gate tail effect",
            f"budget {budget}: rarity_coherence lift={gate_lift:.3g}, rarity lift={rarity_lift:.3g}",
            "medium",
            "Gate survives if lift exceeds rarity and pool base rate.",
            "Carry gate-only/high-gate configs.",
        ),
        (
            "Full dilution",
            f"full lift={full_lift:.3g}, full_no_coherence lift={no_coh_lift:.3g}",
            "medium",
            "Full dilutes the gate if it trails gate-only.",
            "Test full_no_novelty and higher-gate weights.",
        ),
        (
            "Novelty value",
            f"novelty lift={novelty_lift:.3g}; Pareto novelty=0 configs={sum(1 for r in pareto if float(r.get('weight_novelty', 0.0) or 0.0) == 0.0)}",
            "medium",
            "Novelty is weak if it is absent from Pareto-efficient configs.",
            "Do not increase novelty before training evidence.",
        ),
        (
            "Entropy background preference",
            f"Spearman entropy-objectness={uncertainty['entropy_objectness_spearman']:.3g}; on-object mean={uncertainty['entropy_on_object_mean']:.3g}, background mean={uncertainty['entropy_background_mean']:.3g}",
            "high",
            "Plain entropy is risky when it anti-correlates with objectness.",
            "Keep entropy default only as baseline; test OWE.",
        ),
        (
            "Objectness-weighted entropy",
            f"object-positive rate entropy={entropy_object_rate:.3g}, OWE={owe_object_rate:.3g}",
            "medium",
            "Selection outcomes decide uncertainty choice.",
            "Include OWE as a candidate, not silent default.",
        ),
        (
            "Clustering noise",
            f"v2:full mean selection Jaccard across seeds={float(full_stability.get('selection_jaccard_mean', float('nan'))):.3g}",
            "medium",
            "Low fixed-pool Jaccard means clustering threatens selection stability.",
            "Use multiple clustering seeds or stabilize clustering.",
        ),
        (
            "Unknown-filtered clustering",
            f"Unknown-score threshold diagnostics in manifest; top-quartile object rate={manifest['thresholds']['unknown_score']['q75']['object_rate']:.3g}",
            "low",
            "Filtering is justified only if it improves object rate without dropping tail coverage.",
            "Evaluate unknown-filtered clustering before algorithm change.",
        ),
        (
            "Weight tuning sufficiency",
            f"Pareto configs={len(pareto)} from {len(sweep)} zero-training configs",
            "medium",
            "Several Pareto options means tune before redesign.",
            "Take 3-5 Pareto configs into first training run.",
        ),
    ]

    lines = [
        "# Real PROB Stage 1 Report",
        "",
        "## Validation",
        "",
        f"- Checkpoint: `{manifest['checkpoint']['path']}`",
        f"- Checkpoint SHA256: `{manifest['checkpoint']['sha256']}`",
        f"- PROB commit: `{manifest['prob_repo']['commit']}`",
        f"- DAOWOD commit: `{manifest['daowod_repo']['commit']}`",
        f"- Candidate images/proposals: {manifest['candidate']['image_count']} / {manifest['candidate']['proposal_count']}",
        f"- Reference images/proposals: {manifest['reference']['image_count']} / {manifest['reference']['proposal_count']}",
        f"- Posterior dims / embedding dims: {manifest['candidate']['posterior_dimensions']} / {manifest['candidate']['embedding_dimensions']}",
        f"- GT absent from acquisition-time inputs: {manifest['ground_truth_policy']['gt_absent_from_acquisition_inputs']}",
        f"- Class/group GT joined only post hoc: {manifest['ground_truth_policy']['class_group_gt_joined_only_post_hoc']}",
        "",
        "## Direct Answers",
        "",
        f"1. 8x synthetic gate effect survives? Real gate lift is {gate_lift:.3g} at budget {budget}; compare rarity-only {rarity_lift:.3g}.",
        f"2. Full dilutes the gate? Full lift {full_lift:.3g} vs gate-only {gate_lift:.3g}.",
        f"3. Novelty useless? Novelty lift {novelty_lift:.3g}; inspect Pareto configs for novelty-zero dominance.",
        f"4. Entropy prefers background? Entropy/objectness Spearman {uncertainty['entropy_objectness_spearman']:.3g}.",
        f"5. OWE better candidate? Entropy object-positive rate {entropy_object_rate:.3g}; OWE {owe_object_rate:.3g}.",
        f"6. Clustering instability: full mean fixed-pool selection Jaccard {float(full_stability.get('selection_jaccard_mean', float('nan'))):.3g}.",
        "7. Unknown-filtered clustering is justified only as a diagnostic candidate until its stability/tail tradeoff is tested in training.",
        "8. Weight tuning is sufficient before algorithm changes if Pareto configs preserve object rate and class diversity.",
        "9. Recommended first training configs: " + ", ".join(row["strategy"] for row in best),
        f"10. Budgets: {manifest['budgets']}; rounds/seeds: use at least 3 acquisition seeds and 10 clustering seeds for Stage 1-sensitive configs.",
        "",
        "## Decision Table",
        "",
        "| Finding | Real-data evidence | Confidence | Implication | Required action |",
        "|---|---|---:|---|---|",
    ]
    for row in decision_rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `real_stage1_manifest.json`",
            "- `real_proposal_components.csv`",
            "- `real_image_scores.csv`",
            "- `real_selection_metrics.csv`",
            "- `real_clustering_stability.csv`",
            "- `real_weight_sweep.csv`",
            "- `real_pareto_configs.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    budgets = ints(args.budget)
    cluster_seeds = ints(args.cluster_seeds)
    cluster_counts = ints(args.cluster_counts)
    neighbour_counts = ints(args.neighbour_counts)

    candidate_ids = read_ids(args.candidate_ids)
    reference_ids = read_ids(args.reference_ids)
    if set(candidate_ids) & set(reference_ids):
        raise ValueError("candidate and reference image IDs overlap")

    candidate = load_proposals(args.candidate)
    reference = load_proposals(args.reference)
    candidate_summary = validate_proposals("candidate", candidate, candidate_ids)
    reference_summary = validate_proposals("reference", reference, reference_ids)
    groups = build_groups(candidate_ids, args.annotations_dir)
    gt = build_posthoc_gt(
        candidate, candidate_ids, args.annotations_dir, groups, args.iou_threshold
    )
    base = pool_base_rates(gt)

    novelty_cache: dict[str, np.ndarray] = {}
    kth_cache: dict[int, np.ndarray] = {}
    bundles: dict[int, dict[str, Any]] = {}
    for seed in cluster_seeds:
        bundles[seed] = component_bundle(
            candidate,
            reference,
            cluster_seed=seed,
            cluster_count=args.baseline_clusters,
            neighbour_count=args.baseline_neighbours,
            novelty_chunk=args.novelty_candidate_chunk,
            global_kth_cache=kth_cache,
            novelty_cache=novelty_cache,
        )

    baseline = bundles[cluster_seeds[0]]
    scores = strategy_scores(baseline, candidate.image_ids, random_seed=0)
    selection_rows, image_rows, rankings = metric_rows_for_scores(
        scores, candidate.image_ids, gt, base, budgets, args.top_k
    )

    # Per-proposal component table: acquisition-time values first, post-hoc GT at the end.
    ids = np.asarray([str(value) for value in candidate.image_ids], dtype=object)
    image_score_by_strategy = {
        name: aggregate_image_scores(candidate.image_ids, value, args.top_k)
        for name, value in scores.items()
    }
    component_rows: list[dict[str, Any]] = []
    for index in range(ids.size):
        row: dict[str, Any] = {
            "image_id": ids[index],
            "proposal_index": int(index % candidate_summary["proposals_per_image"]),
            "unknown_score": float(candidate.confidence[index]),
            "objectness": float(candidate.objectness[index]),
            "posterior_entropy": float(baseline["raw"]["posterior_entropy"][index]),
            "objectness_weighted_entropy": float(
                baseline["raw"]["objectness_weighted_entropy"][index]
            ),
            "margin_uncertainty": float(baseline["raw"]["margin_uncertainty"][index]),
            "novelty": float(baseline["raw"]["novelty"][index]),
            "rarity": float(baseline["raw"]["rarity"][index]),
            "coherence": float(baseline["raw"]["coherence"][index]),
            "gated_rarity_coherence": float(baseline["raw"]["gated"][index]),
            "pseudo_cluster_id": int(baseline["labels"][index]),
            "cluster_size": int(baseline["cluster_sizes"][index]),
            "isolated": bool(baseline["isolated"][index]),
        }
        for name, values in scores.items():
            safe = name.replace(":", "_")
            row[f"score_{safe}"] = float(values[index])
            row[f"image_score_{safe}"] = float(image_score_by_strategy[name][ids[index]])
        row["posthoc_gt_on_object"] = bool(gt["proposal_on_object"][index])
        row["posthoc_gt_group"] = str(gt["proposal_gt_group"][index])
        row["posthoc_gt_class"] = str(gt["proposal_gt_class"][index])
        row["posthoc_gt_max_iou"] = float(gt["proposal_max_iou"][index])
        component_rows.append(row)

    scores_by_seed = {
        seed: strategy_scores(bundle, candidate.image_ids, random_seed=0)
        for seed, bundle in bundles.items()
    }
    stability_rows = cluster_stability(
        bundles, scores_by_seed, candidate.image_ids, gt, base, max(budgets), args.top_k
    )
    for count in cluster_counts:
        if count == args.baseline_clusters:
            continue
        bundle = component_bundle(
            candidate,
            reference,
            cluster_seed=cluster_seeds[0],
            cluster_count=count,
            neighbour_count=args.baseline_neighbours,
            novelty_chunk=args.novelty_candidate_chunk,
            global_kth_cache=kth_cache,
            novelty_cache=novelty_cache,
        )
        stability_rows.append(
            {
                "scope": "cluster_count",
                "strategy": "__cluster__",
                "cluster_count": count,
                "cluster_assignment_ari_vs_baseline": adjusted_rand_score(
                    baseline["labels"], bundle["labels"]
                ),
                "rarity_rank_spearman_vs_baseline": spearman(
                    baseline["norm"]["rarity"], bundle["norm"]["rarity"]
                ),
                "gated_rank_spearman_vs_baseline": spearman(
                    baseline["norm"]["gated"], bundle["norm"]["gated"]
                ),
            }
        )
    for nn in neighbour_counts:
        if nn == args.baseline_neighbours:
            continue
        bundle = component_bundle(
            candidate,
            reference,
            cluster_seed=cluster_seeds[0],
            cluster_count=args.baseline_clusters,
            neighbour_count=nn,
            novelty_chunk=args.novelty_candidate_chunk,
            global_kth_cache=kth_cache,
            novelty_cache=novelty_cache,
        )
        stability_rows.append(
            {
                "scope": "neighbour_count",
                "strategy": "__coherence__",
                "neighbour_count": nn,
                "coherence_rank_spearman_vs_baseline": spearman(
                    baseline["norm"]["coherence"], bundle["norm"]["coherence"]
                ),
                "gated_rank_spearman_vs_baseline": spearman(
                    baseline["norm"]["gated"], bundle["norm"]["gated"]
                ),
            }
        )

    sweep_rows, pareto_rows = weight_sweep(
        bundles, candidate.image_ids, gt, base, rankings, max(budgets), args.top_k
    )

    on_object = gt["proposal_on_object"]
    entropy = baseline["raw"]["posterior_entropy"]
    owe = baseline["raw"]["objectness_weighted_entropy"]
    uncertainty_report = {
        "entropy_unknown_spearman": spearman(entropy, candidate.confidence),
        "entropy_objectness_spearman": spearman(entropy, candidate.objectness),
        "owe_unknown_spearman": spearman(owe, candidate.confidence),
        "owe_objectness_spearman": spearman(owe, candidate.objectness),
        "entropy_on_object_mean": float(entropy[on_object].mean())
        if on_object.any()
        else float("nan"),
        "entropy_background_mean": float(entropy[~on_object].mean())
        if (~on_object).any()
        else float("nan"),
        "owe_on_object_mean": float(owe[on_object].mean()) if on_object.any() else float("nan"),
        "owe_background_mean": float(owe[~on_object].mean())
        if (~on_object).any()
        else float("nan"),
        "legacy_uncertainty_overlap_jaccard": jaccard(
            select(rankings["v2:uncertainty"], max(budgets)),
            select(rankings["v1:rarity_no_coherence"], max(budgets)),
        ),
    }

    threshold_values = sorted(
        {0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, float(np.quantile(candidate.confidence, 0.75))}
    )
    threshold_summary = {}
    for threshold in threshold_values:
        mask = candidate.confidence >= threshold
        key = (
            "q75"
            if abs(threshold - float(np.quantile(candidate.confidence, 0.75))) < 1e-12
            else str(threshold)
        )
        threshold_summary[key] = {
            "threshold": threshold,
            "proposal_fraction": float(mask.mean()),
            "image_coverage": int(np.unique(ids[mask]).size),
            "object_rate": float(gt["proposal_on_object"][mask].mean())
            if mask.any()
            else float("nan"),
            "tail_proposal_rate": float((gt["proposal_gt_group"][mask] == "tail").mean())
            if mask.any()
            else float("nan"),
        }

    manifest = {
        "schema": "real_prob_stage1_v1",
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
        "prob_repo": git_commit(args.prob_repo),
        "daowod_repo": git_commit(args.daowod_repo),
        "candidate": candidate_summary
        | {"path": str(args.candidate), "sha256": sha256(args.candidate)},
        "reference": reference_summary
        | {"path": str(args.reference), "sha256": sha256(args.reference)},
        "budgets": budgets,
        "cluster_seeds": cluster_seeds,
        "cluster_counts": cluster_counts,
        "neighbour_counts": neighbour_counts,
        "proposal_filtering": {
            "before_filter_candidate_proposals": candidate_summary["proposal_count"],
            "after_filter_candidate_proposals": candidate_summary["proposal_count"],
            "filter": "bridge retained top 100 unknown-score queries per image; no additional Stage 1 analysis filter",
        },
        "image_id_coverage": {
            "candidate_matches_split": True,
            "reference_matches_split": True,
            "candidate_reference_overlap": 0,
        },
        "ground_truth_policy": {
            "gt_absent_from_acquisition_inputs": True,
            "class_group_gt_joined_only_post_hoc": True,
            "posthoc_fields_only_in_component_csv_suffix": "posthoc_*",
        },
        "class_groups": groups,
        "pool_base_rates": base,
        "unknown_score_distribution": summarise(candidate.confidence),
        "objectness_distribution": summarise(candidate.objectness),
        "thresholds": {"unknown_score": threshold_summary},
        "uncertainty_report": uncertainty_report,
    }

    (out / "real_stage1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(out / "real_proposal_components.csv", component_rows)
    write_csv(out / "real_image_scores.csv", image_rows)
    write_csv(out / "real_selection_metrics.csv", selection_rows)
    write_csv(out / "real_clustering_stability.csv", stability_rows)
    write_csv(out / "real_weight_sweep.csv", sweep_rows)
    write_csv(out / "real_pareto_configs.csv", pareto_rows)
    write_report(
        out / "real_stage1_report.md",
        manifest,
        selection_rows,
        stability_rows,
        sweep_rows,
        pareto_rows,
        uncertainty_report,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(out),
                "candidate_proposals": candidate_summary["proposal_count"],
                "reference_proposals": reference_summary["proposal_count"],
                "metrics_rows": len(selection_rows),
                "pareto_configs": len(pareto_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
