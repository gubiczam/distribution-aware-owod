"""A PROB-calibrated synthetic candidate pool.

Purpose and limits, stated up front: this simulator exists to (a) unit-test the
*intended* interaction semantics of rarity and coherence, (b) exercise the full
pipeline deterministically without a GPU, and (c) show exactly what the
diagnostics will report. It is **not** evidence about the real M-OWODB pool.
Every real-pool verdict must come from proposals exported by the bridge from a
real checkpoint.

Calibration comes from a real PROB Task-1 checkpoint sample recorded in
``PROB/results/trained_checkpoint_smoke/diagnostics.json``:

* ``pred_obj`` (the objectness distance head) spans about [69, 467]
* ``objectness = exp(-(obj_temp / hidden_dim) * pred_obj)`` therefore spans
  about [0.09, 0.70] with ``obj_temp = 1.3`` and ``hidden_dim = 256``
* per-class sigmoid scores average about 0.022 with a heavy right tail
* the bridge keeps the top ``max_proposals_per_image`` queries by unknown score
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ObjectArray = NDArray[np.object_]

#: Measured on a real PROB Task-1 checkpoint; see the module docstring.
PRED_OBJ_RANGE: tuple[float, float] = (69.4, 467.2)
OBJECTNESS_TEMPERATURE = 1.3
HIDDEN_DIM = 256
CLASS_SIGMOID_MEAN = 0.022


@dataclass(frozen=True)
class SyntheticPool:
    """A synthetic candidate pool plus the ground truth for post-hoc analysis."""

    image_ids: ObjectArray
    embeddings: FloatArray
    confidence: FloatArray
    posterior: FloatArray
    predicted_labels: IntArray
    objectness: FloatArray
    reference_embeddings: FloatArray
    #: Ground truth. Never pass this into acquisition; post-hoc joins only.
    image_classes: Mapping[str, list[str]]
    class_image_counts: Mapping[str, int]
    true_proposal_class: ObjectArray
    is_on_object: NDArray[np.bool_]
    is_planted_outlier: NDArray[np.bool_]
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def unique_image_ids(self) -> list[str]:
        return list(dict.fromkeys(str(value) for value in self.image_ids.tolist()))

    def class_stats_rows(self) -> list[dict[str, object]]:
        """Rows matching ``class_stats.csv`` so ClassGroups can load them."""

        ordered = sorted(
            self.class_image_counts,
            key=lambda name: (-self.class_image_counts[name], name),
        )
        total = len(ordered)
        head_end, medium_end = (total + 2) // 3, (2 * total + 2) // 3
        rows = []
        for rank, name in enumerate(ordered):
            group = "head" if rank < head_end else "medium" if rank < medium_end else "tail"
            rows.append(
                {
                    "class_name": name,
                    "rank": rank,
                    "group": group,
                    "source_frequency": self.class_image_counts[name],
                    "target_frequency": self.class_image_counts[name],
                    "realised_frequency": self.class_image_counts[name],
                    "absolute_error": 0,
                }
            )
        return rows

    def write_class_stats(self, path) -> None:
        import csv
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self.class_stats_rows()
        with target.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def long_tail_image_counts(*, class_count: int, largest: int, imbalance_ratio: float) -> list[int]:
    """Exponentially decaying per-class image counts, as build_long_tail_pool does."""

    if class_count < 1 or largest < 1 or imbalance_ratio < 1:
        raise ValueError("Invalid long-tail parameters.")
    span = max(class_count - 1, 1)
    return [
        max(1, int(round(largest * imbalance_ratio ** (-rank / span))))
        for rank in range(class_count)
    ]


def simulate_pool(
    *,
    class_count: int = 20,
    largest_class_images: int = 30,
    imbalance_ratio: float = 20.0,
    proposals_per_image: int = 20,
    on_object_fraction: float = 0.35,
    planted_outlier_fraction: float = 0.05,
    embedding_dimension: int = 256,
    class_separation: float = 6.0,
    within_class_spread: float = 1.0,
    background_spread: float = 3.0,
    outlier_radius: float = 25.0,
    known_class_count: int = 40,
    reference_images: int = 30,
    seed: int = 0,
) -> SyntheticPool:
    """Build a deterministic long-tail candidate pool with PROB-like statistics.

    Structure that matters for the audit's questions:

    * class clusters are equally *well formed* and differ only in how many
      proposals they contain, so any coherence difference between head and tail
      is a pure sample-size artifact, which is what S5 is about;
    * ``on_object_fraction`` controls how many kept proposals actually sit on an
      object — the rest are duplicate/background queries, which is what makes the
      real pool's local density largely image-driven;
    * ``planted_outlier_fraction`` adds genuinely isolated proposals, so
      "rare and isolated" can be distinguished from "rare and coherent".
    """

    if not 0.0 < on_object_fraction <= 1.0:
        raise ValueError("on_object_fraction must lie in (0, 1].")
    if not 0.0 <= planted_outlier_fraction < 1.0:
        raise ValueError("planted_outlier_fraction must lie in [0, 1).")

    rng = np.random.default_rng(seed)
    counts = long_tail_image_counts(
        class_count=class_count,
        largest=largest_class_images,
        imbalance_ratio=imbalance_ratio,
    )
    class_names = [f"task_class_{index:02d}" for index in range(class_count)]
    class_image_counts = dict(zip(class_names, counts, strict=True))

    centres = rng.normal(size=(class_count, embedding_dimension)) * class_separation
    background_centre = rng.normal(size=embedding_dimension) * class_separation

    image_ids: list[str] = []
    embeddings: list[FloatArray] = []
    true_classes: list[str] = []
    on_object_flags: list[bool] = []
    outlier_flags: list[bool] = []
    image_classes: dict[str, list[str]] = {}

    image_counter = 0
    for class_index, (class_name, image_count) in enumerate(class_image_counts.items()):
        for _ in range(image_count):
            image_id = f"sim_{image_counter:06d}"
            image_counter += 1
            image_classes[image_id] = [class_name]
            for _ in range(proposals_per_image):
                draw = rng.random()
                if draw < planted_outlier_fraction:
                    direction = rng.normal(size=embedding_dimension)
                    direction /= max(np.linalg.norm(direction), 1e-12)
                    embeddings.append(direction * outlier_radius)
                    true_classes.append("outlier")
                    on_object_flags.append(False)
                    outlier_flags.append(True)
                elif draw < planted_outlier_fraction + on_object_fraction:
                    embeddings.append(
                        centres[class_index]
                        + rng.normal(scale=within_class_spread, size=embedding_dimension)
                    )
                    true_classes.append(class_name)
                    on_object_flags.append(True)
                    outlier_flags.append(False)
                else:
                    embeddings.append(
                        background_centre
                        + rng.normal(scale=background_spread, size=embedding_dimension)
                    )
                    true_classes.append("background")
                    on_object_flags.append(False)
                    outlier_flags.append(False)
                image_ids.append(image_id)

    embedding_matrix = np.asarray(embeddings, dtype=np.float64)
    proposal_count = embedding_matrix.shape[0]
    on_object = np.asarray(on_object_flags, dtype=np.bool_)
    planted = np.asarray(outlier_flags, dtype=np.bool_)

    # Objectness: on-object queries sit at the low-distance end of the measured
    # pred_obj range, background queries at the high end.
    low, high = PRED_OBJ_RANGE
    pred_obj = np.where(
        on_object,
        rng.uniform(low, low + 0.45 * (high - low), proposal_count),
        rng.uniform(low + 0.35 * (high - low), high, proposal_count),
    )
    objectness = np.exp(-(OBJECTNESS_TEMPERATURE / HIDDEN_DIM) * pred_obj)

    # Posterior over the introduced known classes plus the unknown slot.
    posterior_classes = known_class_count + 1
    logits = rng.normal(loc=-3.8, scale=1.2, size=(proposal_count, posterior_classes))
    # An on-object unknown proposal concentrates mass on the unknown slot; a
    # background query stays diffuse. This is what gives entropy information the
    # unknown score alone does not carry.
    logits[on_object, -1] += rng.normal(loc=3.2, scale=0.9, size=int(on_object.sum()))
    logits[planted, -1] += rng.normal(loc=1.2, scale=1.5, size=int(planted.sum()))
    class_probability = 1.0 / (1.0 + np.exp(-logits))
    combined = objectness[:, None] * class_probability
    posterior = combined / np.maximum(combined.sum(axis=1, keepdims=True), 1e-12)
    confidence = np.clip(combined[:, -1], 0.0, 1.0)
    predicted_labels = np.where(
        posterior.argmax(axis=1) == posterior_classes - 1,
        known_class_count,
        posterior.argmax(axis=1),
    ).astype(np.int64)

    references = background_centre + rng.normal(
        scale=background_spread,
        size=(max(reference_images, 1) * 3, embedding_dimension),
    )

    return SyntheticPool(
        image_ids=np.asarray(image_ids, dtype=object),
        embeddings=embedding_matrix,
        confidence=confidence,
        posterior=posterior,
        predicted_labels=predicted_labels,
        objectness=objectness,
        reference_embeddings=references,
        image_classes=image_classes,
        class_image_counts=class_image_counts,
        true_proposal_class=np.asarray(true_classes, dtype=object),
        is_on_object=on_object,
        is_planted_outlier=planted,
        metadata={
            "calibration": "PROB Task-1 checkpoint sample (see module docstring)",
            "seed": seed,
            "class_count": class_count,
            "imbalance_ratio": imbalance_ratio,
            "proposals_per_image": proposals_per_image,
            "on_object_fraction": on_object_fraction,
            "planted_outlier_fraction": planted_outlier_fraction,
            "images": len(image_classes),
            "proposals": proposal_count,
            "mean_objectness": float(objectness.mean()),
            "mean_unknown_score": float(confidence.mean()),
            "posterior_classes": posterior_classes,
            "warning": "synthetic; not evidence about the real M-OWODB pool",
        },
    )


def structured_regime_pool(
    *,
    proposals_per_class: Sequence[int],
    embedding_dimension: int = 32,
    class_separation: float = 8.0,
    within_class_spread: float = 0.6,
    isolated_count: int = 0,
    outlier_radius: float = 30.0,
    seed: int = 0,
) -> tuple[FloatArray, IntArray, ObjectArray]:
    """A minimal pool with exact per-class proposal counts, for regime tests.

    Returns ``(embeddings, true_class_index, image_ids)``. Each class is equally
    well formed; only its size differs. Isolated proposals are appended as their
    own singleton classes so "rare and isolated" is representable.
    """

    rng = np.random.default_rng(seed)
    class_count = len(proposals_per_class)
    centres = rng.normal(size=(class_count, embedding_dimension)) * class_separation
    blocks, labels = [], []
    for index, size in enumerate(proposals_per_class):
        blocks.append(
            centres[index] + rng.normal(scale=within_class_spread, size=(size, embedding_dimension))
        )
        labels.extend([index] * size)
    for offset in range(isolated_count):
        direction = rng.normal(size=embedding_dimension)
        direction /= max(np.linalg.norm(direction), 1e-12)
        blocks.append((direction * outlier_radius)[None, :])
        labels.append(class_count + offset)
    embeddings = np.vstack(blocks)
    image_ids = np.asarray(
        [f"img_{index // 4:04d}" for index in range(embeddings.shape[0])], dtype=object
    )
    return embeddings, np.asarray(labels, dtype=np.int64), image_ids
