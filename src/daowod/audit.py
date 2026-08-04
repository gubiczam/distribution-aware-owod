"""Component-level diagnostics: why an acquisition signal works, or does not.

This module exists because the first full run of Contribution A produced a
negative result — the coherence gate did not improve tail discovery — and a
negative result is only useful if it can be localised. A budget curve says *that*
a strategy failed; these measurements say *which component* failed and whether the
failure is fixable.

Three questions, three families of measurement:

**Is the information in the representation?** :func:`probe_auc` fits a
cross-validated linear probe on the embeddings and reports ROC-AUC for
"unknown object versus background". This is an *upper bound* on what any function
of those features could achieve. It uses ground truth and is therefore a
diagnostic only — never an acquisition signal.

**Do the unsupervised estimators extract it?** :func:`signal_auc` scores the same
targets with the acquisition components themselves. The gap between the two is the
quantity that matters: a large gap means the estimator is at fault, a small gap
means the representation is.

**Is the premise behind coherence true?** :func:`neighbourhood_composition`
measures what a proposal's nearest neighbours actually are. The gate assumes a
rare true class forms a locally consistent group while a false positive does not.
That is a statement about neighbourhood label composition, and it is directly
checkable.

Everything here reads ground truth. Nothing here may be imported by the
acquisition path; :mod:`daowod.discovery` and :mod:`daowod.active` enforce that
separation, and the leakage guard re-derives every score from its recorded
components regardless.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: The contrasts every diagnostic reports. "versus background" is the operative
#: comparison because background is what the budget is actually lost to: 75 % of a
#: real Task-1 candidate pool.
DEFAULT_TARGETS: tuple[str, ...] = (
    "unknown_vs_background",
    "tail_vs_background",
    "onobject_vs_background",
    "unknown_vs_known",
)


class AuditError(ValueError):
    """Raised when a diagnostic cannot be computed from the given inputs."""


@dataclass(frozen=True)
class Strata:
    """Oracle strata for one pool, as boolean masks."""

    is_unknown: BoolArray
    is_known: BoolArray
    is_background: BoolArray
    is_head: BoolArray
    is_medium: BoolArray
    is_tail: BoolArray
    class_name: NDArray[np.object_]

    @classmethod
    def from_oracle(cls, match_kind: ArrayLike, group: ArrayLike, class_name: ArrayLike) -> Strata:
        kind = np.asarray([str(value) for value in np.asarray(match_kind).tolist()], dtype=object)
        groups = np.asarray([str(value) for value in np.asarray(group).tolist()], dtype=object)
        names = np.asarray([str(value) for value in np.asarray(class_name).tolist()], dtype=object)
        return cls(
            is_unknown=(kind == "unknown"),
            is_known=(kind == "known"),
            is_background=(kind == "background"),
            is_head=(groups == "head"),
            is_medium=(groups == "medium"),
            is_tail=(groups == "tail"),
            class_name=names,
        )

    def target(self, name: str) -> tuple[BoolArray, BoolArray]:
        """Positive and negative masks for one named contrast."""

        table = {
            "unknown_vs_background": (self.is_unknown, self.is_background),
            "tail_vs_background": (self.is_tail, self.is_background),
            "head_vs_background": (self.is_head, self.is_background),
            "onobject_vs_background": (~self.is_background, self.is_background),
            "unknown_vs_known": (self.is_unknown, self.is_known),
            "tail_vs_head": (self.is_tail, self.is_head),
        }
        if name not in table:
            raise AuditError(f"Unknown target {name!r}. Supported: {sorted(table)}")
        return table[name]

    def counts(self) -> dict[str, int]:
        return {
            "unknown": int(self.is_unknown.sum()),
            "known": int(self.is_known.sum()),
            "background": int(self.is_background.sum()),
            "head": int(self.is_head.sum()),
            "medium": int(self.is_medium.sum()),
            "tail": int(self.is_tail.sum()),
        }

    def label_for_purity(self) -> NDArray[np.object_]:
        """Class name for objects, one shared label for background.

        Background gets a single label so that "local label homogeneity" is
        comparable across strata: for an unknown proposal it means "neighbours of
        my own class", for a background proposal "neighbours that are also
        background".
        """

        labels = self.class_name.copy()
        labels[self.is_background] = "__background__"
        return labels


def signal_auc(
    signals: Mapping[str, ArrayLike],
    strata: Strata,
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    minimum_positives: int = 10,
) -> list[dict[str, object]]:
    """ROC-AUC of each acquisition signal for each contrast.

    AUC rather than a correlation because the signals are used as *rankings*: the
    budget buys a prefix of the ordering, so the question is exactly "how often is
    a true unknown ranked above a background region".
    """

    rows: list[dict[str, object]] = []
    for target in targets:
        positive, negative = strata.target(target)
        keep = positive | negative
        y = positive[keep].astype(int)
        if int(y.sum()) < minimum_positives:
            continue
        for name, values in signals.items():
            array = np.asarray(values, dtype=np.float64)
            if array.shape[0] != keep.shape[0]:
                raise AuditError(f"Signal {name!r} is not parallel to the strata.")
            rows.append(
                {
                    "target": target,
                    "signal": name,
                    "kind": "unsupervised",
                    "positives": int(y.sum()),
                    "negatives": int(keep.sum() - y.sum()),
                    "roc_auc": float(roc_auc_score(y, array[keep])),
                }
            )
    return rows


def precision_at_budget(
    signals: Mapping[str, ArrayLike],
    strata: Strata,
    *,
    budgets: Sequence[int] = (100, 500, 2000),
    positive: str = "unknown",
) -> list[dict[str, object]]:
    """Precision in the top-K of each signal's ranking — the decision-relevant number.

    ROC-AUC summarises the *whole* ordering, but an annotation budget buys a short
    prefix of it: 2 000 of 48 000 regions is the top 4 %. Two signals with similar
    AUC can differ several-fold in the top 4 %, and the reverse is also true, so AUC
    is not a sufficient basis on which to choose an experiment.

    This function exists because the first version of this audit made exactly that
    mistake. It ranked candidate estimators by AUC, selected the label-anchored
    support term on an AUC of 0.708 against the baseline coherence term's 0.481, and
    thereby over-predicted what the anchored term would deliver in a campaign.
    Measured at the actual budget, the same comparison reads: anchored support 4.7x
    the base rate, baseline gated term **0.31x** — i.e. the baseline's distribution
    term is not merely uninformative in the top 4 %, it is actively anti-selective,
    which the AUC of 0.486 does not reveal.

    ``lift`` is precision divided by the pool's base rate, so 1.0 is
    indistinguishable from random sampling and below 1.0 is worse than random.
    """

    mask = strata.is_tail if positive == "tail" else strata.is_unknown
    base_rate = float(mask.mean())
    rows: list[dict[str, object]] = []
    for name, values in signals.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape[0] != mask.shape[0]:
            raise AuditError(f"Signal {name!r} is not parallel to the strata.")
        order = np.argsort(-array, kind="stable")
        for budget in sorted(int(value) for value in budgets):
            taken = order[: min(budget, array.shape[0])]
            precision = float(mask[taken].mean())
            rows.append(
                {
                    "signal": name,
                    "positive": positive,
                    "budget": int(budget),
                    "budget_fraction": float(min(budget, array.shape[0]) / array.shape[0]),
                    "positives_in_top_k": int(mask[taken].sum()),
                    "precision": precision,
                    "base_rate": base_rate,
                    "lift_over_random": (
                        float(precision / base_rate) if base_rate > 0 else float("nan")
                    ),
                    "worse_than_random": bool(precision < base_rate),
                }
            )
    return rows


def probe_auc(
    feature_sets: Mapping[str, ArrayLike],
    strata: Strata,
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    folds: int = 3,
    seed: int = 0,
    minimum_positives: int = 10,
) -> list[dict[str, object]]:
    """Cross-validated linear-probe ROC-AUC: the representation's upper bound.

    Balanced class weights and cross-validation, because the positive rate is
    around 1 %: an unweighted probe would predict the majority class and a
    single split would put too few positives in the test fold to estimate an AUC.
    """

    rows: list[dict[str, object]] = []
    for target in targets:
        positive, negative = strata.target(target)
        keep = positive | negative
        y = positive[keep].astype(int)
        if int(y.sum()) < max(minimum_positives, folds * 2):
            continue
        for name, features in feature_sets.items():
            matrix = np.asarray(features, dtype=np.float64)
            if matrix.ndim == 1:
                matrix = matrix.reshape(-1, 1)
            observed = matrix[keep]
            scores: list[float] = []
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            for train, test in splitter.split(observed, y):
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2000, class_weight="balanced"),
                )
                model.fit(observed[train], y[train])
                scores.append(
                    float(roc_auc_score(y[test], model.predict_proba(observed[test])[:, 1]))
                )
            rows.append(
                {
                    "target": target,
                    "signal": name,
                    "kind": "supervised_probe",
                    "positives": int(y.sum()),
                    "negatives": int(keep.sum() - y.sum()),
                    "roc_auc": float(np.mean(scores)),
                    "roc_auc_sd": float(np.std(scores, ddof=1))
                    if len(scores) > 1
                    else float("nan"),
                    "folds": folds,
                }
            )
    return rows


def neighbourhood_composition(
    embeddings: ArrayLike,
    strata: Strata,
    *,
    neighbours: int = 10,
    subset: ArrayLike | None = None,
    label: str = "full pool",
) -> list[dict[str, object]]:
    """What each stratum's nearest neighbours actually are.

    This is the direct test of the coherence premise. ``same_label_fraction`` is
    the local label homogeneity a density-based coherence term is implicitly
    measuring; if it is higher for background than for tail classes, then any such
    term ranks background above the tail and the gate cannot help.
    """

    matrix = np.asarray(embeddings, dtype=np.float64)
    indices = (
        np.arange(matrix.shape[0])
        if subset is None
        else np.flatnonzero(np.asarray(subset, dtype=np.bool_))
    )
    if indices.size < neighbours + 2:
        raise AuditError(
            f"{label}: {indices.size} proposals is too few for a {neighbours}-neighbour report."
        )
    vectors = matrix[indices]
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    _, neighbour_index = (
        NearestNeighbors(n_neighbors=neighbours + 1).fit(vectors).kneighbors(vectors)
    )
    neighbour_index = neighbour_index[:, 1:]

    labels = strata.label_for_purity()[indices]
    unknown = strata.is_unknown[indices]
    background = strata.is_background[indices]
    same = np.array(
        [
            float(np.mean(labels[row] == labels[position]))
            for position, row in enumerate(neighbour_index)
        ]
    )
    neighbour_unknown = unknown[neighbour_index].mean(axis=1)
    neighbour_on_object = (~background[neighbour_index]).mean(axis=1)

    masks = {
        "head": strata.is_head[indices],
        "medium": strata.is_medium[indices],
        "tail": strata.is_tail[indices],
        "known": strata.is_known[indices],
        "background": background,
    }
    rows: list[dict[str, object]] = []
    for stratum, mask in masks.items():
        if not mask.any():
            continue
        rows.append(
            {
                "subset": label,
                "subset_size": int(indices.size),
                "subset_unknown_rate": float(unknown.mean()),
                "subset_background_rate": float(background.mean()),
                "neighbours": int(neighbours),
                "stratum": stratum,
                "n": int(mask.sum()),
                "same_label_fraction": float(same[mask].mean()),
                "neighbour_unknown_fraction": float(neighbour_unknown[mask].mean()),
                "neighbour_on_object_fraction": float(neighbour_on_object[mask].mean()),
            }
        )
    return rows


def pseudo_class_quality(
    pseudo_labels: ArrayLike,
    rarity: ArrayLike,
    strata: Strata,
) -> dict[str, object]:
    """Does the pseudo-class assignment recover anything about the real classes?

    Reports cluster/class agreement and, decisively, the rank correlation between
    the *estimated* rarity a proposal receives and the *true* frequency of its
    class. The distribution-aware term is only "distribution-aware" to the extent
    that this correlation is non-zero.
    """

    labels = np.asarray(pseudo_labels, dtype=np.int64)
    values = np.asarray(rarity, dtype=np.float64)
    if labels.shape != values.shape:
        raise AuditError("pseudo_labels and rarity must be parallel.")
    unknown = strata.is_unknown
    coarse = np.where(strata.is_background, "__background__", strata.class_name)

    report: dict[str, object] = {
        "clusters": int(np.unique(labels).size),
        "cluster_size_median": float(np.median(np.bincount(labels - labels.min()))),
        "ari_all_strata": float(adjusted_rand_score(coarse, labels)),
        "nmi_all_strata": float(normalized_mutual_info_score(coarse, labels)),
    }
    if unknown.sum() > 1:
        report["ari_unknown_classes"] = float(
            adjusted_rand_score(strata.class_name[unknown], labels[unknown])
        )
        report["nmi_unknown_classes"] = float(
            normalized_mutual_info_score(strata.class_name[unknown], labels[unknown])
        )
        names, counts = np.unique(strata.class_name[unknown], return_counts=True)
        frequency = dict(zip(names.tolist(), counts.tolist(), strict=True))
        true_frequency = np.array(
            [float(frequency[name]) for name in strata.class_name[unknown].tolist()]
        )
        # Spearman via ranks, to avoid a scipy dependency in the library path.
        report["rarity_vs_true_rarity_spearman"] = float(
            _spearman(values[unknown], -true_frequency)
        )
        report["unknown_classes_present"] = int(names.size)

    background_fraction = np.array(
        [float(strata.is_background[labels == cluster].mean()) for cluster in np.unique(labels)]
    )
    report["cluster_background_fraction_median"] = float(np.median(background_fraction))
    report["clusters_over_90_percent_background"] = int((background_fraction > 0.9).sum())
    return report


def _spearman(left: ArrayLike, right: ArrayLike) -> float:
    from daowod.normalisation import average_ranks

    a = average_ranks(left)
    b = average_ranks(right)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a**2).sum() * (b**2).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def retention_curve(
    ranking: ArrayLike,
    strata: Strata,
    object_index: ArrayLike,
    *,
    fractions: Sequence[float] = (1.0, 0.5, 0.25, 0.1),
    name: str = "ranking",
) -> list[dict[str, object]]:
    """What a ground-truth-free pool filter keeps and what it throws away.

    Reports, per keep-fraction, the resulting unknown rate and how many *distinct*
    unknown and tail objects survive. A filter is only worth applying if it raises
    the unknown rate faster than it destroys the reachable object set — the
    denominator every discovery metric is measured against.
    """

    scores = np.asarray(ranking, dtype=np.float64)
    objects = np.asarray(object_index, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")

    def distinct(mask: BoolArray) -> tuple[int, int]:
        unknown_objects = {
            int(value) for value in objects[mask & strata.is_unknown].tolist() if value >= 0
        }
        tail_objects = {
            int(value) for value in objects[mask & strata.is_tail].tolist() if value >= 0
        }
        return len(unknown_objects), len(tail_objects)

    total_unknown, total_tail = distinct(np.ones(scores.shape[0], dtype=np.bool_))
    rows: list[dict[str, object]] = []
    for fraction in fractions:
        keep = np.zeros(scores.shape[0], dtype=np.bool_)
        keep[order[: max(1, int(scores.shape[0] * float(fraction)))]] = True
        unknown_objects, tail_objects = distinct(keep)
        rows.append(
            {
                "ranking": name,
                "keep_fraction": float(fraction),
                "pool_size": int(keep.sum()),
                "unknown_rate": float(strata.is_unknown[keep].mean()),
                "background_rate": float(strata.is_background[keep].mean()),
                "unknown_proposals": int(strata.is_unknown[keep].sum()),
                "unknown_objects": unknown_objects,
                "unknown_objects_total": total_unknown,
                "tail_objects": tail_objects,
                "tail_objects_total": total_tail,
                "unknown_object_retention": (
                    float(unknown_objects / total_unknown) if total_unknown else float("nan")
                ),
                "tail_object_retention": (
                    float(tail_objects / total_tail) if total_tail else float("nan")
                ),
            }
        )
    return rows


def revealed_sample_complexity(
    embeddings: ArrayLike,
    strata: Strata,
    *,
    revealed_counts: Sequence[int] = (5, 10, 20, 40, 80, 160),
    draws: int = 8,
    negative_ratio: float = 50.0,
    seed: int = 0,
) -> list[dict[str, object]]:
    """How much oracle supervision a label-anchored signal needs to beat chance.

    Simulates what a campaign owns after it has revealed ``m`` unknown regions —
    and, at a measured ~2 % annotation precision, roughly ``m * negative_ratio``
    non-unknown ones — then scores the *held-out* remainder. Held-out because a
    signal fitted on a region cannot be credited with ranking that same region.

    Two estimators: mean similarity to the revealed unknowns (what
    :func:`daowod.revealed.support` computes) and a linear probe on the revealed
    labels. The curve is the evidence for whether anchoring is viable at realistic
    budgets, and it is measured before any campaign is run.
    """

    matrix = np.asarray(embeddings, dtype=np.float64)
    unit = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    unknown_positions = np.flatnonzero(strata.is_unknown)
    other_positions = np.flatnonzero(~strata.is_unknown)
    rows: list[dict[str, object]] = []

    for count in revealed_counts:
        if count > unknown_positions.size:
            continue
        similarity_scores: list[float] = []
        probe_scores: list[float] = []
        tail_scores: list[float] = []
        for draw in range(draws):
            generator = np.random.default_rng(seed + draw)
            positives = generator.choice(unknown_positions, size=count, replace=False)
            negatives = generator.choice(
                other_positions,
                size=int(min(other_positions.size, round(count * negative_ratio))),
                replace=False,
            )
            used = np.zeros(matrix.shape[0], dtype=np.bool_)
            used[positives] = True
            used[negatives] = True
            held = ~used

            similarity = (unit @ unit[positives].T).mean(axis=1)
            similarity_scores.append(_held_out_auc(similarity, strata, held, "unknown"))

            features = np.vstack([unit[positives], unit[negatives]])
            targets = np.r_[np.ones(positives.size), np.zeros(negatives.size)]
            model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")
            )
            model.fit(features, targets)
            predicted = model.predict_proba(unit)[:, 1]
            probe_scores.append(_held_out_auc(predicted, strata, held, "unknown"))
            tail = _held_out_auc(predicted, strata, held, "tail")
            if np.isfinite(tail):
                tail_scores.append(tail)

        rows.append(
            {
                "revealed_unknowns": int(count),
                "revealed_negatives": int(min(other_positions.size, round(count * negative_ratio))),
                "draws": int(draws),
                "similarity_auc_mean": float(np.mean(similarity_scores)),
                "similarity_auc_sd": float(np.std(similarity_scores, ddof=1)),
                "probe_auc_mean": float(np.mean(probe_scores)),
                "probe_auc_sd": float(np.std(probe_scores, ddof=1)),
                "probe_tail_auc_mean": float(np.mean(tail_scores)) if tail_scores else float("nan"),
            }
        )
    return rows


def _held_out_auc(scores: FloatArray, strata: Strata, held: BoolArray, positive: str) -> float:
    mask = strata.is_tail if positive == "tail" else strata.is_unknown
    pos = mask[held]
    keep = pos | strata.is_background[held]
    if int(pos.sum()) < 3:
        return float("nan")
    return float(roc_auc_score(pos[keep].astype(int), scores[held][keep]))


#: The components the distribution-aware term is built from. Separated from the
#: other signals because the audit's central question is how *these* compare with
#: what is freely available, not how the best of everything compares.
DISTRIBUTION_SIGNALS: tuple[str, ...] = (
    "rarity_pseudo_class",
    "coherence_relative_within_cluster",
    "coherence_radius_core",
    "gated_rarity_x_coherence",
)


def summarise_gap(
    rows: Sequence[Mapping[str, object]],
    *,
    target: str,
    distribution_signals: Sequence[str] = DISTRIBUTION_SIGNALS,
) -> dict[str, object]:
    """Decompose the shortfall into the two gaps that imply different fixes.

    A single "supervised ceiling minus best signal" number is not enough, because
    two very different failures produce a similar figure. The audit therefore
    reports both:

    ``estimator_gap`` = best freely available unsupervised signal minus best
        distribution component. This is what the *implementation* leaves on the
        table. A large value means the distribution-aware terms are worse than
        something already free, and the fix is a better estimator.

    ``representation_headroom`` = supervised ceiling minus best freely available
        signal. This is the most a better unsupervised estimator could ever win in
        this feature space. A small value means even a perfect estimator would
        barely beat the free signal, and the fix is a different representation.

    Reporting them separately is what distinguishes "our estimator is broken" from
    "this feature space has nothing more to give" — and the first run's data has
    both properties at once, which a single number would hide.
    """

    supervised = [
        row for row in rows if row["kind"] == "supervised_probe" and row["target"] == target
    ]
    unsupervised = [
        row for row in rows if row["kind"] == "unsupervised" and row["target"] == target
    ]
    if not supervised or not unsupervised:
        return {"target": target, "available": False}

    wanted = set(distribution_signals)
    distribution = [row for row in unsupervised if str(row["signal"]) in wanted]
    other = [row for row in unsupervised if str(row["signal"]) not in wanted]
    ceiling = max(supervised, key=lambda row: float(row["roc_auc"]))
    best_free = max(other or unsupervised, key=lambda row: float(row["roc_auc"]))
    best_distribution = (
        max(distribution, key=lambda row: float(row["roc_auc"])) if distribution else None
    )

    estimator_gap = (
        float(best_free["roc_auc"]) - float(best_distribution["roc_auc"])
        if best_distribution is not None
        else float("nan")
    )
    headroom = float(ceiling["roc_auc"]) - float(best_free["roc_auc"])
    verdicts: list[str] = []
    if np.isfinite(estimator_gap) and estimator_gap >= 0.1:
        verdicts.append(
            "the distribution-aware components score below a signal that is already "
            "free, so the estimator is the first thing to fix"
        )
    if headroom < 0.1:
        verdicts.append(
            "even a perfect unsupervised estimator would gain little over that free "
            "signal here, so a large win needs a different representation"
        )
    else:
        verdicts.append(
            "the representation still holds unexploited signal, so a better "
            "estimator can win without changing features"
        )
    return {
        "target": target,
        "available": True,
        "supervised_ceiling": float(ceiling["roc_auc"]),
        "supervised_ceiling_features": str(ceiling["signal"]),
        "best_free_unsupervised": float(best_free["roc_auc"]),
        "best_free_unsupervised_signal": str(best_free["signal"]),
        "best_distribution_component": (
            float(best_distribution["roc_auc"]) if best_distribution is not None else float("nan")
        ),
        "best_distribution_component_signal": (
            str(best_distribution["signal"]) if best_distribution is not None else ""
        ),
        "estimator_gap": estimator_gap,
        "representation_headroom": headroom,
        "verdict": "; ".join(verdicts),
    }


def static_ranking_discovery(
    ranking: ArrayLike,
    strata: Strata,
    object_index: ArrayLike,
    *,
    budgets: Sequence[int],
    name: str = "ranking",
) -> list[dict[str, object]]:
    """Discovery achieved by ranking the pool once, with no rounds and no feedback.

    This reference puts a floor under the whole apparatus. A multi-round,
    distribution-aware strategy costs a clustering, a neighbour search and an
    oracle-feedback loop; a static ranking by a detector output costs one sort. If
    the strategy does not beat the sort on the metric that matters, the complexity
    is not earning anything, and a reader is entitled to see that comparison in
    discovered-object counts rather than only in AUC.
    """

    scores = np.asarray(ranking, dtype=np.float64)
    objects = np.asarray(object_index, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    total_unknown = {int(value) for value in objects[strata.is_unknown].tolist() if value >= 0}
    total_tail = {int(value) for value in objects[strata.is_tail].tolist() if value >= 0}

    rows: list[dict[str, object]] = []
    for budget in sorted(int(value) for value in budgets):
        selected = np.zeros(scores.shape[0], dtype=np.bool_)
        selected[order[: min(budget, scores.shape[0])]] = True
        found_unknown = {
            int(value) for value in objects[selected & strata.is_unknown].tolist() if value >= 0
        }
        found_tail = {
            int(value) for value in objects[selected & strata.is_tail].tolist() if value >= 0
        }
        rows.append(
            {
                "ranking": name,
                "budget": int(budget),
                "annotated": int(selected.sum()),
                "true_unknown_proposals": int(strata.is_unknown[selected].sum()),
                "annotation_precision": float(strata.is_unknown[selected].mean()),
                "background_selection_rate": float(strata.is_background[selected].mean()),
                "all_objects_found": len(found_unknown),
                "all_discovery_recall": (
                    float(len(found_unknown) / len(total_unknown))
                    if total_unknown
                    else float("nan")
                ),
                "tail_objects_found": len(found_tail),
                "tail_discovery_recall": (
                    float(len(found_tail) / len(total_tail)) if total_tail else float("nan")
                ),
            }
        )
    return rows
