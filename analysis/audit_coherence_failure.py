"""Localise the coherence gate's failure to a component, from real PROB proposals.

Run from the repository root:

    python analysis/audit_coherence_failure.py \\
        --export outputs/real_stage1/reference_proposals.npz \\
        --annotations ~/owod_stage/Annotations \\
        --output outputs/audit_contribution_a

Writes one CSV per diagnostic plus a markdown summary. Every number in
``docs/contribution_a_failure_analysis.md`` comes from this script, so a reader can
regenerate the analysis rather than take it on trust.

The script reads ground truth throughout: it is an analysis of a finished
experiment, not part of any acquisition path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from daowod import annotation_study as study
from daowod import audit, candidates, components, export_cache, reporting
from daowod.normalisation import normalise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, help="Proposal export NPZ from the PROB bridge.")
    parser.add_argument("--annotations", required=True, help="VOC Annotations directory.")
    parser.add_argument("--output", default="outputs/audit_contribution_a")
    parser.add_argument("--evaluation-images", type=int, default=2400)
    parser.add_argument("--pilot-images", type=int, default=600)
    parser.add_argument("--reference-images", type=int, default=1000)
    parser.add_argument("--per-image-limit", type=int, default=20)
    parser.add_argument("--cluster-count", type=int, default=20)
    parser.add_argument("--neighbours", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    export = study.load_export(args.export)
    available = sorted({str(value) for value in export["image_ids"].tolist()})
    splits = export_cache.split_disjoint(
        available,
        counts={
            "reference": args.reference_images,
            "pilot": args.pilot_images,
            "evaluation": args.evaluation_images,
        },
        seed=args.seed,
    )
    config = study.StudyConfig(
        candidate_spec=candidates.CandidatePoolSpec(per_image_limit=args.per_image_limit)
    )
    prepared = study.prepare_pool(
        export=export,
        annotations_dir=args.annotations,
        config=config,
        restrict_to_images=splits["evaluation"],
    )
    table = prepared.table
    strata = audit.Strata.from_oracle(table.gt_match_kind, table.gt_group, table.gt_class)
    print(f"pool={prepared.size} strata={strata.counts()}", flush=True)

    embeddings = prepared.pool.embeddings
    objectness = prepared.pool.objectness
    unknown_score = prepared.pool.confidence
    posterior = prepared.pool.posterior
    boxes = prepared.pool.boxes_cxcywh
    normalised_posterior = posterior / np.maximum(posterior.sum(axis=1, keepdims=True), 1e-12)
    area = boxes[:, 2] * boxes[:, 3]

    reference_mask = np.array(
        [str(value) in set(splits["reference"]) for value in export["image_ids"].tolist()],
        dtype=np.bool_,
    )
    bank = np.asarray(export["embeddings"])[reference_mask][: config.reference_limit]

    # --- the acquisition signals, exactly as the campaign computes them -------
    pseudo_labels = components.assign_pseudo_labels(
        embeddings, cluster_count=args.cluster_count, seed=args.seed
    )
    rarity = components.compute_rarity(pseudo_labels)
    coherence_relative = components.compute_coherence(
        embeddings,
        method="relative_within_cluster",
        pseudo_labels=pseudo_labels,
        neighbour_count=5,
    )
    coherence_radius = components.compute_coherence(
        embeddings,
        method="radius_core",
        pseudo_labels=pseudo_labels,
        neighbour_count=5,
        minimum_samples=4,
    )
    entropy = components.compute_uncertainty(method="entropy", posterior=posterior)
    weighted_entropy = components.compute_uncertainty(
        method="objectness_weighted_entropy", posterior=posterior, confidence=unknown_score
    )
    novelty = components.compute_novelty(embeddings, bank)

    signals = {
        "objectness": objectness,
        "unknown_score": unknown_score,
        "entropy": entropy,
        "objectness_weighted_entropy": weighted_entropy,
        "novelty": novelty,
        "rarity_pseudo_class": rarity,
        "coherence_relative_within_cluster": coherence_relative.coherence,
        "coherence_radius_core": coherence_radius.coherence,
        "gated_rarity_x_coherence": normalise(rarity, "rank") * coherence_relative.coherence,
        "sqrt_box_area": np.sqrt(area),
        "objectness_x_sqrt_area": objectness * np.sqrt(area),
        "one_minus_max_known": 1.0 - normalised_posterior[:, :-1].max(axis=1),
    }
    feature_sets = {
        "decoder_embedding_256d": embeddings,
        "posterior_20d": posterior,
        "objectness_1d": objectness,
        "box_geometry_4d": boxes,
        "embedding_plus_objectness": np.hstack([embeddings, objectness.reshape(-1, 1)]),
    }

    print("measuring signal AUCs...", flush=True)
    auc_rows = audit.signal_auc(signals, strata)
    print("measuring supervised probe ceilings...", flush=True)
    auc_rows += audit.probe_auc(feature_sets, strata, seed=args.seed)
    reporting.write_csv(output / "signal_auc.csv", auc_rows)

    gaps = [
        audit.summarise_gap(auc_rows, target=target)
        for target in ("unknown_vs_background", "tail_vs_background", "onobject_vs_background")
    ]
    reporting.write_csv(output / "representation_vs_estimator_gap.csv", gaps)

    print("measuring precision at the actual annotation budgets...", flush=True)
    precision = audit.precision_at_budget(signals, strata, budgets=(100, 500, 2000))
    precision += audit.precision_at_budget(
        signals, strata, budgets=(100, 500, 2000), positive="tail"
    )
    reporting.write_csv(output / "precision_at_budget.csv", precision)

    print("measuring neighbourhood composition...", flush=True)
    neighbourhood = audit.neighbourhood_composition(
        embeddings, strata, neighbours=args.neighbours, label="full pool"
    )
    conditioning = objectness * np.sqrt(area)
    order = np.argsort(-conditioning, kind="stable")
    for fraction in (0.5, 0.25):
        keep = np.zeros(prepared.size, dtype=np.bool_)
        keep[order[: int(prepared.size * fraction)]] = True
        neighbourhood += audit.neighbourhood_composition(
            embeddings,
            strata,
            neighbours=args.neighbours,
            subset=keep,
            label=f"top {int(fraction * 100)}% by objectness x sqrt(area)",
        )
    reporting.write_csv(output / "neighbourhood_composition.csv", neighbourhood)

    print("measuring pseudo-class quality...", flush=True)
    quality = audit.pseudo_class_quality(pseudo_labels, rarity, strata)
    reporting.write_json(output / "pseudo_class_quality.json", quality)

    print("measuring pool-filter retention...", flush=True)
    retention: list[dict[str, object]] = []
    for name, ranking in (
        ("objectness", objectness),
        ("unknown_score", unknown_score),
        ("sqrt_box_area", np.sqrt(area)),
        ("objectness_x_sqrt_area", conditioning),
    ):
        retention += audit.retention_curve(ranking, strata, table.gt_object_index, name=name)
    reporting.write_csv(output / "pool_filter_retention.csv", retention)

    print("measuring free-heuristic reference discovery...", flush=True)
    budgets = (100, 250, 500, 1000, 2000)
    reference: list[dict[str, object]] = []
    for name, ranking in (
        ("objectness", objectness),
        ("unknown_score", unknown_score),
        ("sqrt_box_area", np.sqrt(area)),
        ("objectness_x_sqrt_area", conditioning),
    ):
        reference += audit.static_ranking_discovery(
            ranking, strata, table.gt_object_index, budgets=budgets, name=name
        )
    reporting.write_csv(output / "static_ranking_discovery.csv", reference)

    print("measuring revealed-label sample complexity...", flush=True)
    complexity = audit.revealed_sample_complexity(
        embeddings, strata, draws=args.draws, seed=args.seed
    )
    reporting.write_csv(output / "revealed_sample_complexity.csv", complexity)

    manifest = {
        "export": args.export,
        "annotations": args.annotations,
        "pool_size": prepared.size,
        "strata": strata.counts(),
        "reachable_objects": {
            group: prepared.targets.object_total(group)
            for group in ("all", "head", "medium", "tail")
        },
        "reachable_classes": {
            group: prepared.targets.class_total(group)
            for group in ("all", "head", "medium", "tail")
        },
        "composition": dict(prepared.composition),
        "cluster_count": args.cluster_count,
        "neighbours": args.neighbours,
        "seed": args.seed,
        "gaps": gaps,
        "pseudo_class_quality": quality,
    }
    reporting.write_json(output / "audit_manifest.json", manifest)
    (output / "audit_summary.md").write_text(
        _summary(manifest, auc_rows, neighbourhood, complexity, reference, precision),
        encoding="utf-8",
    )
    print(json.dumps(gaps, indent=2))
    print(f"\nwrote {output}")


def _table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _summary(manifest, auc_rows, neighbourhood, complexity, reference, precision) -> str:
    lines = ["# Component audit of the coherence-gate failure", ""]
    lines.append(
        f"Pool {manifest['pool_size']} proposals; "
        f"{manifest['reachable_objects']['all']} reachable unknown objects "
        f"({manifest['reachable_objects']['tail']} in the tail group) across "
        f"{manifest['reachable_classes']['all']} classes."
    )
    lines.append("")
    lines.append("## Representation ceiling versus unsupervised estimators")
    lines.append("")
    lines.append(
        _table(
            [
                "target",
                "supervised ceiling",
                "best free signal",
                "best distribution component",
                "estimator gap",
                "representation headroom",
            ],
            [
                [
                    row["target"],
                    f"{row['supervised_ceiling']:.3f} ({row['supervised_ceiling_features']})",
                    f"{row['best_free_unsupervised']:.3f} ({row['best_free_unsupervised_signal']})",
                    f"{row['best_distribution_component']:.3f} "
                    f"({row['best_distribution_component_signal']})",
                    f"{row['estimator_gap']:.3f}",
                    f"{row['representation_headroom']:.3f}",
                ]
                for row in manifest["gaps"]
                if row.get("available")
            ],
        )
    )
    lines.append("")
    for row in manifest["gaps"]:
        if row.get("available"):
            lines.append(f"- **{row['target']}**: {row['verdict']}.")
    lines.append("")
    lines.append("## Every signal, ROC-AUC for unknown versus background")
    lines.append("")
    selected = sorted(
        (row for row in auc_rows if row["target"] == "unknown_vs_background"),
        key=lambda row: -float(row["roc_auc"]),
    )
    lines.append(
        _table(
            ["signal", "kind", "ROC-AUC"],
            [[row["signal"], row["kind"], f"{float(row['roc_auc']):.3f}"] for row in selected],
        )
    )
    lines.append("")
    lines.append("## Precision in the top 4 % — what the budget actually buys")
    lines.append("")
    lines.append(
        "ROC-AUC summarises the whole ordering; a 2 000-region budget buys the top "
        "4 %. `lift` below 1.0 means worse than random sampling."
    )
    lines.append("")
    lines.append(
        _table(
            ["signal", "precision@2000", "unknown proposals", "lift over random"],
            [
                [
                    row["signal"],
                    f"{float(row['precision']):.4f}",
                    row["positives_in_top_k"],
                    f"{float(row['lift_over_random']):.2f}x",
                ]
                for row in sorted(
                    (
                        row
                        for row in precision
                        if int(row["budget"]) == 2000 and row["positive"] == "unknown"
                    ),
                    key=lambda row: -float(row["precision"]),
                )
            ],
        )
    )
    lines.append("")
    lines.append("## The coherence premise: what a proposal's neighbours actually are")
    lines.append("")
    lines.append(
        _table(
            ["subset", "stratum", "n", "same-label fraction", "neighbour on-object fraction"],
            [
                [
                    row["subset"],
                    row["stratum"],
                    row["n"],
                    f"{float(row['same_label_fraction']):.3f}",
                    f"{float(row['neighbour_on_object_fraction']):.3f}",
                ]
                for row in neighbourhood
            ],
        )
    )
    lines.append("")
    lines.append("## Pseudo-class quality")
    lines.append("")
    quality = manifest["pseudo_class_quality"]
    lines.append(
        _table(
            ["quantity", "value"],
            [
                [key, f"{value:.4f}" if isinstance(value, float) else value]
                for key, value in sorted(quality.items())
            ],
        )
    )
    lines.append("")
    lines.append("## What a free heuristic already discovers (no rounds, no feedback)")
    lines.append("")
    lines.append(
        _table(
            ["ranking", "budget", "unknown objects", "tail objects", "annotation precision"],
            [
                [
                    row["ranking"],
                    row["budget"],
                    row["all_objects_found"],
                    row["tail_objects_found"],
                    f"{float(row['annotation_precision']):.3f}",
                ]
                for row in reference
                if row["ranking"] == "objectness_x_sqrt_area"
            ],
        )
    )
    lines.append("")
    lines.append("## Sample complexity of a label-anchored signal")
    lines.append("")
    lines.append(
        _table(
            ["revealed unknowns", "similarity AUC", "probe AUC", "probe tail AUC"],
            [
                [
                    row["revealed_unknowns"],
                    f"{float(row['similarity_auc_mean']):.3f} ± {float(row['similarity_auc_sd']):.3f}",
                    f"{float(row['probe_auc_mean']):.3f} ± {float(row['probe_auc_sd']):.3f}",
                    f"{float(row['probe_tail_auc_mean']):.3f}",
                ]
                for row in complexity
            ],
        )
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
