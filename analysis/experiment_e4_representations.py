"""Representation Experiment E4, Phases 1-4 and 6: compare feature spaces.

    python analysis/experiment_e4_representations.py \\
        --export outputs/real_stage1/reference_proposals.npz \\
        --annotations ~/owod_stage/Annotations \\
        --representations outputs/e4_representations \\
        --output outputs/e4_geometry

Runs on the *evaluation* candidate pool — the same 48 000 proposals the frozen
experiments used — and for each available feature space measures the geometry
statistics the coherence gate depends on, then draws the projections.

This script does not touch the baseline and does not run any acquisition strategy;
Phase 5 lives in ``analysis/run_e4_active_learning.py`` so that the cheap geometry
comparison and the expensive campaign comparison can be run and re-run
independently.

It reads ground truth throughout. Like ``analysis/audit_coherence_failure.py`` it is
an analysis of a finished experiment, never part of an acquisition path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from daowod import (
    audit,
    candidates,
    components,
    detector,
    geometry,
    representations,
    study,
    tables,
)
from daowod.config import resolve_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--representations", default="outputs/e4_representations")
    parser.add_argument("--output", default="outputs/e4_geometry")
    parser.add_argument("--mode", default="MAINREVEALED")
    parser.add_argument("--neighbours", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only",
        action="append",
        help="Restrict to these representation names; repeatable.",
    )
    parser.add_argument(
        "--skip-projections",
        action="store_true",
        help="Skip the PCA/t-SNE figures, which are the slow part.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mode = resolve_mode(args.mode)
    directory = Path(args.representations)

    export = study.load_export(args.export)
    image_ids = np.asarray([str(value) for value in export["image_ids"].tolist()], dtype=object)
    available_images = sorted(set(image_ids.tolist()))
    splits = detector.split_disjoint(
        available_images,
        counts={
            "reference": mode.reference_images,
            "pilot": mode.pilot_images,
            "evaluation": mode.evaluation_images,
        },
        seed=args.seed,
    )
    config = mode.study_config()
    prepared = study.prepare_pool(
        export=export,
        annotations_dir=args.annotations,
        config=config,
        restrict_to_images=list(splits["evaluation"]),
    )
    strata = audit.Strata.from_oracle(
        prepared.table.gt_match_kind, prepared.table.gt_group, prepared.table.gt_class
    )
    print(f"evaluation pool: {prepared.size} proposals; strata {strata.counts()}", flush=True)

    # The pool's rows *within the export*, so a representation can be sliced to
    # exactly the proposals the acquisition would see.
    pool_rows = _pool_rows(
        export, image_ids=image_ids, wanted=splits["evaluation"], spec=config.candidate_spec
    )
    if pool_rows.size != prepared.size:
        raise RuntimeError(
            f"pool row reconstruction gave {pool_rows.size} rows for a "
            f"{prepared.size}-proposal pool; the two must agree."
        )

    # --- representation audit (Phase 1 as a table) -----------------------------
    audit_rows = representations.audit_rows(export=export, directory=directory)
    tables.write_csv(output / "representation_audit.csv", audit_rows)
    ready = [row["name"] for row in audit_rows if row["available"]]
    if args.only:
        wanted = set(args.only)
        ready = [name for name in ready if name in wanted]
    ready = representations.sequence(ready)
    print(f"comparing {len(ready)} representations: {ready}", flush=True)
    for row in audit_rows:
        if not row["available"]:
            print(
                f"  unavailable: {row['name']} — {row.get('blocked_by', 'unknown reason')}",
                flush=True,
            )

    coverage = _coverage(directory, ready, pool_rows)
    tables.write_csv(output / "representation_coverage.csv", coverage)
    for row in coverage:
        if not row["fully_covered"]:
            raise RuntimeError(
                f"{row['representation']}: {row['missing_rows']} pool rows have no "
                "embedding. Re-run the extractor before analysing this space."
            )

    # --- Phase 3: geometry ----------------------------------------------------
    purity: list[dict[str, object]] = []
    density: list[dict[str, object]] = []
    precision: list[dict[str, object]] = []
    mutual: list[dict[str, object]] = []
    clusters: list[dict[str, object]] = []
    compactness: list[dict[str, object]] = []
    overlap: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    sibling: list[dict[str, object]] = []
    pseudo: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    embeddings_by_name: dict[str, np.ndarray] = {}

    for name in ready:
        started = time.time()
        matrix, manifest = representations.load(
            name, export=export, directory=directory, rows=pool_rows, seed=args.seed
        )
        embeddings_by_name[name] = matrix
        manifests.append(manifest)
        bundle = geometry.evaluate_representation(
            matrix, strata, representation=name, neighbours=args.neighbours, seed=args.seed
        )
        purity.append(bundle["purity"])
        density.extend(bundle["density"])
        precision.append(bundle["nn_precision"])
        mutual.append(bundle["mutual"])
        clusters.extend(bundle["clusters"])
        compactness.append(bundle["compactness"])
        overlap.append(bundle["overlap"])
        component_rows.extend(bundle["components"])
        sibling.append(bundle["sibling_rank"])
        pseudo.append(bundle["pseudo_classes"])
        advantage = float(bundle["purity"]["tail_purity_advantage"])
        print(
            f"  {name:26s} dim={matrix.shape[1]:5d} "
            f"tail_same_label={float(bundle['purity']['tail_same_label']):.4f} "
            f"background={float(bundle['purity']['background_same_label']):.4f} "
            f"advantage={advantage:.3f} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

    tables.write_csv(output / "neighbourhood_purity.csv", purity)
    tables.write_csv(output / "local_density.csv", density)
    tables.write_csv(output / "nearest_neighbour_precision.csv", precision)
    tables.write_csv(output / "mutual_neighbour_consistency.csv", mutual)
    tables.write_csv(output / "cluster_quality.csv", clusters)
    tables.write_csv(output / "compactness_separation.csv", compactness)
    tables.write_csv(output / "background_tail_overlap.csv", overlap)
    tables.write_csv(output / "component_distributions.csv", component_rows)
    tables.write_csv(output / "same_class_sibling_rank.csv", sibling)
    tables.write_csv(output / "pseudo_class_alignment.csv", pseudo)
    headline = geometry.headline_table(purity)
    tables.write_csv(output / "headline_tail_purity.csv", headline)

    # --- Phase 4: figures -----------------------------------------------------
    figures: list[str] = []
    figures += [str(path) for path in figures.purity_bars(headline, output)]
    figures += [str(path) for path in figures.purity_panel(purity, output)]
    figures += [str(path) for path in figures.geometry_scatter(clusters, compactness, output)]
    figures += [str(path) for path in figures.density_figure(density, output)]

    projection_manifests: list[dict[str, object]] = []
    if not args.skip_projections:
        objectness = prepared.pool.objectness
        for name in ready:
            matrix = embeddings_by_name[name]
            pseudo_labels = components.assign_pseudo_labels(
                matrix, cluster_count=20, seed=args.seed
            )
            coherence = components.compute_coherence(
                matrix,
                method="relative_within_cluster",
                pseudo_labels=pseudo_labels,
                neighbour_count=3,
            ).coherence
            extra = {"coherence": coherence, "objectness": objectness}
            for method in ("pca", "tsne"):
                for scheme in ("balanced", "natural"):
                    written, manifest = figures.projection_figure(
                        matrix,
                        strata,
                        output,
                        representation=name,
                        method=method,
                        scheme=scheme,
                        extra=extra,
                        seed=args.seed,
                    )
                    figures += [str(path) for path in written]
                    projection_manifests.append(manifest)
                    print(f"  figure: {name} {method} {scheme}", flush=True)

    # --- Phase 6 inputs: the comparison against the baseline space -------------
    baseline = representations.BASELINE_REPRESENTATION
    deltas = _deltas(purity, clusters, compactness, overlap, baseline=baseline)
    tables.write_csv(output / "geometry_vs_baseline.csv", deltas)

    manifest = {
        "export": args.export,
        "mode": mode.name,
        "seed": args.seed,
        "neighbours": args.neighbours,
        "pool_size": int(prepared.size),
        "strata": strata.counts(),
        "reachable_objects": {
            group: prepared.targets.object_total(group)
            for group in ("all", "head", "medium", "tail")
        },
        "representations": manifests,
        "unavailable": [
            {"name": row["name"], "blocked_by": row.get("blocked_by", "")}
            for row in audit_rows
            if not row["available"]
        ],
        "projection_note": figures.unused_grid_note(),
        "projections": projection_manifests,
        "figures": figures,
    }
    tables.write_json(output / "e4_geometry_manifest.json", manifest)
    (output / "e4_geometry_summary.md").write_text(
        _summary(
            manifest, headline, purity, clusters, compactness, overlap, deltas, pseudo, sibling
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {output}")
    print(json.dumps(headline, indent=2))


def _pool_rows(export, *, image_ids, wanted, spec) -> np.ndarray:
    keep = np.array([str(value) in set(wanted) for value in image_ids.tolist()], dtype=np.bool_)
    subset = np.flatnonzero(keep)
    selection = candidates.build_candidate_pool(
        image_ids=image_ids[subset],
        boxes_cxcywh=np.asarray(export["boxes"])[subset],
        objectness=np.asarray(export["objectness"])[subset],
        unknown_score=np.asarray(export["confidence"])[subset],
        posterior=np.asarray(export["posterior"])[subset],
        predicted_labels=np.asarray(export.get("predicted_labels"))[subset]
        if "predicted_labels" in export
        else None,
        spec=spec,
    )
    return subset[selection.indices]


def _coverage(directory: Path, names, pool_rows: np.ndarray) -> list[dict[str, object]]:
    """Verify that every pool row was actually extracted, for each crop space."""

    rows: list[dict[str, object]] = []
    for name in names:
        spec = representations.resolve(name)
        base = spec.name if spec.kind == "crop" else spec.base
        filled_path = directory / f"{base}_filled.npy"
        if spec.kind != "crop" and not (base and (directory / f"{base}_filled.npy").exists()):
            rows.append(
                {
                    "representation": name,
                    "kind": spec.kind,
                    "checked": False,
                    "missing_rows": 0,
                    "fully_covered": True,
                    "note": "derived from the export; no extraction to verify",
                }
            )
            continue
        if not filled_path.exists():
            rows.append(
                {
                    "representation": name,
                    "kind": spec.kind,
                    "checked": False,
                    "missing_rows": 0,
                    "fully_covered": True,
                    "note": "no coverage mask written by the extractor",
                }
            )
            continue
        filled = np.load(filled_path)
        missing = int((~filled[pool_rows]).sum())
        rows.append(
            {
                "representation": name,
                "kind": spec.kind,
                "checked": True,
                "pool_rows": int(pool_rows.size),
                "missing_rows": missing,
                "fully_covered": missing == 0,
                "note": "",
            }
        )
    return rows


def _deltas(purity, clusters, compactness, overlap, *, baseline: str) -> list[dict[str, object]]:
    """Each space against the baseline space, on the statistics that decide E4."""

    def find(rows, name, **match):
        for row in rows:
            if str(row.get("representation")) == name and all(
                str(row.get(key)) == str(value) for key, value in match.items()
            ):
                return row
        return None

    base_purity = find(purity, baseline)
    base_cluster = find(clusters, baseline, label_set="unknown_class")
    base_compact = find(compactness, baseline)
    base_overlap = find(overlap, baseline)
    rows: list[dict[str, object]] = []
    for row in purity:
        name = str(row["representation"])
        cluster = find(clusters, name, label_set="unknown_class")
        compact = find(compactness, name)
        over = find(overlap, name)
        entry: dict[str, object] = {
            "representation": name,
            "is_baseline": name == baseline,
            "tail_purity_advantage": float(row["tail_purity_advantage"]),
            "tail_purity_advantage_baseline": (
                float(base_purity["tail_purity_advantage"]) if base_purity else float("nan")
            ),
            "tail_same_label": float(row["tail_same_label"]),
            "background_same_label": float(row["background_same_label"]),
            "coherence_premise_holds": bool(row["coherence_premise_holds"]),
        }
        entry["tail_purity_advantage_ratio_to_baseline"] = (
            float(entry["tail_purity_advantage"] / entry["tail_purity_advantage_baseline"])
            if base_purity and float(base_purity["tail_purity_advantage"]) > 0
            else float("nan")
        )
        if cluster and base_cluster:
            entry["unknown_class_silhouette"] = float(cluster["silhouette"])
            entry["unknown_class_silhouette_delta"] = float(
                cluster["silhouette"] - base_cluster["silhouette"]
            )
            entry["davies_bouldin"] = float(cluster["davies_bouldin"])
        if compact and compact.get("available") and base_compact and base_compact.get("available"):
            entry["compactness_ratio"] = float(compact["compactness_ratio"])
            entry["compactness_ratio_delta"] = float(
                compact["compactness_ratio"] - base_compact["compactness_ratio"]
            )
        if over and over.get("available") and base_overlap and base_overlap.get("available"):
            entry["nearest_tail_auc"] = float(over["nearest_tail_auc"])
            entry["nearest_tail_auc_delta"] = float(
                over["nearest_tail_auc"] - base_overlap["nearest_tail_auc"]
            )
            entry["centroid_auc"] = float(over["centroid_auc"])
        rows.append(entry)
    return rows


def _table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _fmt(value, digits=4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(number) else f"{number:.{digits}f}"


def _summary(
    manifest, headline, purity, clusters, compactness, overlap, deltas, pseudo, sibling
) -> str:
    lines = ["# E4 geometry comparison", ""]
    lines.append(
        f"Evaluation pool {manifest['pool_size']} proposals; strata {manifest['strata']}; "
        f"{manifest['reachable_objects']['all']} reachable unknown objects "
        f"({manifest['reachable_objects']['tail']} tail)."
    )
    lines.append("")
    lines.append("## The decisive statistic")
    lines.append("")
    lines.append(
        "`tail purity advantage` = tail same-label neighbour fraction / background "
        "same-label neighbour fraction. It must exceed **1.0** for a density-based "
        "coherence term to be able to prefer rare objects over background at all."
    )
    lines.append("")
    lines.append(
        "Two versions. The *raw* ratio is what the earlier audit quoted. The "
        "*normalised* one divides each stratum's purity by the maximum its class "
        "sizes allow — the tail group's ceiling at k=10 is only 0.235, because three "
        "tail classes have a single proposal, so the raw ratio mixes geometry with "
        "the frequencies that define the tail. The **head** column needs no such "
        "caveat: those classes have 12-65 members and a ceiling of exactly 1.0."
    )
    lines.append("")
    lines.append(
        _table(
            [
                "representation",
                "tail same-label",
                "tail ceiling",
                "tail normalised",
                "head same-label (ceiling 1.0)",
                "background same-label",
                "raw advantage",
                "normalised advantage",
                "premise holds",
            ],
            [
                [
                    row["representation"],
                    _fmt(row["tail_same_label"]),
                    _fmt(row["tail_purity_ceiling"], 3),
                    _fmt(row["tail_same_label_normalised"], 3),
                    _fmt(row["head_same_label"]),
                    _fmt(row["background_same_label"]),
                    _fmt(row["tail_purity_advantage"], 3),
                    _fmt(row["tail_purity_advantage_normalised"], 3),
                    "**yes**" if row["coherence_premise_holds"] else "no",
                ]
                for row in headline
            ],
        )
    )
    lines.append("")
    lines.append("## Rank of the nearest same-class sibling (no frequency ceiling)")
    lines.append("")
    lines.append(
        "For each unknown region whose class has at least two members, where does its "
        "closest sibling sit in the full similarity ordering over 48 000 proposals? A "
        "space that clusters the class answers 1-2; a space that does not answers in "
        "the thousands."
    )
    lines.append("")
    lines.append(
        _table(
            [
                "representation",
                "tail median rank",
                "head median rank",
                "unknown median rank",
                "sibling in top 10 (unknown)",
            ],
            [
                [
                    row["representation"],
                    _fmt(row.get("tail_median_sibling_rank"), 1),
                    _fmt(row.get("head_median_sibling_rank"), 1),
                    _fmt(row.get("unknown_median_sibling_rank"), 1),
                    _fmt(row.get("unknown_sibling_within_10"), 3),
                ]
                for row in sibling
            ],
        )
    )
    lines.append("")
    lines.append("## Cluster geometry of the unknown classes")
    lines.append("")
    lines.append(
        _table(
            [
                "representation",
                "silhouette",
                "Davies-Bouldin",
                "within/between distance",
                "structure",
            ],
            [
                [
                    row["representation"],
                    _fmt(row.get("unknown_class_silhouette")),
                    _fmt(row.get("davies_bouldin"), 3),
                    _fmt(row.get("compactness_ratio"), 3),
                    "compact" if float(row.get("compactness_ratio", 2) or 2) < 1.0 else "none",
                ]
                for row in deltas
            ],
        )
    )
    lines.append("")
    lines.append("## Tail versus background separability in each space")
    lines.append("")
    lines.append(
        "`nearest tail AUC` is local (leave-one-out proximity to other tail regions); "
        "`centroid AUC` is a single global direction. Local at chance with global high "
        "means the class information exists but not locally — which is what a k-NN "
        "coherence term reads."
    )
    lines.append("")
    lines.append(
        _table(
            ["representation", "nearest tail AUC (local)", "centroid AUC (global)"],
            [
                [
                    row["representation"],
                    _fmt(row.get("nearest_tail_auc"), 3),
                    _fmt(row.get("centroid_auc"), 3),
                ]
                for row in deltas
            ],
        )
    )
    lines.append("")
    lines.append("## Does clustering each space recover the unknown classes?")
    lines.append("")
    lines.append(
        _table(
            ["representation", "ARI (unknown classes)", "NMI", "rarity vs true rarity (Spearman)"],
            [
                [
                    row["representation"],
                    _fmt(row.get("ari_unknown_classes")),
                    _fmt(row.get("nmi_unknown_classes")),
                    _fmt(row.get("rarity_vs_true_rarity_spearman"), 3),
                ]
                for row in pseudo
            ],
        )
    )
    lines.append("")
    if manifest["unavailable"]:
        lines.append("## Representations that could not be built")
        lines.append("")
        for row in manifest["unavailable"]:
            lines.append(f"- `{row['name']}`: {row['blocked_by']}")
        lines.append("")
    lines.append(manifest["projection_note"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
