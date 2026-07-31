"""Which strategy actually selects tail-class images? Measurement only.

The components anti-correlate (uncertainty vs gated rho = -0.32 in 7/7 pool
configurations), so the composite score is a compromise. This asks whether the
compromise helps or hurts the objective Contribution A is stated in terms of.
"""

import json
import sys

import numpy as np

from daowod.groups import ClassGroups
from daowod.scoring import STRATEGY_REGISTRY, score_pool, select_images
from daowod.simulation import simulate_pool

BUDGET, SEEDS = 10, (0, 1, 2, 3, 4)
BASE = dict(
    class_count=20,
    largest_class_images=30,
    imbalance_ratio=20.0,
    proposals_per_image=20,
    on_object_fraction=0.35,
    planted_outlier_fraction=0.05,
    embedding_dimension=256,
    known_class_count=40,
    reference_images=30,
)
STRATEGIES = (
    "v2:random",
    "v2:uncertainty",
    "v2:novelty",
    "v2:rarity",
    "v2:coherence",
    "v2:rarity_coherence",
    "v2:uncertainty_rarity",
    "v2:full",
    "v2:full_no_coherence",
    "v2:full_no_uncertainty",
    "v2:proposal_formula",
)

pool = simulate_pool(**BASE, seed=0)
groups = ClassGroups.from_mapping(
    {r["class_name"]: r["group"] for r in pool.class_stats_rows()}, source="sim"
)
by_group = {g: set(groups.members(g)) for g in ("head", "medium", "tail")}
unique_images = list(dict.fromkeys(str(v) for v in pool.image_ids.tolist()))
pool_rate = {
    g: np.mean([1 if set(pool.image_classes[i]) & by_group[g] else 0 for i in unique_images])
    for g in by_group
}

rows = []
for name in STRATEGIES:
    spec = STRATEGY_REGISTRY.resolve(name)
    per_seed = {g: [] for g in by_group}
    distinct = []
    for seed in SEEDS:
        if spec.random_selection:
            rng = np.random.default_rng(seed)
            selection = [unique_images[i] for i in rng.permutation(len(unique_images))[:BUDGET]]
        else:
            result = score_pool(
                spec=spec,
                image_ids=pool.image_ids,
                embeddings=pool.embeddings,
                reference_embeddings=pool.reference_embeddings,
                confidence=pool.confidence,
                posterior=pool.posterior,
                predicted_labels=pool.predicted_labels,
                seed=seed,
                compute_all_components=True,
            )
            selection = select_images(result.image_scores, budget=BUDGET)
        for g in by_group:
            per_seed[g].append(
                np.mean([1 if set(pool.image_classes[i]) & by_group[g] else 0 for i in selection])
            )
        distinct.append(len({c for i in selection for c in pool.image_classes[i]}))
    rows.append(
        {
            "strategy": name,
            **{f"{g}_coverage": float(np.mean(per_seed[g])) for g in by_group},
            **{f"{g}_sd": float(np.std(per_seed[g], ddof=1)) for g in by_group},
            "tail_lift_over_pool": float(np.mean(per_seed["tail"]) / max(pool_rate["tail"], 1e-9)),
            "distinct_classes_covered": float(np.mean(distinct)),
        }
    )

print("=" * 94)
print("Tail-class coverage of the selected images (budget 10, mean over 5 clustering seeds)")
print("=" * 94)
print(
    f"pool base rates: head {pool_rate['head']:.2f}  medium {pool_rate['medium']:.2f}  "
    f"tail {pool_rate['tail']:.2f}  ({len(by_group['tail'])} tail classes of 20)"
)
print()
print(f"{'strategy':28} {'tail':>7} {'sd':>6} {'lift':>6} {'medium':>8} {'head':>7} {'classes':>8}")
for r in sorted(rows, key=lambda r: -r["tail_coverage"]):
    print(
        f"{r['strategy']:28} {r['tail_coverage']:>7.2f} {r['tail_sd']:>6.2f} "
        f"{r['tail_lift_over_pool']:>6.2f} {r['medium_coverage']:>8.2f} "
        f"{r['head_coverage']:>7.2f} {r['distinct_classes_covered']:>8.1f}"
    )
best = max(rows, key=lambda r: r["tail_coverage"])
full = next(r for r in rows if r["strategy"] == "v2:full")
print()
print(f"best tail coverage : {best['strategy']} at {best['tail_coverage']:.2f}")
print(f"v2:full            : {full['tail_coverage']:.2f}")
if best["strategy"] != "v2:full":
    print(
        f"=> the composite score gives up {best['tail_coverage'] - full['tail_coverage']:+.2f} of tail"
    )
    print("   coverage relative to its own best single component.")
json.dump(rows, open(sys.argv[1], "w"), indent=2)
