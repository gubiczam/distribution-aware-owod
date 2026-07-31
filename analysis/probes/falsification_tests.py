"""Two falsification tests for the coherence gate. Measurement only."""

import json
import sys

import numpy as np

from daowod.diagnostics import jaccard
from daowod.groups import ClassGroups
from daowod.normalisation import normalise
from daowod.scoring import (
    STRATEGY_REGISTRY,
    StrategySpec,
    aggregate_image_scores,
    combine_components,
    score_pool,
    select_images,
)
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


def combine(result, *, gated_values=None, rarity=0.0, gated=0.0):
    spec = StrategySpec(
        name="probe",
        uncertainty_weight=0.3,
        novelty_weight=0.2,
        rarity_weight=rarity,
        gated_weight=gated,
    )
    values = dict(result.normalised)
    if gated_values is not None:
        values["gated"] = gated_values
    return combine_components(spec, values)


def pick(result, scores):
    return select_images(
        aggregate_image_scores(result.image_ids, scores, method="top_k_mean", top_k=3),
        budget=BUDGET,
    )


pool = simulate_pool(**BASE, seed=0)
groups = ClassGroups.from_mapping(
    {r["class_name"]: r["group"] for r in pool.class_stats_rows()}, source="sim"
)
tail = {n for n, g in groups.as_dict().items() if g == "tail"}
medium = {n for n, g in groups.as_dict().items() if g == "medium"}

print("=" * 88)
print("TEST 1  Is the gate's effect distinguishable from an arbitrary perturbation?")
print("=" * 88)
print("Coherence is replaced by a random permutation of itself: identical marginal")
print("distribution, all relationship to the embedding structure destroyed. If the")
print("shuffled gate moves selection as much as the real gate, the gate contributes")
print("no structural information beyond perturbing a tightly packed ranking.")
print()
real, shuffled, ungated_sel = {}, {}, {}
rows = []
for seed in SEEDS:
    r = score_pool(
        spec=STRATEGY_REGISTRY.resolve("v2:full"),
        image_ids=pool.image_ids,
        embeddings=pool.embeddings,
        reference_embeddings=pool.reference_embeddings,
        confidence=pool.confidence,
        posterior=pool.posterior,
        predicted_labels=pool.predicted_labels,
        seed=seed,
        compute_all_components=True,
    )
    ungated_sel[seed] = pick(r, combine(r, rarity=0.5))
    real[seed] = pick(r, combine(r, gated=0.5))
    per_seed = []
    for trial in range(20):
        rng = np.random.default_rng(1000 * seed + trial)
        fake_coh = r.raw["coherence"][rng.permutation(r.raw["coherence"].size)]
        fake_gated = normalise(r.normalised["rarity"] * np.power(fake_coh, 1.0), "rank")
        per_seed.append(
            1.0
            - jaccard(pick(r, combine(r, gated_values=fake_gated, gated=0.5)), ungated_sel[seed])
        )
    rows.append(
        {
            "seed": seed,
            "real_gate_effect": 1.0 - jaccard(real[seed], ungated_sel[seed]),
            "shuffled_gate_effect_mean": float(np.mean(per_seed)),
            "shuffled_gate_effect_sd": float(np.std(per_seed, ddof=1)),
            "shuffled_trials": len(per_seed),
        }
    )
print(f"{'seed':>5} {'real gate':>10} {'shuffled mean':>14} {'shuffled sd':>12}")
for row in rows:
    print(
        f"{row['seed']:>5} {row['real_gate_effect']:>10.3f} "
        f"{row['shuffled_gate_effect_mean']:>14.3f} {row['shuffled_gate_effect_sd']:>12.3f}"
    )
real_mean = float(np.mean([r["real_gate_effect"] for r in rows]))
fake_mean = float(np.mean([r["shuffled_gate_effect_mean"] for r in rows]))
fake_sd = float(np.mean([r["shuffled_gate_effect_sd"] for r in rows]))
z = (real_mean - fake_mean) / max(fake_sd, 1e-12)
print(f"\nreal gate effect       = {real_mean:.4f}")
print(f"shuffled gate effect   = {fake_mean:.4f} +/- {fake_sd:.4f}")
print(f"z of real vs shuffled  = {z:+.2f}")
print(
    f"VERDICT: the real gate is {'INDISTINGUISHABLE from' if abs(z) < 2 else 'distinguishable from'} an arbitrary perturbation"
)

print()
print("=" * 88)
print("TEST 2  Does the gate move selection toward tail classes, as hypothesised?")
print("=" * 88)


def coverage(selection):
    t = sum(1 for i in selection if set(pool.image_classes[i]) & tail)
    m = sum(1 for i in selection if set(pool.image_classes[i]) & medium)
    return t / len(selection), m / len(selection)


print(
    f"{'seed':>5} {'ungated tail':>13} {'gated tail':>11} {'delta':>8} "
    f"{'ungated med':>12} {'gated med':>10}"
)
deltas = []
for seed in SEEDS:
    ut, um = coverage(ungated_sel[seed])
    gt, gm = coverage(real[seed])
    deltas.append(gt - ut)
    print(f"{seed:>5} {ut:>13.2f} {gt:>11.2f} {gt - ut:>+8.2f} {um:>12.2f} {gm:>10.2f}")
print(
    f"\nmean change in tail coverage from adding the gate = {np.mean(deltas):+.3f} "
    f"(sd {np.std(deltas, ddof=1):.3f})"
)
print(f"tail classes in pool: {len(tail)} of {len(groups.as_dict())}")
print("The gate exists to promote rare-but-coherent proposals. A change in tail")
print("coverage that is zero or negative falsifies its stated purpose.")
json.dump(
    {
        "test1": rows,
        "test1_summary": {"real": real_mean, "shuffled": fake_mean, "shuffled_sd": fake_sd, "z": z},
        "test2_tail_delta_mean": float(np.mean(deltas)),
        "test2_tail_delta_sd": float(np.std(deltas, ddof=1)),
    },
    open(sys.argv[1], "w"),
    indent=2,
)
