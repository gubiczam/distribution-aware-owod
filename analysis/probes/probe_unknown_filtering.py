"""Diagnostic probe: what would unknown-filtered clustering change?

Measurement only. Nothing here is wired into the library; it computes the
statistics that would decide whether the change is worth making.
"""

import json
import sys
from collections import Counter

import numpy as np

from daowod import components as comp
from daowod.diagnostics import spearman
from daowod.normalisation import normalise
from daowod.simulation import simulate_pool

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
UNKNOWN_INDEX = 40  # known_class_count, i.e. PROB's unknown slot
SEEDS = (0, 1, 2, 3, 4)
pool = simulate_pool(**BASE, seed=0)
true_counts = Counter(pool.true_proposal_class.tolist())
true_size = np.array([true_counts[c] for c in pool.true_proposal_class.tolist()], float)
unknown_mask = pool.predicted_labels == UNKNOWN_INDEX

print("=" * 88)
print("PROBE  Unknown-filtered clustering: what would change, quantitatively?")
print("=" * 88)
print(f"proposals                                    {pool.embeddings.shape[0]}")
print(
    f"predicted as unknown (the candidate filter)  {int(unknown_mask.sum())} "
    f"({unknown_mask.mean():.1%})"
)
print(f"of those, actually on an object               {pool.is_on_object[unknown_mask].mean():.1%}")
print(
    f"of the rest, actually on an object            {pool.is_on_object[~unknown_mask].mean():.1%}"
)
print()

rows = []
for label, mask in (
    ("all proposals (current)", np.ones_like(unknown_mask)),
    ("unknown-predicted only", unknown_mask),
):
    fidelity, noise_pairs, spreads = [], [], []
    partitions = {}
    for seed in SEEDS:
        subset = pool.embeddings[mask]
        labels = comp.assign_pseudo_labels(subset, source="cluster", cluster_count=20, seed=seed)
        sizes = comp.cluster_sizes(labels)
        rarity = normalise(comp.compute_rarity(labels, method="log_inverse_frequency"), "rank")
        fidelity.append(spearman(sizes.astype(float), true_size[mask]))
        spreads.append(float(rarity.std()))
        partitions[seed] = (labels, rarity)
    # clustering instability: how much the rarity RANKING moves between seeds
    for i, a in enumerate(SEEDS):
        for b in SEEDS[i + 1 :]:
            noise_pairs.append(spearman(partitions[a][1], partitions[b][1]))
    rows.append(
        {
            "pool": label,
            "proposals": int(mask.sum()),
            "rarity_fidelity_to_true_class_size": float(np.mean(fidelity)),
            "rarity_spread": float(np.mean(spreads)),
            "rarity_rank_stability_across_seeds": float(np.mean(noise_pairs)),
        }
    )

print(f"{'pool':26} {'n':>6} {'rarity fidelity':>16} {'spread':>8} {'rank stability':>15}")
for r in rows:
    print(
        f"{r['pool']:26} {r['proposals']:>6} "
        f"{r['rarity_fidelity_to_true_class_size']:>16.4f} {r['rarity_spread']:>8.4f} "
        f"{r['rarity_rank_stability_across_seeds']:>15.4f}"
    )
print()
print("'rarity fidelity' is Spearman(pseudo-cluster size, TRUE class size), computed")
print("post hoc. Negative means the rarity term ranks frequent classes as rare.")
print("'rank stability' is Spearman between the rarity rankings two clustering seeds")
print("produce on the identical pool: 1.0 would mean the acquisition is deterministic.")
print()
delta_fid = (
    rows[1]["rarity_fidelity_to_true_class_size"] - rows[0]["rarity_fidelity_to_true_class_size"]
)
delta_stab = (
    rows[1]["rarity_rank_stability_across_seeds"] - rows[0]["rarity_rank_stability_across_seeds"]
)
print(f"change in rarity fidelity  : {delta_fid:+.4f}")
print(f"change in rank stability   : {delta_stab:+.4f}")
print()
print("=" * 88)
print("How much of the pool would the filter discard, and does it keep the tail?")
print("=" * 88)
kept = Counter()
total = Counter()
for cls, keep in zip(pool.true_proposal_class.tolist(), unknown_mask.tolist(), strict=True):
    total[cls] += 1
    if keep:
        kept[cls] += 1
task_classes = [c for c in total if c.startswith("task_class_")]
by_size = sorted(task_classes, key=lambda c: -total[c])
print(f"{'true class':18} {'total':>7} {'kept':>6} {'kept %':>8}")
for cls in by_size[:3] + ["..."] + by_size[-3:]:
    if cls == "...":
        print(f"{'...':18}")
        continue
    print(f"{cls:18} {total[cls]:>7} {kept[cls]:>6} {kept[cls] / total[cls]:>7.1%}")
for cls in ("background", "outlier"):
    if cls in total:
        print(f"{cls:18} {total[cls]:>7} {kept[cls]:>6} {kept[cls] / total[cls]:>7.1%}")
json.dump(
    {
        "rows": rows,
        "delta_fidelity": delta_fid,
        "delta_stability": delta_stab,
        "unknown_fraction": float(unknown_mask.mean()),
        "on_object_within_unknown": float(pool.is_on_object[unknown_mask].mean()),
        "on_object_outside_unknown": float(pool.is_on_object[~unknown_mask].mean()),
    },
    open(sys.argv[1], "w"),
    indent=2,
)
