# Diagnostic campaign — component behaviour and experiment design

Answers the Step 10 questions using the instrumentation added in Steps 2–9.

## Provenance and what this is not

**Executed:** a PROB-calibrated pool of 198 images / 3,960 proposals — the pilot
protocol's own parameters (20 Task-2 classes, imbalance ratio 20, 20 proposals
per image, budget 10) — scored with all 19 registry strategies across seeds
0/1/2, plus a coherence-method sweep. Statistics are calibrated to a real PROB
Task-1 checkpoint sample (`PROB/results/trained_checkpoint_smoke/diagnostics.json`):
`pred_obj ∈ [69.4, 467.2]`, so `objectness = exp(-(1.3/256)·pred_obj) ∈ [0.09, 0.70]`;
per-class sigmoid scores averaging 0.022.

**Not executed:** anything on real M-OWODB images. This machine has no CUDA, no
Task-1 checkpoint, and one image in `PROB/data/OWOD/JPEGImages`. The commands to
reproduce every number below on real proposals are in §8; until they are run,
treat §§1–7 as predictions about component *behaviour* that the code will report
identically on real data, not as measurements of the real pool.

Where a finding follows from the arithmetic rather than from the simulated data
distribution, it is marked **[structural]** — those transfer to real data
directly.

---

## 1. Does posterior uncertainty differ from the unknown score? (Q1)

| method | Spearman with unknown score | Spearman with objectness | prefers background |
|---|---|---|---|
| `entropy` | −0.413 | −0.307 | **yes** |
| `margin` | −0.320 | −0.242 | **yes** |
| `one_minus_max` | −0.393 | −0.298 | **yes** |
| `objectness_weighted_entropy` | +0.341 | +0.222 | no |
| `legacy_prob_score` | **+1.000** | +0.732 | no |

Two separate conclusions.

**The legacy transform is confirmed dead. [structural]** `1 − |2c − 1|` reaches
Spearman +1.000 with the unknown score, reproducing audit S2 exactly. It carries
no ranking information of its own.

**But plain entropy trades one defect for another.** PROB's posterior is
`objectness · sigmoid(class logits)`, renormalised. A *confident* unknown
detection therefore has a peaked posterior and **low** entropy, while a
background query has a diffuse posterior and **high** entropy. Measured on the
calibrated pool: mean entropy is 0.855 for background proposals against 0.810 for
on-object proposals, and entropy anti-correlates with objectness at −0.307. A term
that ranks clutter above objects spends the annotation budget on clutter.

This is a consequence of how the posterior is constructed, so it should hold on
real exports too — and it is directly checkable, because `objectness` is in every
NPZ the bridge writes.

`objectness_weighted_entropy` (new) is the geometric mean of the *rank-normalised*
entropy and unknown score. It is the only method that satisfies both criteria at
once: not monotone in the unknown score (+0.341, so it adds information) and
positively correlated with objectness (+0.222, so it prefers objects). A first
implementation combined the *raw* values and scored +0.9997 with the unknown
score — reproducing S2 — because the score spans orders of magnitude while
entropy sits in a narrow band. Combining on ranks fixes it.

**`entropy` remains the configured default, per Decision 1.** The evidence above
says it should probably not be, and that is the first item needing sign-off (§7).

## 2. Is rarity continuous, or a singleton indicator? (Q2)

| | fraction below 0.1 | distinct values |
|---|---|---|
| v1 (`count**-1` + min-max) | **82.9 %** | — |
| v2 (log inverse frequency + rank) | **9.5 %** | 19.3 |

Audit S4 is resolved. **[structural]** Rank normalisation is invariant to any
strictly monotone transform of the raw component, so the concentration was a
property of the transform, not of the ordering; `log_inverse_frequency`,
`inverse_frequency` and `negative_count` are provably identical after ranking
(`test_all_rarity_methods_agree_under_rank_normalisation`).

## 3. What regime is coherence in? (Q3, and the S5 sweep)

| coherence method | regime | Spearman with cluster size | `full` vs `full_no_coherence` Jaccard |
|---|---|---|---|
| `density` (v1) | frequency-confounded / saturated | **−0.435** | 0.818 |
| `relative_within_cluster` (v2 default) | **saturated** | **−0.034** | **0.674** |
| `neighbour_consistency` | frequency-confounded / saturated | −0.486 | **1.000** |

**The chosen default is empirically the right one.** `relative_within_cluster` is
the only variant that removes the frequency confound (−0.435 → −0.034) *and* it
produces the largest change in what gets selected. `neighbour_consistency` is
worst on both counts: still confounded, and Jaccard 1.000 means the gate changes
nothing at all.

**But it lands in a saturated regime.** Spread is 0.042 on a [0,1] scale, so
coherence is nearly constant across the pool and the gate has little to act on.
`Spearman(rarity, gated) = 0.982`: the gate barely reorders proposals.

A sharper statement of S5 than the audit had, from
`test_legacy_density_coherence_collapses_below_the_neighbour_count`: the legacy
absolute-density measure does not degrade gradually with class size, it
**collapses at the neighbour count**. A pseudo-class with fewer than `k = 5`
members necessarily has its k-th nearest neighbour in another cluster, so `d_k`
jumps to the inter-cluster scale. Measured tail/head coherence ratio: 0.14 at tail
size 2–3, 0.83 at 6, 0.93 at 60. **[structural]** In this pool no cluster falls
below the threshold (`clusters_below_neighbour_count` = 0.0), which is why v1's
confound is a moderate −0.435 rather than catastrophic. On a real pool with
sparser clusters it would be far worse — and the diagnostics now report that
fraction so the regime is visible before any conclusion is drawn.

## 4. Does the gate materially change the ranking? (Q4)

`full` versus `full_no_coherence`: Jaccard 0.674, **20 % of selected images
differ** (2 of 10). At proposal level `Spearman(rarity, gated) = 0.982`.

Those two facts only look contradictory until you measure the image ranking:

## 5. The image-aggregation pathology, and which method to use (Q5b, Step 7)

| aggregation | boundary gap / score range | signal (strategy) | noise (seed) | S/N | usable |
|---|---|---|---|---|---|
| `top_k_mean` (k=3) | 0.0089 | 0.326 | 0.222 | **1.47** | yes |
| `max` | 0.0065 | 0.232 | 0.222 | 1.05 | yes |
| `mean` | 0.0061 | 0.232 | 0.567 | 0.41 | no |
| `noisy_or` | **0.00004** | 0.172 | 0.333 | 0.52 | no |

**The selection boundary is razor-thin. [structural]** After rank normalisation
the top image scores saturate near 1.0: on this pool the gap between the 10th and
11th ranked image is 0.9 % of the full score range under `top_k_mean`, and
0.004 % under `noisy_or`. That is why a 0.982 proposal-level correlation can still
flip 20 % of the selected set — and it means image-level selection is
hypersensitive to negligible score differences.

`top_k_mean` has the best signal-to-noise and is kept as the default; `mean` and
`noisy_or` are unusable at this budget (their seed noise exceeds the strategy
effect). This answers Step 7 with evidence rather than preference.

## 6. Effect size, seed noise, and a confound I had introduced (Q5, Q6)

Mean pairwise Jaccard across all 19 strategies is 0.142, so the strategies do
select genuinely different images. `full` versus `random` is 0.018.

The number that matters for the campaign design:

* coherence-gate effect on selection: **0.326**
* `v2:full`'s own seed-to-seed variation: **0.222** (self-Jaccard 0.778, min 0.667)
* **signal / noise = 1.47**

At budget 10 the gate's effect is only about 1.5× the KMeans seed noise. That is
the quantitative form of audit S1, and it is a selection-level bound: whatever
survives into `known_mAP` or `U_Recall` will be smaller still.

### A confound in my own seed derivation, found and fixed

`derive_seed(seed, round, strategy, version)` gave each strategy its own KMeans
partition of the *same* pool. Two strategies scoring identical proposals agreed on
only 88.2 % of pairwise co-memberships, and of an apparent 0.462 selection
difference between `full` and `full_no_coherence`, **0.280 came from the differing
partition rather than from the gate** — 60 % of the measured effect was an
artifact. Pseudo-labelling partitions the pool; it is not part of a strategy's
definition. `derive_pool_seed(seed, round)` now gives every strategy on a pool one
shared clustering, enforced by
`test_all_strategies_share_one_clustering_per_pool`. The offline harness already
shared the seed across strategies, so the numbers in §§1–5 were measured with a
shared clustering and are unaffected.

### Two protocol levers, and why neither is a free win

`pseudo_label_source="predicted"` removes KMeans entirely: seed noise **0.000**,
gate signal **0.889**. Tempting — but the rarity it produces is *actively
misleading*. `Spearman(pseudo-cluster size, true class size)` is **−0.486** for
predicted labels versus **+0.877** for KMeans, because unknown-class objects all
collapse into PROB's single "unknown" slot (23.9 % of proposals, the largest
cluster) while background queries scatter across the 40 known-class labels. The
rarity term would rank frequent classes as rare. **Do not use it as-is.**

Raising the budget does not help either: S/N was 1.47 / 1.42 / 0.88 / 1.38 at
budgets 10 / 20 / 40 / 60. The clustering noise scales with the budget.

The remaining option, not yet implemented, is the obvious synthesis: filter to
proposals PROB predicts as *unknown*, then cluster within them. That keeps
clustering's correct rarity ordering while removing the background proposals that
drive both the noise and the entropy pathology.

## 7. What needs sign-off

1. **Default uncertainty method.** Decision 1 specified `entropy`; §1 shows every
   pure-posterior method prefers background. `objectness_weighted_entropy` is
   implemented and satisfies both criteria. Changing the default is a one-line
   config edit — but it contradicts an explicit instruction, so it is left to you.
2. **Unknown-filtered clustering** (§6, last paragraph). This is the highest-value
   remaining change and it is a genuine design decision: it changes what "rarity"
   means. Roughly half a day.
3. **Whether the coherence gate is worth keeping at all.** It is real
   (20 % of selections) but marginal (S/N 1.47) and saturated (spread 0.042). If
   §6's synthesis does not raise the effect, the honest thesis contribution may be
   the *diagnostic framework and the negative result*, not the gate.

## 8. Reproducing this on real proposals (not executed)

Requires a GPU runtime with the M-OWODB data and a PROB Task-1 checkpoint.

```bash
# 1. Export one candidate pool and its reference set from a frozen checkpoint.
cd /content/PROB
python daowod_prob_bridge.py predict \
  --image-ids  /content/protocol/candidate_ids.txt \
  --checkpoint /content/daowod_checkpoints/t1.pth \
  --output     /content/pool/candidate_proposals.npz \
  --data-root /content/data/OWOD --dataset TOWOD \
  --prev-introduced-classes 20 --current-introduced-classes 20 \
  --num-classes 81 --max-proposals-per-image 20 --device cuda

python daowod_prob_bridge.py predict \
  --image-ids  /content/protocol/base_reference_ids.txt \
  --checkpoint /content/daowod_checkpoints/t1.pth \
  --output     /content/pool/reference_proposals.npz \
  --data-root /content/data/OWOD --dataset TOWOD \
  --prev-introduced-classes 20 --current-introduced-classes 20 \
  --num-classes 81 --max-proposals-per-image 20 --device cuda

# 2. Every number in this report, on real proposals, no retraining.
daowod-run diagnose \
  --candidates /content/pool/candidate_proposals.npz \
  --references /content/pool/reference_proposals.npz \
  --class-stats /content/.../long_tail/class_stats.csv \
  --annotations /content/data/OWOD/Annotations \
  --unknown-classes truck traffic_light fire_hydrant stop_sign parking_meter \
     bench elephant bear zebra giraffe backpack umbrella handbag tie suitcase \
     microwave oven toaster sink refrigerator \
  --seeds 0 1 2 --budget 10 \
  --output /content/diagnostics

# 3. The decisive single check: does entropy prefer background on real data?
python - <<'EOF'
import numpy as np
from daowod import components
from daowod.diagnostics import spearman
d = np.load("/content/pool/candidate_proposals.npz", allow_pickle=True)
for method in ("entropy", "objectness_weighted_entropy", "legacy_prob_score"):
    v = components.compute_uncertainty(
        method=method, posterior=d["posterior"], confidence=d["confidence"])
    print(f"{method:30} r(objectness)={spearman(v, d['objectness']):+.4f}")
EOF
```

Read `offline_headline.json` first: it answers Q1–Q6 directly. `q3_coherence_regime`
tells you which regime the real pool is in, and
`clusters_below_neighbour_count` tells you whether the legacy density measure
would have collapsed on it.
