# Follow-up experiment: label-anchored distribution estimation and the free prior

This reports the second full run of Contribution A. The first run's negative result
and its component audit are in
[`contribution_a_failure_analysis.md`](contribution_a_failure_analysis.md); read
that first, because it is what selects the two interventions tested here.

## Design

Eleven arms, one shared matrix, one PROB export. Identical candidate pool,
identical three long-tail severities, identical three seeds, identical budget grid
(100 / 250 / 500 / 1 000 / 2 000 annotated regions), identical oracle and identical
evaluation protocol for every arm. The baseline is unchanged and still registered
under its original name.

| arm | family | `U(x)` | distribution term |
|---|---|---|---|
| `v2:random` | baseline | — | — |
| `v2:uncertainty` | baseline | posterior entropy | — |
| `v2:uncertainty_novelty` | baseline | posterior entropy | — |
| `v2:full_no_coherence` | baseline | posterior entropy | cluster rarity, ungated |
| **`v2:full`** | **baseline under test** | posterior entropy | cluster rarity × pool density |
| **`v2:objectness_area_prior`** | **free control** | objectness × box scale | — |
| `v2:prior_full` | prior+cluster | objectness × box scale | cluster rarity × pool density |
| `v2:prior_revealed_full` | prior+anchored | objectness × box scale | revealed rarity × revealed support |
| `v2:revealed_support_only` | label-anchored | posterior entropy | revealed support only |
| `v2:revealed_no_gate` | label-anchored | posterior entropy | revealed rarity, ungated |
| `v2:revealed_full` | label-anchored | posterior entropy | revealed rarity × revealed support |

Two interventions are under test, and they are separable because each appears both
alone and in combination.

**Intervention 1 — label-anchored distribution estimation** (`revealed_*`). The
score's shape, weights and gate form are untouched; only the *source* of rarity and
coherence changes. Rarity becomes the inverse frequency of the nearest class the
oracle has already confirmed; coherence becomes similarity to confirmed unknown
regions. Before any unknown is revealed both defer to the unsupervised values, so
round 1 is bit-identical to the baseline and every later difference is attributable
to the labels the campaign bought.

**Intervention 2 — informativeness prior** (`prior_*`, `objectness_area_prior`).
The `U(x)` slot carries `objectness × box scale` instead of posterior entropy. This
is the control the audit demands: it is free, it needs no oracle feedback, and it
measured ROC-AUC 0.777 for unknown-versus-background where every semantic component
sat near 0.48.

## Pre-registered predictions

Recorded before the run, from the audit's measured sample-complexity curve
(held-out ROC-AUC for unknown vs background: 5 revealed unknowns → 0.671 for the
similarity estimator, 20 → 0.694, 160 → 0.689; the linear probe → 0.710 / 0.755 /
0.814):

1. The anchored arms beat the unsupervised baseline on **unknown** discovery,
   because 0.69 ≫ 0.48.
2. The anchored arms do **not** fix tail-versus-head selectivity, because a
   realistic budget reveals 10–40 unknown regions spread over ~20 classes, i.e. one
   or two per class, and per-class frequency cannot be estimated from that.
3. The similarity-based support **plateaus below the free prior** (0.69 versus
   0.777), so `objectness_area_prior` should beat the pure anchored arms on unknown
   discovery.
4. The combination `prior_revealed_full` should be the strongest arm, but the
   distribution term's contribution on top of the prior should be small.

Predictions 1–4 are stated so the outcome can contradict them. It partly does; see
§ "Where the predictions were wrong".

## Results

Distinct unknown ground-truth objects discovered inside a 2 000-region budget
(4.2 % of the pool), mean over three seeds, for each of the three severities. Tail
objects, classes and precision are averaged over severities. Full tables are in
`arm_comparison_unknown.csv`, `arm_comparison_tail.csv`, `discovery_counts.csv` and
`budget_curves.csv` of the run directory.

| arm | unknown objects @2000 (mod / nat / sev) | mean | vs baseline | tail obj | classes | precision |
|---|---|---:|---|---:|---:|---:|
| `v2:objectness_area_prior` **[free control]** | 25.0 / 34.0 / 22.0 | **27.0** | `+++` | 1.7 | 14.0 | 0.0183 |
| `v2:prior_revealed_full` | 20.7 / 28.3 / 16.3 | **21.8** | `++~` | 1.6 | 13.1 | 0.0121 |
| `v2:prior_full` | 16.7 / 19.7 / 13.0 | **16.4** | `~~−` | 1.4 | 11.0 | 0.0092 |
| `v2:revealed_full` | 13.7 / 17.7 / 15.0 | **15.4** | `~~~` | 1.3 | 10.6 | 0.0077 |
| `v2:random` | 14.3 / 22.0 / 9.3 | **15.2** | `~+−` | 1.1 | 10.2 | 0.0077 |
| `v2:full` **[BASELINE]** | 12.0 / 17.0 / 14.0 | **14.3** | — | 0.8 | 9.1 | 0.0072 |
| `v2:full_no_coherence` | 14.3 / 15.7 / 8.7 | **12.9** | `+~−` | 1.0 | 8.7 | 0.0064 |
| `v2:revealed_no_gate` | 13.3 / 13.3 / 10.7 | **12.4** | `~~~` | 1.3 | 9.6 | 0.0062 |
| `v2:uncertainty_novelty` | 11.0 / 15.0 / 9.0 | **11.7** | `−−−` | 1.0 | 9.7 | 0.0058 |
| `v2:uncertainty` | 8.0 / 9.0 / 5.0 | **7.3** | `−−−` | 0.0 | 5.7 | 0.0037 |
| `v2:revealed_support_only` | 2.3 / 4.0 / 4.7 | **3.7** | `−−−` | 0.2 | 3.1 | 0.0018 |

`vs baseline` gives the paired sign per severity in the order moderate / natural /
severe: `+` the arm beat `v2:full` in **every** seed, `−` it lost in every seed, `~`
mixed. Paired because all eleven arms share one pool, one export and one seed set.

Headline numbers:

* the free control finds **1.88× the baseline's** unknown objects, and is the **only**
  arm that beats the baseline in every severity with a consistent sign across seeds;
* the label-anchored gate (`v2:revealed_full`, 15.4) is mixed against the baseline
  (14.3) in all three severities — a small positive trend, no reliable improvement;
* `v2:revealed_support_only` loses to the baseline in every seed of every severity;
* the baseline itself is not reliably better than random (15.2 vs 14.3, signs
  `~+−`).

Tail discovery remains unresolvable: the per-cell counts are 0–2 objects out of 26,
so no arm can be separated on it. The tail column is reported for completeness, not
as evidence.

## Interpretation

**The free prior wins; the distribution-aware term is net-harmful.** The single
arm that clearly beats everything is `objectness_area_prior`, which has no
distribution term, no clustering, no oracle feedback and no rounds. It doubles the
baseline's unknown-object discovery. Every arm that carries a distribution term
scores below the prior alone.

**The baseline is not distinguishable from random.** Averaged over severities,
`v2:random` finds 15.2 unknown objects and `v2:full` 14.3; the paired signs are
`~ + −`, i.e. random wins outright on the natural severity, the baseline wins
outright on severe, and moderate is mixed. That is consistent with §1.4 of the
failure analysis: all three of the baseline's terms select worse than random in the
top 4 % of the ranking, so their weighted sum has no reliable advantage over not
ranking at all.

**Label-anchoring improves the estimator without making the term useful.** The
cleanest way to read this is the cost the distribution term imposes on a score that
already works. Starting from the prior alone and adding a gate (mean over the three
severities):

| arm | unknown objects @ 2 000 | objects lost to the gate |
|---|---:|---:|
| `objectness_area_prior` (no gate) | 27.0 | — |
| `prior_revealed_full` (label-anchored gate) | 21.8 | **−5.2** |
| `prior_full` (unsupervised gate) | 16.4 | **−10.6** |

Label-anchoring halves the damage, which is a real and measurable improvement of the
estimator — the audit's prediction that anchored rarity carries more signal than
k-means rarity is supported. But it does not change the sign: the term still
subtracts from a score that works without it. The same ordering appears in the pure
arms, where the anchored gate (15.4) edges the unsupervised gate (14.3) and both
edge the ungated variants.

**The support term alone is actively harmful, and the reason is the §1.1 geometry.**
`revealed_support_only` finds 3.7 unknown objects against random's 15.2 — the worst
arm in the study, losing to the baseline in every seed of every severity. "Resembles a region the oracle confirmed as unknown" fails because
a confirmed unknown sits *inside the background mass* — its own ten nearest
neighbours are 74 % background — so similarity to it is largely similarity to nearby
background. The first candidate explanation, that the term re-buys objects it has
already found, was tested and **refuted**: proposals-per-discovered-object is ≈1.00
for every arm, including this one. It is not redundant, it is mis-aimed.

**The cold start is not the explanation.** Replaying the campaigns
(deterministic, same seeds) shows the bank warm from round 2 onwards:

| arm | round 1 | round 2 | round 3 | round 4 |
|---|---|---|---|---|
| `revealed_full` | 1 region / 1 class | 10 / 7 | 14 / 9 | 14 / 9 |
| `revealed_no_gate` | 6 / 4 | 8 / 6 | 9 / 7 | 11 / 9 |
| `prior_revealed_full` | 1 / 1 | 6 / 6 | 17 / 13 | 24 / 15 |

Three to four of five rounds ran with a warm, multi-class bank. The anchored term
had data; it simply does not point where discovery is.

**Why the audit over-predicted this.** The sample-complexity curve measured
*proposal-level ROC-AUC with a bank sampled uniformly from all unknowns*. Two things
break when the campaign builds its own bank: the achievable bank is 10–24 regions
rather than a well-spread sample, and — the larger error — AUC is not the
decision-relevant statistic. At the actual budget the anchored support reaches 4.7×
the base rate with a diverse 14-region bank but the campaign's own bank is
self-selected, and the composite score dilutes it with entropy (0.46×) and novelty
(0.71×). `daowod.audit.precision_at_budget` now measures the right quantity; see
§1.4 of the failure analysis.

## Where the predictions were wrong

| prediction | outcome |
|---|---|
| 1. Anchored arms beat the baseline on unknown discovery | **Falsified.** `revealed_full` 15.4 vs baseline 14.3, mixed sign in all three severities. `revealed_support_only` is far worse. |
| 2. Anchored arms do not fix tail selectivity | **Supported**, though tail counts of 0–2 cannot resolve it either way. |
| 3. Anchored support plateaus below the free prior | **Supported**, and by a much larger margin than expected (3.7 vs 27.0 objects for the pure arm). |
| 4. `prior_revealed_full` is the strongest arm | **Falsified.** The prior *alone* is strongest; adding the anchored term costs 5.2 objects. |

Predictions 1 and 4 were wrong in the same direction: I expected the distribution
term to help once its estimator worked. It does not, in this feature space, at this
budget.

## What this means for the research hypothesis

The proposal's hypothesis H-A is that a coherence-weighted, tail-aware score reaches
the same tail level for fewer oracle calls than uncertainty or diversity selection.
After two full runs on real S-OWODB Task-1 proposals:

**Weakened, and for a reason that is now specific.** The hypothesis presumes that
(i) an estimated class distribution over candidate regions is obtainable, and (ii)
local coherence separates a rare true class from an isolated false positive. Both
presumptions are false in PROB's Task-1 decoder feature space: pseudo-class rarity
correlates with true class rarity at ρ = 0.116, and a tail region's neighbourhood is
1.5 % its own class against background's 88.8 %. Anchoring the estimator on oracle
labels — the only in-scope way to obtain a better distribution estimate — measurably
improves it but does not make the term profitable.

**Not refuted, because the premise has only been tested in one space.** The gate is
a statement about feature geometry. It has been falsified in the *decoder* space of a
Task-1 PROB checkpoint; it has not been tested in a space where out-of-vocabulary
categories are known to cluster. That is experiment E4 in the failure analysis, and
it is now the decisive one.

**A finding the proposal did not anticipate, and which changes the ordering of the
work.** The dominant available signal is geometric: `objectness × box scale` finds
1.88× the baseline's unknown objects with no clustering, no feedback and no rounds,
and PROB's own unknown score selects *worse than random* at Task 1. The
distribution-aware question — which rare class to spend the next annotation on — is
downstream of a question the proposal assumed away: which regions contain an object
at all. On this pool the second question dominates the first.

## Recommendations for the next iteration of Contribution A

1. **Adopt the informativeness prior as the `U(x)` term, and make it the control in
   every future comparison.** It is free, and no semantic term measured so far beats
   it. Report every new method against it, not only against `v2:random`.
   *Caveat to check first:* box scale correlates with object size, so the prior may
   systematically miss small rare objects. Stratify discovery by object size before
   adopting it as the headline.
2. **Run E4 (feature-space retest) next.** Re-embed the same candidate boxes with
   DINOv2 or CLIP and recompute §1.1–1.2 of the failure analysis. The single decisive
   number is the tail same-label fraction: 0.015 in decoder space. If it does not
   rise above background's, the coherence gate should be retired for Task-1 OWOD and
   Contribution A refocused on the object-vs-background problem. ~1 GPU-hour.
3. **Adopt E3's protocol before running any further comparison.** Unknown discovery
   as the primary metric (364 objects, not 26); budgets as a fraction of reachable
   objects; ≥ 5 seeds; exact binomial intervals on discovery counts. Tail claims are
   currently unfalsifiable at 0–2 events per cell.
4. **Select estimators on precision at the budget, never on ROC-AUC.** This audit's
   own experiment selection was distorted by using AUC; `precision_at_budget` is now
   in `daowod.audit`.
5. **Keep the label-anchored estimator, as a component rather than a strategy.** It
   halves the damage the distribution term does and it is the only rarity estimator
   measured to carry real signal. If E4 finds a space where unknowns cluster, this is
   the estimator to test there.
6. **Do not tune α, β, γ or `p` on this pool.** With every semantic term below the
   base rate in the top 4 %, a weight sweep would be fitting noise, and the pilot /
   evaluation split would not protect against it because the defect is in the terms
   rather than in their weighting.

## Reproducing

```bash
python - <<'PY'
from daowod.pipeline import PipelineConfig, run_pipeline
run_pipeline(PipelineConfig(
    mode="MAINREVEALED",
    data_root="/path/to/owod_stage",
    split_file="/path/to/owod_stage/ImageSets/OWDETR/pilot_t1_train_4000.txt",
    existing_export="outputs/real_stage1/reference_proposals.npz",
    output_dir="outputs/contribution_a_revealed",
    cache_dir="outputs/contribution_a_revealed/cache",
    require_gpu=False,
))
PY
```

Or set `RUN_MODE = "MAINREVEALED"` in
`notebooks/contribution_a_active_annotation.ipynb`, which exports the proposals
itself on a T4.
