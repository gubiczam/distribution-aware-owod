# Why the coherence gate failed, and what follows from it

Every number in this document is produced by
`analysis/audit_coherence_failure.py` from the real S-OWODB / OWDETR Task-1
proposal export, and is reproducible with:

```bash
python analysis/audit_coherence_failure.py \
    --export outputs/real_stage1/reference_proposals.npz \
    --annotations ~/owod_stage/Annotations \
    --output outputs/audit_contribution_a
```

Pool under analysis: 2 400 images, 48 000 candidate proposals, 364 reachable
unknown objects (273 head / 65 medium / 26 tail) across 38 unknown classes. The
pool is 75.1 % background, 23.9 % known-class objects and 0.98 % unknown objects.

The first full run of Contribution A found that the coherence gate did **not**
improve tail discovery, in all three severities, with a consistent sign across
seeds. This document localises that result to specific components and states which
of the proposal's assumptions survived.

---

## 1. What failed

### 1.1 The coherence premise is false in this feature space

The gate assumes that a rare true class forms a locally consistent group while an
isolated false positive does not — DBSCAN's core-versus-noise intuition. That is a
claim about neighbourhood label composition, and it is directly measurable. Over
each proposal's ten nearest neighbours in the 256-d decoder embedding space:

| stratum | n | same-label fraction | neighbours on an object |
|---|---:|---:|---:|
| tail unknowns | 34 | **0.015** | 0.262 |
| medium unknowns | 83 | 0.005 | 0.183 |
| head unknowns | 354 | 0.021 | 0.277 |
| known objects | 11 460 | 0.491 | 0.619 |
| **background** | 36 069 | **0.888** | 0.112 |

A tail proposal's neighbourhood contains 1.5 % of its own class. A background
proposal's contains 88.8 % background. Background is by a factor of ~60 the most
locally coherent stratum in the pool, so *any* density- or homogeneity-based
coherence measure ranks background highest. Multiplying rarity by such a measure
promotes coherent background, which is exactly what the gate-suppression
counterfactual recorded: it removed 118–327 isolated outliers from the top-2 000
and gained a net 0 to +1 tail objects.

This is consistent with how the detector was trained. PROB's Task-1 decoder is
optimised to discriminate 19 known classes and to score objectness; nothing in its
objective encourages unknown classes to cluster. The known classes do cluster
(same-label fraction 0.491) precisely because they are in the loss. The unknowns
are not, so they land wherever — dispersed through a feature space whose dominant
mass is background.

**Conditioning the pool does not repair it.** Restricting to the top 25 % by
objectness × box scale raises the tail's on-object neighbour fraction from 0.26 to
0.40 and drops background's same-label fraction from 0.888 to 0.712, but the tail's
own same-label fraction stays at 0.011. Class-level local coherence is not
recoverable here by filtering.

### 1.2 No local-structure estimator carries the signal

*(AUC below; see §1.4 for the same comparison at the actual annotation budget, which
is the decision-relevant one.)*

Eight alternative definitions of local structure were measured on the same pool
(ROC-AUC for "unknown object versus background"):

| estimator | AUC |
|---|---:|
| k-NN density | 0.385 |
| coherence, radius-core (DBSCAN-style) | 0.445 |
| shared-nearest-neighbour density | 0.447 |
| coherence, relative-within-cluster (baseline) | 0.481 |
| mutual-k-NN coherence | 0.498 |
| local outlier factor | 0.521 |
| neighbourhood mean objectness | 0.564 |
| objectness-weighted local density | 0.454 |

All lie in [0.385, 0.565]; k-NN density is *inverted*, as the premise analysis
predicts. Neighbourhood mean objectness (0.564) barely exceeds a proposal's own
objectness (0.557), so the neighbourhood adds ≈0.007 AUC over the point value.
Adaptive neighbourhoods, HDBSCAN and mutual-neighbour variants are therefore not
promising in this space; the problem is not the estimator's *shape*.

### 1.3 The rarity estimator is close to uninformative

`rarity` depends on k-means pseudo-classes. Measured:

| quantity | value |
|---|---:|
| clusters | 20 |
| median cluster size | 2 438 |
| median per-cluster background fraction | **0.83** |
| clusters that are >90 % background | **6 / 20** |
| ARI, clusters versus true strata | 0.007 |
| NMI, clusters versus true strata | 0.095 |
| ARI, clusters versus true class (unknowns only) | 0.047 |
| **Spearman(estimated rarity, true class rarity)** | **0.116** |

A rank correlation of 0.116 between the rarity a proposal receives and the true
frequency of its class means the "distribution-aware" term is not, in any
meaningful sense, aware of the distribution. Six of twenty clusters are pure
background, so "rare pseudo-class" frequently means "unusual patch of background".

The gate is the product of these two terms. A near-noise rarity multiplied by an
anti-correlated coherence yields ROC-AUC 0.489 — indistinguishable from chance, and
below its own ungated rarity (0.485 → the two are within noise of each other and of
0.5). That is the whole result: the compared quantities carry no signal, so the
comparison between them measures nothing but variance.

### 1.4 Every term of the baseline score is worse than random at the actual budget

ROC-AUC summarises the whole ordering. An annotation budget buys a *prefix* of it —
2 000 of 48 000 regions is the top 4 % — and that is where the decision is made.
Measuring precision in the top 4 % (`lift` = precision ÷ the pool's 0.98 % base
rate; below 1.0 is worse than random sampling):

| signal | precision@2000 | unknown proposals | lift |
|---|---:|---:|---:|
| objectness × box scale | 0.0550 | 110 | **5.61×** |
| box scale alone | 0.0540 | 108 | 5.50× |
| objectness-weighted entropy | 0.0270 | 54 | 2.75× |
| objectness | 0.0185 | 37 | 1.89× |
| cluster coherence | 0.0160 | 32 | 1.63× |
| novelty | 0.0070 | 14 | **0.71×** |
| radius-core coherence | 0.0050 | 10 | **0.51×** |
| PROB unknown score | 0.0045 | 9 | **0.46×** |
| posterior entropy | 0.0045 | 9 | **0.46×** |
| cluster rarity | 0.0040 | 8 | **0.41×** |
| **gated rarity × coherence** | **0.0030** | **6** | **0.31×** |
| 1 − max known posterior | 0.0020 | 4 | 0.20× |

The baseline strategy is `0.3 × entropy + 0.2 × novelty + 0.5 × gated`. At the
budget it is actually evaluated at, **all three of its terms select worse than
random**: 0.46×, 0.71× and 0.31×. That is the complete, mechanism-level explanation
of the negative result — not that the gate failed to add value on top of a working
score, but that the score it was added to was anti-selective in every component.

**A correction to this audit's own method.** The first version of this analysis
ranked candidate estimators by ROC-AUC alone. On that basis the baseline's gated
term (AUC 0.486) looks merely uninformative rather than harmful, and the
label-anchored support term (AUC 0.708 with a diverse 14-region bank) looks like a
large win. Precision at the budget tells a different story for both: the gated term
is 3× *worse* than random, and the anchored term's advantage — real at 4.7× with a
well-spread bank — depends on a bank the campaign cannot actually build (§ results).
`daowod.audit.precision_at_budget` now reports this, and it should be the quantity
future experiments are selected on.

### 1.5 Statistical power was also insufficient — but that is not the main cause

At the largest budget (2 000 regions, 4.2 % of the pool) the number of *tail
objects* each strategy discovered was 0, 1 or 2 out of 26. With outcomes that
discrete, no gate could have been demonstrated even had it worked. This is a real
limitation of the experiment's design, and it is why the follow-up reports unknown
discovery (364 reachable objects, 471 proposals) as the primary metric and tail
discovery as secondary.

It is, however, secondary to the estimator failure: the AUC analyses above use 471
unknown positives and 36 069 negatives, are well powered, and are unambiguous.

---

## 2. What survived

**The representation is not empty.** A cross-validated linear probe on the same
decoder embeddings separates unknown objects from background at ROC-AUC 0.837, and
tail objects from background at 0.816. The information the proposal wants to exploit
is present; the unsupervised estimators simply do not extract it.

**Region-level annotation is the right unit.** The oracle, the discovery metrics
and the budget-curve protocol behaved as designed; every leakage control passed at
every round of every cell, and the run is reproducible seed-for-seed.

**The long-tail protocol works.** Three severities were realised with genuinely
distinct head:tail object ratios (5.88 / 10.50 / 15.85) and the negative result
reproduced across all of them, which is what makes it a result rather than a fluke.

**Multi-round feedback is available and cheap.** The oracle labels the campaign buys
are usable supervision, and the plan already asks for them to update the estimated
distribution.

---

## 3. The finding that reframes the problem

Two gaps must be distinguished, and the audit reports both:

| target | supervised ceiling | best *free* unsupervised signal | best distribution component | estimator gap | representation headroom |
|---|---:|---:|---:|---:|---:|
| unknown vs background | 0.837 (embeddings) | **0.777** (objectness × box scale) | 0.489 (gated) | **0.288** | 0.061 |
| tail vs background | 0.816 | 0.759 | 0.509 | 0.250 | 0.057 |
| on-object vs background | 0.936 | 0.748 | 0.464 | 0.284 | 0.188 |

* The **estimator gap** (≈0.25–0.29) says the distribution-aware components score
  far below something that is already free.
* The **representation headroom** (≈0.06 for the unknown/tail contrasts) says that
  even a perfect unsupervised estimator would gain little *over that free signal*
  in this feature space.

And the free signal is startlingly strong on the metric that matters. Sorting the
pool once by box scale — no rounds, no clustering, no oracle feedback — and
annotating the prefix:

| ranking | budget | unknown objects found | tail objects | annotation precision |
|---|---:|---:|---:|---:|
| objectness × box scale | 2 000 | **85** | 4 | 0.055 |
| box scale alone | 2 000 | 84 | 4 | 0.054 |
| objectness alone | 2 000 | 30 | 3 | 0.018 |
| PROB unknown score | 2 000 | 8 | 0 | 0.004 |
| — *v2:random* (campaign) | 2 000 | 25 | 2 | 0.013 |
| — *v2:full*, the baseline (campaign) | 2 000 | 16 | 1 | 0.008 |

A one-line sort finds **5.3× more unknown objects than the full
distribution-aware strategy** and 3.4× more than random. PROB's own unknown score
finds zero unknown objects inside a 500-region budget: at Task 1 it ranks
background above unknown objects, so using it to build or rank the pool is
actively harmful.

So the honest diagnosis is not only "the coherence gate failed". It is that **the
acquisition score was assembled from semantic signals that are at or below chance
on this pool, while the dominant available signal — how large the predicted box is
— was not in the score at all.** The first-order problem is *object versus
background*, not *rare versus common*, and the proposal's formula implicitly assumed
the first problem was already solved.

### Literature consistency

This is not anomalous. Open-world detectors deliberately do not rely on
feature-space clustering to find unknowns: OW-DETR and PROB both introduce explicit
objectness or pseudo-labelling mechanisms because a closed-set discriminative
backbone does not organise out-of-vocabulary categories into clusters. The measured
same-label fraction of 0.015 for unknown classes versus 0.491 for known classes is
that design assumption showing up as data. Likewise, the observation that a
geometric prior dominates weak semantic signals under extreme class imbalance is a
familiar failure mode of active learning at low positive rates: with a ~1 % positive
rate, an acquisition function whose AUC is 0.49 performs indistinguishably from
random sampling, and any prior with AUC 0.78 dominates it regardless of how
principled the former is.

---

## 4. Follow-up experiments, ranked by expected information gain

Each answers exactly one question. Costs assume the existing pipeline and one
cached export (CPU only unless stated).

### E1 — Does anchoring the distribution term on revealed labels make it informative? **[implemented]**

* **Motivation.** The estimator gap is 0.29 and active learning supplies labels for
  free; the plan already asks for the observed distribution to be updated from
  revealed classes, and the baseline used revealed labels only for saturation.
* **Hypothesis.** Rarity from the nearest *revealed* class and coherence from
  similarity to *confirmed* unknown regions carry more signal than k-means rarity
  and pool density.
* **Pre-registered expected outcome.** Unknown discovery and annotation precision
  improve substantially; tail-versus-head selectivity does not, because a realistic
  budget reveals 10–40 unknown regions spread over ~20 classes, i.e. one or two per
  class. Measured sample complexity (held-out AUC for unknown vs background): 5
  revealed unknowns → 0.671 similarity / 0.710 probe; 20 → 0.694 / 0.755; 160 →
  0.689 / 0.814. Five labels already beat every unsupervised alternative.
* **Effort.** One module (`daowod.revealed`), one flag on `StrategySpec`, three
  registry entries. **Cost.** ~48 min CPU for the 11-arm matrix.
* **Scientific value.** High: it tests the proposal's hypothesis with a working
  estimator instead of a broken one, and the similarity variant plateaus at 0.69,
  below the free prior — so it also predicts its own ceiling.
* **Risks.** Cold start (round 1 has no labels; handled by deferring to the
  unsupervised value, which makes round 1 bit-identical to the baseline). Reveal
  bias: the anchored term is fitted on regions the strategy itself chose, so it can
  reinforce its own early mistakes — visible as saturating unknown discovery.

### E2 — Does the informativeness prior beat the whole semantic apparatus? **[implemented]**

* **Motivation.** Section 3: a one-line sort finds 5.3× more unknown objects than
  the baseline.
* **Hypothesis.** `objectness × box scale` in the `U(x)` slot dominates every
  posterior-derived uncertainty, and the distribution term adds little on top.
* **Effort.** One uncertainty method, three registry entries. **Cost.** included in
  the same matrix.
* **Scientific value.** Very high, and uncomfortable: it establishes the floor that
  any future version of Contribution A has to clear.
* **Risks.** Box scale correlates with annotation cost and with object size, so a
  scale-driven strategy may systematically miss small rare objects. Must be reported
  per size stratum in the next iteration.

### E3 — Restate the protocol so tail effects are measurable

* **Motivation.** 26 reachable tail objects and 0–2 discoveries per cell cannot
  resolve any effect.
* **Design.** Primary metric = unknown discovery (364 objects); budgets expressed as
  a fraction of reachable objects rather than of the pool; seeds ≥ 5; exact binomial
  intervals on discovery counts; the free prior as a mandatory control arm.
* **Effort.** Low (metrics and modes already exist). **Cost.** ~1.5 h CPU for 5
  seeds. **Value.** High — without it, later experiments will keep producing
  unfalsifiable results.
* **Risks.** None methodological; it weakens the headline claim by admitting the
  tail metric is under-powered, which is the correct thing to do.

### E4 — Is class structure for unknowns present in a *different* feature space? **[implemented; see `e4_representation_results.md`]**

* **Motivation.** The premise failed in decoder space, and the representation
  headroom over the free prior there is only ~0.06. The premise is a property of the
  *space*, so it must be retested elsewhere before the gate is abandoned.
* **Outcome.** Run. Re-embedding the identical boxes with DINO ResNet-50 improves
  semantic neighbourhoods enormously — median nearest-same-class-sibling rank 202 to
  6 — but the tail-versus-background purity ratio stays at 0.30, three times short of
  the 1.0 the gate needs, because background remains the most locally homogeneous
  stratum in every space. Details and the corrected statistic in
  [`e4_representation_results.md`](e4_representation_results.md).
* **Design.** Re-embed the same candidate boxes with a self-supervised or
  vision-language encoder (DINOv2 or CLIP on the cropped regions), then recompute
  exactly the measurements in §1.1–1.2. The decisive statistic is the tail
  same-label fraction: 0.015 in decoder space; if it rises above the background
  value, the gate becomes viable and E1's mechanism should be retested there.
* **Effort.** Medium: crop extraction plus one encoder pass over 48 000 regions;
  the audit script then runs unchanged. **Cost.** ~1 GPU-hour plus the export.
* **Value.** High and decisive for the contribution's future, but it should follow
  E1–E3 because it costs GPU time and because the audit already shows the cheaper
  fixes matter more.
* **Risks.** Cropped-region embeddings discard context, and CLIP's vocabulary
  overlaps the "unknown" classes, which could make the diagnostic optimistic
  relative to a genuinely open world.

### E5 — Does a learned discriminator (not similarity) close the gap?

* **Motivation.** In the sample-complexity curve the probe keeps improving with
  labels (0.710 → 0.814 from 5 → 160 revealed unknowns) while the similarity variant
  plateaus at ~0.69. The probe is the only measured estimator that exceeds the free
  prior.
* **Design.** Replace the similarity-based support with a logistic discriminator
  refitted each round on the revealed labels; keep everything else fixed.
* **Effort.** Low (the support term is one function). **Cost.** ~48 min CPU.
* **Value.** Medium-high — it is the natural successor to E1 and the curve predicts
  it wins where E1 plateaus.
* **Risks.** Refitting on self-selected data amplifies reveal bias more than a
  similarity term does; needs the E3 protocol to detect.

### E6 — Does the candidate pool's composition bound everything else?

* **Motivation.** Retention measurements: conditioning on objectness × box scale at
  25 % doubles the unknown rate (0.98 % → 1.98 %) and halves background, while
  keeping 166/364 unknown and 12/26 tail objects.
* **Design.** Rerun the full matrix on the conditioned pool, reporting absolute
  discovered-object counts against the *unconditioned* reachable set so pools remain
  comparable.
* **Effort.** Low. **Cost.** ~48 min. **Value.** Medium: it changes every strategy's
  precision but does not test the hypothesis.
* **Risks.** Pool conditioning changes recall denominators; comparing recalls across
  pools without the absolute counts would be misleading.

**Deprioritised by the evidence.** Adaptive neighbourhoods, HDBSCAN,
shared-nearest-neighbour density, mutual-neighbour coherence and LOF were all
measured in §1.2 and lie between 0.385 and 0.521 AUC. Implementing them as
acquisition strategies is not justified; the limitation is the space, not the
estimator's shape.

---

## 5. What was implemented, and what it is compared against

E1 and E2, in one 11-arm matrix on a single shared pool (identical severities,
seeds, budgets, oracle and evaluation protocol):

| arm | family | `U(x)` | distribution term |
|---|---|---|---|
| `v2:random` | baseline | — | — |
| `v2:uncertainty` | baseline | entropy | — |
| `v2:uncertainty_novelty` | baseline | entropy | — |
| `v2:full_no_coherence` | baseline | entropy | cluster rarity, ungated |
| `v2:full` | **baseline under test** | entropy | cluster rarity × pool density |
| `v2:objectness_area_prior` | free control | objectness × box scale | — |
| `v2:prior_full` | new | objectness × box scale | cluster rarity × pool density |
| `v2:prior_revealed_full` | new | objectness × box scale | revealed rarity × revealed support |
| `v2:revealed_support_only` | new | entropy | revealed support only |
| `v2:revealed_no_gate` | new | entropy | revealed rarity, ungated |
| `v2:revealed_full` | new | entropy | revealed rarity × revealed support |

The baseline is unchanged and still registered under its original name; the
label-anchored path is a separate `distribution_estimator="revealed"` flag whose
cold-start behaviour makes round 1 bit-identical to the baseline. Results are in
`docs/contribution_a_revealed_results.md`.
