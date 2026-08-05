# Contribution A — Results

What the distribution-aware active annotation experiments measured, and what those
measurements support. Hypotheses are named as in
[`research_design.md`](research_design.md) §2. Reproduction commands are in
[`reproduction.md`](reproduction.md).

Contribution B has no experimental results: only its allocation core exists. See
`research_design.md` §8.

---

## Headline

**H-A1 and H-A2 are falsified in every feature space tested.** The proposal states the
coherence gate is indispensable because uncertainty, diversity and rarity all peak on
isolated outliers, and that local coherence suppresses them. On real S-OWODB Task-1
proposals the gate's *premise* is inverted: background is the most locally homogeneous
stratum in the pool, so any density- or homogeneity-based coherence measure ranks
background highest, and multiplying rarity by it promotes coherent background.

**H-A3 is supported in its diagnosis and refuted in its remedy.** Anchoring the
distribution term on oracle-revealed labels measurably improves the estimator — it halves
the damage the term does — but does not make the term profitable.

**A finding the proposal did not anticipate reorders the work.** The dominant available
signal is geometric. `objectness × box scale` — one line, no clustering, no oracle
feedback, no rounds — finds **1.88×** the baseline's unknown objects. The
distribution-aware question (*which rare class to annotate next*) is downstream of a
question the proposal assumed solved (*which regions contain an object at all*). On this
pool the second dominates the first.

---

## 1. Protocol

One frozen PROB export, S-OWODB / OWDETR Task-1, `per_image_limit = 20`. Every arm shares
one candidate pool, three long-tail severities, three seeds, one budget grid
(100 / 250 / 500 / 1 000 / 2 000 annotated regions), five rounds, one oracle and one
evaluation protocol. Comparisons are paired by seed and severity.

| quantity | value |
|---|---:|
| images / candidate proposals | 2 400 / 48 000 |
| pool composition | 75.1 % background · 23.9 % known-class · 0.98 % unknown |
| unknown proposals | 471 (34 tail / 83 medium / 354 head) |
| known-object · background proposals | 11 460 · 36 069 |
| reachable unknown objects | 364 (273 head / 65 medium / 26 tail) across 38 classes |
| severities realised (head : tail object ratio) | 5.88 / 10.50 / 15.85 |

Budget counts **annotated regions**, which is the proposal's oracle cost. Discovery counts
distinct ground-truth objects, never proposals: forty proposals on one dog are one
discovery.

**Not measured.** No detector retraining happens here, so no known-mAP, U-Recall, WI or
A-OSE number is claimed. The official PROB evaluator remains the only source for those.

---

## 2. The coherence premise is false in the decoder space

Neighbourhood label composition over each proposal's ten nearest neighbours in the 256-d
decoder embedding. This is directly measurable and is exactly what the gate assumes.

| stratum | n | same-label fraction | neighbours on an object |
|---|---:|---:|---:|
| tail unknowns | 34 | **0.015** | 0.262 |
| medium unknowns | 83 | 0.005 | 0.183 |
| head unknowns | 354 | 0.021 | 0.277 |
| known objects | 11 460 | 0.491 | 0.619 |
| **background** | 36 069 | **0.888** | 0.112 |

A tail proposal's neighbourhood is 1.5 % its own class; a background proposal's is 88.8 %
background. The gate-suppression counterfactual confirms the consequence: it removed
118–327 isolated outliers from the top-2 000 and gained a net **0 to +1** tail objects.

This is consistent with how PROB is trained. Its Task-1 decoder discriminates 19 known
classes and scores objectness; nothing in the objective encourages *unknown* classes to
cluster. Known classes do cluster (0.491) because they are in the loss.

**Conditioning the pool does not repair it.** Restricting to the top 25 % by
`objectness × box scale` raises the tail's on-object neighbour fraction 0.26 → 0.40 and
drops background's same-label fraction 0.888 → 0.712, but the tail's own same-label
fraction stays at 0.011.

## 3. No local-structure estimator carries the signal

Eight definitions of local structure, ROC-AUC for "unknown object versus background":

| estimator | AUC | | estimator | AUC |
|---|---:|---|---|---:|
| k-NN density | 0.385 *(inverted)* | | local outlier factor | 0.521 |
| coherence, radius-core (DBSCAN-style) | 0.445 | | neighbourhood mean objectness | 0.564 |
| shared-nearest-neighbour density | 0.447 | | objectness-weighted local density | 0.454 |
| coherence, relative-within-cluster *(baseline)* | 0.481 | | mutual-k-NN coherence | 0.498 |

All lie in [0.385, 0.564]. k-NN density is *inverted*, as the premise analysis predicts.
The best, neighbourhood mean objectness (0.564), barely exceeds a proposal's own
objectness (0.557) — the neighbourhood adds ≈0.007 AUC over the point value. **The problem
is not the estimator's shape**, so adaptive neighbourhoods, HDBSCAN and mutual-neighbour
variants are not promising in this space.

## 4. The rarity estimator is close to uninformative

`rarity` depends on k-means pseudo-classes:

| quantity | value |
|---|---:|
| clusters · median cluster size | 20 · 2 438 |
| median per-cluster background fraction | **0.83** |
| clusters >90 % background | **6 / 20** |
| ARI · NMI, clusters vs true strata | 0.007 · 0.095 |
| ARI, clusters vs true class (unknowns only) | 0.047 |
| **Spearman(estimated rarity, true class rarity)** | **0.116** |

ρ = 0.116 between the rarity a proposal receives and the true frequency of its class means
the "distribution-aware" term is not, in any meaningful sense, aware of the distribution.
Six of twenty clusters are pure background, so "rare pseudo-class" often means "unusual
patch of background".

The gate is the product of these two terms: near-noise rarity × anti-correlated coherence
= ROC-AUC **0.489**, indistinguishable from chance and no better than its own ungated
rarity (0.485).

## 5. Every term of the baseline selects worse than random at the actual budget

ROC-AUC summarises the whole ordering, but a budget buys a *prefix*: 2 000 of 48 000
regions is the top 4 %, and that is where the decision is made. `lift` = precision ÷ the
pool's 0.98 % base rate; below 1.0 is worse than random sampling.

| signal | precision@2000 | unknown proposals | lift |
|---|---:|---:|---:|
| objectness × box scale | 0.0550 | 110 | **5.61×** |
| box scale alone | 0.0540 | 108 | 5.50× |
| objectness-weighted entropy | 0.0270 | 54 | 2.75× |
| objectness | 0.0185 | 37 | 1.89× |
| cluster coherence | 0.0160 | 32 | 1.63× |
| novelty | 0.0070 | 14 | **0.71×** |
| radius-core coherence | 0.0050 | 10 | 0.51× |
| PROB unknown score | 0.0045 | 9 | 0.46× |
| posterior entropy | 0.0045 | 9 | **0.46×** |
| cluster rarity | 0.0040 | 8 | 0.41× |
| **gated rarity × coherence** | **0.0030** | **6** | **0.31×** |
| 1 − max known posterior | 0.0020 | 4 | 0.20× |

The baseline is `0.3 × entropy + 0.2 × novelty + 0.5 × gated`. At the budget it is
evaluated at, **all three of its terms select worse than random**: 0.46×, 0.71×, 0.31×.
That is the mechanism-level explanation — not that the gate failed to add value on top of
a working score, but that the score it was added to was anti-selective in every component.

### The two gaps

| target | supervised ceiling | best *free* signal | best distribution component | estimator gap | representation headroom |
|---|---:|---:|---:|---:|---:|
| unknown vs background | 0.837 | **0.777** | 0.489 | **0.288** | 0.061 |
| tail vs background | 0.816 | 0.759 | 0.509 | 0.250 | 0.057 |
| on-object vs background | 0.936 | 0.748 | 0.464 | 0.284 | 0.188 |

A cross-validated linear probe on the same embeddings reaches 0.837 for unknown-vs-
background, so **the information is present** — the unsupervised estimators do not extract
it. But the representation headroom over the free signal is only ≈0.06, so even a perfect
unsupervised estimator would gain little over something already free.

### A static sort beats the whole apparatus

Sorting the pool once — no rounds, no clustering, no oracle feedback — and annotating the
prefix at budget 2 000:

| ranking | unknown objects found | tail | annotation precision |
|---|---:|---:|---:|
| objectness × box scale | **85** | 4 | 0.055 |
| box scale alone | 84 | 4 | 0.054 |
| objectness alone | 30 | 3 | 0.018 |
| PROB unknown score | 8 | 0 | 0.004 |
| *random* (campaign) | 25 | 2 | 0.013 |
| *full*, the baseline (campaign) | 16 | 1 | 0.008 |

**5.3× more unknown objects than the full distribution-aware strategy** and 3.4× more than
random. PROB's own unknown score finds *zero* unknown objects inside a 500-region budget:
at Task 1 it ranks background above unknown objects, so using it to build or rank the pool
is actively harmful.

## 6. Eleven arms: the free control wins, the distribution term is net-harmful

Distinct unknown objects discovered within a 2 000-region budget (4.2 % of the pool), mean
over three seeds, per severity (moderate / natural / severe). `vs base` is the paired sign
per severity: `+` beat the baseline in every seed, `−` lost in every seed, `~` mixed.

| arm | @2000 (mod/nat/sev) | mean | vs base | tail | classes | precision |
|---|---|---:|---|---:|---:|---:|
| `objectness_area_prior` **[free control]** | 25.0 / 34.0 / 22.0 | **27.0** | `+++` | 1.7 | 14.0 | 0.0183 |
| `prior_revealed_full` | 20.7 / 28.3 / 16.3 | 21.8 | `++~` | 1.6 | 13.1 | 0.0121 |
| `prior_full` | 16.7 / 19.7 / 13.0 | 16.4 | `~~−` | 1.4 | 11.0 | 0.0092 |
| `revealed_full` | 13.7 / 17.7 / 15.0 | 15.4 | `~~~` | 1.3 | 10.6 | 0.0077 |
| `random` | 14.3 / 22.0 / 9.3 | 15.2 | `~+−` | 1.1 | 10.2 | 0.0077 |
| `full` **[BASELINE]** | 12.0 / 17.0 / 14.0 | 14.3 | — | 0.8 | 9.1 | 0.0072 |
| `full_no_coherence` | 14.3 / 15.7 / 8.7 | 12.9 | `+~−` | 1.0 | 8.7 | 0.0064 |
| `revealed_no_gate` | 13.3 / 13.3 / 10.7 | 12.4 | `~~~` | 1.3 | 9.6 | 0.0062 |
| `uncertainty_novelty` | 11.0 / 15.0 / 9.0 | 11.7 | `−−−` | 1.0 | 9.7 | 0.0058 |
| `uncertainty` | 8.0 / 9.0 / 5.0 | 7.3 | `−−−` | 0.0 | 5.7 | 0.0037 |
| `revealed_support_only` | 2.3 / 4.0 / 4.7 | 3.7 | `−−−` | 0.2 | 3.1 | 0.0018 |

* The free control is the **only** arm that beats the baseline in every severity with a
  consistent sign across seeds, at **1.88×** its unknown discovery.
* **The baseline is not distinguishable from random** (14.3 vs 15.2, signs `~+−`), which is
  what §5 predicts when all three of its terms are sub-random at the budget.
* Every arm carrying a distribution term scores below the prior alone.

**The cost the distribution term imposes on a score that works:**

| arm | @2 000 | objects lost to the gate |
|---|---:|---:|
| `objectness_area_prior` (no gate) | 27.0 | — |
| `prior_revealed_full` (label-anchored gate) | 21.8 | **−5.2** |
| `prior_full` (unsupervised gate) | 16.4 | **−10.6** |

Label-anchoring halves the damage — H-A3's diagnosis is right, and anchored rarity does
carry more signal than k-means rarity — but it does not change the sign.

**`revealed_support_only` is actively harmful, and §2 explains why.** At 3.7 objects it is
the worst arm, losing to the baseline in every seed of every severity. "Resembles a region
the oracle confirmed as unknown" fails because a confirmed unknown sits *inside the
background mass*: its own ten nearest neighbours are 74 % background, so similarity to it
is largely similarity to nearby background. The competing explanation — that the term
re-buys objects it already found — was **tested and refuted**:
proposals-per-discovered-object is ≈1.00 for every arm. It is not redundant, it is
mis-aimed.

**Cold start is not the explanation.** Deterministic replay shows the revealed bank warm
from round 2 (regions / classes):

| arm | round 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| `revealed_full` | 1 / 1 | 10 / 7 | 14 / 9 | 14 / 9 |
| `revealed_no_gate` | 6 / 4 | 8 / 6 | 9 / 7 | 11 / 9 |
| `prior_revealed_full` | 1 / 1 | 6 / 6 | 17 / 13 | 24 / 15 |

Three to four of five rounds ran with a warm, multi-class bank. The anchored term had
data; it does not point where discovery is.

## 7. Representation experiment: the embedding is the problem, but fixing it is not enough

The premise is a claim about feature geometry, so it was retested with the acquisition
held completely fixed and only the space varied — nine spaces, same boxes, same pool.

**A correction to the headline statistic first.** The original decisive number was
`tail purity ÷ background purity` = 0.017. **That ratio is confounded:** a class with `c`
members cannot supply more than `min(c − 1, k)` same-class neighbours, so at k = 10 the
tail group (classes of 1–6 proposals, three holding exactly one) has a purity *ceiling* of
0.235 while background's is 1.0. Part of the apparent 60-fold gap was the class frequency
that *defines* the tail, not the geometry under test. Ceiling-normalising raises the
decoder space's advantage 0.017 → **0.058**: the conclusion survives, the magnitude was
overstated roughly fourfold. Everything below is on the corrected basis.

| representation | dim | tail normalised | background same-label | **normalised advantage** | premise holds |
|---|---:|---:|---:|---:|---|
| `imagenet_resnet50` | 2048 | 0.262 | 0.7962 | **0.330** | no |
| `dino_resnet50` | 2048 | 0.241 | 0.8097 | **0.297** | no |
| `prob_decoder_plus_dino` | 2304 | 0.241 | 0.8466 | 0.285 | no |
| `dino_whitened` | 64 | 0.230 | 0.8200 | 0.281 | no |
| `prob_decoder_whitened` | 64 | 0.067 | 0.8721 | 0.076 | no |
| `prob_posterior` | 20 | 0.056 | 0.8709 | 0.064 | no |
| **`prob_decoder`** *(baseline)* | 256 | 0.052 | 0.8879 | **0.058** | no |
| `prob_decoder_minus_top4` | 256 | 0.045 | 0.8658 | 0.052 | no |
| `prob_geometry` | 5 | 0.006 | 0.8083 | 0.008 | no |

**The representation matters, by a large factor** — an appearance encoder raises the
statistic 5–6× over PROB's decoder. **No space clears break-even.** Every advantage is
below 1.0, so in every space a homogeneity-based coherence term still ranks background
above the tail; the best space is 3× short. **No transform of the decoder space helps:**
whitening gives 0.058 → 0.076, and projecting out the four leading principal components
*hurts* (0.052), refuting the alternative explanation that the structure is present but
drowned by high-variance background directions. It is not there to recover.

**The ceiling-free statistic is sharper.** For each unknown region whose class has ≥2
members, where does its nearest same-class sibling sit in the similarity ordering over all
48 000 proposals?

| representation | tail median rank | head median rank | unknown median rank | sibling in top 10 |
|---|---:|---:|---:|---:|
| `dino_resnet50` | 213 | **6.0** | **6.0** | **0.545** |
| `imagenet_resnet50` | **8.0** | 6.5 | 8.0 | 0.517 |
| `prob_decoder_plus_dino` | 66 | 6.0 | 6.0 | 0.536 |
| **`prob_decoder`** *(baseline)* | 843 | 147.5 | **202.0** | **0.126** |
| `prob_geometry` | 4 114 | 334.0 | 605.0 | 0.053 |

In PROB's decoder space the median unknown region's closest same-class sibling is the
**202nd** nearest neighbour; in DINO space it is the **6th** — a **34-fold** improvement —
and the fraction of unknown regions with a sibling in their top-10 rises from 12.6 % to
54.5 %. On the head group, whose ceiling is exactly 1.0 so no artefact can hide, the median
rank falls 147.5 → 6.0. Silhouette over true unknown classes rises −0.366 → −0.011 (DINO)
and −0.002 (ImageNet); Davies–Bouldin falls 4.70 → 3.39 / 3.19. Both remain at or below
the "no structure" boundary: the classes are no longer strongly interleaved, but they are
still not *separated*.

**So the decoder embedding genuinely does not preserve semantic neighbourhoods, and an
off-the-shelf appearance encoder largely does — yet the premise still fails.** Background's
same-label fraction is 0.80–0.89 in *every* space including DINO's, because 36 069 crops of
sky, road, foliage and wall genuinely do look like one another. Improving class
neighbourhoods 34× cannot change that.

The local/global pair makes the point from the other side:

| representation | nearest-tail AUC (local) | centroid AUC (global) |
|---|---:|---:|
| `prob_decoder` | 0.798 | 0.716 |
| `dino_resnet50` | 0.835 | 0.715 |
| `dino_whitened` | 0.860 | 0.913 |

Proximity to other tail regions already discriminates tail from background at AUC ≈ 0.8 in
the *baseline* space. The information a coherence term needs is partly there even in the
decoder. What the gate reads instead is *absolute local homogeneity*, which background
maximises — **and that is a property of the formulation, not of the space.**

**Pseudo-classes are not a representation problem either.** k-means with 20 clusters over
a 75 %-background pool fails to recover unknown classes in any space (ARI ≤ 0.076), and the
correlation between estimated and true rarity does not improve — it goes slightly negative
for both crop encoders (`dino_resnet50` −0.068, `imagenet_resnet50` −0.148, against
`prob_decoder` 0.116). Better neighbourhoods do not help a clustering whose 20 centroids
are spent on background modes.

---

## 8. What survived

* **The information is present.** A linear probe reaches ROC-AUC 0.837 for unknown vs
  background and 0.816 for tail vs background on the same embeddings.
* **Region-level annotation is the right unit.** The oracle, the discovery metrics and the
  budget-curve protocol behaved as designed; every leakage control passed at every round
  of every cell, and runs reproduce seed-for-seed.
* **The controlled long-tail protocol works.** Three severities were realised with
  genuinely distinct head:tail ratios (5.88 / 10.50 / 15.85) and the negative result
  reproduced across all three, which is what makes it a result rather than a fluke.
* **Multi-round feedback is available and cheap**, and the oracle labels are usable
  supervision.

## 9. Limitations

**Tail discovery is under-powered and its claims are unfalsifiable as designed.** At the
largest budget each strategy discovered 0, 1 or 2 tail objects out of 26. With outcomes
that discrete no gate could have been demonstrated even had it worked. Tail columns above
are reported for completeness, never as evidence. The AUC analyses are well powered (471
unknown positives, 36 069 negatives) and are what the conclusions rest on.

**Three methodological corrections were made, and each changed a conclusion.**

1. **Select on precision at the budget, not ROC-AUC.** By AUC the gated term (0.486) looks
   merely uninformative and the anchored support (0.708 with a diverse 14-region bank)
   looks like a large win. At the actual budget the gated term is 3× *worse* than random,
   and the anchored term's advantage depends on a bank the campaign cannot build.
2. **Ceiling-normalise the purity ratio** (§7). The original statistic overstated the
   effect roughly fourfold.
3. **A reported signal-to-noise of 0.28 for the coherence gate was meaningless** and is
   withdrawn. It varied the pool realisation and the clustering seed together; redrawing
   every proposal embedding makes two selections nearly disjoint (self-Jaccard 0.08), so it
   measured "does a different pool select different images", which it trivially does.
   Signal and noise must both be measured on a *fixed* pool, varying only the
   acquisition's own randomness.

**`objectness × box scale` correlates with object size,** so a scale-driven strategy may
systematically miss small rare objects. Discovery must be stratified by object size before
the prior is adopted as anything but a control.

## 10. Status of the proposal's hypotheses

| Hypothesis | Status |
|---|---|
| **H-A1** the gate is indispensable because U, D, w all peak on outliers | **Falsified as stated.** The premise inverts: background is the most locally homogeneous stratum, so the gate promotes coherent background. The *motivation* is sound — ungated rarity is also sub-random (0.41×) — but so is the gated form (0.31×) |
| **H-A2** rarity fires only when rare AND locally supported | **Falsified in nine spaces.** Normalised advantage 0.008–0.330, all below the 1.0 break-even |
| **H-A3** `ĉ`, `n̂_c` are estimates, refreshed each round | **Diagnosis supported, remedy insufficient.** Label anchoring halves the gate's damage (−10.6 → −5.2 objects) but does not make the term profitable |
| Controlled long-tail construction | **Implemented and validated**, three distinct severities |
| Grouped metrics · annotation-efficiency curve | **Implemented.** Tail arm under-powered (§9) |

**Not refuted as a research direction.** These are results about PROB's Task-1 decoder
space and eight alternatives at a ~1 % positive rate. What they establish is narrower and
more useful than "the idea is wrong": the *formulation* reads absolute local homogeneity,
which background maximises in any appearance space, and the first-order problem on this
pool is object-versus-background rather than rare-versus-common.

**Literature consistency.** This is not anomalous. OW-DETR and PROB both introduce explicit
objectness or pseudo-labelling mechanisms precisely because a closed-set discriminative
backbone does not organise out-of-vocabulary categories into clusters; the measured
same-label fraction of 0.015 for unknown versus 0.491 for known classes is that design
assumption showing up as data. Likewise, a geometric prior dominating weak semantic signals
under extreme imbalance is a familiar active-learning failure mode: at a ~1 % positive rate
an acquisition function with AUC 0.49 is indistinguishable from random, and any prior at
AUC 0.78 dominates it however principled the former is.

## 11. Research debt

**The representation experiment's acquisition phase is incomplete.** The geometry
comparison (§7) is complete across all nine spaces and is what the conclusions above rest
on. Re-running the *acquisition arms* in a different space finished only for
`prob_decoder`; the `dino_resnet50` run produced a pilot ablation and one severity's study
state but no comparison tables. So "do the same strategies behave differently in DINO
space?" is **not yet answered**, and the earlier write-up's Phase 5 and Phase 7 sections
were never filled in. §7's geometry result predicts the gate still fails there (advantage
0.297, still 3× short of break-even), but that is a prediction, not a measurement.

**If Contribution A is continued**, the measurements select the next steps: adopt the
informativeness prior as the `U(x)` term and report every future method against it, not
only against random; complete the DINO acquisition run; restate the protocol so tail
effects are measurable (unknown discovery as the primary metric over 364 objects rather
than 26, budgets as a fraction of reachable objects, ≥5 seeds, exact binomial intervals);
and do not tune α, β, γ or `p` on this pool, because with every semantic term below the
base rate a weight sweep would be fitting noise.
