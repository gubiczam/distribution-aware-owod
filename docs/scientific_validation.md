# Is the current Contribution A scientifically defensible?

Validation of the acquisition function as it stands. No algorithm was changed and
no component was added; every number below comes from
`analysis/validate_contribution_a.py` plus two falsification probes, all of which
call only existing library functions.

**Provenance.** Measurements are on the PROB-calibrated synthetic pool
(`daowod.simulation`), 20 Task-2 classes, imbalance ratio 20, 20 proposals per
image, 198 images / 3,960 proposals — the pilot protocol's own parameters — over
7 pool configurations × 3 pool realisations × 5 clustering seeds. No real
M-OWODB proposals were available on this machine (no CUDA, no Task-1 checkpoint,
one image in `JPEGImages`). `docs/diagnostics_report.md` §8 has the commands that
reproduce all of it on real exports. Conclusions that follow from the arithmetic
rather than from the simulated data distribution are marked **[structural]**.

**A methodological correction made during this work.** A first version of the
analysis varied the pool realisation and the clustering seed together and reported
a signal-to-noise of 0.28 for the coherence gate. That number was meaningless:
redrawing every proposal embedding makes two selections nearly disjoint
(self-Jaccard 0.08), so it measured "does a different pool select different
images", which it trivially does. Signal and noise are now both measured on a
**fixed** pool, varying only the acquisition's own randomness. Pool realisations
are used solely to ask whether a conclusion is stable.

---

## Phase 1 — Does each component carry unique information?

Pairwise redundancy on normalised components (mean over 7 configurations × 3
realisations):

| pair | \|Spearman\| | normalised MI |
|---|---|---|
| **rarity ↔ gated** | **0.982** | **0.815** |
| novelty ↔ rarity | 0.658 | 0.344 |
| novelty ↔ gated | 0.620 | 0.295 |
| uncertainty ↔ gated | 0.322 | 0.044 |
| uncertainty ↔ novelty | 0.321 | 0.043 |
| uncertainty ↔ rarity | 0.321 | 0.050 |
| novelty ↔ coherence | 0.235 | 0.095 |
| coherence ↔ gated | 0.133 | 0.115 |
| uncertainty ↔ coherence | 0.026 | 0.008 |
| **rarity ↔ coherence** | **0.007** | 0.025 |

Three findings.

**Uncertainty is genuinely independent of the distribution terms.** ρ = 0.32 with
rarity, 0.03 with coherence, mutual information 0.05 and 0.008. Whatever else is
true, the score is not measuring one thing twice here.

**Rarity and coherence are independent of each other** (ρ = 0.007). They do
measure different properties, which is the premise Contribution A needs.

**But the gated term is almost entirely redundant with rarity: ρ = 0.982,
normalised MI = 0.815.** Multiplying rarity by coherence and re-ranking recovers
almost the same ordering. This is the central quantitative fact of this report.

Novelty is the weakest contributor: 0.658 redundant with rarity, and the smallest
unique share of the composite score.

*One caution on a statistic that looks supportive and is not.* Incremental R²
against the full score is gated 0.402, uncertainty 0.179, novelty 0.062. That
ordering is largely **tautological** — the full score is
`0.3·u + 0.2·n + 0.5·gated`, so the term with the largest weight necessarily
explains the most variance in it. It is evidence about the weights, not about
information content. The non-circular evidence is the rarity↔gated redundancy
above and the selection-level tests below.

## Phase 2 — What does each component contribute to the selection?

Budget 10. "Signal" is how much two strategies' selected image sets differ;
"noise" is how much one strategy's own selection moves across clustering seeds on
the identical pool.

| contrast | selection difference | ρ(proposal scores) | signal / noise | range across configurations |
|---|---|---|---|---|
| full vs random | 0.974 | — | 3.07 | [0.96, 0.99] |
| **full vs full_no_rarity** | **0.802** | 0.301 | **2.53** | [0.74, 0.91] |
| **full vs full_no_uncertainty** | **0.759** | 0.897 | **2.39** | [0.54, 0.92] |
| **full vs full_no_coherence** | **0.294** | **0.988** | **0.93** | **[0.13, 0.41]** |
| rarity vs coherence | 0.993 | 0.006 | 1.33 | [0.97, 1.00] |
| gated vs additive R+C | 0.848 | 0.779 | 1.23 | [0.68, 0.99] |
| uncertainty vs rarity | 0.963 | −0.324 | ∞ | [0.95, 0.98] |
| uncertainty vs coherence | 0.968 | −0.025 | ∞ | [0.94, 0.98] |

Clustering noise of `v2:full` = **0.317**. So:

* **Removing rarity changes 80 % of the selection (S/N 2.53). Removing
  uncertainty changes 76 % (S/N 2.39).** Both terms are doing substantial,
  separable work.
* **Removing the coherence gate changes 29 % — less than the acquisition's own
  clustering noise (S/N 0.93 < 1).** At budget 10, running the same strategy with
  a different KMeans seed changes the selection more than removing the gate does.
* The gate's effect varies **3-fold** across pool configurations ([0.13, 0.41]),
  so it is the one contrast whose size depends heavily on assumptions the
  simulator cannot validate.

Clustering noise is itself strongly strategy-dependent: `v2:uncertainty` **0.000**
(deterministic — it needs no clustering), `v2:full` 0.317, `v2:rarity` 0.819,
`v2:random` 0.978. Any strategy whose score depends on pseudo-labels inherits the
partition's instability.

**The gate gets weaker as the budget grows**, which is the opposite of what a
useful mechanism should do: selection difference 0.339 / 0.294 / 0.255 / 0.204 at
budgets 5 / 10 / 20 / 40.

## Phase 3 — Is the method stable?

**Coherence exponent p** (selection difference vs ungated, budget 10):

| p | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|
| effect | 0.000 | **0.061** | 0.326 | 0.368 | 0.455 |

The gate does essentially nothing below p = 1. **The pilot campaign's `full_p05`
variant (p = 0.5) had a selection effect of 0.061** — under a fifth of the
clustering noise. That comparison could not have produced a result.

**Cluster count** — the parameter the method does not determine:

| k | 5 | 10 | 20 | 40 | 60 | 80 |
|---|---|---|---|---|---|---|
| gate effect | 0.437 | 0.061 | 0.326 | 0.121 | 0.061 | 0.121 |
| S/N | 0.69 | 0.14 | 1.47 | 0.25 | 0.26 | 0.67 |

**Non-monotone, a 7× range, no discernible pattern.** The configured k = 20
happens to be the best of the six values tried. That is the signature of a
quantity dominated by partition noise, not of a mechanism responding to a
parameter.

**Neighbour count** is stable (0.326 flat for k = 2, 3, 5; declining after).
**top-k aggregation** is non-monotone (0.232 / 0.326 / 0.326 / 0.368 / 0.121 for
k = 1, 2, 3, 5, 10) with a collapse at 10.

**λ × γ response surface.** Jaccard with the default configuration ranges 0.34 to
1.00 across the grid. Moving γ from 0.5 to 0.25 or 0.75 changes ~28 % of the
selection. The default sits on a slope, not a plateau — and these weights were
never tuned.

## The outcome measurement that overturns part of the analysis above

Everything up to here measured *selection agreement* — how much two strategies'
image sets overlap, and how far a component's ranking correlates with another's.
Those are the wrong statistics for the question Contribution A actually asks,
which is about **which classes get annotated**. Measuring the objective directly
changes the conclusion.

**Tail lift** = fraction of selected images containing a tail-group class, divided
by that fraction in the pool. 1.00 means no preference. Budget 10, 3 clustering
seeds, 5 pool configurations, tail base rate 0.07:

| strategy | reference | few_on_object | many_on_object | mild_imbalance | more_outliers | **mean** |
|---|---|---|---|---|---|---|
| `v2:rarity_coherence` (gated term alone) | 13.20 | 10.37 | 5.66 | 3.64 | 7.54 | **8.08** |
| `v2:coherence` | 12.26 | 10.37 | 5.19 | 3.64 | 4.71 | 7.23 |
| `v2:rarity` | 4.24 | 0.00 | 5.66 | 8.59 | 9.43 | 5.58 |
| `v2:uncertainty` | 2.83 | 4.24 | 1.41 | 0.00 | 4.24 | 2.54 |
| **`v2:full`** | 3.30 | 6.60 | 0.94 | 0.00 | 0.47 | **2.26** |
| `v2:random` | 0.94 | 0.94 | 0.94 | 0.99 | 0.94 | 0.95 |
| **`v2:full_no_coherence`** | 1.41 | 0.47 | 0.00 | 0.33 | 0.94 | **0.63** |
| `v2:novelty` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |

Four results, each stable across all five configurations:

1. **The gated rarity × coherence term is a strong mechanism: 8.08× tail lift**,
   above 3.6× in every configuration. It is the best tail-selector of any variant
   tested.
2. **The composite `full` score retains only 28 % of it** (2.26 vs 8.08). The
   `α·uncertainty + β·novelty` terms pull against the gate — consistent with the
   signed correlations, which are **negative in 7/7 configurations**
   (uncertainty↔gated −0.32, uncertainty↔novelty −0.32). `full` is a tug-of-war,
   not a sum.
3. **Without the gate the composite has no tail preference at all: 0.63, below the
   pool base rate.** Inside `full`, the gate is worth a **3.58× tail lift** and is
   the only reason `full` prefers the tail whatsoever.
4. **Novelty scores 0.00 tail lift in every single configuration** — it never
   selects a tail image, and it covers only 2.0 distinct classes against `full`'s
   7.0. It is carrying weight β = 0.2 while working directly against the stated
   objective.

### An interpretive correction to Phase 1

I reported rarity↔gated redundancy of 0.982 and read it as evidence that the gate
does little. **That reading was wrong.** A rank correlation of 0.982 coexists with
an outcome difference of 8.08 versus 0.63 in tail lift. The gate does not reorder
the rarity ranking much — it reorders it *exactly where it matters*, at the top,
which is the only region a budget-10 selection ever sees. Rank correlation over
3,960 proposals is simply not the right statistic for a decision that depends on
the top 30. The Phase 2 finding stands as measured (29 % selection difference,
S/N 0.93), but it too must be read as "29 % of the selection, and that 29 % is
where all of the long-tail behaviour lives".

**Caveats that keep this from being a clean win.** `v2:coherence` and
`v2:rarity_coherence` achieve high tail lift partly by chasing the tail
exclusively: 4.2 distinct classes covered versus `full`'s 7.0, and 0.00 head
coverage. For OWOD, known-class performance must also be maintained, so maximum
tail lift is not automatically the best strategy — that trade-off can only be
settled by retraining. And the novelty result is the most simulator-dependent
number in this report: the simulated reference set is drawn around the background
centre, whereas the real reference set is the top-20 unknown-scoring queries of
Task-1 images. The audit flagged novelty's reference semantics as muddy (S8) on
independent grounds, so the two lines of evidence agree, but this number in
particular needs the real-data check.

## The two decisive falsification tests

### Test 1 — Is the gate distinguishable from an arbitrary perturbation?

Coherence was replaced by a random permutation of itself: identical marginal
distribution, every relationship to the embedding structure destroyed. 20 shuffles
per seed, 5 seeds.

| | selection difference vs ungated |
|---|---|
| real gate | 0.268 |
| shuffled gate | 0.104 ± 0.091 |
| z | **+1.82** |

The real gate moves selection about 2.6× more than a structure-destroying shuffle,
but at z = 1.82 that is **suggestive, not established** at 5 seeds. Note also that
a *shuffled* gate still changes 10 % of the selection — a direct consequence of the
razor-thin selection boundary (the 10th-to-11th image gap is 0.9 % of the score
range). Part of what looks like a mechanism is a tightly packed ranking being
jostled.

### Test 2 — Does the gate move selection toward the tail, as hypothesised?

| seed | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| ungated tail coverage | 0.00 | 0.20 | 0.10 | 0.00 | 0.20 | |
| gated tail coverage | 0.20 | 0.30 | 0.20 | 0.00 | 0.30 | |
| Δ | +0.20 | +0.10 | +0.10 | 0.00 | +0.10 | **+0.100 ± 0.071** |

**The gate works in exactly the direction the hypothesis predicts**, and it never
works against it: 4 of 5 seeds positive, 1 tie, 0 negative, with medium-class
coverage falling correspondingly (0.70 → 0.60). A sign test gives p ≈ 0.06
one-sided — again suggestive, not significant, at 5 seeds.

This is the strongest evidence *for* Contribution A in the whole report, and it is
why the verdict below is not "abandon".

---

## Phase 4 — What experiment schedule do these effects justify?

What is measurable without a GPU is **selection**. What the thesis needs is a
**detector-metric** difference, which requires training. The selection numbers
bound it: two strategies that select 71 % of the same images cannot produce a
large metric difference.

Seeds per arm for 80 % power at α = 0.05 (two-sided, normal approximation, no
multiplicity correction):

| assumed metric sd / seed | effect 0.002 | 0.005 | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|---|
| 0.005 | 99 | 16 | 4 | 1 | 1 |
| 0.010 | 393 | 63 | 16 | 4 | 1 |
| 0.020 | 1570 | 252 | 63 | 16 | 3 |
| 0.030 | 3532 | 566 | 142 | 36 | 6 |

On a 200-image evaluation split a per-seed sd of 0.01–0.02 in known mAP is the
realistic band. A 1-point (0.01) mAP effect then needs **16–63 seeds per arm**.
The pilot ran **one**.

The recommended schedule follows from the measured S/N, not from a budget guess:

| | recommendation | why |
|---|---|---|
| **Do not run** | the 4-variant × 3-round × 1-seed retraining campaign again | `full_p05` has a 0.061 selection effect; the comparison is undefined |
| **Stage 1** (no GPU) | offline selection, 5+ seeds, all 12 ablations, on **real** exported proposals | reproduces this whole report at the cost of two `predict` calls |
| **Stage 2** (cheap GPU) | retrain only `random`, `v2:uncertainty`, `v2:full`, `v2:full_no_coherence` | the only contrasts with S/N > 1 that also differ in mechanism |
| **Seeds** | **≥ 5**, budget **10–20**, rounds 3 | 5 seeds resolves a 0.02 metric effect at sd 0.01; budget above 20 weakens the gate |
| **Report** | selection-level results as the primary evidence, metrics as secondary | the selection evidence is well-powered; the metric evidence will not be |

---

## Phase 5 — Answers

### Does the current Contribution A appear scientifically defensible?

**The mechanism is defensible. The composite score, as currently weighted, is
not.** Three claims, separated:

*"Uncertainty and distribution-awareness carry independent information"* —
**defensible.** ρ = 0.32 with rarity, 0.03 with coherence; MI 0.05 and 0.008;
removals change 76–80 % of the selection at S/N ≈ 2.4–2.5, stable across all 7
configurations.

*"Gating rarity by local coherence improves selection for long-tail categories"* —
**supported, and more strongly than the selection statistics suggested.** The
gated term alone achieves **8.08× tail lift**, above 3.6× in all 5 configurations
tested, and is the best tail-selector of any variant. Inside `full`, the gate is
worth **3.58×** and is the sole reason `full` prefers the tail at all. Its
direction is consistent on 5/5 seeds in the coverage test, and it beats a
structure-destroying shuffle at z = +1.82.

*"The current `full` score is a good way to combine these"* — **not defensible.**
`full` retains only 28 % of its own gated term's tail lift, because its components
anti-correlate (negative in 7/7 configurations) and the weights were never tuned.
`v2:novelty` carries β = 0.2 while achieving **0.00 tail lift in every
configuration**. And the gate's contribution is throttled to a 29 % selection
effect that sits below the acquisition's own clustering noise (0.317), with a 7×
range across an untuned cluster count.

The honest position: **the idea works; the implementation dilutes it by roughly
3.5× and then buries what remains in pseudo-labelling noise.** That is a
composition and protocol problem, not a failure of the hypothesis.

### Which assumptions are supported?

1. Uncertainty carries information independent of the distribution terms.
2. Rarity and coherence measure different things (ρ = 0.007).
3. Rarity, after the rank-normalisation fix, is a graded signal (9.5 % of
   proposals below 0.1, was 82.9 %). **[structural]**
4. The gate's *direction* matches the hypothesis: it trades medium-class for
   tail-class coverage, consistently in sign.
5. `relative_within_cluster` removes the frequency confound the audit found
   (Spearman with cluster size −0.435 → −0.034).
6. `top_k_mean` is the right aggregation (best signal-to-noise, 1.47 vs 1.05 /
   0.41 / 0.52).

### Which remain speculative?

1. **That the gate's effect is large enough to matter.** S/N 0.93; z = 1.82 vs a
   shuffle.
2. **That the effect survives at a useful budget.** It falls monotonically with
   budget over the range tested.
3. **That any of the weights (α, β, γ, p) are near-optimal.** Never tuned; the
   λ×γ surface shows the default sits on a slope.
4. **That coherence's saturated regime (spread 0.042) leaves it anything to do.**
5. **That any of this transfers to real PROB embeddings.** The whole report is
   synthetic. This is the largest single uncertainty.
6. **That the detector metrics respond to a 29 % selection change at all.**
   Untested and untestable without GPU time.

### Which diagnostics justify changing the algorithm?

* rarity ↔ gated redundancy 0.982 / MI 0.815 — the gate barely re-ranks
* clustering noise 0.317 > gate signal 0.294 — noise dominates the mechanism
* cluster-count instability: 7× range, non-monotone
* `v2:rarity` clustering noise 0.819 vs `v2:uncertainty` 0.000 — all instability
  enters through pseudo-labelling
* the probe below: filtering to unknown-predicted proposals raises rarity rank
  stability from 0.736 to 0.991

### Which argue against changing it?

* Test 2: the gate moves tail coverage the right way on 5/5 seeds. A redesign
  risks losing a mechanism that is working, just quietly.
* Test 1: z = +1.82. The gate *is* doing something beyond perturbation; two more
  seeds might make that conventional.
* Rarity and coherence are genuinely independent (ρ = 0.007), so the premise is
  sound; it is the *combination and the noise floor* that are weak, not the idea.
* Neighbour count is stable — the coherence estimator itself is not fragile.
* **No real-data measurement exists yet.** Redesigning on synthetic evidence alone
  would be the same mistake as the original pilot, in the opposite direction.

### Would unknown-filtered clustering strengthen the work or invalidate the pilot?

Measured, without implementing it (`probe_filtering.json`):

| | all proposals (current) | unknown-predicted only |
|---|---|---|
| proposals | 3,960 | 945 (23.9 %) |
| on an object | 30 % | **89.5 %** |
| rarity fidelity to true class size | **0.849** | 0.711 |
| rarity rank stability across seeds | 0.736 | **0.991** |

The filter discards 97.9 % of background proposals while keeping 54–70 % of every
task class **including the tail** (53.8 %, 60.0 %, 70.0 % for the three smallest).

**It would strengthen the work.** It attacks the dominant problem: clustering noise
(0.317) currently exceeds the gate signal (0.294), and the filter raises rarity
rank stability from 0.736 to 0.991 — a nearly deterministic acquisition. The price
is a 0.139 drop in rarity fidelity, which is the wrong thing to optimise while the
binding constraint is noise.

**And it would invalidate direct comparison with the pilot** — it changes what
rarity is computed over, so `v2:full` before and after are different estimators.
That is not an argument against doing it; it is an argument about *how*.

### Recommendation

**Do not change the algorithm yet — but for a different reason than I expected.**
The evidence now says the mechanism is strong and the *weights* are wrong. That
matters because a weight change is a configuration edit, not a redesign, and it is
cheap to test. The order should be:

1. **Run Stage 1 on real proposals first.** Two `predict` calls, no training,
   `daowod-run diagnose --seeds 0 1 2 3 4`. Three numbers decide everything that
   follows: the gated term's tail lift, `v2:novelty`'s tail lift, and the
   clustering noise. Every quantitative claim in this report is synthetic, and the
   novelty result in particular depends on how the reference set is built — which
   the simulator gets wrong by construction.
2. **Then run a weight sweep, not an algorithm change.** The λ×γ surface and the
   tail-lift table together say the interesting region is γ ≫ α, β — plausibly
   β = 0 given novelty's 0.00 tail lift. That is `daowod-run diagnose` with
   different weight vectors, expressible entirely as new `StrategySpec` presets,
   and it costs no GPU time. Adding `v2:full_no_novelty` and
   `v2:rarity_coherence` to the retraining set is likely worth more than any new
   component.
3. **Only then consider unknown-filtered clustering, and add it as `v3:*`.** It
   remains well-motivated (rarity rank stability 0.736 → 0.991) but it is now the
   *second* problem, not the first: dilution costs 3.5× of the tail lift, noise
   costs less. Keep `v1:*` and `v2:*` byte-identical and report v3 alongside, so
   the pilot comparison survives and the improvement is measured against it.
4. **Reframe the thesis claim around the measured mechanism.** The defensible
   story is now stronger than "the framework plus a negative result": *a
   coherence-gated rarity term selects long-tail images at ~8× the base rate, and
   a diagnostic framework that shows why the naive composite recovers only a
   quarter of that.* Both halves are supported by measurement.
5. **If only one thing is done:** drop `full_p05` from every future protocol. Its
   selection effect is 0.061 — under a fifth of the clustering noise — so the
   comparison is undefined, and it consumed a quarter of the pilot's GPU budget.

---

## Artifacts

`analysis/probes/` holds the three targeted probes:
`falsification_tests.py` (shuffle test, tail-coverage test),
`probe_unknown_filtering.py` (the v3 decision), `probe_tail_coverage.py`
(tail lift per strategy).

`analysis/validate_contribution_a.py` writes, per run:
`phase1_distributions.csv`, `phase1_correlations.csv`,
`phase1_unique_information.csv`, `phase2_pairwise.csv`,
`phase2_component_effects.csv`, `phase2_clustering_noise.csv`,
`phase3_surfaces.csv`, `phase3_sweeps.csv`, `phase4_power_nomogram.csv`,
`validation_summary.json`, and four figures in PNG and PDF
(`figure1_component_redundancy`, `figure2_ablation_overlap`,
`figure3_response_surfaces`, `figure4_hyperparameter_stability`).

Reproduce with:

```bash
python analysis/validate_contribution_a.py outputs/validation   # ~5 min
```
