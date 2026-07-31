# Phase 1 — Repository Audit

Audit of `distribution-aware-owod` @ `8199d2b` + `PROB` @ `980cf3a` (branch
`feat/daowod-bridge-v2`), plus the five notebooks and the untracked
`notebooks/contribution_a_multiround_prob.ipynb`.

Nothing in the pipeline was modified. Every quantitative claim below was
produced by running the installed package; the commands are reproduced inline so
they can be re-run.

Baseline state: `pytest` 20 passed, `ruff check` clean.

---

## 1. Architecture as built

Three layers, with a deliberately narrow boundary between them.

```
                        ┌────────────────────────────────────────┐
  configs/experiment.yaml (legacy) ── load_config ──► ExperimentConfig
                        └────────────────────────────────────────┘
                                        │  (used only by legacy notebooks)
  notebooks/contribution_a_multiround_prob.ipynb  ← the live experiment driver
        │  builds AcquisitionConfig / AcquisitionWeights directly in Python
        │  (the YAML is bypassed entirely)
        ▼
  daowod.experiment.run_active_round      one AL round, detector-backed
        │
        ├─ daowod.prob_adapter.ProbAdapter ── subprocess ──► PROB/daowod_prob_bridge.py
        │      .predict()   → proposal NPZ  (image_ids, confidence, embeddings,
        │                                    posterior, predicted_labels, boxes, objectness)
        │      .train()     → checkpoint.pth
        │      .evaluate()  → metrics JSON  (known_mAP, U_Recall, WI, A_OSE, …)
        │
        └─ daowod.acquisition.score_proposals
                 compute_uncertainty → compute_novelty → assign_pseudo_labels
                 → compute_rarity → compute_coherence → compute_proposal_scores
                 → aggregate_image_scores (mean of top-k) → select_images (budget)
```

### Data flow of one round

1. `adapter.predict(candidates)` — PROB forward pass over every pool image.
   Per image the bridge keeps the `MAX_PROPOSALS_PER_IMAGE = 20` queries with the
   highest **unknown score** `objectness · sigmoid(logit_unknown)`, where
   `objectness = exp(-(obj_temp/hidden_dim) · pred_obj)`. Verified identical to
   PROB's own `PostProcess` (`models/prob_deformable_detr.py:542,650`), and the
   bridge's `index_select` over `[0..seen) ∪ {num_classes-1}` is exactly the
   complement of PROB's `invalid_cls_logits` (`:607`). **The adapter boundary is
   faithful to PROB — this part is correct.**
2. `adapter.predict(references)` — the same export over Task-1 reference images
   ∪ everything labelled so far.
3. `score_proposals` — five component functions, then a weighted combination.
4. `aggregate_image_scores` — mean of each image's top-`k` proposal scores.
5. `select_images` — sort by image score, take `budget`, tie-break on `str(id)`.
6. `adapter.train(labelled ∪ selected)` — one PROB fine-tuning epoch.
7. `adapter.evaluate(checkpoint)` — official OWOD evaluator.
8. Round manifest + SHA-256 of every artifact; the notebook adds
   `round_identity.json` / `round_record.json` and a validated Drive copy.

The reproducibility scaffolding around steps 6–8 (pinned commits, clean-checkout
validation, digest-verified persistence, resumable rounds, refusal to overwrite a
completed round) is genuinely strong and above the level normally seen in a
thesis repository. **The weaknesses are all in steps 3–4 and in the experimental
design, not in the plumbing.**

### The active-learning loop exists twice

| | `run_active_round` (function) | `ActiveLearningExperiment._run_single` (class) |
|---|---|---|
| Used by | the live notebook | `colab_experiment.ipynb` only |
| Tested | 14 references in the suite | none |
| Strategies accepted | `random`, `rarity_no_coherence`, `full` | whatever the config says |
| Random RNG | `random.Random(f"{seed}:{round}")` | `np.random.default_rng(seed)` |
| Manifest / leakage guards | yes | no |
| Writes `pseudo_label` per proposal | no | yes (`_write_diagnostics`) |
| Calls `grouped_unknown_recall` | no | yes — **the only caller** |

Two implementations of one concept, already diverged in seeding, artifacts and
validation. The unused one owns the only path to the long-tail metrics.

---

## 2. Contribution A: what the code computes

| Component | Implementation | Range / normalisation |
|---|---|---|
| `uncertainty` (`ambiguity`) | `1 − |2c − 1|`, `c` = PROB unknown score | none |
| `uncertainty` (`entropy`, `margin`) | over exported `posterior` | `entropy` ÷ log C |
| `novelty` | `1 − max cos(candidate, reference)` | min-max |
| `pseudo_labels` | `KMeans(k=20)` on L2-normalised embeddings | — |
| `rarity` | `count(pseudo_label)^(−rarity_power)` | min-max |
| `coherence` | `1 / (1 + d_k / median(d_k))`, `d_k` = k-th NN distance | clipped [0,1] |

Combination (`compute_proposal_scores`):

```
uncertainty          → u
uncertainty_novelty  → (α·u + β·n) / (α + β)
rarity               → r
rarity_coherence     → r · cohᵗ
ungated_full         → α·u + β·n + γ·r
full                 → α·u + β·n + γ·(r · cohᵗ)
```

with `α, β, γ, t = ALPHA, BETA, GAMMA, coherence_power = 0.3, 0.2, 0.5, {0.5|1.0}`.

### Mismatch against the written research goal

The proposal states

```
S(x) = U(x) + λ·D(x) + γ·w(ĉ)·coh(x)
```

The implementation is `α·u + β·n + γ·r·cohᵗ`. Three distinct discrepancies:

* **An extra term.** `β·n` (novelty) appears in the code and in no version of
  the proposal formula.
* **A missing term.** The proposal has an *ungated* distribution term `λ·D(x)`
  **in addition to** the gated `γ·w(ĉ)·coh(x)`. No implemented variant has both:
  `full` has only the gated term, `ungated_full` has only the ungated one. The
  formula as written by the proposal is not implemented by any strategy.
* **The proposal is itself ambiguous.** `D(x)` is described as
  "rarity / distribution-awareness" and `w(ĉ)` as "class-frequency weighting" —
  these are the same quantity under two names. Until this is resolved on paper
  the formula cannot be said to be implemented correctly or incorrectly.

`γ·w(ĉ)·coh(x)` with `w(ĉ) = rarity` **does** equal the code's gated term at
`t = 1`. So the code is a superset-plus-omission, not a contradiction. It still
must be reconciled and documented before anything is written up.

---

## 3. Scientific findings, severity-ranked

### S1 — The campaign's headline contrast is numerically tiny, and the design has no power to detect it. **Critical.**

The live protocol compares four variants: `random`, `rarity_no_coherence`,
`full` at `p=0.5`, `full` at `p=1.0`. The last three differ only in the exponent
on the coherence gate. Measured effect of that exponent on how strongly the
score prefers tail proposals, on a simulated 20-class pool with realistic
proposal counts (≥6 proposals per class):

| `p` | mean gated score, head | tail | tail/head |
|---|---|---|---|
| 0.00 | 0.02850 | 0.56086 | 19.68 |
| 0.25 | 0.02392 | 0.46575 | 19.47 |
| 0.50 | 0.02008 | 0.38680 | 19.26 |
| 1.00 | 0.01415 | 0.26682 | 18.85 |
| 2.00 | 0.00703 | 0.12706 | 18.07 |

Removing the gate entirely (`p=0`) versus `p=1` changes the tail preference by
**4 %**. That is the entire designed difference between `rarity_no_coherence`,
`full_p05` and `full_p1`.

Against that, the measurement apparatus is: **one seed**, 3 rounds × 10 images
= 30 labelled images per variant, **one** fine-tuning epoch at `lr = 2e-5` with
`freeze_prob_model=True`, evaluated on a **200-image** custom split. There are no
repeated seeds, therefore no error bars, therefore no defensible claim in either
direction. Whatever ordering the twelve runs produce will be SGD and
split-sampling noise.

This is the finding that most threatens the project, and it is a *design*
problem, not a bug. Section 6 proposes a protocol that fits the same compute.

### S2 — "Uncertainty" is not uncertainty. **Critical.**

`confidence` from the bridge is PROB's unknown-class *detection score*. From a
real Task-1 checkpoint (`PROB/results/trained_checkpoint_smoke/diagnostics.json`):
`pred_obj ∈ [69.4, 467.2]` ⇒ `objectness ∈ [0.093, 0.703]`; class sigmoid scores
mean 0.022. The product, restricted to the kept top-20 queries, lands in
`[0.012, 0.29]` — median 0.024.

`ambiguity = 1 − |2c − 1|` equals `2c` for every `c < 0.5`. Measured over that
distribution: **fraction of proposals with `c > 0.5` = 0.000000**, and
**Spearman(uncertainty, unknown score) = +1.000000**.

So `α·U(x)` is `2α ·` (the unknown score) — a strictly increasing rescaling of a
quantity the pipeline already used to pick which proposals exist at all. It adds
no ranking information and it is not an uncertainty. Note that `posterior` *is*
exported by the bridge, so the genuine `entropy` and `margin` modes are
available and simply unused.

Consequence for the thesis: the term labelled `U(x)` in the score cannot be
described as uncertainty in the write-up as it stands.

### S3 — The long-tail metrics that define the hypothesis are never computed. **Critical.**

`grouped_unknown_recall` (head/medium/tail U-Recall) is gated on
`metrics["detections_path"]`. The bridge's `evaluate()` writes `known_mAP`,
`U_Recall`, `WI`, `A_OSE`, `known_AP50`, `unknown_AP50`, `current_known_AP50`,
`previous_known_AP50`, `official_metrics`, `coco_eval_bbox` — and **no
`detections_path`**. Its only caller is `ActiveLearningExperiment`, which the
live campaign does not use.

The hypothesis is specifically about long-tail categories. There is currently no
head/medium/tail measurement anywhere in the executed pipeline. The ingredients
exist and are never joined: `build_long_tail_pool` writes a per-class
`group` column to `class_stats.csv`, and each round records
`selected_task_class_counts` / `pool_task_class_counts`. Nothing merges them.

### S4 — Rarity degenerates into a near-binary "is a singleton cluster" indicator. **Major.**

`count^(−1)` followed by min-max. On a 20-class pool with a 20:1 imbalance
(cluster sizes 300 … 1):

| cluster size | 300 | 150 | 52 | 25 | 13 | 6 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| rarity | 0.000 | 0.003 | 0.016 | 0.037 | 0.074 | 0.164 | 0.331 | 0.498 | 1.000 |

**97.0 %** of proposals get rarity < 0.1; 0.4 % get > 0.5. The hyperbola plus
min-max means the single smallest cluster defines the scale and everything else
collapses. `rarity` is not a graded frequency signal; it is an outlier flag.
Log-inverse-frequency or rank normalisation fixes this without changing the
concept.

### S5 — Coherence is a non-scale-free density estimate; its interaction with class frequency is regime-dependent and unmeasured. **Major.**

`coherence = 1/(1 + d_k/median(d_k))` is an *absolute* local-density statistic.
In a long-tail pool, density is confounded with class frequency. Simulation with
20 equally-well-formed Gaussian classes differing **only** in sample count:

| regime | mean coherence, head | tail | effect |
|---|---|---|---|
| ≤3 proposals per tail class | 0.502 | 0.141 | tail penalised 3.6× |
| ≥6 proposals per tail class | 0.498 | 0.479 | flat |

Both regimes are bad, differently:

* In the **sparse** regime the gate suppresses exactly the rare-but-genuine
  classes the rarity term exists to promote. It cannot distinguish "isolated
  noise proposal" from "rare well-formed class", because it only sees density.
* In the **dense** regime `coherence ≈ 0.5 ± 0.05` for everything, so
  `r · coh ≈ 0.5 r` and `full` is `ungated_full` with `γ` halved. The
  contribution silently vanishes into a rescaling of one weight.

Which regime the real pool occupies has never been measured. With 20 proposals
kept per image and a `imbalance_ratio=20` pool, the tail classes probably land in
the dense regime — i.e. the most likely outcome of the current campaign is that
`full` and `rarity_no_coherence` are indistinguishable *by construction*, and
that null result would be uninterpretable.

Note also the two flaws in S4 and S5 partly cancel: 1/n rarity over-favours the
tail by ~29×, density coherence penalises it by ~3.6×, net ~8×. The tuned values
of `γ` and `p` are therefore compensating for normalisation artifacts rather than
expressing a modelling choice.

A scale-free alternative measured on the same simulation — a silhouette-style
contrast `(d_other − d_same)/max(·)` — gives head 0.642 / tail 0.828: it does
*not* penalise rarity, while still collapsing for genuinely isolated points. A
neighbourhood-purity variant was also tried and is **worse** (head 1.00 /
tail 0.18) because a singleton class has no same-class neighbours by
construction. The choice needs to be made empirically on real proposals, not
argued from first principles.

### S6 — The components are not on a common scale, so the stated weights do not describe the model. **Major.**

`novelty` and `rarity` are min-maxed; `uncertainty` and `coherence` are not.
Empirically `u ∈ [0.024, 0.58]`, `coherence ≈ 0.5 ± 0.05`, `rarity` spans [0,1]
but with 97 % of mass below 0.1. Writing `α:β:γ = 0.3:0.2:0.5` in the thesis
would misrepresent the actual influence of each term by a large factor. Any
component ablation is also uninterpretable until the components are commensurate.

### S7 — `KMeans(k=C)` does not recover the class partition on long-tail data. **Major.**

Same simulation, `k=20` against 20 true classes:

* true class 0 (n=300) → split across clusters {282, 18}
* true class 1 (n=210) → split across 3 clusters {92, 87, 31}
* true classes 12, 13, 15, 16, 17 → **merged into one cluster**
* `Spearman(pseudo-cluster size, true class size) = +0.80`

KMeans minimises within-cluster variance, so it splits populous clusters and
absorbs sparse ones. "Rarity of the pseudo-class" is therefore only loosely tied
to rarity of the actual class — and the direction of the error is systematically
against the tail. `pseudo_label_source="predicted"` exists and
`predicted_labels` are exported; that path is never exercised in the campaign
and should at minimum be an ablation axis.

### S8 — Novelty's reference distribution is semantically muddy and not comparable across rounds. **Moderate.**

The reference embeddings are the top-20 *unknown-scoring* queries of the Task-1
reference images ∪ everything labelled so far. So "distance from the labelled
distribution" is measured against unknown-looking proposals rather than against
representations of the known classes. Additionally the reference set grows every
round while `novelty` is re-min-maxed each round, so novelty values are not
comparable across rounds — yet they are plotted and averaged across rounds.

### S9 — Non-standard evaluation split, and every round evaluates twice. **Moderate.**

Metrics come from a bespoke 200-image split (100 unknown-bearing + 100
known-only) rather than `owod_all_task_test`. Defensible for a pilot and
correctly disclosed in the notebook, but known mAP on 100 known-only images is
high-variance and not comparable to any published number.

Separately: `PROB_EVAL_EVERY = 1` forces PROB's training loop to run the full
evaluator on that same split at the end of the fine-tuning epoch
(`main_open_world.py:346`), and then the bridge runs `evaluate` again. Every
round pays for two official evaluations; ~12 evaluator runs per campaign are
discarded. (Checked for leakage: `checkpoint.pth` is saved unconditionally at
`main_open_world.py:343-364` with no best-checkpoint selection, so the in-training
evaluation does **not** leak the eval split into model selection. It is waste,
not contamination.)

### S10 — The `random` baseline pays the full proposal-export cost. **Minor.**

`run_active_round` calls `adapter.predict(candidates)` before branching on
strategy, so the random baseline runs a PROB forward pass over the entire pool
and discards it. Harmless scientifically; it is 25 % of the campaign's inference
budget.

---

## 4. Engineering findings

**E1 — Four inconsistent strategy registries, and the config accepts names that
crash.** Verified by execution:

| strategy | `AcquisitionConfig` | `score_proposals` |
|---|---|---|
| `random` | accepted | `ValueError: Unknown strategy: random` |
| `rarity_no_coherence` | accepted | `ValueError: Unknown strategy` |
| `uncertainty`, `uncertainty_novelty`, `rarity`, `rarity_coherence`, `ungated_full`, `full` | accepted | OK |

`ActiveLearningExperiment._run_single` passes the config strategy straight to
`score_proposals`, so a config naming `rarity_no_coherence` — a name the config
validator explicitly allows and the live notebook uses — crashes mid-run in that
code path. Registries: `config.py` 8 names, `acquisition.Strategy` 6,
`run_active_round` 3, `compare_acquisition_strategies` 5. No two agree.

**E2 — The scoring formulas are duplicated in four places and have already
drifted.** `compute_proposal_scores` (`acquisition.py:228`),
`_offline_strategy_scores` (`:332`), the `rarity_bonus` expression in
`experiment.py:229`, and the `rarity_bonus` expression in `acquisition.py:463`.
The drift is real: `compute_proposal_scores("uncertainty_novelty")` divides by
`(α+β)`; `_offline_strategy_scores("uncertainty_novelty")` does not. The offline
comparison and the online loop therefore rank that baseline on different scales.

**E3 — Dead or unreachable code.** No caller outside its own module or the test
suite: `frequency_groups`, `unknown_class_counts`, `read_voc_classes`,
`load_detection_json`, `box_iou`, `grouped_unknown_recall` (unreachable, S3),
`DatasetState` (only used by the unused class), `ActiveLearningExperiment` +
`_prepare_pool` + `_run_single` + `_write_diagnostics` (~200 lines),
`read_image_ids`. `configs/experiment.yaml` and `load_config` are used only by
superseded notebooks — **the live experiment is configured by Python constants in
notebook cell B, and the YAML is inert.**

**E4 — Notebook sprawl.** 5 notebooks, 9,637 source lines; 4 are superseded
(`colab_experiment`, `contribution_a`, `coherence_p05_retraining`, `multiround`)
and all 4 duplicate the same ~600-line Colab bootstrap. The live notebook is
5,395 lines, of which roughly 3,500 are environment verification and ~800 are
the experiment. The verification is good work but it does not belong in the same
file as the science.

**E5 — Quadratic image aggregation.** `aggregate_image_scores` and
`_top_k_indices` do `ids == image_id` per image, O(N·M). Measured: 2,000
proposals 1.8 ms → 8,000 22.7 ms → 16,000 86.6 ms (4× data, 48× time). Fine at
today's 6,000; a one-line `np.unique`/`argsort` grouping removes it.

**E6 — Seed derivation collides.** `seed = seed + round_index` for KMeans
(`experiment.py:221,474`) maps (seed 0, round 1) and (seed 1, round 0) to the
same value. Two different RNG mechanisms are used for the random baseline in the
two loop implementations. Multi-seed runs need a hash-derived per-(seed, round,
strategy) stream.

**E7 — No entry point.** No `[project.scripts]`, no `__main__`, no CLI. The
Phase 9 requirement "one command reproduces an experiment" cannot be met today;
reproduction means running 25 Colab cells in order.

**E8 — Phase 7 is not satisfiable from current artifacts.** `proposal_scores.csv`
holds `image_id, uncertainty, novelty, rarity, coherence, rarity_bonus, score`.
Missing: pseudo-label / cluster id, cluster size, predicted class, GT class,
objectness, box, head/medium/tail group, selected flag. Note the CSV also has no
stable proposal index, so it cannot be joined back to the NPZ rows other than by
implicit row order.

**E9 — Reporting is thinner than required.** 3 figures (learning curves,
component means, overlap heatmap) against the 11 requested; PNG only, no PDF; no
per-group recall, no strategy ranking, no score-distribution plot.

**E10 — `prob_adapter.py` style and memory.** Formatted one-argument-per-line,
inconsistent with every other module. `_run` uses `capture_output=True` for
commands with an 86,400 s timeout, buffering an entire training log in memory and
printing it in full; a day-long run's stdout is held as one string.

---

## 5. Hidden assumptions

1. **`confidence ∈ [0,1]` and is calibrated enough for `1−|2c−1|` to mean
   something.** True for the range, false for the calibration (S2).
2. **Cluster size ≈ class frequency.** False in the direction that matters (S7).
3. **Local density distinguishes noise from rare classes.** It does not; it only
   measures density (S5).
4. **Components are comparable, so the weights mean what they say.** False (S6).
5. **Task-2 introduced classes are a valid proxy for "unknowns".** Reasonable and
   standard for M-OWODB, but it means "unknown recall" is recall on 20 specific
   classes, not on open-world novelty in general.
6. **One fine-tuning epoch on 10 images produces a measurable change in OWOD
   metrics.** Untested and unlikely; `verify_checkpoint_save` only proves *some*
   weights changed, not that anything measurable did.
7. **Ground truth never reaches the acquisition function.** Verified true, and
   correctly documented in `protocol.json` — a genuine strength.
8. **The 200-image eval split is representative.** Unverified; no variance
   estimate exists.

---

## 6. What this implies for Phases 2–11

The repository is in better engineering shape than most thesis code and in worse
scientific shape than its polish suggests. Concretely: the plumbing is
publication-grade, the measurement design is not yet able to support any claim.
Priority order should therefore be **S3 → S1 → S2/S4/S5/S6 → the requested
feature phases**, not the phase order as numbered.

Recommended sequencing (Phase numbers in brackets):

1. **Unify the scoring core** [2, 3, 4]. One generalised score, one registry,
   one place where any formula lives:
   `S = w_u·û + w_n·n̂ + w_r·r̂ + w_g·(r̂ · cohᵖ)` over **rank-normalised**
   components. Every one of the 11 required baselines becomes a weight vector,
   including the proposal's `U + λD + γ·w(ĉ)·coh` (which is `w_u=1, w_r=λ, w_g=γ`)
   — resolving S8 and E1/E2 at once and making Phase 5's ablation mechanical.
2. **Make the long-tail metrics real** [6]. Have the bridge emit
   `detections_path`; join `class_stats.csv` groups into every round's metrics;
   report head/medium/tail recall and tail U-Recall per round. Without this there
   is no Contribution A result at all.
3. **Fix the offline evaluation protocol** [1 new, enables 4, 5, 11]. A
   frozen-checkpoint acquisition-only harness — which
   `compare_acquisition_strategies` already almost is — lets all 11 baselines run
   over many seeds for the cost of one proposal export. Retraining is then spent
   only on the 3–4 variants that show a real selection difference. This is the
   single highest value-per-effort change available and it is what makes S1
   tractable within the existing compute.
4. **Proposal-level CSV with GT join** [7]. Needs a stable proposal index and a
   box↔GT IoU match; this is also what will finally answer whether the coherence
   gate helps or hurts the tail.
5. **Figures, ablation tables, reproducibility CLI, cleanup** [8, 9, 10].
6. **Re-audit against the hypothesis** [11].

### Rough effort

| Work | Estimate |
|---|---|
| Unified scoring core + registry + 11 baselines + tests | 1.5 days |
| `detections_path` + head/medium/tail metrics end to end | 1 day |
| Offline multi-seed acquisition harness | 1 day |
| Proposal-level CSV with GT/cluster/group join | 0.5 day |
| Ablation driver + summary tables | 0.5 day |
| 11 figures, PNG+PDF | 1 day |
| CLI, config consolidation, run manifest | 0.5 day |
| Dead-code removal, notebook split, type hints, logging | 1 day |
| Scientific validation pass + write-up notes | 0.5 day |
| **Total** | **~7.5 days** |

Comfortably inside a two-week sprint, leaving room for the GPU campaign.

---

## 7. Decisions needed before writing code

These change what gets built, so they should not be guessed.

**D1 — Which formula is canonical?** Recommendation: adopt the generalised
four-weight form above and define the proposal's `S` and the current `full` as
two named presets of it. Nothing is lost, both are reproducible, and the thesis
can state one equation.

**D2 — What happens to "uncertainty"?** Recommendation: rename the current mode
to `unknown_score` (honest), add true `entropy` / `margin` over the exported
posterior, and make the uncertainty mode an ablation axis. Do **not** silently
keep calling `1−|2c−1|` uncertainty.

**D3 — Which coherence definition?** Recommendation: keep `density` as
`coherence_mode="density"`, add `silhouette`, and select between them on real
exported proposals rather than by argument. Report the correlation between
coherence and true class frequency as a headline diagnostic either way.

**D4 — Compute allocation.** The current 12-run campaign cannot answer the
research question (S1). Recommendation: offline multi-seed selection over all 11
baselines (cheap, gives the selection-behaviour result with error bars), then
retraining restricted to `random` / `uncertainty` / `full` / `full-without-coherence`
with **≥3 seeds** and a larger per-round budget. Same GPU hours, an order of
magnitude more statistical power.
