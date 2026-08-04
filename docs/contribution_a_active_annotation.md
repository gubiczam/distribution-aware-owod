# Contribution A — offline active annotation over real PROB proposals

This document describes the experiment implemented by
`notebooks/contribution_a_active_annotation.ipynb` and the modules behind it: what
is measured, which parts of the design are forced by measurements on real data,
and what the results do and do not support.

> **Result status.** The first full run found that the coherence gate did **not**
> improve tail discovery, reproducibly across all three severities.
> [`contribution_a_failure_analysis.md`](contribution_a_failure_analysis.md)
> localises that to specific components and ranks the follow-up experiments;
> [`contribution_a_revealed_results.md`](contribution_a_revealed_results.md)
> reports the eleven-arm follow-up. Read those before interpreting any number
> here as support for the hypothesis.

## The question

Under a long-tailed unknown-class distribution, does feeding the *estimated* class
distribution back into the annotation decision find more rare unknown objects per
oracle call than uncertainty- or diversity-driven selection?

The acquisition score is

```text
s(x) = alpha * uncertainty(x) + beta * novelty(x) + gamma * rarity(x) * coherence(x)**p
```

`coherence` multiplies **only** the rarity term. An isolated proposal keeps its
uncertainty and novelty but loses the rarity bonus, which is the mechanism that is
supposed to stop the strategy spending its budget on isolated false positives that
merely *look* rare. The gate is a product, not a third additive term; the ablation
grid contrasts it against exactly that weaker alternative at equal total gamma.

## What is measured, and what is not

Measured: the quality of the **annotation set** as a function of region-level
annotation cost — unknown / head / medium / tail discovery recall over distinct
ground-truth objects, distinct classes discovered, annotation precision,
background selection rate, isolated-outlier selection rate, embedding diversity,
and the budget needed to reach a fixed tail level.

Not measured: detector performance. No PROB retraining happens, so no known-mAP,
U-Recall, WI or A-OSE number is claimed. The official PROB evaluator remains the
only source for those; `selected_proposals.csv` is the hand-off point for a
downstream retraining experiment.

## Pipeline

```text
preflight
  -> disjoint reference / pilot / evaluation image splits
  -> cached PROB inference (daowod_prob_bridge.py predict)
  -> candidate proposals            (PROB outputs only)
  -> region-level oracle            (VOC XML, IoU 0.5)
  -> controlled long-tail pools     (validated distinct)
  -> leakage proof
  -> pilot hyperparameter choice    (disjoint pool)
  -> runtime projection & downscaling
  -> acquisition strategies x severities x seeds
  -> metrics, plots, CSVs, markdown summary, ZIP
```

`daowod.pipeline.run_pipeline` runs all of it; the notebook is a driver.

## Modules

| Module | Responsibility |
|---|---|
| `daowod/candidates.py` | Candidate-pool construction from a raw export. PROB outputs only. |
| `daowod/oracle.py` | Region-level ground-truth matching, PROB naming and box conventions. |
| `daowod/longtail.py` | Controlled severities, retention profiles, reachable denominators. |
| `daowod/active.py` | Multi-round proposal-level acquisition loop and feedback. |
| `daowod/discovery.py` | Discovery, efficiency, robustness, diversity metrics and AUCs. |
| `daowod/annotation_study.py` | The study matrix, ablation grid, pilot selection, leakage check. |
| `daowod/modes.py` | `DEBUG` / `FAST` / `MAIN` as data. |
| `daowod/runtime.py` | Pilot timing, runtime projection, budget-driven downscaling. |
| `daowod/preflight.py` | Environment, GPU, PROB checkout, dataset, checkpoint validation. |
| `daowod/export_cache.py` | Cached, resumable, content-keyed detector inference. |
| `daowod/audit.py` | Component diagnostics: probe ceilings, neighbourhood premise, estimator quality. |
| `daowod/revealed.py` | Label-anchored rarity and support, from the regions the oracle confirmed. |
| `daowod/plots.py` | Publication figures. |
| `daowod/reporting.py` | CSVs, contrasts, markdown summary, ZIP, artifact verification. |
| `daowod/pipeline.py` | Stage sequencing and resume. |

Scoring itself is unchanged: everything goes through `daowod.scoring.score_pool`,
the repository's single canonical scorer.

## Ground-truth discipline

Annotations are read in exactly two places:

1. **Protocol** — building the long-tail evaluation pool. The oracle's class labels
   decide which proposals exist; the strategies then see only PROB outputs for the
   survivors. This is the same licence the long-tail literature uses to build
   LT-CIFAR or LVIS-style splits.
2. **Oracle** — revealing the true class of a proposal *after* it has been
   selected. That is the definition of active learning, not leakage.

Four checks enforce the boundary, and the run stops if any fails:

- `discovery.assert_selection_is_ground_truth_free` re-derives every acquisition
  score from its recorded components. An unrecorded oracle term breaks the
  identity. This is the strong check — it constrains arithmetic, not names.
- `active.score_round` is verified by introspection to accept no oracle argument.
- `diagnostics.assert_no_ground_truth` runs against the actual acquisition records.
- Scoring is re-run at a fixed seed and required to be bit-identical.

Results are written to `leakage_report.json` and into the summary.

## Design decisions forced by measurements on real data

These are not preferences; each one was measured on real S-OWODB Task-1 exports.

**Candidate ranking is objectness, not the unknown score.** On the 500-image
export, per-image top-20 by objectness retains 51.9 % of true-unknown proposals at
a 39.4 % on-object rate, versus 43.8 % / 26.5 % for the unknown score. Objectness
also has the higher AUC for "sits on an object" (0.879 vs 0.711).

**The pool must be filtered at all.** A raw export is roughly 85 % background.
Restricting to object-like proposals raised pseudo-class rarity rank stability from
0.736 to 0.991.

**Denominators come from the pool, not the dataset.** The annotations of the
500-image export hold 50 unknown classes; PROB's proposals reach 22 of them, and
the ones it misses are the rarest. Grouping over annotation frequency would leave
the tail group with 2 objects — too few to resolve a curve. Grouping over
*reachable* objects keeps all three groups populated.

**Two retention profiles are needed, because one cannot move the imbalance in both
directions.** The natural reachable distribution is already extreme (on 3 500
images: head class 73 objects, many classes at 1). An absolute exponential target
therefore sits *above* the natural count for most middle and tail classes, so
`min(target, available)` keeps everything and a "severe" setting silently
reproduces "natural" — measured head:tail 15.64 versus 15.44, a 1 % gap. The
`relative` profile scales each class by its own count and does sharpen. The
`absolute` profile with a head cap is the only way to flatten. Both are implemented
and each severity records which it used.

**Small pools cannot express a sharpening axis at all.** On 500 images the
reachable tail group holds 8 objects across 7 classes — already at the
one-object-per-class floor, so nothing can be removed from it. `DEBUG` and `FAST`
therefore use the flatten-only axis
(`longtail.FLATTENING_IMBALANCE_SETTINGS`), and `validate_settings_distinct` fails
loudly rather than letting a run report two names for one regime.

**Novelty had to be blocked over candidate rows.** `candidates @ references.T` at
70 000 x 20 000 allocates 11.2 GB and is killed on a Colab CPU runtime; the blocked
form is bounded at 128 MB and measured 22x faster (47 s to 2.1 s) with identical
rank order.

## Measured pool sizes

Real S-OWODB / OWDETR Task-1 exports, `per_image_limit = 20`:

| Images | Candidate proposals | Reachable unknown objects | Classes | Tail objects |
|---:|---:|---:|---:|---:|
| 500 | 10 000 | 104 | 22 | 8 |
| 2 400 | 48 000 | 364 | 38 | 26 |
| 3 500 | 70 000 | 508 | 44 | 25 |

The tail denominator is the limiting quantity, which is why `MAIN` infers 4 000
images and why the summary's limitations section names the tail count explicitly
whenever it is below 20.

## Execution modes

| Mode | Eval / pilot / reference images | Per-image limit | Budgets | Rounds | Seeds | Arms | Severity axis | Reportable |
|---|---|---:|---|---:|---:|---:|---|---|
| `DEBUG` | 200 / 60 / 150 | 12 | 25–100 | 2 | 2 | 5 | flatten-only | no |
| `FAST` | 500 / 150 / 350 | 20 | 50–400 | 4 | 2 | 5 | flatten-only | no |
| `MAIN` | 2 400 / 600 / 1 000 | 20 | 100–2 000 | 5 | 3 | 5 | flatten + sharpen | yes |
| `MAINREVEALED` | 2 400 / 600 / 1 000 | 20 | 100–2 000 | 5 | 3 | 11 | flatten + sharpen | yes |

`MAINREVEALED` is the follow-up A/B: the same protocol with the free
informativeness-prior control and the label-anchored distribution term running
beside the untouched baseline, so every arm shares one pool, one severity axis, one
seed set and one budget grid.

The three image pools are disjoint by construction (`export_cache.split_disjoint`):
the pilot chooses the coherence definition, the evaluation pool produces the
reported numbers, and the reference pool is the novelty bank — a bank overlapping
the pool would make novelty partly self-referential.

## Runtime control

The run measures rather than assumes:

1. A 25-image probe export gives detector seconds-per-image **before** the bulk
   export, so reducing the image count can still save GPU time.
2. One real acquisition campaign on the real pool gives seconds-per-cell.
3. The projection is compared against the budget (`RUNTIME_BUDGET_HOURS`).

If the projection does not fit, the *pool* shrinks — fewer evaluation images, then
a smaller per-image candidate limit, then the ablation grid — and never the seeds,
strategies or severities, because those are what the experiment is. If it still
does not fit, the run raises and asks for a smaller mode explicitly. Every
reduction is recorded in `runtime_plan.json` and in the summary's limitations.

## Resume

`export_cache` keys each chunk on the bridge settings, the checkpoint digest and
the exact image IDs, so a reused chunk cannot come from a different configuration.
The study matrix caches one severity at a time under `output_dir/state`. After a
dropped session, re-run the notebook: finished chunks and finished severities are
reused, and the test suite asserts that a resumed run reproduces the cached numbers
exactly.

## Outputs

| File | Contents |
|---|---|
| `budget_curves.csv` | Every metric per (severity, strategy, seed, budget). |
| `per_strategy/*.csv` | The same records split one file per strategy (curve, AUC, selected). |
| `budget_curves_aggregated.csv` | Mean and sample sd over seeds; sd is NaN at n=1. |
| `strategy_auc.csv` | Normalised AUCs and final-budget values per campaign. |
| `strategy_summary.csv` | Tail AUC mean/sd per (severity, strategy). |
| `headline_contrasts.csv` | Gated minus each weaker rung, paired by seed. |
| `cost_to_target.csv` | Budget to reach a fixed tail recall; misses reported as misses. |
| `selected_proposals.csv` | Every annotated region with its post-hoc oracle verdict. |
| `component_distributions.csv` | Component values per oracle stratum. |
| `gate_suppression.csv` | The gate counterfactual on the same top-K. |
| `long_tail_pools.csv`, `class_frequency.csv`, `severity_report.csv` | The protocol. |
| `ablations.csv`, `pilot_ablation.csv` | Gate form x coherence definition; pilot grid. |
| `arm_comparison_unknown.csv`, `arm_comparison_tail.csv` | Every arm against the baseline, paired by seed. |
| `discovery_counts.csv` | Absolute discovered-object counts, so a recall of 0.038 is legible as one object. |
| `preflight.csv`, `run_manifest.json`, `runtime_plan.json`, `leakage_report.json` | Provenance. |
| `figure_*.png` / `.pdf` | Eleven figures, publication resolution. |
| `research_summary.md` | Headline tables, mechanism evidence, stated verdicts, limitations. |
| `daowod_contribution_a_<mode>.zip` | Everything above, excluding the proposal cache. |

## How to run

See the notebook's own instructions. In short: open
`notebooks/contribution_a_active_annotation.ipynb` in Colab on a T4, edit only the
configuration cell, run `DEBUG` first, then `MAIN`.

To run the offline study head-less against an existing export:

```python
from daowod.pipeline import PipelineConfig, run_pipeline

result = run_pipeline(
    PipelineConfig(
        mode="FAST",
        data_root="/path/to/owod_stage",
        existing_export="/path/to/proposals.npz",
        output_dir="outputs/contribution_a",
        cache_dir="outputs/contribution_a/cache",
        require_gpu=False,
    )
)
```

## Limitations

- Annotation-set quality only; no detector retraining, so no official OWOD metric.
- Pseudo-classes come from k-means over decoder embeddings, so rarity is an
  estimate. The ablation grid includes the pseudo-label-free `radius_core`
  coherence definition precisely because clustering instability is a known
  confound.
- Recall denominators are pool-reachable objects; unreachable objects are excluded
  from both numerator and denominator, and the count used is in every row.
- The tail denominator is small even at `MAIN` scale (26 objects on 2 400 images),
  so tail recall moves in steps of about 4 %. The summary says so whenever the
  count is below 20.
- The severity axis is bounded by the data. Sharpening beyond one object per tail
  class is impossible; `maximum_expressible_ratio` and `imbalance_ratio_saturated`
  are reported per severity.
