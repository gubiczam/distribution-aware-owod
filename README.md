# Distribution-Aware OWOD

Minimal research code for Contribution A of a BME TDK project on distribution-aware
active annotation for Open-World Object Detection (OWOD).

The project studies fixed-budget image selection when unknown classes follow a long-tail
distribution. Standard active-learning scores can over-select isolated unknown-looking
outliers. This repository adds a rarity term that is useful only when the proposal is
locally coherent with nearby proposals.

## Contribution A

For each candidate proposal, the acquisition score is

```text
score = alpha * uncertainty
      + beta * novelty
      + gamma * rarity * coherence**p
```

`coherence` gates only the rarity contribution. An isolated proposal may still be
uncertain or novel, but its low local coherence suppresses the rarity bonus.

Acquisition arms, all resolved through the one canonical scorer
(`daowod.scoring.STRATEGY_REGISTRY`, `v2:` names):

| arm | `U(x)` | distribution term |
|---|---|---|
| `random` | — | — |
| `uncertainty` | posterior entropy | — |
| `uncertainty_novelty` | posterior entropy | — |
| `full_no_coherence` | posterior entropy | cluster rarity, ungated |
| `full` | posterior entropy | cluster rarity x pool density **(baseline)** |
| `objectness_area_prior` | objectness x box scale | — **(free control)** |
| `prior_full` | objectness x box scale | cluster rarity x pool density |
| `prior_revealed_full` | objectness x box scale | revealed rarity x revealed support |
| `revealed_support_only` | posterior entropy | revealed support only |
| `revealed_no_gate` | posterior entropy | revealed rarity, ungated |
| `revealed_full` | posterior entropy | revealed rarity x revealed support |

Plus single-component and ablation variants (`novelty`, `rarity`, `coherence`,
`rarity_plus_coherence`, `proposal_formula`, the `full_no_*` family) and the
version-1 specs that reproduce pre-audit numbers bit for bit.

## Two experiments

> **Where the science stands.** The coherence gate, as specified, does **not**
> improve tail discovery on real S-OWODB Task-1 proposals. Three experiments, all
> frozen and reproducible:
>
> 1. [`contribution_a_failure_analysis.md`](docs/contribution_a_failure_analysis.md)
>    — the component audit. Every term of the baseline score selects *worse than
>    random* in the top 4 % of the ranking, while a one-line `objectness × box scale`
>    prior finds 1.9× more unknown objects.
> 2. [`contribution_a_revealed_results.md`](docs/contribution_a_revealed_results.md)
>    — the eleven-arm follow-up. Anchoring the distribution term on oracle-revealed
>    labels halves the damage it does but does not make it profitable.
> 3. [`e4_representation_results.md`](docs/e4_representation_results.md) — is the
>    failure the embedding or the formulation? Re-embedding the same boxes with DINO
>    ResNet-50 moves the median unknown region's nearest same-class sibling from the
>    **202nd** neighbour to the **6th**, so the decoder embedding really does destroy
>    semantic neighbourhoods — yet background stays the most locally homogeneous
>    stratum in every space, so the gate's premise still fails.
>
> The baseline is retained unchanged in all of them.

**Region-level annotation study (the main deliverable).** An offline active
annotation simulation over real PROB proposals: every strategy spends the same
region-level oracle budget on the same candidate pool, and the metrics describe the
resulting *annotation set*. One cached export supports the whole matrix, so the loop
never touches a GPU. Runs end to end from
`notebooks/contribution_a_active_annotation.ipynb` on a single T4 in about four to
five hours. See [`docs/contribution_a_active_annotation.md`](docs/contribution_a_active_annotation.md).

```text
preflight -> disjoint reference/pilot/evaluation splits -> cached PROB inference
-> candidate proposals -> region-level oracle -> controlled long-tail pools
-> leakage proof -> pilot hyperparameter choice -> runtime projection
-> acquisition strategies -> metrics -> plots -> CSVs -> summary -> ZIP
```

**Image-level campaign with retraining.** The older loop that spends an *image*
budget and calls PROB to retrain between rounds, ending in official OWOD metrics.

```text
PROB proposals
-> uncertainty
-> novelty against labelled reference embeddings
-> pseudo-label assignment by prediction or clustering
-> rarity from estimated pseudo-class frequency
-> local coherence
-> proposal score
-> top-k image aggregation
-> fixed-budget image selection
-> annotation
-> PROB retraining
-> OWOD evaluation
```

## Repository Structure

```text
configs/experiment.yaml         Editable experiment configuration

notebooks/contribution_a_active_annotation.ipynb
                                Main executable Colab notebook (annotation study)
analysis/build_contribution_a_notebook.py
                                Generator for that notebook
analysis/audit_coherence_failure.py
                                Component audit behind the failure analysis
analysis/e4_required_rows.py    Which export rows E4 must re-embed
analysis/extract_region_embeddings.py
                                Re-embed the boxes with DINO / ImageNet (torch env)
analysis/experiment_e4_representations.py
                                E4 Phases 3, 4, 6: geometry and figures
analysis/run_e4_active_learning.py
                                E4 Phase 5: same strategies, different space

src/daowod/scoring.py           The one canonical acquisition scorer and registry
src/daowod/components.py        Uncertainty, novelty, rarity, coherence
src/daowod/normalisation.py     Component normalisation

src/daowod/candidates.py        Candidate-pool construction (PROB outputs only)
src/daowod/oracle.py            Region-level ground-truth matching
src/daowod/longtail.py          Controlled long-tail severities and denominators
src/daowod/active.py            Multi-round region-level acquisition loop
src/daowod/discovery.py         Discovery / efficiency / robustness metrics
src/daowod/annotation_study.py  Study matrix, ablations, pilot selection
src/daowod/modes.py             DEBUG / FAST / MAIN execution modes
src/daowod/runtime.py           Runtime projection and budget-driven downscaling
src/daowod/preflight.py         Environment, GPU, dataset, checkpoint validation
src/daowod/export_cache.py      Cached, resumable PROB inference
src/daowod/audit.py             Component diagnostics for a negative result
src/daowod/revealed.py          Label-anchored rarity and support
src/daowod/representations.py   E4 feature spaces: PROB, crop encoders, transforms
src/daowod/geometry.py          E4 geometry metrics for a feature space
src/daowod/representation_plots.py
                                E4 projections and comparison figures
src/daowod/plots.py             Publication figures
src/daowod/reporting.py         CSVs, summary, ZIP, artifact verification
src/daowod/pipeline.py          Stage sequencing and resume

src/daowod/acquisition.py       Legacy scoring facade (kept for reproducibility)
src/daowod/config.py            YAML configuration loading
src/daowod/dataset.py           VOC image IDs, annotations, pools
src/daowod/prob_adapter.py      Explicit subprocess boundary to PROB
src/daowod/experiment.py        Multi-seed image-level campaign orchestration
src/daowod/metrics.py           Head/medium/tail unknown recall diagnostics

tests/test_active_annotation.py Unit tests for the annotation study
tests/test_revealed_distribution.py
                                Tests for label-anchored estimation and the prior
tests/test_representation_geometry.py
                                Tests for E4 feature spaces and geometry metrics
tests/test_study_pipeline.py    End-to-end pipeline test on a fabricated export
tests/test_smoke.py             Small correctness smoke suite
```

## Current Status

Implemented in this repository:

- deterministic labelled/pool state;
- image-level long-tail pool construction without partial annotations;
- proposal-level uncertainty, novelty, rarity, coherence, and scoring;
- deterministic fixed-budget image selection;
- ingestion of official PROB metrics JSON;
- grouped unknown recall diagnostics;
- the complete region-level annotation study: candidate pool, region oracle,
  controlled long-tail severities, multi-round acquisition, discovery metrics,
  ablations, plots, reports, and a resumable Colab pipeline.

Still external to this repository:

- real PROB training;
- real PROB evaluation;
- the proposal export itself, which runs inside a PROB checkout through
  `daowod_prob_bridge.py` (the pipeline calls it and caches the result).

The official PROB evaluator remains the source for known mAP, standard U-Recall,
Wilderness Impact, and A-OSE. This repository adds grouped diagnostic recall and
annotation-set quality metrics; it does not claim detector numbers.

## Installation

```bash
python -m pip install --editable ".[dev]"
```

The package supports Python 3.11.

## Checks

```bash
ruff check .
ruff format --check .
pytest
python -m compileall -q src
```

## Colab Notebook

Open `notebooks/contribution_a_active_annotation.ipynb` in Google Colab (T4
runtime) for the region-level annotation study. Edit only the configuration cell,
run `RUN_MODE = "DEBUG"` first (minutes, no GPU needed), then `"MAIN"`. It clones
this repository and the PROB fork, builds the deformable-attention CUDA extension,
validates every precondition, exports proposals once into a resumable cache, runs
the full strategy matrix, and writes CSVs, figures, a markdown research summary and
a ZIP.

Set `RUN_MODE = "MAINREVEALED"` for the eleven-arm follow-up that adds the
free informativeness-prior control and the label-anchored distribution term beside
the baseline.

`notebooks/contribution_a.ipynb` remains the older image-level notebook. It clones
this repository and the official PROB repository, installs DAOWOD, runs the local
checks, and executes detector-independent synthetic validation without requiring
external data.

Optional real PROB integration requires this Google Drive layout:

```text
MyDrive/DAOWOD/
|-- data/
|   `-- OWOD/
|       |-- JPEGImages/
|       |-- Annotations/
|       `-- ImageSets/
|           `-- TOWOD/
`-- checkpoints/
    `-- MOWODB/
        `-- t1.pth
```

When those assets and a compatible PROB attention backend are available, the notebook
can run a one-image proposal export smoke test and score the exported proposal
features. Missing Drive data or checkpoints are reported as `MISSING` or `SKIPPED`,
not as code failures.

## PROB Schemas

The main repository talks to PROB only through `src/daowod/prob_adapter.py`.

Proposal NPZ files must contain:

- `image_ids`
- `confidence`
- `embeddings`

They may also contain:

- `posterior`
- `predicted_labels`
- `boxes`
- `objectness`

Metrics JSON files must contain:

- `known_mAP`
- `U_Recall`
- `WI`
- `A_OSE`

They may also contain:

- `detections_path`, pointing to a JSON file with `ground_truth` and `detections`
  entries for grouped unknown recall diagnostics.

## Configuration

Start from `configs/experiment.yaml`, then replace the dataset paths, PROB checkout
path, checkpoint path, and exact unknown-class list for the OWOD task/protocol being
run. The sample file intentionally does not guess protocol-specific unknown classes.
