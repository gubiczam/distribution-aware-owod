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

Implemented acquisition strategies:

- `random`
- `uncertainty`
- `uncertainty_novelty`
- `rarity`
- `rarity_coherence`
- `ungated_full`
- `full`

## Pipeline

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
configs/experiment.yaml      Editable experiment configuration
notebooks/contribution_a.ipynb Thin notebook entry point
src/daowod/acquisition.py    Contribution-A scoring logic
src/daowod/config.py         YAML configuration loading
src/daowod/dataset.py        VOC image IDs, annotations, pools
src/daowod/prob_adapter.py   Explicit subprocess boundary to PROB
src/daowod/experiment.py     Multi-seed active-learning orchestration
src/daowod/metrics.py        Head/medium/tail unknown recall diagnostics
tests/test_smoke.py          Small correctness smoke suite
```

## Current Status

Implemented in this repository:

- deterministic labelled/pool state;
- image-level long-tail pool construction without partial annotations;
- proposal-level uncertainty, novelty, rarity, coherence, and scoring;
- deterministic fixed-budget image selection;
- ingestion of official PROB metrics JSON;
- grouped unknown recall diagnostics.

Still external to this repository:

- real PROB training;
- real PROB evaluation;
- proposal export from an installed PROB checkout and checkpoint.

The official PROB evaluator remains the source for known mAP, standard U-Recall,
Wilderness Impact, and A-OSE. This repository only adds grouped diagnostic recall.

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
