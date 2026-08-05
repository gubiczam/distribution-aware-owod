# Distribution-Aware Open-World Object Detection

Research code for a BME TDK project on the effect of **long-tailed unknown-class
distributions** in Open-World Object Detection (OWOD), and on two ways of acting on it.

Existing OWOD methods report a single aggregate U-Recall and known-mAP. The project's
premise is that this hides structure: rare unknown classes are suppressed by frequent
ones, labelling cost is badly distributed, and rare new classes are lost during
incremental update. All three are manifestations of one long-tail problem, and
head/medium/tail-resolved evaluation is the instrument that exposes it.

The contract for everything here is [`docs/proposal.docx`](docs/proposal.docx), read into
[`docs/research_design.md`](docs/research_design.md).

## The two contributions

**A — distribution-aware active annotation.** Under a bounded oracle, which candidate
region should be labelled next? The score is the proposal's equation (1):

```text
s(x) = U(x) + λ·D(x) + γ·w(ĉ(x))·coh(x)          w(c) ∝ 1/n_c
```

`coh` gates **only** the rarity term, so an isolated proposal keeps its uncertainty and
novelty but loses its rarity bonus. The claim under test is that this AND relation
separates a genuinely rare class from an isolated false positive.

> **Where the science stands.** The gate's premise is **falsified** on real S-OWODB
> Task-1 proposals: background is the most locally homogeneous stratum in the pool
> (tail same-label fraction 0.015 against background's 0.888), so a homogeneity-based
> coherence term ranks background highest. A one-line `objectness × box scale` prior
> finds **1.88×** more unknown objects than the full distribution-aware score. This
> holds in all nine feature spaces tested. Read [`docs/results.md`](docs/results.md)
> before interpreting any output — it is a result about a named hypothesis, not a bug.

**B — distribution-aware exemplar allocation.** Replay memory is normally split evenly
per class, which hurts the tail. The rule under test allocates `m_c ∝ n_c^α` with
`Σm_c = M`: `α=0` uniform (the standard), `α=1` size-proportional, `α<0` tail-favouring.

> **Status: allocation core only.** [`src/daowod/memory.py`](src/daowod/memory.py) is
> mathematically complete and unit-tested — exact integer conservation, deterministic
> ties, both granularities. The *research question* — the optimal `α` and how it moves
> with tail severity — needs real incremental model updates and is **not measured**.
> No offline forgetting proxy is provided, because forgetting is a property of a trained
> model, not of a buffer. See `docs/research_design.md` §8.

## Reproduce

```bash
python -m pip install --editable ".[dev]"     # Python 3.11
pytest                                        # 215 tests, no GPU, seconds
```

| Experiment | Command |
|---|---|
| **Contribution A** (the study) | `python experiments/contribution_a.py study --mode DEBUG --no-gpu ...` |
| Component audit (why the gate failed) | `python experiments/contribution_a.py audit ...` |
| Representation geometry (nine spaces) | `python experiments/contribution_a.py representation ...` |
| **Contribution B** (α sweep) | `python experiments/contribution_b.py --config configs/contribution_b.yaml` |
| Strategy registry | `daowod-run strategies` |

Start with `--mode DEBUG`: it exercises every stage in minutes with no GPU, and prints
that its numbers are not reportable. Then `FAST`, then `MAIN`. On Colab use
[`notebooks/contribution_a_colab.ipynb`](notebooks/contribution_a_colab.ipynb), which is
a thin shim over the same entrypoint.

Full protocol, splits, PROB boundary and the exact commands:
[`docs/reproduction.md`](docs/reproduction.md). Large binaries and their digests:
[`docs/artifacts.md`](docs/artifacts.md).

## Layout

```text
docs/
  proposal.docx        the contract
  research_design.md   requirements, cited to the proposal; §7 what it does NOT require
  reproduction.md      protocol, splits, PROB boundary, commands, measured design decisions
  results.md           Contribution A's measured results, including the falsification
  artifacts.md         large-binary manifest: digests, locations, regeneration

configs/
  contribution_a.yaml  THE protocol: pool sizes, budgets, rounds, seeds, arms, severities
  contribution_b.yaml  memory budget, α grid, granularity

data/protocol/         version-controlled split IDs and class groups (digest-pinned)

src/daowod/            18 modules, flat
  cli.py               registry inspection
  config.py            execution modes: the YAML schema and loader
  scoring.py         ★ equation (1): the one scorer and the one registry
  components.py      ★ U · D · rarity · coh, cluster and label-anchored estimators
  candidates.py        candidate pool from PROB outputs only, ground-truth free
  oracle.py            region-level ground-truth matching + the no-leakage guard
  dataset.py           VOC image IDs, annotations, pools
  longtail.py          head/medium/tail groups + controlled tail severities
  representations.py   the nine feature spaces
  annotation.py      ★A multi-round region acquisition, iterative ĉ / n̂_c refresh
  study.py           ★A arm × severity × seed matrix, pilot/evaluation split
  memory.py          ★B m_c ∝ n_c^α allocation core
  discovery.py         annotation-set quality, tail discovery, efficiency curve
  geometry.py          feature-space geometry against the oracle strata
  audit.py             component-level mechanism diagnostics
  figures.py           every publication figure
  tables.py            CSVs, JSON manifests, markdown summary
  pipeline.py          preflight + cost estimate + stage sequencing + resume
  detector.py          the only module that talks to PROB: adapter + export cache

experiments/           one file per experiment; contribution_a.py is the entrypoint
notebooks/             one thin Colab shim per contribution
tests/                 215 tests, fixtures.py is the only synthetic pool
```

### Where to make a change

| To change… | Edit |
|---|---|
| the acquisition formula | `src/daowod/scoring.py`, `src/daowod/components.py` |
| how tail rarity is estimated | `src/daowod/components.py` |
| which arms are compared, pool size, seeds | `configs/contribution_a.yaml` |
| the memory allocation rule | `src/daowod/memory.py` |
| a metric | `discovery.py` / `geometry.py` / `audit.py` |
| a figure | `src/daowod/figures.py` |
| add an experiment | a file in `experiments/` + a subcommand |

## Design rules this repository holds to

1. **One scorer.** Exactly one implementation of equation (1) and one strategy registry.
   Names are plain (`full`, not `v2:full`); there is no second semantics version.
2. **Ground truth is a protocol and evaluation input, never an acquisition input.**
   Enforced by two automated checks, one of which re-derives every score from its
   recorded components, so it constrains arithmetic rather than naming.
3. **Real data or nothing.** No synthetic-pool result is ever reported. The one
   deterministic fixture lives in `tests/` and says so.
4. **The protocol is never mutated to fit a clock.** An earlier version shrank the
   evaluation pool when the runtime projection did not fit, silently moving every
   reported denominator. The preflight now estimates runtime, disk and memory, prints
   them, and **refuses to start** if a declared limit is exceeded.
5. **The library never imports torch.** PROB is reached only through a subprocess and a
   content-keyed export cache whose fingerprint covers everything that could change one
   exported number.
6. **Negative results are first-class.** The falsified gate premise keeps its code, its
   data path and its document, because it falsifies a hypothesis the proposal states.

## What is not implemented

Named here so nobody looks for it: PROB training and evaluation (external, and the
official evaluator remains the only source of known mAP, U-Recall, WI and A-OSE);
incremental retraining and per-group forgetting; the closed A→B feedback loop; LVIS
confirmation; and the acquisition half of the representation experiment, which finished
only for the baseline space (`docs/results.md` §11).
