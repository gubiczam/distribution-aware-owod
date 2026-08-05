# Refactor Plan (revised under Phase 3 constraints)

Working document for the architecture refactor. Deleted at completion; its permanent
content is `docs/research_design.md`, `docs/reproduction.md`, `docs/results.md` and the
final repository map in `README.md`.

## Safety record

| Item | Value |
|---|---|
| Refactor branch | `refactor/clean-architecture` — the only branch used |
| **Safety snapshot commit** | **`1ecb585bbeb9c86d3248d7d724ef6b5da8d04de8`** (2026-08-04) |
| Snapshot contents | All 34 previously-untracked files: the Contribution A region-level study, the E4 representation experiment, their documents and tests |
| Baseline commit | `8cbc33d` = `main`, never modified |
| Proposed tag | `pre-refactor-snapshot` on `1ecb585` |
| Archive of everything deleted | Git history at `8cbc33d` (tracked files) and `1ecb585` (previously-untracked files). **No archive module is kept in the live source tree.** |
| Push / merge | Not performed. Requires an explicit request. |

## Scope

**In scope:** remove dead and superseded code; consolidate Contribution A; preserve the
scientifically relevant negative findings; create a clean boundary for Contribution B;
simplify configs, docs and entrypoints; rewrite tests; produce thin Colab notebooks.

**Out of scope, explicitly:** any offline substitute for catastrophic-forgetting
evaluation; any claim that Contribution B is experimentally complete; a new
incremental-learning framework; expensive experiments; changes to the scientific
protocol; moving large binaries into Git; **any NPZ or output deletion**.

---

## 1. Design rules applied to reach the revised tree

1. **Flat package.** No `core/`, `run/`, `report/` or `loop/` package: each would have had
   one consumer group and would exist only to look tidy. `src/daowod/` is flat, and the
   contribution boundary is carried by module names and docstrings.
2. **No single-module package for future work.** Contribution B is one module,
   `memory.py`, not a `memory/` package. An empty package is architecture for work that
   does not exist yet.
3. **No `feedback.py`.** The closed loop of `research_design.md` §4 has no experiment yet;
   the module would have had zero users.
4. **Merge where the split had one consumer:** `preflight` into `pipeline` (same
   entrypoint); PROB adapter and export cache into `detector` (one detector);
   `normalisation` into `scoring`; `groups` into `longtail`; the leakage guard into
   `oracle`; `revealed` into `components` (it is an alternative estimator for two
   existing components, not a separate concern).
5. **Keep `figures` and `tables` separate.** At ~560 and ~550 lines each they are
   independently large, and they answer different questions (what a reader looks at
   versus what a reader re-analyses).
6. **No `runtime.py`.** Adaptive protocol mutation is deleted outright. A small
   deterministic estimate — runtime, memory, disk — is printed by `pipeline.py`'s
   preflight, which **fails before execution** if a declared limit is exceeded. It never
   shrinks the pool.
7. **One CLI, one entrypoint per contribution.**

## 2. Revised target tree

```
distribution-aware-owod/
├── README.md                       what · where A and B are · how to reproduce · where to extend
├── pyproject.toml
├── .gitignore                      keeps the data/protocol/ whitelist; outputs/ stays ignored
│
├── docs/
│   ├── proposal.docx               the contract (moved from the repository root)
│   ├── research_design.md          the architecture contract, cited to the proposal
│   ├── reproduction.md             datasets, splits, severities, seeds, PROB boundary, exact commands
│   ├── results.md                  ONE Contribution A results document
│   └── artifacts.md                external artifact manifest: filenames, SHA256, paths, regeneration
│
├── configs/
│   ├── contribution_a.yaml         region-level study; debug · fast · main modes (absorbs modes.py)
│   └── contribution_b.yaml         allocation core: M, α grid, granularity
│
├── data/protocol/                  version-controlled split IDs + class groups (unchanged)
│
├── src/daowod/                     19 modules, flat
│   ├── __init__.py                 exports nothing eagerly  ← this is what decoupled the programs
│   ├── cli.py                      THE command-line entrypoint
│   ├── config.py                   YAML → validated, digest-stamped config
│   │
│   ├── scoring.py                  ★A  equation (1): the one scorer, registry, normalisation
│   ├── components.py               ★A  U · D · rarity · coh, cluster and revealed estimators
│   ├── candidates.py               candidate pool from PROB outputs only, ground-truth free
│   ├── oracle.py                   region-level GT matching + the no-leakage assertion
│   ├── dataset.py                  VOC image IDs, annotations, image pools
│   ├── longtail.py                 head/medium/tail groups + controlled tail severities
│   ├── representations.py          feature spaces for the representation experiment
│   │
│   ├── annotation.py               ★A  multi-round region acquisition, iterative ĉ / n̂_c refresh
│   ├── study.py                    ★A  arm × severity × seed matrix, pilot/evaluation split
│   │
│   ├── memory.py                   ★B  m_c ∝ n_c^α allocation core (mathematics only)
│   │
│   ├── discovery.py                annotation-set quality, tail discovery, efficiency curve
│   ├── geometry.py                 feature-space geometry against the oracle strata
│   ├── audit.py                    component-level mechanism diagnostics
│   │
│   ├── figures.py                  every publication figure
│   ├── tables.py                   CSVs, JSON manifests, markdown summary
│   │
│   ├── pipeline.py                 preflight + deterministic estimate + stage sequencing + resume
│   └── detector.py                 PROB subprocess adapter + content-keyed export cache
│
├── experiments/
│   ├── contribution_a.py           THE Contribution A entrypoint (--stage study|audit|representation)
│   ├── contribution_b.py           THE Contribution B entrypoint (α allocation sweep)
│   └── extract_embeddings.py       torch-side subprocess; runs OUTSIDE this environment
│
├── notebooks/
│   ├── contribution_a_colab.ipynb  thin shim over experiments/contribution_a.py
│   └── contribution_b_colab.ipynb  thin shim over experiments/contribution_b.py
│
└── tests/
    ├── fixtures.py                 the one deterministic fabricated export
    ├── test_scoring.py             equation (1), registry, normalisation, components
    ├── test_annotation.py          acquisition loop, study matrix, revealed estimator
    ├── test_memory.py              ★B  α < 0, α = 0, α = 1, budget conservation, ties
    ├── test_evaluation.py          discovery, geometry, audit
    ├── test_pipeline.py            end-to-end on the fixture
    └── test_clean_clone.py         every required input survives a clean clone
```

Requirements check: one CLI entrypoint (`cli.py`); one Contribution A experiment
entrypoint (`experiments/contribution_a.py`); one Contribution B experiment entrypoint
(`experiments/contribution_b.py`); no package with a single trivial module; no generic
abstraction without multiple concrete users; no empty architecture for future work.

## 3. Contribution B — exactly what is built, and what is not

`src/daowod/memory.py`, ~200 lines, mathematics only:

* allocate a fixed integer budget `M` across classes with `m_c ∝ n_c^α`;
* **exact integer conservation**: `Σ m_c == M` for every α and every count vector;
* **deterministic tie handling**: largest-remainder with a documented, stable tie-break
  (no dependence on dict ordering or floating-point luck);
* object-level allocation from class counts;
* image-level allocation for multi-label images (one image may carry several classes);
* explicit α values with validation — α is a declared parameter, never inferred;
* unit tests covering α < 0 (tail-favouring), α = 0 (uniform, the current standard) and
  α = 1 (size-proportional), plus conservation and tie determinism.

**Not built, deliberately:** no offline forgetting proxy; no replay buffer training loop;
no PROB retraining; no `NotImplementedError` scaffolding. The integration point for future
replay/retraining is a single documented function signature — the allocation returns a
plain mapping from class to exemplar count (and an image selection for the image-level
case), which is all a future replay trainer needs to consume.

`docs/research_design.md` §8 states the non-claim explicitly: measuring optimal α
requires real incremental updates and remains a separate future research step.

## 4. Contribution A results — one document

`docs/results.md` consolidates the conclusions of five pieces of work and narrates no
history:

| Retained result | Hypothesis | Role |
|---|---|---|
| Coherence-gate experiment (region-level study) | H-A1, H-A2 | primary experiment |
| Component-level failure audit | H-A1 | localises the failure to components |
| Representation experiment | H-A1, H-A2 | separates embedding from formulation |
| Revealed-label estimator experiment | H-A3 | bounds how much is estimation error |
| Objectness × box-scale prior | — | **non-contribution control only**, labelled as such |

The four current narrative documents (`contribution_a_active_annotation.md`,
`contribution_a_failure_analysis.md`, `contribution_a_revealed_results.md`,
`e4_representation_results.md`) collapse into it. Numbers preserved before their sources
are deleted: pool composition (75.1 % background, 23.9 % known, 0.98 % unknown; 364
reachable unknowns, 273/65/26 head/medium/tail); per-component signal AUC; the 1.9×
objectness-prior contrast; the eleven-arm comparison; same-class sibling rank 202 → 6;
tail versus background same-label fraction 0.015 / 0.888. Plus one sentence recording the
signal-to-noise methodological correction.

Code is retained only where it reproduces a number that remains in this document.

## 5. Complexity: current versus revised target

| Dimension | Current (measured) | Revised target | Δ |
|---|---|---|---|
| `src/` modules | 33 | **19** (+ `__init__`) | −42 % |
| `src/` lines | 18 159 | ~9 500 | −48 % |
| Analysis / experiment scripts | 16 | **3** | −81 % |
| Analysis / experiment lines | 8 402 | ~1 000 | −88 % |
| Test files | 10 | **7** | −30 % |
| Test lines | 5 253 | ~2 200 | −58 % |
| **Total lines (src + experiments + tests)** | **31 814** | **~12 700** | **−60 %** |
| Notebooks | 7 | 2 | −71 % |
| Configs | 6 | 2 | −67 % |
| Documentation files | 14 (3 337 lines) | 5 | −64 % |
| Public entrypoints | ~24 | 1 CLI + 2 experiments + 2 notebooks | −79 % |
| Contribution A implemented | yes | yes | — |
| Contribution B | 0 lines | allocation core only, unit-tested | — |

Line figures for the target column are estimates from per-module budgets; module, script,
notebook, config, doc and entrypoint counts are exact commitments.

### Capability cuts considered and rejected

Three options were offered to push below ~12 700 lines. **All three were declined**, so
none is applied and the target stays ~12 700.

| Option | Would have saved | Decision |
|---|---|---|
| **C1** Drop the E4 embedding-projection figures and trim `geometry.py` | ~320 lines | **DO NOT APPLY** — the visual representation evidence is retained |
| **C2** Drop ZIP packaging and artifact verification from `tables.py` | ~150 lines | **DO NOT APPLY** — both retained |
| **C3** Coarsen `pipeline.py` resume from per-stage to per-phase | ~250 lines | **DO NOT APPLY** — per-stage resume retained |

## 6. Resolved decisions

| # | Decision | Resolution |
|---|---|---|
| **D1** | `outputs/` curation | Delete `stage1b/*.npz` — **deferred** to the separate, separately-approved output-cleanup operation. Nothing in `outputs/` is touched by this refactor |
| **D2** | Image-level retraining loop | **Superseded by D6a** |
| **D3** | `runtime.py` adaptive downscaling | **Delete.** Replaced by a deterministic preflight estimate that fails before execution and never shrinks the pool |
| **D4** | Scope of Contribution B | **Reduced to the mathematical allocation core only** (§3). Not experimentally complete, and not claimed to be |
| **D5** | v1 compatibility semantics | **Delete.** Remove `acquisition.py` and `scoring._VERSION_1`; drop the `v1:`/`v2:` prefix from the live API; move `AcquisitionWeights` into the current config/scoring module. **No archive module in the live tree** — Git history and `1ecb585` are the archive |
| **D6** | Never-executed image-level campaign | **D6a — delete.** `experiment.py` (878), `metrics.py` (411) and the campaign-specific `ProtocolConfig` / `validate_command_parity` code in `config.py` (~250) are removed. No slimmed campaign is retained. The incremental retraining path is rebuilt later, when Contribution B enters experimental execution and its protocol is fixed |

The tree in §2 reflects **D6a** and all of the above.

## 7. Execution sequence

Commit discipline: one purpose per commit; imports valid at every commit; deletion, large
moves and logic rewrites never combined in one commit; each step revertable with
`git revert`.

| # | Purpose | Files | Validation | Expected |
|---|---|---|---|---|
| 1 | Persist architecture documentation | `docs/research_design.md`, `docs/refactor_plan.md` | read | contract on disk before destructive work |
| 2 | Tag the snapshot | — | `git tag` | `pre-refactor-snapshot` → `1ecb585` |
| 3 | Remove generated files and obsolete prose | `.DS_Store`, `.pytest_cache/`, `.ruff_cache/`, 7 documents, `docs/figures/` (8), 5 notebooks, 4 configs | `pytest -q` | unchanged pass count; nothing imports them |
| 4 | Write `docs/results.md` + `docs/reproduction.md` **before** deleting their sources | +2 docs, −6 docs | read; every number traceable to a surviving CSV | §4 numbers preserved |
| 5 | Remove synthetic-only and disqualified-pool code | `simulation.py`, `validate_contribution_a.py`, `real_stage1_analysis.py`, `stage2_plan.py`, `prepare_*`, `compare_*`, `probes/` | `pytest -q` | obsolete tests removed with their subject |
| 6 | Remove v1 compatibility semantics (**D5**) | `acquisition.py`, `scoring._VERSION_1`, `config.py`, `test_scoring_core.py`, configs, docs | `pytest -q`; `daowod-run strategies` | plain strategy names, no `v1:`/`v2:` prefix |
| 7 | Break the eager `__init__.py`; delete `offline.py`, `runtime.py`, most of `diagnostics.py` | `__init__.py`, 3 modules, `cli.py`, `pipeline.py` | `pytest -q`; `python -c "import daowod"` | legacy modules no longer load transitively; preflight prints an estimate and fails over budget |
| 8 | Consolidate Contribution A — `git mv` and merges, **no scientific behaviour change** | all of `src/daowod/` | `pytest -q`; `compileall` | identical numeric behaviour; a failure here is a broken import |
| 9 | Simplify CLI and configs | `cli.py`, `config.py`, `configs/` (2) | `daowod-run --help`; `validate` | one CLI, two configs, modes as YAML |
| 10 | Consolidate tests | `tests/` (7 files) | `pytest -q` | every retained behaviour tested |
| 11 | Add the Contribution B allocation core | `memory.py`, `experiments/contribution_b.py`, `configs/contribution_b.yaml`, `tests/test_memory.py` | `pytest -q` | α < 0 / 0 / 1, Σ`m_c` = M, deterministic ties |
| 12 | Rebuild `experiments/` | 3 files | `--help` each | one entrypoint per contribution |
| 13 | Rewrite both notebooks as thin shims | 2 notebooks | `nbformat` validation | valid JSON, ~10 cells |
| 14 | Rewrite `README.md`, `docs/reproduction.md`, `docs/artifacts.md`; move `owod.docx` | root + `docs/` | the 10-minute test | a newcomer names both contributions and both entrypoints |
| 15 | Full validation | — | the §8 suite | all green |
| 16 | **STOP** | — | — | no output or NPZ cleanup; that is a separately approved operation |

## 8. Validation suite

```bash
pytest -q
ruff check .
ruff format --check .
python -m compileall -q src experiments
python -c "import daowod"                                   # import smoke test
daowod-run --help                                            # CLI
python experiments/contribution_a.py --help
python experiments/contribution_b.py --help
python -c "import json,sys; [json.load(open(p)) for p in sys.argv[1:]]" notebooks/*.ipynb
pytest -q tests/test_clean_clone.py                          # clean-clone asset audit
```

**Test invariant** — not a fixed count. Every retained behaviour is tested; all retained
tests pass; no test exists solely to preserve deleted compatibility behaviour; no
synthetic result is presented as scientific evidence.

## 9. Outputs and large files — postponed

No NPZ is moved, deleted or duplicated during this refactor. `outputs/` stays gitignored
and untouched; `real_stage1/*.npz` stays external to Git. The only artifact work in scope
is `docs/artifacts.md`: expected filenames, SHA256 hashes, documented external paths and
regeneration instructions. The recorded **D1** decision to delete `stage1b/*.npz`
(466 MB) waits for the separate, separately-approved output-cleanup operation.

## 10. Stop conditions

Work halts and asks before: deleting any NPZ or output directory; changing the active
scoring equation; changing candidate-pool semantics; changing long-tail severity
construction; changing oracle matching; removing an experiment whose numeric conclusion
appears in `docs/results.md`; implementing PROB retraining; pushing or merging.
