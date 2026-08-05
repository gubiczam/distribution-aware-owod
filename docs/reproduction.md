# Reproduction

Everything needed to re-run the experiments in [`results.md`](results.md): the protocol,
the splits, the PROB boundary, the commands, and the traps that cost real time.

---

## 1. Protocol constants

Frozen. Every number in `results.md` was produced under exactly these settings.

| Setting | Value |
|---|---|
| Detector | PROB (Probabilistic Objectness), Deformable-DETR backbone, ResNet-50 DINO-initialised |
| Task protocol | S-OWODB / **OWDETR** |
| `previous_introduced_classes` | 0 |
| `current_introduced_classes` | 19 |
| `num_classes` | 81 |
| `objectness_temperature` | 1 |
| Checkpoint | Task-1 S-OWODB; its SHA256 is recorded in every run manifest |
| Canonical evaluation split | PROB `data/OWOD/ImageSets/OWDETR/owdetr_test.txt`, byte-for-byte |
| Proposals per image (export) | 100 (every decoder query) |
| `per_image_limit` (candidate pool) | 20 |
| Oracle matching | region-level, VOC XML, IoU 0.5 |

A mismatch in any of the first six changes the scientific meaning of a run, so
configuration validation rejects train/predict/evaluate drift before execution rather
than after.

## 2. Splits, and one permanently disqualified pool

Version-controlled under `data/protocol/`, so a clean clone carries them:

| File | Rows | Role |
|---|---:|---|
| `stage1b/stage1b_candidate_500.txt` | 500 | candidate (acquisition) pool, official Task-1 train side |
| `stage1b/stage1b_reference_3500.txt` | 3 500 | reference bank for novelty |
| `stage2/stage2_class_groups.csv` | 62 | head / medium / tail assignment |

Preflight proves zero overlap: candidate/evaluation = 0, reference/evaluation = 0,
candidate/reference = 0.

> ### The 500-image evaluation pool is disqualified. Do not re-derive it.
>
> An earlier 500-image pool was **a subset of `owdetr_test`**, the evaluation split. Any
> acquisition or training result computed on it is contaminated and cannot be published.
> It was replaced by the leak-free candidate/reference splits above.
>
> The *proposal export* taken from that pool is still a valid **input** for the offline
> geometry and acquisition analyses in `results.md`, because those never train and never
> evaluate a detector — they read proposals and compare rankings. The disqualification is
> about using it for training or for detector evaluation.

The three image pools — reference, pilot, evaluation — are disjoint by construction. The
pilot chooses the coherence definition; the evaluation pool produces the reported numbers;
the reference pool is the novelty bank. A bank overlapping the pool would make novelty
partly self-referential.

## 3. Ground-truth discipline

Annotations are read in exactly two places, and both are licensed:

1. **Protocol** — building the long-tail evaluation pool. Class labels decide which
   proposals exist; strategies then see only PROB outputs for the survivors. This is the
   same licence the long-tail literature uses to build LT-CIFAR or LVIS-style splits.
2. **Oracle** — revealing a proposal's true class *after* it has been selected. That is
   the definition of active learning, not leakage.

Four checks enforce the boundary and the run stops if any fails:

* `discovery.assert_selection_is_ground_truth_free` re-derives every acquisition score
  from its recorded components; an unrecorded oracle term breaks the identity. This is the
  strong check — it constrains arithmetic, not names.
* the round scorer is verified by introspection to accept no oracle argument;
* the no-ground-truth assertion runs against the actual acquisition records;
* scoring is re-run at a fixed seed and required to be bit-identical.

Results land in `leakage_report.json` and in the run summary.

## 4. Where the embeddings come from

PROB is Deformable-DETR with a probabilistic objectness head:

```text
image
 └─ backbone (ResNet-50, DINO-initialised)   -> multi-scale feature maps
     └─ input projections                    -> src_flatten  [B, sum(H*W), 256]
         └─ encoder (6 layers)               -> memory       [B, sum(H*W), 256]
             └─ decoder (6 layers)           -> hs           [6, B, 100, 256]
                 ├─ class_embed(hs[-1])      -> pred_logits  [B, 100, 81]
                 ├─ bbox_embed(hs[-1])       -> pred_boxes   [B, 100, 4]
                 ├─ prob_obj_head(hs[-1])    -> pred_obj     [B, 100]
                 └─ hs[-1]                   -> pred_features[B, 100, 256]
```

The bridge converts these into the export the repository consumes: `embeddings` =
`pred_features` = `hs[-1]`; `posterior` = `objectness · sigmoid(logits)` renormalised;
`objectness` = `exp(-T · pred_obj)`; `boxes` = `pred_boxes`; `predicted_labels` = argmax of
the posterior.

**So every result in `results.md` §2–§6 is computed in one space: the final decoder
layer's 256-d hidden state per decoder query.** §7 is the experiment that varies it.

Three PROB-internal spaces would be genuinely different from `hs[-1]` — intermediate
decoder layers, encoder tokens, backbone maps — but each needs a source patch **and** a
fresh GPU inference pass, and the latter two are spatial tokens requiring RoI pooling over
four feature levels per box. They are the natural next step given a GPU session. What is
reachable without one, and more informative for the hypothesis, is re-embedding the same
predicted boxes with an encoder trained under a *different objective*.

The nine spaces compared in `results.md` §7:

| space | kind | dim | trained for |
|---|---|---:|---|
| `prob_decoder` | export | 256 | Task-1 detection: 19 known classes + objectness **(baseline)** |
| `prob_posterior` | export | 20 | the same, projected onto class scores |
| `prob_geometry` | export | 5 | nothing learned: objectness, box scale, aspect, centre |
| `dino_resnet50` | crop re-embed | 2048 | **self-supervised** DINO, no class labels |
| `imagenet_resnet50` | crop re-embed | 2048 | **closed-set supervised** ImageNet-1k |
| `prob_decoder_whitened` | derived | 64 | PCA-whitened decoder space |
| `prob_decoder_minus_top4` | derived | 256 | decoder space, 4 leading PCs projected out |
| `dino_whitened` | derived | 64 | dimension-matched partner for the whitened decoder |
| `prob_decoder_plus_dino` | derived | 2304 | equal-weight L2-normalised concatenation |

DINO ResNet-50 weights are already in the PROB checkout — PROB uses them to initialise its
own backbone — so the comparison needs no download.

## 5. The PROB boundary

`daowod` never imports torch. PROB is reached only as a subprocess that writes files:

**Proposal NPZ** must contain `image_ids`, `confidence`, `embeddings`; may contain
`posterior`, `predicted_labels`, `boxes`, `objectness`.

**Metrics JSON** must contain `known_mAP`, `U_Recall`, `WI`, `A_OSE`; may contain
`detections_path`.

Detector inference is the only GPU cost and the only step that cannot be repeated cheaply,
so it is done once into a content-keyed cache. A chunk is reused only when the fingerprint
matches: bridge settings (data root, dataset, task class counts, objectness temperature,
proposals per image, seed, device), the checkpoint digest, and the exact image IDs in that
chunk. Anything that could change one exported number is inside the fingerprint, so a
stale cache cannot silently contaminate a run. A disconnected session resumes at the first
missing chunk; an identical rerun does no GPU work at all.

## 6. Running it

Install:

```bash
python -m pip install --editable ".[dev]"    # Python 3.11
```

Checks:

```bash
pytest
ruff check .
ruff format --check .
python -m compileall -q src experiments
```

Contribution A — the region-level annotation study, the component audit and the
representation geometry, all from one entrypoint:

```bash
# the reported experiment; --mode overrides `mode:` in the config
python experiments/contribution_a.py study \
    --mode DEBUG --no-gpu \
    --data-root <data-root> --split <ids>.txt --checkpoint <t1>.pth \
    --output outputs/contribution_a_debug --cache outputs/cache

# the component audit behind results.md sections 2-5
python experiments/contribution_a.py audit \
    --export <frozen-export>.npz --annotations <data-root>/Annotations \
    --output outputs/audit_contribution_a

# the nine-space geometry comparison behind results.md section 7
python experiments/contribution_a.py representation \
    --export <frozen-export>.npz --annotations <data-root>/Annotations \
    --representations outputs/e4_representations --output outputs/e4_geometry

# the acquisition arms in a different space -- INCOMPLETE, see results.md section 11
python experiments/contribution_a.py representation-acquisition \
    --export <frozen-export>.npz --annotations <data-root>/Annotations \
    --representations outputs/e4_representations --output outputs/e4_active_learning
```

The authoritative Colab handoff for Contribution A is
[`../notebooks/contribution_a_master_colab.ipynb`](../notebooks/contribution_a_master_colab.ipynb).
It orchestrates these repository entrypoints, records environment/commit metadata, and
keeps official detector metrics marked unavailable unless a real retraining/evaluation
run produces them.

`study` takes only paths: sizes, budgets, rounds, seeds, arms and the severity axis all
come from the config. Each stage is also runnable directly
(`python experiments/component_audit.py --help`); the dispatcher forwards its arguments
untouched.

The representation stages need region embeddings produced **outside** this environment,
because `daowod` deliberately has no torch dependency and the only interpreter with torch
is the PROB checkout's. Select the rows first: re-embedding all 400 000 exported rows
costs about two and a half hours, and only a fraction is ever read.

```bash
python experiments/select_rows.py \
    --export <frozen-export>.npz --output outputs/e4_representations

~/Documents/PROB/.venv/bin/python experiments/extract_embeddings.py \
    --export <frozen-export>.npz --images <data-root>/JPEGImages \
    --output outputs/e4_representations \
    --rows outputs/e4_representations/rows.npy
```

Contribution B — the allocation core:

```bash
python experiments/contribution_b.py --config configs/contribution_b.yaml
```

Start in `DEBUG` mode: it exercises every stage on a few hundred images in minutes with no
GPU, and the run prints that its numbers are not reportable. Then `FAST`, then `MAIN`.
Mode names are matched case-insensitively.

`python -m daowod.cli` (installed as `daowod-run`) does **not** run experiments. Its only
subcommand is `strategies`, which lists the acquisition registry:

```bash
daowod-run strategies --verbose
```

## 7. Execution modes

A mode fixes every quantity that trades runtime against statistical power. Modes live in
the config, not in code, so a protocol change is a version-controlled diff.

| Mode | eval / pilot / reference images | per-image limit | budgets | rounds | seeds | arms | severity axis | reportable |
|---|---|---:|---|---:|---:|---:|---|---|
| `debug` | 200 / 60 / 150 | 12 | 25–100 | 2 | 2 | 5 | flatten-only | **no** |
| `fast` | 500 / 150 / 350 | 20 | 50–400 | 4 | 2 | 5 | flatten-only | **no** |
| `main` | 2 400 / 600 / 1 000 | 20 | 100–2 000 | 5 | 3 | 5 | flatten + sharpen | yes |
| `main_revealed` | 2 400 / 600 / 1 000 | 20 | 100–2 000 | 5 | 3 | 11 | flatten + sharpen | yes |

`main_revealed` is the eleven-arm comparison in `results.md` §6: the same protocol with the
free informativeness-prior control and the label-anchored distribution term running beside
the untouched baseline, so every arm shares one pool, one severity axis, one seed set and
one budget grid.

## 8. Measured pool sizes

Real S-OWODB / OWDETR Task-1 exports, `per_image_limit = 20`:

| Images | Candidate proposals | Reachable unknown objects | Classes | Tail objects |
|---:|---:|---:|---:|---:|
| 500 | 10 000 | 104 | 22 | 8 |
| 2 400 | 48 000 | 364 | 38 | 26 |
| 3 500 | 70 000 | 508 | 44 | 25 |

The tail denominator is the limiting quantity, and it is why the reported study infers
2 400 images and why any summary names the tail count explicitly whenever it is below 20.

## 9. Design decisions forced by measurements

Not preferences. Each was measured on real S-OWODB Task-1 exports, and each will bite
anyone who changes it back.

**Rank candidates by objectness, not the unknown score.** Per-image top-20 by objectness
retains 51.9 % of true-unknown proposals at a 39.4 % on-object rate, versus 43.8 % / 26.5 %
for the unknown score. Objectness also has the higher AUC for "sits on an object" (0.879 vs
0.711). `results.md` §5 shows PROB's unknown score is actively harmful at Task 1.

**Filter the pool at all.** A raw export is ~85 % background. Restricting to object-like
proposals raised pseudo-class rarity rank stability from 0.736 to 0.991.

**Take denominators from the pool, not the dataset.** The 500-image export's annotations
hold 50 unknown classes; PROB's proposals reach 22, and the ones it misses are the rarest.
Grouping over annotation frequency would leave the tail group with 2 objects. Grouping over
*reachable* objects keeps all three groups populated.

**Two retention profiles are needed**, because one cannot move the imbalance in both
directions. The natural reachable distribution is already extreme (3 500 images: head class
73 objects, many classes at 1), so an absolute exponential target sits *above* the natural
count for most middle and tail classes; `min(target, available)` then keeps everything and
a "severe" setting silently reproduces "natural" — measured head:tail 15.64 versus 15.44, a
1 % gap. The `relative` profile scales each class by its own count and does sharpen; the
`absolute` profile with a head cap is the only way to flatten. Each severity records which
it used, and validation fails loudly rather than reporting two names for one regime.

**Small pools cannot express a sharpening axis at all.** On 500 images the reachable tail
group holds 8 objects across 7 classes — already at the one-object-per-class floor — so
`debug` and `fast` use the flatten-only axis.

**Novelty must be blocked over candidate rows.** `candidates @ references.T` at
70 000 × 20 000 allocates 11.2 GB and is killed on a Colab CPU runtime. The blocked form is
bounded at 128 MB and measured 22× faster (47 s → 2.1 s) with identical rank order.

## 10. Runtime, and what the software will not do

Preflight measures rather than assumes, and prints a deterministic estimate of runtime,
memory and disk before any GPU time is spent. **If the projection exceeds the declared
limit the run fails before starting.**

It does **not** silently shrink the evaluation pool to fit a budget. An earlier version did,
which mutates the protocol after the protocol was declared and makes a reported number
untraceable to the design it was supposed to test. Pool size is an explicit choice: pick a
smaller mode, or raise the limit deliberately.

## 11. Artifacts

The frozen proposal exports are large binaries and are **not** in Git; `outputs/` is
gitignored. Expected filenames, SHA256 hashes, external paths and regeneration
instructions are in [`artifacts.md`](artifacts.md).

## 12. Strategy naming

Strategy names are plain (`full`, `random`, `objectness_area_prior`, `revealed_full`, …) and
resolve through one registry. There is no second semantics version and no `v1:` / `v2:`
prefix: equation (1) is one formula, and the pre-audit variant reproduced numbers from the
disqualified pool of §2. That variant is in Git history at tag `pre-refactor-snapshot`,
which is where a superseded definition belongs.
