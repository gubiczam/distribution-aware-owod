# Artifacts

Large binaries are **not** in Git, and `outputs/` is gitignored. This file is the
manifest: what each artifact is, its digest, where it lives, and how to regenerate it.

Everything reported in [`results.md`](results.md) was computed from the exports below.
A digest mismatch means the numbers in that document do not describe the file you have.

---

## 1. Frozen PROB proposal exports

The input to every real-data result. Produced once by `daowod_prob_bridge predict`
inside a PROB checkout, from an S-OWODB / OWDETR Task-1 checkpoint.

| Artifact | Size | SHA256 |
|---|---:|---|
| `reference_proposals.npz` | 466 MB | `76305eb41d79bce4ea48ded9870c669f69f5c5e330b2479487c01e30f91b5883` |
| `candidate_proposals.npz` | 58 MB | `c6d540535da006d5b61fa7dba99915043853f52cc29344b3c474fe7b79e3143c` |

`reference_proposals.npz` contents — 4 000 images, 100 decoder queries each:

| Array | Shape | Meaning |
|---|---|---|
| `image_ids` | (400 000,) | one row per decoder query |
| `embeddings` | (400 000, 256) | `hs[-1]`, the final decoder hidden state |
| `posterior` | (400 000, 20) | `objectness · sigmoid(logits)`, renormalised |
| `confidence` | (400 000,) | PROB's unknown score |
| `objectness` | (400 000,) | `exp(-T · pred_obj)` |
| `boxes` | (400 000, 4) | normalised `cxcywh`, as the detector emitted them |
| `predicted_labels` | (400 000,) | argmax of the posterior |

**Current location.** `outputs/real_stage1/` in the working tree, which is gitignored.
These files have never been committed and are **not recoverable from Git history**.
Regenerating them needs a GPU and the Task-1 checkpoint.

**Provenance caveat, which matters.** This export comes from the 500-image pool that was
later disqualified for training and detector evaluation, because it is a subset of
`owdetr_test`. It remains a valid *input* to the offline geometry and ranking analyses,
which never train and never evaluate a detector — they read proposals and compare
orderings. See [`reproduction.md`](reproduction.md) section 2 before using it for
anything else.

### Regenerating

Inside a PROB checkout with the Task-1 checkpoint, via the bridge the pipeline uses:

```bash
python daowod_prob_bridge.py predict \
    --checkpoint <t1.pth> \
    --data-root <data>/OWOD \
    --dataset OWDETR \
    --prev-introduced-classes 0 \
    --current-introduced-classes 19 \
    --num-classes 81 \
    --objectness-temperature 1 \
    --max-proposals-per-image 100 \
    --image-ids <split>.txt \
    --output reference_proposals.npz
```

Any change to those flags changes the exported numbers, which is why they are also
inside the export cache's fingerprint (`reproduction.md` section 5).

## 2. Re-embedded region features

For the representation experiment. Produced outside the `daowod` environment, because
the library has no torch dependency:

```bash
python experiments/select_rows.py --export <export>.npz --output outputs/e4_representations
<prob-venv>/bin/python experiments/extract_embeddings.py \
    --export <export>.npz --images <data>/JPEGImages \
    --output outputs/e4_representations --rows outputs/e4_representations/rows.npy
```

Two encoders, `dino_resnet50` and `imagenet_resnet50`, both already present in the PROB
checkout because PROB initialises its backbone from DINO. `select_rows.py` restricts
extraction to the rows the experiment actually reads — the evaluation and pilot candidate
pools plus the reference bank prefix — which is a fraction of the 400 000 exported rows
and saves about two and a half hours.

## 3. Result tables and figures

Small, regenerable from the exports above, and cited by `results.md`:

| Directory | What |
|---|---|
| `outputs/audit_contribution_a/` | component-audit tables (signal AUC, precision at budget, neighbourhood composition) |
| `outputs/e4_geometry/` | the nine-space geometry comparison, plus projection figures |
| `outputs/e4_active_learning/prob_decoder/` | the acquisition arms in the baseline space |

`outputs/e4_active_learning/dino_resnet50/` holds a pilot ablation and one severity's
study state and **no comparison tables**: that run did not finish. See `results.md`
section 11.

For Colab handoff, use
[`../notebooks/contribution_a_master_colab.ipynb`](../notebooks/contribution_a_master_colab.ipynb).
It packages compact CSV/JSON/Markdown/figure outputs and records excluded large caches in
`archive_manifest.json`; proposal exports and representation arrays remain external
artifacts under the storage policy below.

## 4. Storage policy

* `outputs/` is gitignored, and no artifact is copied into the repository. A 466 MB
  binary in Git would make every clone pay for it forever.
* The version-controlled inputs are only the small ones that a clean clone must carry:
  `data/protocol/stage1b/*.txt` and `data/protocol/stage2/stage2_class_groups.csv`,
  whose digests are pinned in `tests/test_clean_clone.py`.
* Curating `outputs/` — archiving the exports outside the working tree and deleting the
  superseded diagnostics — is a **separate, separately approved operation**. Nothing in
  the refactor that produced this repository moved, deleted or duplicated any NPZ.
