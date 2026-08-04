# E4 Phase 1 — where proposal embeddings come from, and what is obtainable

Representation Experiment E4 asks whether the coherence gate's failure is a property
of PROB's decoder embedding rather than of the coherence formulation. Before
comparing spaces, this documents exactly which spaces exist inside PROB, which are
exported today, and which can be obtained **without retraining** on the hardware
available.

## 1. Where the embeddings originate

PROB is Deformable-DETR with a probabilistic objectness head. Following
`PROB/models/prob_deformable_detr.py` and `PROB/models/deformable_transformer.py`:

```text
image
 └─ backbone (ResNet-50, DINO-initialised)         -> multi-scale feature maps  srcs
     └─ input projections                          -> src_flatten  [B, sum(H*W), 256]
         └─ transformer encoder (6 layers)         -> memory       [B, sum(H*W), 256]
             └─ transformer decoder (6 layers)     -> hs           [6, B, 100, 256]
                 ├─ class_embed(hs[-1])            -> pred_logits  [B, 100, 81]
                 ├─ bbox_embed(hs[-1])             -> pred_boxes   [B, 100, 4]
                 ├─ prob_obj_head(hs[-1])          -> pred_obj     [B, 100]  (scalar)
                 └─ hs[-1]                         -> pred_features [B, 100, 256]
```

The forward pass returns exactly four tensors:

```python
out = {
    "pred_logits": outputs_class[-1],
    "pred_boxes": outputs_coord[-1],
    "pred_obj": outputs_objectness[-1],
    "pred_features": hs[-1],  # the fork's addition; without it E1-E3 are impossible
}
```

`daowod_prob_bridge.py predict` converts these into the export the whole repository
consumes: `embeddings` = `pred_features` = `hs[-1]`, `posterior` =
`objectness · sigmoid(logits)` renormalised, `objectness` = `exp(-T·pred_obj)`,
`boxes` = `pred_boxes`, `predicted_labels` = argmax of the posterior.

**So every number in every experiment so far has been computed in one space: the
final decoder layer's 256-d hidden state, per decoder query.**

## 2. Representation inventory

| representation | where it lives | in the export? | obtainable without retraining? |
|---|---|---|---|
| **Decoder, final layer** `hs[-1]` | model output | **yes**, as `embeddings` | already have it |
| Decoder, intermediate layers `hs[0..4]` | `torch.stack(intermediate)` inside `DeformableTransformerDecoder` | no | yes, but needs a bridge patch **and a GPU re-run** of inference |
| Encoder tokens `memory` | `DeformableTransformer.forward` | no | yes in principle, but they are *spatial tokens*, not per-proposal: a region embedding requires RoI-pooling `memory` over each predicted box across four feature levels. Needs a patch **and** a GPU re-run. |
| Backbone maps `srcs` | `Joiner`/`Backbone` output | no | same as encoder: spatial, needs RoI pooling, patch and GPU |
| Objectness "features" | `prob_obj_head(hs[-1])` | **yes**, as the `objectness` scalar | the head is an MLP `256 -> 1`; its *feature space* is `hs[-1]` itself, so "objectness features" is not a distinct space — only a 1-d projection of the decoder space. Exported. |
| Query embeddings `query_embed.weight` | learned parameter | no | irrelevant as a *proposal* representation: it is a fixed 100×512 table, identical for every image, so it carries no per-region information. |
| Class posterior | `class_embed(hs[-1])` | **yes**, as `posterior` | already have it (a 20-d supervised-semantic projection of the decoder space) |

**Consequence.** The three PROB-internal spaces that would be genuinely different
from `hs[-1]` — intermediate decoder layers, encoder tokens, backbone maps — all
require both a source patch and a fresh GPU inference pass over 4 000 images. That
is out of reach here (no CUDA device; MPS cannot run PROB's compiled
`MultiScaleDeformableAttention`). They are recorded as the natural next step if a
GPU session becomes available.

What *is* reachable is more valuable for the hypothesis anyway: re-embedding the
same predicted boxes with an encoder trained under a **different objective**. That
tests H1 directly, because H1 is a claim about what an embedding must preserve, not
about which layer of PROB it comes from.

## 3. What E4 actually compares

All of these were confirmed to exist on this machine, and none needs a download.

| space | kind | dim | objective it was trained for |
|---|---|---:|---|
| `prob_decoder` | export | 256 | Task-1 detection: 19 known classes + objectness **(frozen baseline)** |
| `prob_posterior` | export | 20 | the same, projected onto class scores (log-probabilities) |
| `prob_geometry` | export | 5 | nothing learned: objectness, box scale, aspect, centre |
| `dino_resnet50` | crop re-embed | 2048 | **self-supervised** DINO, no class labels at all |
| `imagenet_resnet50` | crop re-embed | 2048 | **closed-set supervised** ImageNet-1k classification |
| `prob_decoder_whitened` | derived | 64 | PCA-whitened decoder space |
| `prob_decoder_minus_top4` | derived | 256 | decoder space with its 4 leading PCs projected out |
| `dino_whitened` | derived | 64 | dimension-matched partner for the whitened decoder |
| `prob_decoder_plus_dino` | derived | 2304 | equal-weight L2-normalised concatenation |

Two encoders and one control, chosen so the result is interpretable either way:

* **DINO ResNet-50** is the representation whose neighbourhoods are reported to
  transfer to categories that were never labelled — precisely the property the
  coherence gate needs. Its weights are already in the PROB checkout, because PROB
  uses them to initialise its own backbone. That makes the comparison unusually
  clean: *the same weights*, before and after PROB's detection fine-tuning.
* **ImageNet ResNet-50** separates two confounded explanations. If it also clusters
  the unknown classes, then closed-set supervision is not what destroys the
  neighbourhoods and the cause is specific to the detection objective. If it fails
  where DINO succeeds, the cause is supervision on a fixed label set.
* The **derived** spaces test the cheapest alternative explanation — that the
  decoder space contains the structure but a few high-variance directions carrying
  the background mass dominate the metric. If whitening or removing the leading
  components repairs the tail's neighbourhood purity, the diagnosis is the metric,
  not the representation.

CLIP was considered and **excluded**: a `laion/CLIP-ViT-B-32` snapshot is in the
local Hugging Face cache, but neither `transformers` nor `open_clip` is installed in
the torch environment and there is no network access to add them. Loading the raw
weights would mean hand-implementing a ViT forward pass, which is a large amount of
untested code for a third encoder that would not change the design of the
experiment. SAM was excluded for the same reason plus size. Both are recorded as
available extensions once a network-connected environment exists.

## 4. Cost and where it is spent

Re-embedding is the only expensive step. Measured on this machine (Apple MPS,
ResNet-50, 224² crops): **94 crops/s**, both encoders in one pass over each crop.

The export holds 400 000 proposals, but only 80 000 are ever read by the
experiment — the evaluation split's candidate pool (48 000), the pilot split's
candidate pool (12 000) and the 20 000-row novelty bank. Those row sets are
functions of objectness, boxes and NMS **only**, never of the embeddings, so they
are byte-identical for every representation. `analysis/e4_required_rows.py`
computes them and the extractor is restricted to them, taking the cost from
2.4 hours to **28 minutes**. The saving is a checked precondition, not an
assumption: the merge step refuses to write a representation if any requested row
is missing, and the geometry driver verifies coverage again before use.

## 5. What is deliberately *not* changed

E4 varies the representation and nothing else. The candidate pool, the region
oracle, the three long-tail severities, the seeds, the annotation budgets, the
strategy definitions, the weights α/β/γ, the coherence exponent, the round count
and every metric stay exactly as the frozen experiments left them. The substitution
point is a single field:

```python
replaced = daowod.representations.substituted_export(export, new_embeddings)
```

Because the candidate filter, the oracle, the severity masks and the budget
prefixes are all functions of fields this does not touch, the pool each strategy
searches is identical across representations, and the only quantity that varies is
the geometry in which novelty, rarity and coherence are computed.

Results: [`e4_representation_results.md`](e4_representation_results.md).
