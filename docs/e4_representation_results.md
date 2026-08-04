# E4 — Is the failure the representation or the formulation?

The frozen experiments established, reproducibly, that the coherence gate does not
improve tail discovery, and localised the cause to the geometry of PROB's decoder
space. That is a claim about one feature space. E4 tests whether it is a claim about
the *formulation* by holding the acquisition completely fixed and varying only the
space its neighbourhoods are computed in.

Phase 1 (where embeddings come from, what is obtainable) is in
[`e4_representation_audit.md`](e4_representation_audit.md). Reproduce everything
here with:

```bash
python analysis/e4_required_rows.py --export outputs/real_stage1/reference_proposals.npz \
    --output outputs/e4_representations
~/Documents/PROB/.venv/bin/python analysis/extract_region_embeddings.py \
    --export outputs/real_stage1/reference_proposals.npz --images ~/owod_stage/JPEGImages \
    --output outputs/e4_representations --rows outputs/e4_representations/rows.npy
python analysis/experiment_e4_representations.py --export outputs/real_stage1/reference_proposals.npz \
    --annotations ~/owod_stage/Annotations --output outputs/e4_geometry
python analysis/run_e4_active_learning.py --export outputs/real_stage1/reference_proposals.npz \
    --annotations ~/owod_stage/Annotations --output outputs/e4_active_learning
```

Pool: the same 48 000-proposal evaluation pool as the frozen runs — 471 unknown
proposals (34 tail, 83 medium, 354 head), 11 460 known-object, 36 069 background;
364 reachable unknown objects, 26 of them tail.

---

## A correction to the headline statistic, before any result

The earlier audit's decisive number was

    tail purity advantage = tail same-label neighbour fraction
                          / background same-label neighbour fraction

measured at 0.017 in the decoder space. **That ratio is confounded.** A class with
`c` members in the pool cannot supply more than `min(c - 1, k)` same-class
neighbours, so at k = 10 its purity is capped at `min(c - 1, 10) / 10`. The tail
group's classes hold 1 to 6 proposals — three of them hold exactly one — giving the
tail a mean *ceiling* of **0.235**, while background's ceiling is 1.0. Part of the
apparent 60-fold gap was the class frequency that *defines* the tail, not the
geometry under test.

E4 therefore reports three things instead of one:

1. the raw ratio, for continuity with the frozen result;
2. a **ceiling-normalised** ratio, which divides each stratum's purity by the most
   its class sizes allow — this is the fair comparison and the verdict uses it;
3. the **rank of the nearest same-class sibling**, which has no ceiling at all, plus
   the *head* unknown group whose classes hold 12–65 proposals and therefore have a
   ceiling of exactly 1.0.

Normalising raises the decoder space's advantage from 0.017 to **0.058** — the
conclusion survives, but the magnitude was overstated roughly fourfold, and every
number below is reported on the corrected basis.

---

## Phase 3 — the decisive statistic across nine feature spaces

| representation | dim | tail same-label | tail ceiling | tail normalised | head same-label (ceiling 1.0) | background same-label | **normalised advantage** | premise holds |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `imagenet_resnet50` | 2048 | 0.0647 | 0.235 | 0.262 | 0.0975 | 0.7962 | **0.330** | no |
| `dino_resnet50` | 2048 | 0.0588 | 0.235 | 0.241 | 0.1062 | 0.8097 | **0.297** | no |
| `prob_decoder_plus_dino` | 2304 | 0.0588 | 0.235 | 0.241 | 0.1127 | 0.8466 | 0.285 | no |
| `dino_whitened` | 64 | 0.0559 | 0.235 | 0.230 | 0.0952 | 0.8200 | 0.281 | no |
| `prob_decoder_whitened` | 64 | 0.0176 | 0.235 | 0.067 | 0.0350 | 0.8721 | 0.076 | no |
| `prob_posterior` | 20 | 0.0176 | 0.235 | 0.056 | 0.0282 | 0.8709 | 0.064 | no |
| **`prob_decoder`** (baseline) | 256 | 0.0147 | 0.235 | 0.052 | 0.0209 | 0.8879 | **0.058** | no |
| `prob_decoder_minus_top4` | 256 | 0.0118 | 0.235 | 0.045 | 0.0226 | 0.8658 | 0.052 | no |
| `prob_geometry` | 5 | 0.0029 | 0.235 | 0.006 | 0.0076 | 0.8083 | 0.008 | no |

**The representation matters, by a large factor.** Re-embedding the identical boxes
with an appearance encoder raises the decisive statistic 5–6× over PROB's decoder
(0.058 → 0.297 for DINO, 0.330 for ImageNet).

**No space clears the break-even line.** Every advantage is below 1.0, so in every
space tested a density- or homogeneity-based coherence term still ranks background
above the tail. The best space is 3× short.

**No transform of the decoder space helps.** Whitening moves 0.058 → 0.076; removing
the four leading principal components *hurts* (0.052). The alternative explanation
that the structure is present in the decoder space but drowned by high-variance
background directions is **refuted**: it is not there to recover.

## Phase 3 — the ceiling-free statistic, which is much sharper

For each unknown region whose class has at least two members, where does its nearest
same-class sibling sit in the similarity ordering over all 48 000 proposals?

| representation | tail median rank | head median rank | unknown median rank | sibling in top 10 |
|---|---:|---:|---:|---:|
| `dino_resnet50` | 213 | **6.0** | **6.0** | **0.545** |
| `prob_decoder_plus_dino` | 66 | 6.0 | 6.0 | 0.536 |
| `imagenet_resnet50` | **8.0** | 6.5 | 8.0 | 0.517 |
| `dino_whitened` | 27 | 9.5 | 10.0 | 0.509 |
| `prob_posterior` | 997 | 110.5 | 187.5 | 0.214 |
| `prob_decoder_whitened` | 973 | 79.5 | 95.0 | 0.212 |
| `prob_decoder_minus_top4` | 731 | 128.0 | 210.5 | 0.143 |
| **`prob_decoder`** (baseline) | 843 | 147.5 | **202.0** | **0.126** |
| `prob_geometry` | 4114 | 334.0 | 605.0 | 0.053 |

This is the strongest evidence in E4, and it is free of the frequency ceiling. In
PROB's decoder space the median unknown region's closest same-class sibling is the
**202nd** nearest neighbour; in DINO space it is the **6th** — a **34-fold**
improvement — and the fraction of unknown regions with a sibling inside their top-10
rises from 12.6 % to 54.5 %. On the head group, whose ceiling is exactly 1.0 and
where no artefact can hide, the median rank falls from 147.5 to 6.0.

So **the decoder embedding genuinely does not preserve semantic neighbourhoods, and
an off-the-shelf appearance encoder largely does.** H0's specific claim is supported.

Supporting statistics agree. Silhouette over the true unknown classes rises from
−0.366 (decoder) to −0.011 (DINO) and −0.002 (ImageNet); Davies–Bouldin falls from
4.70 to 3.39 and 3.19. Both remain at or below the "no structure" boundary, so the
classes are still not *separated* — they are merely no longer strongly interleaved.

## Phase 3 — why the premise still fails despite better neighbourhoods

Background's same-label fraction is 0.80–0.89 in **every** space, including DINO's.
Background is intrinsically the most locally homogeneous stratum of a detector's
proposal pool in any appearance space: 36 069 crops of sky, road, foliage and wall
genuinely do look like one another. Improving the *class* neighbourhoods 34× does
not change that, so the ratio stays below 1.

The local/global separability pair makes the same point from the other side:

| representation | nearest tail AUC (local) | centroid AUC (global) |
|---|---:|---:|
| `prob_decoder` | 0.798 | 0.716 |
| `dino_resnet50` | 0.835 | 0.715 |
| `dino_whitened` | 0.860 | 0.913 |
| `prob_decoder_whitened` | 0.834 | 0.927 |

Proximity to other tail regions already discriminates tail from background at
AUC ≈ 0.8 in the *baseline* space. The information a coherence term would need is
partly there even in the decoder. What the gate reads instead is *absolute local
homogeneity*, which background maximises — and that is a property of the
formulation, not of the space.

## Phase 3 — pseudo-classes are not repaired either

| representation | ARI (unknown classes) | NMI | rarity vs true rarity (Spearman) |
|---|---:|---:|---:|
| `prob_decoder` | 0.047 | 0.253 | **0.116** |
| `dino_resnet50` | 0.033 | 0.263 | −0.068 |
| `imagenet_resnet50` | 0.076 | 0.315 | **−0.148** |
| `prob_decoder_plus_dino` | 0.043 | 0.243 | 0.036 |

k-means with 20 clusters over a pool that is 75 % background still fails to recover
the unknown classes in any space (ARI ≤ 0.076), and the correlation between
*estimated* rarity and *true* class rarity does not improve — it goes slightly
negative for the two crop encoders. The rarity half of the distribution-aware term is
therefore **not** a representation problem: better neighbourhoods do not help a
clustering whose 20 centroids are spent on background modes.

## Phase 4 — projections

`figure_e4_<space>_{pca,tsne}_{balanced,natural}.png`, each panel coloured by oracle
stratum, coherence and objectness. In DINO's t-SNE the object-like regions occupy a
broad connected area with background forming its own tight island — the
object/background split is visible — but the tail regions (pink) are scattered
*among* the head regions rather than grouped, which is the same conclusion the
silhouette reports. The `natural`-sampled versions show the honest proportions: at
0.07 % of the pool the tail is nearly invisible, which is itself part of why the tail
metric is under-powered.

UMAP is absent: `umap-learn` is not installed and there is no network access. PCA
(linear, deterministic, cannot manufacture clusters) and t-SNE are both reported so a
structural claim can be checked against both.

## Phase 5 — the same strategies in a different space

<!-- PHASE5 -->

## Phase 6 — statistical treatment

Differences are paired by seed and severity, since every representation shares one
candidate pool, one export and one seed set; 95 % intervals come from the
t distribution on the paired differences with three seeds, and are reported so that a
difference smaller than its interval is not read as an effect. Two arms —
`v2:random` and `v2:objectness_area_prior` — cannot depend on the embedding, so their
spread across representations must be exactly zero; `e4_invariance_check.csv` tests
this, and a violation would void the comparison.

## Phase 7 — interpretation

<!-- PHASE7 -->
