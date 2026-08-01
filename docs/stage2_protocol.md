# Stage 2 Protocol

## Status

**CONDITIONAL GO.** Stage 2 is now rebuilt from leak-free Stage 1B evidence. The canonical evaluation split is staged and disjoint from candidate, reference, and initial-labelled IDs. The remaining blocker is the real T4 smoke execution.

## Stage 1 Audit Outcome

- Stage 1B uses real PROB proposals from official Task-1 training-side IDs only; the old 500-image eval-pool diagnostics are retained only as a contaminated baseline for comparison.
- Full tail lift at budget 50: 2.308; gate-only tail lift at budget 50: 2.308.
- Full object-positive rate at budget 50: 0.680; gate-only object-positive rate at budget 50: 0.020.
- Full and gate-only select the same images at budget 50: False.
- Rarity alignment with true inverse class frequency, post hoc Spearman: -0.054.
- Identical tail lift is an image-level composition tie, not proof that proposal-level scores agree.
- Ground truth is used only after scoring for diagnostics and never as an acquisition input.

## Authoritative Protocol Object

- Every Stage 2 YAML has a required `protocol:` block consumed by validation, acquisition, train, predict, evaluate, reports, and manifests.
- The frozen task settings are `dataset_protocol=OWDETR`, `previous_introduced_classes=0`, `current_introduced_classes=19`, `num_classes=81`, `objectness_temperature=1`, and the Task-1 SOWODB checkpoint digest recorded in each config.
- Command parity validation rejects train/predict/evaluate drift before execution; each round manifest records the fully resolved train, candidate-predict, reference-predict, and evaluate command lines plus a resolved-command parity report.

## Pool Decision

- Decision: Option A.
- Candidate pool: leak-free Stage 1B candidate split, `data/protocol/stage1b/stage1b_candidate_500.txt`.
- Representation reference bank: leak-free Stage 1B fixed bank, `data/protocol/stage1b/stage1b_reference_3500.txt`.
- Long-tail transformation: disabled. `dataset.long_tail.enabled=false` and `protocol.long_tail_transformation=none` preserve real Stage 1B comparability.
- Initial labelled split: none. Stage 2 starts with zero labelled candidate-pool images so first-round acquisition sees the exact Stage 1B candidate pool.
- Evaluation split: fixed `owdetr_test`, SHA256 `f58a4a97a8c4c84af337e6ab8dfb4ec97b5d96c6269a601d4f3d4dc3bddef49d`.
- Candidate/evaluation overlap under the canonical split: 0 images.
- Reference/evaluation overlap under the canonical split: 0 images.

## Reference Semantics

- Fixed representation reference bank: the 3,500-image Stage 1B reference split, sliced from the official train-side 4,000-image real PROB export, used only to compute proposal-space novelty/coherence references; it does not change by round.
- Growing labelled training set: the cumulative selected candidate IDs, starting empty and growing by exactly 20 images per round. This is what `train --labelled-ids` receives.
- Newly selected candidate images are added to the labelled training set, not to the fixed representation reference bank.
- Selected candidate images are removed from the candidate pool after selection and never selected twice.
- Candidate/reference overlap is forbidden. Candidate/evaluation and labelled/evaluation overlap are recorded and must be zero unless explicitly permitted by a future protocol revision.
- The detector checkpoint is used to recompute candidate and fixed-reference proposal features each scored round; reference proposals may be cached only for the identical checkpoint/reference-list pair.
- The fixed reference split can contain known and unknown objects; it is a representation bank, not an initial labelled set.

## Evaluation

- Evaluation split name: `owdetr_test`, fixed across all rounds, seeds, and strategies.
- Evaluation support is written before training to `outputs/stage2_plan/evaluation_support_report.csv`.
- Local preflight reports `asset_status=ready` and `protocol_status=ready`.
- Evaluation disjointness is satisfied; config validation should pass before T4 smoke.
- Official metrics come from the PROB bridge JSON (`known_mAP`, `U_Recall`, `WI`, `A_OSE`, plus unknown AP50 when exported). Custom grouped metrics use the same detections artifact, IoU 0.5, and frozen Stage 1 candidate-frequency thirds.

## Clustering Stabilisation

- Best observed zero-training stability method: `fixed_deterministic_kmeans` with mean selection Jaccard 1.000.
- Chosen smallest stability change: `fixed_deterministic_kmeans` with mean selection Jaccard 1.000.
- Current v2 baseline remains the scientific baseline; full fixed-pool clustering-seed Jaccard was 0.291 in Stage 1B.
- v3 included in training: False. Consensus is stable but would require a new production scorer path and adds runtime; fixed deterministic clustering is the smallest stability fix and is redundant with a controlled v2 selected set, so a separate v3 arm is not scientifically distinguishable.

## Stage 2 Arms

### v2:random

- StrategySpec: `v2:random`
- Weights: random
- Uncertainty: none
- Clustering: none
- Aggregation: top_k_mean/top3 not used for random
- Reason: Lower-bound active-learning control.
- Hypothesis: Detector gains beyond this arm come from acquisition signal, not round size.

### v2:uncertainty_objectness_weighted_entropy

- StrategySpec: `v2:uncertainty with uncertainty_method=objectness_weighted_entropy`
- Weights: uncertainty=1.0
- Uncertainty: objectness_weighted_entropy
- Clustering: none for scoring
- Aggregation: top_k_mean, top_k=3
- Reason: Stage 1B object-positive rate was 0.90 vs 0.76 for plain entropy.
- Hypothesis: Object-like ambiguous proposals improve downstream unknown recall without unnecessary known-mAP loss.

### v2:full

- StrategySpec: `v2:full`
- Weights: entropy=0.3, novelty=0.2, gated=0.5
- Uncertainty: entropy
- Clustering: current v2 KMeans, explicitly controlled pool seed
- Aggregation: top_k_mean, top_k=3
- Reason: Current contribution baseline under the real PROB pool.
- Hypothesis: The written full score improves tail-specific detector recall on a leak-free acquisition pool.

### v2:full_no_novelty

- StrategySpec: `v2:full_no_novelty`
- Weights: gated=0.5, uncertainty=0.3
- Uncertainty: entropy
- Clustering: current v2 KMeans, no novelty term
- Aggregation: top_k_mean, top_k=3
- Reason: Best Pareto-efficient executable zero-novelty Stage 1B configuration.
- Hypothesis: Removing unsupported novelty while retaining the gate yields clearer detector gains than v2:full.

## Training Protocol

- Seeds: [0, 1, 2]
- Rounds: 3
- Budget per round: 20 images.
- The 20-image round size is chosen because Stage 1 budget curves already separate OWE/full/gate behavior by budgets 20-50 while keeping the campaign small.
- Every strategy/seed starts from the same Task-1 checkpoint digest recorded in `run_matrix.csv`.
- No checkpoint is shared across strategies; each round resumes only from its own previous completed round.
- Completed-round overwrite protection is required: `round_manifest.json` with `completed=true` must abort reruns.
- Safe resume means rerunning the same command after a technical failure; completed rounds are immutable and permanently failed runs are reported rather than replaced silently.
- PROB schedule is 10 additional fine-tuning epochs at learning rate 2e-5 with evaluation every 2 epochs and the PROB model unfrozen; this is intentionally more meaningful than the one-epoch smoke setting.

## Cost

- Training runs: 36
- Evaluations: 36
- Estimated T4 hours: 63.0
- Expected storage: 52.8 GB
- Expected Colab sessions: 4-6 T4 sessions assuming 10-16 usable GPU hours per session.
- Predict calls: 2 for scored strategies (candidate pool and fixed reference bank); 0 for random unless export_proposals_for_random is enabled
- Safe caching: Reference proposal exports may be cached per checkpoint/round because the fixed reference bank does not change; candidate exports may be cached only within an identical checkpoint/candidate-list round.

## Preregistration

- Primary outcome: tail-U-Recall at IoU 0.5 on the fixed evaluation split
- Secondary outcomes: known mAP, aggregate U-Recall, WI, A-OSE, unknown AP50, head/medium/tail recall, class coverage
- Expected direction: tail-U-Recall higher than random without unacceptable known-mAP degradation
- Acceptable known-mAP degradation: 0.02 absolute.
- Mean, sample standard deviation, and 95% t-interval are computed over seeds; best-seed selection is forbidden.
- Stage 1 tail lift is diagnostic only and is not treated as proof of detector improvement.
- Seed policy: derive_seed('pool', model_seed, round_index), shared across strategies for the same seed/round; paired comparisons are preserved by running every strategy on the same seed/round pre-selection pool.
- Scientific early stopping is removed. All arms complete the same matrix unless a technical, numerical, catastrophic-safety, or resource stop occurs.

## T4 Smoke

- Smoke config: `configs/smoke_stage2_t4.yaml`.
- It uses one strategy, one seed, one round, budget 2, one epoch, the same checkpoint, same OWDETR/SOWODB command arguments, same predict/evaluate bridge, grouped metrics, Drive persistence, and resume/overwrite checks.
- Exact Colab cells are in `docs/stage2_t4_smoke.md`.

Do not start the 36-run campaign until the real T4 smoke cycle passes.
