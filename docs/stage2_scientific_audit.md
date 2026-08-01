# Stage 2 Scientific Audit

> Supersession note: this document records the pre-fix NO-GO audit. The current
> post-remediation verdict is in `docs/stage2_scientific_audit_addendum.md` and
> remains CONDITIONAL GO pending the real T4 smoke run. Staged evaluation assets
> and leak-free Stage 1B splits are now resolved.

## Executive Decision

**NO GO today.**

The current Stage 2 design is close to a defensible minimum-cost experiment, but I would
not spend the estimated 63 T4 hours yet. The protocol is not fully reconstructable from
the current files, and two execution details can change the scientific meaning of the
campaign:

1. The generated PROB `train_command` does not pass the OWDETR/SOWODB protocol arguments
   used in Stage 1. The bridge defaults are `dataset=TOWOD`, `prev_introduced_classes=20`,
   `current_introduced_classes=20`, and `objectness_temperature=1.3`; Stage 1 used
   `dataset=OWDETR`, `prev=0`, `current=19`, and `objectness_temperature=1`.
2. The generated configs enable `dataset.long_tail`, so the campaign runner derives a
   416-image eligible pool from the 500-image Stage 1 candidate split, then removes 20
   initial labelled images per seed. The actual first acquisition pool is therefore
   396 images, not the 500-image real Stage 1 pool that justified the strategy choices.

These are not implementation-feature requests. They are scientific-definition blockers:
another researcher could run a valid-looking campaign that is not the intended experiment.

## Part 1 - Entire Protocol Review

Files reviewed:

- `docs/stage2_protocol.md`
- `outputs/stage2_plan/preregistration.json`
- `outputs/stage2_plan/runtime_estimate.json`
- `outputs/stage2_plan/strategy_comparison.csv`
- `outputs/stage2_plan/run_matrix.csv`
- `outputs/real_stage1/real_stage1_report.md`
- `outputs/real_stage1/real_stage1_manifest.json`
- `configs/stage2_v2_*.yaml`
- `src/daowod/config.py`
- `src/daowod/experiment.py`
- `src/daowod/scoring.py`
- `src/daowod/dataset.py`
- `src/daowod/metrics.py`
- `src/daowod/prob_adapter.py`
- `PROB/daowod_prob_bridge.py`

### Reconstruction

Intended experiment:

- Initial checkpoint: `/Users/gubiczam/Downloads/results/SOWODB/t1.pth`,
  SHA256 `dba5390bffdfdf63058a995f241696df8d06b7fb859aecc8292d9ea02d459a22`.
- Strategies:
  - `v2:random`
  - `v2:uncertainty` with `uncertainty_method=objectness_weighted_entropy`
  - `v2:full`
  - `v2:full_no_novelty`
- Seeds: `0, 1, 2`.
- Rounds: `3`.
- Budget: `20` images per round.
- Training schedule: 10 additional epochs, LR `2e-5`, eval every 2 epochs, unfrozen PROB.
- Primary outcome: tail-U-Recall at IoU 0.5.
- Secondary outcomes: known mAP, aggregate U-Recall, WI, A-OSE, unknown AP50,
  head/medium/tail recall, class coverage.
- Estimated cost: 36 training runs, 36 evaluations, about 63 T4 hours, about 52.8 GB.

### Ambiguities and Gaps

1. **Protocol-specific PROB args are missing from training.** The train command omits
   `--data-root`, `--dataset OWDETR`, `--prev-introduced-classes 0`,
   `--current-introduced-classes 19`, `--num-classes 81`, and
   `--objectness-temperature 1`. This is the most serious ambiguity.
2. **Evaluation protocol args are incomplete.** The evaluate command passes `--data-root`
   and `--dataset OWDETR`, but not `prev/current/objectness_temperature`. If bridge
   defaults affect the evaluator/model build, metrics may not match Stage 1 protocol.
3. **Candidate pool definition is inconsistent.** Stage 1 evidence is from 500 candidate
   images. The campaign configs enable a long-tail builder that produces 416 eligible
   images from those 500 and then removes 20 initial labelled images. First acquisition
   therefore sees 396 images.
4. **Reference pool definition is inconsistent.** Stage 1 reference was 4,000 images.
   The campaign runner uses the current labelled set as `reference_ids`. At round 1 this
   is only the 20 initial images, not the 4,000-image Stage 1 reference set.
5. **Run matrix is not executable.** It records intended split paths and checkpoint digest,
   but the runner does not consume `run_matrix.csv`. The configs are the executable files.
6. **Initial labelled set is underspecified scientifically.** It is seeded random from the
   derived pool, but the protocol does not state whether this initial set is identical
   across strategies within each seed. The implementation does make it identical per seed,
   but the protocol should say so.
7. **Evaluation split is underspecified.** `test_set` is not in configs, so bridge default
   `owod_all_task_test` is used. This should be explicit.
8. **Colab commands contain placeholders.** `<DAOWOD_REPO_URL>` and `<PROB_REPO_URL>` are
   not enough for independent execution.
9. **Training schedule is named, not justified.** Ten epochs is better than a smoke run, but
   no evidence shows it is enough for stable fine-tuning or not too much for 20-image rounds.
10. **Stopping criteria can bias the experiment.** Stopping after round 1 if all non-random
    arms trail random directionally risks stopping on noise with only 3 seeds.
11. **Primary metric support is unknown.** Tail-U-Recall is defensible, but the tail support
    count in the evaluation split is not recorded in the protocol.
12. **Class-name aliases are not protocolized.** Stage 1 needed COCO/VOC aliases
    (`airplane/aeroplane`, `motorcycle/motorbike`, etc.). Stage 2 grouped metrics must
    prove the same normalization.

If another researcher received only the files, they could run a campaign, but not necessarily
the intended campaign.

## Part 2 - Experimental Design Audit

| Decision | Current choice | Why chosen | Stronger alternative? | Judgment |
|---|---:|---|---|---|
| Strategies | 4 arms | Minimum non-redundant set: random, OWE, current full, no-novelty full | Add only if it resolves a specific mechanism | Reasonable |
| Seeds | 3 | Minimum required by prompt and budget | 5 seeds improves uncertainty estimates | Weak but acceptable for TDK if framed as pilot |
| Rounds | 3 | Shows learning trajectory, total 60 images | 2 rounds reduces cost; 4 rounds improves curves | 3 is reasonable |
| Budget | 20/round | Stage 1 separation visible by 20-50 | 30/round may give more detector signal | 20 is cost-efficient, but may be underpowered |
| Candidate pool | Config-derived 416 eligible images, 396 after initial labels | Long-tail control | Use exact 500 Stage 1 candidate pool or re-run Stage 1 on executable pool | Current mismatch is blocker |
| Reference pool | Initial labelled set, then cumulative labelled | Real active-learning setup | Use fixed 4,000 reference only for scoring, but then it is transductive/not live AL | Must be stated; Stage 1 evidence is not directly transferable |
| Evaluation protocol | Bridge default test set | Convenient | Explicit fixed test set and support counts | Blocker until explicit |
| Metrics | Tail-U-Recall primary, standard OWOD secondary | Matches research question | Add per-round selection diagnostics and support counts | Good, incomplete support reporting |
| Grouped metrics | Head/medium/tail unknown recall/AP | Needed for long-tail claim | Predefine class groups from training pool and freeze for all seeds | Must avoid seed-dependent groups if comparing strategies |
| Clustering | Current v2 with deterministic pool seed in runner | Avoid unstable seed cherry-picking | Fixed deterministic KMeans or consensus | Current protocol wording is confusing; deterministic control must be explicit |
| Image aggregation | top-3 mean | Already used in Stage 1 | Sensitivity check top-1 or top-5 offline only | Acceptable |
| Uncertainty | OWE vs entropy-in-full | Stage 1 OWE object-positive rate 0.72 vs entropy 0.52 | Include plain entropy only if extra cheap offline/diagnostic | Good |
| Rarity | KMeans pseudo-cluster rarity | Core hypothesis | Add no-gate or gate-only only if resources permit | Scientifically weak after Stage 1 |
| Coherence | Relative within-cluster | Avoid density confound | Consensus/unknown-filtered not ready | Acceptable as tested mechanism |
| Weighting | Full 0.3/0.2/0.5, no-novelty 0.3/0.5 | Stage 1 Pareto and proposal | Normalize weights? It does not change ranks if scale constant | Acceptable |
| Checkpoint reuse | Same initial checkpoint for all arms/seeds | Fair comparison | Also record digest in every round manifest | Good |
| Stopping criteria | Stop after round 1 if all trail random | Cost control | Continue full 3 rounds unless technical failure; analyze sequentially only as appendix | Current criterion risks bias |

## Part 3 - Statistical Power

With 3 seeds, inferential power is weak. For paired strategy comparisons with `n=3`,
the 95% CI half-width is approximately:

`4.303 * s_paired / sqrt(3) = 2.48 * s_paired`

So a mean difference must be very large relative to seed-to-seed variability before it
is statistically convincing. A rough 80% powered paired two-sided test needs an effect
on the order of 3 to 4 paired standard deviations. This is not a conventional
publication-grade sample size.

Detectable practical effects, assuming typical OWOD seed variability:

| Metric | If paired SD is... | 95% CI half-width with 3 seeds | Practical implication |
|---|---:|---:|---|
| known mAP | 0.5-1.0 mAP pts | 1.2-2.5 pts | 2 pt degradation threshold is barely resolvable |
| U-Recall | 1-3 pts | 2.5-7.4 pts | Only large changes are credible |
| tail-U-Recall | 3-8 pts | 7.4-19.8 pts | Primary endpoint likely underpowered |
| WI | 0.005-0.02 | 0.012-0.050 | Only large WI changes are interpretable |
| A-OSE | 50-200 detections | 124-496 | Large variance likely; use paired plots |

Smallest modification if power is weak:

1. Keep 3 seeds for full 3-round campaign.
2. Add **2 extra seeds only for the final-round checkpoint** of the two most important
   comparisons if early variance is high: OWE vs random and `v2:full_no_novelty` vs
   `v2:full`.
3. Predefine this escalation: add seeds 3 and 4 only when the round-1 paired SD of
   tail-U-Recall exceeds 5 points or when confidence intervals reverse the apparent
   ordering.

This preserves the minimum campaign but avoids a paper built on three noisy points.

## Part 4 - Threats to Validity

### Internal Validity

| Threat | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Train/eval protocol defaults differ from Stage 1 | High | Critical | Pass all OWDETR/SOWODB args explicitly in every train/predict/evaluate command |
| Executable pool differs from Stage 1 pool | High | Critical | Freeze exact acquisition pool or rerun Stage 1 audit on the executable pool |
| Initial labelled set confounds strategies | Medium | High | Same initial labels per seed across strategies; record files and digests |
| Reference set differs from Stage 1 reference | High | High | State live-AL reference semantics; do not cite Stage 1 novelty/gate evidence as directly predictive |
| Random arm skips proposal export | Medium | Medium | Fine scientifically, but record selection RNG and selected IDs |
| Early stopping based on noisy round 1 | Medium | High | Remove scientific stopping; keep only technical/resource stop |
| KMeans nondeterminism/GPU nondeterminism | Medium | Medium | Record pool seed, software versions, deterministic flags where feasible |
| GT leakage through class groups | Low-Medium | High | Define class groups from predeclared training pool only; never per-strategy selected labels |

### External Validity

| Threat | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Results limited to SOWODB/OWDETR Task 1 | High | Medium | Present as Task-1 long-tail AL study; do not claim general OWOD superiority |
| Candidate pool only 500/416 images | High | Medium | Be explicit: minimum-cost pilot, not full benchmark |
| Tail classes in evaluation may be sparse | Medium | High | Report support counts and per-class recall |
| PROB-specific proposal behavior | High | Medium | Claim contribution as PROB-backed acquisition, not detector-agnostic proof |

### Construct Validity

| Threat | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Stage 1 tail lift may not measure detector improvement | High | High | Already stated; keep as diagnostic only |
| Tail-U-Recall ignores known-class cost | Medium | High | Primary success requires known mAP drop <= 2 pts |
| Pseudo-cluster rarity not aligned with true rarity | High | High | Report Stage 1 Spearman -0.024; interpret full/no-novelty cautiously |
| OWE may select easy known objects, not useful unknowns | Medium | Medium | Analyze unknown AP50, U-Recall, selected GT composition post hoc |

### Statistical Conclusion Validity

| Threat | Likelihood | Severity | Mitigation |
|---|---|---|---|
| n=3 seeds underpowered | High | High | Paired plots, CIs, optional seed escalation |
| Multiple comparisons | Medium | Medium | Predefine primary comparisons and mark others exploratory |
| Best-seed cherry-picking | Low | High | Report mean/std/CI; no strategy chosen by best seed |
| Non-normal metrics | Medium | Medium | Use paired seed-level differences and bootstrap/permutation as appendix |

## Part 5 - Failure Analysis

| Outcome | Can conclude | Cannot conclude |
|---|---|---|
| Random wins | Acquisition signals failed under this protocol; selection may be noisy or harmful | All distribution-aware OWOD is invalid |
| OWE wins | Object-like uncertainty is stronger than cluster rarity for this setup | OWE is universally best or tail-aware |
| `v2:full` wins | Original combined score has downstream value despite weak Stage 1 diagnostics | The gate specifically caused the gain |
| `v2:full_no_novelty` wins | Novelty removal helped; novelty was unnecessary or harmful here | Coherence/rarity alone is proven causal |
| No strategy differs | Campaign is underpowered or effects are small | Strategies are equivalent in general |
| Known mAP collapses | AL additions damage known-class retention or training schedule too aggressive | Unknown selection itself is the only cause |
| Tail improves, known mAP stable | Strongest positive outcome for Contribution A | Mechanism is isolated without further ablations |
| Tail improves, aggregate U-Recall does not | Method may specifically help rare unknown classes | Overall OWOD performance improved |
| Aggregate U-Recall improves, tail does not | Acquisition helps unknowns but not long tail | Contribution A tail hypothesis is supported |
| WI improves but recall drops | Fewer unknown mistakes, but less recall | Detector is better overall |
| A-OSE drops, known mAP drops | Trade-off shifted predictions, ambiguous value | Scientific success unless within mAP tolerance |
| High variance across seeds | Design may be unstable/underpowered | Any single-seed winner is meaningful |
| Training failures clustered by strategy | Strategy may produce bad data or config bug | Other completed strategies are directly comparable without caveats |

## Part 6 - Ablation Completeness

Current ablations are enough for a minimum TDK experiment only if the question is:

"Does the best Stage 1-motivated acquisition strategy improve tail-U-Recall over random
and OWE under a fixed PROB protocol?"

They are not enough to prove every component's causal role.

Minimum additional ablations by information gain per GPU hour:

1. **No expensive addition: offline executable-pool Stage 1 audit.** Recompute top-budget
   diagnostics on the exact 416/396-image executable pool and 20-image initial references.
2. **No expensive addition: selection-overlap report for each actual round.** This tells
   whether arms remain non-redundant after round 1.
3. **Cheap detector addition if budget permits: `v2:uncertainty` plain entropy for one seed
   and one round only.** Confirms OWE's training advantage is not just Stage 1 proposal purity.
4. **Cheap schedule ablation: one strategy, one seed, 5 vs 10 epochs.** Ensures schedule is
   not driving all results.

Do not add novelty-only, `full_p05`, gate-only, or consensus unless the main campaign gives
a specific reason. Their Stage 1 evidence is too weak for the GPU cost.

## Part 7 - Figure Planning

| Figure | Axes | Caption | Message |
|---|---|---|---|
| Main learning curve | x=round/cumulative labels, y=tail-U-Recall | Tail-U-Recall over active-learning rounds | Primary result |
| Known retention curve | x=round, y=known mAP | Known mAP cost of tail acquisition | Trade-off safety |
| Unknown recall curve | x=round, y=aggregate U-Recall | Aggregate unknown recall | OWOD utility beyond tail |
| Head/medium/tail grouped bars | x=strategy, y=U-Recall by group | Final-round grouped recall with support counts | Long-tail specificity |
| Strategy overlap heatmap | x/y=strategy, color=selected-image Jaccard | Round-wise selection redundancy | Shows arms test different selections |
| Stage 1 diagnostic scatter | x=objectness, y=entropy/OWE, color=post-hoc object | Why OWE was included | Links Stage 1 to design |
| Clustering stability plot | x=method, y=selection Jaccard | Current vs stabilised clustering | Justifies no v3 arm |
| Component contribution plot | x=component, y=mean contribution in selected images | Full vs no-novelty score anatomy | Mechanism interpretation |
| WI/A-OSE trade-off | x=WI, y=A-OSE or U-Recall | Open-world error trade-offs | Avoids single-metric story |

## Part 8 - Table Planning

| Table | Values | Priority |
|---|---|---|
| Protocol table | strategies, weights, uncertainty, clustering, aggregation, budget | Primary |
| Main final metrics | tail-U-Recall, known mAP, U-Recall, WI, A-OSE, unknown AP50 | Primary |
| Pairwise differences | OWE-random, full-OWE, no-novelty-full with mean/std/CI | Primary |
| Grouped supports | unknown GT support and matched count by head/medium/tail | Primary |
| Run matrix summary | seeds, rounds, checkpoint digest, pool digest | Secondary |
| Stage 1 audit | OWE object-positive, novelty lift, clustering Jaccard, rarity alignment | Secondary |
| Runtime/storage | observed GPU hours, storage, failures/retries | Secondary |
| Per-class tail recall | recall per tail class | Appendix |
| Full selection IDs | selected images by strategy/seed/round | Appendix |
| Failed/retried runs | failure reason and handling | Appendix |

## Part 9 - Reviewer Simulation

### Reviewer A - OWOD Expert

Criticism: "This is not a standard full OWOD benchmark; a 500-image or 416-image pool is too small."

Response: Correct. The claim must be limited to a minimum-cost Task-1 active-learning study.
Missing evidence for a full benchmark claim: full SOWODB/TOWOD campaign.

Criticism: "Tail-U-Recall is not enough; known mAP and open-set errors matter."

Response: The preregistration includes known mAP, WI, A-OSE, unknown AP50, and aggregate U-Recall.
Missing evidence: explicit support counts and acceptable trade-off decision rule in the final tables.

Criticism: "The training command may not match OWDETR Task 1."

Response: Current evidence supports the criticism. The command must be fixed before running.

### Reviewer B - Active Learning Expert

Criticism: "Three seeds are too few for active learning."

Response: It is a minimum-cost design, not a definitive benchmark. Missing evidence: variance estimate
from round 1 and optional escalation to seeds 3-4.

Criticism: "The reference set changes from Stage 1 to Stage 2."

Response: Correct. Stage 1 used a large fixed reference to diagnose components; Stage 2 is live AL
with labelled-set reference. Missing evidence: executable-pool/round-1 dry audit.

Criticism: "Early stopping after round 1 biases the result."

Response: I agree. Remove scientific early stopping; retain only technical/resource stop.

### Reviewer C - ML Methodology Expert

Criticism: "The primary hypothesis was weakened after Stage 1; rarity is not aligned with true rarity."

Response: True. Stage 1 Spearman is -0.024, so any claim must be empirical downstream improvement,
not validated rarity estimation.

Criticism: "Multiple comparisons with n=3 can produce arbitrary winners."

Response: Predefine primary comparisons and report paired CIs; do not select by best seed.
Missing evidence: seed escalation or a clear pilot framing.

Criticism: "Protocol files are not sufficient for exact reproduction."

Response: Currently correct. Missing evidence: explicit fixed train/eval args, test set, pool digests,
initial labelled IDs, and concrete repo URLs/commits.

## Part 10 - Final GO / NO-GO

I would **not** spend the 63 GPU hours today.

Blocking issues ranked by expected impact:

1. **Critical: train/evaluate protocol arguments are incomplete.** This can run the wrong
   OWOD task semantics.
2. **Critical: executable pool differs from Stage 1 evidence.** Either freeze the exact
   Stage 1 pool for Stage 2 or rerun the Stage 1 audit on the exact executable pool.
3. **High: reference-set semantics differ from Stage 1.** The protocol must state that
   Stage 2 uses live labelled references, not the 4,000-image Stage 1 reference.
4. **High: n=3 is underpowered for publication-level statistical claims.** Add a predefined
   seed-escalation rule or frame as a minimum-cost pilot.
5. **Medium: early stopping can bias conclusions.** Remove scientific early stopping.
6. **Medium: evaluation split and support counts are not explicit.** Fix before running.
7. **Medium: Colab commands are placeholders.** Replace with concrete repo URLs/commits or
   archive instructions.

After blockers 1-3 are resolved and a tiny real T4 smoke train/evaluate passes, the design
can become **CONDITIONAL GO**. It should become full **GO** only if the first-round variance
does not make the 3-seed design scientifically uninterpretable.
