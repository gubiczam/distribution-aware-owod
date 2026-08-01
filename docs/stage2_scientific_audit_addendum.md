# Stage 2 Scientific Audit Addendum

Verdict after Stage 1B leak-free remediation: CONDITIONAL GO.

Resolved blockers:

- Command drift is blocked by an authoritative `protocol:` object and command-parity validation. The exact audit mismatch (`TOWOD`, 20/20 class defaults, objectness temperature 1.3, inconsistent eval split) now fails tests.
- Pool decision is Option A using Stage 1B only. Stage 2 uses a leak-free official Task-1 train-side candidate split and a disjoint Stage 1B representation reference bank. The long-tail transformation is disabled to preserve Stage 1B parity.
- Canonical evaluation split is PROB `data/OWOD/ImageSets/OWDETR/owdetr_test.txt`, copied unchanged to the staged data root. Its SHA256 is recorded in every Stage 2 YAML and in `outputs/stage2_plan/protocol_preflight.json`.
- Evaluation XML annotations were generated from local COCO val2017 annotations using PROB's official `datasets/coco2voc.py:coco_to_voc_detection` conversion path. Evaluation JPEGs are present or symlinked from the local COCO val2017 image cache.
- Reference semantics are no longer overloaded. The fixed representation bank is for proposal-space scoring; cumulative selected candidate IDs are the labelled training set.
- Round manifests now record candidate, reference, labelled-before, selected, training, remaining, and evaluation IDs with SHA256 digests, overlaps, support counts, and resolved command lines.
- Scientific early stopping has been removed. Technical/numerical/catastrophic/resource stops remain preregistered.
- Seed policy is frozen: model/data-loader seed is passed to PROB, random acquisition uses deterministic seed-round-strategy shuffling, clustering uses `derive_seed('pool', model_seed, round_index)` shared across strategies for paired comparisons.

Stage 1B disposition:

- The old 500-image Stage 1 pool is permanently disqualified because it is a subset of `owdetr_test`. Stage 1B replaces it with official Task-1 train-side candidate and reference splits.
- Stage 1B preflight proves zero overlap: candidate/evaluation=0, reference/evaluation=0, candidate/reference=0.
- The remaining blocker is empirical rather than design-level: the real T4 smoke train/evaluate cycle has not yet run.
