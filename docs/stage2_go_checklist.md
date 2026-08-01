# Stage 2 GO Checklist

Current verdict: CONDITIONAL GO.

- [x] Train/predict/evaluate protocol parity is validated from the authoritative `protocol:` object.
- [x] Executable pool is frozen to leak-free Stage 1B, the official Task-1 train-side candidate pool.
- [x] Reference semantics are frozen: fixed Stage 1B representation bank is separate from cumulative labelled training IDs.
- [x] Stage 1B diagnostics correspond to the executable acquisition pool; the old eval-pool diagnostics are disqualified from acquisition/training.
- [x] Canonical evaluation split is resolved and frozen as PROB `OWDETR/owdetr_test.txt`.
- [x] Evaluation annotations/images are present in local staged assets.
- [x] Evaluation split is disjoint from acquisition/training. Current preflight: candidate/evaluation, reference/evaluation, and initial/evaluation overlaps are all zero.
- [x] Scientific early stopping is removed from preregistration.
- [x] Seed policy and variance-based escalation rule are preregistered.
- [x] Stage 2 configs validate locally. They must pass locally before the smoke run.
- [x] Tests pass locally.
- [x] Resolved commands are written into every round manifest before training.
- [x] Data-lineage ID lists, hashes, overlaps, and support counts are written per round.
- [ ] Real T4 smoke train/evaluate passes. It is the remaining blocker before a full GO.
- [ ] Smoke artifacts persist in Drive and resume refusal is observed.
- [x] Run matrix, runtime/storage estimate, and safe caching policy are updated.
