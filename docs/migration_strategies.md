# Migration notes — strategy names, scoring core, and round artifacts

Applies to the refactor that followed `docs/audit_phase1.md`. Read the first
section before running anything: it is the one change that will stop an existing
configuration from loading.

---

## 1. Unqualified strategy names are now invalid (intentional safety break)

`full` is rejected. So is any other name that exists in both semantics versions.

```
ConfigError: Strategy 'full' exists with different semantics in versions [1, 2].
Request it explicitly, e.g. 'v1:full' to reproduce published results or
'v2:full' for the current definition.
```

**Why this is a hard error and not a default.** `full` meant
`0.3*(1-|2c-1|) + 0.2*minmax(novelty) + 0.5*minmax(count^-1)*density_coherence`.
It now means `0.3*rank(entropy) + 0.2*rank(novelty) + 0.5*rank(rank(rarity)*coherence)`.
Those are different experiments. Silently picking either one would make a number
in the thesis untraceable to a definition, so the loader refuses to guess.

**What to write instead:**

| Intent | Use |
|---|---|
| Reproduce the pilot campaign numbers | `v1:full_p1`, `v1:full_p05`, `v1:rarity_no_coherence` |
| Run the current definition | `v2:full`, `v2:full_no_coherence`, … |
| Anything else | prefix with `v1:` or `v2:` |

`random` needs no prefix: it is behaviourally identical in both versions, and the
registry detects that by comparing behaviour fields (not descriptions), so it
resolves without complaint.

Repository-owned files already updated: `configs/experiment.yaml`, every test
fixture, `tests/test_campaign_integration.py`, this document. The pinned Colab
notebook `notebooks/contribution_a_multiround_prob.ipynb` clones a **pinned
DAOWOD commit** and is unaffected — it keeps working exactly as before.

### Version-1 specs ignore configuration overrides

`acquisition.uncertainty_method`, `coherence_method`, `normalisation`,
`pseudo_label_source`, `cluster_count`, `neighbour_count`, `top_k` and
`image_aggregation` are applied only to v2 specs. A v1 spec exists to stay fixed;
overriding it would destroy the reproducibility it is there for. The one
exception is `coherence_exponent`, which was always a declared v1 knob (`p=0.5`
vs `p=1.0`).

---

## 2. Strategy names, old to new

| Pre-audit name | Exact-reproduction name | Current-semantics equivalent |
|---|---|---|
| `random` | `random` | `random` |
| `uncertainty` | `v1:uncertainty` (deprecated) | `v2:uncertainty` (entropy) |
| `uncertainty_novelty` | `v1:uncertainty_novelty` | `v2:uncertainty_novelty` |
| `rarity` | `v1:rarity` (deprecated) | `v2:rarity` |
| `rarity_coherence` | `v1:rarity_coherence` (deprecated) | `v2:rarity_coherence` |
| `ungated_full` | `v1:ungated_full` | `v2:full_no_coherence` |
| `rarity_no_coherence` | `v1:rarity_no_coherence` | `v2:full_no_coherence` |
| `full` (p=1) | `v1:full_p1` | `v2:full` |
| `full` (p=0.5) | `v1:full_p05` | `v2:full` + `coherence_exponent: 0.5` |
| — | — | `v2:novelty`, `v2:coherence`, `v2:uncertainty_rarity`, `v2:uncertainty_coherence`, `v2:rarity_plus_coherence`, `v2:full_no_rarity`, `v2:full_no_uncertainty`, `v2:full_no_novelty`, `v2:proposal_formula` |

Three v1 names carry a `deprecated` message and print a note on load. They still
run; the note records *why* the audit considers them unsound
(`v1:uncertainty` is a monotone rescaling of the unknown score, `v1:rarity` is a
near-binary singleton indicator, `v1:rarity_coherence` is frequency-confounded).

`v2:proposal_formula` is new: it is the formula exactly as the research proposal
writes it, `S = U + λ·D + γ·w(ĉ)·coh`, i.e. with **both** an ungated and a gated
distribution term. No pre-audit variant implemented it.

---

## 3. Backward compatibility: what is guaranteed, and how it is proven

`daowod.acquisition` is now a shim that delegates to `daowod.components`,
`daowod.normalisation` and `daowod.scoring`. Every weighted sum routes through
`daowod.scoring.combine_components`, so there is exactly one implementation of
the arithmetic. What the shim still owns is the *version-1 conventions*: which
component is min-maxed (novelty, rarity), which is not (uncertainty, coherence),
and which entry point weight-normalises.

Guarantees, each backed by a test:

| Guarantee | Test |
|---|---|
| Every legacy formula reproduces to 1e-12 | `test_scoring_core.py::test_legacy_specs_reproduce_pre_audit_scores_exactly` |
| A live round with a v1 spec reproduces the exact pre-audit proposal scores | `test_smoke.py::test_run_active_round_full_regression_scores_image_scores_and_selection` |
| The ungated variant is genuinely ungated | `test_smoke.py::test_ungated_strategy_scores_are_invariant_to_the_coherence_method` |
| Fast aggregation equals the quadratic original | `test_scoring_core.py::test_fast_aggregation_matches_the_legacy_implementation` |
| Legacy public API keeps its behaviour | the pre-existing `test_smoke.py` suite, unchanged in intent |

### One historical inconsistency is preserved on purpose

`compute_proposal_scores("uncertainty_novelty")` divides by `alpha + beta`.
`_offline_strategy_scores("uncertainty_novelty")` does not. That divergence is
what the audit found (E2). Both are kept under their own names rather than
silently unified, because published offline comparisons used the un-normalised
form. New code should use `v2:uncertainty_novelty`, which is unambiguous.

---

## 4. Behavioural compatibility changes

1. **Unqualified ambiguous strategy names now raise `ConfigError`.** Section 1.
2. **`run_active_round` takes `spec: StrategySpec`, not `strategy: str` +
   `acquisition_config`.** It returns a `RoundResult` dataclass instead of a
   dict; field names are unchanged (`result.selected_image_ids` rather than
   `result["selected_image_ids"]`).
3. **Any registry strategy can now run in the live loop.** The old
   `{random, rarity_no_coherence, full}` allowlist is gone.
4. **A random strategy no longer exports proposals it discards.**
   `candidate_proposals.npz` is not written for `random` unless
   `export_proposals_for_random=True`. This was ~25 % of the pilot campaign's
   inference budget. The pinned notebook's `validate_round` expects that file for
   random rounds, which is one reason it stays pinned to the old commit.
5. **`proposal_scores.csv` columns changed.** Old:
   `uncertainty, novelty, rarity, coherence, rarity_bonus, score`. New: `raw_*`
   and `norm_*` for all five components, plus `proposal_score`, `image_score`,
   `cluster_id`, `cluster_size`, `posterior_entropy`, `isolated_outlier`,
   `proposal_selected`, `image_selected`, `run_id`, `seed`, `round`, `strategy`.
   `raw_gated` is now *always* the gated interaction `norm_rarity * coh^p`; the
   old `rarity_bonus` column meant "the rarity contribution this strategy
   actually applied", which is now read off the spec's weights instead.
6. **`ActiveLearningExperiment` was removed**, replaced by
   `ActiveLearningCampaign`. It was unused by the live campaign, untested, and
   the only caller of the grouped metrics. Its useful behaviour (grouped
   long-tail metrics, per-proposal diagnostics) moved into the single round path.
   `notebooks/colab_experiment.ipynb` used it and is superseded.
7. **Seed derivation changed.** `seed + round_index` collided across
   `(seed, round)` pairs; it is now
   `sha256(seed | round | strategy | semantics_version)`. Selections from a
   pre-audit run will not reproduce bit-for-bit under the new derivation — use
   the pinned commit for that, or read the recorded `scoring_seed` from the old
   manifest.
8. **`evaluation.require_detections` defaults to `true`.** A round now fails if
   the evaluator produced no detections artifact, because that silently disabled
   every long-tail metric (S3). Set it to `false` to opt out explicitly.
9. **The PROB bridge gained `evaluate --detections-output` and a `detections`
   subcommand.** Existing `evaluate` invocations keep working and now also write
   `<metrics stem>_detections.json` and a `detections_path` key. Use
   `--no-detections` to restore the old output exactly.
10. **`AcquisitionConfig.uncertainty_mode` was removed** as a stored field. The
    YAML key is still accepted and translated, and the translation is recorded in
    `ExperimentConfig.legacy_aliases`, which appears in the run manifest — so the
    provenance is preserved where it belongs rather than duplicated on the
    config object.
11. **`daowod.experiment.acquisition_config_specs` was removed** (no callers).
    Use `config.acquisition.resolved_specs()`.

---

## 5. Running things

```bash
# Validate a configuration without touching a GPU, and write a run manifest.
daowod-run validate --config configs/experiment.yaml --manifest outputs/manifest.json

# List every strategy, or just the thesis ablation matrix.
daowod-run strategies --required-only --verbose

# Full campaign: seeds x strategies x rounds, one command.
daowod-run campaign --config configs/experiment.yaml

# Offline multi-seed diagnostics over an exported pool (no retraining).
daowod-run diagnose \
  --candidates outputs/.../candidate_proposals.npz \
  --references outputs/.../reference_proposals.npz \
  --class-stats outputs/long_tail_seed_0/class_stats.csv \
  --annotations /path/to/data/OWOD/Annotations \
  --unknown-classes $(cat unknown_classes.txt) \
  --seeds 0 1 2 --budget 10 \
  --output outputs/diagnostics
```
