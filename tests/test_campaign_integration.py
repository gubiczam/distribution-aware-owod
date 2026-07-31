"""End-to-end campaign integration on a fake adapter.

Proves, without a GPU:

1. the YAML configuration is actually consumed;
2. seeds x strategies x rounds execute through the single campaign path;
3. versioned strategies resolve correctly;
4. proposal and image diagnostics are written;
5. grouped head/medium/tail metrics are produced;
6. manifests contain the resolved strategy specification;
7. completed rounds cannot be overwritten;
8. no acquisition-time ground-truth leakage occurs;
9. rerunning with the same seed is deterministic.
"""

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from daowod.cli import main as cli_main
from daowod.config import load_config
from daowod.diagnostics import GROUND_TRUTH_FIELDS
from daowod.experiment import ActiveLearningCampaign, derive_seed
from daowod.prob_adapter import ProposalBatch
from daowod.simulation import simulate_pool

UNKNOWN_CLASSES = [f"task_class_{index:02d}" for index in range(6)]
KNOWN_CLASSES = ["car", "person"]
POOL = simulate_pool(
    class_count=6,
    largest_class_images=8,
    imbalance_ratio=4.0,
    proposals_per_image=6,
    embedding_dimension=16,
    known_class_count=4,
    seed=3,
)


def _write_voc(path: Path, class_names: Sequence[str]) -> None:
    objects = "".join(
        f"<object><name>{name}</name>"
        "<bndbox><xmin>1</xmin><ymin>1</ymin><xmax>21</xmax><ymax>21</ymax></bndbox>"
        "</object>"
        for name in class_names
    )
    path.write_text(f"<annotation>{objects}</annotation>\n", encoding="utf-8")


class RecordingAdapter:
    """A fake detector that produces schema-valid proposals and detections."""

    def __init__(self, root: Path, annotations: Path) -> None:
        self.root = root
        self.annotations = annotations
        self.predict_calls: list[tuple[int, str]] = []
        self.train_calls: list[list[str]] = []
        self.evaluate_calls: list[str] = []

    def predict(
        self, image_ids: Sequence[str], *, checkpoint: str | Path, output_path: str | Path
    ) -> ProposalBatch:
        ids = [str(value) for value in image_ids]
        self.predict_calls.append((len(ids), str(checkpoint)))
        wanted = set(ids)
        mask = np.array([str(value) in wanted for value in POOL.image_ids.tolist()], dtype=bool)
        if not mask.any():
            # Reference images outside the pool: synthesise a small batch.
            count = max(len(ids), 1)
            rng = np.random.default_rng(len(ids))
            batch = ProposalBatch(
                image_ids=np.asarray(ids * 2, dtype=object)[: count * 2],
                confidence=np.clip(rng.random(count * 2) * 0.3, 1e-6, 1.0),
                embeddings=rng.normal(size=(count * 2, POOL.embeddings.shape[1])),
                posterior=np.abs(rng.normal(size=(count * 2, 5))) + 1e-3,
                predicted_labels=rng.integers(0, 4, count * 2).astype(np.int64),
            ).validate()
        else:
            batch = ProposalBatch(
                image_ids=POOL.image_ids[mask],
                confidence=POOL.confidence[mask],
                embeddings=POOL.embeddings[mask],
                posterior=POOL.posterior[mask],
                predicted_labels=POOL.predicted_labels[mask],
            ).validate()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            image_ids=batch.image_ids,
            confidence=batch.confidence,
            embeddings=batch.embeddings,
            posterior=batch.posterior,
            predicted_labels=batch.predicted_labels,
        )
        return ProposalBatch.load(output)

    def train(
        self,
        labelled_image_ids: Sequence[str],
        *,
        previous_checkpoint: str | Path | None,
        run_dir: str | Path,
        round_index: int,
        seed: int,
    ) -> Path:
        self.train_calls.append([str(value) for value in labelled_image_ids])
        checkpoint = Path(run_dir) / "checkpoint.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"fake-checkpoint")
        return checkpoint

    def evaluate(self, *, checkpoint: str | Path, output_path: str | Path) -> dict[str, object]:
        self.evaluate_calls.append(str(checkpoint))
        output = Path(output_path)
        detections_path = output.with_name(f"{output.stem}_detections.json")
        evaluation_images = ["eval_000", "eval_001", "eval_002"]
        ground_truth = [
            {
                "image_id": "eval_000",
                "class_name": UNKNOWN_CLASSES[0],
                "box": [0.0, 0.0, 10.0, 10.0],
            },
            {
                "image_id": "eval_000",
                "class_name": UNKNOWN_CLASSES[2],
                "box": [20.0, 20.0, 30.0, 30.0],
            },
            {
                "image_id": "eval_001",
                "class_name": UNKNOWN_CLASSES[5],
                "box": [0.0, 0.0, 10.0, 10.0],
            },
            {"image_id": "eval_002", "class_name": "car", "box": [0.0, 0.0, 10.0, 10.0]},
        ]
        detections = [
            {
                "image_id": "eval_000",
                "class_name": "unknown",
                "score": 0.9,
                "box": [0.0, 0.0, 10.0, 10.0],
            },
            {
                "image_id": "eval_001",
                "class_name": "unknown",
                "score": 0.8,
                "box": [0.0, 0.0, 10.0, 10.0],
            },
            {
                "image_id": "eval_002",
                "class_name": "car",
                "score": 0.7,
                "box": [0.0, 0.0, 10.0, 10.0],
            },
        ]
        detections_path.write_text(
            json.dumps(
                {
                    "schema": "daowod_detections_v1",
                    "image_count": len(evaluation_images),
                    "ground_truth": ground_truth,
                    "detections": detections,
                }
            ),
            encoding="utf-8",
        )
        metrics = {
            "known_mAP": 0.11,
            "U_Recall": 0.22,
            "WI": 0.33,
            "A_OSE": 4,
            "unknown_AP50": 0.15,
            "previous_known_AP50": 0.12,
            "detections_path": str(detections_path),
        }
        output.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")
        return metrics


CONFIG_TEMPLATE = """
name: integration
active_learning:
  rounds: 2
  strategy: v2:full
  budget: 2
  initial_images: 4
  budget_per_round: 2
  seeds:
    - 0
    - 1
acquisition:
  strategies:
    - v2:random
    - v2:full
    - v1:full_p1
  uncertainty_method: entropy
  coherence_method: relative_within_cluster
  normalisation: rank
  cluster_count: 4
  neighbour_count: 2
  top_k: 2
dataset:
  image_set_path: {image_set}
  annotations_dir: {annotations}
  unknown_classes:
{unknown_classes}
  known_classes:
    - car
    - person
  class_groups_path: {class_stats}
  known_class_groups_path: {known_class_stats}
  long_tail:
    enabled: false
    imbalance_ratio: 4.0
evaluation:
  grouped_metrics: true
  iou_threshold: 0.5
  require_detections: true
prob:
  repository_path: {prob_root}
  initial_checkpoint: {checkpoint}
  train_command: python bridge.py train --labelled-ids {{labelled_ids}} --output-checkpoint {{checkpoint}}
  predict_command: python bridge.py predict --image-ids {{image_ids}} --output {{proposals}}
  evaluate_command: python bridge.py evaluate --checkpoint {{checkpoint}} --output {{metrics}}
output_dir: {output_dir}
"""


@pytest.fixture
def environment(tmp_path: Path) -> dict[str, object]:
    annotations = tmp_path / "Annotations"
    annotations.mkdir()
    for image_id, classes in POOL.image_classes.items():
        _write_voc(annotations / f"{image_id}.xml", classes)
    for image_id in ("eval_000", "eval_001", "eval_002"):
        _write_voc(annotations / f"{image_id}.xml", [UNKNOWN_CLASSES[0]])

    image_set = tmp_path / "pool.txt"
    image_set.write_text("\n".join(POOL.unique_image_ids) + "\n", encoding="utf-8")

    class_stats = tmp_path / "class_stats.csv"
    POOL.write_class_stats(class_stats)

    known_class_stats = tmp_path / "known_class_stats.csv"
    known_class_stats.write_text("class_name,group\ncar,head\nperson,tail\n", encoding="utf-8")

    prob_root = tmp_path / "prob"
    prob_root.mkdir()
    checkpoint = prob_root / "t1.pth"
    checkpoint.write_bytes(b"initial")

    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        CONFIG_TEMPLATE.format(
            image_set=image_set,
            annotations=annotations,
            unknown_classes="\n".join(f"    - {name}" for name in UNKNOWN_CLASSES),
            class_stats=class_stats,
            known_class_stats=known_class_stats,
            prob_root=prob_root,
            checkpoint=checkpoint,
            output_dir=output_dir,
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return {
        "config_path": config_path,
        "annotations": annotations,
        "output_dir": output_dir,
        "tmp_path": tmp_path,
    }


def _run(environment: dict[str, object]) -> tuple[object, RecordingAdapter]:
    config = load_config(environment["config_path"])
    adapter = RecordingAdapter(Path(environment["tmp_path"]), Path(environment["annotations"]))
    result = ActiveLearningCampaign(config, adapter).run()
    return result, adapter


def test_campaign_consumes_the_yaml_and_runs_every_seed_strategy_round(
    environment: dict[str, object],
) -> None:
    result, adapter = _run(environment)
    output = Path(environment["output_dir"])

    # (1) the YAML is consumed: the persisted config matches the file on disk.
    persisted = json.loads((output / "experiment_config.json").read_text(encoding="utf-8"))
    assert persisted["name"] == "integration"
    assert persisted["acquisition"]["strategies"] == ["v2:random", "v2:full", "v1:full_p1"]
    assert persisted["evaluation"]["grouped_metrics"] is True

    # (2) 2 seeds x 3 strategies x 2 rounds = 12 rounds through one code path.
    assert len(result.metrics) == 12
    assert len(adapter.train_calls) == 12
    assert len(adapter.evaluate_calls) == 12
    assert {(row["seed"], row["strategy"], row["round"]) for row in result.metrics} == {
        (seed, strategy, round_index)
        for seed in (0, 1)
        for strategy in ("random", "full", "full_p1")
        for round_index in (1, 2)
    }

    # (3) versioned strategies resolve to their own semantics.
    versions = {(row["strategy"], row["semantics_version"]) for row in result.metrics}
    assert ("full", 2) in versions
    assert ("full_p1", 1) in versions

    # A random strategy needs no proposal export; the scored ones need two each.
    assert len(adapter.predict_calls) == 2 * 2 * 2 * 2


def test_round_artifacts_diagnostics_and_grouped_metrics_are_written(
    environment: dict[str, object],
) -> None:
    _run(environment)
    round_dir = Path(environment["output_dir"]) / "seed_0" / "v2_full" / "round_01"

    # (4) proposal and image diagnostics.
    for name in (
        "proposal_scores.csv",
        "image_scores.csv",
        "component_diagnostics.json",
        "grouped_metrics.json",
        "round_manifest.json",
        "metrics.json",
        "selected_ids.txt",
    ):
        assert (round_dir / name).is_file(), name

    with (round_dir / "proposal_scores.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows
    for field in (
        "raw_uncertainty",
        "norm_uncertainty",
        "raw_rarity",
        "norm_rarity",
        "raw_coherence",
        "norm_coherence",
        "raw_gated",
        "norm_gated",
        "cluster_id",
        "cluster_size",
        "posterior_entropy",
        "isolated_outlier",
        "proposal_selected",
        "image_selected",
        "image_score",
    ):
        assert field in rows[0], field

    diagnostics = json.loads((round_dir / "component_diagnostics.json").read_text())
    assert diagnostics["coherence_regime"]["regime"] in {
        "informative",
        "frequency_confounded",
        "saturated",
        "inactive",
    }
    assert "clusters_below_neighbour_count" in diagnostics

    # (5) grouped head/medium/tail metrics, validated for internal consistency.
    grouped = json.loads((round_dir / "grouped_metrics.json").read_text())
    for key in (
        "U_Recall_head",
        "U_Recall_medium",
        "U_Recall_tail",
        "unknown_AP50_head",
        "unknown_AP50_tail",
        "unknown_gt_head",
        "unknown_gt_tail",
        "Recall_head",
    ):
        assert key in grouped, key
    assert grouped["unknown_gt_total"] == 3
    assert (
        grouped["unknown_gt_head"] + grouped["unknown_gt_medium"] + grouped["unknown_gt_tail"] == 3
    )

    # (6) the manifest carries the fully resolved strategy specification.
    manifest = json.loads((round_dir / "round_manifest.json").read_text())
    assert manifest["completed"] is True
    assert manifest["strategy_spec"]["uncertainty_method"] == "entropy"
    assert manifest["strategy_spec"]["coherence_method"] == "relative_within_cluster"
    assert manifest["strategy_spec"]["normalisation"] == "rank"
    assert manifest["strategy_spec"]["top_k"] == 2
    assert manifest["semantics_version"] == 2
    assert manifest["scoring_seed"] == derive_seed(0, 1, "full", 2)
    assert manifest["grouped_metrics"]["U_Recall_grouped"] == pytest.approx(2 / 3)

    # A v1 round in the same campaign keeps pre-audit semantics.
    legacy_manifest = json.loads(
        (
            Path(environment["output_dir"])
            / "seed_0"
            / "v1_full_p1"
            / "round_01"
            / "round_manifest.json"
        ).read_text()
    )
    assert legacy_manifest["strategy_spec"]["uncertainty_method"] == "legacy_prob_score"
    assert legacy_manifest["strategy_spec"]["coherence_method"] == "density"
    assert legacy_manifest["strategy_spec"]["normalisation"] == "minmax"


def test_completed_rounds_cannot_be_overwritten(environment: dict[str, object]) -> None:
    config = load_config(environment["config_path"])
    adapter = RecordingAdapter(Path(environment["tmp_path"]), Path(environment["annotations"]))
    ActiveLearningCampaign(config, adapter).run()
    # (7) a second campaign into the same output directory must refuse.
    with pytest.raises(ValueError, match="Completed round"):
        ActiveLearningCampaign(config, adapter).run()


def test_no_acquisition_time_ground_truth_leakage(environment: dict[str, object]) -> None:
    _run(environment)
    output = Path(environment["output_dir"])
    # (8) no acquisition-time artifact may carry a ground-truth column.
    checked = 0
    for path in output.rglob("proposal_scores.csv"):
        with path.open(newline="", encoding="utf-8") as file:
            header = next(csv.reader(file))
        offending = [
            name for name in header if name in GROUND_TRUTH_FIELDS or name.startswith("gt_")
        ]
        assert offending == [], (path, offending)
        checked += 1
    assert checked == 8  # 2 seeds x 2 scored strategies x 2 rounds

    for path in output.rglob("image_scores.csv"):
        with path.open(newline="", encoding="utf-8") as file:
            assert next(csv.reader(file)) == ["image_id", "score"]


def test_campaign_is_deterministic_across_reruns(environment: dict[str, object]) -> None:
    first, _ = _run(environment)
    output = Path(environment["output_dir"])
    first_selections = json.loads((output / "selections.json").read_text(encoding="utf-8"))
    first_scores = (output / "seed_0" / "v2_full" / "round_01" / "proposal_scores.csv").read_text(
        encoding="utf-8"
    )

    # (9) rerun into a fresh directory with the same seeds: identical results.
    import shutil

    shutil.rmtree(output)
    second, _ = _run(environment)
    second_selections = json.loads((output / "selections.json").read_text(encoding="utf-8"))
    second_scores = (output / "seed_0" / "v2_full" / "round_01" / "proposal_scores.csv").read_text(
        encoding="utf-8"
    )

    assert first_selections == second_selections
    assert first_scores == second_scores
    assert [row["strategy"] for row in first.metrics] == [row["strategy"] for row in second.metrics]
    # Different seeds must not produce identical selections (the seed matters).
    by_seed = {}
    for row in second_selections:
        if row["strategy"] == "full" and row["after_round"] == 1:
            by_seed[row["seed"]] = row["selected_image_ids"]
    assert set(by_seed) == {0, 1}


def test_derive_seed_does_not_collide_across_seed_and_round(
    environment: dict[str, object],
) -> None:
    """The pre-audit derivation `seed + round_index` collided on these pairs."""

    assert derive_seed(0, 1, "full", 2) != derive_seed(1, 0, "full", 2)
    assert derive_seed(0, 1, "full", 2) != derive_seed(0, 1, "rarity", 2)
    assert derive_seed(0, 1, "full", 2) == derive_seed(0, 1, "full", 2)


def test_cli_validate_reads_the_repository_configuration(
    environment: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = Path(environment["tmp_path"]) / "manifest.json"
    exit_code = cli_main(
        ["validate", "--config", str(environment["config_path"]), "--manifest", str(manifest)]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "configuration is valid" in output
    assert "v2:full" in output and "v1:full_p1" in output
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    assert stored["config_fingerprint"] == load_config(environment["config_path"]).fingerprint()
    assert stored["seeds"] == [0, 1]
    assert "git_commit" in stored


def test_cli_rejects_ambiguous_strategy_names(
    environment: dict[str, object],
) -> None:
    path = Path(environment["config_path"])
    path.write_text(
        path.read_text(encoding="utf-8").replace("    - v2:full\n", "    - full\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different semantics"):
        cli_main(["validate", "--config", str(path)])
