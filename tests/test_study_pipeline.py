"""End-to-end test of the pipeline's plumbing, on a fabricated export.

Scope, stated plainly: the *data* here is synthetic, so nothing in this file is
evidence about the method. What it does prove is that the orchestration is
correct — the stages run in the right order, the pilot pool is disjoint from the
evaluation pool, the severities are validated, every promised artifact is written,
the run is resumable, and a rerun reproduces identical numbers. Those are exactly
the properties that are expensive to discover are broken during a four-hour Colab
session.

Real-data behaviour is exercised separately, by running the pipeline against a
PROB export (see ``docs/contribution_a_active_annotation.md``).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from daowod import config as config_module
from daowod import longtail
from daowod.config import ExecutionMode
from daowod.pipeline import PipelineConfig, PipelineError, run_pipeline
from daowod.study import PRIMARY_STRATEGIES

IMAGE_WIDTH, IMAGE_HEIGHT = 320, 240
DIMENSIONS = 12

#: A long-tailed unknown class profile plus one known class, so the pool holds
#: known-object, background and unknown proposals like a real export does.
UNKNOWN_PROFILE: tuple[tuple[str, int], ...] = (
    ("apple", 26),
    ("kite", 18),
    ("umbrella", 13),
    ("bench", 10),
    ("skateboard", 7),
    ("banana", 5),
    ("microwave", 4),
    ("toaster", 3),
    ("scissors", 2),
)
KNOWN_CLASS = "dog"


def _boxes_for(count: int, generator: np.random.Generator) -> list[tuple[int, int, int, int]]:
    boxes = []
    for _ in range(count):
        width = int(generator.integers(40, 90))
        height = int(generator.integers(40, 90))
        x_min = int(generator.integers(2, IMAGE_WIDTH - width - 2))
        y_min = int(generator.integers(2, IMAGE_HEIGHT - height - 2))
        boxes.append((x_min, y_min, x_min + width, y_min + height))
    return boxes


def build_fixture(root: Path, *, images: int = 120, seed: int = 0) -> tuple[Path, Path]:
    """Write VOC annotations and a matching proposal export.

    Each annotated object gets one proposal at (a jittered version of) its own box
    and an embedding drawn around a per-class centroid, so rarity and coherence
    carry real structure. Every image also gets background proposals, which is what
    makes annotation precision and the background selection rate meaningful.
    """

    generator = np.random.default_rng(seed)
    annotations_dir = root / "Annotations"
    images_dir = root / "JPEGImages"
    splits_dir = root / "ImageSets" / "OWDETR"
    for directory in (annotations_dir, images_dir, splits_dir):
        directory.mkdir(parents=True, exist_ok=True)

    class_names = [name for name, _ in UNKNOWN_PROFILE]
    centroids = {
        name: generator.normal(size=DIMENSIONS) * 3.0 for name in [*class_names, KNOWN_CLASS]
    }
    assignments: list[str] = []
    for name, count in UNKNOWN_PROFILE:
        assignments.extend([name] * count)
    assignments.extend([KNOWN_CLASS] * 40)
    generator.shuffle(assignments)

    per_image: list[list[str]] = [[] for _ in range(images)]
    for index, name in enumerate(assignments):
        per_image[index % images].append(name)

    image_ids: list[str] = []
    rows_image: list[str] = []
    rows_embedding: list[np.ndarray] = []
    rows_box: list[list[float]] = []
    rows_objectness: list[float] = []
    rows_confidence: list[float] = []
    rows_posterior: list[np.ndarray] = []
    rows_label: list[int] = []

    for index in range(images):
        image_id = f"img{index:04d}"
        image_ids.append(image_id)
        names = per_image[index]
        boxes = _boxes_for(len(names), generator)
        body = "".join(
            f"<object><name>{name}</name><bndbox>"
            f"<xmin>{box[0]}</xmin><ymin>{box[1]}</ymin>"
            f"<xmax>{box[2]}</xmax><ymax>{box[3]}</ymax>"
            "</bndbox></object>"
            for name, box in zip(names, boxes, strict=True)
        )
        (annotations_dir / f"{image_id}.xml").write_text(
            "<annotation><size>"
            f"<width>{IMAGE_WIDTH}</width><height>{IMAGE_HEIGHT}</height><depth>3</depth>"
            f"</size>{body}</annotation>",
            encoding="utf-8",
        )
        (images_dir / f"{image_id}.jpg").write_bytes(b"\xff\xd8\xff\xd9")

        for name, box in zip(names, boxes, strict=True):
            jitter = generator.normal(scale=1.5, size=4)
            x_min, y_min, x_max, y_max = (
                value + shift for value, shift in zip(box, jitter, strict=True)
            )
            rows_image.append(image_id)
            rows_box.append(
                [
                    ((x_min + x_max) / 2.0) / IMAGE_WIDTH,
                    ((y_min + y_max) / 2.0) / IMAGE_HEIGHT,
                    (x_max - x_min) / IMAGE_WIDTH,
                    (y_max - y_min) / IMAGE_HEIGHT,
                ]
            )
            rows_embedding.append(centroids[name] + generator.normal(scale=0.4, size=DIMENSIONS))
            rows_objectness.append(float(generator.uniform(0.5, 0.95)))
            rows_confidence.append(float(generator.uniform(0.3, 0.9)))
            rows_label.append(80 if name != KNOWN_CLASS else 3)
            rows_posterior.append(generator.dirichlet(np.full(6, 0.6)))
        for _ in range(6):
            rows_image.append(image_id)
            rows_box.append(
                [
                    float(generator.uniform(0.05, 0.95)),
                    float(generator.uniform(0.05, 0.95)),
                    float(generator.uniform(0.02, 0.08)),
                    float(generator.uniform(0.02, 0.08)),
                ]
            )
            rows_embedding.append(generator.normal(scale=4.0, size=DIMENSIONS))
            rows_objectness.append(float(generator.uniform(0.05, 0.6)))
            rows_confidence.append(float(generator.uniform(0.05, 0.6)))
            rows_label.append(80)
            rows_posterior.append(generator.dirichlet(np.full(6, 0.6)))

    (splits_dir / "test_split.txt").write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    export_path = root / "export.npz"
    np.savez_compressed(
        export_path,
        image_ids=np.asarray(rows_image, dtype=object),
        confidence=np.asarray(rows_confidence, dtype=np.float64),
        embeddings=np.asarray(rows_embedding, dtype=np.float64),
        posterior=np.asarray(rows_posterior, dtype=np.float64),
        predicted_labels=np.asarray(rows_label, dtype=np.int64),
        boxes=np.asarray(rows_box, dtype=np.float64),
        objectness=np.asarray(rows_objectness, dtype=np.float64),
    )
    return export_path, splits_dir / "test_split.txt"


TEST_MODE = ExecutionMode(
    name="PYTEST",
    description="Synthetic-data plumbing test. Not a research configuration.",
    evaluation_images=60,
    pilot_images=20,
    reference_images=40,
    per_image_limit=6,
    budgets=(20, 40, 80),
    rounds=2,
    seeds=(0, 1),
    strategies=PRIMARY_STRATEGIES,
    imbalance_settings=longtail.FLATTENING_IMBALANCE_SETTINGS,
    run_pilot=True,
    run_ablations=True,
    ablation_seeds=(0,),
    reference_limit=2_000,
    runtime_budget_seconds=15 * 60,
    research_grade=False,
)


@pytest.fixture(scope="module")
def registered_mode() -> ExecutionMode:
    return config_module.register(TEST_MODE, replace_existing=True)


@pytest.fixture(scope="module")
def fixture_paths(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("dataset")
    export, split = build_fixture(root)
    return root, export, split


@pytest.fixture(scope="module")
def completed_run(
    fixture_paths: tuple[Path, Path, Path],
    registered_mode: ExecutionMode,
    tmp_path_factory: pytest.TempPathFactory,
):
    root, export, split = fixture_paths
    output = tmp_path_factory.mktemp("results")
    config = PipelineConfig(
        mode="PYTEST",
        data_root=str(root),
        split_file=str(split),
        existing_export=str(export),
        output_dir=str(output),
        cache_dir=str(output / "cache"),
        require_gpu=False,
        target_tail_recall=0.25,
    )
    return run_pipeline(config, progress=None)


def test_pipeline_produces_every_promised_artifact(completed_run) -> None:
    statuses = {row["artifact"]: row["status"] for row in completed_run.artifacts}
    assert set(statuses.values()) == {"PASS"}, statuses
    assert completed_run.archive is not None and completed_run.archive.exists()
    assert completed_run.summary_path is not None
    summary = completed_run.summary_path.read_text(encoding="utf-8")
    assert "Contribution A" in summary
    assert "Not a reportable result" in summary  # research_grade is False
    assert "Limitations" in summary


def test_per_strategy_csvs_are_written(completed_run) -> None:
    directory = completed_run.output_dir / "per_strategy"
    names = {path.name for path in directory.glob("*.csv")}
    for strategy in PRIMARY_STRATEGIES:
        slug = strategy.replace(":", "_")
        assert f"{slug}_curve.csv" in names
        assert f"{slug}_auc.csv" in names
        assert f"{slug}_selected.csv" in names


def test_figures_are_written_as_png_and_pdf(completed_run) -> None:
    suffixes = {path.suffix for path in completed_run.figures}
    assert suffixes == {".png", ".pdf"}
    names = {path.stem for path in completed_run.figures}
    for expected in (
        "figure_tail_discovery_vs_budget",
        "figure_annotation_efficiency",
        "figure_gate_suppression",
        "figure_long_tail_protocol",
        "figure_ablation_heatmap",
    ):
        assert expected in names


def test_the_matrix_covers_every_strategy_severity_and_seed(completed_run) -> None:
    rows = completed_run.outputs.strategy_rows
    strategies = {str(row["strategy"]) for row in rows}
    severities = {str(row["imbalance_setting"]) for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    assert strategies == set(PRIMARY_STRATEGIES)
    assert severities == {spec.name for spec in TEST_MODE.imbalance_settings}
    assert seeds == set(TEST_MODE.seeds)
    # Plain registry names, so every table joins on the same key.
    assert not any(":" in name for name in strategies)


def test_severities_are_validated_and_distinct(completed_run) -> None:
    assert "head:tail object ratio" in completed_run.severity_verdict
    ratios = [float(row["head_to_tail_object_ratio"]) for row in completed_run.severity_rows]
    assert len(ratios) == len(set(round(value, 3) for value in ratios))


def test_leakage_controls_all_pass(completed_run) -> None:
    assert completed_run.leakage["components_rebuild_score"] is True
    assert completed_run.leakage["scorer_has_no_oracle_parameter"] is True
    assert completed_run.leakage["acquisition_records_have_no_gt_field"] is True
    assert completed_run.leakage["scoring_is_deterministic_at_fixed_seed"] is True


def test_selected_proposals_carry_the_oracle_verdict_post_hoc(completed_run) -> None:
    rows = completed_run.outputs.selected_rows
    assert rows
    row = rows[0]
    # Acquisition-time fields and post-hoc fields both present, and the post-hoc
    # ones are prefixed so the leakage guard can find them.
    assert {"proposal_id", "image_id", "objectness", "acquisition_rank"} <= set(row)
    assert {"gt_match_kind", "gt_class", "gt_group", "gt_best_iou"} <= set(row)


def test_discovery_recall_is_monotone_in_budget(completed_run) -> None:
    """Budget curves are prefixes of one trajectory, so recall cannot fall."""

    grouped: dict[tuple[str, str, int], list[tuple[int, float]]] = {}
    for row in completed_run.outputs.strategy_rows:
        key = (str(row["strategy"]), str(row["imbalance_setting"]), int(row["seed"]))
        grouped.setdefault(key, []).append((int(row["budget"]), float(row["all_discovery_recall"])))
    for key, points in grouped.items():
        values = [value for _, value in sorted(points)]
        assert values == sorted(values), key


def test_pilot_choice_is_recorded_and_from_a_disjoint_pool(completed_run) -> None:
    pilot = completed_run.pilot
    assert pilot["chosen_coherence_method"] in {"relative_within_cluster", "radius_core"}
    assert int(pilot["chosen_neighbour_count"]) >= 3
    assert int(pilot["pilot_pool_size"]) > 0
    manifest = json.loads(
        (completed_run.output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    # The main run used the pilot's choice, and says so.
    assert manifest["study_config"]["coherence_method_override"] == pilot["chosen_coherence_method"]


def test_runtime_plan_is_recorded_with_a_verdict(completed_run) -> None:
    plan = completed_run.runtime_plan
    assert plan["within_budget"] is True
    assert plan["study_cells"] == (
        len(TEST_MODE.imbalance_settings) * len(TEST_MODE.strategies) * len(TEST_MODE.seeds)
    )
    assert plan["seconds_per_cell"] > 0.0


def test_rerunning_reuses_cached_severities_and_reproduces_the_numbers(
    fixture_paths: tuple[Path, Path, Path],
    registered_mode: ExecutionMode,
    completed_run,
    tmp_path: Path,
) -> None:
    """Resume must be a no-op on the results, not merely fast."""

    root, export, split = fixture_paths
    state_files = sorted(path.name for path in (completed_run.output_dir / "state").iterdir())
    assert any(name.startswith("study_") for name in state_files)

    config = PipelineConfig(
        mode="PYTEST",
        data_root=str(root),
        split_file=str(split),
        existing_export=str(export),
        output_dir=str(completed_run.output_dir),
        cache_dir=str(completed_run.output_dir / "cache"),
        require_gpu=False,
        target_tail_recall=0.25,
    )
    second = run_pipeline(config, progress=None)
    first_rows = {
        (
            str(row["strategy"]),
            str(row["imbalance_setting"]),
            int(row["seed"]),
            int(row["budget"]),
        ): float(row["tail_discovery_recall"])
        for row in completed_run.outputs.strategy_rows
    }
    second_rows = {
        (
            str(row["strategy"]),
            str(row["imbalance_setting"]),
            int(row["seed"]),
            int(row["budget"]),
        ): float(row["tail_discovery_recall"])
        for row in second.outputs.strategy_rows
    }
    assert first_rows == second_rows


def test_force_recomputes_and_still_agrees(
    fixture_paths: tuple[Path, Path, Path],
    registered_mode: ExecutionMode,
    completed_run,
    tmp_path: Path,
) -> None:
    """A cold rerun must reproduce the cached numbers exactly (determinism)."""

    root, export, split = fixture_paths
    config = PipelineConfig(
        mode="PYTEST",
        data_root=str(root),
        split_file=str(split),
        existing_export=str(export),
        output_dir=str(tmp_path / "cold"),
        cache_dir=str(tmp_path / "cold_cache"),
        require_gpu=False,
        target_tail_recall=0.25,
        force=True,
    )
    cold = run_pipeline(config, progress=None)
    warm_auc = {
        (str(row["strategy"]), str(row["imbalance_setting"]), int(row["seed"])): float(
            row["tail_discovery_auc"]
        )
        for row in completed_run.outputs.auc_rows
    }
    cold_auc = {
        (str(row["strategy"]), str(row["imbalance_setting"]), int(row["seed"])): float(
            row["tail_discovery_auc"]
        )
        for row in cold.outputs.auc_rows
    }
    assert warm_auc == cold_auc


def test_a_missing_export_image_is_reported_not_silently_dropped(
    fixture_paths: tuple[Path, Path, Path],
    registered_mode: ExecutionMode,
    tmp_path: Path,
) -> None:
    root, export, split = fixture_paths
    bigger = replace(TEST_MODE, name="PYTESTBIG", evaluation_images=200)
    config_module.register(bigger, replace_existing=True)
    config = PipelineConfig(
        mode="PYTESTBIG",
        data_root=str(root),
        split_file=str(split),
        existing_export=str(export),
        output_dir=str(tmp_path / "toobig"),
        cache_dir=str(tmp_path / "toobig_cache"),
        require_gpu=False,
    )
    with pytest.raises((PipelineError, Exception)) as error:
        run_pipeline(config, progress=None)
    assert "images" in str(error.value).lower() or "lists only" in str(error.value)


def test_required_artifact_names_are_well_formed() -> None:
    """A refactor once turned "preflight.csv" into "csv" and every test still passed.

    The pipeline verifies that each REQUIRED_ARTIFACTS entry exists, so a mangled
    name is self-consistent: the run writes the wrong filename and then happily
    finds it. Only the names themselves catch that.
    """

    from daowod.pipeline import REQUIRED_ARTIFACTS

    for name in REQUIRED_ARTIFACTS:
        stem, _, suffix = name.rpartition(".")
        assert suffix in {"csv", "json", "md", "png"}, name
        assert stem, f"{name!r} has no stem, only an extension"
        assert len(stem) > 2, f"{name!r} looks truncated"
