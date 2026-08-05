"""The only module that communicates directly with PROB.

Configured commands must produce these standard files:

proposal NPZ:
    image_ids, confidence, embeddings,
    optionally posterior, predicted_labels, boxes, objectness

metrics JSON:
    known_mAP, U_Recall, WI, A_OSE,
    optionally detections_path
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ObjectArray = NDArray[np.object_]


@dataclass(frozen=True)
class ProposalBatch:
    image_ids: ObjectArray
    confidence: FloatArray
    embeddings: FloatArray
    posterior: FloatArray | None = None
    predicted_labels: IntArray | None = None
    boxes: FloatArray | None = None
    objectness: FloatArray | None = None

    def validate(self) -> ProposalBatch:
        count = self.image_ids.shape[0]

        if self.image_ids.ndim != 1:
            raise ValueError("image_ids must be one-dimensional.")

        if self.confidence.shape != (count,):
            raise ValueError("confidence must match proposal count.")

        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != count:
            raise ValueError("embeddings must be [proposal, feature].")

        if self.posterior is not None and self.posterior.shape[0] != count:
            raise ValueError("posterior must match proposal count.")

        if self.predicted_labels is not None and self.predicted_labels.shape != (count,):
            raise ValueError("predicted_labels must match proposal count.")

        if self.boxes is not None and self.boxes.shape != (count, 4):
            raise ValueError("boxes must have shape [proposal, 4].")

        if self.objectness is not None and self.objectness.shape != (count,):
            raise ValueError("objectness must match proposal count.")

        return self

    @classmethod
    def load(cls, path: str | Path) -> ProposalBatch:
        with np.load(path, allow_pickle=True) as data:
            missing = {"image_ids", "confidence", "embeddings"} - set(data.files)

            if missing:
                raise ValueError(f"Missing proposal NPZ fields: {sorted(missing)}")

            return cls(
                image_ids=np.asarray(data["image_ids"], dtype=object),
                confidence=np.asarray(
                    data["confidence"],
                    dtype=np.float64,
                ),
                embeddings=np.asarray(
                    data["embeddings"],
                    dtype=np.float64,
                ),
                posterior=(
                    np.asarray(data["posterior"], dtype=np.float64) if "posterior" in data else None
                ),
                predicted_labels=(
                    np.asarray(data["predicted_labels"], dtype=np.int64)
                    if "predicted_labels" in data
                    else None
                ),
                boxes=(np.asarray(data["boxes"], dtype=np.float64) if "boxes" in data else None),
                objectness=(
                    np.asarray(data["objectness"], dtype=np.float64)
                    if "objectness" in data
                    else None
                ),
            ).validate()


class ProbAdapter:
    """Run configured PROB commands and read their outputs."""

    def __init__(
        self,
        *,
        repository_path: str | Path,
        train_command: str,
        predict_command: str,
        evaluate_command: str,
        timeout_seconds: int,
    ) -> None:
        self.repository_path = Path(repository_path).resolve()
        self.train_command = train_command
        self.predict_command = predict_command
        self.evaluate_command = evaluate_command
        self.timeout_seconds = timeout_seconds

        if not self.repository_path.exists():
            raise FileNotFoundError(f"PROB repository not found: {self.repository_path}")

    @staticmethod
    def _write_ids(
        path: Path,
        image_ids: Sequence[str],
    ) -> None:
        path.write_text(
            "\n".join(str(image_id) for image_id in image_ids) + "\n",
            encoding="utf-8",
        )

    def _run(
        self,
        template: str,
        **values: object,
    ) -> None:
        command = self.render_command(template, **values)

        print("=" * 80)
        print("COMMAND:")
        print(command)
        print("=" * 80)

        result = subprocess.run(
            shlex.split(
                command,
                posix=os.name != "nt",
            ),
            cwd=self.repository_path,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
        )

        print("RETURN CODE:", result.returncode)
        print("=" * 80)
        print("STDOUT:")
        print(result.stdout)
        print("=" * 80)
        print("STDERR:")
        print(result.stderr)
        print("=" * 80)

        result.check_returncode()

    def render_command(self, template: str, **values: object) -> str:
        """Resolve a configured command template without executing it."""

        return template.format(
            repo=str(self.repository_path),
            **{key: str(value) for key, value in values.items()},
        )

    def resolved_train_command(
        self,
        *,
        labelled_ids: str | Path,
        previous_checkpoint: str | Path | None,
        checkpoint: str | Path,
        output_dir: str | Path,
        round_index: int,
        seed: int,
    ) -> str:
        return self.render_command(
            self.train_command,
            labelled_ids=labelled_ids,
            previous_checkpoint=previous_checkpoint or "",
            checkpoint=checkpoint,
            output_dir=output_dir,
            round=round_index,
            seed=seed,
        )

    def resolved_predict_command(
        self,
        *,
        image_ids: str | Path,
        checkpoint: str | Path,
        proposals: str | Path,
    ) -> str:
        return self.render_command(
            self.predict_command,
            image_ids=image_ids,
            checkpoint=checkpoint,
            proposals=proposals,
            output_dir=Path(proposals).parent,
        )

    def resolved_evaluate_command(self, *, checkpoint: str | Path, metrics: str | Path) -> str:
        return self.render_command(
            self.evaluate_command,
            checkpoint=checkpoint,
            metrics=metrics,
            output_dir=Path(metrics).parent,
        )

    def train(
        self,
        labelled_image_ids: Sequence[str],
        *,
        previous_checkpoint: str | Path | None,
        run_dir: str | Path,
        round_index: int,
        seed: int,
    ) -> Path:
        """Train or fine-tune PROB and return its checkpoint."""

        directory = Path(run_dir)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        labelled_path = directory / "labelled_ids.txt"
        checkpoint_path = directory / "checkpoint.pth"

        self._write_ids(
            labelled_path,
            labelled_image_ids,
        )

        self._run(
            self.resolved_train_command(
                labelled_ids=labelled_path,
                previous_checkpoint=previous_checkpoint,
                checkpoint=checkpoint_path,
                output_dir=directory,
                round_index=round_index,
                seed=seed,
            ),
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"PROB did not create checkpoint: {checkpoint_path}")

        return checkpoint_path

    def predict(
        self,
        image_ids: Sequence[str],
        *,
        checkpoint: str | Path,
        output_path: str | Path,
    ) -> ProposalBatch:
        """Export proposal features for acquisition."""

        output = Path(output_path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ids_path = output.with_suffix(".ids.txt")

        self._write_ids(
            ids_path,
            image_ids,
        )

        self._run(
            self.resolved_predict_command(
                image_ids=ids_path,
                checkpoint=checkpoint,
                proposals=output,
            ),
        )

        if not output.exists():
            raise FileNotFoundError(f"PROB did not create proposals: {output}")

        proposals = ProposalBatch.load(output)

        ids_path.unlink(missing_ok=True)

        return proposals

    def evaluate(
        self,
        *,
        checkpoint: str | Path,
        output_path: str | Path,
    ) -> dict[str, object]:
        """Run official PROB evaluation and read its JSON summary."""

        output = Path(output_path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._run(
            self.resolved_evaluate_command(checkpoint=checkpoint, metrics=output),
        )

        if not output.exists():
            raise FileNotFoundError(f"PROB did not create metrics: {output}")

        metrics = json.loads(output.read_text(encoding="utf-8"))

        required = {
            "known_mAP",
            "U_Recall",
            "WI",
            "A_OSE",
        }

        missing = required - set(metrics)

        if missing:
            raise ValueError(f"Missing PROB metrics: {sorted(missing)}")

        return metrics


# =============================================================================
# Cached, resumable export
#
# Cached, resumable PROB proposal export.
#
# Detector inference is the only GPU cost in this experiment and the only step that
# cannot be repeated cheaply, so it is done once, in chunks, into a content-keyed
# cache. A disconnected Colab session resumes at the first missing chunk; a rerun
# with identical settings does no GPU work at all.
#
# Cache identity
# --------------
# A chunk is reused only when the *fingerprint* matches: bridge settings (data root,
# dataset, task class counts, objectness temperature, proposals per image, seed,
# device), the checkpoint's digest, and the exact image IDs in that chunk. Anything
# that could change a single exported number is inside the fingerprint, so a stale
# cache cannot silently contaminate a run — the failure mode of every "just cache
# it" implementation.
#
# Why chunks rather than one call
# -------------------------------
# Three reasons, all measured on real runs: a 4 000-image export takes long enough
# that a session drop mid-way is likely; the bridge's per-call overhead (model
# build, checkpoint load) is a few seconds and amortises fine over 250 images; and
# chunk timings give the runtime planner a real images-per-second rate after the
# first chunk instead of a guess.
#
# Merged with the adapter because there is one detector: the subprocess boundary
# and the cache that fronts it are two halves of the same contract, and the cache
# was the adapter's only consumer.
# =============================================================================

EXPORT_FIELDS: tuple[str, ...] = (
    "image_ids",
    "confidence",
    "embeddings",
    "posterior",
    "predicted_labels",
    "boxes",
    "objectness",
)


class ExportError(RuntimeError):
    """Raised when an export is impossible, incomplete or inconsistent."""


@dataclass(frozen=True)
class BridgeSettings:
    """Every flag the bridge's ``predict`` subcommand needs.

    Defaults are the S-OWODB / OWDETR Task-1 protocol: 19 known classes
    introduced, 81 total (80 COCO + unknown), objectness temperature 1, and the
    full 100-query export so the candidate filter — not the bridge — decides what
    a candidate is.
    """

    prob_repository: str
    checkpoint: str
    data_root: str
    dataset: str = "OWDETR"
    previous_introduced_classes: int = 0
    current_introduced_classes: int = 19
    num_classes: int = 81
    objectness_temperature: float = 1.0
    batch_size: int = 2
    num_workers: int = 2
    device: str = "cuda"
    seed: int = 0
    max_proposals_per_image: int = 100
    minimum_unknown_score: float = 0.0
    python_executable: str = "python"
    timeout_seconds: int = 7_200

    def __post_init__(self) -> None:
        if self.max_proposals_per_image < 1:
            raise ExportError("max_proposals_per_image must be positive.")
        if not 0.0 <= self.minimum_unknown_score <= 1.0:
            raise ExportError("minimum_unknown_score must lie in [0, 1].")
        if self.num_classes < 2:
            raise ExportError("num_classes must be at least 2.")

    def protocol_arguments(self) -> list[str]:
        return [
            "--data-root",
            str(self.data_root),
            "--dataset",
            str(self.dataset),
            "--prev-introduced-classes",
            str(self.previous_introduced_classes),
            "--current-introduced-classes",
            str(self.current_introduced_classes),
            "--num-classes",
            str(self.num_classes),
            "--objectness-temperature",
            str(self.objectness_temperature),
            "--batch-size",
            str(self.batch_size),
            "--num-workers",
            str(self.num_workers),
            "--device",
            str(self.device),
            "--seed",
            str(self.seed),
        ]

    def predict_command(self) -> str:
        """The command template :class:`~daowod.detector.ProbAdapter` renders.

        Written as a template rather than a list because the adapter is the one
        boundary to PROB in this repository and it takes templates; keeping that
        contract means the bridge is still called exactly one way.
        """

        return (
            f"{self.python_executable} daowod_prob_bridge.py predict "
            "--image-ids {image_ids} --checkpoint {checkpoint} --output {proposals} "
            f"--max-proposals-per-image {self.max_proposals_per_image} "
            f"--minimum-unknown-score {self.minimum_unknown_score} "
            + " ".join(self.protocol_arguments())
        )

    def as_dict(self) -> dict[str, object]:
        return dict(asdict(self))

    def fingerprint(self, *, checkpoint_digest: str) -> str:
        """Digest of every setting that can change an exported number."""

        payload = {
            key: value
            for key, value in self.as_dict().items()
            # The interpreter path and the timeout cannot change a number, and
            # they differ between a local check and Colab; excluding them is what
            # makes a cache portable between the two.
            if key not in {"python_executable", "timeout_seconds", "num_workers", "prob_repository"}
        }
        payload["checkpoint_digest"] = checkpoint_digest
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


def checkpoint_digest(path: str | Path, *, sample_bytes: int = 64 * 1024 * 1024) -> str:
    """Digest of a checkpoint, over a bounded prefix plus its size.

    A full SHA-256 of a 500 MB checkpoint on a Colab Drive mount costs minutes on
    every run. Hashing the first ``sample_bytes`` together with the exact file
    size distinguishes any two checkpoints this pipeline could plausibly confuse
    (different training runs differ in the first megabytes of their state dict)
    while staying under a second. The full digest is still available through
    :func:`daowod.dataset.file_sha256` where an exact match is required, such as
    the notebook's asset validation.
    """

    target = Path(path)
    if not target.exists():
        raise ExportError(f"Missing checkpoint: {target}")
    digest = hashlib.sha256()
    digest.update(str(target.stat().st_size).encode("utf-8"))
    with target.open("rb") as handle:
        remaining = int(sample_bytes)
        while remaining > 0:
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def chunk_image_ids(image_ids: Sequence[str], *, chunk_images: int) -> list[list[str]]:
    """Deterministic, order-independent chunking.

    IDs are de-duplicated and sorted first, so the same request produces the same
    chunks — and therefore hits the same cache — no matter how the caller ordered
    its split file.
    """

    if chunk_images < 1:
        raise ExportError("chunk_images must be positive.")
    unique = sorted(dict.fromkeys(str(value) for value in image_ids))
    if not unique:
        raise ExportError("No image IDs were requested.")
    return [unique[start : start + chunk_images] for start in range(0, len(unique), chunk_images)]


@dataclass(frozen=True)
class ChunkRecord:
    """One exported chunk on disk."""

    index: int
    path: Path
    image_count: int
    proposal_count: int
    seconds: float
    reused: bool
    digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "path": str(self.path),
            "image_count": self.image_count,
            "proposal_count": self.proposal_count,
            "seconds": round(self.seconds, 2),
            "reused": self.reused,
            "digest": self.digest,
        }


@dataclass
class ExportResult:
    """Every chunk that makes up one logical export."""

    cache_dir: Path
    fingerprint: str
    chunks: list[ChunkRecord] = field(default_factory=list)

    @property
    def image_count(self) -> int:
        return sum(chunk.image_count for chunk in self.chunks)

    @property
    def proposal_count(self) -> int:
        return sum(chunk.proposal_count for chunk in self.chunks)

    @property
    def exported_seconds(self) -> float:
        """Wall clock actually spent on the GPU, excluding reused chunks."""

        return sum(chunk.seconds for chunk in self.chunks if not chunk.reused)

    @property
    def exported_images(self) -> int:
        return sum(chunk.image_count for chunk in self.chunks if not chunk.reused)

    def seconds_per_image(self) -> float | None:
        """Measured inference rate, or ``None`` when everything was cached."""

        if self.exported_images < 1:
            return None
        return self.exported_seconds / self.exported_images

    def paths(self) -> list[Path]:
        return [chunk.path for chunk in self.chunks]

    def as_dict(self) -> dict[str, object]:
        return {
            "cache_dir": str(self.cache_dir),
            "fingerprint": self.fingerprint,
            "images": self.image_count,
            "proposals": self.proposal_count,
            "chunks": [chunk.as_dict() for chunk in self.chunks],
            "exported_images": self.exported_images,
            "exported_seconds": round(self.exported_seconds, 1),
            "seconds_per_image": self.seconds_per_image(),
        }


def _chunk_digest(fingerprint: str, image_ids: Sequence[str]) -> str:
    payload = fingerprint + "|" + "\n".join(image_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _validate_chunk(path: Path, *, expected_images: int) -> tuple[int, int]:
    """Row and image counts of a cached chunk, or raise if it is unusable."""

    with np.load(path, allow_pickle=True) as handle:
        missing = [name for name in EXPORT_FIELDS if name not in handle.files]
        if missing:
            raise ExportError(f"{path}: chunk is missing {missing}.")
        ids = np.asarray(handle["image_ids"], dtype=object)
        rows = int(ids.shape[0])
        images = int(np.unique(ids.astype(str)).size)
    if rows < 1:
        raise ExportError(f"{path}: chunk contains no proposals.")
    if images > expected_images:
        raise ExportError(
            f"{path}: chunk holds {images} images but only {expected_images} were requested."
        )
    return rows, images


def export_proposals(
    *,
    settings: BridgeSettings,
    image_ids: Sequence[str],
    cache_dir: str | Path,
    chunk_images: int = 250,
    progress: Callable[[str], None] | None = None,
    stop_after_chunks: int | None = None,
) -> ExportResult:
    """Export proposals for ``image_ids``, reusing whatever the cache already has.

    ``stop_after_chunks`` exports only the first N chunks and returns, which is how
    the pilot measures the inference rate before committing to the full export.
    """

    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    digest = checkpoint_digest(settings.checkpoint)
    fingerprint = settings.fingerprint(checkpoint_digest=digest)
    blocks = chunk_image_ids(image_ids, chunk_images=chunk_images)
    result = ExportResult(cache_dir=directory, fingerprint=fingerprint)

    adapter = ProbAdapter(
        repository_path=settings.prob_repository,
        train_command="{checkpoint}",  # unused here; the adapter requires a value
        predict_command=settings.predict_command(),
        evaluate_command="{metrics}",  # unused here
        timeout_seconds=settings.timeout_seconds,
    )

    for index, block in enumerate(blocks):
        if stop_after_chunks is not None and index >= stop_after_chunks:
            break
        chunk_key = _chunk_digest(fingerprint, block)
        path = directory / f"chunk_{index:04d}_{chunk_key}.npz"
        if path.exists():
            rows, images = _validate_chunk(path, expected_images=len(block))
            if progress is not None:
                progress(
                    f"chunk {index + 1}/{len(blocks)}: reused {images} image(s), "
                    f"{rows} proposal(s) from cache"
                )
            result.chunks.append(
                ChunkRecord(
                    index=index,
                    path=path,
                    image_count=images,
                    proposal_count=rows,
                    seconds=0.0,
                    reused=True,
                    digest=chunk_key,
                )
            )
            continue

        if progress is not None:
            progress(f"chunk {index + 1}/{len(blocks)}: exporting {len(block)} image(s) on GPU")
        started = time.perf_counter()
        adapter.predict(block, checkpoint=settings.checkpoint, output_path=path)
        seconds = time.perf_counter() - started
        rows, images = _validate_chunk(path, expected_images=len(block))
        result.chunks.append(
            ChunkRecord(
                index=index,
                path=path,
                image_count=images,
                proposal_count=rows,
                seconds=seconds,
                reused=False,
                digest=chunk_key,
            )
        )
        if progress is not None:
            progress(
                f"chunk {index + 1}/{len(blocks)}: {images} image(s), {rows} proposal(s) "
                f"in {seconds:.1f}s ({seconds / max(images, 1):.2f}s/image)"
            )

    write_manifest(result, settings=settings, checkpoint_digest=digest)
    return result


def write_manifest(
    result: ExportResult, *, settings: BridgeSettings, checkpoint_digest: str
) -> Path:
    """Record what the cache holds, so a later run can audit it without loading it."""

    path = result.cache_dir / "export_manifest.json"
    payload = {
        "settings": settings.as_dict(),
        "checkpoint_digest": checkpoint_digest,
        **result.as_dict(),
    }
    if payload.get("seconds_per_image") is None and path.exists():
        # A fully cached rerun has no fresh timing. The rate measured when these
        # chunks were produced is still the right number for the runtime
        # projection, so it is carried forward rather than lost — otherwise a
        # resumed session would have to re-run inference just to time it.
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        if previous.get("checkpoint_digest") == checkpoint_digest:
            payload["seconds_per_image"] = previous.get("seconds_per_image")
            payload["seconds_per_image_source"] = "carried forward from a previous run"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def load_chunks(paths: Sequence[str | Path]) -> dict[str, NDArray[np.generic]]:
    """Concatenate exported chunks into one in-memory export.

    Chunks are concatenated in the order given, which is chunk order, which is
    sorted image-ID order — so the merged export is itself deterministic and every
    downstream index is stable across runs.
    """

    if not paths:
        raise ExportError("No export chunk was given.")
    collected: dict[str, list[NDArray[np.generic]]] = {name: [] for name in EXPORT_FIELDS}
    for path in paths:
        with np.load(path, allow_pickle=True) as handle:
            missing = [name for name in EXPORT_FIELDS if name not in handle.files]
            if missing:
                raise ExportError(f"{path}: chunk is missing {missing}.")
            for name in EXPORT_FIELDS:
                collected[name].append(handle[name])
    merged: dict[str, NDArray[np.generic]] = {}
    for name, blocks in collected.items():
        merged[name] = (
            np.concatenate([np.asarray(block, dtype=object) for block in blocks])
            if name == "image_ids"
            else np.concatenate(blocks)
        )
    counts = {name: int(array.shape[0]) for name, array in merged.items()}
    if len(set(counts.values())) != 1:
        raise ExportError(f"Merged export columns disagree in length: {counts}")
    return merged


def load_export_file(path: str | Path) -> dict[str, NDArray[np.generic]]:
    """Load a single pre-existing export NPZ, validating the study's schema."""

    return load_chunks([path])


def stage_dataset(
    *,
    source: str | Path,
    destination: str | Path,
    image_ids: Sequence[str],
    dataset: str = "OWDETR",
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Copy the images and annotations one run needs onto local disk.

    Why this exists: on Colab the dataset lives on a Drive FUSE mount, where the
    cost of reading a file is dominated by per-file latency rather than bandwidth.
    The detector opens one JPEG per forward pass, so inference over a few thousand
    images pays that latency a few thousand times, and it shows up as detector
    seconds-per-image — the term the runtime budget is most sensitive to. Copying
    the needed files once turns thousands of high-latency reads into one bulk
    transfer.

    Only the requested IDs are copied, existing files are skipped (so an
    interrupted stage resumes), and the ``ImageSets`` tree is copied whole because
    the bridge writes its temporary split there.
    """

    root = Path(source)
    target = Path(destination)
    wanted = sorted(dict.fromkeys(str(value) for value in image_ids))
    if not wanted:
        raise ExportError("stage_dataset was given no image IDs.")
    copied = 0
    skipped = 0
    missing: list[str] = []
    for directory, suffix in (("Annotations", ".xml"), ("JPEGImages", ".jpg")):
        (target / directory).mkdir(parents=True, exist_ok=True)
        for image_id in wanted:
            origin = root / directory / f"{image_id}{suffix}"
            landing = target / directory / f"{image_id}{suffix}"
            if landing.exists() and landing.stat().st_size > 0:
                skipped += 1
                continue
            if not origin.exists():
                missing.append(str(origin))
                continue
            shutil.copyfile(origin, landing)
            copied += 1
        if progress is not None:
            progress(f"staged {directory}: {copied} copied, {skipped} already present")
    if missing:
        raise ExportError(
            f"{len(missing)} dataset file(s) are missing at the source, e.g. {missing[:3]}."
        )
    splits_source = root / "ImageSets" / dataset
    splits_target = target / "ImageSets" / dataset
    splits_target.mkdir(parents=True, exist_ok=True)
    for split in sorted(splits_source.glob("*.txt")):
        landing = splits_target / split.name
        if not landing.exists():
            shutil.copyfile(split, landing)
    return {
        "source": str(root),
        "destination": str(target),
        "images": len(wanted),
        "files_copied": copied,
        "files_already_present": skipped,
        "splits": sorted(path.name for path in splits_target.glob("*.txt")),
    }


def split_disjoint(
    image_ids: Sequence[str],
    *,
    counts: Mapping[str, int],
    seed: int = 0,
) -> dict[str, list[str]]:
    """Partition images into named, provably disjoint pools.

    The pilot pool must not overlap the evaluation pool — that separation is what
    makes the hyperparameter choice honest — and the reference bank must not
    overlap either, or novelty would be measured against the very proposals it is
    scoring. Returning a dict of disjoint lists, built once from one shuffle, makes
    the property structural rather than a convention the caller has to remember.
    """

    unique = sorted(dict.fromkeys(str(value) for value in image_ids))
    required = sum(int(value) for value in counts.values())
    if required > len(unique):
        raise ExportError(
            f"Requested {required} images across {list(counts)} but the split "
            f"lists only {len(unique)}."
        )
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(unique))
    pools: dict[str, list[str]] = {}
    cursor = 0
    for name, size in counts.items():
        take = int(size)
        chosen = sorted(unique[int(position)] for position in order[cursor : cursor + take])
        pools[name] = chosen
        cursor += take
    overlaps = {
        f"{left}&{right}": sorted(set(pools[left]) & set(pools[right]))[:5]
        for index, left in enumerate(pools)
        for right in list(pools)[index + 1 :]
        if set(pools[left]) & set(pools[right])
    }
    if overlaps:  # pragma: no cover - structurally impossible, kept as a tripwire
        raise ExportError(f"Pools overlap: {overlaps}")
    return pools
