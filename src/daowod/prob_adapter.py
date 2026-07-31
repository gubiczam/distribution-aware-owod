"""The only module that communicates directly with PROB.

Configured commands must produce these standard files:

proposal NPZ:
    image_ids, confidence, embeddings,
    optionally posterior, predicted_labels, boxes, objectness

metrics JSON:
    known_mAP, U_Recall, WI, A_OSE,
    optionally detections_path
"""

import json
import os
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
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

    def validate(self) -> "ProposalBatch":
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
    def load(cls, path: str | Path) -> "ProposalBatch":
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
        command = template.format(
            repo=str(self.repository_path),
            **{key: str(value) for key, value in values.items()},
        )

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
            self.train_command,
            labelled_ids=labelled_path,
            previous_checkpoint=previous_checkpoint or "",
            checkpoint=checkpoint_path,
            output_dir=directory,
            round=round_index,
            seed=seed,
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
            self.predict_command,
            image_ids=ids_path,
            checkpoint=checkpoint,
            proposals=output,
            output_dir=output.parent,
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
            self.evaluate_command,
            checkpoint=checkpoint,
            metrics=output,
            output_dir=output.parent,
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
