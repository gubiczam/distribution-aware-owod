"""Everything that must be true before a GPU hour is spent.

Each check returns a row rather than printing, so the notebook can render one
table and the pipeline can refuse to start on a single ``FAIL``. Three statuses
are distinguished on purpose:

``PASS``   the requirement is satisfied and was verified, not assumed.
``FAIL``   the run cannot produce valid results; the pipeline stops.
``SKIP``   the requirement does not apply to this run (no GPU needed in DEBUG).
``WARN``   the run can proceed but a reported number will be weaker for it.

The distinction matters because the failure mode this module exists to prevent is
a three-hour session that ends in a missing-file error, or worse, one that
finishes and reports numbers derived from the wrong split.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from daowod import oracle

#: Marker the bridge's proposal export depends on. PROB's forward pass must
#: publish the final decoder hidden states as ``pred_features``; without them the
#: export has no embeddings and novelty, rarity and coherence are all undefined.
PROB_FEATURE_MARKER = "pred_features"

#: Files the bridge contract requires inside the PROB checkout.
PROB_REQUIRED_FILES: tuple[tuple[str, str], ...] = (
    ("daowod_prob_bridge.py", "def predict("),
    ("main_open_world.py", "def get_args_parser("),
    ("models/prob_deformable_detr.py", PROB_FEATURE_MARKER),
)

#: Packages the offline study needs, with the import name where it differs.
REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("PyYAML", "yaml"),
)


class PreflightError(RuntimeError):
    """Raised when a required precondition is not met."""


@dataclass(frozen=True)
class Check:
    """One verified precondition."""

    name: str
    status: str
    detail: str
    value: object = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "check": self.name,
            "status": self.status,
            "detail": self.detail,
            "value": self.value,
        }


def _run(command: Sequence[str], *, timeout: int = 60) -> tuple[int, str]:
    try:
        result = subprocess.run(
            [str(part) for part in command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:  # pragma: no cover - env specific
        return 1, str(error)
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def check_python() -> Check:
    """Python must be 3.11 or 3.12: the package pins that range."""

    major, minor = sys.version_info[:2]
    supported = (major, minor) in ((3, 11), (3, 12))
    return Check(
        name="python_version",
        status="PASS" if supported else "FAIL",
        detail=f"{platform.python_version()} on {platform.platform()}",
        value=f"{major}.{minor}",
    )


def check_packages(packages: Sequence[tuple[str, str]] = REQUIRED_PACKAGES) -> list[Check]:
    """Every offline dependency must import and report a version."""

    checks: list[Check] = []
    for distribution, module_name in packages:
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            checks.append(
                Check(
                    name=f"package:{distribution}",
                    status="FAIL",
                    detail=f"import {module_name} failed: {error}",
                )
            )
            continue
        checks.append(
            Check(
                name=f"package:{distribution}",
                status="PASS",
                detail=f"import {module_name} ok",
                value=getattr(module, "__version__", "unknown"),
            )
        )
    return checks


def check_gpu(*, required: bool, require_t4_or_better: bool = True) -> list[Check]:
    """CUDA, a visible device, and enough memory for PROB inference.

    ``required=False`` turns every GPU check into ``SKIP`` rather than removing
    it, so a CPU-only DEBUG run still reports what it did not verify.
    """

    if not required:
        return [
            Check(
                name="gpu",
                status="SKIP",
                detail="This mode runs the offline study only; no detector inference.",
            )
        ]
    checks: list[Check] = []
    code, output = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    visible = code == 0 and output.strip() != ""
    checks.append(
        Check(
            name="nvidia_smi",
            status="PASS" if visible else "FAIL",
            detail=output.strip()[:300] if visible else "nvidia-smi unavailable or reported no GPU",
            value=output.strip().splitlines()[0] if visible else "",
        )
    )
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        checks.append(
            Check(name="torch_cuda", status="FAIL", detail=f"import torch failed: {error}")
        )
        return checks
    available = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if available else ""
    total_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if available else 0.0
    )
    checks.append(
        Check(
            name="torch_cuda",
            status="PASS" if available else "FAIL",
            detail=f"torch {torch.__version__}, cuda available={available}, device={name!r}",
            value=name,
        )
    )
    if available:
        # A T4 has 16 GB and is the reference device. Anything with at least 12 GB
        # runs the same configuration; less than that and PROB inference at the
        # protocol's resolution will not fit, which is a FAIL rather than a WARN
        # because the alternative is an out-of-memory crash mid-session.
        enough = total_gb >= 12.0
        recognised = "T4" in name.upper()
        status = "PASS" if enough else "FAIL"
        if enough and require_t4_or_better and not recognised:
            status = "WARN"
        checks.append(
            Check(
                name="gpu_memory",
                status=status,
                detail=(
                    f"{total_gb} GB on {name!r}. The runtime budget is calibrated for "
                    "a 16 GB T4; a different device changes the projection, not the "
                    "validity of the results."
                ),
                value=total_gb,
            )
        )
    return checks


def check_repository(path: str | Path, *, name: str) -> list[Check]:
    """A checkout must exist, be a git repository, and report its commit."""

    root = Path(path)
    if not root.exists():
        return [Check(name=f"{name}_checkout", status="FAIL", detail=f"missing: {root}")]
    code, head = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    commit = head.strip() if code == 0 else "unavailable"
    code, status = _run(["git", "-C", str(root), "status", "--short"])
    dirty = status.strip()
    return [
        Check(
            name=f"{name}_checkout",
            status="PASS",
            detail=f"{root} at commit {commit[:12]}",
            value=commit,
        ),
        Check(
            name=f"{name}_worktree",
            status="PASS" if not dirty else "WARN",
            detail=(
                "clean"
                if not dirty
                else f"{len(dirty.splitlines())} modified path(s); results are not "
                "reproducible from the commit alone"
            ),
            value=len(dirty.splitlines()),
        ),
    ]


def check_prob_checkout(path: str | Path) -> list[Check]:
    """The PROB checkout must expose the bridge and the decoder feature export."""

    root = Path(path)
    checks = check_repository(root, name="prob")
    if not root.exists():
        return checks
    for relative, marker in PROB_REQUIRED_FILES:
        target = root / relative
        if not target.exists():
            checks.append(
                Check(
                    name=f"prob_file:{relative}",
                    status="FAIL",
                    detail=f"missing: {target}",
                )
            )
            continue
        text = target.read_text(encoding="utf-8", errors="replace")
        present = marker in text
        checks.append(
            Check(
                name=f"prob_file:{relative}",
                status="PASS" if present else "FAIL",
                detail=(
                    f"contains {marker!r}"
                    if present
                    else (
                        f"{target} exists but does not contain {marker!r}. The "
                        "proposal export depends on this symbol; a renamed "
                        "function or a dropped model output would otherwise fail "
                        "hours into the run."
                    )
                ),
            )
        )
    return checks


def check_bridge_cli(path: str | Path, *, python_executable: str | None = None) -> Check:
    """``daowod_prob_bridge.py check`` must exit zero inside the PROB checkout."""

    root = Path(path)
    bridge = root / "daowod_prob_bridge.py"
    if not bridge.exists():
        return Check(name="prob_bridge_check", status="FAIL", detail=f"missing: {bridge}")
    interpreter = python_executable or sys.executable
    code, output = _run([interpreter, str(bridge), "check"], timeout=600)
    return Check(
        name="prob_bridge_check",
        status="PASS" if code == 0 else "FAIL",
        detail=(output.strip()[-400:] or "no output"),
        value=code,
    )


def check_dataset(
    root: str | Path,
    *,
    dataset: str = "OWDETR",
    split_file: str | Path | None = None,
    sample: int = 5,
) -> list[Check]:
    """The VOC-style tree, the split file, and a sample of real image/annotation pairs."""

    base = Path(root)
    checks: list[Check] = []
    for directory in ("Annotations", "JPEGImages", "ImageSets"):
        target = base / directory
        checks.append(
            Check(
                name=f"dataset_dir:{directory}",
                status="PASS" if target.is_dir() else "FAIL",
                detail=str(target),
                value=len(list(target.iterdir())) if target.is_dir() else 0,
            )
        )
    split_directory = base / "ImageSets" / dataset
    checks.append(
        Check(
            name=f"dataset_splits:{dataset}",
            status="PASS" if split_directory.is_dir() else "FAIL",
            detail=str(split_directory),
            value=(
                sorted(item.name for item in split_directory.glob("*.txt"))
                if split_directory.is_dir()
                else []
            ),
        )
    )
    if split_file is None:
        return checks

    path = Path(split_file)
    if not path.exists():
        checks.append(Check(name="split_file", status="FAIL", detail=f"missing: {path}"))
        return checks
    ids = read_image_ids(path)
    checks.append(
        Check(
            name="split_file",
            status="PASS" if ids else "FAIL",
            detail=f"{path} lists {len(ids)} image id(s)",
            value=len(ids),
        )
    )
    missing_images = [
        image_id
        for image_id in ids[: max(sample, 1)]
        if not (base / "JPEGImages" / f"{image_id}.jpg").exists()
    ]
    missing_annotations = [
        image_id
        for image_id in ids[: max(sample, 1)]
        if not (base / "Annotations" / f"{image_id}.xml").exists()
    ]
    checks.append(
        Check(
            name="split_assets",
            status="PASS" if not (missing_images or missing_annotations) else "FAIL",
            detail=(
                f"sampled {min(len(ids), max(sample, 1))} id(s); missing images "
                f"{missing_images[:3]}, missing annotations {missing_annotations[:3]}"
            ),
        )
    )
    if ids and not missing_annotations:
        try:
            parsed = oracle.read_voc_annotation(ids[0], base / "Annotations")
            unknown = [item.class_name for item in parsed.objects if not item.is_known]
            checks.append(
                Check(
                    name="annotation_parse",
                    status="PASS",
                    detail=(
                        f"{ids[0]}: {parsed.width}x{parsed.height}, "
                        f"{len(parsed.objects)} object(s), "
                        f"{len(unknown)} unknown at Task 1"
                    ),
                    value=len(parsed.objects),
                )
            )
        except oracle.OracleError as error:
            checks.append(Check(name="annotation_parse", status="FAIL", detail=str(error)))
    return checks


def check_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    load: bool = False,
) -> list[Check]:
    """The detector checkpoint must exist, be non-trivial, and optionally match a digest."""

    target = Path(path)
    if not target.exists():
        return [Check(name="checkpoint", status="FAIL", detail=f"missing: {target}")]
    size_mb = round(target.stat().st_size / 1024**2, 1)
    checks = [
        Check(
            name="checkpoint",
            status="PASS" if size_mb > 1.0 else "FAIL",
            detail=f"{target} ({size_mb} MB)",
            value=size_mb,
        )
    ]
    if expected_sha256:
        from daowod.dataset import file_sha256

        digest = file_sha256(target)
        checks.append(
            Check(
                name="checkpoint_sha256",
                status="PASS" if digest == expected_sha256 else "FAIL",
                detail=f"{digest} (expected {expected_sha256})",
                value=digest,
            )
        )
    if load:
        try:
            torch = importlib.import_module("torch")
            state = torch.load(str(target), map_location="cpu")
            keys = sorted(state)[:6] if isinstance(state, dict) else []
            checks.append(
                Check(
                    name="checkpoint_load",
                    status="PASS",
                    detail=f"loaded; top-level keys {keys}",
                )
            )
        except Exception as error:  # pragma: no cover - depends on torch build
            checks.append(
                Check(name="checkpoint_load", status="FAIL", detail=f"torch.load failed: {error}")
            )
    return checks


def check_disk(path: str | Path, *, required_gb: float) -> Check:
    """Free space for the proposal export, which is the largest artifact."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    free_gb = round(usage.free / 1024**3, 1)
    return Check(
        name="disk_space",
        status="PASS" if free_gb >= required_gb else "FAIL",
        detail=f"{free_gb} GB free at {target}, need about {required_gb} GB",
        value=free_gb,
    )


def estimate_export_gigabytes(*, images: int, proposals_per_image: int, dimensions: int) -> float:
    """Size of one export: embeddings dominate, at float32 on disk via ``np.savez``.

    Used by the disk check so "not enough space" is reported before the export
    rather than at the last chunk.
    """

    rows = max(int(images), 0) * max(int(proposals_per_image), 0)
    per_row_bytes = 4 * (int(dimensions) + 4 + 20 + 4)  # embeddings, box, posterior, scalars
    return round(rows * per_row_bytes / 1024**3, 2)


def read_image_ids(path: str | Path) -> list[str]:
    """Image IDs from a VOC ``ImageSets`` file, first whitespace field per line."""

    text = Path(path).read_text(encoding="utf-8")
    return [line.split()[0] for line in text.splitlines() if line.strip()]


def summarise(checks: Iterable[Check]) -> dict[str, object]:
    """Counts per status plus the failing names, for the notebook's header line."""

    rows = list(checks)
    counts: dict[str, int] = {}
    for check in rows:
        counts[check.status] = counts.get(check.status, 0) + 1
    return {
        "checks": len(rows),
        "counts": counts,
        "failed": [check.name for check in rows if check.status == "FAIL"],
        "warned": [check.name for check in rows if check.status == "WARN"],
    }


def require_all_pass(checks: Sequence[Check]) -> None:
    """Raise a single error naming every failed check and its detail."""

    failures = [check for check in checks if check.status == "FAIL"]
    if not failures:
        return
    lines = "\n".join(f"  - {check.name}: {check.detail}" for check in failures)
    raise PreflightError(f"{len(failures)} precondition(s) failed:\n{lines}")


def rows(checks: Sequence[Check]) -> list[Mapping[str, object]]:
    """Check rows for ``preflight.csv``."""

    return [check.as_dict() for check in checks]


def environment_report() -> dict[str, object]:
    """Free-form context recorded with every run for reproducibility."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
    }
