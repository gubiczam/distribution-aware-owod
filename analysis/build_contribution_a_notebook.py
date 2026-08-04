"""Generate ``notebooks/contribution_a_active_annotation.ipynb``.

The notebook is generated rather than hand-edited so that its JSON stays valid,
its cells stay small, and a change to the driver is reviewable as a diff of Python
rather than of embedded JSON strings. Run from the repository root:

    python analysis/build_contribution_a_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md(
    """
# Contribution A — Distribution-Aware Active Annotation for Open-World Detection

Runs the full offline active-annotation experiment on **real PROB proposals**, on one
NVIDIA T4, in roughly four to five hours.

**What this measures.** For a fixed *region-level* annotation budget, which candidate
regions does each acquisition strategy buy? The acquisition score is

```text
s(x) = alpha * uncertainty(x) + beta * novelty(x) + gamma * rarity(x) * coherence(x)**p
```

where coherence enters **only as a multiplicative gate on rarity**: an isolated
proposal keeps its uncertainty and novelty but loses the rarity bonus. Five
strategies are compared — random, uncertainty, uncertainty+novelty, +estimated
rarity (ungated), and the gated coherence-aware selection — across three
controlled long-tail severities and several seeds.

**What this does not measure.** No detector retraining happens here, so no
known-mAP / U-Recall / WI / A-OSE number is claimed. Those come from the official
PROB evaluator; this notebook measures the quality of the annotation set itself.

**Ground-truth discipline.** Annotations are used in exactly two places: to build
the long-tail evaluation pool (a protocol step) and to answer the oracle *after* a
region has been selected. The pipeline proves this rather than asserting it — it
re-derives every acquisition score from its recorded components and stops if an
unexplained term is present.

**Pipeline.**

```text
preflight -> disjoint reference/pilot/evaluation splits -> cached PROB inference
-> candidate proposals -> region-level oracle -> controlled long-tail pools
-> leakage proof -> pilot hyperparameter choice -> runtime projection
-> acquisition strategies -> metrics -> plots -> CSVs -> summary -> ZIP
```

**Outputs** land in `OUTPUT_DIR`: per-strategy CSVs, publication figures (PNG+PDF),
`research_summary.md`, a run manifest, and one ZIP with everything.

**Execution modes.** `DEBUG` (a few minutes; exercises every stage), `FAST` (about
an hour; under-powered but real), `MAIN` (the five-strategy experiment, T4, 4-5 h),
and `MAINREVEALED` (the eleven-arm follow-up: the same protocol with the free
informativeness-prior control and the label-anchored distribution term alongside the
baseline). All of them run the real detector over their own image budget — `DEBUG`
over about 410 images — unless `EXISTING_EXPORT` points at a proposal NPZ, which
skips every GPU step and makes the whole notebook CPU-only.

**Read `docs/contribution_a_failure_analysis.md` before interpreting `MAIN`.** The
first full run found the coherence gate did *not* improve tail discovery, and the
component audit localises why: in PROB's Task-1 decoder space a tail proposal's ten
nearest neighbours contain 1.5 % of its own class while a background proposal's
contain 88.8 % background, so density-based coherence ranks background highest.
`MAINREVEALED` is the follow-up experiment that test-drives the two fixes the audit
motivates.

**Resumable.** Every stage caches. If the session drops, re-run the notebook top to
bottom: finished detector chunks and finished severities are reused.
"""
)

md(
    """
## 1. Configuration

Edit **only** the next cell. Leave `RUN_MODE = "DEBUG"` for the first pass — it
exercises every stage on about 410 images in a few minutes. Switch to `FAST`, then
`MAIN`, once it passes.
"""
)

code(
    """
# ============================ USER CONFIGURATION ============================
RUN_MODE = "DEBUG"                  # "DEBUG" | "FAST" | "MAIN" | "MAINREVEALED"

# Repositories. The PROB fork must export decoder features as `pred_features`
# and ship `daowod_prob_bridge.py`; the pinned commit below does both.
DAOWOD_GIT_URL = "https://github.com/gubiczam/distribution-aware-owod.git"
DAOWOD_COMMIT = ""                  # "" = default branch tip
PROB_GIT_URL = "https://github.com/gubiczam/PROB.git"
PROB_COMMIT = "980cf3a796f064dd4c56f573ba10cc755143e116"

# Google Drive layout. DATASET_DIR must be a VOC-style tree:
#   DATASET_DIR/{Annotations,JPEGImages,ImageSets/OWDETR}
DRIVE_ROOT = "/content/drive/MyDrive"
DATASET_DIR = f"{DRIVE_ROOT}/owod_stage"
DATASET_ARCHIVE = f"{DRIVE_ROOT}/DAOWOD/assets/owod_stage.tar.gz"   # optional fallback
CHECKPOINT_PATH = f"{DRIVE_ROOT}/results/SOWODB/t1.pth"             # PROB Task-1 checkpoint
DINO_WEIGHTS_PATH = f"{DRIVE_ROOT}/PROB/models/dino_resnet50_pretrain.pth"

# Which images the detector runs over. Task-1 train side of S-OWODB / OWDETR.
SPLIT_NAME = "pilot_t1_train_4000"   # ImageSets/OWDETR/<SPLIT_NAME>.txt

# Protocol constants for S-OWODB / OWDETR Task 1.
DATASET = "OWDETR"
PREVIOUS_INTRODUCED_CLASSES = 0
CURRENT_INTRODUCED_CLASSES = 19
NUM_CLASSES = 81
OBJECTNESS_TEMPERATURE = 1.0
BATCH_SIZE = 2
DEVICE = "cuda"                      # "cpu" only works with EXISTING_EXPORT set

# Where results and the proposal cache go. Keep the cache on Drive to survive a
# disconnect; keep results local and copy the ZIP at the end (faster).
OUTPUT_DIR = "/content/daowod_results"
CACHE_DIR = f"{DRIVE_ROOT}/DAOWOD/proposal_cache"
COPY_RESULTS_TO_DRIVE = f"{DRIVE_ROOT}/DAOWOD/contribution_a"

# Behaviour switches.
ALLOW_INSTALL = True                 # install dependencies and build the CUDA op
STAGE_DATASET_LOCALLY = True         # copy the needed images off the Drive mount first
RUNTIME_BUDGET_HOURS = 4.5           # the run downscales its pool to fit this
TARGET_TAIL_RECALL = 0.5             # "cost to reach" target in the report
FORCE_RERUN = False                  # True ignores every cached stage
EXPECTED_CHECKPOINT_SHA256 = ""      # optional integrity pin

# Reuse a proposal export instead of running the detector (skips all GPU work).
EXISTING_EXPORT = ""
# ===========================================================================
"""
)

md("## 2. Environment and GPU")

code(
    '''
import json, os, platform, subprocess, sys, time
from pathlib import Path

NOTEBOOK_STARTED = time.time()
IN_COLAB = "google.colab" in sys.modules or Path("/content").is_dir()
print("python  :", sys.version.split()[0], "|", platform.platform())
print("colab   :", IN_COLAB)


def sh(command, cwd=None, check=True, timeout=3600, quiet=False):
    """Run a shell command, echoing it, and fail loudly with its output."""
    print("$", " ".join(str(part) for part in command))
    result = subprocess.run(
        [str(part) for part in command], cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    if not quiet and result.stdout:
        print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:], file=sys.stderr)
        if check:
            raise RuntimeError(f"Command failed ({result.returncode})")
    return result


gpu = sh(["nvidia-smi"], check=False).stdout
print(gpu or "no nvidia-smi: only a run with EXISTING_EXPORT set will work here")
'''
)

md("## 3. Google Drive")

code(
    """
if IN_COLAB:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except Exception as error:  # already mounted, or running outside Colab
        print("Drive mount skipped:", error)
print("Drive root exists:", Path(DRIVE_ROOT).exists())
"""
)

md("## 4. Repositories")

code(
    """
DAOWOD_REPO = Path("/content/distribution-aware-owod")
PROB_REPO = Path("/content/PROB")


def checkout(url, commit, target):
    if not target.exists():
        sh(["git", "clone", url, str(target)], timeout=1800)
    if commit:
        sh(["git", "-C", str(target), "fetch", "--all", "--tags"], timeout=1800, quiet=True)
        sh(["git", "-C", str(target), "checkout", commit])
    head = sh(["git", "-C", str(target), "rev-parse", "HEAD"], quiet=True).stdout.strip()
    print(f"{target.name} at {head}")
    return head


DAOWOD_HEAD = checkout(DAOWOD_GIT_URL, DAOWOD_COMMIT, DAOWOD_REPO) if IN_COLAB else None
PROB_HEAD = checkout(PROB_GIT_URL, PROB_COMMIT, PROB_REPO) if IN_COLAB else None
if not IN_COLAB:
    # Local development: use the working tree this notebook lives in.
    DAOWOD_REPO = Path.cwd() if (Path.cwd() / "src" / "daowod").is_dir() else Path.cwd().parent
    PROB_REPO = Path(os.environ.get("PROB_REPO", PROB_REPO))
    print("local DAOWOD:", DAOWOD_REPO, "| local PROB:", PROB_REPO)
"""
)

md(
    """
## 5. Dependencies and the deformable-attention CUDA extension

DAOWOD is installed editable. PROB needs two pure-Python packages Colab lacks and
the compiled `MultiScaleDeformableAttention` op — without the op, CUDA inference
falls back to nothing and the bridge fails. The build takes a few minutes and is
skipped when the extension already imports.

If pip reports that it changed `numpy` or `torch`, restart the runtime and re-run
from the top before continuing: the already-imported binary extensions would
otherwise be linked against the previous version.
"""
)

code(
    """
if ALLOW_INSTALL:
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]"], cwd=DAOWOD_REPO, timeout=1800)
    if not EXISTING_EXPORT:
        sh([sys.executable, "-m", "pip", "install", "-q", "einops", "wandb", "pycocotools"], timeout=1800)

if not EXISTING_EXPORT and DEVICE == "cuda":
    try:
        import MultiScaleDeformableAttention  # noqa: F401

        print("deformable attention extension already available")
    except ImportError:
        sh([sys.executable, "-m", "pip", "install", "-v", "."], cwd=PROB_REPO / "models" / "ops", timeout=3600)
        sh([sys.executable, "test.py"], cwd=PROB_REPO / "models" / "ops", timeout=1800)

import daowod
import numpy

print("daowod package:", Path(daowod.__file__).parent)
print("numpy:", numpy.__version__, "- restart the runtime if pip changed this")
"""
)

md(
    """
## 6. Dataset and checkpoint

The dataset directory is used directly if present, otherwise the archive is
extracted once into the cache. PROB loads its DINO backbone initialisation from a
path relative to its own checkout, so that file is linked into place — the
checkpoint overwrites those weights, but the file must exist for the model to
build.
"""
)

code(
    """
import shutil, tarfile, zipfile

data_root = Path(DATASET_DIR)
if not data_root.is_dir():
    archive = Path(DATASET_ARCHIVE)
    if not archive.exists():
        raise FileNotFoundError(
            f"Neither DATASET_DIR ({data_root}) nor DATASET_ARCHIVE ({archive}) exists."
        )
    extracted = Path("/content/owod_stage")
    if not extracted.is_dir():
        extracted.mkdir(parents=True, exist_ok=True)
        print("extracting", archive)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(extracted)
        else:
            with tarfile.open(archive) as handle:
                handle.extractall(extracted)
    inner = [item for item in extracted.iterdir() if (item / "Annotations").is_dir()]
    data_root = inner[0] if inner else extracted
DATA_ROOT = str(data_root)
SPLIT_FILE = str(data_root / "ImageSets" / DATASET / f"{SPLIT_NAME}.txt")
print("data root :", DATA_ROOT)
print("split file:", SPLIT_FILE, "->", Path(SPLIT_FILE).exists())

if not EXISTING_EXPORT:
    dino_target = PROB_REPO / "models" / "dino_resnet50_pretrain.pth"
    if not dino_target.exists():
        source = Path(DINO_WEIGHTS_PATH)
        if not source.exists():
            raise FileNotFoundError(
                f"PROB initialises its backbone from {dino_target}, and "
                f"DINO_WEIGHTS_PATH ({source}) is missing. Download "
                "dino_resnet50_pretrain.pth into Drive, or point DINO_WEIGHTS_PATH at it."
            )
        dino_target.symlink_to(source)
    print("dino weights:", dino_target, dino_target.exists())
"""
)

md(
    """
## 6b. Stage the dataset onto local disk

The detector opens one JPEG per forward pass. On a Drive FUSE mount each of those
opens costs tens of milliseconds, which lands directly in detector
seconds-per-image — the term the runtime budget is most sensitive to. Copying the
images this run needs turns thousands of high-latency reads into one bulk transfer.
Already-copied files are skipped, so an interrupted stage resumes.
"""
)

code(
    """
from daowod import export_cache, preflight

if STAGE_DATASET_LOCALLY and not EXISTING_EXPORT:
    wanted = preflight.read_image_ids(SPLIT_FILE)
    report = export_cache.stage_dataset(
        source=DATA_ROOT,
        destination="/content/owod_local",
        image_ids=wanted,
        dataset=DATASET,
        progress=print,
    )
    print(json.dumps(report, indent=2))
    DATA_ROOT = report["destination"]
    SPLIT_FILE = str(Path(DATA_ROOT) / "ImageSets" / DATASET / f"{SPLIT_NAME}.txt")
print("data root in use:", DATA_ROOT)
"""
)

md(
    """
## 7. Repository self-check

The library's own tests and linters run here, before any GPU time. A failure means
the checkout is broken, not the experiment.
"""
)

code(
    """
sh([sys.executable, "-m", "compileall", "-q", "src"], cwd=DAOWOD_REPO)
sh([sys.executable, "-m", "pytest", "-q", "-x"], cwd=DAOWOD_REPO, timeout=1800)
sh([sys.executable, "-m", "ruff", "check", "."], cwd=DAOWOD_REPO, check=False)
"""
)

md(
    """
## 8. Build the run configuration

`PipelineConfig` is the whole experiment as data. Printing the resolved mode shows
exactly how many images the detector will see and how large the comparison matrix
is, before anything expensive starts.
"""
)

code(
    """
from daowod import modes
from daowod.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(
    mode=RUN_MODE,
    data_root=DATA_ROOT,
    split_file=SPLIT_FILE,
    prob_repository=str(PROB_REPO),
    checkpoint=CHECKPOINT_PATH,
    output_dir=OUTPUT_DIR,
    cache_dir=CACHE_DIR,
    dataset=DATASET,
    previous_introduced_classes=PREVIOUS_INTRODUCED_CLASSES,
    current_introduced_classes=CURRENT_INTRODUCED_CLASSES,
    num_classes=NUM_CLASSES,
    objectness_temperature=OBJECTNESS_TEMPERATURE,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    python_executable=sys.executable,
    expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
    existing_export=EXISTING_EXPORT,
    require_gpu=not EXISTING_EXPORT and DEVICE == "cuda",
    runtime_budget_seconds=RUNTIME_BUDGET_HOURS * 3600,
    target_tail_recall=TARGET_TAIL_RECALL,
    force=FORCE_RERUN,
)
mode = config.execution_mode()
print(json.dumps(mode.as_dict(), indent=2, default=str))
print(
    f"\\ndetector images: {mode.total_images}"
    f"  |  study cells: {len(mode.imbalance_settings)} severities"
    f" x {len(mode.strategies)} strategies x {len(mode.seeds)} seeds"
)
"""
)

md(
    """
## 9. Preflight

Every precondition is verified, not assumed: interpreter, packages, GPU and memory,
PROB checkout markers, the bridge's own `check` subcommand, the checkpoint, disk
space, the dataset tree, and one real annotation parse. A single `FAIL` stops here
rather than three hours in.
"""
)

code(
    """
import pandas as pd

from daowod import preflight
from daowod.pipeline import stage_preflight

checks = stage_preflight(config, mode)
display(pd.DataFrame(preflight.rows(checks)))
print(json.dumps(preflight.summarise(checks), indent=2))
"""
)

md(
    """
## 10. Run the experiment

One call. It exports proposals (cached, resumable), builds the pools, validates the
severities, proves the leakage controls, picks the coherence definition on a
disjoint pilot pool, projects runtime and shrinks the pool if the projection
exceeds the budget, runs the strategy matrix and the ablation grid, then writes
every CSV, figure, the markdown summary and the ZIP.

Re-running after a disconnect reuses finished detector chunks and finished
severities.
"""
)

code(
    """
result = run_pipeline(config)
print("\\nstage seconds:", json.dumps(result.stage_seconds, indent=2))
print("archive:", result.archive)
"""
)

md("## 11. Headline results")

code(
    """
summary = pd.DataFrame(result.headline())
display(summary.sort_values(["imbalance_setting", "tail_discovery_auc_mean"], ascending=[True, False]))

print("Paired contrasts (positive = the gate wins):")
display(pd.DataFrame(result.contrasts))

# Every arm against the baseline, and the absolute discovered-object counts. Read
# these before the recalls: the tail denominator is small enough that a recall of
# 0.038 is a single object.
for table in ("arm_comparison_unknown.csv", "discovery_counts.csv"):
    path = result.output_dir / table
    if path.exists() and path.stat().st_size:
        print()
        print(table)
        display(pd.read_csv(path))
"""
)

code(
    """
print("Long-tail severities actually achieved:")
display(
    pd.DataFrame(result.severity_rows)[
        [
            "setting",
            "requested_imbalance_ratio",
            "achieved_imbalance_ratio",
            "head_to_tail_object_ratio",
            "unknown_objects_before",
            "unknown_objects_after",
            "tail_objects_after",
            "imbalance_ratio_saturated",
        ]
    ]
)
print(result.severity_verdict)
"""
)

md("## 12. Figures")

code(
    """
from IPython.display import Image, display as show

for name in (
    "figure_headline_comparison",
    "figure_family_panels",
    "figure_all_arms_auc",
    "figure_tail_discovery_vs_budget",
    "figure_unknown_discovery_vs_budget",
    "figure_group_discovery",
    "figure_annotation_efficiency",
    "figure_tail_auc_by_severity",
    "figure_unique_classes",
    "figure_component_distributions",
    "figure_gate_suppression",
    "figure_long_tail_protocol",
    "figure_ablation_heatmap",
    "figure_cost_to_target",
):
    path = result.output_dir / f"{name}.png"
    if path.exists():
        print(name)
        show(Image(filename=str(path)))
"""
)

md("## 13. Research summary")

code(
    """
from IPython.display import Markdown, display as show_markdown

show_markdown(Markdown(result.summary_path.read_text(encoding="utf-8")))
"""
)

md(
    """
## 14. Artifact verification and delivery

Every promised file is checked for existence and non-zero size. The ZIP is copied
to Drive so it survives the runtime, and offered as a browser download.
"""
)

code(
    """
display(pd.DataFrame(result.artifacts))
print("all artifacts present:", all(row["status"] == "PASS" for row in result.artifacts))

if COPY_RESULTS_TO_DRIVE:
    destination = Path(COPY_RESULTS_TO_DRIVE)
    destination.mkdir(parents=True, exist_ok=True)
    copied = shutil.copy2(result.archive, destination / result.archive.name)
    shutil.copy2(result.summary_path, destination / result.summary_path.name)
    print("copied to Drive:", copied)

if IN_COLAB:
    try:
        from google.colab import files

        files.download(str(result.archive))
    except Exception as error:
        print("download skipped:", error)

print(f"total notebook runtime: {(time.time() - NOTEBOOK_STARTED) / 3600:.2f} h")
"""
)

md(
    """
## 15. Representation Experiment E4 — is the failure the embedding or the formulation?

The first two experiments established, reproducibly, that the coherence gate does not
improve tail discovery, and the component audit localised the cause to the *geometry*
of PROB's decoder space: a tail region's ten nearest neighbours are 1.5 % its own
class while a background region's are 88.8 % background, so any density-based
coherence term ranks background highest.

That is a claim about one feature space. E4 tests whether it is a claim about the
*formulation* by holding the acquisition completely fixed and varying only the space
its neighbourhoods are computed in. The decisive statistic is

```text
tail purity advantage = tail same-label neighbour fraction
                      / background same-label neighbour fraction
```

which must exceed 1.0 for a coherence term to be able to prefer rare objects at all.
It is 0.017 in the baseline space.

Three steps, each cached and each runnable on its own. The re-embedding needs a torch
environment; on Colab that is the same runtime, so `sys.executable` is used instead of
the local PROB virtual environment.
"""
)

code(
    """
E4_DIR = f"{OUTPUT_DIR}/e4"
E4_REPRESENTATIONS = f"{E4_DIR}/representations"
E4_EXPORT = EXISTING_EXPORT or str(
    next(Path(CACHE_DIR).glob("*.npz"), Path(CACHE_DIR) / "merged_export.npz")
)
print("E4 will re-embed the boxes in:", E4_EXPORT)

# 1. Which export rows the experiment actually reads (about 20 % of them).
sh([sys.executable, "analysis/e4_required_rows.py",
    "--export", E4_EXPORT, "--output", E4_REPRESENTATIONS], cwd=DAOWOD_REPO, timeout=1800)
"""
)

code(
    """
# 2. Re-embed those regions with encoders trained under different objectives.
#    DINO ResNet-50 comes from the PROB checkout; ImageNet ResNet-50 from the
#    torchvision cache. Nothing is downloaded. Resumable: finished chunks are reused.
sh([sys.executable, "analysis/extract_region_embeddings.py",
    "--export", E4_EXPORT,
    "--images", f"{DATA_ROOT}/JPEGImages",
    "--output", E4_REPRESENTATIONS,
    "--rows", f"{E4_REPRESENTATIONS}/rows.npy",
    "--device", "cuda" if DEVICE == "cuda" else "cpu",
    "--chunk-images", "200"],
   cwd=DAOWOD_REPO, timeout=14400)
"""
)

code(
    """
# 3. Phase 3 and 4: geometry statistics and the projections, for every space.
sh([sys.executable, "analysis/experiment_e4_representations.py",
    "--export", E4_EXPORT,
    "--annotations", f"{DATA_ROOT}/Annotations",
    "--representations", E4_REPRESENTATIONS,
    "--output", f"{E4_DIR}/geometry"],
   cwd=DAOWOD_REPO, timeout=14400)

display(pd.read_csv(Path(DAOWOD_REPO) / f"{E4_DIR}/geometry/headline_tail_purity.csv"))
show_markdown(Markdown((Path(DAOWOD_REPO) / f"{E4_DIR}/geometry/e4_geometry_summary.md").read_text()))
"""
)

code(
    """
for figure in sorted((Path(DAOWOD_REPO) / f"{E4_DIR}/geometry").glob("figure_e4_*.png")):
    print(figure.name)
    show(Image(filename=str(figure)))
"""
)

md(
    """
### Phase 5 — the same strategies, a different space

This re-runs the frozen strategy ladder once per representation. Nothing about the
acquisition changes; only `embeddings` in the export differs. `v2:random` and
`v2:objectness_area_prior` cannot depend on the embedding, so their results must be
identical across every space — the run checks that, and a violation means the
comparison is void.

Budget roughly 30 minutes of CPU per representation.
"""
)

code(
    """
sh([sys.executable, "analysis/run_e4_active_learning.py",
    "--export", E4_EXPORT,
    "--annotations", f"{DATA_ROOT}/Annotations",
    "--representations", E4_REPRESENTATIONS,
    "--output", f"{E4_DIR}/active_learning"],
   cwd=DAOWOD_REPO, timeout=28800)

base = Path(DAOWOD_REPO) / f"{E4_DIR}/active_learning"
display(pd.read_csv(base / "e4_invariance_check.csv"))
display(pd.read_csv(base / "e4_representation_comparison.csv"))
show_markdown(Markdown((base / "e4_active_learning_summary.md").read_text()))
"""
)

md(
    """
## 16. How to read the output

**`budget_curves.csv`** — one row per (severity, strategy, seed, budget) with every
metric: unknown / head / medium / tail discovery recall, unique classes discovered,
annotation precision, background selection rate, isolated-outlier selection rate,
and embedding-space diversity. Discovery counts **distinct ground-truth objects**,
so forty proposals on one object count once.

**`strategy_auc.csv`** — each budget curve collapsed to a normalised AUC (a mean
recall over the budget sweep, in [0, 1]) plus the final-budget values.

**`headline_contrasts.csv`** — the gated strategy minus each weaker rung, paired by
seed and severity. Paired because all strategies share one pool and one export.

**`gate_suppression.csv`** — the mechanism. Rank the pool by ungated rarity, rank it
by the gated interaction, and count what changes hands in the top-K:
`suppressed_isolated` is the isolated outliers the gate removed,
`promoted_true_unknown` and `promoted_tail` are what it bought instead.

**`ablations.csv`** — gate form (multiplicative / additive / none) x coherence
definition x neighbourhood size. The comparison against `additive` at equal total
gamma is the one that isolates the gate's *form* rather than the presence of a
coherence signal.

**Denominators.** Recall denominators are the unknown objects *reachable from the
candidate pool*. Objects no proposal covers are excluded from both numerator and
denominator; `*_objects_reachable` in every row states the number used.

**Next step.** Feed `selected_proposals.csv` for a chosen strategy and budget into
PROB fine-tuning to measure the downstream effect on the official OWOD metrics.
That is a separate, detector-side experiment; this notebook stops at the annotation
set.
"""
)


def notebook() -> dict:
    cells = []
    for kind, source in CELLS:
        lines = source.split("\n")
        payload = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cell = {
            "cell_type": kind,
            "id": f"daowod-{len(cells):02d}",
            "metadata": {},
            "source": payload,
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


target = Path("notebooks/contribution_a_active_annotation.ipynb")
target.write_text(json.dumps(notebook(), indent=1) + "\n", encoding="utf-8")
print(f"wrote {target} with {len(CELLS)} cells")
