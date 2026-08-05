"""Feature spaces for Representation Experiment E4, as data.

E4 asks one question: is the coherence gate's failure a property of *PROB's decoder
embedding* or of the coherence formulation? Answering it requires holding the
acquisition formulation fixed and varying only the space its neighbourhoods are
computed in. This module enumerates the spaces that can be produced from the
existing proposal export plus encoders already on disk, so that "swap the
representation" is a one-line change rather than a new pipeline.

Three kinds of space
--------------------
``prob``
    Read straight out of the export the detector wrote. No extra computation, and
    ``prob_decoder`` is the frozen baseline's representation — the reference every
    other space is measured against.

``crop``
    Produced by ``experiments/extract_embeddings.py``, which re-embeds the same
    predicted boxes with an encoder trained under a different objective. Requires a
    torch environment, so it runs as a subprocess and lands as an NPZ that this
    module only reads.

``derived``
    A deterministic transform of another space — whitening, removal of leading
    principal components, concatenation. These cost nothing and they test a
    specific alternative explanation: that the information is present in the
    decoder space but *drowned* by a few high-variance directions carrying the
    background mass, rather than absent. If removing those directions repairs the
    neighbourhoods, the diagnosis is a metric problem, not a representation problem.

Nothing here fits anything on ground truth. PCA is fitted on the pool's own
embeddings, which is as unsupervised as the k-means the baseline already uses, so a
derived space remains a legitimate acquisition input rather than an oracle-assisted
one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA

FloatArray = NDArray[np.float64]

#: The representation the frozen experiments used. Every comparison is read against
#: this, and it is never modified.
BASELINE_REPRESENTATION = "prob_decoder"


class RepresentationError(ValueError):
    """Raised when a representation cannot be produced from the given inputs."""


@dataclass(frozen=True)
class RepresentationSpec:
    """One feature space, and everything needed to reproduce it."""

    name: str
    kind: str
    description: str
    source: str = ""
    base: str = ""
    parameters: Mapping[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.kind not in ("prob", "crop", "derived"):
            raise RepresentationError(f"{self.name}: unknown kind {self.kind!r}.")
        if self.kind == "derived" and not self.base:
            raise RepresentationError(f"{self.name}: a derived space needs a base.")
        object.__setattr__(self, "parameters", dict(self.parameters or {}))

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "source": self.source,
            "base": self.base,
            "parameters": dict(self.parameters),
        }


#: Every space E4 can build on this machine, without network access and without
#: retraining anything.
REGISTRY: tuple[RepresentationSpec, ...] = (
    RepresentationSpec(
        name="prob_decoder",
        kind="prob",
        description="PROB's final decoder hidden state hs[-1], 256-d per query. The "
        "frozen baseline's representation.",
        source="export['embeddings']",
    ),
    RepresentationSpec(
        name="prob_posterior",
        kind="prob",
        description="The exported class posterior, 20-d (19 known + unknown). A "
        "semantic space the detector was explicitly trained to shape.",
        source="export['posterior']",
    ),
    RepresentationSpec(
        name="prob_geometry",
        kind="prob",
        description="Objectness and box geometry only: objectness, box scale, "
        "aspect ratio, centre. The space behind the free informativeness prior.",
        source="export['objectness'] + export['boxes']",
    ),
    RepresentationSpec(
        name="dino_resnet50",
        kind="crop",
        description="Self-supervised DINO ResNet-50 pooled features over the "
        "cropped predicted region, 2048-d.",
        source="experiments/extract_embeddings.py",
    ),
    RepresentationSpec(
        name="imagenet_resnet50",
        kind="crop",
        description="Supervised ImageNet ResNet-50 pooled features over the same "
        "crops, 2048-d. The closed-set-supervision control.",
        source="experiments/extract_embeddings.py",
    ),
    RepresentationSpec(
        name="prob_decoder_whitened",
        kind="derived",
        description="PROB decoder space, PCA-whitened. Tests whether the "
        "neighbourhood structure is present but dominated by a few high-variance "
        "directions.",
        base="prob_decoder",
        parameters={"transform": "whiten", "components": 64},
    ),
    RepresentationSpec(
        name="prob_decoder_minus_top4",
        kind="derived",
        description="PROB decoder space with its four leading principal components "
        "removed. If the background mass lives in those directions, removing them "
        "should raise tail neighbourhood purity.",
        base="prob_decoder",
        parameters={"transform": "remove_leading", "components": 4},
    ),
    RepresentationSpec(
        name="dino_whitened",
        kind="derived",
        description="DINO ResNet-50 space, PCA-whitened to 64 dimensions, so the "
        "comparison against the whitened decoder space is dimension-matched.",
        base="dino_resnet50",
        parameters={"transform": "whiten", "components": 64},
    ),
    RepresentationSpec(
        name="prob_decoder_plus_dino",
        kind="derived",
        description="L2-normalised concatenation of the decoder and DINO spaces, "
        "equally weighted. Tests whether the detector's semantics add anything to a "
        "self-supervised appearance space.",
        base="prob_decoder",
        parameters={"transform": "fuse", "with": "dino_resnet50"},
    ),
)

SPEC_BY_NAME: Mapping[str, RepresentationSpec] = {spec.name: spec for spec in REGISTRY}


def resolve(name: str) -> RepresentationSpec:
    if name not in SPEC_BY_NAME:
        raise RepresentationError(
            f"Unknown representation {name!r}. Available: {sorted(SPEC_BY_NAME)}"
        )
    return SPEC_BY_NAME[name]


def available(
    *, export: Mapping[str, NDArray[np.generic]], directory: str | Path | None = None
) -> list[str]:
    """Names that can actually be built from what is on disk right now.

    Reported rather than assumed so a run states which spaces it compared and which
    it could not obtain, instead of silently comparing fewer.
    """

    ready: list[str] = []
    for spec in REGISTRY:
        try:
            _require(spec, export=export, directory=directory)
        except RepresentationError:
            continue
        ready.append(spec.name)
    return ready


def _require(
    spec: RepresentationSpec,
    *,
    export: Mapping[str, NDArray[np.generic]],
    directory: str | Path | None,
) -> None:
    if spec.kind == "prob":
        needed = {
            "prob_decoder": ("embeddings",),
            "prob_posterior": ("posterior",),
            "prob_geometry": ("objectness", "boxes"),
        }[spec.name]
        missing = [field for field in needed if field not in export]
        if missing:
            raise RepresentationError(f"{spec.name}: export is missing {missing}.")
        return
    if spec.kind == "crop":
        if directory is None:
            raise RepresentationError(f"{spec.name}: no representation directory given.")
        path = Path(directory) / f"{spec.name}.npz"
        if not path.exists():
            raise RepresentationError(
                f"{spec.name}: {path} does not exist. Run experiments/extract_embeddings.py first."
            )
        return
    _require(resolve(spec.base), export=export, directory=directory)
    partner = str(spec.parameters.get("with", ""))
    if partner:
        _require(resolve(partner), export=export, directory=directory)


def _unit(matrix: ArrayLike) -> FloatArray:
    array = np.asarray(matrix, dtype=np.float64)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def load(
    name: str,
    *,
    export: Mapping[str, NDArray[np.generic]],
    directory: str | Path | None = None,
    rows: ArrayLike | None = None,
    seed: int = 0,
) -> tuple[FloatArray, dict[str, object]]:
    """The embedding matrix for one representation, plus a reproducibility manifest.

    ``rows`` restricts to a subset of the export's proposals *before* any derived
    transform is fitted, which matters: whitening fitted on the whole export and
    whitening fitted on the candidate pool are different spaces, and the one the
    acquisition sees is the pool's.
    """

    spec = resolve(name)
    _require(spec, export=export, directory=directory)
    selector = None if rows is None else np.asarray(rows, dtype=np.int64)

    if spec.kind == "prob":
        matrix = _load_prob(spec, export)
    elif spec.kind == "crop":
        path = Path(directory) / f"{spec.name}.npz"  # type: ignore[arg-type]
        with np.load(path, allow_pickle=True) as handle:
            matrix = np.asarray(handle["embeddings"], dtype=np.float64)
        if matrix.shape[0] != np.asarray(export["embeddings"]).shape[0]:
            raise RepresentationError(
                f"{spec.name}: {matrix.shape[0]} embeddings for "
                f"{np.asarray(export['embeddings']).shape[0]} exported proposals; the "
                "alignment contract is broken."
            )
    else:
        matrix, _ = load(spec.base, export=export, directory=directory, rows=rows, seed=seed)
        return _derive(spec, matrix, export=export, directory=directory, rows=rows, seed=seed)

    if selector is not None:
        matrix = matrix[selector]
    manifest = dict(spec.as_dict())
    manifest.update(
        {
            "rows": int(matrix.shape[0]),
            "dimensions": int(matrix.shape[1]),
            "mean_l2_norm": float(np.linalg.norm(matrix, axis=1).mean()),
        }
    )
    if spec.kind == "crop":
        side = Path(directory) / f"{spec.name}.json"  # type: ignore[arg-type]
        if side.exists():
            manifest["extraction"] = json.loads(side.read_text(encoding="utf-8"))
    return matrix, manifest


def _load_prob(spec: RepresentationSpec, export: Mapping[str, NDArray[np.generic]]) -> FloatArray:
    if spec.name == "prob_decoder":
        return np.asarray(export["embeddings"], dtype=np.float64)
    if spec.name == "prob_posterior":
        posterior = np.asarray(export["posterior"], dtype=np.float64)
        # Log-probabilities rather than probabilities: the posterior spans orders of
        # magnitude and a Euclidean or cosine neighbourhood over raw probabilities is
        # dominated by whichever class happens to be largest.
        return np.log(np.maximum(posterior, 1e-12))
    objectness = np.asarray(export["objectness"], dtype=np.float64).reshape(-1, 1)
    boxes = np.asarray(export["boxes"], dtype=np.float64)
    scale = np.sqrt(np.clip(boxes[:, 2] * boxes[:, 3], 1e-12, 1.0)).reshape(-1, 1)
    aspect = np.log(np.maximum(boxes[:, 2], 1e-6) / np.maximum(boxes[:, 3], 1e-6)).reshape(-1, 1)
    return np.hstack([objectness, scale, aspect, boxes[:, :2]])


def _derive(
    spec: RepresentationSpec,
    base: FloatArray,
    *,
    export: Mapping[str, NDArray[np.generic]],
    directory: str | Path | None,
    rows: ArrayLike | None,
    seed: int,
) -> tuple[FloatArray, dict[str, object]]:
    transform = str(spec.parameters.get("transform", ""))
    manifest = dict(spec.as_dict())

    if transform == "whiten":
        components = int(spec.parameters.get("components", 64))
        components = int(min(components, base.shape[1], base.shape[0]))
        model = PCA(n_components=components, whiten=True, random_state=seed)
        matrix = np.asarray(model.fit_transform(base), dtype=np.float64)
        manifest["explained_variance_ratio_sum"] = float(model.explained_variance_ratio_.sum())
    elif transform == "remove_leading":
        components = int(spec.parameters.get("components", 4))
        components = int(min(components, base.shape[1] - 1, base.shape[0] - 1))
        model = PCA(n_components=components, random_state=seed)
        model.fit(base)
        centred = base - model.mean_
        # Project out the leading directions rather than dropping coordinates: the
        # remaining space keeps the original dimensionality, so a neighbourhood in it
        # is comparable with one in the untransformed space.
        matrix = centred - (centred @ model.components_.T) @ model.components_
        manifest["removed_variance_ratio"] = float(model.explained_variance_ratio_.sum())
    elif transform == "fuse":
        partner_name = str(spec.parameters["with"])
        partner, partner_manifest = load(
            partner_name, export=export, directory=directory, rows=rows, seed=seed
        )
        # Each side is L2-normalised first, so the concatenation weights the two
        # spaces equally instead of by whichever happens to have larger norms.
        matrix = np.hstack([_unit(base), _unit(partner)])
        manifest["partner"] = partner_manifest.get("name", partner_name)
        manifest["partner_dimensions"] = int(partner.shape[1])
    else:
        raise RepresentationError(f"{spec.name}: unknown transform {transform!r}.")

    manifest.update(
        {
            "rows": int(matrix.shape[0]),
            "dimensions": int(matrix.shape[1]),
            "mean_l2_norm": float(np.linalg.norm(matrix, axis=1).mean()),
        }
    )
    return matrix, manifest


def substituted_export(
    export: Mapping[str, NDArray[np.generic]],
    embeddings: ArrayLike,
) -> dict[str, NDArray[np.generic]]:
    """A copy of the export with its ``embeddings`` replaced, everything else intact.

    This is the whole isolation mechanism for Phase 5. Because the candidate filter,
    the oracle, the severity masks and the budget prefixes are all functions of
    fields this does *not* touch, the pool a strategy searches is bit-identical
    across representations, and the only quantity that varies is the geometry the
    score's novelty, rarity and coherence terms are computed in.
    """

    matrix = np.asarray(embeddings, dtype=np.float64)
    base = np.asarray(export["embeddings"])
    if matrix.shape[0] != base.shape[0]:
        raise RepresentationError(
            f"Replacement embeddings have {matrix.shape[0]} rows against the "
            f"export's {base.shape[0]}."
        )
    replaced = {name: np.asarray(values) for name, values in export.items()}
    replaced["embeddings"] = matrix
    return replaced


def write_substituted_export(
    path: str | Path,
    *,
    export: Mapping[str, NDArray[np.generic]],
    embeddings: ArrayLike,
    manifest: Mapping[str, object] | None = None,
) -> Path:
    """Write a drop-in export NPZ carrying a different representation."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    replaced = substituted_export(export, embeddings)
    np.savez(target, **replaced)
    if manifest is not None:
        target.with_suffix(".json").write_text(
            json.dumps(dict(manifest), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return target


def describe_registry() -> list[dict[str, object]]:
    """Rows for ``representation_registry.csv``."""

    return [spec.as_dict() for spec in REGISTRY]


def audit_rows(
    *,
    export: Mapping[str, NDArray[np.generic]],
    directory: str | Path | None = None,
) -> list[dict[str, object]]:
    """What each candidate space is, and whether it is obtainable right now."""

    ready = set(available(export=export, directory=directory))
    rows: list[dict[str, object]] = []
    for spec in REGISTRY:
        row = dict(spec.as_dict())
        row["available"] = spec.name in ready
        if not row["available"]:
            try:
                _require(spec, export=export, directory=directory)
            except RepresentationError as error:
                row["blocked_by"] = str(error)
        rows.append(row)
    return rows


def sequence(names: Sequence[str] | None = None) -> list[str]:
    """Requested names, defaulting to the whole registry, baseline first."""

    if names is None:
        chosen = [spec.name for spec in REGISTRY]
    else:
        chosen = [str(name) for name in names]
        for name in chosen:
            resolve(name)
    if BASELINE_REPRESENTATION in chosen:
        chosen = [BASELINE_REPRESENTATION] + [
            name for name in chosen if name != BASELINE_REPRESENTATION
        ]
    return chosen
