"""Which export rows Representation Experiment E4 must re-embed, and no others.

Re-embedding all 400 000 exported proposals with two encoders costs about two and a
half hours. Only a fraction of them is ever read: the candidate pool of the
evaluation split, the candidate pool of the pilot split, and the prefix of the
reference split that becomes the novelty bank. Those row sets are functions of the
detector's objectness and boxes alone — never of the embeddings — so they are
identical for every representation, and restricting extraction to them changes
nothing about the experiment while making it five times cheaper.

The row list is written out and consumed by ``analysis/extract_region_embeddings.py``
through ``--rows``. ``analysis/experiment_e4_representations.py`` then *verifies*
that every row it actually reads was extracted, so the saving is a checked
precondition rather than an assumption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from daowod import candidates, detector, study
from daowod.modes import resolve_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--output", required=True, help="Where to write rows.npy and its manifest.")
    parser.add_argument("--mode", default="MAINREVEALED")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def candidate_rows(export, *, image_ids, wanted, spec) -> np.ndarray:
    """Export-row indices of the candidate pool built from ``wanted`` images."""

    keep = np.array([str(value) in set(wanted) for value in image_ids.tolist()], dtype=np.bool_)
    subset = np.flatnonzero(keep)
    selection = candidates.build_candidate_pool(
        image_ids=image_ids[subset],
        boxes_cxcywh=np.asarray(export["boxes"])[subset],
        objectness=np.asarray(export["objectness"])[subset],
        unknown_score=np.asarray(export["confidence"])[subset],
        posterior=np.asarray(export["posterior"])[subset],
        predicted_labels=np.asarray(export.get("predicted_labels"))[subset]
        if "predicted_labels" in export
        else None,
        spec=spec,
    )
    return subset[selection.indices]


def main() -> None:
    args = parse_args()
    mode = resolve_mode(args.mode)
    export = study.load_export(args.export)
    image_ids = np.asarray([str(value) for value in export["image_ids"].tolist()], dtype=object)
    available = sorted(set(image_ids.tolist()))
    splits = detector.split_disjoint(
        available,
        counts={
            "reference": mode.reference_images,
            "pilot": mode.pilot_images,
            "evaluation": mode.evaluation_images,
        },
        seed=args.seed,
    )
    spec = mode.study_config().candidate_spec

    evaluation = candidate_rows(export, image_ids=image_ids, wanted=splits["evaluation"], spec=spec)
    pilot = candidate_rows(export, image_ids=image_ids, wanted=splits["pilot"], spec=spec)
    reference_mask = np.array(
        [str(value) in set(splits["reference"]) for value in image_ids.tolist()], dtype=np.bool_
    )
    reference = np.flatnonzero(reference_mask)[: mode.reference_limit]

    rows = np.unique(np.concatenate([evaluation, pilot, reference]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "rows.npy", rows)
    manifest = {
        "export": args.export,
        "mode": mode.name,
        "seed": args.seed,
        "total_export_rows": int(image_ids.shape[0]),
        "required_rows": int(rows.size),
        "fraction_of_export": float(rows.size / image_ids.shape[0]),
        "evaluation_pool_rows": int(evaluation.size),
        "pilot_pool_rows": int(pilot.size),
        "reference_bank_rows": int(reference.size),
        "images_touched": int(np.unique(image_ids[rows]).size),
        "candidate_spec": spec.as_dict(),
    }
    (output / "required_rows.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
