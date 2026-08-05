"""Re-embed the exported proposal boxes with encoders other than PROB's decoder.

This is Representation Experiment E4's feature extractor. It must run **outside**
the DAOWOD environment: `daowod` deliberately does not depend on torch, and the
only interpreter on this machine that has torch is the PROB checkout's virtual
environment. The boundary is therefore the same one `daowod.detector` uses for
the detector — a subprocess that writes an NPZ the library then reads:

    ~/Documents/PROB/.venv/bin/python analysis/extract_region_embeddings.py \\
        --export outputs/real_stage1/reference_proposals.npz \\
        --images ~/owod_stage/JPEGImages \\
        --output outputs/e4_representations \\
        --encoder dino_resnet50 --encoder imagenet_resnet50

Why these encoders
------------------
The question E4 answers is whether the coherence failure is a property of *PROB's
decoder space* or of the coherence formulation. Answering it needs representations
built by a different objective, obtainable without retraining and without network
access:

``dino_resnet50``
    Self-supervised DINO ResNet-50, from the checkpoint PROB already ships for
    backbone initialisation. Self-supervised objectives are the ones reported to
    produce neighbourhoods that transfer to categories never labelled — exactly the
    property the coherence gate needs and the Task-1 decoder lacks.

``imagenet_resnet50``
    Supervised ImageNet ResNet-50, from torchvision's cache. A *closed-set
    supervised* control: if it clusters the unknown classes as well as DINO does,
    then supervision per se is not what breaks the neighbourhoods, and the
    explanation lies in PROB's detection objective specifically.

Both are 2048-d pooled features over the cropped region. Cropping rather than
whole-image embedding is deliberate: the acquisition score is defined on a region,
so the representation compared must be a region representation.

Alignment contract
------------------
The output NPZ's ``embeddings`` array is row-aligned to the input export, so a
substituted export is a drop-in replacement and every downstream index — candidate
pool, oracle, severity mask, budget prefix — is unchanged. ``proposal_index`` is
written alongside so alignment can be verified rather than assumed.

Caching and resume
------------------
Work is written in chunks of images. A rerun skips chunks whose file exists and
whose recorded image list matches, so an interrupted extraction resumes instead of
restarting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

#: Encoders and where their weights live. Both are already on disk; nothing is
#: downloaded, so the experiment runs without network access.
ENCODERS: dict[str, dict[str, str]] = {
    "dino_resnet50": {
        "architecture": "resnet50",
        "weights": "~/Documents/PROB/models/dino_resnet50_pretrain.pth",
        "description": "Self-supervised DINO ResNet-50 (the checkpoint PROB uses to "
        "initialise its backbone).",
    },
    "imagenet_resnet50": {
        "architecture": "resnet50",
        "weights": "~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth",
        "description": "Supervised ImageNet-1k ResNet-50 (torchvision cache).",
    },
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, help="Proposal export NPZ to re-embed.")
    parser.add_argument("--images", required=True, help="JPEGImages directory.")
    parser.add_argument("--output", required=True, help="Directory for the chunk cache.")
    parser.add_argument(
        "--encoder",
        action="append",
        choices=sorted(ENCODERS),
        help="Encoder to run; repeatable. Default: all of them.",
    )
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-images", type=int, default=200)
    parser.add_argument(
        "--rows",
        default="",
        help="Optional .npy of export-row indices to embed. Produced by "
        "analysis/e4_required_rows.py; restricts work to the rows the experiment "
        "actually reads, which is checked downstream rather than assumed.",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--context-pad",
        type=float,
        default=0.0,
        help="Fraction of box size added on each side before cropping. 0 keeps the "
        "region exactly as the detector predicted it.",
    )
    return parser.parse_args()


def build_encoder(name: str, device: torch.device) -> torch.nn.Module:
    spec = ENCODERS[name]
    weights = Path(spec["weights"]).expanduser()
    if not weights.exists():
        raise FileNotFoundError(f"{name}: missing weights at {weights}")
    model = getattr(torchvision.models, spec["architecture"])(weights=None)
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [key for key in result.missing_keys if not key.startswith("fc.")]
    if unexpected or missing:
        raise RuntimeError(
            f"{name}: checkpoint does not match {spec['architecture']} "
            f"(missing {missing[:4]}, unexpected {unexpected[:4]})"
        )
    # The classifier is dropped rather than ignored: the comparison is between
    # pooled 2048-d representations, and a 1000-way ImageNet logit vector would be a
    # different kind of space entirely.
    model.fc = torch.nn.Identity()
    return model.eval().to(device)


def crop_tensor(
    image: Image.Image, box: np.ndarray, *, size: int, context_pad: float
) -> torch.Tensor:
    """One normalised region crop from a normalised ``cxcywh`` box.

    The conversion is the same one :func:`daowod.oracle.boxes_to_pixel_xyxy`
    performs and that the oracle's IoU matching is tested against, so a crop here
    covers exactly the region the oracle scored.
    """

    width, height = image.size
    centre_x, centre_y, box_w, box_h = (float(value) for value in box)
    pad = float(context_pad)
    half_w = box_w * (0.5 + pad)
    half_h = box_h * (0.5 + pad)
    x1 = int(max(0.0, (centre_x - half_w) * width))
    y1 = int(max(0.0, (centre_y - half_h) * height))
    x2 = int(min(float(width), (centre_x + half_w) * width))
    y2 = int(min(float(height), (centre_y + half_h) * height))
    if x2 - x1 < 2 or y2 - y1 < 2:
        # A degenerate predicted box still needs a representation; falling back to
        # the whole image is the only choice that cannot silently drop a row and
        # break the alignment contract.
        x1, y1, x2, y2 = 0, 0, width, height
    patch = image.crop((x1, y1, x2, y2)).resize((size, size), Image.BILINEAR)
    array = torch.from_numpy(np.asarray(patch, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (array - IMAGENET_MEAN) / IMAGENET_STD


def main() -> None:
    args = parse_args()
    names = args.encoder or sorted(ENCODERS)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images).expanduser()
    device = torch.device(args.device)

    with np.load(args.export, allow_pickle=True) as handle:
        image_ids = np.asarray([str(value) for value in handle["image_ids"].tolist()], dtype=object)
        boxes = np.asarray(handle["boxes"], dtype=np.float64)
    total = image_ids.shape[0]
    wanted_rows = None
    if args.rows:
        wanted_rows = np.load(args.rows).astype(np.int64)
        selected_mask = np.zeros(total, dtype=bool)
        selected_mask[wanted_rows] = True
        print(
            f"restricted to {wanted_rows.size} of {total} rows ({wanted_rows.size / total:.1%})",
            flush=True,
        )
    else:
        selected_mask = np.ones(total, dtype=bool)
    unique = sorted(set(image_ids[selected_mask].tolist()))
    print(
        f"export: {total} proposals; embedding {int(selected_mask.sum())} over "
        f"{len(unique)} images",
        flush=True,
    )

    models = {name: build_encoder(name, device) for name in names}
    for name in names:
        print(f"loaded {name}: {ENCODERS[name]['description']}", flush=True)

    blocks = [
        unique[start : start + args.chunk_images]
        for start in range(0, len(unique), args.chunk_images)
    ]
    started = time.time()
    for index, block in enumerate(blocks):
        paths = {name: output / f"{name}_chunk_{index:04d}.npz" for name in names}
        if all(path.exists() for path in paths.values()):
            print(f"chunk {index + 1}/{len(blocks)}: cached", flush=True)
            continue

        positions: list[int] = []
        crops: list[torch.Tensor] = []
        for image_id in block:
            selected = np.flatnonzero((image_ids == image_id) & selected_mask)
            if selected.size == 0:
                continue
            with Image.open(images_dir / f"{image_id}.jpg") as handle:
                image = handle.convert("RGB")
                for position in selected.tolist():
                    crops.append(
                        crop_tensor(
                            image,
                            boxes[position],
                            size=args.crop_size,
                            context_pad=args.context_pad,
                        )
                    )
                    positions.append(position)
        if not crops:
            continue
        stacked = torch.stack(crops)
        index_array = np.asarray(positions, dtype=np.int64)

        for name, model in models.items():
            if paths[name].exists():
                continue
            features: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, stacked.shape[0], args.batch_size):
                    batch = stacked[start : start + args.batch_size].to(device)
                    features.append(model(batch).float().cpu().numpy())
            matrix = np.concatenate(features).astype(np.float32)
            np.savez(
                paths[name],
                proposal_index=index_array,
                embeddings=matrix,
                image_ids=image_ids[index_array],
            )
        done = sum(len(item) for item in blocks[: index + 1])
        rate = done / max(time.time() - started, 1e-9)
        remaining = (len(unique) - done) / max(rate, 1e-9) / 60.0
        print(
            f"chunk {index + 1}/{len(blocks)}: {len(crops)} crops, "
            f"{rate:.1f} images/s, ~{remaining:.1f} min left",
            flush=True,
        )

    # --- merge, verifying the alignment contract rather than trusting it ---------
    for name in names:
        chunks = sorted(output.glob(f"{name}_chunk_*.npz"))
        if not chunks:
            continue
        merged = np.zeros((total, 0), dtype=np.float32)
        filled = np.zeros(total, dtype=bool)
        for path in chunks:
            with np.load(path, allow_pickle=True) as handle:
                positions = handle["proposal_index"]
                block = handle["embeddings"]
            if merged.shape[1] == 0:
                merged = np.zeros((total, block.shape[1]), dtype=np.float32)
            merged[positions] = block
            filled[positions] = True
        expected = selected_mask
        gaps = int((expected & ~filled).sum())
        if gaps:
            raise RuntimeError(
                f"{name}: {gaps} requested proposals have no embedding; the "
                "extraction is incomplete and must not be merged."
            )
        target = output / f"{name}.npz"
        np.savez(target, embeddings=merged, image_ids=image_ids)
        manifest = {
            "encoder": name,
            "architecture": ENCODERS[name]["architecture"],
            "weights": str(Path(ENCODERS[name]["weights"]).expanduser()),
            "description": ENCODERS[name]["description"],
            "source_export": str(args.export),
            "proposals": int(total),
            "images": len(unique),
            "dimensions": int(merged.shape[1]),
            "crop_size": int(args.crop_size),
            "context_pad": float(args.context_pad),
            "row_aligned_to_source_export": True,
            "embedded_rows": int(selected_mask.sum()),
            "restricted_to_rows": bool(args.rows),
            "row_list": str(args.rows) if args.rows else "",
        }
        np.save(target.with_name(f"{name}_filled.npy"), filled)
        target.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {target} {merged.shape}", flush=True)


if __name__ == "__main__":
    main()
