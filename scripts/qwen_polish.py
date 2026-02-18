#!/usr/bin/env python3
"""
Stage 7.5: Qwen Image Edit — Object Crop Polish + Multi-Angle Generation

Given SAM3D object crops, this script:
  1. Polishes each crop (removes background noise, sharpens, normalizes lighting)
     using Qwen-Image-Edit-2511 via a targeted edit prompt.
  2. Generates multi-angle canonical views (front/back/left/right) per object
     to use as stronger SAM3D seed priors in a second detection pass.

Usage:
    python3 qwen_polish.py \
        --crops-dir /data/output/sam3d_out/object_crops \
        --index /data/output/sam3d_out/object_point_cloud_index.json \
        --output-dir /data/output/sam3d_out/polished \
        [--model-path /opt/qwen-image-edit] \
        [--no-multi-angle]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image


QWEN_MODEL_PATH = os.getenv(
    "QWEN_IMAGE_EDIT_MODEL_PATH", "/opt/qwen-image-edit"
)

POLISH_PROMPT = (
    "Clean up this image of a {label}. Remove any background clutter, "
    "shadows, and noise. Keep only the {label} with a clean white background. "
    "Preserve the exact shape, color, and texture. Make the object sharp and well-lit."
)

MULTI_ANGLE_PROMPTS = {
    "front": "Show this {label} from the front, centered, white background, high quality.",
    "back":  "Show this {label} from the back, centered, white background, high quality.",
    "left":  "Show this {label} from the left side, centered, white background, high quality.",
    "right": "Show this {label} from the right side, centered, white background, high quality.",
}


def load_pipeline(model_path: str):
    from diffusers import QwenImageEditPlusPipeline
    print(f"[qwen-polish] Loading Qwen-Image-Edit from {model_path}...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")
    print("[qwen-polish] Model loaded.")
    return pipe


def get_best_crops_per_object(crops_dir: Path, index_path: Path):
    """
    Group crop files by object label, pick the best (largest area) representative.
    Also reads reference_crop_path from the index JSON if available.
    """
    # Parse index for object metadata
    objects = []
    if index_path and index_path.exists():
        with open(index_path) as f:
            data = json.load(f)
        objects = data.get("objects", [])

    # Group crop files by label prefix
    crop_files = sorted(crops_dir.glob("*.png")) + sorted(crops_dir.glob("*.jpg"))
    label_crops = defaultdict(list)
    for cf in crop_files:
        # Filename format: {label}_{frame:03d}_{instance:03d}.png
        # Use label prefix (everything before last two _NNN segments)
        parts = cf.stem.rsplit("_", 2)
        if len(parts) >= 3:
            label = parts[0]
        else:
            label = cf.stem
        label_crops[label].append(cf)

    # Pick best crop per label (largest file as proxy for largest visible area)
    best = {}
    for label, files in label_crops.items():
        best_file = max(files, key=lambda f: f.stat().st_size)
        best[label] = best_file

    # Override with reference_crop_path from index if present
    for obj in objects:
        ref = obj.get("reference_crop_path")
        if ref and Path(ref).exists():
            label_key = re.sub(r"[\s/]", "_", obj.get("label", "").lower())
            best[label_key] = Path(ref)

    return best, objects


def polish_crop(pipe, image: Image.Image, label: str) -> Image.Image:
    prompt = POLISH_PROMPT.format(label=label.replace("_", " "))
    print(f"[qwen-polish]   Polishing: {label}")
    result = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=20,
        guidance_scale=7.5,
    )
    return result.images[0]


def generate_angles(pipe, image: Image.Image, label: str, out_dir: Path):
    """Generate front/back/left/right views of the polished crop."""
    views = {}
    for angle, template in MULTI_ANGLE_PROMPTS.items():
        prompt = template.format(label=label.replace("_", " "))
        print(f"[qwen-polish]     Generating {angle} view for: {label}")
        result = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=25,
            guidance_scale=8.0,
        )
        out_path = out_dir / f"{label}_{angle}.png"
        result.images[0].save(out_path)
        views[angle] = str(out_path)
    return views


def main():
    parser = argparse.ArgumentParser(description="Qwen object crop polish + multi-angle generation")
    parser.add_argument("--crops-dir", required=True, help="SAM3D object_crops directory")
    parser.add_argument("--index", required=True, help="object_point_cloud_index.json path")
    parser.add_argument("--output-dir", required=True, help="Output directory for polished crops")
    parser.add_argument("--model-path", default=QWEN_MODEL_PATH)
    parser.add_argument("--no-multi-angle", action="store_true", help="Skip multi-angle generation")
    parser.add_argument("--only-labels", nargs="*", help="Only process these labels (debug)")
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    index_path = Path(args.index)
    out_dir = Path(args.output_dir)
    polished_dir = out_dir / "polished_crops"
    angles_dir = out_dir / "multi_angle_crops"
    polished_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_multi_angle:
        angles_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.model_path).exists():
        print(f"[qwen-polish] ERROR: Model not found at {args.model_path}")
        print("[qwen-polish] Set QWEN_IMAGE_EDIT_MODEL_PATH or pass --model-path")
        sys.exit(1)

    pipe = load_pipeline(args.model_path)

    best_crops, objects = get_best_crops_per_object(crops_dir, index_path)
    print(f"[qwen-polish] Found {len(best_crops)} unique object labels with crops")

    results = {}
    for label, crop_path in sorted(best_crops.items()):
        if args.only_labels and label not in args.only_labels:
            continue

        print(f"[qwen-polish] Processing: {label} (crop: {crop_path.name})")
        image = Image.open(crop_path).convert("RGB")

        # Stage 7.5a: Polish
        polished = polish_crop(pipe, image, label)
        polished_path = polished_dir / f"{label}_polished.png"
        polished.save(polished_path)

        results[label] = {
            "original_crop": str(crop_path),
            "polished_crop": str(polished_path),
            "multi_angle": {},
        }

        # Stage 7.5b: Multi-angle generation
        if not args.no_multi_angle:
            views = generate_angles(pipe, polished, label, angles_dir)
            results[label]["multi_angle"] = views
            print(f"[qwen-polish]   Generated {len(views)} angle views for {label}")

    # Write summary
    summary_path = out_dir / "qwen_polish_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[qwen-polish] Done. {len(results)} objects processed.")
    print(f"[qwen-polish] Summary: {summary_path}")
    print(f"[qwen-polish] Polished crops: {polished_dir}")
    if not args.no_multi_angle:
        print(f"[qwen-polish] Multi-angle views: {angles_dir}")
    print("\n[qwen-polish] Multi-angle crops can be used as SAM3D seed frames:")
    print("[qwen-polish]   sam3_detect.py --extra-seed-frames <angles_dir> ...")


if __name__ == "__main__":
    main()
