"""Prepare original-resolution frames and task masks for direct image editing."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from .local_reconstruction_adapters import _sha256_file
from .website_task_masks import decode_track_mask


def prepare_object_removal_frames(*, frames: Sequence[Mapping[str, Any]], task_masks: Mapping[str, Any],
                                  output_root: Path) -> list[dict[str, Any]]:
    """Keep original pixels; only the editor supplies the newly exposed background."""
    output_root.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, frame in enumerate(frames):
        source = Path(frame["source_image_path"])
        if _sha256_file(source) != frame["source_image_digest"]:
            raise ValueError("website_object_removal_source_changed")
        rotation = float(frame["display_rotation_degrees"])
        if not np.isfinite(rotation) or rotation % 90:
            raise ValueError("website_source_rotation_not_supported")
        with Image.open(source) as image:
            original = image.convert("RGB").rotate(rotation, expand=True)
        mask = np.zeros((original.height, original.width), dtype=bool)
        for target in task_masks["targets"]:
            if target["task_effect"] != "manipulated" or target["disposition"] != "remove":
                continue
            track = target.get("source_track") or target["track"]
            for observation in track["observations"]:
                if observation["source_frame_id"] != frame["frame_id"]:
                    continue
                source_mask = decode_track_mask(observation)
                if target.get("source_track") and source_mask.shape != mask.shape:
                    raise ValueError("website_object_removal_source_mask_mismatch")
                mask |= np.asarray(Image.fromarray(source_mask).resize(original.size, Image.Resampling.NEAREST))
        # Cover motion-blurred object edges and leave a narrow background-only
        # transition for compositing. Scale the margin with the source resolution.
        edge_margin = max(2, round(min(original.size) * 0.01))
        feather_pixels = max(1, round(min(original.size) * 0.005))
        if mask.any():
            mask = binary_dilation(mask, iterations=edge_margin + feather_pixels)
        image_path, mask_path = output_root / f"{index:06d}.png", output_root / f"{index:06d}.mask.png"
        original.save(image_path)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        digest = _sha256_file(image_path)
        prepared.append({"frame_id": frame["frame_id"], "image_path": str(image_path), "image_digest": digest,
                         "original_image_path": str(image_path), "original_image_digest": digest,
                         "original_source_digest": frame["source_image_digest"],
                         "remaining_mask_path": str(mask_path), "remaining_mask_digest": _sha256_file(mask_path),
                         "edge_feather_pixels": feather_pixels,
                         "remaining_pixel_count": int(mask.sum()), "generated_pixels_present": False,
                         "metric_measurement_proven": False})
    return prepared
