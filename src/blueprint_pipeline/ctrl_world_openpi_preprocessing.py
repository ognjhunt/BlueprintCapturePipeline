"""Exact released Ctrl-World-to-OpenPI image preprocessing seam."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CTRL_WORLD_ROLLOUT_SOURCE_SHA256 = (
    "cb270ebefdbe102ee78d3f6b1b0947345388d9d64a2403205515ffba66b38663"
)
OPENPI_IMAGE_TOOLS_SOURCE_REVISION = (
    "15a9616a00943ada6c20a0f158e3adb39df2ccac"
)


def preprocess_ctrl_world_frame_for_openpi_policy(value: str | Path | Any) -> np.ndarray:
    """Apply the public 192x320 -> 180x320 -> padded 224x224 path.

    Imports stay lazy because the exact implementation belongs in the pinned
    OpenPI/Ctrl-World GPU environment.  A missing dependency is an admission
    failure, never a reason to silently choose a different interpolation path.
    """

    if isinstance(value, (str, Path)):
        with Image.open(Path(value).expanduser().resolve()) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(value)
    if array.shape != (192, 320, 3) or array.dtype != np.uint8:
        raise ValueError("ctrl_world_policy_source_frame_must_be_uint8_192x320x3")
    try:
        import torch
        import torch.nn.functional as functional
        from openpi_client import image_tools
    except ImportError as exc:  # pragma: no cover - exercised in pinned GPU runtime
        raise RuntimeError("ctrl_world_openpi_preprocessing_dependencies_missing") from exc
    tensor = torch.from_numpy(array.copy()).to(torch.uint8)
    resized = functional.interpolate(
        tensor.permute(2, 0, 1).unsqueeze(0).float(),
        size=(180, 320),
        mode="bilinear",
        align_corners=False,
    )
    resized_array = (
        resized.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    )
    result = np.asarray(image_tools.resize_with_pad(resized_array, 224, 224))
    if result.shape != (224, 224, 3) or result.dtype != np.uint8:
        raise ValueError("ctrl_world_openpi_policy_preprocessing_output_invalid")
    return result


__all__ = [
    "CTRL_WORLD_ROLLOUT_SOURCE_SHA256",
    "OPENPI_IMAGE_TOOLS_SOURCE_REVISION",
    "preprocess_ctrl_world_frame_for_openpi_policy",
]
