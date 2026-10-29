"""Run-length track masks, decoded without the segmentation runtime.

Kept apart from ``website_task_masks`` so CPU-only readers (preparation status,
the site's object question) can decode a mask without importing the SAM and
GPU provider modules that the mask producer needs.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def decode_track_mask(observation: Mapping[str, Any]) -> np.ndarray:
    height, width = int(observation["height"]), int(observation["width"])
    if height <= 0 or width <= 0 or height * width > 100_000_000:
        raise ValueError("website_track_mask_dimensions_invalid")
    mask = np.zeros(height * width, dtype=bool)
    for run in observation["runs"]:
        start, length = int(run["start"]), int(run["length"])
        if start < 0 or length <= 0 or start + length > mask.size:
            raise ValueError("website_track_mask_run_invalid")
        mask[start:start + length] = True
    return mask.reshape(height, width)
