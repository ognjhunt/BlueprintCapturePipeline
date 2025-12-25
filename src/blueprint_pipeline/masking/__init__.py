"""Dynamic object masking for 3D reconstruction.

This package provides tools to detect and mask dynamic objects (people, cars, etc.)
from video frames before SLAM processing.
"""

from .dynamic_mask import (
    DynamicMaskGenerator,
    MaskConfig,
    generate_dynamic_masks,
)

__all__ = [
    "DynamicMaskGenerator",
    "MaskConfig",
    "generate_dynamic_masks",
]
