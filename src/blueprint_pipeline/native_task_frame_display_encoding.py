"""Display-encode linear HDR radiance for retained camera frames.

Isaac's own LDR annotator hard-clips linear radiance per channel: the
scene-839873 franka-controls run retained HDR buffers where 17% of samples
exceeded 1.0 (p99 = 11, max = 55), and the per-channel clip rendered them as
white blobs with chromatic fringes where channels saturated at different
points. The rolloff here is driven by the max channel so a saturated pixel
keeps its channel ratios exactly, and content at or below the knee keeps its
plain sRGB encoding. The linear HDR ``.npy`` retained beside the PNG stays
the raw truth either way.
"""

from __future__ import annotations

from typing import Any

DISPLAY_ENCODE_KNEE = 0.9
DISPLAY_ENCODE_SHOULDER = 4.0


def display_encode_hdr(hdr: Any) -> Any:
    """Encode linear HDR radiance to a display uint8 frame without clip fringes."""

    import numpy as np

    linear = np.nan_to_num(
        np.asarray(hdr, dtype=np.float64),
        nan=0.0,
        posinf=65504.0,
        neginf=0.0,
    )
    linear = np.clip(linear, 0.0, None)
    peak = linear.max(axis=-1)
    compressed = np.where(
        peak <= DISPLAY_ENCODE_KNEE,
        peak,
        DISPLAY_ENCODE_KNEE
        + (1.0 - DISPLAY_ENCODE_KNEE)
        * (1.0 - np.exp(-(peak - DISPLAY_ENCODE_KNEE) / DISPLAY_ENCODE_SHOULDER)),
    )
    scale = np.where(peak > 0.0, compressed / np.maximum(peak, 1.0e-12), 0.0)
    mapped = np.clip(linear * scale[..., None], 0.0, 1.0)
    srgb = np.where(
        mapped <= 0.0031308,
        12.92 * mapped,
        1.055 * np.power(mapped, 1.0 / 2.4) - 0.055,
    )
    return (np.clip(srgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


__all__ = [
    "DISPLAY_ENCODE_KNEE",
    "DISPLAY_ENCODE_SHOULDER",
    "display_encode_hdr",
]
