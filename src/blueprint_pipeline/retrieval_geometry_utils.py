"""Geometry and storage-URI helpers for retrieval-memory artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .local_capture import LocalCaptureContext
from .site_memory_utils import p95 as _site_memory_p95


def mat_tx(transform: Any) -> float:
    try:
        return float(transform[0][3])
    except (TypeError, IndexError, KeyError):
        return 0.0


def mat_ty(transform: Any) -> float:
    try:
        return float(transform[1][3])
    except (TypeError, IndexError, KeyError):
        return 0.0


def mat_tz(transform: Any) -> float:
    try:
        return float(transform[2][3])
    except (TypeError, IndexError, KeyError):
        return 0.0


def euclidean(
    a: Tuple[float, float, float],
    b: Optional[Tuple[float, float, float]],
) -> float:
    if b is None:
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def p95(values: List[float]) -> float:
    return _site_memory_p95(values)


def local_to_gs_uri(path: Path, ctx: LocalCaptureContext) -> Optional[str]:
    try:
        rel = path.relative_to(ctx.storage_root / ctx.bucket)
        return f"gs://{ctx.bucket}/{rel.as_posix()}"
    except ValueError:
        return None


def arkit_depth_uri(frame_id: str, ctx: LocalCaptureContext) -> Optional[str]:
    path = ctx.raw_root / "arkit" / "depth" / f"{frame_id}.png"
    return local_to_gs_uri(path, ctx) if path.is_file() else None


def arkit_confidence_uri(frame_id: str, ctx: LocalCaptureContext) -> Optional[str]:
    path = ctx.raw_root / "arkit" / "confidence" / f"{frame_id}.png"
    return local_to_gs_uri(path, ctx) if path.is_file() else None
