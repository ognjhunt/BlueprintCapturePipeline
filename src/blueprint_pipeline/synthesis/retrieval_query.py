"""K-NN retrieval from the site reference index.

Two retrieval modes:
  spatial   — nearest by Euclidean camera-centre distance in site frame (requires aligned index)
  embedding — nearest by DINOv3 cosine similarity (works cross-session before alignment)
  hybrid    — spatial distance re-ranked by embedding similarity

SWM uses K=5 during training and K=1 at inference. For Blueprint depth-splat synthesis,
K=1 (nearest frame) is the primary path; K=3 is available for multi-frame blending.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def query_site(
    *,
    site_index_path: Path,
    target_T_world_camera: np.ndarray,   # 4x4, in site frame (needs aligned index)
    k: int = 5,
    mode: str = "spatial",               # "spatial" | "embedding" | "hybrid"
    query_embedding: Optional[np.ndarray] = None,  # [1024] float32, normalised
    storage_root: Optional[Path] = None, # needed for embedding mode
    bucket: Optional[str] = None,
    max_distance_m: Optional[float] = None,  # cap spatial search radius
) -> List[Dict[str, Any]]:
    """
    Return up to K reference records from the site index, ordered best-first.

    spatial:   ranks by ||t_ref_site - t_target||₂ (camera-centre distance in site frame)
    embedding: ranks by DINOv3 cosine similarity; requires query_embedding
    hybrid:    spatial distance × (1 - cosine_sim) combined score; requires both

    Records must have site_frame_transform populated for spatial/hybrid mode.
    Falls back to per-session spatial search if the index is not fully aligned.
    """
    records = _load_site_index(site_index_path)
    if not records:
        return []

    t_target = _translation(target_T_world_camera)

    if mode == "spatial":
        return _query_spatial(
            records=records,
            t_target=t_target,
            k=k,
            max_distance_m=max_distance_m,
        )
    if mode == "embedding":
        if query_embedding is None:
            raise ValueError("query_embedding is required for embedding mode")
        return _query_embedding(
            records=records,
            query_embedding=query_embedding,
            storage_root=storage_root,
            bucket=bucket,
            k=k,
        )
    if mode == "hybrid":
        if query_embedding is None:
            raise ValueError("query_embedding is required for hybrid mode")
        return _query_hybrid(
            records=records,
            t_target=t_target,
            query_embedding=query_embedding,
            storage_root=storage_root,
            bucket=bucket,
            k=k,
            max_distance_m=max_distance_m,
        )
    raise ValueError(f"Unknown query mode: {mode}")


# ---------------------------------------------------------------------------
# Spatial retrieval
# ---------------------------------------------------------------------------


def _query_spatial(
    *,
    records: List[Dict[str, Any]],
    t_target: np.ndarray,   # [3]
    k: int,
    max_distance_m: Optional[float],
) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for rec in records:
        T_ref = _effective_pose(rec)
        if T_ref is None:
            continue
        dist = float(np.linalg.norm(_translation(T_ref) - t_target))
        if max_distance_m is not None and dist > max_distance_m:
            continue
        scored.append((dist, rec))
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:k]]


# ---------------------------------------------------------------------------
# Embedding retrieval
# ---------------------------------------------------------------------------


def _query_embedding(
    *,
    records: List[Dict[str, Any]],
    query_embedding: np.ndarray,  # [1024] L2-normalised
    storage_root: Optional[Path],
    bucket: Optional[str],
    k: int,
) -> List[Dict[str, Any]]:
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for rec in records:
        emb = _load_embedding(rec, storage_root=storage_root, bucket=bucket)
        if emb is None:
            continue
        sim = float(np.dot(query_norm, emb))
        scored.append((-sim, rec))  # negate for ascending sort
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:k]]


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------


def _query_hybrid(
    *,
    records: List[Dict[str, Any]],
    t_target: np.ndarray,
    query_embedding: np.ndarray,
    storage_root: Optional[Path],
    bucket: Optional[str],
    k: int,
    max_distance_m: Optional[float],
    spatial_weight: float = 0.6,
    embedding_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Combined score = spatial_weight * normalised_distance + embedding_weight * (1 - sim).
    Lower is better.
    """
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    candidates: List[Tuple[float, float, Dict[str, Any]]] = []

    for rec in records:
        T_ref = _effective_pose(rec)
        if T_ref is None:
            continue
        dist = float(np.linalg.norm(_translation(T_ref) - t_target))
        if max_distance_m is not None and dist > max_distance_m:
            continue
        emb = _load_embedding(rec, storage_root=storage_root, bucket=bucket)
        sim = float(np.dot(query_norm, emb)) if emb is not None else 0.0
        candidates.append((dist, 1.0 - sim, rec))

    if not candidates:
        return []

    # Normalise distances into [0, 1]
    max_dist = max(d for d, _, _ in candidates) or 1.0
    scored = [
        (spatial_weight * (d / max_dist) + embedding_weight * ds, rec)
        for d, ds, rec in candidates
    ]
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:k]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _effective_pose(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Return the record's pose in site frame.
    If site_frame_transform is set: T_site = site_frame_transform @ T_world_camera_session.
    If not set (unaligned): T_world_camera_session (per-session frame, not comparable cross-session).
    """
    T_raw = rec.get("T_world_camera")
    if T_raw is None:
        return None
    T = np.array(T_raw, dtype=np.float64)
    if T.shape != (4, 4):
        return None

    site_xform = rec.get("site_frame_transform")
    if site_xform is not None:
        T_site = np.array(site_xform, dtype=np.float64)
        if T_site.shape == (4, 4):
            return T_site @ T
    return T


def _translation(T: np.ndarray) -> np.ndarray:
    return T[:3, 3]


def _load_embedding(
    rec: Dict[str, Any],
    *,
    storage_root: Optional[Path],
    bucket: Optional[str],
) -> Optional[np.ndarray]:
    uri = rec.get("embedding_uri")
    if not uri:
        return None
    local = _uri_to_local(uri, storage_root=storage_root, bucket=bucket)
    if local is None or not local.is_file():
        return None
    try:
        vec = np.fromfile(str(local), dtype=np.float32)
        if vec.shape[0] < 64:
            return None
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)
    except Exception:
        return None


def _load_site_index(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _uri_to_local(
    uri: str,
    *,
    storage_root: Optional[Path],
    bucket: Optional[str],
) -> Optional[Path]:
    if not uri.startswith("gs://"):
        return Path(uri)
    remainder = uri[5:]
    bkt, _, key = remainder.partition("/")
    if storage_root is None:
        return None
    candidate = storage_root / bkt / key
    if candidate.is_file():
        return candidate
    flat = storage_root / key
    if flat.is_file():
        return flat
    return candidate
