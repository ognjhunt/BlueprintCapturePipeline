"""Phase 3B: Cross-session ARKit coordinate frame alignment.

Each ARKit session generates its own world coordinate frame. T_world_camera from
capture A is NOT comparable to capture B of the same site. This stage computes a
canonical site frame by aligning all sessions to the first qualifying capture
(reference session) using DINOv3 visual place recognition + SE(3) RANSAC.

Once aligned, site_frame_transform on each site reference index record is populated
with the 4x4 SE(3) matrix that maps from that record's ARKit session frame into the
canonical site frame. After this, spatial nearest-neighbor retrieval works correctly
across all captures of the same site.

Algorithm:
  1. Group site_reference_index.jsonl records by coordinate_frame_session_id
  2. Designate the earliest session as the reference (site frame = reference world frame)
  3. For each non-reference session S_k:
     a. Load DINOv3 embeddings for S_k and the reference session
     b. Find candidate cross-session frame matches via cosine similarity NN
     c. RANSAC over candidate SE(3) transforms T_site_from_Sk = T_ref_i @ inv(T_Sk_j)
     d. Refine translation over RANSAC inliers
     e. Store result as site_frame_transform on all records in session S_k
  4. Patch site_reference_index.jsonl in-place
  5. Rewrite site_reference_manifest.json with site_frame_established = true
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .common import (
    PipelineError,
    ensure_dir,
    parse_bool,
    read_json,
    utc_now_iso,
    write_json,
)
from .local_capture import LocalCaptureContext, resolve_local_capture_context


# ---------------------------------------------------------------------------
# Alignment thresholds
# ---------------------------------------------------------------------------

# Visual similarity: DINOv3 cosine similarity required for a frame pair to be a candidate match
_SIM_THRESHOLD = 0.75

# RANSAC inlier thresholds
_INLIER_TRANSLATION_M = 0.5   # max positional error (metres) for a pair to be an inlier
_INLIER_ROTATION_DEG = 20.0   # max angular error (degrees) for a pair to be an inlier

# Minimum candidate matches to attempt RANSAC
_MIN_CANDIDATES = 8

# Minimum RANSAC inlier fraction to accept alignment
_MIN_INLIER_FRACTION = 0.35

# RANSAC iterations
_RANSAC_ITERS = 500

# DINOv3 embedding dimension
_EMBED_DIM = 1024


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_frame_alignment_stage(
    *,
    capture_root: str | Path,
    force_realign: bool = False,
) -> Dict[str, Any]:
    """
    Align all ARKit sessions for the site associated with this capture to a common
    coordinate frame. Patches site_frame_transform on every site_reference_index record
    that can be aligned.

    Idempotent: skips sessions that are already aligned unless force_realign=True.
    """
    ctx = resolve_local_capture_context(capture_root)
    descriptor = _load_descriptor(ctx)
    site_id = _resolve_site_id(descriptor)

    if not site_id:
        return {"status": "skipped", "reason": "no_site_id", "capture_id": ctx.capture_id}

    site_root = ctx.storage_root / ctx.bucket / "sites" / site_id / "reference_memory"
    site_index_path = site_root / "site_reference_index.jsonl"

    if not site_index_path.is_file():
        return {"status": "skipped", "reason": "no_site_index", "site_id": site_id}

    records = _load_jsonl(site_index_path)
    if not records:
        return {"status": "skipped", "reason": "empty_site_index", "site_id": site_id}

    sessions = _group_by_session(records)
    if len(sessions) < 2:
        return {
            "status": "skipped",
            "reason": "single_session_no_alignment_needed",
            "site_id": site_id,
            "session_count": len(sessions),
        }

    # Reference session: earliest first-record timestamp
    ref_session_id = _pick_reference_session(sessions)
    ref_records = sessions[ref_session_id]

    results: List[Dict[str, Any]] = []
    aligned_sessions: List[str] = [ref_session_id]
    # Reference is trivially aligned (identity)
    transforms: Dict[str, Optional[List[List[float]]]] = {ref_session_id: _identity_4x4()}

    # Load reference embeddings once
    ref_embeddings, ref_record_index = _load_session_embeddings(ref_records, ctx)
    if ref_embeddings is None or len(ref_embeddings) == 0:
        return {
            "status": "failed",
            "reason": "reference_session_has_no_embeddings",
            "site_id": site_id,
            "reference_session": ref_session_id,
        }

    for session_id, session_records in sessions.items():
        if session_id == ref_session_id:
            continue

        # Skip if already aligned and not forcing realign
        if not force_realign and _session_already_aligned(session_records):
            transforms[session_id] = session_records[0].get("site_frame_transform")
            aligned_sessions.append(session_id)
            results.append({
                "session_id": session_id,
                "status": "skipped",
                "reason": "already_aligned",
            })
            continue

        session_result = _align_session(
            session_id=session_id,
            session_records=session_records,
            ref_session_id=ref_session_id,
            ref_records=ref_records,
            ref_embeddings=ref_embeddings,
            ref_record_index=ref_record_index,
            ctx=ctx,
        )
        results.append(session_result)

        if session_result["status"] == "aligned":
            transforms[session_id] = session_result["site_frame_transform"]
            aligned_sessions.append(session_id)

    # Patch index records with computed transforms
    n_patched = _patch_site_index(
        site_index_path=site_index_path,
        transforms=transforms,
    )

    # Update site manifest
    _update_site_manifest_alignment(
        site_root=site_root,
        site_index_path=site_index_path,
        site_id=site_id,
        aligned_sessions=aligned_sessions,
    )

    return {
        "status": "completed",
        "site_id": site_id,
        "sessions_total": len(sessions),
        "sessions_aligned": len(aligned_sessions),
        "records_patched": n_patched,
        "session_results": results,
    }


# ---------------------------------------------------------------------------
# Session alignment
# ---------------------------------------------------------------------------


def _align_session(
    *,
    session_id: str,
    session_records: List[Dict[str, Any]],
    ref_session_id: str,
    ref_records: List[Dict[str, Any]],
    ref_embeddings: np.ndarray,
    ref_record_index: List[Dict[str, Any]],
    ctx: LocalCaptureContext,
) -> Dict[str, Any]:
    """Estimate T_site_from_session for a single non-reference session."""
    session_embeddings, session_record_index = _load_session_embeddings(session_records, ctx)

    if session_embeddings is None or len(session_embeddings) == 0:
        return {
            "session_id": session_id,
            "status": "failed",
            "reason": "no_embeddings",
        }

    candidates = _find_candidate_matches(
        query_embeddings=session_embeddings,
        query_record_index=session_record_index,
        ref_embeddings=ref_embeddings,
        ref_record_index=ref_record_index,
        sim_threshold=_SIM_THRESHOLD,
    )

    if len(candidates) < _MIN_CANDIDATES:
        return {
            "session_id": session_id,
            "status": "failed",
            "reason": "insufficient_matches",
            "candidate_count": len(candidates),
        }

    T_site_from_session, inlier_count, inlier_fraction = _ransac_se3(
        candidates=candidates,
        n_iters=_RANSAC_ITERS,
        translation_threshold_m=_INLIER_TRANSLATION_M,
        rotation_threshold_deg=_INLIER_ROTATION_DEG,
    )

    if T_site_from_session is None or inlier_fraction < _MIN_INLIER_FRACTION:
        return {
            "session_id": session_id,
            "status": "failed",
            "reason": "ransac_failed",
            "candidate_count": len(candidates),
            "inlier_count": inlier_count,
            "inlier_fraction": round(inlier_fraction, 3),
        }

    # Refine translation estimate over all inliers
    T_site_from_session = _refine_translation(
        T_site_from_session=T_site_from_session,
        candidates=candidates,
        translation_threshold_m=_INLIER_TRANSLATION_M,
        rotation_threshold_deg=_INLIER_ROTATION_DEG,
    )

    T_list = T_site_from_session.tolist()
    return {
        "session_id": session_id,
        "status": "aligned",
        "reference_session": ref_session_id,
        "site_frame_transform": T_list,
        "candidate_count": len(candidates),
        "inlier_count": inlier_count,
        "inlier_fraction": round(inlier_fraction, 3),
    }


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------


def _load_session_embeddings(
    records: List[Dict[str, Any]],
    ctx: LocalCaptureContext,
) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
    """
    Load DINOv3 embeddings for all records in a session that have embedding_uri.
    Returns (embeddings [N, 1024], filtered_records).
    Embeddings are L2-normalised for cosine similarity via dot product.
    """
    filtered: List[Dict[str, Any]] = []
    vecs: List[np.ndarray] = []

    for rec in records:
        emb_uri = rec.get("embedding_uri")
        if not emb_uri:
            continue
        local_path = _uri_to_local(emb_uri, ctx)
        if local_path is None or not local_path.is_file():
            continue
        try:
            vec = np.fromfile(str(local_path), dtype=np.float32)
            if vec.shape[0] != _EMBED_DIM:
                continue
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                continue
            vecs.append(vec / norm)
            filtered.append(rec)
        except Exception:
            continue

    if not vecs:
        return None, []

    return np.stack(vecs, axis=0), filtered  # [N, 1024]


# ---------------------------------------------------------------------------
# Visual place recognition: cosine similarity NN
# ---------------------------------------------------------------------------


def _find_candidate_matches(
    *,
    query_embeddings: np.ndarray,    # [N_q, D]
    query_record_index: List[Dict[str, Any]],
    ref_embeddings: np.ndarray,      # [N_r, D]
    ref_record_index: List[Dict[str, Any]],
    sim_threshold: float,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """
    Find candidate frame pairs between a query session and the reference session.
    Returns list of (query_record, ref_record, similarity) tuples above threshold,
    one per query frame (best match only).
    """
    # Cosine similarity matrix: query × ref (both already L2-normalised)
    sims = query_embeddings @ ref_embeddings.T  # [N_q, N_r]

    candidates: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
    for q_idx in range(len(query_record_index)):
        row = sims[q_idx]
        best_r_idx = int(np.argmax(row))
        best_sim = float(row[best_r_idx])
        if best_sim >= sim_threshold:
            q_rec = query_record_index[q_idx]
            r_rec = ref_record_index[best_r_idx]
            # Only keep pairs that have valid T_world_camera on both sides
            if _has_valid_pose(q_rec) and _has_valid_pose(r_rec):
                candidates.append((q_rec, r_rec, best_sim))

    # Also run ref→query direction and merge (mutual NN promotes reliability)
    sims_T = sims.T  # [N_r, N_q]
    for r_idx in range(len(ref_record_index)):
        row = sims_T[r_idx]
        best_q_idx = int(np.argmax(row))
        best_sim = float(row[best_q_idx])
        if best_sim >= sim_threshold:
            r_rec = ref_record_index[r_idx]
            q_rec = query_record_index[best_q_idx]
            if _has_valid_pose(q_rec) and _has_valid_pose(r_rec):
                pair = (q_rec, r_rec, best_sim)
                # Dedup by query frame_id
                if not any(c[0].get("frame_id") == q_rec.get("frame_id") for c in candidates):
                    candidates.append(pair)

    # Sort descending by similarity so RANSAC samples the best pairs first
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def _has_valid_pose(rec: Dict[str, Any]) -> bool:
    T = rec.get("T_world_camera")
    if not isinstance(T, list) or len(T) < 4:
        return False
    return True


# ---------------------------------------------------------------------------
# RANSAC SE(3) estimation
# ---------------------------------------------------------------------------


def _ransac_se3(
    *,
    candidates: List[Tuple[Dict[str, Any], Dict[str, Any], float]],
    n_iters: int,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> Tuple[Optional[np.ndarray], int, float]:
    """
    RANSAC over candidate frame pair transforms.

    Each pair (q_rec, r_rec) gives a candidate:
        T_site_from_session = T_world_ref @ inv(T_world_session)
    where T_world_ref is the reference session pose and T_world_session is the
    query session pose (both in their respective session frames).

    Returns (T_site_from_session [4,4], inlier_count, inlier_fraction).
    If RANSAC fails, returns (None, 0, 0.0).
    """
    n = len(candidates)
    best_inliers = 0
    best_T: Optional[np.ndarray] = None
    cos_thresh = math.cos(math.radians(rotation_threshold_deg))

    # Pre-compute poses
    q_poses: List[np.ndarray] = [_pose_matrix(c[0]) for c in candidates]
    r_poses: List[np.ndarray] = [_pose_matrix(c[1]) for c in candidates]

    for _ in range(n_iters):
        idx = random.randint(0, n - 1)
        T_q = q_poses[idx]   # session frame pose
        T_r = r_poses[idx]   # reference (site) frame pose

        # Candidate alignment: maps session frame → site frame
        # T_r ≈ T_candidate @ T_q  →  T_candidate = T_r @ inv(T_q)
        try:
            T_q_inv = _mat_inv(T_q)
        except Exception:
            continue
        T_candidate = T_r @ T_q_inv

        # Count inliers
        inliers = 0
        for j in range(n):
            # Apply candidate transform to session pose; compare to ref pose
            T_q_j = q_poses[j]
            T_r_j = r_poses[j]
            T_predicted = T_candidate @ T_q_j

            # Translation residual
            dt = np.linalg.norm(T_predicted[:3, 3] - T_r_j[:3, 3])
            if dt > translation_threshold_m:
                continue

            # Rotation residual: trace(R_pred.T @ R_ref) ≥ 1 + 2*cos(theta) for small theta
            # cos(theta) ≥ cos_thresh  →  (trace - 1) / 2 ≥ cos_thresh
            R_pred = T_predicted[:3, :3]
            R_ref = T_r_j[:3, :3]
            cos_angle = (np.trace(R_pred.T @ R_ref) - 1.0) / 2.0
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            if cos_angle < cos_thresh:
                continue

            inliers += 1

        if inliers > best_inliers:
            best_inliers = inliers
            best_T = T_candidate.copy()

    if best_T is None or best_inliers == 0:
        return None, 0, 0.0

    inlier_fraction = best_inliers / n
    return best_T, best_inliers, inlier_fraction


def _refine_translation(
    *,
    T_site_from_session: np.ndarray,
    candidates: List[Tuple[Dict[str, Any], Dict[str, Any], float]],
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> np.ndarray:
    """
    Refine the translation of T_site_from_session by averaging over inliers.
    Rotation is taken from RANSAC result (it's harder to average and RANSAC is good enough).
    """
    cos_thresh = math.cos(math.radians(rotation_threshold_deg))
    R_align = T_site_from_session[:3, :3]

    translation_residuals: List[np.ndarray] = []
    for q_rec, r_rec, _ in candidates:
        T_q = _pose_matrix(q_rec)
        T_r = _pose_matrix(r_rec)
        T_predicted = T_site_from_session @ T_q
        dt = np.linalg.norm(T_predicted[:3, 3] - T_r[:3, 3])
        if dt > translation_threshold_m:
            continue
        cos_angle = float(np.clip((np.trace(T_predicted[:3, :3].T @ T_r[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
        if cos_angle < cos_thresh:
            continue
        # Residual: what translation would perfectly explain this pair?
        # T_r[:3,3] = R_align @ T_q[:3,3] + t_align  →  t_align = T_r[:3,3] - R_align @ T_q[:3,3]
        t_implied = T_r[:3, 3] - R_align @ T_q[:3, 3]
        translation_residuals.append(t_implied)

    if len(translation_residuals) >= 3:
        t_refined = np.mean(np.stack(translation_residuals, axis=0), axis=0)
        T_refined = T_site_from_session.copy()
        T_refined[:3, 3] = t_refined
        return T_refined

    return T_site_from_session


# ---------------------------------------------------------------------------
# Site index patching
# ---------------------------------------------------------------------------


def _patch_site_index(
    *,
    site_index_path: Path,
    transforms: Dict[str, Optional[List[List[float]]]],
) -> int:
    """
    Rewrite site_reference_index.jsonl with site_frame_transform populated for
    sessions that were successfully aligned. Returns count of patched records.
    """
    records = _load_jsonl(site_index_path)
    patched = 0
    out_lines: List[str] = []
    for rec in records:
        session_id = rec.get("coordinate_frame_session_id", "")
        if session_id in transforms and transforms[session_id] is not None:
            rec = dict(rec)
            rec["site_frame_transform"] = transforms[session_id]
            patched += 1
        out_lines.append(json.dumps(rec, separators=(",", ":")))

    site_index_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return patched


def _update_site_manifest_alignment(
    *,
    site_root: Path,
    site_index_path: Path,
    site_id: str,
    aligned_sessions: List[str],
) -> None:
    """Update site_reference_manifest.json to reflect alignment status."""
    manifest_path = site_root / "site_reference_manifest.json"
    if not manifest_path.is_file():
        return

    manifest = read_json(manifest_path)
    manifest = dict(manifest)
    captures = manifest.get("captures") or []
    for cap in captures:
        session = cap.get("coordinate_frame_session_id", "")
        cap["site_frame_aligned"] = session in aligned_sessions

    manifest["site_frame_established"] = len(aligned_sessions) > 1
    manifest["aligned_session_count"] = len(aligned_sessions)
    manifest["last_updated"] = utc_now_iso()
    write_json(manifest_path, manifest)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _pose_matrix(rec: Dict[str, Any]) -> np.ndarray:
    """Extract 4x4 SE(3) pose matrix from a site index record."""
    T = rec.get("T_world_camera")
    if T is None:
        raise ValueError(f"Record has no T_world_camera: {rec.get('frame_id')}")
    arr = np.array(T, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"T_world_camera is not 4x4: shape {arr.shape}")
    return arr


def _mat_inv(T: np.ndarray) -> np.ndarray:
    """Efficient SE(3) inverse: [R|t]^-1 = [R^T | -R^T t]"""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ t)
    return T_inv


def _identity_4x4() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


# ---------------------------------------------------------------------------
# Session grouping and selection
# ---------------------------------------------------------------------------


def _group_by_session(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    sessions: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        sid = rec.get("coordinate_frame_session_id", "")
        sessions.setdefault(sid, []).append(rec)
    return sessions


def _pick_reference_session(sessions: Dict[str, List[Dict[str, Any]]]) -> str:
    """Reference session = session whose first record has the earliest captured_at."""
    def _first_ts(recs: List[Dict[str, Any]]) -> str:
        return min((r.get("captured_at") or "9999") for r in recs)
    return min(sessions.keys(), key=lambda sid: _first_ts(sessions[sid]))


def _session_already_aligned(records: List[Dict[str, Any]]) -> bool:
    return any(r.get("site_frame_transform") is not None for r in records)


# ---------------------------------------------------------------------------
# Descriptor helpers
# ---------------------------------------------------------------------------


def _load_descriptor(ctx: LocalCaptureContext) -> Dict[str, Any]:
    if not ctx.descriptor_path.is_file():
        raise PipelineError(f"capture_descriptor.json not found: {ctx.descriptor_path}")
    return read_json(ctx.descriptor_path)


def _resolve_site_id(descriptor: Dict[str, Any]) -> Optional[str]:
    meta = descriptor.get("metadata") or {}
    site_identity = meta.get("site_identity") or {}
    return site_identity.get("site_id") or descriptor.get("site_id") or None


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def _uri_to_local(uri: str, ctx: LocalCaptureContext) -> Optional[Path]:
    """Resolve a gs:// URI to a local path via the storage root."""
    if not uri.startswith("gs://"):
        return Path(uri)
    remainder = uri[5:]
    bucket, _, key = remainder.partition("/")
    if not bucket or not key:
        return None
    # Try bucket-prefixed path first, then flat
    candidate_bucket = ctx.storage_root / bucket / key
    candidate_flat = ctx.storage_root / key
    if candidate_bucket.is_file():
        return candidate_bucket
    if candidate_flat.is_file():
        return candidate_flat
    # Best-effort: return bucket path even if it doesn't exist yet
    return candidate_bucket
