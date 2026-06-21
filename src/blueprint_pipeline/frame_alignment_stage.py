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
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .common import (
    PipelineError,
    read_json,
    utc_now_iso,
    write_json,
)
from .local_capture import LocalCaptureContext, resolve_local_capture_context
from .retrieval_index_stage import (
    _write_retrieval_validation,
    _write_site_manifest,
    _write_site_memory_indices,
    _write_site_reference_summary_projection,
    _update_coverage_map,
)
from .site_memory_utils import (
    backproject_depth_points,
    effective_pose,
    fingerprint_similarity,
    gs_uri_to_local,
    load_embedding,
    load_jsonl as _sm_load_jsonl,
    load_numeric_array as _sm_load_numeric_array,
    p95 as _sm_p95,
    plane_summaries,
    pose_matrix,
    rotation_cosine,
    write_ascii_pointcloud,
)


# ---------------------------------------------------------------------------
# Alignment thresholds
# ---------------------------------------------------------------------------

# Visual similarity: DINOv3 cosine similarity required for a frame pair to be a candidate match
_SIM_THRESHOLD = 0.75
_PAIR_SCORE_THRESHOLD = 0.62

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

    ref_session_id = _pick_reference_session(sessions)

    results: List[Dict[str, Any]] = []
    aligned_sessions: List[str] = [ref_session_id]
    transforms: Dict[str, Optional[List[List[float]]]] = {ref_session_id: _identity_4x4()}
    alignment_edges: List[Dict[str, Any]] = []

    pending = {
        session_id: session_records
        for session_id, session_records in sessions.items()
        if session_id != ref_session_id
    }

    for session_id, session_records in list(pending.items()):
        if not force_realign and _session_already_aligned(session_records):
            transforms[session_id] = session_records[0].get("site_frame_transform")
            aligned_sessions.append(session_id)
            results.append(
                {
                    "session_id": session_id,
                    "status": "skipped",
                    "reason": "already_aligned",
                }
            )
            pending.pop(session_id, None)

    progress = True
    while pending and progress:
        progress = False
        for session_id in list(pending.keys()):
            session_records = pending[session_id]
            best_result: Optional[Dict[str, Any]] = None
            for anchor_session_id in list(aligned_sessions):
                anchor_records = sessions[anchor_session_id]
                session_result = _align_session(
                    session_id=session_id,
                    session_records=session_records,
                    anchor_session_id=anchor_session_id,
                    anchor_records=anchor_records,
                    anchor_transform=transforms.get(anchor_session_id),
                    ctx=ctx,
                )
                if best_result is None or float(session_result.get("combined_score") or 0.0) > float(best_result.get("combined_score") or 0.0):
                    best_result = session_result
            if best_result is None:  # pragma: no cover - defensive; aligned_sessions is seeded with the reference session.
                continue
            results.append(best_result)
            if best_result["status"] == "aligned":
                transforms[session_id] = best_result["site_frame_transform"]
                aligned_sessions.append(session_id)
                alignment_edges.append(
                    {
                        "from_session_id": session_id,
                        "to_session_id": str(best_result.get("reference_session") or ref_session_id),
                        "candidate_count": best_result.get("candidate_count"),
                        "inlier_count": best_result.get("inlier_count"),
                        "inlier_fraction": best_result.get("inlier_fraction"),
                        "combined_score": best_result.get("combined_score"),
                    }
                )
                pending.pop(session_id, None)
                progress = True
            elif best_result.get("reason") == "insufficient_matches":
                continue
            else:
                pending.pop(session_id, None)

    n_patched = _patch_site_index(site_index_path=site_index_path, transforms=transforms)
    _write_site_transforms_manifest(site_root=site_root, site_id=site_id, transforms=transforms, results=results)
    _write_alignment_validation(site_root=site_root, site_id=site_id, results=results, aligned_sessions=aligned_sessions)
    _update_overlap_graph_alignment(site_root=site_root, alignment_edges=alignment_edges)
    _write_static_memory_artifacts(site_root=site_root, site_index_path=site_index_path, site_id=site_id, storage_root=ctx.storage_root)
    _update_coverage_map(site_root=site_root, site_index_path=site_index_path, site_id=site_id)
    _write_site_manifest(site_root=site_root, site_index_path=site_index_path, site_id=site_id)
    _write_site_memory_indices(site_root=site_root, site_index_path=site_index_path, site_id=site_id, storage_root=ctx.storage_root)
    _write_retrieval_validation(site_root=site_root, site_index_path=site_index_path, site_id=site_id)
    _update_site_manifest_alignment(
        site_root=site_root,
        site_index_path=site_index_path,
        site_id=site_id,
        aligned_sessions=aligned_sessions,
    )
    _write_site_reference_summary_projection(
        site_root=site_root,
        site_index_path=site_index_path,
        site_id=site_id,
        storage_root=ctx.storage_root,
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
    anchor_session_id: str,
    anchor_records: List[Dict[str, Any]],
    anchor_transform: Optional[List[List[float]]],
    ctx: LocalCaptureContext,
) -> Dict[str, Any]:
    """Estimate T_site_from_session for a session against one already-aligned anchor session."""
    session_embeddings, session_record_index = _load_session_embeddings(session_records, ctx)
    anchor_embeddings, anchor_record_index = _load_session_embeddings(anchor_records, ctx)

    if session_embeddings is None or len(session_embeddings) == 0:
        return {
            "session_id": session_id,
            "status": "failed",
            "reason": "no_embeddings",
            "combined_score": 0.0,
        }
    if anchor_embeddings is None or len(anchor_embeddings) == 0:
        return {
            "session_id": session_id,
            "status": "failed",
            "reason": "anchor_session_has_no_embeddings",
            "reference_session": anchor_session_id,
            "combined_score": 0.0,
        }

    candidates = _find_candidate_matches(
        query_embeddings=session_embeddings,
        query_record_index=session_record_index,
        ref_embeddings=anchor_embeddings,
        ref_record_index=anchor_record_index,
        sim_threshold=_SIM_THRESHOLD,
    )

    if len(candidates) < _MIN_CANDIDATES:
        return {
            "session_id": session_id,
            "status": "failed",
            "reason": "insufficient_matches",
            "reference_session": anchor_session_id,
            "candidate_count": len(candidates),
            "combined_score": round(float(np.mean([float(c[2]) for c in candidates]) if candidates else 0.0), 4),
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
            "reference_session": anchor_session_id,
            "candidate_count": len(candidates),
            "inlier_count": inlier_count,
            "inlier_fraction": round(inlier_fraction, 3),
            "combined_score": round(float(np.mean([float(c[2]) for c in candidates]) if candidates else 0.0), 4),
        }

    T_anchor_from_session = _refine_translation(
        T_site_from_session=T_site_from_session,
        candidates=candidates,
        translation_threshold_m=_INLIER_TRANSLATION_M,
        rotation_threshold_deg=_INLIER_ROTATION_DEG,
    )
    anchor_xform = np.array(anchor_transform or _identity_4x4(), dtype=np.float64)
    T_site_from_session = anchor_xform @ T_anchor_from_session

    residuals = _candidate_residuals(T_anchor_from_session=T_anchor_from_session, candidates=candidates)
    return {
        "session_id": session_id,
        "status": "aligned",
        "reference_session": anchor_session_id,
        "site_frame_transform": T_site_from_session.tolist(),
        "candidate_count": len(candidates),
        "inlier_count": inlier_count,
        "inlier_fraction": round(inlier_fraction, 3),
        "combined_score": round(float(np.mean([float(c[2]) for c in candidates]) if candidates else 0.0), 4),
        "residual_translation_p95_m": round(_sm_p95([item["translation_m"] for item in residuals]), 4) if residuals else 0.0,
        "residual_rotation_p95_deg": round(_sm_p95([item["rotation_deg"] for item in residuals]), 4) if residuals else 0.0,
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
        vec = load_embedding(embedding_uri=str(emb_uri), storage_root=ctx.storage_root, expected_dim=_EMBED_DIM)
        if vec is None:
            continue
        vecs.append(vec)
        filtered.append(rec)

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
    sims = query_embeddings @ ref_embeddings.T  # [N_q, N_r]

    candidates: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
    for q_idx in range(len(query_record_index)):
        row = sims[q_idx]
        best_r_idx = int(np.argmax(row))
        best_sim = float(row[best_r_idx])
        q_rec = query_record_index[q_idx]
        r_rec = ref_record_index[best_r_idx]
        pair_score = _candidate_pair_score(q_rec=q_rec, r_rec=r_rec, visual_sim=best_sim)
        if pair_score >= _PAIR_SCORE_THRESHOLD and best_sim >= sim_threshold and _has_valid_pose(q_rec) and _has_valid_pose(r_rec):
            candidates.append((q_rec, r_rec, pair_score))

    sims_T = sims.T  # [N_r, N_q]
    for r_idx in range(len(ref_record_index)):
        row = sims_T[r_idx]
        best_q_idx = int(np.argmax(row))
        best_sim = float(row[best_q_idx])
        r_rec = ref_record_index[r_idx]
        q_rec = query_record_index[best_q_idx]
        pair_score = _candidate_pair_score(q_rec=q_rec, r_rec=r_rec, visual_sim=best_sim)
        if pair_score >= _PAIR_SCORE_THRESHOLD and best_sim >= sim_threshold and _has_valid_pose(q_rec) and _has_valid_pose(r_rec):
            pair = (q_rec, r_rec, pair_score)
            if not any(c[0].get("frame_id") == q_rec.get("frame_id") for c in candidates):
                candidates.append(pair)

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def _candidate_pair_score(
    *,
    q_rec: Dict[str, Any],
    r_rec: Dict[str, Any],
    visual_sim: float,
) -> float:
    q_geometry = q_rec.get("geometry_fingerprint") if isinstance(q_rec.get("geometry_fingerprint"), dict) else {}
    r_geometry = r_rec.get("geometry_fingerprint") if isinstance(r_rec.get("geometry_fingerprint"), dict) else {}
    geometry_score = fingerprint_similarity(q_geometry, r_geometry) if q_geometry and r_geometry else 0.0
    q_anchors = set(str(item).strip() for item in (q_rec.get("anchor_observations") or []) if str(item).strip())
    r_anchors = set(str(item).strip() for item in (r_rec.get("anchor_observations") or []) if str(item).strip())
    topology_score = 0.0
    if q_anchors and r_anchors:
        topology_score += 0.7 if (q_anchors & r_anchors) else 0.0
    if q_rec.get("zone_id") and q_rec.get("zone_id") == r_rec.get("zone_id"):
        topology_score += 0.3
    staticness = float(q_rec.get("staticness_score") or 0.0) + float(r_rec.get("staticness_score") or 0.0)
    staticness_score = min(staticness / 2.0, 1.0)
    return round(
        (0.55 * float(np.clip(visual_sim, 0.0, 1.0)))
        + (0.20 * geometry_score)
        + (0.15 * min(topology_score, 1.0))
        + (0.10 * staticness_score),
        4,
    )


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


def _candidate_residuals(
    *,
    T_anchor_from_session: np.ndarray,
    candidates: List[Tuple[Dict[str, Any], Dict[str, Any], float]],
) -> List[Dict[str, float]]:
    residuals: List[Dict[str, float]] = []
    for q_rec, r_rec, _ in candidates:
        T_q = _pose_matrix(q_rec)
        T_r = _pose_matrix(r_rec)
        predicted = T_anchor_from_session @ T_q
        translation_m = float(np.linalg.norm(predicted[:3, 3] - T_r[:3, 3]))
        cos_angle = rotation_cosine(predicted, T_r)
        rotation_deg = math.degrees(math.acos(float(np.clip(cos_angle, -1.0, 1.0))))
        residuals.append(
            {
                "translation_m": translation_m,
                "rotation_deg": rotation_deg,
            }
        )
    return residuals


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
            T_raw = pose_matrix(rec.get("T_world_camera"))
            T_site = pose_matrix(transforms[session_id])
            if T_raw is not None and T_site is not None:
                rec["T_site_camera"] = (T_site @ T_raw).tolist()
            patched += 1
        out_lines.append(json.dumps(rec, separators=(",", ":")))

    site_index_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return patched


def _write_site_transforms_manifest(
    *,
    site_root: Path,
    site_id: str,
    transforms: Dict[str, Optional[List[List[float]]]],
    results: List[Dict[str, Any]],
) -> None:
    entries: List[Dict[str, Any]] = []
    result_by_session = {str(item.get("session_id") or ""): item for item in results}
    for session_id, transform in transforms.items():
        result = result_by_session.get(session_id, {})
        entries.append(
            {
                "session_id": session_id,
                "status": "aligned" if transform is not None else str(result.get("status") or "failed"),
                "site_frame_transform": transform,
                "reference_session": result.get("reference_session"),
                "candidate_count": result.get("candidate_count"),
                "inlier_count": result.get("inlier_count"),
                "inlier_fraction": result.get("inlier_fraction"),
                "combined_score": result.get("combined_score"),
                "residual_translation_p95_m": result.get("residual_translation_p95_m"),
                "residual_rotation_p95_deg": result.get("residual_rotation_p95_deg"),
            }
        )
    write_json(
        site_root / "site_transforms.json",
        {
            "schema_version": "v1",
            "site_id": site_id,
            "generated_at": utc_now_iso(),
            "entries": entries,
        },
    )


def _write_alignment_validation(
    *,
    site_root: Path,
    site_id: str,
    results: List[Dict[str, Any]],
    aligned_sessions: List[str],
) -> None:
    aligned = [item for item in results if item.get("status") == "aligned"]
    write_json(
        site_root / "alignment_validation.json",
        {
            "schema_version": "v1",
            "site_id": site_id,
            "generated_at": utc_now_iso(),
            "session_results": results,
            "aligned_session_count": len(aligned_sessions),
            "mean_inlier_fraction": round(
                float(np.mean([float(item.get("inlier_fraction") or 0.0) for item in aligned])) if aligned else 0.0,
                4,
            ),
            "translation_residual_p95_m": round(
                _sm_p95([float(item.get("residual_translation_p95_m") or 0.0) for item in aligned]) if aligned else 0.0,
                4,
            ),
            "rotation_residual_p95_deg": round(
                _sm_p95([float(item.get("residual_rotation_p95_deg") or 0.0) for item in aligned]) if aligned else 0.0,
                4,
            ),
        },
    )


def _update_overlap_graph_alignment(
    *,
    site_root: Path,
    alignment_edges: List[Dict[str, Any]],
) -> None:
    graph_path = site_root / "site_overlap_graph.json"
    if not graph_path.is_file():
        return
    graph = read_json(graph_path)
    edges = list(graph.get("edges") or [])
    accepted_pairs = {
        (str(item.get("from_session_id") or ""), str(item.get("to_session_id") or ""))
        for item in alignment_edges
    }
    for edge in edges:
        pair = (str(edge.get("from_session_id") or ""), str(edge.get("to_session_id") or ""))
        reverse_pair = (pair[1], pair[0])
        if pair in accepted_pairs or reverse_pair in accepted_pairs:
            edge["accepted_for_alignment"] = True
    graph["generated_at"] = utc_now_iso()
    graph["alignment_edges"] = alignment_edges
    write_json(graph_path, graph)


def _write_static_memory_artifacts(
    *,
    site_root: Path,
    site_index_path: Path,
    site_id: str,
    storage_root: Path,
) -> None:
    records = _sm_load_jsonl(site_index_path)
    aligned_records = [record for record in records if record.get("site_frame_transform") is not None]
    if not aligned_records:
        return

    static_root = site_root / "static_memory"
    static_root.mkdir(parents=True, exist_ok=True)
    point_rows: List[np.ndarray] = []
    dynamic_rows: List[Dict[str, Any]] = []
    visibility_graph: Dict[str, List[str]] = {}
    for record in aligned_records:
        reference_id = str(record.get("reference_id") or "")
        visibility_graph[reference_id] = list(record.get("visibility_cells") or [])
        if float(record.get("staticness_score") or 0.0) < 0.55:
            dynamic_rows.append(
                {
                    "reference_id": reference_id,
                    "capture_id": record.get("capture_id"),
                    "chunk_id": record.get("chunk_id"),
                    "staticness_score": record.get("staticness_score"),
                    "anchor_observations": record.get("anchor_observations") or [],
                }
            )
            continue
        depth = _sm_load_numeric_array(record.get("depth_uri"), storage_root=storage_root)
        if depth is None:
            continue
        confidence = _sm_load_numeric_array(record.get("confidence_uri"), storage_root=storage_root)
        intrinsics = record.get("intrinsics") if isinstance(record.get("intrinsics"), dict) else {}
        T_site = effective_pose(record)
        if T_site is None:
            continue
        points = backproject_depth_points(
            depth=depth,
            intrinsics=intrinsics,
            T_world_camera=T_site,
            confidence=confidence,
            static_weight=float(record.get("staticness_score") or 1.0),
        )
        if points.size == 0:
            continue
        point_rows.append(points)

    all_points = np.concatenate(point_rows, axis=0) if point_rows else np.zeros((0, 4), dtype=np.float32)
    pointcloud_path = static_root / "static_pointcloud.ply"
    write_ascii_pointcloud(pointcloud_path, all_points)
    planes = plane_summaries(all_points)
    planes_path = static_root / "planes.json"
    write_json(planes_path, {"schema_version": "v1", "site_id": site_id, "planes": planes})
    visibility_path = static_root / "visibility_graph.json"
    write_json(visibility_path, {"schema_version": "v1", "site_id": site_id, "visibility": visibility_graph})
    dynamic_path = static_root / "dynamic_observations.jsonl"
    with dynamic_path.open("w", encoding="utf-8") as handle:
        for row in dynamic_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    write_json(
        site_root / "site_static_map_manifest.json",
        {
            "schema_version": "v1",
            "site_id": site_id,
            "generated_at": utc_now_iso(),
            "point_count": int(all_points.shape[0]),
            "aligned_reference_frame_count": len(aligned_records),
            "dynamic_observation_count": len(dynamic_rows),
            "artifacts": {
                "static_pointcloud": str(pointcloud_path),
                "planes": str(planes_path),
                "visibility_graph": str(visibility_path),
                "dynamic_observations": str(dynamic_path),
            },
        },
    )


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
    manifest["aligned_capture_fraction"] = round(
        len(aligned_sessions) / float(len(captures) or 1),
        4,
    )
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
    return _sm_load_jsonl(path)


def _uri_to_local(uri: str, ctx: LocalCaptureContext) -> Optional[Path]:
    """Resolve a gs:// URI to a local path via the storage root."""
    local = gs_uri_to_local(uri, storage_root=ctx.storage_root)
    if local is not None and local.is_file():
        return local
    if uri.startswith("gs://"):
        trimmed = uri[5:]
        bucket, _, key = trimmed.partition("/")
        if bucket and key:
            flat = ctx.storage_root / key
            if flat.is_file():
                return flat
    return local
