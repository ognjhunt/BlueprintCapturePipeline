"""Lane-aware capture pipeline entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .common import PipelineError, parse_bool, parse_gs_uri, resolve_gs_uri_to_path
from .evaluation_prep_stage import run_evaluation_prep_stage
from .geometry_sources import load_capture_geometry
from .local_capture import resolve_local_capture_context
from .materialization import materialize_capture_bundle
from .qualification import run_qualification_pipeline
from .frame_alignment_stage import run_frame_alignment_stage
from .retrieval_index_stage import run_retrieval_index_stage
from .synthesis.synthesize import synthesize_view

_SUPPORTED_LANES = {
    "qualification", "scene_memory", "evaluation_prep",
    "retrieval_index", "frame_alignment",
    "synthesis_coverage_validation",
    "all",
}


@dataclass(frozen=True)
class PipelineConfig:
    gcs_root: Path = Path(os.getenv("GCS_ROOT", "/mnt/gcs"))


def _normalize_lane_value(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value not in _SUPPORTED_LANES:
        raise ValueError(f"Unsupported pipeline lane: {raw}")
    return value


def _normalize_requested_lanes(values: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    for value in values or []:
        lane = _normalize_lane_value(value)
        if lane is None:
            continue
        if lane == "all":
            for expanded in (
                "qualification",
                "scene_memory",
                "retrieval_index",
                "frame_alignment",
                "evaluation_prep",
                "synthesis_coverage_validation",
            ):
                if expanded not in normalized:
                    normalized.append(expanded)
            continue
        if lane in {"retrieval_index", "frame_alignment", "evaluation_prep"} and "qualification" not in normalized:
            normalized.append("qualification")
        if lane not in normalized:
            normalized.append(lane)
    ordered: List[str] = []
    for lane in (
        "qualification",
        "scene_memory",
        "retrieval_index",
        "frame_alignment",
        "evaluation_prep",
        "synthesis_coverage_validation",
    ):
        if lane in normalized and lane not in ordered:
            ordered.append(lane)
    return ordered


def _descriptor_is_native_default_candidate(raw_payload: Mapping[str, Any]) -> bool:
    capture_mode = raw_payload.get("capture_mode")
    metadata = raw_payload.get("metadata") if isinstance(raw_payload.get("metadata"), Mapping) else {}
    if not isinstance(capture_mode, Mapping) and isinstance(metadata.get("capture_mode"), Mapping):
        capture_mode = metadata.get("capture_mode")
    scene_memory_capture = raw_payload.get("scene_memory_capture")
    if not isinstance(scene_memory_capture, Mapping) and isinstance(metadata.get("scene_memory_capture"), Mapping):
        scene_memory_capture = metadata.get("scene_memory_capture")
    quality = raw_payload.get("quality") if isinstance(raw_payload.get("quality"), Mapping) else {}
    resolved_mode = str((capture_mode or {}).get("resolved_mode") or "").strip().lower()
    return resolved_mode == "site_world_candidate" and bool(
        (scene_memory_capture or {}).get("world_model_candidate")
        or quality.get("world_model_candidate")
    )


def _load_descriptor_requested_lanes(descriptor_gcs_uri: str, gcs_root: Any) -> List[str]:
    descriptor_path = resolve_gs_uri_to_path(descriptor_gcs_uri, gcs_root)
    try:
        raw_payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw_payload = {}
    if isinstance(raw_payload, Mapping) and _descriptor_is_native_default_candidate(raw_payload):
        return [
            "qualification",
            "scene_memory",
            "retrieval_index",
            "frame_alignment",
            "evaluation_prep",
        ]
    raw_requested_outputs = raw_payload.get("requested_outputs")
    if isinstance(raw_requested_outputs, str):
        requested_outputs = [raw_requested_outputs]
    elif isinstance(raw_requested_outputs, (list, tuple, set)):
        requested_outputs = [str(value) for value in raw_requested_outputs]
    else:
        requested_outputs = []
    normalized_outputs = {str(value).strip().lower() for value in requested_outputs if str(value).strip()}
    if "deeper_evaluation" in normalized_outputs:
        return ["qualification", "scene_memory", "retrieval_index", "frame_alignment", "evaluation_prep"]
    if normalized_outputs & {"managed_tuning", "data_licensing"}:
        return ["qualification", "scene_memory", "retrieval_index", "frame_alignment", "evaluation_prep"]
    if normalized_outputs & {"scene_memory", "preview_simulation", "evaluation_prep"}:
        return ["qualification", "scene_memory", "retrieval_index", "frame_alignment", "evaluation_prep"]
    return ["qualification"]


def resolve_requested_lanes(
    *,
    descriptor_gcs_uri: str,
    gcs_root: Any,
    lane: Optional[str] = None,
    requested_lanes: Optional[List[str]] = None,
) -> List[str]:
    explicit_lane = _normalize_lane_value(lane)
    if explicit_lane:
        return _normalize_requested_lanes([explicit_lane])

    env_lane = _normalize_lane_value(os.getenv("PIPELINE_LANE"))
    if env_lane:
        return _normalize_requested_lanes([env_lane])

    normalized_requested = _normalize_requested_lanes(requested_lanes)
    if normalized_requested:
        return normalized_requested

    descriptor_requested = _normalize_requested_lanes(_load_descriptor_requested_lanes(descriptor_gcs_uri, gcs_root))
    return descriptor_requested or ["qualification"]


def run_capture_pipeline(
    *,
    descriptor_gcs_uri: str,
    lane: Optional[str] = None,
    requested_lanes: Optional[List[str]] = None,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    cfg = config or PipelineConfig()
    lanes = resolve_requested_lanes(
        descriptor_gcs_uri=descriptor_gcs_uri,
        gcs_root=cfg.gcs_root,
        lane=lane,
        requested_lanes=requested_lanes,
    )

    results: List[Dict[str, Any]] = []
    qualification_result: Optional[Dict[str, Any]] = None
    for selected_lane in lanes:
        if selected_lane in {"qualification", "scene_memory"}:
            if qualification_result is None:
                qualification_result = run_qualification_pipeline(
                    descriptor_gcs_uri=descriptor_gcs_uri,
                    config=cfg,
                    requested_lanes=lanes,
                )
            if selected_lane == "qualification":
                results.append(qualification_result)
            else:
                results.append(
                    {
                        "status": "completed",
                        "lane": "scene_memory",
                        "scene_id": qualification_result.get("scene_id"),
                        "capture_id": qualification_result.get("capture_id"),
                        "pipeline_prefix": qualification_result.get("pipeline_prefix"),
                        "source": "qualification_artifacts",
                    }
                )
            continue
        if selected_lane == "evaluation_prep":
            if qualification_result is None:
                qualification_result = run_qualification_pipeline(
                    descriptor_gcs_uri=descriptor_gcs_uri,
                    config=cfg,
                    requested_lanes=lanes,
                )
            evaluation_prep_result = run_evaluation_prep_stage(
                capture_root=resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root).parent,
                provider_name="manual",
            )
            lane_result = {
                "status": "completed",
                "lane": "evaluation_prep",
                "scene_id": qualification_result.get("scene_id"),
                "capture_id": qualification_result.get("capture_id"),
                "pipeline_prefix": qualification_result.get("pipeline_prefix"),
                "source": "evaluation_prep_artifacts",
                "manifest_path": evaluation_prep_result.get("manifest_path"),
            }
            results.append(lane_result)
            continue
        if selected_lane == "retrieval_index":
            capture_root = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root).parent
            retrieval_result = run_retrieval_index_stage(
                capture_root=capture_root,
                force_rebuild=parse_bool(os.getenv("RETRIEVAL_INDEX_FORCE_REBUILD"), default=False),
            )
            results.append({"lane": "retrieval_index", **retrieval_result})
            continue
        if selected_lane == "frame_alignment":
            capture_root = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root).parent
            alignment_result = run_frame_alignment_stage(
                capture_root=capture_root,
                force_realign=parse_bool(os.getenv("FRAME_ALIGNMENT_FORCE_REALIGN"), default=False),
            )
            results.append({"lane": "frame_alignment", **alignment_result})
            continue
        if selected_lane == "synthesis_coverage_validation":
            capture_root = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root).parent
            synthesis_result = _run_synthesis_coverage_validation(
                capture_root=capture_root,
                descriptor_gcs_uri=descriptor_gcs_uri,
                cfg=cfg,
            )
            results.append({"lane": "synthesis_coverage_validation", **synthesis_result})
            continue
        raise ValueError(f"Unsupported pipeline lane: {selected_lane}")

    parsed = parse_gs_uri(descriptor_gcs_uri)
    return {
        "status": "completed",
        "descriptor_gcs_uri": descriptor_gcs_uri,
        "bucket": parsed.bucket,
        "lanes": lanes,
        "results": results,
    }


def _run_synthesis_coverage_validation(
    *,
    capture_root: Path,
    descriptor_gcs_uri: str,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    return run_capture_synthesis_validation(
        capture_root=capture_root,
        descriptor_gcs_uri=descriptor_gcs_uri,
        cfg=cfg,
        mode="splat_only",
    )


def run_capture_synthesis_validation(
    *,
    capture_root: Path,
    descriptor_gcs_uri: str,
    cfg: PipelineConfig,
    mode: str = "splat_only",
) -> Dict[str, Any]:
    """
    Run a single-frame synthesis validation QA check.

    Gates:
    1. capture_descriptor.json must have world_model_candidate=true
    2. The site's reference index must contain at least one record from a
       different pass_id than this capture (so there is a prior reference to
       warp from).

    Returns a dict with status "completed", "skipped", or "failed".
    Non-blocking: exceptions from synthesis are caught and returned as "failed".
    """
    import datetime

    # --- Load descriptor to check world_model_candidate gate ---
    descriptor_path = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root)
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "failed", "reason": f"descriptor_unreadable: {exc}"}

    quality = descriptor.get("quality") if isinstance(descriptor.get("quality"), Mapping) else {}
    if not (descriptor.get("world_model_candidate") or quality.get("world_model_candidate")):
        return {"status": "skipped", "reason": "not_world_model_candidate"}

    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    site_identity = metadata.get("site_identity") if isinstance(metadata.get("site_identity"), Mapping) else {}
    topology = metadata.get("capture_topology") if isinstance(metadata.get("capture_topology"), Mapping) else {}
    site_id = site_identity.get("site_id") or descriptor.get("site_id")
    capture_id = descriptor.get("capture_id")
    pass_id = topology.get("pass_id")

    if not site_id:
        return {"status": "skipped", "reason": "no_site_id_in_descriptor"}

    # --- Check site reference index exists and has prior pass records ---
    parsed = parse_gs_uri(descriptor_gcs_uri)
    index_path = cfg.gcs_root / parsed.bucket / "sites" / site_id / "reference_memory" / "site_reference_index.jsonl"
    if not index_path.is_file():
        return {"status": "skipped", "reason": "no_site_reference_index"}

    try:
        index_records = [
            json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "failed", "reason": f"index_unreadable: {exc}"}

    # Only synthesize against a reference from a different pass (not this capture's own frames)
    prior_records = [r for r in index_records if r.get("pass_id") != pass_id]
    if not prior_records:
        return {"status": "skipped", "reason": "no_prior_pass_in_index"}

    # Use spatial retrieval only when the site frame is established (Phase 3B aligned).
    # Before alignment, site_frame_transform is null, so cross-session spatial distances
    # are meaningless — fall back to embedding (appearance-based, works pre-alignment).
    index_aligned = any(r.get("site_frame_transform") is not None for r in prior_records)
    query_mode = "spatial" if index_aligned else "embedding"

    geometry = load_capture_geometry(
        context=resolve_local_capture_context(capture_root),
        descriptor=descriptor,
    )
    pose_rows = list(geometry.get("poses") or [])
    target_T = None
    target_intrinsics = geometry.get("intrinsics") if isinstance(geometry.get("intrinsics"), Mapping) else None
    if pose_rows:
        midpoint_row = pose_rows[len(pose_rows) // 2]
        target_T = midpoint_row.get("T_world_camera") or midpoint_row.get("transform")

    if target_T is None:
        return {"status": "skipped", "reason": "no_geometry_poses"}

    import numpy as np
    T = np.array(target_T, dtype=np.float64)
    if T.ndim == 1 and T.shape[0] == 16:
        T = T.reshape(4, 4)
    if T.shape != (4, 4):
        return {"status": "skipped", "reason": "invalid_pose_shape"}

    if target_intrinsics is None:
        # Fall back to a reasonable iPhone Pro default
        target_intrinsics = {"fx": 1462.0, "fy": 1462.0, "cx": 960.0, "cy": 720.0, "width": 1920, "height": 1440}

    target_h = int(target_intrinsics.get("height", 1440))
    target_w = int(target_intrinsics.get("width", 1920))

    # --- Run synthesis (non-blocking) ---
    output_stem = "cosmos" if mode == "cosmos_i2w" else "splat"
    output_path = (
        cfg.gcs_root / parsed.bucket / "sites" / site_id / "coverage_validation"
        / f"{capture_id}_{output_stem}.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        synth_result = synthesize_view(
            site_id=site_id,
            storage_root=cfg.gcs_root,
            bucket=parsed.bucket,
            target_T_world_camera=T,
            target_intrinsics=target_intrinsics,
            target_h=target_h,
            target_w=target_w,
            output_path=output_path,
            mode=mode,
            k=1,
            query_mode=query_mode,
            depth_scale=0.001,
        )
    except Exception as exc:  # non-blocking: synthesis failure never blocks the pipeline
        return {"status": "failed", "reason": str(exc)}

    return {
        "status": synth_result.get("status", "completed"),
        "capture_id": capture_id,
        "site_id": site_id,
        "synthesis_mode": mode,
        "retrieval_mode": query_mode,
        "coverage_frac": synth_result.get("coverage_frac"),
        "ref_frame_distance_m": synth_result.get("retrieval_dist_m"),
        "output_uri": f"gs://{parsed.bucket}/sites/{site_id}/coverage_validation/{capture_id}_{output_stem}.jpg",
        "output_video_uri": (
            f"gs://{parsed.bucket}/sites/{site_id}/coverage_validation/{capture_id}_{output_stem}.mp4"
            if mode == "cosmos_i2w"
            else None
        ),
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


def run_capture_pipeline_for_capture(
    *,
    bucket: str,
    scene_id: str,
    capture_id: str,
    lane: Optional[str] = None,
    requested_lanes: Optional[List[str]] = None,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    cfg = config or PipelineConfig()
    materialized = materialize_capture_bundle(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=cfg.gcs_root,
    )
    return run_capture_pipeline(
        descriptor_gcs_uri=str(materialized["descriptor_uri"]),
        lane=lane,
        requested_lanes=requested_lanes,
        config=cfg,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run lane-aware capture pipeline")
    parser.add_argument(
        "--descriptor-gcs-uri",
        default=(os.getenv("PIPELINE_DESCRIPTOR_GCS_URI") or "").strip() or None,
        help="gs:// URI for capture_descriptor.json",
    )
    parser.add_argument("--bucket", default=(os.getenv("PIPELINE_BUCKET") or "").strip() or None)
    parser.add_argument("--scene-id", default=(os.getenv("PIPELINE_SCENE_ID") or "").strip() or None)
    parser.add_argument("--capture-id", default=(os.getenv("PIPELINE_CAPTURE_ID") or "").strip() or None)
    parser.add_argument(
        "--lane",
        default=None,
        help="qualification, scene_memory, evaluation_prep, retrieval_index, frame_alignment, synthesis_coverage_validation, or all",
    )
    args = parser.parse_args(argv)

    try:
        if args.descriptor_gcs_uri:
            cfg = PipelineConfig()
            descriptor_path = resolve_gs_uri_to_path(args.descriptor_gcs_uri, cfg.gcs_root)
            if descriptor_path.exists() or not (args.bucket and args.scene_id and args.capture_id):
                run_capture_pipeline(
                    descriptor_gcs_uri=args.descriptor_gcs_uri,
                    lane=args.lane,
                    config=cfg,
                )
            else:
                run_capture_pipeline_for_capture(
                    bucket=args.bucket,
                    scene_id=args.scene_id,
                    capture_id=args.capture_id,
                    lane=args.lane,
                    config=cfg,
                )
        elif args.bucket and args.scene_id and args.capture_id:
            run_capture_pipeline_for_capture(
                bucket=args.bucket,
                scene_id=args.scene_id,
                capture_id=args.capture_id,
                lane=args.lane,
            )
        else:
            parser.error("--descriptor-gcs-uri or --bucket/--scene-id/--capture-id is required")
    except (PipelineError, ValueError) as exc:
        print(f"[capture-orchestrator] FAILED: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - safety net
        print(f"[capture-orchestrator] FAILED (unexpected): {exc}")
        return 1

    print("[capture-orchestrator] completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
