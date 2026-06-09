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
from .robot_eval_job_orchestrator import run_robot_eval_job_request_inbox
from .simulation_automation import build_simulation_automation
from .synthesis.synthesize import synthesize_view

_CURRENT_PIPELINE_LANES = ("qualification", "evaluation_prep", "simulation_automation")
_LEGACY_PIPELINE_LANES = (
    "scene_memory",
    "retrieval_index",
    "frame_alignment",
    "synthesis_coverage_validation",
    "cosmos_single_capture_smoke",
)
_LANE_ORDER = (
    "qualification",
    "scene_memory",
    "retrieval_index",
    "frame_alignment",
    "evaluation_prep",
    "simulation_automation",
    "synthesis_coverage_validation",
    "cosmos_single_capture_smoke",
)
_SUPPORTED_LANES = {*_CURRENT_PIPELINE_LANES, *_LEGACY_PIPELINE_LANES, "current", "all"}
_LANE_ALIASES = {
    "robot_eval_dataset": "evaluation_prep",
    "task_evaluation_run": "simulation_automation",
}
_ANDROID_XR_VIDEO_ONLY_PROFILE = "android_xr_glasses"
_ANDROID_XR_VIDEO_ONLY_MODALITY = "android_xr_video_only"


@dataclass(frozen=True)
class PipelineConfig:
    gcs_root: Path = Path(os.getenv("GCS_ROOT", "/mnt/gcs"))


def _normalize_lane_value(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    value = _LANE_ALIASES.get(value, value)
    if value not in _SUPPORTED_LANES:
        raise ValueError(f"Unsupported pipeline lane: {raw}")
    return value


def _normalize_requested_lanes(values: Any) -> List[str]:
    if values is None:
        raw_values: List[str] = []
    elif isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, (list, tuple, set)):
        raw_values = [str(value) for value in values]
    else:
        raw_values = [str(values)]

    normalized: List[str] = []
    for value in raw_values:
        lane = _normalize_lane_value(value)
        if lane is None:
            continue
        if lane in {"all", "current"}:
            for expanded in _CURRENT_PIPELINE_LANES:
                if expanded not in normalized:
                    normalized.append(expanded)
            continue
        if lane in {"retrieval_index", "frame_alignment", "evaluation_prep"} and "qualification" not in normalized:
            normalized.append("qualification")
        if lane == "simulation_automation":
            if "qualification" not in normalized:
                normalized.append("qualification")
            if "evaluation_prep" not in normalized:
                normalized.append("evaluation_prep")
        if lane not in normalized:
            normalized.append(lane)
    ordered: List[str] = []
    for lane in _LANE_ORDER:
        if lane in normalized and lane not in ordered:
            ordered.append(lane)
    return ordered


def _mapping_value(payload: Mapping[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is not None:
        return value
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    if key in metadata:
        return metadata.get(key)
    capture_bundle = payload.get("capture_bundle") if isinstance(payload.get("capture_bundle"), Mapping) else {}
    return capture_bundle.get(key)


def _descriptor_is_android_xr_video_only(raw_payload: Mapping[str, Any]) -> bool:
    capture_profile_id = str(_mapping_value(raw_payload, "capture_profile_id") or "").strip().lower()
    capture_modality = str(_mapping_value(raw_payload, "capture_modality") or "").strip().lower()
    return (
        capture_profile_id == _ANDROID_XR_VIDEO_ONLY_PROFILE
        or capture_profile_id.startswith("android_xr_")
        or capture_modality == _ANDROID_XR_VIDEO_ONLY_MODALITY
    )


def _descriptor_is_native_default_candidate(raw_payload: Mapping[str, Any]) -> bool:
    if _descriptor_is_android_xr_video_only(raw_payload):
        return False
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
    raw_requested_outputs = raw_payload.get("requested_outputs") or raw_payload.get("requestedOutputs")
    if isinstance(raw_requested_outputs, str):
        requested_outputs = [raw_requested_outputs]
    elif isinstance(raw_requested_outputs, (list, tuple, set)):
        requested_outputs = [str(value) for value in raw_requested_outputs]
    else:
        requested_outputs = []
    normalized_outputs = {str(value).strip().lower() for value in requested_outputs if str(value).strip()}
    if isinstance(raw_payload, Mapping) and _descriptor_is_android_xr_video_only(raw_payload):
        return ["qualification"]
    descriptor_requested_lanes = _normalize_requested_lanes(
        raw_payload.get("requested_lanes") or raw_payload.get("requestedLanes")
    )
    if descriptor_requested_lanes:
        if not normalized_outputs and descriptor_requested_lanes == ["qualification", "scene_memory"]:
            return ["qualification"]
        return descriptor_requested_lanes
    if isinstance(raw_payload, Mapping) and _descriptor_is_native_default_candidate(raw_payload):
        return list(_CURRENT_PIPELINE_LANES)
    if "task_evaluation_run" in normalized_outputs:
        return list(_CURRENT_PIPELINE_LANES)
    if "robot_eval_dataset" in normalized_outputs:
        return ["qualification", "evaluation_prep"]
    if normalized_outputs & {
        "preview",
        "preview_simulation",
        "evaluation_prep",
        "deeper_evaluation",
        "managed_tuning",
        "data_licensing",
    }:
        return list(_CURRENT_PIPELINE_LANES)
    if "scene_memory" in normalized_outputs:
        return ["qualification", "scene_memory"]
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


def _build_derived_lane_result(
    *,
    lane: str,
    source: str,
    qualification_result: Mapping[str, Any],
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "completed",
        "lane": lane,
        "scene_id": qualification_result.get("scene_id"),
        "capture_id": qualification_result.get("capture_id"),
        "pipeline_prefix": qualification_result.get("pipeline_prefix"),
        "source": source,
    }
    if extra_fields:
        result.update(dict(extra_fields))
    return result


def _robot_eval_job_request_inbox_for_capture(capture_root: Path) -> Optional[Path]:
    """Return the first configured inbox containing WebApp robot-eval job requests."""

    candidates: List[Path] = []
    env_inbox = os.getenv("ROBOT_EVAL_JOB_REQUEST_INBOX_DIR")
    if env_inbox:
        candidates.append(Path(env_inbox))
    candidates.append(capture_root / "pipeline" / "robot_eval_job_requests" / "inbox")

    for candidate in candidates:
        if candidate.is_dir() and any(path.is_file() for path in candidate.glob("*.json")):
            return candidate
    return None


def _run_robot_eval_job_inbox_if_ready(capture_root: Path) -> Dict[str, Any]:
    inbox = _robot_eval_job_request_inbox_for_capture(capture_root)
    if inbox is None:
        return {
            "status": "waiting_for_job_requests",
            "processed_count": 0,
            "inbox_dir": None,
            "manifest_path": None,
            "claim_boundary": "no_robot_eval_job_request_v1_files_found",
        }
    result = run_robot_eval_job_request_inbox(
        capture_root=capture_root,
        inbox_dir=inbox,
        provisioner=os.getenv("ROBOT_EVAL_JOB_DEFAULT_PROVISIONER", "fixture_local"),
        simulator=os.getenv("ROBOT_EVAL_JOB_DEFAULT_SIMULATOR", "fixture"),
    )
    return {
        "status": result.get("status"),
        "processed_count": result.get("processed_count", 0),
        "inbox_dir": str(inbox),
        "manifest_path": str(
            capture_root / "pipeline" / "robot_eval_job_requests" / "inbox_run_manifest.json"
        ),
        "claim_boundary": "job_requests_processed_with_gated_default_execution",
    }


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
                    _build_derived_lane_result(
                        lane="scene_memory",
                        source="qualification_artifacts",
                        qualification_result=qualification_result,
                    )
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
            lane_result = _build_derived_lane_result(
                lane="evaluation_prep",
                source="evaluation_prep_artifacts",
                qualification_result=qualification_result,
                extra_fields={"manifest_path": evaluation_prep_result.get("manifest_path")},
            )
            results.append(lane_result)
            continue
        if selected_lane == "simulation_automation":
            capture_root = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root).parent
            automation_result = build_simulation_automation(capture_root=capture_root)
            robot_eval_jobs = _run_robot_eval_job_inbox_if_ready(capture_root)
            lane_result = _build_derived_lane_result(
                lane="simulation_automation",
                source="simulation_automation_artifacts",
                qualification_result=qualification_result or {},
                extra_fields={
                    "manifest_path": automation_result.get("manifest_path"),
                    "plan_path": automation_result.get("plan_path"),
                    "automation_status": automation_result.get("status"),
                    "robot_eval_job_inbox_status": robot_eval_jobs.get("status"),
                    "robot_eval_job_inbox_processed_count": robot_eval_jobs.get(
                        "processed_count",
                        0,
                    ),
                    "robot_eval_job_inbox_manifest_path": robot_eval_jobs.get("manifest_path"),
                },
            )
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
        if selected_lane == "cosmos_single_capture_smoke":
            from .synthesis.cosmos_benchmark import run_cosmos_single_capture_smoke_lane

            capture_root = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root).parent
            smoke_result = run_cosmos_single_capture_smoke_lane(
                capture_root=capture_root,
                descriptor_gcs_uri=descriptor_gcs_uri,
                cfg=cfg,
            )
            results.append({"lane": "cosmos_single_capture_smoke", **smoke_result})
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
        help=(
            "current/all, qualification, evaluation_prep, simulation_automation, "
            "or explicit legacy lanes: scene_memory, retrieval_index, frame_alignment, "
            "synthesis_coverage_validation, cosmos_single_capture_smoke"
        ),
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
