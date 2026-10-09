"""Lane-aware capture pipeline entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .core.common import PipelineError, parse_bool, parse_gs_uri, resolve_gs_uri_to_path
from .evaluation_prep_stage import run_evaluation_prep_stage
from .geometry_sources import load_capture_geometry
from .core.logging_utils import log_event
from .local_capture import resolve_local_capture_context
from .materialization import materialize_capture_bundle
from .core.pipeline_settings import PipelineConfig, PipelineSettings
from .site_package_orchestrator import run_qualification_pipeline
from .frame_alignment_stage import run_frame_alignment_stage
from .core.lane_resume import (
    lane_ledger_input_fingerprint,
    lane_resume_disabled,
    read_completed_lane_result,
    record_lane_completion,
)
from .core.lane_admission import (
    LEGACY_CAPTURE_LANES,
    require_legacy_lane_admission,
)
from .retrieval_index_stage import run_retrieval_index_stage
from .agent_execution_offer import publish_agent_execution_offer
from .simulation_automation import build_simulation_automation
from .synthesis.synthesize import synthesize_view

logger = logging.getLogger(__name__)

_CURRENT_PIPELINE_LANES = ("qualification", "evaluation_prep", "simulation_automation")
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
_SUPPORTED_LANES = {*_CURRENT_PIPELINE_LANES, *LEGACY_CAPTURE_LANES, "current", "all"}
_LANE_ALIASES = {
    "robot_eval_dataset": "evaluation_prep",
    "task_evaluation_run": "simulation_automation",
}
_ANDROID_XR_VIDEO_ONLY_PROFILE = "android_xr_glasses"
_ANDROID_XR_VIDEO_ONLY_MODALITY = "android_xr_video_only"
SIM_ONLY_BETA_AUTONOMY_ENV = "BLUEPRINT_SIM_ONLY_BETA_AUTONOMY"
SIM_ONLY_BETA_DEFAULT_TASK_EVAL_ENV = "BLUEPRINT_SIM_ONLY_BETA_DEFAULT_TASK_EVAL"


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


def _read_json_mapping(path: Path) -> Dict[str, Any]:
    try:
        from .task_evaluation_scene_retirement_generations import read_selected_capture_input_bytes
        payload = json.loads(read_selected_capture_input_bytes(path))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _descriptor_requested_outputs(raw_payload: Mapping[str, Any]) -> set[str]:
    raw_requested_outputs = raw_payload.get("requested_outputs") or raw_payload.get("requestedOutputs")
    if isinstance(raw_requested_outputs, str):
        values = [raw_requested_outputs]
    elif isinstance(raw_requested_outputs, (list, tuple, set)):
        values = [str(value) for value in raw_requested_outputs]
    else:
        values = []
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _sim_only_beta_autonomy_enabled() -> bool:
    return _env_truthy(SIM_ONLY_BETA_AUTONOMY_ENV)


def _sim_only_beta_default_task_eval_enabled() -> bool:
    return _sim_only_beta_autonomy_enabled() or _env_truthy(SIM_ONLY_BETA_DEFAULT_TASK_EVAL_ENV)


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
    raw_payload = _read_json_mapping(descriptor_path)
    normalized_outputs = _descriptor_requested_outputs(raw_payload)
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
    if _sim_only_beta_default_task_eval_enabled():
        return list(_CURRENT_PIPELINE_LANES)
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


def run_capture_pipeline(
    *,
    descriptor_gcs_uri: str,
    lane: Optional[str] = None,
    requested_lanes: Optional[List[str]] = None,
    allow_legacy_lanes: bool = False,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    cfg = config or PipelineConfig()
    lanes = resolve_requested_lanes(
        descriptor_gcs_uri=descriptor_gcs_uri,
        gcs_root=cfg.gcs_root,
        lane=lane,
        requested_lanes=requested_lanes,
    )
    require_legacy_lane_admission(
        lanes,
        allow_legacy_lanes=allow_legacy_lanes,
    )
    allow_lane_fault_isolation = len(lanes) > 1
    descriptor_path = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root)
    # Production dispatch (storage_trigger -> run_capture_pipeline) bypasses the
    # run_e2e stage ledger, so Cloud Tasks x Cloud Run Job retries would re-run
    # every lane. The lane ledger skips lanes already completed for the same
    # capture input fingerprint. BLUEPRINT_LANE_RESUME_DISABLED=1 disables it.
    resume_root = resolve_local_capture_context(descriptor_path).capture_root
    lane_ledger_fingerprint: Optional[Dict[str, Any]] = None
    if not lane_resume_disabled():
        try:
            lane_ledger_fingerprint = lane_ledger_input_fingerprint(
                capture_root=resume_root,
                descriptor_path=descriptor_path,
            )
        except OSError:
            logger.warning(
                "capture_pipeline.lane_resume_fingerprint_failed descriptor=%s",
                descriptor_gcs_uri,
                exc_info=True,
            )
    log_event(
        logger,
        logging.INFO,
        "capture_pipeline.started",
        descriptor_gcs_uri=descriptor_gcs_uri,
        descriptor_path=str(descriptor_path),
        requested_lane=lane,
        lanes=lanes,
    )

    results: List[Dict[str, Any]] = []
    lane_failures: List[Dict[str, Any]] = []
    qualification_result: Optional[Dict[str, Any]] = None

    def _append_lane_result(selected_lane: str, lane_result: Mapping[str, Any]) -> None:
        if selected_lane in {"evaluation_prep", "simulation_automation"}:
            lane_result = {**lane_result, "artifact_purpose": "evaluation_preparation",
                           "robot_evaluation_performed": False}
        results.append(dict(lane_result))
        log_event(
            logger,
            logging.INFO,
            "capture_pipeline.lane_completed",
            descriptor_gcs_uri=descriptor_gcs_uri,
            selected_lane=selected_lane,
            lane=lane_result.get("lane") or selected_lane,
            lane_status=lane_result.get("status"),
            result_count=len(results),
            manifest_path=lane_result.get("manifest_path"),
            source=lane_result.get("source"),
        )
        if lane_ledger_fingerprint is not None and not lane_result.get(
            "resumed_from_lane_ledger"
        ):
            record_lane_completion(
                capture_root=resume_root,
                lane=selected_lane,
                fingerprint=lane_ledger_fingerprint,
                lane_result=lane_result,
            )

    def _append_lane_failure(selected_lane: str, exc: BaseException) -> None:
        failure = {
            "lane": selected_lane,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        results.append(failure)
        lane_failures.append(failure)
        log_event(
            logger,
            logging.ERROR,
            "capture_pipeline.lane_failed",
            descriptor_gcs_uri=descriptor_gcs_uri,
            selected_lane=selected_lane,
            error_type=failure["error_type"],
            error=failure["error"],
            result_count=len(results),
        )

    def _run_lane_call(selected_lane: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - lane failures must not discard prior lanes
            if not allow_lane_fault_isolation:
                raise
            _append_lane_failure(selected_lane, exc)
            return None

    def _qualification_for_lane(selected_lane: str) -> Optional[Dict[str, Any]]:
        nonlocal qualification_result
        if qualification_result is None:
            qualification_result = _run_lane_call(
                selected_lane,
                run_qualification_pipeline,
                descriptor_gcs_uri=descriptor_gcs_uri,
                config=cfg,
                requested_lanes=lanes,
            )
        return qualification_result

    for selected_lane in lanes:
        if lane_ledger_fingerprint is not None:
            resumed_result = read_completed_lane_result(
                capture_root=resume_root,
                lane=selected_lane,
                fingerprint=lane_ledger_fingerprint,
            )
            if selected_lane in {"evaluation_prep", "simulation_automation"} and resumed_result is not None:
                if (resumed_result.get("artifact_purpose") != "evaluation_preparation"
                        or resumed_result.get("robot_evaluation_performed") is not False):
                    # Refresh only preparation projections; construction and
                    # qualification keep their completed-prefix adoption.
                    resumed_result = None
            if resumed_result is not None:
                if selected_lane == "qualification" and qualification_result is None:
                    qualification_result = dict(resumed_result)
                log_event(
                    logger,
                    logging.INFO,
                    "capture_pipeline.lane_skipped_already_completed",
                    descriptor_gcs_uri=descriptor_gcs_uri,
                    selected_lane=selected_lane,
                    reason="lane_ledger_marker_matches_capture_input_fingerprint",
                    marker_dir="pipeline/lane_ledger",
                )
                _append_lane_result(
                    selected_lane,
                    {**resumed_result, "resumed_from_lane_ledger": True},
                )
                continue
        log_event(
            logger,
            logging.INFO,
            "capture_pipeline.lane_started",
            descriptor_gcs_uri=descriptor_gcs_uri,
            selected_lane=selected_lane,
        )
        if selected_lane in {"qualification", "scene_memory"}:
            qualification = _qualification_for_lane(selected_lane)
            if qualification is None:
                continue
            if selected_lane == "qualification":
                _append_lane_result(selected_lane, qualification)
            else:
                _append_lane_result(
                    selected_lane,
                    _build_derived_lane_result(
                        lane="scene_memory",
                        source="qualification_artifacts",
                        qualification_result=qualification,
                    )
                )
            continue
        if selected_lane == "evaluation_prep":
            qualification = _qualification_for_lane(selected_lane)
            if qualification is None:
                continue
            evaluation_prep_result = _run_lane_call(
                selected_lane,
                run_evaluation_prep_stage,
                capture_root=resume_root,
                provider_name="manual",
            )
            if evaluation_prep_result is None:
                continue
            lane_result = _build_derived_lane_result(
                lane="evaluation_prep",
                source="evaluation_prep_artifacts",
                qualification_result=qualification,
                extra_fields={
                    "manifest_path": evaluation_prep_result.get("manifest_path"),
                    "evaluation_prep_result": dict(evaluation_prep_result),
                },
            )
            _append_lane_result(selected_lane, lane_result)
            continue
        if selected_lane == "simulation_automation":
            capture_root = resume_root
            automation_result = _run_lane_call(
                selected_lane,
                build_simulation_automation,
                capture_root=capture_root,
            )
            if automation_result is None:
                continue
            # Website captures publish what a self-serve robot-team run needs
            # from here (capture root, the one scenario, episode count). Best
            # effort: a missing offer only keeps the WebApp plan unpayable.
            publish_agent_execution_offer(capture_root)
            lane_result = _build_derived_lane_result(
                lane="simulation_automation",
                source="simulation_automation_artifacts",
                qualification_result=qualification_result or {},
                extra_fields={
                    "manifest_path": automation_result.get("manifest_path"),
                    "plan_path": automation_result.get("plan_path"),
                    "automation_status": automation_result.get("status"),
                    "artifact_purpose": "evaluation_preparation",
                    "robot_evaluation_performed": False,
                    "robot_eval_job_inbox_status": "separate_evaluation_request_required",
                    "robot_eval_job_inbox_processed_count": 0,
                },
            )
            _append_lane_result(selected_lane, lane_result)
            continue
        if selected_lane == "retrieval_index":
            capture_root = resume_root
            retrieval_result = _run_lane_call(
                selected_lane,
                run_retrieval_index_stage,
                capture_root=capture_root,
                force_rebuild=parse_bool(os.getenv("RETRIEVAL_INDEX_FORCE_REBUILD"), default=False),
            )
            if retrieval_result is None:
                continue
            _append_lane_result(selected_lane, {"lane": "retrieval_index", **retrieval_result})
            continue
        if selected_lane == "frame_alignment":
            capture_root = resume_root
            alignment_result = _run_lane_call(
                selected_lane,
                run_frame_alignment_stage,
                capture_root=capture_root,
                force_realign=parse_bool(os.getenv("FRAME_ALIGNMENT_FORCE_REALIGN"), default=False),
            )
            if alignment_result is None:
                continue
            _append_lane_result(selected_lane, {"lane": "frame_alignment", **alignment_result})
            continue
        if selected_lane == "synthesis_coverage_validation":
            capture_root = resume_root
            synthesis_result = _run_lane_call(
                selected_lane,
                _run_synthesis_coverage_validation,
                capture_root=capture_root,
                descriptor_gcs_uri=descriptor_gcs_uri,
                cfg=cfg,
            )
            if synthesis_result is None:
                continue
            _append_lane_result(
                selected_lane,
                {"lane": "synthesis_coverage_validation", **synthesis_result},
            )
            continue
        if selected_lane == "cosmos_single_capture_smoke":
            from .synthesis.cosmos_benchmark import run_cosmos_single_capture_smoke_lane

            capture_root = resume_root
            smoke_result = _run_lane_call(
                selected_lane,
                run_cosmos_single_capture_smoke_lane,
                capture_root=capture_root,
                descriptor_gcs_uri=descriptor_gcs_uri,
                cfg=cfg,
            )
            if smoke_result is None:
                continue
            _append_lane_result(
                selected_lane,
                {"lane": "cosmos_single_capture_smoke", **smoke_result},
            )
            continue
        log_event(
            logger,
            logging.ERROR,
            "capture_pipeline.unsupported_lane",
            descriptor_gcs_uri=descriptor_gcs_uri,
            selected_lane=selected_lane,
        )
        unsupported_error = ValueError(f"Unsupported pipeline lane: {selected_lane}")
        if not allow_lane_fault_isolation:
            raise unsupported_error
        _append_lane_failure(selected_lane, unsupported_error)

    parsed = parse_gs_uri(descriptor_gcs_uri)
    result = {
        "status": "completed_with_lane_failures" if lane_failures else "completed",
        "artifact_purpose": "evaluation_preparation",
        "robot_evaluation_performed": False,
        "descriptor_gcs_uri": descriptor_gcs_uri,
        "bucket": parsed.bucket,
        "lanes": lanes,
        "results": results,
    }
    if lane_failures:
        result["lane_failure_count"] = len(lane_failures)
    log_event(
        logger,
        logging.INFO,
        "capture_pipeline.completed",
        descriptor_gcs_uri=descriptor_gcs_uri,
        bucket=parsed.bucket,
        lanes=lanes,
        result_count=len(results),
        lane_failure_count=len(lane_failures),
    )
    return result


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
        from .task_evaluation_scene_retirement_generations import read_selected_capture_input_bytes
        descriptor = json.loads(read_selected_capture_input_bytes(descriptor_path))
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
    allow_legacy_lanes: bool = False,
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
        allow_legacy_lanes=allow_legacy_lanes,
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
    parser.add_argument(
        "--allow-legacy-lanes",
        action="store_true",
        help=(
            "Explicitly admit deprecated scene-memory/retrieval/alignment/synthesis/"
            "Cosmos lanes. Current product lanes do not require this flag."
        ),
    )
    args = parser.parse_args(argv)

    try:
        settings = PipelineSettings.from_env()
        cfg = PipelineConfig.from_settings(settings)
        if args.descriptor_gcs_uri:
            descriptor_path = resolve_gs_uri_to_path(args.descriptor_gcs_uri, cfg.gcs_root)
            if descriptor_path.exists() or not (args.bucket and args.scene_id and args.capture_id):
                run_capture_pipeline(
                    descriptor_gcs_uri=args.descriptor_gcs_uri,
                    lane=args.lane,
                    allow_legacy_lanes=bool(args.allow_legacy_lanes),
                    config=cfg,
                )
            else:
                run_capture_pipeline_for_capture(
                    bucket=args.bucket,
                    scene_id=args.scene_id,
                    capture_id=args.capture_id,
                    lane=args.lane,
                    allow_legacy_lanes=bool(args.allow_legacy_lanes),
                    config=cfg,
                )
        elif args.bucket and args.scene_id and args.capture_id:
            run_capture_pipeline_for_capture(
                bucket=args.bucket,
                scene_id=args.scene_id,
                capture_id=args.capture_id,
                lane=args.lane,
                allow_legacy_lanes=bool(args.allow_legacy_lanes),
                config=cfg,
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
