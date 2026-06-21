"""Lane-aware capture pipeline entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .common import PipelineError, parse_bool, parse_gs_uri, resolve_gs_uri_to_path
from .evaluation_prep_stage import run_evaluation_prep_stage
from .geometry_sources import load_capture_geometry
from .logging_utils import log_event
from .local_capture import resolve_local_capture_context
from .materialization import materialize_capture_bundle
from .qualification import run_qualification_pipeline
from .frame_alignment_stage import run_frame_alignment_stage
from .retrieval_index_stage import run_retrieval_index_stage
from .robot_eval_job_orchestrator import (
    REQUIRED_ROBOT_EVAL_INPUTS,
    run_robot_eval_job_request_inbox,
)
from .scenario_variation_instantiator import SCENARIO_VARIATION_NAMES
from .simulation_automation import (
    SIMULATOR_FRAMEWORKS,
    WORLD_MODEL_ENGINE_TARGETS,
    build_simulation_automation,
)
from .synthesis.synthesize import synthesize_view

logger = logging.getLogger(__name__)

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
_STANDARD_ROBOT_EVAL_SCORECARD_METRICS = (
    "success_rate",
    "cycle_time",
    "intervention_rate",
    "unsafe_proximity",
    "collision_risk",
    "object_drop",
    "wrong_object",
    "timeout",
    "recovery_success",
    "world_model_uncertainty",
    "sim_vs_real_calibration_score",
)
_STANDARD_SCENARIO_VARIATION_NAMES = tuple(SCENARIO_VARIATION_NAMES)
_STANDARD_SIMULATOR_ENGINE_NAMES = tuple(SIMULATOR_FRAMEWORKS)
_STANDARD_WORLD_MODEL_ENGINE_NAMES = tuple(WORLD_MODEL_ENGINE_TARGETS)
_STANDARD_SIMULATOR_PLUGIN_REQUIRED_INPUT_KEYS = (
    "simulation_automation_plan",
    "asset_conversion_plan",
    "scenario_variation_instances",
    "episode_spec",
    "cpu_preflight_manifest",
)
_SIMULATOR_PLUGIN_ENGINE_REQUIRED_INPUT_KEYS = {
    "isaac_lab_arena": ("arena_environment_packet",),
}
_STANDARD_WORLD_MODEL_PLUGIN_REQUIRED_INPUT_KEYS = (
    "simulation_automation_plan",
    "scenario_variation_instances",
    "site_card",
    "task_cards",
    "scenario_cards",
)
_WORLD_MODEL_PLUGIN_OPTIONAL_INPUT_KEYS = {
    "world_manifest",
    "simready_bridge",
    "gpu_handoff_packet",
    "dense_world_model_export",
    "site_reference_projection",
}
_OPTIONAL_MISSING_SOURCE_STATUSES = {"", "missing", "not_available", "optional_missing"}
SIM_ONLY_BETA_AUTONOMY_ENV = "BLUEPRINT_SIM_ONLY_BETA_AUTONOMY"
SIM_ONLY_BETA_DEFAULT_TASK_EVAL_ENV = "BLUEPRINT_SIM_ONLY_BETA_DEFAULT_TASK_EVAL"
ALLOW_SIMULATOR_EXECUTION_ENV = "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"
MUJOCO_G1_MODEL_ROOT_ENV = "BLUEPRINT_MUJOCO_G1_MODEL_ROOT"
MUJOCO_ALLOW_FETCH_G1_ASSETS_ENV = "BLUEPRINT_MUJOCO_ALLOW_FETCH_G1_ASSETS"
MUJOCO_BETA_STEPS_ENV = "BLUEPRINT_MUJOCO_BETA_STEPS"
MUJOCO_BETA_SKIP_RENDER_ENV = "BLUEPRINT_MUJOCO_BETA_SKIP_RENDER_FRAMES"


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


def _read_json_mapping(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _safe_slug(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip().lower()
    cleaned = "".join(char if char.isalnum() else "-" for char in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or fallback


def _descriptor_requested_outputs(raw_payload: Mapping[str, Any]) -> set[str]:
    raw_requested_outputs = raw_payload.get("requested_outputs") or raw_payload.get("requestedOutputs")
    if isinstance(raw_requested_outputs, str):
        values = [raw_requested_outputs]
    elif isinstance(raw_requested_outputs, (list, tuple, set)):
        values = [str(value) for value in raw_requested_outputs]
    else:
        values = []
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _descriptor_requests_task_evaluation_run(descriptor_path: Path) -> bool:
    payload = _read_json_mapping(descriptor_path)
    return "task_evaluation_run" in _descriptor_requested_outputs(payload)


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _sim_only_beta_autonomy_enabled() -> bool:
    return _env_truthy(SIM_ONLY_BETA_AUTONOMY_ENV)


def _sim_only_beta_default_task_eval_enabled() -> bool:
    return _sim_only_beta_autonomy_enabled() or _env_truthy(SIM_ONLY_BETA_DEFAULT_TASK_EVAL_ENV)


def _call_requests_task_evaluation_run(
    *,
    lane: Optional[str],
    requested_lanes: Optional[List[str]],
) -> bool:
    requested = [lane] if lane else []
    requested.extend(requested_lanes or [])
    return any(str(value).strip().lower() == "task_evaluation_run" for value in requested)


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


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _mujoco_beta_simulator_command(capture_root: Path) -> str:
    explicit = str(os.getenv("ROBOT_EVAL_JOB_DEFAULT_SIMULATOR_COMMAND") or "").strip()
    if explicit:
        return explicit
    command = [
        sys.executable,
        "-m",
        "blueprint_pipeline.mujoco_g1_simulator_command",
        "--capture-root",
        str(capture_root),
        "--steps",
        str(_env_int(MUJOCO_BETA_STEPS_ENV, 32)),
    ]
    g1_root = str(os.getenv(MUJOCO_G1_MODEL_ROOT_ENV) or "").strip()
    if g1_root:
        command.extend(["--g1-model-root", g1_root])
    elif _env_truthy(MUJOCO_ALLOW_FETCH_G1_ASSETS_ENV):
        command.append("--allow-fetch-g1-assets")
    else:
        return ""
    if _env_truthy(MUJOCO_BETA_SKIP_RENDER_ENV):
        command.append("--skip-render-frames")
    return " ".join(shlex.quote(part) for part in command)


def _default_robot_eval_job_runtime(capture_root: Path) -> Dict[str, Any]:
    simulator = os.getenv("ROBOT_EVAL_JOB_DEFAULT_SIMULATOR")
    provisioner = os.getenv("ROBOT_EVAL_JOB_DEFAULT_PROVISIONER")
    if _sim_only_beta_autonomy_enabled():
        simulator = simulator or "mujoco"
        provisioner = provisioner or "fixture_local"
    else:
        simulator = simulator or "fixture"
        provisioner = provisioner or "fixture_local"

    allowed_simulators: List[str] = []
    raw_allowed = os.getenv("ROBOT_EVAL_JOB_ALLOWED_SIMULATORS")
    if raw_allowed:
        allowed_simulators = [
            item.strip()
            for item in raw_allowed.replace(";", ",").split(",")
            if item.strip()
        ]
    elif _sim_only_beta_autonomy_enabled() and simulator != "fixture":
        allowed_simulators = [simulator]

    simulator_commands: Dict[str, str] = {}
    if _sim_only_beta_autonomy_enabled() and simulator == "mujoco":
        command = _mujoco_beta_simulator_command(capture_root)
        if command:
            simulator_commands["mujoco"] = command

    return {
        "provisioner": provisioner,
        "simulator": simulator,
        "allow_simulator_execution": (
            _sim_only_beta_autonomy_enabled()
            and _env_truthy(ALLOW_SIMULATOR_EXECUTION_ENV)
            and simulator != "fixture"
        ),
        "allowed_simulators": allowed_simulators,
        "simulator_commands": simulator_commands,
        "allow_cpu_simulator_preflight": _sim_only_beta_autonomy_enabled()
        and _env_truthy("BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT"),
    }


def _load_cards(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json_mapping(path)
    cards = payload.get("cards")
    if not isinstance(cards, list):
        return []
    return [dict(card) for card in cards if isinstance(card, Mapping)]


def _metric_ids_from_methodology(path: Path) -> set[str]:
    payload = _read_json_mapping(path)
    raw_metrics = (
        payload.get("metrics")
        or payload.get("scorecard_metrics")
        or payload.get("standard_scorecard_metrics")
    )
    metric_ids: set[str] = set()
    if isinstance(raw_metrics, Mapping):
        for key, value in raw_metrics.items():
            if isinstance(value, Mapping):
                metric_id = value.get("metric_id") or value.get("metricId") or value.get("id")
            else:
                metric_id = key
            metric_id = str(metric_id or "").strip()
            if metric_id:
                metric_ids.add(metric_id)
        return metric_ids
    if not isinstance(raw_metrics, list):
        return metric_ids
    for metric in raw_metrics:
        if isinstance(metric, Mapping):
            metric_id = metric.get("metric_id") or metric.get("metricId") or metric.get("id")
        else:
            metric_id = metric
        metric_id = str(metric_id or "").strip()
        if metric_id:
            metric_ids.add(metric_id)
    return metric_ids


def _task_ids_from_thresholds(path: Path) -> set[str]:
    payload = _read_json_mapping(path)
    raw_thresholds = (
        payload.get("task_thresholds")
        or payload.get("taskThresholds")
        or payload.get("thresholds")
    )
    task_ids: set[str] = set()
    if isinstance(raw_thresholds, Mapping):
        for key, value in raw_thresholds.items():
            if isinstance(value, Mapping):
                task_id = value.get("task_id") or value.get("taskId") or value.get("id")
            else:
                task_id = key
            task_id = str(task_id or "").strip()
            if task_id:
                task_ids.add(task_id)
        return task_ids
    if not isinstance(raw_thresholds, list):
        return task_ids
    for threshold in raw_thresholds:
        if not isinstance(threshold, Mapping):
            continue
        task_id = threshold.get("task_id") or threshold.get("taskId") or threshold.get("id")
        task_id = str(task_id or "").strip()
        if task_id:
            task_ids.add(task_id)
    return task_ids


def _failure_mode_ids_from_taxonomy(path: Path) -> set[str]:
    payload = _read_json_mapping(path)
    raw_modes = payload.get("failure_modes") or payload.get("failureModes")
    mode_ids: set[str] = set()
    if not isinstance(raw_modes, list):
        return mode_ids
    for mode in raw_modes:
        if isinstance(mode, Mapping):
            mode_id = mode.get("failure_mode_id") or mode.get("failureModeId") or mode.get("id")
        else:
            mode_id = mode
        mode_id = str(mode_id or "").strip()
        if mode_id:
            mode_ids.add(mode_id)
    return mode_ids


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _relative_to(base: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _local_artifact(path: Path, *, base_dir: Path) -> Dict[str, Any]:
    return {
        "path": _relative_to(base_dir, path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def _automation_local_reference_path(
    value: Any,
    *,
    capture_root: Path,
    automation_root: Path,
) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith("file://"):
        return Path(text[7:]).expanduser()
    if text.startswith("gs://"):
        default_gcs_root = capture_root.parents[3] if len(capture_root.parents) > 3 else capture_root
        return resolve_gs_uri_to_path(text, Path(os.getenv("GCS_ROOT", str(default_gcs_root))))
    if "://" in text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    automation_candidate = automation_root / path
    if automation_candidate.exists():
        return automation_candidate
    if path.parts and path.parts[0] in {"pipeline", "raw", "privacy"}:
        return capture_root / path
    return automation_candidate


def _string_list(values: Any) -> List[str]:
    if isinstance(values, str):
        return [values] if values.strip() else []
    if not isinstance(values, (list, tuple, set)):
        return []
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _scenario_family_rows(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json_mapping(path)
    rows = payload.get("families") or payload.get("scenario_families")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _scenario_family_variation_names(family: Mapping[str, Any]) -> set[str]:
    variations = family.get("variations")
    if not isinstance(variations, list):
        return set()
    names: set[str] = set()
    for variation in variations:
        if not isinstance(variation, Mapping):
            continue
        variation_name = (
            variation.get("variation_name")
            or variation.get("variationName")
            or variation.get("variation_id")
            or variation.get("variationId")
        )
        variation_name = str(variation_name or "").strip()
        if variation_name:
            names.add(variation_name)
    return names


def _variation_instance_rows(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json_mapping(path)
    rows = payload.get("instances") or payload.get("scenario_variation_instances")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _validate_scenario_variation_artifacts(
    *,
    capture_root: Path,
    requested_tasks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    robot_eval_root = capture_root / "pipeline" / "robot_eval_dataset"
    automation_root = capture_root / "pipeline" / "simulation_automation"
    scenario_family_path = robot_eval_root / "scenario_family_library.json"
    variation_instances_path = automation_root / "scenario_variation_instances.json"
    required_variation_names = list(_STANDARD_SCENARIO_VARIATION_NAMES)
    requested_task_ids = [
        str(task.get("task_id") or "").strip()
        for task in requested_tasks
        if str(task.get("task_id") or "").strip()
    ]
    requested_scenario_ids = sorted(
        {
            str(scenario_id or "").strip()
            for task in requested_tasks
            for scenario_id in task.get("scenario_ids", [])
            if str(scenario_id or "").strip()
        }
    )
    blockers: List[str] = []
    family_rows: List[Dict[str, Any]] = []
    missing_family_task_ids: List[str] = []
    missing_family_scenario_ids: List[str] = []
    missing_family_variations_by_family: List[Dict[str, Any]] = []
    if not scenario_family_path.is_file():
        blockers.append("robot_eval_scenario_family_library_missing")
    else:
        family_rows = _scenario_family_rows(scenario_family_path)
        if not family_rows:
            blockers.append("robot_eval_scenario_family_library_empty")
        family_task_ids = {
            str(row.get("task_id") or row.get("taskId") or "").strip()
            for row in family_rows
            if str(row.get("task_id") or row.get("taskId") or "").strip()
        }
        family_scenario_ids = {
            str(row.get("scenario_id") or row.get("scenarioId") or "").strip()
            for row in family_rows
            if str(row.get("scenario_id") or row.get("scenarioId") or "").strip()
        }
        missing_family_task_ids = [
            task_id for task_id in requested_task_ids if task_id not in family_task_ids
        ]
        missing_family_scenario_ids = [
            scenario_id
            for scenario_id in requested_scenario_ids
            if scenario_id not in family_scenario_ids
        ]
        if missing_family_task_ids:
            blockers.append("robot_eval_scenario_family_library_missing_task_coverage")
        if missing_family_scenario_ids:
            blockers.append("robot_eval_scenario_family_library_missing_scenario_coverage")
        for family in family_rows:
            present = _scenario_family_variation_names(family)
            missing = [
                variation_name
                for variation_name in required_variation_names
                if variation_name not in present
            ]
            if missing:
                missing_family_variations_by_family.append(
                    {
                        "family_id": str(family.get("family_id") or family.get("familyId") or ""),
                        "task_id": str(family.get("task_id") or family.get("taskId") or ""),
                        "scenario_id": str(
                            family.get("scenario_id") or family.get("scenarioId") or ""
                        ),
                        "missing_variation_names": missing,
                    }
                )
        if missing_family_variations_by_family:
            blockers.append("robot_eval_scenario_family_library_missing_required_variations")

    variation_payload = _read_json_mapping(variation_instances_path)
    variation_rows = _variation_instance_rows(variation_instances_path)
    missing_variation_names: List[str] = []
    missing_variation_names_by_scenario: List[Dict[str, Any]] = []
    if not variation_instances_path.is_file():
        blockers.append("robot_eval_scenario_variation_instances_missing")
    else:
        if variation_payload.get("status") != "completed":
            blockers.append("robot_eval_scenario_variation_instances_not_completed")
        if not variation_rows:
            blockers.append("robot_eval_scenario_variation_instances_empty")
        instantiated_variation_names = _string_list(
            variation_payload.get("variation_names_instantiated")
            or variation_payload.get("variationNamesInstantiated")
        )
        if not instantiated_variation_names:
            instantiated_variation_names = _string_list(
                [
                    row.get("variation_name")
                    or row.get("variationName")
                    or row.get("variation_id")
                    or row.get("variationId")
                    for row in variation_rows
                ]
            )
        missing_variation_names = [
            variation_name
            for variation_name in required_variation_names
            if variation_name not in set(instantiated_variation_names)
        ]
        if missing_variation_names:
            blockers.append("robot_eval_scenario_variation_instances_missing_required_variations")
        names_by_scenario: Dict[str, set[str]] = {}
        for row in variation_rows:
            scenario_id = str(row.get("scenario_id") or row.get("scenarioId") or "").strip()
            variation_name = (
                row.get("variation_name")
                or row.get("variationName")
                or row.get("variation_id")
                or row.get("variationId")
            )
            variation_name = str(variation_name or "").strip()
            if scenario_id and variation_name:
                names_by_scenario.setdefault(scenario_id, set()).add(variation_name)
        for scenario_id in requested_scenario_ids:
            present = names_by_scenario.get(scenario_id, set())
            missing = [
                variation_name
                for variation_name in required_variation_names
                if variation_name not in present
            ]
            if missing:
                missing_variation_names_by_scenario.append(
                    {
                        "scenario_id": scenario_id,
                        "missing_variation_names": missing,
                    }
                )
        if missing_variation_names_by_scenario:
            blockers.append(
                "robot_eval_scenario_variation_instances_missing_required_variations_per_scenario"
            )

    return {
        "blockers": blockers,
        "scenario_family_library_path": str(scenario_family_path),
        "scenario_variation_instances_path": str(variation_instances_path),
        "required_scenario_variation_names": required_variation_names,
        "missing_scenario_family_task_ids": missing_family_task_ids,
        "missing_scenario_family_scenario_ids": missing_family_scenario_ids,
        "missing_scenario_family_variations_by_family": missing_family_variations_by_family,
        "missing_scenario_variation_names": missing_variation_names,
        "missing_scenario_variation_names_by_scenario": missing_variation_names_by_scenario,
    }


def _validate_robot_eval_dataset_required_inputs(*, capture_root: Path) -> Dict[str, Any]:
    pipeline_root = capture_root / "pipeline"
    missing_inputs: List[str] = []
    artifacts: Dict[str, Dict[str, Any]] = {}
    for key, relative_path in REQUIRED_ROBOT_EVAL_INPUTS.items():
        path = pipeline_root / relative_path
        artifacts[key] = _local_artifact(path, base_dir=pipeline_root)
        if not path.is_file():
            missing_inputs.append(key)
    return {
        "blockers": [f"{key}_missing" for key in missing_inputs],
        "missing_robot_eval_dataset_inputs": missing_inputs,
        "robot_eval_dataset_input_artifacts": artifacts,
    }


def _required_simulator_plugin_input_keys(engine: str) -> List[str]:
    required = list(_STANDARD_SIMULATOR_PLUGIN_REQUIRED_INPUT_KEYS)
    required.extend(_SIMULATOR_PLUGIN_ENGINE_REQUIRED_INPUT_KEYS.get(engine, ()))
    return required


def _required_world_model_plugin_input_keys(_engine: str) -> List[str]:
    return list(_STANDARD_WORLD_MODEL_PLUGIN_REQUIRED_INPUT_KEYS)


def _plugin_local_input_artifact_audit(
    *,
    inputs: Mapping[str, Any],
    capture_root: Path,
    automation_root: Path,
    optional_input_keys: set[str] | None = None,
    optional_missing_source: bool = False,
) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
    optional_keys = optional_input_keys or set()
    local_artifacts: Dict[str, Dict[str, Any]] = {}
    missing_local_input_keys: List[str] = []
    for key, value in sorted(inputs.items()):
        local_path = _automation_local_reference_path(
            value,
            capture_root=capture_root,
            automation_root=automation_root,
        )
        if local_path is None:
            continue
        artifact = _local_artifact(local_path, base_dir=automation_root)
        local_artifacts[key] = artifact
        if not artifact["exists"] and not (optional_missing_source and key in optional_keys):
            missing_local_input_keys.append(key)
    return local_artifacts, missing_local_input_keys


def _validate_simulator_engine_plugin_registry(*, capture_root: Path) -> Dict[str, Any]:
    automation_root = capture_root / "pipeline" / "simulation_automation"
    registry_path = automation_root / "simulator_engine_plugin_registry.json"
    payload = _read_json_mapping(registry_path)
    plugins = _as_mapping(payload.get("plugins"))
    world_model_plugins = _as_mapping(payload.get("world_model_plugins"))
    required_plugins = set(_STANDARD_SIMULATOR_ENGINE_NAMES)
    required_world_model_plugins = set(_STANDARD_WORLD_MODEL_ENGINE_NAMES)
    present_plugins = set(plugins)
    present_world_model_plugins = set(world_model_plugins)
    missing_simulator_plugins = sorted(required_plugins - present_plugins)
    missing_world_model_plugins = sorted(required_world_model_plugins - present_world_model_plugins)
    unready_simulator_plugins: List[Dict[str, Any]] = []
    unready_world_model_plugins: List[Dict[str, Any]] = []
    missing_simulator_input_keys: Dict[str, List[str]] = {}
    missing_world_model_input_keys: Dict[str, List[str]] = {}
    simulator_local_input_artifacts: Dict[str, Dict[str, Dict[str, Any]]] = {}
    world_model_local_input_artifacts: Dict[str, Dict[str, Dict[str, Any]]] = {}
    missing_simulator_local_inputs: Dict[str, List[str]] = {}
    missing_world_model_local_inputs: Dict[str, List[str]] = {}
    for engine in sorted(required_plugins & present_plugins):
        plugin = _as_mapping(plugins.get(engine))
        inputs = _as_mapping(plugin.get("inputs"))
        missing_required_input_keys = [
            key
            for key in _required_simulator_plugin_input_keys(engine)
            if not str(inputs.get(key) or "").strip()
        ]
        if missing_required_input_keys:
            missing_simulator_input_keys[engine] = missing_required_input_keys
        local_artifacts, missing_local_input_keys = _plugin_local_input_artifact_audit(
            inputs=inputs,
            capture_root=capture_root,
            automation_root=automation_root,
        )
        if local_artifacts:
            simulator_local_input_artifacts[engine] = local_artifacts
        if missing_local_input_keys:
            missing_simulator_local_inputs[engine] = missing_local_input_keys
        if plugin.get("adapter_contract_status") != "ready" or not bool(
            plugin.get("managed_execution_supported")
        ):
            unready_simulator_plugins.append(
                {
                    "engine": engine,
                    "adapter_contract_status": plugin.get("adapter_contract_status"),
                    "managed_execution_supported": bool(
                        plugin.get("managed_execution_supported")
                    ),
                }
            )
    for engine in sorted(required_world_model_plugins & present_world_model_plugins):
        plugin = _as_mapping(world_model_plugins.get(engine))
        inputs = _as_mapping(plugin.get("inputs"))
        missing_required_input_keys = [
            key
            for key in _required_world_model_plugin_input_keys(engine)
            if not str(inputs.get(key) or "").strip()
        ]
        if missing_required_input_keys:
            missing_world_model_input_keys[engine] = missing_required_input_keys
        source_status = str(plugin.get("source_status") or "").strip().lower()
        local_artifacts, missing_local_input_keys = _plugin_local_input_artifact_audit(
            inputs=inputs,
            capture_root=capture_root,
            automation_root=automation_root,
            optional_input_keys=_WORLD_MODEL_PLUGIN_OPTIONAL_INPUT_KEYS,
            optional_missing_source=source_status in _OPTIONAL_MISSING_SOURCE_STATUSES,
        )
        if local_artifacts:
            world_model_local_input_artifacts[engine] = local_artifacts
        if missing_local_input_keys:
            missing_world_model_local_inputs[engine] = missing_local_input_keys
        if plugin.get("adapter_contract_status") != "ready" or not bool(
            plugin.get("managed_execution_supported")
        ):
            unready_world_model_plugins.append(
                {
                    "engine": engine,
                    "adapter_contract_status": plugin.get("adapter_contract_status"),
                    "managed_execution_supported": bool(
                        plugin.get("managed_execution_supported")
                    ),
                }
            )

    blockers: List[str] = []
    if not registry_path.is_file():
        blockers.append("robot_eval_simulator_engine_plugin_registry_missing")
    elif not plugins:
        blockers.append("robot_eval_simulator_engine_plugin_registry_empty")
    if missing_simulator_plugins:
        blockers.append("robot_eval_simulator_engine_plugin_registry_missing_required_engines")
    if missing_world_model_plugins:
        blockers.append(
            "robot_eval_simulator_engine_plugin_registry_missing_required_world_model_engines"
        )
    if unready_simulator_plugins:
        blockers.append("robot_eval_simulator_engine_plugins_not_ready")
    if unready_world_model_plugins:
        blockers.append("robot_eval_world_model_engine_plugins_not_ready")
    if missing_simulator_input_keys:
        blockers.append("robot_eval_simulator_engine_plugin_registry_missing_required_input_keys")
    if missing_world_model_input_keys:
        blockers.append("robot_eval_world_model_engine_plugin_registry_missing_required_input_keys")
    if missing_simulator_local_inputs:
        blockers.append("robot_eval_simulator_engine_plugin_registry_missing_local_input_artifacts")
    if missing_world_model_local_inputs:
        blockers.append("robot_eval_world_model_engine_plugin_registry_missing_local_input_artifacts")

    return {
        "blockers": blockers,
        "simulator_engine_plugin_registry_path": str(registry_path),
        "required_simulator_plugins": sorted(required_plugins),
        "required_world_model_plugins": sorted(required_world_model_plugins),
        "simulator_plugins": sorted(present_plugins),
        "world_model_plugins": sorted(present_world_model_plugins),
        "missing_simulator_plugins": missing_simulator_plugins,
        "missing_world_model_plugins": missing_world_model_plugins,
        "unready_simulator_plugins": unready_simulator_plugins,
        "unready_world_model_plugins": unready_world_model_plugins,
        "missing_simulator_plugin_input_keys": missing_simulator_input_keys,
        "missing_world_model_plugin_input_keys": missing_world_model_input_keys,
        "simulator_plugin_local_input_artifacts": simulator_local_input_artifacts,
        "world_model_plugin_local_input_artifacts": world_model_local_input_artifacts,
        "missing_simulator_plugin_local_inputs": missing_simulator_local_inputs,
        "missing_world_model_plugin_local_inputs": missing_world_model_local_inputs,
        "simulator_plugin_count": len(plugins),
        "world_model_plugin_count": len(world_model_plugins),
    }


def _auto_stage_robot_eval_job_request(
    *,
    capture_root: Path,
    descriptor_path: Path,
) -> Dict[str, Any]:
    descriptor = _read_json_mapping(descriptor_path)
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    site_identity = (
        metadata.get("site_identity") if isinstance(metadata.get("site_identity"), Mapping) else {}
    )
    scene_id = (
        descriptor.get("scene_id")
        or metadata.get("scene_id")
        or capture_root.parent.parent.name
        if len(capture_root.parents) > 1
        else "scene"
    )
    capture_id = descriptor.get("capture_id") or metadata.get("capture_id") or capture_root.name
    site_id = (
        site_identity.get("site_id")
        or descriptor.get("site_id")
        or _safe_slug(scene_id, fallback="site")
    )
    robot_eval_root = capture_root / "pipeline" / "robot_eval_dataset"
    task_cards_path = robot_eval_root / "task_cards.json"
    scenario_cards_path = robot_eval_root / "scenario_cards.json"
    task_thresholds_path = robot_eval_root / "task_thresholds.json"
    scoring_methodology_path = robot_eval_root / "scoring_methodology.json"
    failure_taxonomy_path = robot_eval_root / "failure_taxonomy.json"
    task_cards = _load_cards(task_cards_path)
    scenario_cards = _load_cards(scenario_cards_path)
    blockers: List[str] = []
    if not task_cards:
        blockers.append("robot_eval_task_cards_missing")
    if not scenario_cards:
        blockers.append("robot_eval_scenario_cards_missing")
    scenario_ids_by_task: Dict[str, List[str]] = {}
    for scenario in scenario_cards:
        task_id = str(scenario.get("task_id") or "").strip()
        scenario_id = str(scenario.get("scenario_id") or "").strip()
        if task_id and scenario_id:
            scenario_ids_by_task.setdefault(task_id, []).append(scenario_id)
    requested_tasks: List[Dict[str, Any]] = []
    task_ids_without_scenarios: List[str] = []
    for task in task_cards:
        task_id = str(task.get("task_id") or "").strip()
        if not task_id:
            continue
        scenario_ids = scenario_ids_by_task.get(task_id, [])
        if not scenario_ids:
            task_ids_without_scenarios.append(task_id)
        requested_tasks.append(
            {
                "task_id": task_id,
                "scenario_ids": scenario_ids,
                "task_thresholds_uri": "pipeline/robot_eval_dataset/task_thresholds.json",
            }
        )
    if task_ids_without_scenarios:
        blockers.append("robot_eval_task_scenario_links_missing")
    requested_task_ids = [task["task_id"] for task in requested_tasks]
    missing_threshold_task_ids: List[str] = []
    if not task_thresholds_path.is_file():
        blockers.append("robot_eval_task_thresholds_missing")
    else:
        threshold_task_ids = _task_ids_from_thresholds(task_thresholds_path)
        missing_threshold_task_ids = [
            task_id for task_id in requested_task_ids if task_id not in threshold_task_ids
        ]
        if missing_threshold_task_ids:
            blockers.append("robot_eval_task_thresholds_missing_requested_tasks")
    missing_scorecard_metrics: List[str] = []
    if not scoring_methodology_path.is_file():
        blockers.append("robot_eval_scoring_methodology_missing")
    else:
        metric_ids = _metric_ids_from_methodology(scoring_methodology_path)
        missing_scorecard_metrics = [
            metric_id
            for metric_id in _STANDARD_ROBOT_EVAL_SCORECARD_METRICS
            if metric_id not in metric_ids
        ]
        if missing_scorecard_metrics:
            blockers.append("robot_eval_scoring_methodology_missing_standard_metrics")
    scenario_variation_validation: Dict[str, Any] = {
        "blockers": [],
        "missing_scenario_family_task_ids": [],
        "missing_scenario_family_scenario_ids": [],
        "missing_scenario_family_variations_by_family": [],
        "missing_scenario_variation_names": [],
        "missing_scenario_variation_names_by_scenario": [],
        "required_scenario_variation_names": list(_STANDARD_SCENARIO_VARIATION_NAMES),
    }
    simulator_plugin_validation: Dict[str, Any] = {
        "blockers": [],
        "missing_simulator_plugins": [],
        "missing_world_model_plugins": [],
        "unready_simulator_plugins": [],
        "unready_world_model_plugins": [],
        "missing_simulator_plugin_input_keys": {},
        "missing_world_model_plugin_input_keys": {},
        "missing_simulator_plugin_local_inputs": {},
        "missing_world_model_plugin_local_inputs": {},
        "simulator_plugin_count": 0,
        "world_model_plugin_count": 0,
        "required_simulator_plugins": list(_STANDARD_SIMULATOR_ENGINE_NAMES),
        "required_world_model_plugins": list(_STANDARD_WORLD_MODEL_ENGINE_NAMES),
    }
    robot_eval_dataset_validation: Dict[str, Any] = {
        "blockers": [],
        "missing_robot_eval_dataset_inputs": [],
        "robot_eval_dataset_input_artifacts": {},
    }
    failure_taxonomy_mode_count = 0
    if (
        task_cards
        and scenario_cards
        and not missing_threshold_task_ids
        and not missing_scorecard_metrics
        and task_thresholds_path.is_file()
        and scoring_methodology_path.is_file()
    ):
        scenario_variation_validation = _validate_scenario_variation_artifacts(
            capture_root=capture_root,
            requested_tasks=requested_tasks,
        )
        blockers.extend(scenario_variation_validation.get("blockers", []))
        if not failure_taxonomy_path.is_file():
            blockers.append("robot_eval_failure_taxonomy_missing")
        else:
            failure_taxonomy_mode_count = len(_failure_mode_ids_from_taxonomy(failure_taxonomy_path))
            if failure_taxonomy_mode_count <= 0:
                blockers.append("robot_eval_failure_taxonomy_empty")
        if not blockers:
            robot_eval_dataset_validation = _validate_robot_eval_dataset_required_inputs(
                capture_root=capture_root,
            )
            blockers.extend(robot_eval_dataset_validation.get("blockers", []))
        if not blockers:
            simulator_plugin_validation = _validate_simulator_engine_plugin_registry(
                capture_root=capture_root,
            )
            blockers.extend(simulator_plugin_validation.get("blockers", []))
    if blockers:
        return {
            "status": "blocked",
            "job_id": None,
            "inbox_dir": str(capture_root / "pipeline" / "robot_eval_job_requests" / "inbox"),
            "request_path": None,
            "blockers": blockers,
            "missing_task_scenario_links": task_ids_without_scenarios,
            "missing_threshold_task_ids": missing_threshold_task_ids,
            "missing_scorecard_metrics": missing_scorecard_metrics,
            "standard_scorecard_metrics": list(_STANDARD_ROBOT_EVAL_SCORECARD_METRICS),
            "missing_robot_eval_dataset_inputs": robot_eval_dataset_validation.get(
                "missing_robot_eval_dataset_inputs",
                [],
            ),
            "robot_eval_dataset_input_artifacts": robot_eval_dataset_validation.get(
                "robot_eval_dataset_input_artifacts",
                {},
            ),
            "required_scenario_variation_names": scenario_variation_validation.get(
                "required_scenario_variation_names",
                list(_STANDARD_SCENARIO_VARIATION_NAMES),
            ),
            "missing_scenario_family_task_ids": scenario_variation_validation.get(
                "missing_scenario_family_task_ids",
                [],
            ),
            "missing_scenario_family_scenario_ids": scenario_variation_validation.get(
                "missing_scenario_family_scenario_ids",
                [],
            ),
            "missing_scenario_family_variations_by_family": scenario_variation_validation.get(
                "missing_scenario_family_variations_by_family",
                [],
            ),
            "missing_scenario_variation_names": scenario_variation_validation.get(
                "missing_scenario_variation_names",
                [],
            ),
            "missing_scenario_variation_names_by_scenario": scenario_variation_validation.get(
                "missing_scenario_variation_names_by_scenario",
                [],
            ),
            "required_simulator_plugins": simulator_plugin_validation.get(
                "required_simulator_plugins",
                list(_STANDARD_SIMULATOR_ENGINE_NAMES),
            ),
            "required_world_model_plugins": simulator_plugin_validation.get(
                "required_world_model_plugins",
                list(_STANDARD_WORLD_MODEL_ENGINE_NAMES),
            ),
            "missing_simulator_plugins": simulator_plugin_validation.get(
                "missing_simulator_plugins",
                [],
            ),
            "missing_world_model_plugins": simulator_plugin_validation.get(
                "missing_world_model_plugins",
                [],
            ),
            "unready_simulator_plugins": simulator_plugin_validation.get(
                "unready_simulator_plugins",
                [],
            ),
            "unready_world_model_plugins": simulator_plugin_validation.get(
                "unready_world_model_plugins",
                [],
            ),
            "missing_simulator_plugin_input_keys": simulator_plugin_validation.get(
                "missing_simulator_plugin_input_keys",
                {},
            ),
            "missing_world_model_plugin_input_keys": simulator_plugin_validation.get(
                "missing_world_model_plugin_input_keys",
                {},
            ),
            "missing_simulator_plugin_local_inputs": simulator_plugin_validation.get(
                "missing_simulator_plugin_local_inputs",
                {},
            ),
            "missing_world_model_plugin_local_inputs": simulator_plugin_validation.get(
                "missing_world_model_plugin_local_inputs",
                {},
            ),
            "simulator_plugin_local_input_artifacts": simulator_plugin_validation.get(
                "simulator_plugin_local_input_artifacts",
                {},
            ),
            "world_model_plugin_local_input_artifacts": simulator_plugin_validation.get(
                "world_model_plugin_local_input_artifacts",
                {},
            ),
            "task_cards_path": str(task_cards_path),
            "scenario_cards_path": str(scenario_cards_path),
            "task_thresholds_path": str(task_thresholds_path),
            "scoring_methodology_path": str(scoring_methodology_path),
            "failure_taxonomy_path": str(failure_taxonomy_path),
            "scenario_family_library_path": scenario_variation_validation.get(
                "scenario_family_library_path",
            ),
            "scenario_variation_instances_path": scenario_variation_validation.get(
                "scenario_variation_instances_path",
            ),
            "simulator_engine_plugin_registry_path": simulator_plugin_validation.get(
                "simulator_engine_plugin_registry_path",
            ),
            "task_card_count": len(task_cards),
            "scenario_card_count": len(scenario_cards),
            "failure_taxonomy_mode_count": failure_taxonomy_mode_count,
            "simulator_plugin_count": simulator_plugin_validation.get(
                "simulator_plugin_count",
                0,
            ),
            "world_model_plugin_count": simulator_plugin_validation.get(
                "world_model_plugin_count",
                0,
            ),
            "claim_boundary": (
                "task_eval_auto_stage_requires_complete_dataset_scenario_scorecard_plugin_artifacts"
            ),
        }
    first_scenario = scenario_cards[0] if scenario_cards else {}
    robot_profile_id = str(
        first_scenario.get("robot_profile_id")
        or first_scenario.get("robotProfileId")
        or "mobile_manipulator_rgb_v1"
    )
    job_id = (
        f"auto-task-eval-{_safe_slug(scene_id, fallback='scene')}-"
        f"{_safe_slug(capture_id, fallback='capture')}"
    )
    buyer_request_id = f"buyer-request-{job_id}"
    beta_runtime = _default_robot_eval_job_runtime(capture_root)
    request = {
        "schema_version": "robot_eval_job_request.v1",
        "job_id": job_id,
        "buyer_request_id": buyer_request_id,
        "customer": {
            "id": "blueprint-auto-eval-baseline",
            "name": "Blueprint auto-staged baseline evaluation",
            "contact_email": None,
        },
        "site_package": {
            "site_id": site_id,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "capture_root": str(capture_root.resolve()),
            "buyer_request_id": buyer_request_id,
            "package_uri": "pipeline/robot_eval_dataset/robot_eval_dataset_manifest.json",
        },
        "requested_tasks": requested_tasks,
        "robot_profile": {
            "robot_profile_id": robot_profile_id,
            "embodiment": "mobile_manipulator",
            "sensors": ["rgb", "depth"],
        },
        "policy_package": {
            "high_level_skill_trace": {
                "skill_taxonomy_version": "blueprint_baseline_skill_trace.v1",
                "ordered_skill_sequence": [
                    "inspect_site_package",
                    "navigate_to_task_anchor",
                    "execute_task",
                    "report_outcome",
                ],
                "preconditions_postconditions": (
                    "Pre: task/scenario cards and generated robot POV are available. "
                    "Post: baseline reference trace is scored with proof boundaries."
                ),
                "failure_labels": "use_pipeline_failure_taxonomy",
                "source_type": "blueprint_default_baseline_trace",
                "confidence_coverage_note": (
                    "Auto-staged baseline for harness automation; not a robot-team "
                    "policy or deployment-readiness claim."
                ),
            }
        },
        "operation": "evaluate_only",
        "simulator_preference": beta_runtime["simulator"],
        "cosmos_training_preference": {"mode": "export_only"},
        "budget": {"budget_usd": 0.0, "timeout_seconds": 120},
        "rights_privacy_scope": {
            "status": "review_required",
            "external_use_allowed": True,
            "privacy_scope": "derived_deidentified_environment",
        },
        "owner_system": {
            "name": "BlueprintCapturePipeline",
            "request_id": job_id,
            "buyer_request_id": buyer_request_id,
            "capture_id": capture_id,
        },
        "source": {
            "system": "BlueprintCapturePipeline.auto_stage",
            "capture_descriptor": str(descriptor_path.resolve()),
            "requested_outputs": sorted(_descriptor_requested_outputs(descriptor)),
            "task_cards": "pipeline/robot_eval_dataset/task_cards.json",
            "scenario_cards": "pipeline/robot_eval_dataset/scenario_cards.json",
            "task_thresholds": "pipeline/robot_eval_dataset/task_thresholds.json",
            "scoring_methodology": "pipeline/robot_eval_dataset/scoring_methodology.json",
            "failure_taxonomy": "pipeline/robot_eval_dataset/failure_taxonomy.json",
            "failure_taxonomy_mode_count": failure_taxonomy_mode_count,
            "standard_scorecard_metrics": list(_STANDARD_ROBOT_EVAL_SCORECARD_METRICS),
            "required_scenario_variation_names": list(_STANDARD_SCENARIO_VARIATION_NAMES),
            "sim_only_beta_autonomy_enabled": _sim_only_beta_autonomy_enabled(),
            "simulator_profile": {
                "simulator": beta_runtime["simulator"],
                "provisioner": beta_runtime["provisioner"],
                "allow_simulator_execution": beta_runtime["allow_simulator_execution"],
                "allowed_simulators": beta_runtime["allowed_simulators"],
                "packaged_simulator_command_configured": bool(
                    beta_runtime["simulator_commands"].get("mujoco")
                ),
            },
            "simulator_engine_plugin_registry": (
                "pipeline/simulation_automation/simulator_engine_plugin_registry.json"
            ),
            "required_simulator_plugins": list(_STANDARD_SIMULATOR_ENGINE_NAMES),
            "required_world_model_plugins": list(_STANDARD_WORLD_MODEL_ENGINE_NAMES),
            "auto_staged": True,
        },
        "proof_boundary": {
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "robot_policy_execution_proven": False,
            "physics_contact_validated": False,
            "safety_validated": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    inbox = capture_root / "pipeline" / "robot_eval_job_requests" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    envelope_path = inbox / f"{job_id}.json"
    if not envelope_path.is_file():
        envelope = {
            "queue_contract": "robot_eval_job_request_inbox.v1",
            "status": "queued_for_pipeline",
            "job_id": job_id,
            "buyer_request_id": buyer_request_id,
            "pipeline_command": "blueprint-run-robot-eval-job",
            "pipeline_consumer": "BlueprintCapturePipeline",
            "job_request": request,
        }
        envelope_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
        status = "staged"
    else:
        status = "already_staged"
    return {
        "status": status,
        "job_id": job_id,
        "inbox_dir": str(inbox),
        "request_path": str(envelope_path),
        "failure_taxonomy_mode_count": failure_taxonomy_mode_count,
        "simulator_plugin_count": simulator_plugin_validation.get("simulator_plugin_count", 0),
        "world_model_plugin_count": simulator_plugin_validation.get(
            "world_model_plugin_count",
            0,
        ),
        "claim_boundary": "auto_staged_baseline_job_request_not_robot_team_submission",
    }


def _run_robot_eval_job_inbox_if_ready(
    capture_root: Path,
    *,
    auto_stage_task_eval: bool = False,
    descriptor_path: Optional[Path] = None,
) -> Dict[str, Any]:
    inbox = _robot_eval_job_request_inbox_for_capture(capture_root)
    auto_stage: Dict[str, Any] = {"status": "not_requested"}
    if inbox is None and auto_stage_task_eval and descriptor_path is not None:
        auto_stage = _auto_stage_robot_eval_job_request(
            capture_root=capture_root,
            descriptor_path=descriptor_path,
        )
        if auto_stage.get("status") == "blocked":
            return {
                "status": "blocked_missing_task_eval_inputs",
                "processed_count": 0,
                "inbox_dir": auto_stage.get("inbox_dir"),
                "manifest_path": None,
                "auto_stage": auto_stage,
                "claim_boundary": "task_eval_auto_stage_blocked_before_job_processing",
            }
        inbox = _robot_eval_job_request_inbox_for_capture(capture_root)
    if inbox is None:
        return {
            "status": "waiting_for_job_requests",
            "processed_count": 0,
            "inbox_dir": None,
            "manifest_path": None,
            "auto_stage": auto_stage,
            "claim_boundary": "no_robot_eval_job_request_v1_files_found",
        }
    runtime = _default_robot_eval_job_runtime(capture_root)
    result = run_robot_eval_job_request_inbox(
        capture_root=capture_root,
        inbox_dir=inbox,
        provisioner=runtime["provisioner"],
        simulator=runtime["simulator"],
        allow_simulator_execution=runtime["allow_simulator_execution"],
        allowed_simulators=runtime["allowed_simulators"],
        simulator_commands=runtime["simulator_commands"],
        allow_cpu_simulator_preflight=runtime["allow_cpu_simulator_preflight"],
    )
    return {
        "status": result.get("status"),
        "processed_count": result.get("processed_count", 0),
        "inbox_dir": str(inbox),
        "manifest_path": str(
            capture_root / "pipeline" / "robot_eval_job_requests" / "inbox_run_manifest.json"
        ),
        "auto_stage": auto_stage,
        "runtime": {
            "provisioner": runtime["provisioner"],
            "simulator": runtime["simulator"],
            "allow_simulator_execution": runtime["allow_simulator_execution"],
            "allowed_simulators": runtime["allowed_simulators"],
            "simulator_command_configured": bool(runtime["simulator_commands"]),
        },
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
    descriptor_path = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root)
    auto_stage_task_eval = _descriptor_requests_task_evaluation_run(
        descriptor_path
    ) or _call_requests_task_evaluation_run(
        lane=lane,
        requested_lanes=requested_lanes,
    ) or _sim_only_beta_default_task_eval_enabled()
    log_event(
        logger,
        logging.INFO,
        "capture_pipeline.started",
        descriptor_gcs_uri=descriptor_gcs_uri,
        descriptor_path=str(descriptor_path),
        requested_lane=lane,
        lanes=lanes,
        auto_stage_task_eval=auto_stage_task_eval,
    )

    results: List[Dict[str, Any]] = []
    qualification_result: Optional[Dict[str, Any]] = None

    def _append_lane_result(selected_lane: str, lane_result: Mapping[str, Any]) -> None:
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

    for selected_lane in lanes:
        log_event(
            logger,
            logging.INFO,
            "capture_pipeline.lane_started",
            descriptor_gcs_uri=descriptor_gcs_uri,
            selected_lane=selected_lane,
        )
        if selected_lane in {"qualification", "scene_memory"}:
            if qualification_result is None:
                qualification_result = run_qualification_pipeline(
                    descriptor_gcs_uri=descriptor_gcs_uri,
                    config=cfg,
                    requested_lanes=lanes,
                )
            if selected_lane == "qualification":
                _append_lane_result(selected_lane, qualification_result)
            else:
                _append_lane_result(
                    selected_lane,
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
                capture_root=descriptor_path.parent,
                provider_name="manual",
            )
            lane_result = _build_derived_lane_result(
                lane="evaluation_prep",
                source="evaluation_prep_artifacts",
                qualification_result=qualification_result,
                extra_fields={"manifest_path": evaluation_prep_result.get("manifest_path")},
            )
            _append_lane_result(selected_lane, lane_result)
            continue
        if selected_lane == "simulation_automation":
            capture_root = descriptor_path.parent
            automation_result = build_simulation_automation(capture_root=capture_root)
            robot_eval_jobs = _run_robot_eval_job_inbox_if_ready(
                capture_root,
                auto_stage_task_eval=auto_stage_task_eval,
                descriptor_path=descriptor_path,
            )
            auto_stage = robot_eval_jobs.get("auto_stage") if isinstance(
                robot_eval_jobs.get("auto_stage"), Mapping
            ) else {}
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
                    "robot_eval_job_runtime": robot_eval_jobs.get("runtime", {}),
                    "robot_eval_job_inbox_manifest_path": robot_eval_jobs.get("manifest_path"),
                    "robot_eval_job_auto_stage_status": auto_stage.get("status"),
                    "robot_eval_job_auto_stage_request_path": auto_stage.get("request_path"),
                    "robot_eval_job_auto_stage_blockers": auto_stage.get("blockers", []),
                    "robot_eval_job_auto_stage_missing_threshold_task_ids": auto_stage.get(
                        "missing_threshold_task_ids",
                        [],
                    ),
                    "robot_eval_job_auto_stage_missing_scorecard_metrics": auto_stage.get(
                        "missing_scorecard_metrics",
                        [],
                    ),
                    "robot_eval_job_auto_stage_missing_robot_eval_dataset_inputs": (
                        auto_stage.get(
                            "missing_robot_eval_dataset_inputs",
                            [],
                        )
                    ),
                    "robot_eval_job_auto_stage_missing_scenario_variation_names": auto_stage.get(
                        "missing_scenario_variation_names",
                        [],
                    ),
                    "robot_eval_job_auto_stage_missing_scenario_variation_names_by_scenario": (
                        auto_stage.get(
                            "missing_scenario_variation_names_by_scenario",
                            [],
                        )
                    ),
                    "robot_eval_job_auto_stage_failure_taxonomy_mode_count": auto_stage.get(
                        "failure_taxonomy_mode_count",
                        0,
                    ),
                    "robot_eval_job_auto_stage_missing_simulator_plugins": auto_stage.get(
                        "missing_simulator_plugins",
                        [],
                    ),
                    "robot_eval_job_auto_stage_missing_world_model_plugins": auto_stage.get(
                        "missing_world_model_plugins",
                        [],
                    ),
                    "robot_eval_job_auto_stage_unready_simulator_plugins": auto_stage.get(
                        "unready_simulator_plugins",
                        [],
                    ),
                    "robot_eval_job_auto_stage_unready_world_model_plugins": auto_stage.get(
                        "unready_world_model_plugins",
                        [],
                    ),
                    "robot_eval_job_auto_stage_missing_simulator_plugin_input_keys": (
                        auto_stage.get(
                            "missing_simulator_plugin_input_keys",
                            {},
                        )
                    ),
                    "robot_eval_job_auto_stage_missing_world_model_plugin_input_keys": (
                        auto_stage.get(
                            "missing_world_model_plugin_input_keys",
                            {},
                        )
                    ),
                    "robot_eval_job_auto_stage_missing_simulator_plugin_local_inputs": (
                        auto_stage.get(
                            "missing_simulator_plugin_local_inputs",
                            {},
                        )
                    ),
                    "robot_eval_job_auto_stage_missing_world_model_plugin_local_inputs": (
                        auto_stage.get(
                            "missing_world_model_plugin_local_inputs",
                            {},
                        )
                    ),
                    "robot_eval_job_auto_stage_simulator_plugin_count": auto_stage.get(
                        "simulator_plugin_count",
                        0,
                    ),
                    "robot_eval_job_auto_stage_world_model_plugin_count": auto_stage.get(
                        "world_model_plugin_count",
                        0,
                    ),
                },
            )
            _append_lane_result(selected_lane, lane_result)
            continue
        if selected_lane == "retrieval_index":
            capture_root = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root).parent
            retrieval_result = run_retrieval_index_stage(
                capture_root=capture_root,
                force_rebuild=parse_bool(os.getenv("RETRIEVAL_INDEX_FORCE_REBUILD"), default=False),
            )
            _append_lane_result(selected_lane, {"lane": "retrieval_index", **retrieval_result})
            continue
        if selected_lane == "frame_alignment":
            capture_root = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root).parent
            alignment_result = run_frame_alignment_stage(
                capture_root=capture_root,
                force_realign=parse_bool(os.getenv("FRAME_ALIGNMENT_FORCE_REALIGN"), default=False),
            )
            _append_lane_result(selected_lane, {"lane": "frame_alignment", **alignment_result})
            continue
        if selected_lane == "synthesis_coverage_validation":
            capture_root = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root).parent
            synthesis_result = _run_synthesis_coverage_validation(
                capture_root=capture_root,
                descriptor_gcs_uri=descriptor_gcs_uri,
                cfg=cfg,
            )
            _append_lane_result(
                selected_lane,
                {"lane": "synthesis_coverage_validation", **synthesis_result},
            )
            continue
        if selected_lane == "cosmos_single_capture_smoke":
            from .synthesis.cosmos_benchmark import run_cosmos_single_capture_smoke_lane

            capture_root = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root).parent
            smoke_result = run_cosmos_single_capture_smoke_lane(
                capture_root=capture_root,
                descriptor_gcs_uri=descriptor_gcs_uri,
                cfg=cfg,
            )
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
        raise ValueError(f"Unsupported pipeline lane: {selected_lane}")

    parsed = parse_gs_uri(descriptor_gcs_uri)
    result = {
        "status": "completed",
        "descriptor_gcs_uri": descriptor_gcs_uri,
        "bucket": parsed.bucket,
        "lanes": lanes,
        "results": results,
    }
    log_event(
        logger,
        logging.INFO,
        "capture_pipeline.completed",
        descriptor_gcs_uri=descriptor_gcs_uri,
        bucket=parsed.bucket,
        lanes=lanes,
        result_count=len(results),
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
