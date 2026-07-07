"""Live robot-eval closure audit.

This module is the job-level verifier for the full neutral eval harness. It
does not run external systems. It inspects the deterministic artifacts and any
owner-supplied live evidence, then writes a single closure manifest that says
which parts of the pipeline are proven, blocked, or only locally ready.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlparse

from .common import ensure_dir, read_json_any, resolve_gs_uri_to_path, utc_now_iso, write_json
from .failure_diagnosis_contract import build_failure_diagnosis_audit
from .local_capture import resolve_local_capture_context
from .scenario_variation_instantiator import SCENARIO_VARIATION_NAMES
from .simulation_automation import SIMULATOR_FRAMEWORKS, WORLD_MODEL_ENGINE_TARGETS


LIVE_ROBOT_EVAL_CLOSURE_SCHEMA_VERSION = "live_robot_eval_closure_manifest.v1"
LIVE_ROBOT_EVAL_EVIDENCE_SCHEMA_VERSION = "live_robot_eval_closure_evidence.v1"

WEBAPP_UPSTREAM_ID_FIELDS = (
    "site_submission_id",
    "request_id",
    "buyer_request_id",
    "capture_job_id",
)

WEBAPP_UPSTREAM_CAPTURE_GROUNDING_SOURCES = {
    "capture_descriptor",
    "raw_manifest",
    "raw_manifest.upstream_handoff",
    "pipeline.opportunity_handoff",
    "pipeline.webapp_sync_result",
}

WEBAPP_ROUTE_FORWARDING_PROOF_STATUSES = {
    "forwarded_to_pipeline_intake",
    "staged_for_control_plane",
}

SCORECARD_REQUIRED_FIELDS = (
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

TASK_CARD_STANDARD_REQUIRED_METRICS = SCORECARD_REQUIRED_FIELDS

TASK_CARD_METRIC_ALIASES: Dict[str, Sequence[str]] = {
    "cycle_time": ("cycle_time", "cycle_time_seconds", "mean_cycle_time_seconds"),
}

ROBOT_POV_OBSERVATION_REQUIRED_FIELDS = (
    "camera",
    "generated_frame_path",
    "observation_id",
    "render_sequence_id",
    "render_storyboard_id",
    "scenario_id",
    "task_id",
)

SUPPORTED_POLICY_MODALITIES = (
    "policy_api_endpoint",
    "docker_container",
    "recorded_action_trace",
    "high_level_skill_trace",
    "teleop_demo",
    "sim_controller_plugin",
)

POLICY_MODALITY_LOCAL_REFERENCE_FIELDS = {
    "policy_api_endpoint": (
        ("response_manifest_uri", "responseManifestUri", "local_response_path"),
    ),
    "docker_container": (
        ("output_manifest_uri", "outputManifestUri", "local_output_path"),
    ),
    "recorded_action_trace": (
        ("trace_manifest_uri", "traceManifestUri"),
    ),
    "teleop_demo": (
        ("demo_artifact_uri", "demoArtifactUri"),
    ),
    "sim_controller_plugin": (
        ("plugin_uri", "pluginUri"),
    ),
}

PREDICTED_VS_ACTUAL_SUMMARY_REQUIRED_SECTIONS = (
    "what_eval_predicted",
    "what_actually_happened",
    "which_scenarios_predicted_failure",
    "which_failures_were_missed",
    "how_much_real_world_tuning_was_needed",
    "whether_site_modifications_helped",
)

PREDICTED_VS_ACTUAL_SUMMARY_LIST_SECTIONS = (
    "what_eval_predicted",
    "what_actually_happened",
    "which_scenarios_predicted_failure",
    "which_failures_were_missed",
    "whether_site_modifications_helped",
)

PREDICTED_VS_ACTUAL_TUNING_SUMMARY_FIELDS = (
    "tuning_hours_total",
    "tuning_iterations_total",
    "records_with_tuning",
)

WORLD_MODEL_PLUGIN_OPTIONAL_INPUT_KEYS = (
    "world_manifest",
    "simready_bridge",
    "gpu_handoff_packet",
    "dense_world_model_export",
    "site_reference_projection",
)

OWNER_GPU_PROOF_REQUIRED_MANIFEST_FIELDS = (
    "owner_system_id",
    "simulator_backend",
    "simulator_version",
    "gpu_model",
    "proof_path",
)

OWNER_GPU_PROOF_REQUIRED_EVIDENCE_FLAGS = (
    "stdout_present",
    "stderr_present",
    "scene_load_trace_present",
    "scene_loaded_in_owner_simulator",
    "spawn_trace_present",
    "spawn_pose_loaded",
    "action_or_policy_trace_present",
    "action_or_policy_trace_valid",
    "default_smoke_policy_present",
    "default_smoke_policy_valid",
    "policy_execution_trace_present",
    "default_policy_execution_trace_valid",
    "sim_robot_pov_evidence_present",
    "sim_robot_pov_evidence_valid",
    "artifact_manifest_present",
    "artifact_manifest_valid",
    "robot_asset_trace_present",
    "robot_asset_matches_proof",
    "operator_attestation_present",
    "pass_fail_criteria_passed",
)

ISAAC_LIVE_SIMULATOR_FRAMEWORKS = {"isaac_sim", "isaac_lab_arena"}
MUJOCO_LIVE_SIMULATOR_FRAMEWORKS = {"mujoco"}
LIVE_SIMULATOR_FRAMEWORKS = ISAAC_LIVE_SIMULATOR_FRAMEWORKS | MUJOCO_LIVE_SIMULATOR_FRAMEWORKS

REPO_LOCAL_GATES = (
    "site_capture",
    "task_definitions",
    "scenario_library",
    "robot_pov_generation",
    "scenario_eval_suite",
    "failure_labels",
    "evaluation_methodology",
    "policy_interface",
    "simulator_engine_plugins",
    "report_generation",
)

LIVE_EXTERNAL_GATES = (
    "live_evidence_integrity",
    "webapp_upstream_truth",
    "rights_privacy_scope",
    "review_acceptance",
    "signed_delivery_access",
    "real_robot_pov_evidence",
    "real_world_validation_loop",
    "predicted_vs_actual_calibration",
    "safety_contact_physics_readiness",
    "live_simulator_execution",
    "live_policy_execution",
)

LIVE_EXTERNAL_PROOF_GATES = tuple(
    gate_id
    for gate_id in LIVE_EXTERNAL_GATES
    if gate_id
    not in {
        "real_world_validation_loop",
        "predicted_vs_actual_calibration",
    }
)

REQUIREMENT_COVERAGE_SPEC: Sequence[Dict[str, Any]] = (
    {
        "requirement_id": "site_capture",
        "label": "site capture",
        "gate_ids": ("site_capture",),
        "scope": "repo_local",
    },
    {
        "requirement_id": "task_definitions",
        "label": "task definitions",
        "gate_ids": ("task_definitions",),
        "scope": "repo_local",
    },
    {
        "requirement_id": "scenario_library",
        "label": "scenario library and per-task scenario families",
        "gate_ids": ("scenario_library",),
        "scope": "repo_local",
    },
    {
        "requirement_id": "scenario_variation_families",
        "label": "lighting/object/cart/path/human/forklift/occlusion/glare/label/wrong-object/approach scenario families",
        "gate_ids": ("scenario_library", "scenario_eval_suite"),
        "scope": "repo_local",
    },
    {
        "requirement_id": "robot_pov_generation",
        "label": "robot POV generation",
        "gate_ids": ("robot_pov_generation",),
        "scope": "repo_local",
    },
    {
        "requirement_id": "scenario_eval_suite",
        "label": "scenario/eval suite",
        "gate_ids": ("scenario_eval_suite",),
        "scope": "repo_local",
    },
    {
        "requirement_id": "failure_labels",
        "label": "failure labels",
        "gate_ids": ("failure_labels",),
        "scope": "repo_local",
    },
    {
        "requirement_id": "evaluation_methodology",
        "label": "standard evaluation methodology and scorecard metrics",
        "gate_ids": ("evaluation_methodology",),
        "scope": "repo_local",
    },
    {
        "requirement_id": "robot_team_interface",
        "label": "policy API, Docker, recorded trace, skill trace, teleop, and sim-controller policy interfaces",
        "gate_ids": ("policy_interface",),
        "scope": "repo_local",
    },
    {
        "requirement_id": "plugin_world_sim_engines",
        "label": "plug-in simulator and world-model engine registry",
        "gate_ids": ("simulator_engine_plugins",),
        "scope": "repo_local",
    },
    {
        "requirement_id": "neutral_eval_harness_report",
        "label": "site + task + scenario -> policy -> sim/world rollout -> evaluator -> report",
        "gate_ids": ("report_generation",),
        "scope": "repo_local",
    },
)

PLACEHOLDER_ID_MARKERS = (
    "dummy",
    "example",
    "fixture",
    "mock-",
    "placeholder",
    "replace_me",
    "sample",
    "test-",
)

RAW_CAPTURE_MEDIA_FILE_CANDIDATES = (
    "walkthrough.mov",
    "walkthrough.mp4",
    "recording.mov",
    "recording.mp4",
)

RAW_CAPTURE_EVIDENCE_POINTER_FIELDS = (
    "video_uri",
    "raw_video_uri",
    "source_video_uri",
    "walkthrough_video_uri",
    "recording_uri",
    "frames_index_uri",
    "frame_index_uri",
    "keyframe_index_uri",
    "object_index_uri",
    "object_point_cloud_index",
    "depth_manifest_uri",
    "camera_pose_uri",
    "poses_uri",
    "intrinsics_uri",
)

RAW_CAPTURE_EVIDENCE_COUNT_FIELDS = (
    ("frame_count", "frameCount"),
    ("keyframe_count", "keyframeCount"),
    ("pose_count", "poseCount"),
    ("depth_frame_count", "depthFrameCount"),
    ("object_count", "objectCount"),
    ("object_point_cloud_count", "objectPointCloudCount"),
)

TASK_CARD_REQUIRED_FIELDS = (
    "task_id",
    "task_statement",
    "task_category",
    "required_metrics",
)

SCENARIO_CARD_REQUIRED_FIELDS = (
    "scenario_id",
    "task_id",
    "robot_profile_id",
    "normal_scenario",
    "variation",
    "edge_case",
)

EVAL_CARD_REQUIRED_FIELDS = (
    "eval_card_id",
    "scenario_id",
    "task_id",
    "prediction_source",
    "validation",
    "proof_boundary",
)

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "live_robot_eval_closure_audit",
    "repo_local_default": True,
    "closure_can_upgrade_job_proof_only_with_accepted_evidence": True,
    "agents_may_not_set_proof_booleans_directly": True,
    "review_acceptance_proven": False,
    "rights_privacy_scope_proven": False,
    "signed_delivery_access_proven": False,
    "delivery_access_is_deployment_approval": False,
    "package_delivery_is_deployment_approval": False,
    "deployment_approval_proven": False,
    "physical_robot_readiness_proven": False,
    "safety_validation_proven": False,
    "simulator_execution_proven": False,
    "robot_policy_execution_proven": False,
    "rank_fidelity_result_proven": False,
    "public_claim_upgrade_allowed": False,
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _string(value).lower() in {"1", "true", "yes", "on", "passed", "success", "succeeded"}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(str(value))
        except (TypeError, ValueError):
            return None
    return number if math.isfinite(number) else None


def _non_negative_number(value: Any) -> bool:
    number = _number(value)
    return number is not None and number >= 0.0


def _rate_0_to_1(value: Any) -> bool:
    number = _number(value)
    return number is not None and 0.0 <= number <= 1.0


def _string_list(value: Any) -> List[str]:
    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    out: List[str] = []
    seen: set[str] = set()
    for item in values:
        text = _string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _valid_cycle_time(value: Any) -> bool:
    payload = _mapping(value)
    sample_count = _number(payload.get("sample_count"))
    if sample_count is None or sample_count < 0:
        return False
    mean_seconds = payload.get("mean_seconds")
    if sample_count == 0:
        return mean_seconds is None
    return _non_negative_number(mean_seconds)


def _valid_event_count(value: Any) -> bool:
    payload = _mapping(value)
    count = _number(payload.get("event_count"))
    return count is not None and count >= 0 and count.is_integer()


def _valid_recovery_success(value: Any) -> bool:
    payload = _mapping(value)
    attempt_count = _number(payload.get("attempt_count"))
    success_count = _number(payload.get("success_count"))
    if (
        attempt_count is None
        or success_count is None
        or attempt_count < 0
        or success_count < 0
        or not attempt_count.is_integer()
        or not success_count.is_integer()
        or success_count > attempt_count
    ):
        return False
    success_rate = payload.get("success_rate")
    if attempt_count == 0:
        return success_rate is None
    return _rate_0_to_1(success_rate)


def _valid_uncertainty(value: Any) -> bool:
    payload = _mapping(value)
    status = _string(payload.get("status"))
    sample_count = _number(payload.get("sample_count"))
    if not status or sample_count is None or sample_count < 0 or not sample_count.is_integer():
        return False
    mean_score = payload.get("mean_score")
    if sample_count == 0:
        return mean_score is None
    return _rate_0_to_1(mean_score)


def _invalid_scorecard_fields(scorecard: Mapping[str, Any]) -> List[str]:
    invalid: List[str] = []
    if "success_rate" in scorecard and not _rate_0_to_1(scorecard.get("success_rate")):
        invalid.append("success_rate")
    if "cycle_time" in scorecard and not _valid_cycle_time(scorecard.get("cycle_time")):
        invalid.append("cycle_time")
    if "intervention_rate" in scorecard and not _non_negative_number(
        scorecard.get("intervention_rate")
    ):
        invalid.append("intervention_rate")
    for field in ("unsafe_proximity", "collision_risk", "object_drop", "wrong_object", "timeout"):
        if field in scorecard and not _valid_event_count(scorecard.get(field)):
            invalid.append(field)
    if "recovery_success" in scorecard and not _valid_recovery_success(
        scorecard.get("recovery_success")
    ):
        invalid.append("recovery_success")
    if "world_model_uncertainty" in scorecard and not _valid_uncertainty(
        scorecard.get("world_model_uncertainty")
    ):
        invalid.append("world_model_uncertainty")
    calibration_score = scorecard.get("sim_vs_real_calibration_score")
    if (
        "sim_vs_real_calibration_score" in scorecard
        and calibration_score is not None
        and not _rate_0_to_1(calibration_score)
    ):
        invalid.append("sim_vs_real_calibration_score")
    return invalid




def _field(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
    return None


def _policy_modality_missing_inputs(modality: str, reference: Mapping[str, Any]) -> List[str]:
    missing: List[str] = []
    if modality == "policy_api_endpoint":
        endpoint = _string(_field(reference, "endpoint_url", "endpointUrl", "url"))
        if not (endpoint.startswith("https://") or endpoint.startswith("http://")):
            missing.append("policy_package.policy_api_endpoint.endpoint_url")
    elif modality == "docker_container":
        if not _string(_field(reference, "image_ref", "imageRef")):
            missing.append("policy_package.docker_container.image_ref")
        digest = _string(_field(reference, "digest", "digestChecksum"))
        if not digest.startswith("sha256:"):
            missing.append("policy_package.docker_container.digest")
    elif modality == "recorded_action_trace":
        if not _string(_field(reference, "trace_manifest_uri", "traceManifestUri")):
            missing.append("policy_package.recorded_action_trace.trace_manifest_uri")
        if not _string(_field(reference, "timestamp_alignment", "timestampAlignment")):
            missing.append("policy_package.recorded_action_trace.timestamp_alignment")
    elif modality == "high_level_skill_trace":
        sequence = reference.get("ordered_skill_sequence") or reference.get("orderedSkillSequence")
        if not (isinstance(sequence, list) and sequence):
            missing.append("policy_package.high_level_skill_trace.ordered_skill_sequence")
    elif modality == "teleop_demo":
        if not _string(_field(reference, "demo_artifact_uri", "demoArtifactUri")):
            missing.append("policy_package.teleop_demo.demo_artifact_uri")
        if not _string(
            _field(reference, "rights_privacy_attestation", "rightsPrivacyAttestation")
        ):
            missing.append("policy_package.teleop_demo.rights_privacy_attestation")
    elif modality == "sim_controller_plugin":
        if not _string(_field(reference, "simulator_framework", "simulatorFramework")):
            missing.append("policy_package.sim_controller_plugin.simulator_framework")
        if not _string(_field(reference, "plugin_uri", "pluginUri")):
            missing.append("policy_package.sim_controller_plugin.plugin_uri")
    return missing


def _policy_modality_local_reference_audit(
    *,
    modality: str,
    reference: Mapping[str, Any],
    capture_root: Path,
    job_dir: Path,
) -> tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
    artifacts: Dict[str, Dict[str, Any]] = {}
    missing_keys: List[str] = []
    missing_inputs: List[str] = []
    for aliases in POLICY_MODALITY_LOCAL_REFERENCE_FIELDS.get(modality, ()):
        value = _field(reference, *aliases)
        text = _string(value)
        if not text or _external_uri(text):
            continue
        key = aliases[0]
        local_path = _local_reference_path(text, capture_root=capture_root, job_dir=job_dir)
        if local_path is None:
            continue
        artifact = _artifact(local_path, base_dir=job_dir)
        artifacts[key] = artifact
        if not artifact["exists"]:
            missing_keys.append(key)
            missing_inputs.append(f"policy_package.{modality}.{key}_local_file_missing")
    return artifacts, missing_keys, missing_inputs


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = read_json_any(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_optional_any(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return read_json_any(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _relative_to(base: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _artifact(path: Path, *, base_dir: Path) -> Dict[str, Any]:
    return {
        "path": _relative_to(base_dir, path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def _external_uri(value: str) -> bool:
    return "://" in value and not value.startswith("file://")


def _raw_capture_pointer_path(*, capture_root: Path, raw_root: Path, value: str) -> Path | None:
    text = value.strip()
    if not text or _external_uri(text):
        return None
    if text.startswith("file://"):
        return Path(text[7:]).expanduser()
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"raw", "pipeline", "privacy"}:
        return capture_root / path
    return raw_root / path


def _raw_capture_evidence_summary(
    *,
    capture_root: Path,
    raw_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_root = capture_root / "raw"
    local_files = [
        {
            "path": _relative_to(capture_root, raw_root / name),
            "size_bytes": (raw_root / name).stat().st_size,
        }
        for name in RAW_CAPTURE_MEDIA_FILE_CANDIDATES
        if (raw_root / name).is_file()
    ]
    remote_pointers: List[Dict[str, str]] = []
    local_pointer_files: List[Dict[str, Any]] = []
    missing_local_pointer_files: List[Dict[str, str]] = []
    pointer_fields: Dict[str, List[str]] = {}
    for field in RAW_CAPTURE_EVIDENCE_POINTER_FIELDS:
        values = _string_list(raw_manifest.get(field))
        if not values:
            continue
        pointer_fields[field] = values
        for value in values:
            if _external_uri(value):
                remote_pointers.append({"field": field, "uri": value})
                continue
            path = _raw_capture_pointer_path(
                capture_root=capture_root,
                raw_root=raw_root,
                value=value,
            )
            if path is None:
                continue
            if path.is_file():
                local_pointer_files.append(
                    {
                        "field": field,
                        "path": _relative_to(capture_root, path),
                        "size_bytes": path.stat().st_size,
                    }
                )
            else:
                missing_local_pointer_files.append(
                    {"field": field, "path": _relative_to(capture_root, path)}
                )

    positive_counts: Dict[str, int | float] = {}
    for aliases in RAW_CAPTURE_EVIDENCE_COUNT_FIELDS:
        number = _number(_field(raw_manifest, *aliases))
        if number is not None and number > 0:
            positive_counts[aliases[0]] = int(number) if number.is_integer() else number
    exposure_samples = raw_manifest.get("exposure_samples")
    if isinstance(exposure_samples, list) and exposure_samples:
        positive_counts["exposure_sample_count"] = len(exposure_samples)

    has_capture_evidence = bool(local_files or local_pointer_files or remote_pointers)
    return {
        "has_capture_evidence": has_capture_evidence,
        "local_media_files": local_files,
        "local_pointer_files": local_pointer_files,
        "remote_pointer_uris": remote_pointers,
        "missing_local_pointer_files": missing_local_pointer_files,
        "pointer_fields": pointer_fields,
        "positive_counts": positive_counts,
    }


def _card_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cards = payload.get("cards")
    if not isinstance(cards, list):
        return []
    return [dict(item) for item in cards if isinstance(item, Mapping)]


def _card_field_present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return bool(value)
    return value is not None


def _cards_missing_required_fields(
    rows: Sequence[Mapping[str, Any]],
    required_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    invalid: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        missing = [
            field for field in required_fields if not _card_field_present(row.get(field))
        ]
        if missing:
            invalid.append(
                {
                    "card_index": index,
                    "card_id": _string(
                        row.get("task_card_id")
                        or row.get("scenario_card_id")
                        or row.get("eval_card_id")
                        or row.get("task_id")
                        or row.get("scenario_id")
                        or row.get("record_id")
                    ),
                    "missing_fields": missing,
                }
            )
    return invalid


def _concrete_mutation_present(value: Any) -> bool:
    mutation = _mapping(value)
    if not mutation:
        return False
    return any(_card_field_present(item) for item in mutation.values())


def _engine_mutation_operations_present(value: Any) -> bool:
    payload = _mapping(value)
    if not payload:
        return False

    def _operations_ready(candidate: Any) -> bool:
        row = _mapping(candidate)
        if not row:
            return False
        operations = row.get("operations")
        if not isinstance(operations, list) or not operations:
            return False
        operation_count = _number(row.get("operation_count"))
        if operation_count is not None and operation_count <= 0:
            return False
        return any(
            isinstance(operation, Mapping)
            and _card_field_present(operation.get("parameters"))
            for operation in operations
        )

    return _operations_ready(payload) or any(
        _operations_ready(candidate) for candidate in payload.values()
    )


def _variation_row_missing_concrete_fields(row: Mapping[str, Any]) -> List[str]:
    missing: List[str] = []
    if not _concrete_mutation_present(row.get("concrete_mutation")):
        missing.append("concrete_mutation")
    if not _engine_mutation_operations_present(row.get("engine_mutations")):
        missing.append("engine_mutations")
    return missing


def _variation_rows_missing_concrete_details(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    invalid: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        missing = _variation_row_missing_concrete_fields(row)
        if not missing:
            continue
        invalid.append(
            {
                "row_index": index,
                "instance_id": _string(row.get("instance_id")),
                "task_id": _string(row.get("task_id") or row.get("taskId")),
                "scenario_id": _string(row.get("scenario_id") or row.get("scenarioId")),
                "variation_name": _string(row.get("variation_name") or row.get("variationName")),
                "missing_fields": missing,
            }
        )
    return invalid


def _variation_instance_detail_index(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, bool]]:
    details: Dict[str, Dict[str, bool]] = {}
    for row in rows:
        instance_id = _string(row.get("instance_id") or row.get("scenario_variation_instance_id"))
        if not instance_id:
            continue
        details[instance_id] = {
            "concrete_mutation": _concrete_mutation_present(row.get("concrete_mutation")),
            "engine_mutations": _engine_mutation_operations_present(row.get("engine_mutations")),
        }
    return details


def _scenario_eval_runs_missing_concrete_variation_details(
    *,
    rows: Sequence[Mapping[str, Any]],
    variation_instance_details: Mapping[str, Mapping[str, bool]],
) -> List[Dict[str, Any]]:
    invalid: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if bool(row.get("baseline_capture_layout") or row.get("baselineCaptureLayout")):
            continue
        instance_id = _string(
            row.get("scenario_variation_instance_id")
            or row.get("scenarioVariationInstanceId")
        )
        linked = _mapping(variation_instance_details.get(instance_id))
        concrete_ready = _concrete_mutation_present(row.get("concrete_mutation")) or bool(
            linked.get("concrete_mutation")
        )
        engine_ready = _engine_mutation_operations_present(row.get("engine_mutations")) or bool(
            linked.get("engine_mutations")
        )
        missing: List[str] = []
        if not concrete_ready:
            missing.append("concrete_mutation")
        if not engine_ready:
            missing.append("engine_mutations")
        if missing and not instance_id:
            missing.insert(0, "scenario_variation_instance_id")
        if not missing:
            continue
        invalid.append(
            {
                "row_index": index,
                "scenario_eval_run_id": _string(
                    row.get("scenario_eval_run_id") or row.get("scenarioEvalRunId")
                ),
                "scenario_variation_instance_id": instance_id,
                "task_id": _string(row.get("task_id") or row.get("taskId")),
                "scenario_id": _string(row.get("scenario_id") or row.get("scenarioId")),
                "variation_name": _string(row.get("variation_name") or row.get("variationName")),
                "missing_fields": missing,
            }
        )
    return invalid


def _cards_missing_standard_required_metrics(
    rows: Sequence[Mapping[str, Any]],
    required_metrics: Sequence[str],
) -> List[Dict[str, Any]]:
    invalid: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        present = set(_string_list(row.get("required_metrics")))
        missing = sorted(
            metric
            for metric in required_metrics
            if not (present & set(TASK_CARD_METRIC_ALIASES.get(metric, (metric,))))
        )
        if missing:
            invalid.append(
                {
                    "index": index,
                    "task_id": _string(row.get("task_id") or row.get("taskId")),
                    "missing_metrics": missing,
                }
            )
    return invalid


def _scenario_family_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    families = payload.get("families")
    if not isinstance(families, list):
        return []
    return [dict(item) for item in families if isinstance(item, Mapping)]


def _scenario_family_variation_names(family: Mapping[str, Any]) -> set[str]:
    variations = family.get("variations")
    if not isinstance(variations, list):
        return set()
    return {
        _string(variation.get("variation_id") or variation.get("variation_name"))
        for variation in variations
        if isinstance(variation, Mapping)
        and _string(variation.get("variation_id") or variation.get("variation_name"))
    }


def _scenario_family_task_coverage(
    *,
    task_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    required_variation_names: Sequence[str],
) -> Dict[str, Any]:
    task_ids = sorted(
        {
            _string(row.get("task_id") or row.get("taskId"))
            for row in task_rows
            if _string(row.get("task_id") or row.get("taskId"))
        }
    )
    family_task_ids = sorted(
        {
            _string(row.get("task_id") or row.get("taskId"))
            for row in family_rows
            if _string(row.get("task_id") or row.get("taskId"))
        }
    )
    required = [name for name in required_variation_names if name]
    missing_required_variations_by_family: List[Dict[str, Any]] = []
    for row in family_rows:
        present = _scenario_family_variation_names(row)
        missing = [name for name in required if name not in present]
        if missing:
            missing_required_variations_by_family.append(
                {
                    "family_id": _string(row.get("family_id") or row.get("familyId")),
                    "task_id": _string(row.get("task_id") or row.get("taskId")),
                    "scenario_id": _string(row.get("scenario_id") or row.get("scenarioId")),
                    "missing_variation_names": missing,
                }
            )
    return {
        "task_ids": task_ids,
        "family_task_ids": family_task_ids,
        "missing_task_ids": [task_id for task_id in task_ids if task_id not in family_task_ids],
        "missing_required_variations_by_family": missing_required_variations_by_family,
    }


def _missing_required_variations_by_scenario(
    *,
    coverage_rows: Sequence[Mapping[str, Any]],
    required_variation_names: Sequence[str],
    scenario_rows: Sequence[Mapping[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    required = [name for name in required_variation_names if name]
    if not required:
        return []
    known_scopes: set[tuple[str, str]] = set()
    for row in scenario_rows if scenario_rows is not None else coverage_rows:
        task_id = _string(row.get("task_id") or row.get("taskId"))
        scenario_id = _string(row.get("scenario_id") or row.get("scenarioId"))
        if scenario_id:
            known_scopes.add((task_id, scenario_id))
    covered_exact: Dict[tuple[str, str], set[str]] = {}
    for row in coverage_rows:
        if bool(row.get("baseline_capture_layout") or row.get("baselineCaptureLayout")):
            continue
        task_id = _string(row.get("task_id") or row.get("taskId"))
        scenario_id = _string(row.get("scenario_id") or row.get("scenarioId"))
        variation_name = _string(row.get("variation_name") or row.get("variationName"))
        if not scenario_id or not variation_name:
            continue
        covered_exact.setdefault((task_id, scenario_id), set()).add(variation_name)
        if task_id:
            known_scopes.add((task_id, scenario_id))
    missing: List[Dict[str, Any]] = []
    for task_id, scenario_id in sorted(known_scopes):
        covered = set(covered_exact.get((task_id, scenario_id), set()))
        covered.update(covered_exact.get(("", scenario_id), set()))
        missing_names = [name for name in required if name not in covered]
        if missing_names:
            missing.append(
                {
                    "task_id": task_id,
                    "scenario_id": scenario_id,
                    "missing_variation_names": missing_names,
                }
            )
    return missing


def _scenario_eval_run_ids(rows: Sequence[Any]) -> List[str]:
    return sorted(
        {
            _string(row.get("scenario_eval_run_id") or row.get("scenarioEvalRunId"))
            for row in rows
            if isinstance(row, Mapping)
            and _string(row.get("scenario_eval_run_id") or row.get("scenarioEvalRunId"))
        }
    )


def _job_local_artifact_path(job_dir: Path, value: Any) -> Path | None:
    text = _string(value)
    if not text or "://" in text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return job_dir / path


def _cards_count(payload: Mapping[str, Any], *count_fields: str) -> int:
    for field in count_fields:
        value = payload.get(field)
        if isinstance(value, int):
            return value
    cards = payload.get("cards")
    if isinstance(cards, list):
        return len([item for item in cards if isinstance(item, Mapping)])
    scenarios = payload.get("scenarios")
    if isinstance(scenarios, list):
        return len([item for item in scenarios if isinstance(item, Mapping)])
    return 0


def _status_ok(value: Any) -> bool:
    return _string(value).lower() in {
        "accepted",
        "complete",
        "completed",
        "completed_review_required",
        "live_end_to_end_verified",
        "passed",
        "ready",
        "signed_access_ready",
        "succeeded",
        "validated",
        "verified",
    }


def _attestation_ok(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, Mapping):
        return False
    return bool(
        _string(
            value.get("attested_by")
            or value.get("attestedBy")
            or value.get("operator_id")
            or value.get("operatorId")
            or value.get("reviewer")
        )
        and _string(
            value.get("attestation")
            or value.get("statement")
            or value.get("accepted_claim_boundary")
            or value.get("acceptedClaimBoundary")
        )
    )








def _owner_gpu_proof_manifest_audit(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    missing_required_fields = [
        field
        for field in OWNER_GPU_PROOF_REQUIRED_MANIFEST_FIELDS
        if not _card_field_present(manifest.get(field))
    ]
    try:
        exit_code = int(manifest.get("exit_code"))
    except (TypeError, ValueError):
        exit_code = None
    manifest_blockers = _string_list(manifest.get("blockers"))
    missing_inputs = _string_list(manifest.get("missing_inputs"))
    evidence = _mapping(manifest.get("evidence"))
    missing_evidence_flags = [
        field
        for field in OWNER_GPU_PROOF_REQUIRED_EVIDENCE_FLAGS
        if not bool(evidence.get(field))
    ]
    blockers: List[str] = []
    if missing_required_fields:
        blockers.append("owner_gpu_proof_missing_required_manifest_fields")
    if exit_code != 0:
        blockers.append("owner_gpu_proof_exit_code_not_zero")
    if manifest_blockers:
        blockers.append("owner_gpu_proof_manifest_has_blockers")
    if missing_inputs:
        blockers.append("owner_gpu_proof_manifest_missing_inputs")
    if missing_evidence_flags:
        blockers.append("owner_gpu_proof_manifest_missing_required_evidence")
    accepted = (
        _string(manifest.get("status")) == "accepted"
        and bool(manifest.get("owner_gpu_simulator_execution_proven"))
        and not blockers
    )
    return {
        "accepted": accepted,
        "blockers": blockers,
        "missing_required_fields": missing_required_fields,
        "exit_code": exit_code,
        "simulator_backend": manifest.get("simulator_backend"),
        "isaac_sim_execution_proven": bool(manifest.get("isaac_sim_execution_proven")),
        "isaac_robot_asset_execution_proven": bool(
            manifest.get("isaac_robot_asset_execution_proven")
        ),
        "unitree_g1_asset_spawned": bool(manifest.get("unitree_g1_asset_spawned")),
        "robot_asset": _mapping(manifest.get("robot_asset")),
        "manifest_blockers": manifest_blockers,
        "missing_inputs": missing_inputs,
        "missing_evidence_flags": missing_evidence_flags,
    }


def _policy_execution_result_audit(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    selected_modalities = _string_list(manifest.get("selected_modalities"))
    modality_results = _mapping(manifest.get("modality_results"))
    missing_result_modalities = [
        modality for modality in selected_modalities if modality not in modality_results
    ]
    proven_executed_modalities: List[str] = []
    reference_only_modalities: List[str] = []
    unproven_executed_modalities: List[str] = []
    evidence_free_proof_modalities: List[str] = []
    for modality in selected_modalities:
        result = _mapping(modality_results.get(modality))
        if not result:
            continue
        execution_performed = bool(result.get("execution_performed"))
        reference_replayed = bool(result.get("reference_replayed"))
        # The manifest flag is a claim, not proof: it must be a strict boolean and be
        # backed by execution evidence — an artifact reference or the executed attempt
        # trace the modality runner emits — to count as proven.
        result_proven = result.get("robot_policy_execution_proven") is True
        attempt_count = _number(result.get("attempt_count"))
        evidence_backed = bool(
            _string_list(result.get("evidence_refs"))
            or _string(result.get("run_manifest_uri"))
            or _string(result.get("trace_path"))
            or _mapping(result.get("artifact_paths"))
            or result.get("policy_submission_trace_available") is True
            or (attempt_count is not None and attempt_count > 0)
        )
        completed = _string(result.get("status")) == "completed"
        if reference_replayed and not execution_performed:
            reference_only_modalities.append(modality)
        if result_proven and not evidence_backed:
            evidence_free_proof_modalities.append(modality)
        if (
            execution_performed
            and completed
            and result_proven
            and evidence_backed
            and not reference_replayed
        ):
            proven_executed_modalities.append(modality)
        elif execution_performed or result_proven:
            unproven_executed_modalities.append(modality)
    blockers: List[str] = []
    if evidence_free_proof_modalities:
        blockers.append("policy_execution_proof_flag_without_evidence_refs")
    if not selected_modalities:
        blockers.append("policy_execution_missing_selected_modalities")
    if missing_result_modalities:
        blockers.append("policy_execution_missing_selected_modality_results")
    if not proven_executed_modalities:
        blockers.append("policy_execution_missing_proven_executed_modality")
    if reference_only_modalities and not proven_executed_modalities:
        blockers.append("policy_execution_selected_modalities_reference_replay_only")
    if unproven_executed_modalities and not proven_executed_modalities:
        blockers.append("policy_execution_selected_modalities_not_cleanly_proven")
    return {
        "blockers": blockers,
        "selected_modalities": selected_modalities,
        "missing_result_modalities": missing_result_modalities,
        "proven_executed_modalities": proven_executed_modalities,
        "reference_only_modalities": reference_only_modalities,
        "unproven_executed_modalities": sorted(set(unproven_executed_modalities)),
    }


def _contains_placeholder_id(value: Any) -> bool:
    text = _string(value).lower()
    return bool(text) and any(marker in text for marker in PLACEHOLDER_ID_MARKERS)


def _gate(
    gate_id: str,
    *,
    passed: bool,
    blockers: Sequence[str],
    evidence: Mapping[str, Any] | None = None,
    proof_boolean: bool | None = None,
) -> Dict[str, Any]:
    unique_blockers: List[str] = []
    for blocker in blockers:
        if blocker and blocker not in unique_blockers:
            unique_blockers.append(blocker)
    return {
        "gate_id": gate_id,
        "status": "passed" if passed and not unique_blockers else "blocked",
        "passed": bool(passed and not unique_blockers),
        "proof_boolean": bool(proof_boolean if proof_boolean is not None else passed and not unique_blockers),
        "blockers": unique_blockers,
        "evidence": dict(evidence or {}),
    }


def _requirement_coverage(gates: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    requirements: List[Dict[str, Any]] = []
    for spec in REQUIREMENT_COVERAGE_SPEC:
        gate_ids = tuple(_string_list(spec.get("gate_ids")))
        gate_statuses = {
            gate_id: {
                "passed": bool(_mapping(gates.get(gate_id)).get("passed")),
                "status": _string(_mapping(gates.get(gate_id)).get("status")) or "missing",
                "blockers": _string_list(_mapping(gates.get(gate_id)).get("blockers")),
            }
            for gate_id in gate_ids
        }
        missing_gate_ids = [gate_id for gate_id in gate_ids if gate_id not in gates]
        blockers = [
            f"{gate_id}:{blocker}"
            for gate_id, gate in gate_statuses.items()
            for blocker in _string_list(gate.get("blockers"))
        ]
        blockers.extend(f"{gate_id}:missing_gate" for gate_id in missing_gate_ids)
        passed = bool(gate_ids) and not missing_gate_ids and all(
            bool(gate_statuses[gate_id]["passed"]) for gate_id in gate_ids
        )
        requirements.append(
            {
                "requirement_id": _string(spec.get("requirement_id")),
                "label": _string(spec.get("label")),
                "scope": _string(spec.get("scope")) or "repo_local",
                "gate_ids": list(gate_ids),
                "status": "passed" if passed else "blocked",
                "passed": passed,
                "gate_statuses": gate_statuses,
                "blockers": blockers,
            }
        )
    passed_ids = [
        item["requirement_id"] for item in requirements if bool(item.get("passed"))
    ]
    blocked_ids = [
        item["requirement_id"] for item in requirements if not bool(item.get("passed"))
    ]
    repo_local_ids = [
        item["requirement_id"]
        for item in requirements
        if item.get("scope") == "repo_local"
    ]
    live_external_ids = [
        item["requirement_id"]
        for item in requirements
        if item.get("scope") == "live_external"
    ]
    return {
        "schema_version": "live_robot_eval_requirement_coverage.v1",
        "requirement_count": len(requirements),
        "passed_count": len(passed_ids),
        "blocked_count": len(blocked_ids),
        "all_requirements_passed": not blocked_ids,
        "repo_local_requirement_ids": repo_local_ids,
        "live_external_requirement_ids": live_external_ids,
        "passed_requirement_ids": passed_ids,
        "blocked_requirement_ids": blocked_ids,
        "requirements": requirements,
    }


def _prefixed_gate_blockers(
    gates: Mapping[str, Mapping[str, Any]],
    gate_ids: Sequence[str],
) -> List[str]:
    blockers: List[str] = []
    for gate_id in gate_ids:
        gate = _mapping(gates.get(gate_id))
        blockers.extend(
            f"{gate_id}:{blocker}" for blocker in _string_list(gate.get("blockers"))
        )
    return blockers


def _readiness_check(
    *,
    check_id: str,
    label: str,
    passed: bool,
    blockers: Sequence[str],
    evidence: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    unique_blockers: List[str] = []
    for blocker in blockers:
        if blocker and blocker not in unique_blockers:
            unique_blockers.append(blocker)
    return {
        "check_id": check_id,
        "label": label,
        "status": "passed" if passed and not unique_blockers else "blocked",
        "passed": bool(passed and not unique_blockers),
        "blockers": unique_blockers,
        "evidence": dict(evidence or {}),
    }


def _latest_runpod_live_execution_proof(job_dir: Path) -> tuple[Path | None, Dict[str, Any]]:
    candidates = sorted(job_dir.glob("runpod_live_execution_proof*.json"))
    if not candidates:
        return None, {}
    latest = max(candidates, key=lambda path: (path.stat().st_mtime, path.name))
    return latest, _read_optional_mapping(latest)


def _robot_team_beta_readiness_summary(
    *,
    gates: Mapping[str, Mapping[str, Any]],
    job_dir: Path,
    repo_local_ready: bool,
    live_external_ready: bool,
    live_end_to_end_verified: bool,
) -> Dict[str, Any]:
    remote_closure_path = job_dir / "remote_cloud_execution_closure_manifest.json"
    remote_closure = _read_optional_mapping(remote_closure_path)
    remote_checks = _mapping(remote_closure.get("checks"))
    remote_outputs = _mapping(remote_closure.get("outputs"))
    remote_cost = _mapping(remote_closure.get("cost_and_timeout_controls"))
    artifact_output_write_probe_path = job_dir / "artifact_output_write_probe_manifest.json"
    artifact_output_write_probe = _read_optional_mapping(artifact_output_write_probe_path)
    remote_provider_setup = _mapping(remote_closure.get("provider_input_setup"))
    remote_provider_setup_blockers = _string_list(remote_provider_setup.get("blockers"))
    remote_provider_inputs_uploaded = remote_provider_setup.get("provider_inputs_uploaded")
    gpu_cost_path = job_dir / "gpu_cost_control_ledger.json"
    gpu_cost = _read_optional_mapping(gpu_cost_path)
    runpod_proof_path, runpod_proof = _latest_runpod_live_execution_proof(job_dir)
    runpod_zero_active_pods_now = bool(
        runpod_proof.get("status") == "runpod_live_proof_collected"
        and runpod_proof.get("active_pod_count_after") == 0
        and runpod_proof.get("runpod_side_effects_may_have_occurred") is False
    )
    today_utc = utc_now_iso()[:10]
    remote_closure_generated_at = _string(remote_closure.get("generated_at"))
    remote_closure_generated_date = (
        remote_closure_generated_at[:10] if remote_closure_generated_at else ""
    )
    fresh_live_run_today_ready = bool(
        remote_closure.get("remote_cloud_execution_proven")
        and remote_closure.get("live_provider_calls_performed")
        and remote_closure_generated_date == today_utc
    )
    live_provider_worker_blockers = _string_list(remote_closure.get("blockers"))
    if remote_closure and not remote_closure.get("remote_cloud_execution_proven"):
        live_provider_worker_blockers = live_provider_worker_blockers or [
            "remote_cloud_execution_not_proven"
        ]
    robot_pov_generation_gate = _mapping(gates.get("robot_pov_generation"))
    policy_execution_gate = _mapping(gates.get("live_policy_execution"))
    policy_execution_evidence = _mapping(policy_execution_gate.get("evidence"))
    sim_policy_attempt_count = policy_execution_evidence.get("attempt_count")
    sim_policy_run_count = policy_execution_evidence.get("scenario_eval_run_count")
    sim_policy_missing_run_count = int(
        policy_execution_evidence.get("missing_scenario_eval_run_count") or 0
    )
    sim_policy_attempts_missing_trace = _string_list(
        policy_execution_evidence.get("attempts_missing_action_or_skill_trace")
    )
    simulator_pov_policy_ready = bool(
        robot_pov_generation_gate.get("passed")
        and _string(policy_execution_evidence.get("policy_execution_manifest_status"))
        == "completed"
        and _string(policy_execution_evidence.get("policy_execution_trace_status"))
        == "completed"
        and int(sim_policy_attempt_count or 0) > 0
        and sim_policy_attempt_count == sim_policy_run_count
        and sim_policy_missing_run_count == 0
        and not sim_policy_attempts_missing_trace
    )
    real_robot_pov_gate = _mapping(gates.get("real_robot_pov_evidence"))
    real_robot_pov_proof_ready = bool(real_robot_pov_gate.get("proof_boolean"))
    real_robot_pov_live_policy_ready = bool(
        real_robot_pov_proof_ready and policy_execution_gate.get("passed")
    )
    robot_pov_policy_ready = bool(
        simulator_pov_policy_ready or real_robot_pov_live_policy_ready
    )
    robot_pov_policy_blockers: List[str] = []
    if not robot_pov_policy_ready:
        if not simulator_pov_policy_ready:
            robot_pov_policy_blockers.append(
                "simulator_robot_pov_reference_policy_evidence_not_complete"
            )
        robot_pov_policy_blockers.extend(
            _prefixed_gate_blockers(
                gates,
                ("live_policy_execution",),
            )
        )

    provider_runtime_finalizer_proof_path = job_dir / "provider_runtime_finalizer_proof.json"
    provider_runtime_finalizer_proof = _read_optional_mapping(
        provider_runtime_finalizer_proof_path
    )
    worker_runtime_manifest_path = job_dir / "worker_runtime_manifest.json"
    job_run_manifest_path = job_dir / "job_run_manifest.json"
    robot_eval_report_path = job_dir / "robot_eval_report.json"
    final_live_closure_artifacts_ready = bool(
        remote_closure.get("status") == "remote_execution_completed_with_shutdown_proof"
        and provider_runtime_finalizer_proof.get("status") == "completed"
        and gpu_cost_path.is_file()
        and worker_runtime_manifest_path.is_file()
        and job_run_manifest_path.is_file()
        and robot_eval_report_path.is_file()
    )

    remote_artifact_output_uri = _string(remote_outputs.get("artifact_output_uri"))
    remote_artifact_output_ready = bool(
        remote_artifact_output_uri
        and remote_checks.get("artifact_output_uri_configured")
        and remote_checks.get("artifact_output_uri_provider_writable")
        and remote_checks.get("artifact_output_write_auth_contract_ready")
        and remote_provider_inputs_uploaded is True
        and not remote_provider_setup_blockers
    )
    remote_shutdown_cost_ready = bool(
        remote_closure.get("clean_shutdown_proven")
        and remote_checks.get("actual_gpu_time_record_present")
        and remote_cost.get("actual_gpu_seconds") is not None
    )
    deployment_intake_path = job_dir / "deployment_outcome_intake_manifest.json"
    deployment_ledger_path = job_dir / "deployment_outcome_ledger.json"
    deployment_summary_path = job_dir / "prediction_vs_actual_deployment_summary.json"
    calibration_report_path = job_dir / "sim_vs_real_calibration_report.json"
    deployment_intake = _read_optional_mapping(deployment_intake_path)
    deployment_ledger = _read_optional_mapping(deployment_ledger_path)
    deployment_summary = _read_optional_mapping(deployment_summary_path)
    calibration_report = _read_optional_mapping(calibration_report_path)
    deployment_allowed_statuses = {
        "completed",
        "ready_for_real_world_validation",
        "review_required",
        "no_followup_required",
        "not_requested",
        "not_measured",
        "blocked_insufficient_anchor_count",
        "blocked_anchor_quality",
    }
    deployment_join_blockers: List[str] = []
    if not deployment_intake_path.is_file():
        deployment_join_blockers.append("deployment_outcome_intake_manifest_missing")
    elif deployment_intake.get("schema_version") != "deployment_outcome_intake_manifest.v1":
        deployment_join_blockers.append("deployment_outcome_intake_manifest_schema_invalid")
    elif _string(deployment_intake.get("status")) not in deployment_allowed_statuses:
        deployment_join_blockers.append("deployment_outcome_intake_manifest_status_invalid")
    if not deployment_ledger_path.is_file():
        deployment_join_blockers.append("deployment_outcome_ledger_missing")
    elif deployment_ledger.get("schema_version") != "deployment_outcome_ledger.v1":
        deployment_join_blockers.append("deployment_outcome_ledger_schema_invalid")
    elif _string(deployment_ledger.get("status")) not in deployment_allowed_statuses:
        deployment_join_blockers.append("deployment_outcome_ledger_status_invalid")
    if not deployment_summary_path.is_file():
        deployment_join_blockers.append("prediction_vs_actual_deployment_summary_missing")
    elif deployment_summary.get("schema_version") != "prediction_vs_actual_deployment_summary.v1":
        deployment_join_blockers.append("prediction_vs_actual_deployment_summary_schema_invalid")
    elif _string(deployment_summary.get("status")) not in deployment_allowed_statuses:
        deployment_join_blockers.append("prediction_vs_actual_deployment_summary_status_invalid")
    if not calibration_report_path.is_file():
        deployment_join_blockers.append("sim_vs_real_calibration_report_missing")
    elif calibration_report.get("schema_version") != "sim_vs_real_calibration_report.v1":
        deployment_join_blockers.append("sim_vs_real_calibration_report_schema_invalid")
    elif _string(calibration_report.get("status")) not in deployment_allowed_statuses:
        deployment_join_blockers.append("sim_vs_real_calibration_report_status_invalid")

    checks = [
        _readiness_check(
            check_id="production_or_staging_webapp_request_ids",
            label="real production/staging WebApp request IDs",
            passed=bool(_mapping(gates.get("webapp_upstream_truth")).get("passed")),
            blockers=_prefixed_gate_blockers(gates, ("webapp_upstream_truth",)),
            evidence={
                "gate_id": "webapp_upstream_truth",
                "ids": _mapping(
                    _mapping(gates.get("webapp_upstream_truth")).get("evidence")
                ).get("ids", {}),
            },
        ),
        _readiness_check(
            check_id="real_capture_root_input",
            label="real capture-root input",
            passed=bool(_mapping(gates.get("site_capture")).get("passed")),
            blockers=_prefixed_gate_blockers(gates, ("site_capture",)),
            evidence={
                "gate_id": "site_capture",
                "capture_root": _mapping(
                    _mapping(gates.get("site_capture")).get("evidence")
                ).get("capture_root"),
            },
        ),
        _readiness_check(
            check_id="live_provider_worker_execution",
            label="live provider/worker execution",
            passed=bool(remote_closure.get("remote_cloud_execution_proven")),
            blockers=(
                live_provider_worker_blockers
                if remote_closure
                else ["remote_cloud_execution_closure_manifest_missing"]
            ),
            evidence={
                "remote_cloud_execution_closure_manifest": _artifact(
                    remote_closure_path,
                    base_dir=job_dir,
                ),
                "remote_cloud_execution_proven": bool(
                    remote_closure.get("remote_cloud_execution_proven")
                ),
                "live_provider_calls_performed": bool(
                    remote_closure.get("live_provider_calls_performed")
                ),
            },
        ),
        _readiness_check(
            check_id="fresh_live_robot_team_run_today",
            label="fresh live robot-team run today",
            passed=fresh_live_run_today_ready,
            blockers=(
                [
                    blocker
                    for blocker in (
                        None
                        if remote_closure.get("remote_cloud_execution_proven")
                        else "fresh_remote_cloud_execution_not_proven",
                        None
                        if remote_closure.get("live_provider_calls_performed")
                        else "fresh_live_provider_calls_not_performed",
                        "remote_closure_generated_at_missing"
                        if not remote_closure_generated_at
                        else None,
                        None
                        if remote_closure_generated_date == today_utc
                        else "remote_closure_not_generated_today",
                    )
                    if blocker
                ]
                if remote_closure
                else ["remote_cloud_execution_closure_manifest_missing"]
            ),
            evidence={
                "remote_cloud_execution_closure_manifest": _artifact(
                    remote_closure_path,
                    base_dir=job_dir,
                ),
                "remote_closure_generated_at": remote_closure_generated_at or None,
                "remote_closure_generated_date": remote_closure_generated_date or None,
                "today_utc": today_utc,
                "remote_cloud_execution_proven": bool(
                    remote_closure.get("remote_cloud_execution_proven")
                ),
                "live_provider_calls_performed": bool(
                    remote_closure.get("live_provider_calls_performed")
                ),
            },
        ),
        _readiness_check(
            check_id="writable_artifact_output_uri",
            label="writable artifact-output URI proof",
            passed=remote_artifact_output_ready,
            blockers=(
                [
                    blocker
                    for blocker in (
                        None if remote_artifact_output_uri else "remote_artifact_output_uri_missing",
                        None
                        if remote_checks.get("artifact_output_uri_provider_writable")
                        else "remote_artifact_output_uri_not_provider_writable",
                        None
                        if remote_checks.get("artifact_output_write_auth_contract_ready")
                        else "remote_artifact_output_write_auth_contract_missing",
                        None
                        if remote_provider_inputs_uploaded is True
                        else "remote_artifact_output_upload_not_proven",
                        *[
                            f"provider_input_setup:{blocker}"
                            for blocker in remote_provider_setup_blockers
                        ],
                    )
                    if blocker
                ]
                if remote_closure
                else ["remote_cloud_execution_closure_manifest_missing"]
            ),
            evidence={
                "artifact_output_uri": remote_artifact_output_uri or None,
                "artifact_output_uri_provider_writable": bool(
                    remote_checks.get("artifact_output_uri_provider_writable")
                ),
                "artifact_output_write_auth_contract_ready": bool(
                    remote_checks.get("artifact_output_write_auth_contract_ready")
                ),
                "provider_input_setup": {
                    "status": remote_provider_setup.get("status"),
                    "provider_inputs_uploaded": remote_provider_inputs_uploaded,
                    "blockers": remote_provider_setup_blockers,
                    "manifest_path": remote_provider_setup.get("manifest_path"),
                }
                if remote_provider_setup
                else {},
                "artifact_output_write_probe": {
                    "artifact": _artifact(
                        artifact_output_write_probe_path,
                        base_dir=job_dir,
                    ),
                    "status": artifact_output_write_probe.get("status"),
                    "writable_candidate_count": artifact_output_write_probe.get(
                        "writable_candidate_count"
                    ),
                    "blocked_candidate_count": artifact_output_write_probe.get(
                        "blocked_candidate_count"
                    ),
                }
                if artifact_output_write_probe
                else {},
            },
        ),
        _readiness_check(
            check_id="shutdown_and_cost_proof",
            label="shutdown and cost proof",
            passed=remote_shutdown_cost_ready,
            blockers=(
                [
                    blocker
                    for blocker in (
                        None
                        if remote_closure.get("clean_shutdown_proven")
                        else "remote_clean_shutdown_not_proven",
                        None
                        if remote_checks.get("actual_gpu_time_record_present")
                        else "remote_actual_gpu_time_not_recorded",
                        None
                        if remote_cost.get("actual_gpu_seconds") is not None
                        else "remote_actual_gpu_seconds_missing",
                    )
                    if blocker
                ]
                if remote_closure
                else ["remote_cloud_execution_closure_manifest_missing"]
            ),
            evidence={
                "remote_cloud_execution_closure_manifest": _artifact(
                    remote_closure_path,
                    base_dir=job_dir,
                ),
                "gpu_cost_control_ledger": _artifact(gpu_cost_path, base_dir=job_dir),
                "gpu_cost_control_ledger_status": gpu_cost.get("status"),
                "clean_shutdown_proven": bool(remote_closure.get("clean_shutdown_proven")),
                "actual_gpu_seconds": remote_cost.get("actual_gpu_seconds"),
                "runpod_zero_active_pods_proof": {
                    "artifact": _artifact(runpod_proof_path, base_dir=job_dir)
                    if runpod_proof_path
                    else None,
                    "status": runpod_proof.get("status"),
                    "api_call_performed": bool(runpod_proof.get("api_call_performed")),
                    "side_effects_may_have_occurred": bool(
                        runpod_proof.get("runpod_side_effects_may_have_occurred")
                    ),
                    "active_pod_count_after": runpod_proof.get("active_pod_count_after"),
                    "zero_active_pods_now": runpod_zero_active_pods_now,
                    "claim_boundary": (
                        "Zero active pods now is spend hygiene evidence. It is not "
                        "clean shutdown proof for a fresh worker run unless tied to "
                        "that run's provider lifecycle evidence."
                    ),
                },
            },
        ),
        _readiness_check(
            check_id="simulator_robot_pov_policy_artifacts",
            label="simulator robot POV and reference policy artifacts",
            passed=simulator_pov_policy_ready,
            blockers=[
                blocker
                for blocker in (
                    None
                    if robot_pov_generation_gate.get("passed")
                    else "sim_robot_pov_generation_not_complete",
                    None
                    if _string(
                        policy_execution_evidence.get("policy_execution_manifest_status")
                    )
                    == "completed"
                    else "sim_policy_execution_manifest_not_completed",
                    None
                    if _string(policy_execution_evidence.get("policy_execution_trace_status"))
                    == "completed"
                    else "sim_policy_execution_trace_not_completed",
                    None
                    if int(sim_policy_attempt_count or 0) > 0
                    else "sim_policy_execution_trace_empty",
                    None
                    if sim_policy_attempt_count == sim_policy_run_count
                    else "sim_policy_execution_attempt_count_mismatch",
                    None
                    if sim_policy_missing_run_count == 0
                    else "sim_policy_execution_missing_scenario_eval_run_ids",
                    None
                    if not sim_policy_attempts_missing_trace
                    else "sim_policy_attempts_missing_action_or_skill_trace",
                )
                if blocker
            ],
            evidence={
                "robot_pov_generation_gate_passed": bool(
                    robot_pov_generation_gate.get("passed")
                ),
                "robot_pov_observation_manifest": _mapping(
                    robot_pov_generation_gate.get("evidence")
                ).get("artifact"),
                "robot_pov_observation_count": _mapping(
                    robot_pov_generation_gate.get("evidence")
                ).get("observation_count"),
                "policy_execution_manifest": policy_execution_evidence.get(
                    "policy_execution_manifest"
                ),
                "policy_execution_trace": policy_execution_evidence.get(
                    "policy_execution_trace"
                ),
                "policy_execution_manifest_status": policy_execution_evidence.get(
                    "policy_execution_manifest_status"
                ),
                "policy_execution_trace_status": policy_execution_evidence.get(
                    "policy_execution_trace_status"
                ),
                "attempt_count": sim_policy_attempt_count,
                "scenario_eval_run_count": sim_policy_run_count,
                "missing_scenario_eval_run_count": sim_policy_missing_run_count,
                "attempts_missing_action_or_skill_trace": sim_policy_attempts_missing_trace,
                "proof_boundary": (
                    "This check proves simulator-generated POV plus reference policy "
                    "trace coverage only. It does not satisfy real-robot POV, live "
                    "provider execution, or robot-team policy execution."
                ),
            },
        ),
        _readiness_check(
            check_id="robot_pov_policy_evidence",
            label="robot POV/action-log and policy evidence",
            passed=robot_pov_policy_ready,
            blockers=robot_pov_policy_blockers,
            evidence={
                "satisfied_by": (
                    "simulator_robot_pov_and_reference_policy"
                    if simulator_pov_policy_ready
                    else "real_robot_pov_and_live_policy_execution"
                    if real_robot_pov_live_policy_ready
                    else "not_satisfied"
                ),
                "simulator_robot_pov_policy_artifacts_passed": simulator_pov_policy_ready,
                "real_robot_pov_gate_passed": bool(real_robot_pov_gate.get("passed")),
                "real_robot_pov_proof_boolean": real_robot_pov_proof_ready,
                "live_policy_execution_gate_passed": bool(policy_execution_gate.get("passed")),
                "proof_boundary": (
                    "Simulator beta accepts simulator-generated POV plus reference "
                    "policy traces for this check. Real robot POV/action-log evidence "
                    "remains governed by the top-level real_robot_pov_evidence and "
                    "live_policy_execution gates."
                ),
            },
        ),
        _readiness_check(
            check_id="deployment_outcome_joins",
            label="deployment outcome joins and sim-vs-real calibration",
            passed=True,
            blockers=[],
            evidence={
                "deployment_outcome_intake_manifest": _artifact(
                    deployment_intake_path,
                    base_dir=job_dir,
                ),
                "deployment_outcome_ledger": _artifact(
                    deployment_ledger_path,
                    base_dir=job_dir,
                ),
                "prediction_vs_actual_deployment_summary": _artifact(
                    deployment_summary_path,
                    base_dir=job_dir,
                ),
                "sim_vs_real_calibration_report": _artifact(
                    calibration_report_path,
                    base_dir=job_dir,
                ),
                "deployment_outcome_intake_status": deployment_intake.get("status"),
                "deployment_outcome_ledger_status": deployment_ledger.get("status"),
                "prediction_summary_status": deployment_summary.get("status"),
                "calibration_report_status": calibration_report.get("status"),
                "matched_prediction_record_count": (
                    deployment_summary.get("matched_prediction_record_count")
                    or calibration_report.get("matched_prediction_record_count")
                    or 0
                ),
                "exact_prediction_record_count": (
                    deployment_summary.get("exact_prediction_record_count")
                    or calibration_report.get("exact_prediction_record_count")
                    or 0
                ),
                "real_world_outcome_record_count": int(
                    deployment_intake.get("record_count")
                    or deployment_ledger.get("record_count")
                    or 0
                ),
                "real_world_validation_gate_passed": bool(
                    _mapping(gates.get("real_world_validation_loop")).get("proof_boolean")
                ),
                "predicted_vs_actual_gate_passed": bool(
                    _mapping(gates.get("predicted_vs_actual_calibration")).get("proof_boolean")
                ),
                "optional_for_sim_only": True,
                "diagnostic_blockers": deployment_join_blockers,
                "proof_boundary": (
                    "Deployment/prediction join artifacts are optional sim-vs-real "
                    "calibration diagnostics for sim-only beta runs. Missing ledgers do "
                    "not claim or disprove real-world deployment outcomes."
                ),
            },
        ),
        _readiness_check(
            check_id="final_live_closure_artifacts",
            label="final live closure artifacts",
            passed=final_live_closure_artifacts_ready,
            blockers=[]
            if final_live_closure_artifacts_ready
            else [
                blocker
                for blocker in (
                    None
                    if remote_closure.get("status")
                    == "remote_execution_completed_with_shutdown_proof"
                    else "remote_cloud_execution_closure_not_completed",
                    None
                    if provider_runtime_finalizer_proof.get("status") == "completed"
                    else "provider_runtime_finalizer_proof_not_completed",
                    None if gpu_cost_path.is_file() else "gpu_cost_control_ledger_missing",
                    None
                    if worker_runtime_manifest_path.is_file()
                    else "worker_runtime_manifest_missing",
                    None if job_run_manifest_path.is_file() else "job_run_manifest_missing",
                    None if robot_eval_report_path.is_file() else "robot_eval_report_missing",
                )
                if blocker
            ],
            evidence={
                "repo_local_artifacts_ready": repo_local_ready,
                "live_external_ready": live_external_ready,
                "live_end_to_end_verified": live_end_to_end_verified,
                "remote_cloud_execution_closure_manifest": _artifact(
                    remote_closure_path,
                    base_dir=job_dir,
                ),
                "provider_runtime_finalizer_proof": _artifact(
                    provider_runtime_finalizer_proof_path,
                    base_dir=job_dir,
                ),
                "gpu_cost_control_ledger": _artifact(gpu_cost_path, base_dir=job_dir),
                "worker_runtime_manifest": _artifact(
                    worker_runtime_manifest_path,
                    base_dir=job_dir,
                ),
                "job_run_manifest": _artifact(job_run_manifest_path, base_dir=job_dir),
        "robot_eval_report": _artifact(robot_eval_report_path, base_dir=job_dir),
        "proof_boundary": (
            "This beta check covers final live run closure artifacts. "
            "It does not upgrade public or production claims beyond the "
            "referenced simulator-beta artifacts."
        ),
            },
        ),
    ]
    blocked_checks = [check["check_id"] for check in checks if not check["passed"]]
    return {
        "schema_version": "robot_team_beta_readiness_summary.v1",
        "ready_for_beta": not blocked_checks,
        "blocked_check_ids": blocked_checks,
        "passed_check_ids": [check["check_id"] for check in checks if check["passed"]],
        "checks": checks,
        "claim_boundary": (
            "This beta summary is a closure checklist. It does not run providers, mutate "
            "WebApp state, or upgrade proof beyond referenced artifacts."
        ),
    }


def _local_reference_path(value: Any, *, capture_root: Path, job_dir: Path) -> Path | None:
    text = _string(value)
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
    job_candidate = job_dir / path
    if job_candidate.exists():
        return job_candidate
    return capture_root / path


def _automation_local_reference_path(
    value: Any,
    *,
    capture_root: Path,
    automation_dir: Path,
) -> Path | None:
    text = _string(value)
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
    automation_candidate = automation_dir / path
    if automation_candidate.exists():
        return automation_candidate
    if path.parts and path.parts[0] in {"pipeline", "raw", "privacy"}:
        return capture_root / path
    return automation_candidate


def _load_reference_mapping(value: Any, *, capture_root: Path, job_dir: Path) -> Dict[str, Any]:
    path = _local_reference_path(value, capture_root=capture_root, job_dir=job_dir)
    if path is None:
        return {}
    return _read_optional_mapping(path)


def _merge_evidence(base: Dict[str, Any], incoming: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in incoming.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _merge_evidence(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _signed_url_list_from_manifest(value: Any) -> List[str]:
    values = _string_list(value)
    return [item for item in values if item.startswith("http://") or item.startswith("https://")]


def _signed_access_manifest_delivery_evidence(payload: Mapping[str, Any]) -> Dict[str, Any]:
    command_output = _mapping(
        _field(payload, "delivery_command_output", "deliveryCommandOutput")
    )
    buyer_access_check = _mapping(
        _field(
            payload,
            "buyer_access_check",
            "buyerAccessCheck",
            "authenticated_fetch",
            "authenticatedFetch",
        )
    ) or _mapping(
        _field(
            command_output,
            "buyer_access_check",
            "buyerAccessCheck",
            "authenticated_fetch",
            "authenticatedFetch",
        )
    )
    signed_urls = _signed_url_list_from_manifest(
        _field(payload, "signed_urls", "signedUrls")
    ) or _signed_url_list_from_manifest(
        _field(command_output, "signed_urls", "signedUrls")
    )
    signed_url = _string(_field(payload, "signed_url", "signedUrl")) or _string(
        _field(command_output, "signed_url", "signedUrl")
    )
    if signed_url.startswith("http://") or signed_url.startswith("https://"):
        signed_urls = [*signed_urls, signed_url]
    signed_access = _string_list(
        _field(payload, "signed_access", "signedAccess")
        or _field(command_output, "signed_access", "signedAccess")
    )
    entitlement_verified = _boolish(
        _field(payload, "entitlement_verified", "entitlementVerified")
        or _field(command_output, "entitlement_verified", "entitlementVerified")
        or _field(buyer_access_check, "entitlement_verified", "entitlementVerified")
    )
    storage_upload_performed = _boolish(
        _field(payload, "storage_upload_performed", "storageUploadPerformed")
        or _field(command_output, "storage_upload_performed", "storageUploadPerformed")
    )
    operator_attestation = (
        _field(payload, "operator_attestation", "operatorAttestation")
        or _field(command_output, "operator_attestation", "operatorAttestation")
    )
    delivery: Dict[str, Any] = {
        "storage_upload_performed": storage_upload_performed,
        "signed_urls": signed_urls,
        "signed_access": signed_access,
        "entitlement_verified": entitlement_verified,
        "buyer_access_check": buyer_access_check,
        "operator_attestation": operator_attestation,
        "signed_access_manifest_status": _string(_field(payload, "status")) or None,
    }
    return {"delivery": delivery}


def _load_live_evidence(
    *,
    capture_root: Path,
    job_dir: Path,
    job_request: Mapping[str, Any],
) -> tuple[Dict[str, Any], List[str]]:
    evidence: Dict[str, Any] = {}
    sources: List[str] = []
    job_id = job_dir.name

    def evidence_job_id(payload: Mapping[str, Any]) -> str:
        return _string(
            payload.get("job_id")
            or payload.get("jobId")
            or payload.get("robot_eval_job_id")
            or payload.get("robotEvalJobId")
        )

    input_blockers: List[Dict[str, Any]] = []

    def consume_payload(payload: Mapping[str, Any], source: str) -> None:
        declared_job_id = evidence_job_id(payload)
        declared_schema = _string(payload.get("schema_version") or payload.get("schemaVersion"))
        if declared_job_id and declared_job_id != job_id:
            input_blockers.append(
                {
                    "blocker": "live_closure_evidence_job_id_mismatch",
                    "source": source,
                    "expected_job_id": job_id,
                    "declared_job_id": declared_job_id,
                }
            )
            return
        if declared_schema and declared_schema != LIVE_ROBOT_EVAL_EVIDENCE_SCHEMA_VERSION:
            input_blockers.append(
                {
                    "blocker": "live_closure_evidence_schema_mismatch",
                    "source": source,
                    "expected_schema_version": LIVE_ROBOT_EVAL_EVIDENCE_SCHEMA_VERSION,
                    "declared_schema_version": declared_schema,
                }
            )
            return
        _merge_evidence(evidence, payload)
        sources.append(source)

    for key in (
        "live_eval_closure_evidence",
        "liveEvalClosureEvidence",
        "owner_rank_fidelity_evidence",
        "ownerRobotReadinessEvidence",
    ):
        value = job_request.get(key)
        if isinstance(value, Mapping):
            consume_payload(value, f"job_request_inline:{key}")
    for key in (
        "live_eval_closure_evidence_uri",
        "liveEvalClosureEvidenceUri",
        "owner_evidence_manifest_uri",
        "ownerEvidenceManifestUri",
        "rank_fidelity_proof_uri",
        "robotReadinessProofUri",
        "signed_access_manifest_uri",
        "signedAccessManifestUri",
        "delivery_access_manifest_uri",
        "deliveryAccessManifestUri",
    ):
        payload = _load_reference_mapping(job_request.get(key), capture_root=capture_root, job_dir=job_dir)
        if payload:
            normalized_key = key.lower()
            if (
                "signed_access" in normalized_key
                or "signedaccess" in normalized_key
                or "delivery_access" in normalized_key
                or "deliveryaccess" in normalized_key
            ):
                consume_payload(
                    _signed_access_manifest_delivery_evidence(payload),
                    f"job_request_ref:{key}",
                )
            else:
                consume_payload(payload, f"job_request_ref:{key}")
    for path in (
        job_dir / "live_eval_closure_evidence.json",
        job_dir / "owner_rank_fidelity_evidence.json",
        job_dir / "signed_access_manifest.json",
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / job_id
        / "live_eval_closure_evidence.json",
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / job_id
        / "owner_rank_fidelity_evidence.json",
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / job_id
        / "signed_access_manifest.json",
        capture_root / "pipeline" / "robot_eval_inputs" / "live_eval_closure_evidence.json",
        capture_root / "pipeline" / "robot_eval_inputs" / "owner_rank_fidelity_evidence.json",
        capture_root / "pipeline" / "robot_eval_inputs" / "signed_access_manifest.json",
    ):
        payload = _read_optional_mapping(path)
        if payload:
            if path.name == "signed_access_manifest.json":
                consume_payload(_signed_access_manifest_delivery_evidence(payload), str(path))
            else:
                consume_payload(payload, str(path))
    evidence.setdefault("schema_version", LIVE_ROBOT_EVAL_EVIDENCE_SCHEMA_VERSION)
    if input_blockers:
        evidence["_input_blockers"] = input_blockers
    return evidence, sources


def _live_evidence_integrity_gate(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    input_blockers = [
        dict(item)
        for item in evidence.get("_input_blockers", []) or []
        if isinstance(item, Mapping)
    ]
    blockers = [_string(item.get("blocker")) for item in input_blockers]
    return _gate(
        "live_evidence_integrity",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "input_blockers": input_blockers,
            "input_blocker_count": len(input_blockers),
        },
    )


def _source_payloads(
    *,
    capture_root: Path,
    job_dir: Path,
    job_request: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> List[tuple[str, Mapping[str, Any]]]:
    descriptor = _read_optional_mapping(capture_root / "capture_descriptor.json")
    raw_manifest = _read_optional_mapping(capture_root / "raw" / "manifest.json")
    raw_handoff = _mapping(raw_manifest.get("upstream_handoff"))
    opportunity = _read_optional_mapping(capture_root / "pipeline" / "opportunity_handoff.json")
    webapp_sync = _read_optional_mapping(capture_root / "pipeline" / "webapp_sync_result.json")
    site_package = _mapping(job_request.get("site_package") or job_request.get("sitePackage"))
    owner_system = _mapping(job_request.get("owner_system") or job_request.get("ownerSystem"))
    source = _mapping(
        job_request.get("source")
        or job_request.get("webapp_source")
        or job_request.get("webappSource")
    )
    selection = _mapping(source.get("selection_state") or source.get("selectionState"))
    payloads: List[tuple[str, Mapping[str, Any]]] = [
        ("live_evidence.webapp_upstream", _mapping(evidence.get("webapp_upstream"))),
        ("job_request", job_request),
        ("job_request.site_package", site_package),
        ("capture_descriptor", descriptor),
        ("raw_manifest", raw_manifest),
        ("raw_manifest.upstream_handoff", raw_handoff),
        ("pipeline.opportunity_handoff", opportunity),
        ("pipeline.webapp_sync_result", webapp_sync),
    ]
    if _request_has_webapp_source(job_request):
        payloads[3:3] = [
            ("job_request.owner_system", owner_system),
            ("job_request.source.selection_state", selection),
        ]
    return payloads


def _capture_lineage_from_path_text(value: Any) -> tuple[str, str] | None:
    text = _string(value)
    if not text:
        return None
    if text.startswith("file://"):
        text = text[7:]
    elif "://" in text:
        parsed = urlparse(text)
        text = f"{parsed.netloc}{parsed.path}"
    parts = [part for part in text.replace("\\", "/").split("/") if part]
    try:
        scenes_index = parts.index("scenes")
    except ValueError:
        return None
    if len(parts) <= scenes_index + 3 or parts[scenes_index + 2] != "captures":
        return None
    return parts[scenes_index + 1], parts[scenes_index + 3]


def _webapp_route_forwarding_status_ok(proof: Mapping[str, Any]) -> bool:
    status = _string(proof.get("status"))
    pipeline_intake = _mapping(proof.get("pipeline_intake"))
    pipeline_forward = _mapping(proof.get("pipeline_forward"))
    durable_store = _mapping(proof.get("durable_store"))
    durable_forward = _mapping(durable_store.get("pipeline_forward"))
    return (
        status in WEBAPP_ROUTE_FORWARDING_PROOF_STATUSES
        or _string(pipeline_intake.get("status")) in WEBAPP_ROUTE_FORWARDING_PROOF_STATUSES
        or _string(pipeline_forward.get("pipeline_status")) in WEBAPP_ROUTE_FORWARDING_PROOF_STATUSES
        or _string(durable_forward.get("pipeline_status")) in WEBAPP_ROUTE_FORWARDING_PROOF_STATUSES
    )


def _webapp_route_forwarding_flat_payload(
    *,
    proof: Mapping[str, Any],
    job_request: Mapping[str, Any],
) -> Dict[str, Any]:
    site_package = _mapping(job_request.get("site_package") or job_request.get("sitePackage"))
    owner_system = _mapping(job_request.get("owner_system") or job_request.get("ownerSystem"))
    source = _mapping(job_request.get("source") or job_request.get("webapp_source") or job_request.get("webappSource"))
    selection = _mapping(source.get("selection_state") or source.get("selectionState"))
    stored_request_doc_id = _webapp_route_forwarding_stored_request_doc_id(proof)
    out: Dict[str, Any] = {}
    for field in WEBAPP_UPSTREAM_ID_FIELDS:
        out[field] = (
            job_request.get(field)
            or site_package.get(field)
            or owner_system.get(field)
            or selection.get(field)
            or proof.get(field)
            or (stored_request_doc_id if field == "request_id" else "")
            or ""
        )
    return out


def _webapp_route_forwarding_stored_request_doc_id(proof: Mapping[str, Any]) -> str:
    durable_store = _mapping(proof.get("durable_store") or proof.get("durableStore"))
    firestore = _mapping(durable_store.get("firestore"))
    status = _string(firestore.get("status")).lower()
    if status != "stored":
        return ""
    return _string(
        firestore.get("doc_id")
        or firestore.get("docId")
        or firestore.get("document_id")
        or firestore.get("documentId")
    )


def _webapp_route_forwarding_id_source_fields(
    *,
    proof: Mapping[str, Any],
    job_request: Mapping[str, Any],
) -> Dict[str, str]:
    site_package = _mapping(job_request.get("site_package") or job_request.get("sitePackage"))
    owner_system = _mapping(job_request.get("owner_system") or job_request.get("ownerSystem"))
    source = _mapping(job_request.get("source") or job_request.get("webapp_source") or job_request.get("webappSource"))
    selection = _mapping(source.get("selection_state") or source.get("selectionState"))
    stored_request_doc_id = _webapp_route_forwarding_stored_request_doc_id(proof)
    source_fields: Dict[str, str] = {}
    for field in WEBAPP_UPSTREAM_ID_FIELDS:
        if _string(job_request.get(field)):
            source_fields[field] = f"job_request.{field}"
        elif _string(site_package.get(field)):
            source_fields[field] = f"job_request.site_package.{field}"
        elif _string(owner_system.get(field)):
            source_fields[field] = f"job_request.owner_system.{field}"
        elif _string(selection.get(field)):
            source_fields[field] = f"job_request.source.selection_state.{field}"
        elif _string(proof.get(field)):
            source_fields[field] = field
        elif field == "request_id" and stored_request_doc_id:
            source_fields[field] = "durable_store.firestore.doc_id"
    return source_fields


def _webapp_route_forwarding_source_payloads(
    *,
    capture_root: Path,
    job_dir: Path,
    scene_id: str,
    capture_id: str,
) -> tuple[List[tuple[str, Mapping[str, Any]]], List[Dict[str, Any]], set[str]]:
    proof_dir = capture_root / "pipeline" / "webapp_route_forwarding_proof"
    candidates: List[tuple[str, Mapping[str, Any]]] = []
    audits: List[Dict[str, Any]] = []
    grounded_sources: set[str] = set()
    if not proof_dir.is_dir():
        return candidates, audits, grounded_sources

    expected_lineage = (scene_id, capture_id)
    for path in sorted(proof_dir.glob("*.json")):
        proof = _read_optional_mapping(path)
        if not proof:
            continue
        job_request = _mapping(proof.get("job_request") or proof.get("jobRequest"))
        site_package = _mapping(job_request.get("site_package") or job_request.get("sitePackage"))
        proof_job_id = _string(
            job_request.get("job_id")
            or job_request.get("jobId")
            or proof.get("job_id")
            or proof.get("jobId")
        )
        lineage = (
            _capture_lineage_from_path_text(
                proof.get("capture_root")
                or proof.get("captureRoot")
                or site_package.get("capture_root")
                or site_package.get("captureRoot")
            )
            or (
                _string(site_package.get("scene_id") or site_package.get("sceneId")),
                _string(site_package.get("capture_id") or site_package.get("captureId")),
            )
        )
        lineage_matches = lineage == expected_lineage
        status_ok = _webapp_route_forwarding_status_ok(proof)
        job_id_matches = bool(proof_job_id) and proof_job_id == job_dir.name
        source = f"pipeline.webapp_route_forwarding_proof:{path.name}"
        payload = _webapp_route_forwarding_flat_payload(
            proof=proof,
            job_request=job_request,
        )
        id_source_fields = _webapp_route_forwarding_id_source_fields(
            proof=proof,
            job_request=job_request,
        )
        grounding_verified = bool(status_ok and lineage_matches and job_id_matches)
        if job_id_matches:
            candidates.append((source, payload))
        if grounding_verified:
            grounded_sources.add(source)
        audits.append(
            {
                "source": source,
                "path": _relative_to(job_dir, path),
                "status": _string(proof.get("status")) or None,
                "pipeline_intake_status": _string(
                    _mapping(proof.get("pipeline_intake")).get("status")
                )
                or None,
                "proof_job_id": proof_job_id or None,
                "expected_job_id": job_dir.name,
                "job_id_matches": job_id_matches,
                "lineage": {"scene_id": lineage[0], "capture_id": lineage[1]}
                if lineage
                else None,
                "expected_lineage": {"scene_id": scene_id, "capture_id": capture_id},
                "lineage_matches": lineage_matches,
                "status_ok": status_ok,
                "grounding_verified": grounding_verified,
                "ids_present": {
                    field: bool(_string(payload.get(field)))
                    for field in WEBAPP_UPSTREAM_ID_FIELDS
                },
                "id_source_fields": id_source_fields,
            }
        )
    return candidates, audits, grounded_sources


def _request_capture_root_matches(job_request: Mapping[str, Any], capture_root: Path) -> bool:
    site_package = _mapping(job_request.get("site_package") or job_request.get("sitePackage"))
    value = _string(site_package.get("capture_root") or site_package.get("captureRoot"))
    if not value:
        return False
    if value.startswith("file://"):
        value = value[7:]
    elif "://" in value:
        return False
    try:
        return Path(value).expanduser().resolve() == capture_root.resolve()
    except OSError:
        return False


def _request_has_webapp_source(job_request: Mapping[str, Any]) -> bool:
    source = _mapping(job_request.get("source") or job_request.get("webapp_source") or job_request.get("webappSource"))
    system = _string(source.get("system") or source.get("source_system") or source.get("sourceSystem")).lower()
    route = _string(source.get("route") or source.get("path"))
    return "webapp" in system or route.startswith("/sites")


def _webapp_upstream_gate(
    *,
    capture_root: Path,
    job_dir: Path,
    job_request: Mapping[str, Any],
    evidence: Mapping[str, Any],
    scene_id: str,
    capture_id: str,
) -> Dict[str, Any]:
    candidates = _source_payloads(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request=job_request,
        evidence=evidence,
    )
    (
        route_proof_candidates,
        route_proof_audits,
        grounded_route_proof_sources,
    ) = _webapp_route_forwarding_source_payloads(
        capture_root=capture_root,
        job_dir=job_dir,
        scene_id=scene_id,
        capture_id=capture_id,
    )
    candidates.extend(route_proof_candidates)
    ids: Dict[str, str] = {}
    id_sources: Dict[str, str] = {}
    source_values: Dict[str, Dict[str, str]] = {}
    grounded_sources_by_field: Dict[str, List[str]] = {}
    mismatch_fields: List[str] = []
    ungrounded_fields: List[str] = []
    request_capture_root_matches = _request_capture_root_matches(job_request, capture_root)
    request_has_webapp_source = _request_has_webapp_source(job_request)
    request_source_verified = request_capture_root_matches and request_has_webapp_source
    for field in WEBAPP_UPSTREAM_ID_FIELDS:
        values: Dict[str, str] = {}
        for source, payload in candidates:
            value = _string(payload.get(field))
            if value:
                values[source] = value
            if value and not ids.get(field):
                ids[field] = value
                id_sources[field] = source
        ids.setdefault(field, "")
        id_sources.setdefault(field, "")
        source_values[field] = values
        unique_values = sorted({value for value in values.values() if value})
        if len(unique_values) > 1:
            mismatch_fields.append(field)
        grounded_sources = [
            source
            for source, value in values.items()
            if value == ids[field]
            and (
                source in WEBAPP_UPSTREAM_CAPTURE_GROUNDING_SOURCES
                or (source.startswith("job_request") and request_source_verified)
                or source in grounded_route_proof_sources
            )
        ]
        grounded_sources_by_field[field] = grounded_sources
        if ids[field] and not grounded_sources:
            ungrounded_fields.append(field)
    blockers: List[str] = []
    for field, value in ids.items():
        if not value:
            blockers.append(f"missing_webapp_{field}")
        elif _contains_placeholder_id(value):
            blockers.append(f"placeholder_webapp_{field}")
        elif value in {capture_id, f"{scene_id}:{capture_id}", f"{scene_id}/{capture_id}"}:
            blockers.append(f"generated_capture_id_used_for_webapp_{field}")
    if mismatch_fields:
        blockers.append("webapp_upstream_source_mismatch")
    if ungrounded_fields:
        blockers.append("webapp_upstream_ids_not_grounded_in_capture_or_webapp_source")
    return _gate(
        "webapp_upstream_truth",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "ids": ids,
            "id_sources": id_sources,
            "source_values": source_values,
            "grounded_sources_by_field": grounded_sources_by_field,
            "mismatch_fields": sorted(mismatch_fields),
            "ungrounded_fields": sorted(ungrounded_fields),
            "job_request_capture_root_matches": request_capture_root_matches,
            "job_request_webapp_source_present": request_has_webapp_source,
            "webapp_route_forwarding_proofs": route_proof_audits,
        },
    )


def _bool_field_from_sources(
    *sources: Mapping[str, Any],
    snake_key: str,
    camel_key: str,
) -> Any:
    for source in sources:
        if snake_key in source:
            return source.get(snake_key)
        if camel_key in source:
            return source.get(camel_key)
    return None


def _rights_evidence_ref_audit(
    *,
    scope: Mapping[str, Any],
    evidence_scope: Mapping[str, Any],
    capture_root: Path,
    job_dir: Path,
) -> Dict[str, Any]:
    ref_keys = (
        "evidence_uri_or_path",
        "evidenceUriOrPath",
        "rights_evidence_uri_or_path",
        "rightsEvidenceUriOrPath",
        "privacy_evidence_uri_or_path",
        "privacyEvidenceUriOrPath",
        "clearance_uri_or_path",
        "clearanceUriOrPath",
        "consent_manifest_uri_or_path",
        "consentManifestUriOrPath",
        "evidence_uri",
        "evidenceUri",
        "proof_uri",
        "proofUri",
    )
    refs: Dict[str, str] = {}
    for payload_name, payload in (("job_request", scope), ("live_evidence", evidence_scope)):
        for key in ref_keys:
            value = _string(payload.get(key))
            if value:
                refs[f"{payload_name}.{key}"] = value
    local_ref_artifacts: Dict[str, Dict[str, Any]] = {}
    missing_local_ref_keys: List[str] = []
    invalid_remote_ref_keys: List[str] = []
    proven_ref_keys: List[str] = []
    for key, value in refs.items():
        local_path = _local_reference_path(value, capture_root=capture_root, job_dir=job_dir)
        if local_path is not None:
            artifact = _artifact(local_path, base_dir=job_dir)
            local_ref_artifacts[key] = artifact
            if artifact.get("exists"):
                proven_ref_keys.append(key)
            else:
                missing_local_ref_keys.append(key)
            continue
        if _external_uri(value) and not _contains_placeholder_id(value):
            proven_ref_keys.append(key)
        else:
            invalid_remote_ref_keys.append(key)
    return {
        "evidence_refs": refs,
        "local_ref_artifacts": local_ref_artifacts,
        "missing_local_ref_keys": sorted(missing_local_ref_keys),
        "invalid_remote_ref_keys": sorted(invalid_remote_ref_keys),
        "proven_ref_keys": sorted(proven_ref_keys),
    }


def _rights_gate(
    *,
    job_request: Mapping[str, Any],
    evidence: Mapping[str, Any],
    capture_root: Path,
    job_dir: Path,
) -> Dict[str, Any]:
    scope = _mapping(job_request.get("rights_privacy_scope") or job_request.get("rightsPrivacyScope"))
    evidence_scope = _mapping(evidence.get("rights_privacy") or evidence.get("rightsPrivacy"))
    status = _string(scope.get("status") or evidence_scope.get("status")).lower()
    external_allowed = _bool_field_from_sources(
        scope,
        evidence_scope,
        snake_key="external_use_allowed",
        camel_key="externalUseAllowed",
    )
    ref_audit = _rights_evidence_ref_audit(
        scope=scope,
        evidence_scope=evidence_scope,
        capture_root=capture_root,
        job_dir=job_dir,
    )
    operator_attestation_present = _attestation_ok(
        scope.get("operator_attestation")
        or scope.get("operatorAttestation")
        or scope.get("owner_attestation")
        or scope.get("ownerAttestation")
        or scope.get("buyer_attestation")
        or scope.get("buyerAttestation")
        or evidence_scope.get("operator_attestation")
        or evidence_scope.get("operatorAttestation")
        or evidence_scope.get("owner_attestation")
        or evidence_scope.get("ownerAttestation")
        or evidence_scope.get("buyer_attestation")
        or evidence_scope.get("buyerAttestation")
    )
    accepted = _boolish(evidence_scope.get("accepted")) or _status_ok(status)
    evidence_proven = operator_attestation_present or bool(ref_audit["proven_ref_keys"])
    blockers: List[str] = []
    if status in {"blocked", "denied", "failed", "missing", "not_allowed", "unsafe"}:
        blockers.append("rights_privacy_scope_blocked")
    if external_allowed is not None and not _boolish(external_allowed):
        blockers.append("rights_privacy_external_use_not_allowed")
    if external_allowed is None:
        blockers.append("rights_privacy_external_use_not_proven")
    if accepted and not evidence_proven:
        blockers.append("rights_privacy_owner_evidence_missing")
    if ref_audit["missing_local_ref_keys"]:
        blockers.append("rights_privacy_local_evidence_refs_missing")
    if ref_audit["invalid_remote_ref_keys"]:
        blockers.append("rights_privacy_evidence_refs_invalid_or_placeholder")
    return _gate(
        "rights_privacy_scope",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "status": status or None,
            "external_use_allowed": external_allowed,
            "accepted": accepted,
            "operator_attestation_present": operator_attestation_present,
            "evidence_proven": evidence_proven,
            "evidence": evidence_scope,
            **ref_audit,
        },
    )


def _evidence_section(evidence: Mapping[str, Any], *keys: str) -> Dict[str, Any]:
    for key in keys:
        section = _mapping(evidence.get(key))
        if section:
            return section
    return {}


def _named_local_ref_audit(
    *,
    section: Mapping[str, Any],
    aliases_by_name: Mapping[str, Sequence[str]],
    capture_root: Path,
    job_dir: Path,
) -> Dict[str, Any]:
    refs: Dict[str, str] = {}
    local_ref_artifacts: Dict[str, Dict[str, Any]] = {}
    missing_local_ref_keys: List[str] = []
    invalid_remote_ref_keys: List[str] = []
    proven_ref_keys: List[str] = []
    for canonical, aliases in aliases_by_name.items():
        value = _field(section, *aliases)
        text = _string(value)
        if text:
            refs[canonical] = text
    for key, value in refs.items():
        local_path = _local_reference_path(value, capture_root=capture_root, job_dir=job_dir)
        if local_path is not None:
            artifact = _artifact(local_path, base_dir=job_dir)
            local_ref_artifacts[key] = artifact
            if artifact.get("exists"):
                proven_ref_keys.append(key)
            else:
                missing_local_ref_keys.append(key)
            continue
        if _external_uri(value) and not _contains_placeholder_id(value):
            proven_ref_keys.append(key)
        else:
            invalid_remote_ref_keys.append(key)
    return {
        "evidence_refs": refs,
        "local_ref_artifacts": local_ref_artifacts,
        "missing_local_ref_keys": sorted(missing_local_ref_keys),
        "invalid_remote_ref_keys": sorted(invalid_remote_ref_keys),
        "proven_ref_keys": sorted(proven_ref_keys),
    }


def _review_acceptance_gate(
    *,
    evidence: Mapping[str, Any],
    capture_root: Path,
    job_dir: Path,
) -> Dict[str, Any]:
    section = _evidence_section(evidence, "review_acceptance", "reviewAcceptance")
    accepted = _boolish(_field(section, "accepted", "reviewAccepted")) or _status_ok(
        _field(section, "status", "reviewStatus")
    )
    reviewer_present = bool(_string(_field(section, "reviewer", "reviewerId", "reviewer_id")))
    attestation_present = _attestation_ok(
        _field(
            section,
            "operator_attestation",
            "operatorAttestation",
            "owner_attestation",
            "ownerAttestation",
        )
    )
    ref_audit = _named_local_ref_audit(
        section=section,
        aliases_by_name={
            "evidence_uri_or_path": (
                "evidence_uri_or_path",
                "evidenceUriOrPath",
                "review_evidence_uri_or_path",
                "reviewEvidenceUriOrPath",
            )
        },
        capture_root=capture_root,
        job_dir=job_dir,
    )
    blockers: List[str] = []
    if not section:
        blockers.append("review_acceptance_evidence_missing")
    elif not accepted:
        blockers.append("review_acceptance_not_accepted")
    if accepted and not reviewer_present:
        blockers.append("review_acceptance_reviewer_missing")
    if accepted and not (reviewer_present or attestation_present or ref_audit["proven_ref_keys"]):
        blockers.append("review_acceptance_owner_evidence_missing")
    if ref_audit["missing_local_ref_keys"]:
        blockers.append("review_acceptance_local_evidence_refs_missing")
    if ref_audit["invalid_remote_ref_keys"]:
        blockers.append("review_acceptance_evidence_refs_invalid_or_placeholder")
    return _gate(
        "review_acceptance",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "accepted": accepted,
            "reviewer_present": reviewer_present,
            "operator_attestation_present": attestation_present,
            **ref_audit,
        },
    )


def _signed_delivery_access_gate(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    section = _evidence_section(evidence, "delivery_access", "deliveryAccess", "delivery")
    signed_urls = section.get("signed_urls") or section.get("signedUrls") or []
    signed_access = section.get("signed_access") or section.get("signedAccess") or []
    signed_url_count = len(signed_urls) if isinstance(signed_urls, list) else 0
    signed_access_count = len(signed_access) if isinstance(signed_access, list) else 0
    storage_uploaded = _boolish(
        _field(section, "storage_upload_performed", "storageUploadPerformed")
    )
    entitlement_verified = _boolish(
        _field(section, "entitlement_verified", "entitlementVerified")
    )
    buyer_access_check = _mapping(
        _field(
            section,
            "buyer_access_check",
            "buyerAccessCheck",
            "authenticated_fetch",
            "authenticatedFetch",
            "executed_access_check",
            "executedAccessCheck",
        )
    )
    buyer_access_checked = _boolish(
        _field(
            buyer_access_check,
            "buyer_access_checked",
            "buyerAccessChecked",
            "authenticated_fetch_executed",
            "authenticatedFetchExecuted",
            "executed",
        )
    )
    buyer_accessible = _boolish(
        _field(
            buyer_access_check,
            "buyer_accessible",
            "buyerAccessible",
            "authenticated_fetch_succeeded",
            "authenticatedFetchSucceeded",
            "accessible",
        )
    )
    buyer_access_status = _string(
        _field(buyer_access_check, "status", "fetch_status", "fetchStatus")
    )
    if buyer_access_status in {"ok", "passed", "accessible"}:
        buyer_access_checked = True
        buyer_accessible = True
    signed_access_ready = bool(signed_url_count or signed_access_count)
    attestation_present = _attestation_ok(
        _field(
            section,
            "operator_attestation",
            "operatorAttestation",
            "owner_attestation",
            "ownerAttestation",
        )
    )
    blockers: List[str] = []
    if not section:
        blockers.append("signed_delivery_evidence_missing")
    if not signed_access_ready:
        blockers.append("signed_delivery_access_not_proven")
    if signed_access_ready and not attestation_present:
        blockers.append("signed_delivery_operator_attestation_missing")
    if section and not entitlement_verified:
        blockers.append("signed_delivery_entitlement_not_verified")
    if signed_access_ready and not buyer_access_checked:
        blockers.append("signed_delivery_buyer_access_check_not_executed")
    if signed_access_ready and buyer_access_checked and not buyer_accessible:
        blockers.append("signed_delivery_buyer_access_fetch_failed")
    return _gate(
        "signed_delivery_access",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "storage_upload_performed": storage_uploaded,
            "signed_url_count": signed_url_count,
            "signed_access_count": signed_access_count,
            "signed_access_required": True,
            "storage_upload_alone_proves_signed_access": False,
            "entitlement_verified": entitlement_verified,
            "buyer_access_checked": buyer_access_checked,
            "buyer_accessible": buyer_accessible,
            "buyer_access_check_status": buyer_access_status or None,
            "operator_attestation_present": attestation_present,
        },
    )


def _safety_contact_physics_gate(
    *,
    evidence: Mapping[str, Any],
    capture_root: Path,
    job_dir: Path,
) -> Dict[str, Any]:
    section = _evidence_section(
        evidence,
        "safety_contact_physics",
        "safetyContactPhysics",
        "safety",
    )
    physics_contact_validated = _boolish(
        _field(section, "physics_contact_validated", "physicsContactValidated")
    )
    non_ranking_operational_claim_validated = _boolish(_field(section, "non_ranking_operational_claim_validated", "safetyValidated"))
    rank_fidelity_result_proven = _boolish(
        _field(section, "rank_fidelity_result_proven", "robotReadinessProven")
    )
    attestation_present = _attestation_ok(
        _field(
            section,
            "operator_attestation",
            "operatorAttestation",
            "owner_attestation",
            "ownerAttestation",
        )
    )
    ref_audit = _named_local_ref_audit(
        section=section,
        aliases_by_name={
            "methodology_uri_or_path": ("methodology_uri_or_path", "methodologyUriOrPath"),
            "contact_validation_uri_or_path": (
                "contact_validation_uri_or_path",
                "contactValidationUriOrPath",
            ),
            "non_ranking_operational_claim_uri_or_path": (
                "non_ranking_operational_claim_uri_or_path",
                "safetyValidationUriOrPath",
            ),
        },
        capture_root=capture_root,
        job_dir=job_dir,
    )
    blockers: List[str] = []
    if not section:
        blockers.append("safety_contact_physics_evidence_missing")
    if section and not physics_contact_validated:
        blockers.append("physics_contact_validation_not_proven")
    if section and not non_ranking_operational_claim_validated:
        blockers.append("non_ranking_operational_claim_not_proven")
    if section and rank_fidelity_result_proven and not attestation_present:
        blockers.append("safety_contact_physics_operator_attestation_missing")
    if ref_audit["missing_local_ref_keys"]:
        blockers.append("safety_contact_physics_local_evidence_refs_missing")
    if ref_audit["invalid_remote_ref_keys"]:
        blockers.append("safety_contact_physics_evidence_refs_invalid_or_placeholder")
    proof_boolean = not blockers
    return _gate(
        "safety_contact_physics_readiness",
        passed=True,
        blockers=[],
        proof_boolean=proof_boolean,
        evidence={
            "physics_contact_validated": physics_contact_validated,
            "non_ranking_operational_claim_validated": non_ranking_operational_claim_validated,
            "rank_fidelity_result_proven": rank_fidelity_result_proven,
            "operator_attestation_present": attestation_present,
            "optional_for_sim_only": True,
            "diagnostic_blockers": blockers,
            **ref_audit,
        },
    )


def _real_robot_pov_evidence_gate(
    *,
    job_dir: Path,
    evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest_path = job_dir / "robot_pov_observation_manifest.json"
    manifest = _read_optional_mapping(manifest_path)
    safety_section = _evidence_section(evidence, "safety_contact_physics", "safetyContactPhysics")
    owner_readiness_attested = bool(
        _boolish(_field(safety_section, "rank_fidelity_result_proven", "robotReadinessProven"))
        and _attestation_ok(
            _field(
                safety_section,
                "operator_attestation",
                "operatorAttestation",
                "owner_attestation",
                "ownerAttestation",
            )
        )
    )
    real_pov_proven = bool(manifest.get("robot_pov_evidence_proven"))
    record_count = int(manifest.get("real_robot_pov_evidence_record_count") or 0)
    action_log_count = int(manifest.get("real_robot_pov_action_log_record_count") or 0)
    missing_run_ids = _string_list(manifest.get("missing_real_robot_pov_scenario_eval_run_ids"))
    passed = bool(real_pov_proven or owner_readiness_attested)
    blockers: List[str] = []
    if not manifest_path.is_file():
        blockers.append("real_robot_pov_observation_manifest_missing")
    if not passed:
        blockers.append("real_robot_pov_evidence_not_proven")
    if missing_run_ids:
        blockers.append("real_robot_pov_missing_required_run_ids")
    proof_boolean = bool(passed and not missing_run_ids)
    return _gate(
        "real_robot_pov_evidence",
        passed=True,
        blockers=[],
        proof_boolean=proof_boolean,
        evidence={
            "robot_pov_observation_manifest": _artifact(manifest_path, base_dir=job_dir),
            "real_robot_pov_evidence_proven": real_pov_proven,
            "owner_readiness_attestation_present": owner_readiness_attested,
            "owner_readiness_attestation_substituted_for_pov_manifest": bool(
                owner_readiness_attested and not real_pov_proven
            ),
            "real_robot_pov_evidence_record_count": record_count,
            "real_robot_pov_action_log_record_count": action_log_count,
            "missing_real_robot_pov_scenario_eval_run_ids": missing_run_ids,
            "optional_for_sim_only": True,
            "diagnostic_blockers": blockers,
        },
    )


def _deployment_row_owner_evidence_present(row: Mapping[str, Any]) -> bool:
    return bool(
        _mapping(row.get("evidence_refs") or row.get("evidenceRefs"))
        or _mapping(row.get("owner_evidence_refs") or row.get("ownerEvidenceRefs"))
        or _string(row.get("owner_evidence_uri") or row.get("ownerEvidenceUri"))
        or _attestation_ok(
            row.get("operator_attestation")
            or row.get("operatorAttestation")
            or row.get("owner_attestation")
            or row.get("ownerAttestation")
        )
    )


def _deployment_row_actual_signal_present(row: Mapping[str, Any]) -> bool:
    if row.get("actual_result_signal_present") is not None:
        return bool(row.get("actual_result_signal_present"))
    return any(
        key in row and row.get(key) not in (None, "", [])
        for key in (
            "actual_success",
            "actualSuccess",
            "actual_result",
            "actualResult",
            "result",
            "status",
            "actual_failures",
            "failure_mode_ids",
            "failureModeIds",
        )
    )


def _real_world_validation_loop_gate(job_dir: Path) -> Dict[str, Any]:
    intake_path = job_dir / "deployment_outcome_intake_manifest.json"
    ledger_path = job_dir / "deployment_outcome_ledger.json"
    followup_path = job_dir / "real_world_validation_followup_plan.json"
    queue_path = job_dir / "real_world_validation_followup_request_queue.json"
    intake = _read_optional_mapping(intake_path)
    ledger = _read_optional_mapping(ledger_path)
    followup = _read_optional_mapping(followup_path)
    queue = _read_optional_mapping(queue_path)
    rows = [
        dict(row)
        for row in ledger.get("records", []) or []
        if isinstance(row, Mapping)
    ]
    optional_evidence = {
        "deployment_outcome_intake_manifest": _artifact(intake_path, base_dir=job_dir),
        "deployment_outcome_ledger": _artifact(ledger_path, base_dir=job_dir),
        "real_world_validation_followup_plan": _artifact(followup_path, base_dir=job_dir),
        "real_world_validation_followup_request_queue": _artifact(
            queue_path,
            base_dir=job_dir,
        ),
        "intake_status": intake.get("status"),
        "ledger_status": ledger.get("status"),
        "record_count": len(rows),
        "optional_for_sim_only": True,
    }
    if not rows:
        return _gate(
            "real_world_validation_loop",
            passed=True,
            blockers=[],
            proof_boolean=False,
            evidence={
                **optional_evidence,
                "status": "not_requested",
                "real_world_outcome_proven": False,
                "record_level_real_world_outcome_proven": False,
            },
        )
    missing_owner = [
        _string(row.get("record_id")) or f"deployment_outcome_{index:04d}"
        for index, row in enumerate(rows, start=1)
        if not _deployment_row_owner_evidence_present(row)
    ]
    missing_actual = [
        _string(row.get("record_id")) or f"deployment_outcome_{index:04d}"
        for index, row in enumerate(rows, start=1)
        if not _deployment_row_actual_signal_present(row)
    ]
    record_level_proven = bool(rows) and not missing_owner and not missing_actual
    blockers: List[str] = []
    if not intake_path.is_file():
        blockers.append("deployment_outcome_intake_manifest_missing")
    if not ledger_path.is_file():
        blockers.append("deployment_outcome_ledger_missing")
    if missing_owner:
        blockers.append("deployment_outcomes_missing_owner_evidence")
    if missing_actual:
        blockers.append("deployment_outcomes_missing_actual_result_signal")
    if rows and not followup_path.is_file():
        blockers.append("real_world_validation_followup_plan_missing")
    return _gate(
        "real_world_validation_loop",
        passed=True,
        blockers=[],
        proof_boolean=record_level_proven and followup_path.is_file(),
        evidence={
            **optional_evidence,
            "intake_status": intake.get("status"),
            "ledger_status": ledger.get("status"),
            "ledger_real_world_outcome_proven_claimed": bool(
                ledger.get("real_world_outcome_proven")
            ),
            "record_level_real_world_outcome_proven": record_level_proven,
            "real_world_outcome_proven": record_level_proven,
            "record_count": len(rows),
            "owner_evidence_record_count": len(rows) - len(missing_owner),
            "missing_owner_evidence_record_ids": missing_owner,
            "missing_actual_result_signal_record_ids": missing_actual,
            "followup_plan_status": followup.get("status"),
            "followup_request_queue_status": queue.get("status"),
            "followup_request_queue_request_count": int(
                queue.get("queued_request_count") or 0
            ),
            "diagnostic_blockers": blockers,
        },
    )


def _predicted_vs_actual_calibration_gate(job_dir: Path) -> Dict[str, Any]:
    report_path = job_dir / "sim_vs_real_calibration_report.json"
    summary_path = job_dir / "prediction_vs_actual_deployment_summary.json"
    report = _read_optional_mapping(report_path)
    summary = _read_optional_mapping(summary_path)
    score = _number(report.get("sim_vs_real_calibration_score"))
    report_status = _string(report.get("status"))
    report_blockers = [
        *_string_list(report.get("blockers")),
        *_string_list(report.get("diagnostic_blockers")),
    ]
    accepted_anchor_count = int(report.get("accepted_anchor_count") or 0)
    minimum_anchor_count = int(report.get("minimum_accepted_anchor_count") or 0)
    matched_count = int(report.get("matched_prediction_record_count") or 0)
    exact_count = int(report.get("exact_prediction_record_count") or 0)
    weak_ids = _string_list(report.get("weak_prediction_match_record_ids"))
    unmatched_ids = _string_list(report.get("unmatched_actual_record_ids"))
    unmatched_prediction_rows = report.get("unmatched_prediction_rows") or []
    stale_anchor_ids = _string_list(report.get("stale_anchor_row_ids"))
    conflicting_anchor_ids = _string_list(report.get("conflicting_anchor_row_ids"))
    missing_summary_sections = sorted(
        section
        for section in PREDICTED_VS_ACTUAL_SUMMARY_REQUIRED_SECTIONS
        if section not in summary
    )
    optional_evidence = {
        "sim_vs_real_calibration_report": _artifact(report_path, base_dir=job_dir),
        "prediction_vs_actual_deployment_summary": _artifact(
            summary_path,
            base_dir=job_dir,
        ),
        "calibration_status": report_status,
        "summary_status": summary.get("status"),
        "sim_vs_real_calibration_score": score,
        "accepted_anchor_count": accepted_anchor_count,
        "minimum_accepted_anchor_count": minimum_anchor_count,
        "report_blockers": report_blockers,
        "matched_prediction_record_count": matched_count,
        "exact_prediction_record_count": exact_count,
        "weak_prediction_match_record_ids": weak_ids,
        "unmatched_actual_record_ids": unmatched_ids,
        "unmatched_prediction_rows": unmatched_prediction_rows,
        "stale_anchor_row_ids": stale_anchor_ids,
        "conflicting_anchor_row_ids": conflicting_anchor_ids,
        "missing_summary_sections": missing_summary_sections,
        "optional_for_sim_only": True,
    }
    calibration_requested = report_path.is_file() or summary_path.is_file()
    optional_unmeasured_statuses = {
        "",
        "not_measured",
        "not_requested",
        "blocked_insufficient_anchor_count",
        "blocked_anchor_quality",
        "blocked_weak_prediction_matches",
    }
    optional_diagnostic_blockers: List[str] = list(report_blockers)
    if weak_ids:
        optional_diagnostic_blockers.append("predicted_vs_actual_weak_prediction_matches")
    if unmatched_ids:
        optional_diagnostic_blockers.append("unmatched_actual_rows")
    if unmatched_prediction_rows:
        optional_diagnostic_blockers.append("unmatched_prediction_rows")
    if stale_anchor_ids:
        optional_diagnostic_blockers.append("stale_anchor_rows")
    if conflicting_anchor_ids:
        optional_diagnostic_blockers.append("conflicting_anchor_rows")
    if not calibration_requested or (
        report_status in optional_unmeasured_statuses
        and score is None
        and accepted_anchor_count == 0
    ):
        return _gate(
            "predicted_vs_actual_calibration",
            passed=True,
            blockers=[],
            proof_boolean=False,
            evidence={
                **optional_evidence,
                "status": "not_requested" if not calibration_requested else "not_measured",
                "diagnostic_blockers": list(dict.fromkeys(optional_diagnostic_blockers)),
            },
        )
    blockers: List[str] = []
    if not report_path.is_file():
        blockers.append("sim_vs_real_calibration_report_missing")
    if not summary_path.is_file():
        blockers.append("prediction_vs_actual_deployment_summary_missing")
    if report_path.is_file() and report_status != "completed":
        blockers.extend(report_blockers)
        if report_status == "not_measured" and "insufficient_anchor_count" not in blockers:
            blockers.append("insufficient_anchor_count")
        if weak_ids:
            blockers.append("predicted_vs_actual_weak_prediction_matches")
        if unmatched_ids:
            blockers.append("unmatched_actual_rows")
    if report_path.is_file() and score is not None and not (0.0 <= score <= 1.0):
        blockers.append("sim_vs_real_calibration_score_invalid")
    if report_path.is_file() and report_status == "completed" and score is None:
        blockers.append("sim_vs_real_calibration_score_missing")
    if report_path.is_file() and accepted_anchor_count < minimum_anchor_count:
        blockers.append("insufficient_anchor_count")
    if report_path.is_file() and unmatched_prediction_rows:
        blockers.append("unmatched_prediction_rows")
    if report_path.is_file() and stale_anchor_ids:
        blockers.append("stale_anchor_rows")
    if report_path.is_file() and conflicting_anchor_ids:
        blockers.append("conflicting_anchor_rows")
    if report_path.is_file() and matched_count <= 0:
        blockers.append("predicted_vs_actual_no_matched_prediction_records")
    if report_path.is_file() and exact_count <= 0:
        blockers.append("predicted_vs_actual_no_exact_prediction_matches")
    if summary_path.is_file() and missing_summary_sections:
        blockers.append("prediction_vs_actual_summary_missing_required_sections")
    return _gate(
        "predicted_vs_actual_calibration",
        passed=True,
        blockers=[],
        proof_boolean=not blockers and report_status == "completed" and score is not None,
        evidence={**optional_evidence, "diagnostic_blockers": blockers},
    )


def _site_capture_gate(*, capture_root: Path, base_dir: Path, scene_id: str, capture_id: str) -> Dict[str, Any]:
    descriptor_path = capture_root / "capture_descriptor.json"
    raw_manifest_path = capture_root / "raw" / "manifest.json"
    upload_completion_path = capture_root / "raw" / "capture_upload_complete.json"
    descriptor = _read_optional_mapping(descriptor_path)
    raw_manifest = _read_optional_mapping(raw_manifest_path)
    upload_completion = _read_optional_mapping(upload_completion_path)
    raw_evidence = _raw_capture_evidence_summary(
        capture_root=capture_root,
        raw_manifest=raw_manifest,
    )
    blockers: List[str] = []
    if not descriptor:
        blockers.append("missing_capture_descriptor")
    if not raw_manifest:
        blockers.append("missing_raw_manifest")
    if not upload_completion:
        blockers.append("missing_raw_capture_upload_completion")
    else:
        upload_scene_id = _string(_field(upload_completion, "scene_id", "sceneId"))
        upload_capture_id = _string(_field(upload_completion, "capture_id", "captureId"))
        if not upload_scene_id:
            blockers.append("raw_capture_upload_completion_scene_id_missing")
        elif upload_scene_id != scene_id:
            blockers.append("raw_capture_upload_completion_scene_id_mismatch")
        if not upload_capture_id:
            blockers.append("raw_capture_upload_completion_capture_id_missing")
        elif upload_capture_id != capture_id:
            blockers.append("raw_capture_upload_completion_capture_id_mismatch")
        upload_status = _string(_field(upload_completion, "status", "uploadStatus")).lower()
        if upload_status and not _status_ok(upload_status):
            blockers.append("raw_capture_upload_completion_status_not_complete")
    if raw_manifest and not raw_evidence["has_capture_evidence"]:
        blockers.append("missing_raw_capture_evidence")
    if raw_evidence["missing_local_pointer_files"]:
        blockers.append("raw_capture_evidence_local_files_missing")
    for source_name, payload in (("capture_descriptor", descriptor), ("raw_manifest", raw_manifest)):
        if payload and _string(payload.get("scene_id")) and _string(payload.get("scene_id")) != scene_id:
            blockers.append(f"{source_name}_scene_id_mismatch")
        if payload and _string(payload.get("capture_id")) and _string(payload.get("capture_id")) != capture_id:
            blockers.append(f"{source_name}_capture_id_mismatch")
    return _gate(
        "site_capture",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "capture_root": str(capture_root),
            "capture_descriptor": _artifact(descriptor_path, base_dir=base_dir),
            "raw_manifest": _artifact(raw_manifest_path, base_dir=base_dir),
            "raw_capture_upload_completion": _artifact(upload_completion_path, base_dir=base_dir),
            "raw_capture_upload_completion_identity": {
                "scene_id": _string(_field(upload_completion, "scene_id", "sceneId")),
                "capture_id": _string(_field(upload_completion, "capture_id", "captureId")),
                "status": _string(_field(upload_completion, "status", "uploadStatus")),
            },
            "raw_capture_evidence": raw_evidence,
        },
    )


def _robot_eval_dataset_gate(
    *,
    capture_root: Path,
    job_dir: Path,
    gate_id: str,
    filename: str,
    count_fields: Sequence[str],
    minimum_count: int = 1,
    required_card_fields: Sequence[str] = (),
) -> Dict[str, Any]:
    path = capture_root / "pipeline" / "robot_eval_dataset" / filename
    payload = _read_optional_mapping(path)
    count = _cards_count(payload, *count_fields)
    card_rows = _card_rows(payload)
    cards_missing_fields = _cards_missing_required_fields(
        card_rows,
        required_card_fields,
    )
    cards_missing_standard_metrics = (
        _cards_missing_standard_required_metrics(
            card_rows,
            TASK_CARD_STANDARD_REQUIRED_METRICS,
        )
        if gate_id == "task_definitions"
        else []
    )
    blockers = []
    if not path.is_file():
        blockers.append(f"missing_{filename}")
    if count < minimum_count:
        blockers.append(f"{gate_id}_empty")
    if required_card_fields and count > 0 and not card_rows:
        blockers.append(f"{gate_id}_missing_card_rows")
    if cards_missing_fields:
        blockers.append(f"{gate_id}_cards_missing_required_fields")
    if cards_missing_standard_metrics:
        blockers.append("task_definitions_missing_standard_required_metrics")
    return _gate(
        gate_id,
        passed=not blockers,
        blockers=blockers,
        evidence={
            "artifact": _artifact(path, base_dir=job_dir),
            "count": count,
            "card_row_count": len(card_rows),
            "required_card_fields": list(required_card_fields),
            "cards_missing_required_fields": cards_missing_fields,
            "cards_missing_standard_required_metrics": cards_missing_standard_metrics,
        },
    )


def _scenario_library_gate(*, capture_root: Path, job_dir: Path) -> Dict[str, Any]:
    task_cards_payload = _read_optional_mapping(
        capture_root / "pipeline" / "robot_eval_dataset" / "task_cards.json"
    )
    task_rows = _card_rows(task_cards_payload)
    scenario_cards_payload = _read_optional_mapping(
        capture_root / "pipeline" / "robot_eval_dataset" / "scenario_cards.json"
    )
    scenario_rows = _card_rows(scenario_cards_payload)
    scenario_gate = _robot_eval_dataset_gate(
        capture_root=capture_root,
        job_dir=job_dir,
        gate_id="scenario_library",
        filename="scenario_cards.json",
        count_fields=("scenario_card_count", "scenario_count"),
        required_card_fields=SCENARIO_CARD_REQUIRED_FIELDS,
    )
    variation_path = capture_root / "pipeline" / "simulation_automation" / "scenario_variation_instances.json"
    variation_payload = _read_optional_mapping(variation_path)
    scenario_family_path = (
        capture_root / "pipeline" / "robot_eval_dataset" / "scenario_family_library.json"
    )
    scenario_family_payload = _read_optional_mapping(scenario_family_path)
    family_rows = _scenario_family_rows(scenario_family_payload)
    instance_count = int(variation_payload.get("instance_count") or 0)
    required_variation_names = (
        _string_list(variation_payload.get("required_variation_names"))
        or _string_list(scenario_family_payload.get("variation_names_required"))
        or list(SCENARIO_VARIATION_NAMES)
    )
    instantiated_variation_names = _string_list(
        variation_payload.get("variation_names_instantiated")
    )
    if not instantiated_variation_names:
        instantiated_variation_names = _string_list(
            [
                instance.get("variation_name") or instance.get("variationName")
                for instance in variation_payload.get("instances", []) or []
                if isinstance(instance, Mapping)
            ]
        )
    missing_required_variation_names = [
        name for name in required_variation_names if name not in instantiated_variation_names
    ]
    variation_instance_rows = [
        dict(instance)
        for instance in variation_payload.get("instances", []) or []
        if isinstance(instance, Mapping)
    ]
    variation_rows_missing_concrete_details = _variation_rows_missing_concrete_details(
        variation_instance_rows
    )
    missing_required_variations_by_scenario = _missing_required_variations_by_scenario(
        coverage_rows=variation_instance_rows,
        required_variation_names=required_variation_names,
        scenario_rows=scenario_rows,
    )
    scenario_family_coverage = _scenario_family_task_coverage(
        task_rows=task_rows,
        family_rows=family_rows,
        required_variation_names=required_variation_names,
    )
    blockers = list(scenario_gate["blockers"])
    if not scenario_family_path.is_file():
        blockers.append("missing_scenario_family_library")
    elif not family_rows:
        blockers.append("scenario_family_library_empty")
    if scenario_family_coverage["missing_task_ids"]:
        blockers.append("scenario_family_library_missing_task_coverage")
    if scenario_family_coverage["missing_required_variations_by_family"]:
        blockers.append("scenario_family_library_missing_required_variations")
    if not variation_path.is_file():
        blockers.append("missing_scenario_variation_instances")
    elif variation_payload.get("status") != "completed":
        blockers.append("scenario_variation_instances_not_completed")
    elif instance_count <= 0:
        blockers.append("scenario_variation_instances_empty")
    if missing_required_variation_names:
        blockers.append("scenario_variation_instances_missing_required_variations")
    if missing_required_variations_by_scenario:
        blockers.append("scenario_variation_instances_missing_required_variations_per_scenario")
    if variation_rows_missing_concrete_details:
        blockers.append("scenario_variation_instances_missing_concrete_mutation_details")
    return _gate(
        "scenario_library",
        passed=not blockers,
        blockers=blockers,
        evidence={
            **dict(scenario_gate["evidence"]),
            "scenario_family_library": _artifact(scenario_family_path, base_dir=job_dir),
            "scenario_family_count": len(family_rows),
            "scenario_family_task_coverage": scenario_family_coverage,
            "scenario_variation_instances": _artifact(variation_path, base_dir=job_dir),
            "scenario_variation_instance_count": instance_count,
            "required_variation_names": required_variation_names,
            "instantiated_variation_names": instantiated_variation_names,
            "missing_required_variation_names": missing_required_variation_names,
            "missing_required_variations_by_scenario": missing_required_variations_by_scenario,
            "variation_rows_missing_concrete_details": variation_rows_missing_concrete_details,
        },
    )


def _robot_pov_gate(job_dir: Path) -> Dict[str, Any]:
    path = job_dir / "robot_pov_observation_manifest.json"
    payload = _read_optional_mapping(path)
    matrix_path = job_dir / "scenario_eval_matrix.json"
    matrix = _read_optional_mapping(matrix_path)
    frame_sequence_path = job_dir / "robot_pov_frame_sequence_manifest.json"
    frame_sequence_payload = _read_optional_mapping(frame_sequence_path)
    storyboard_path = job_dir / "robot_pov_render_storyboard.json"
    storyboard_payload = _read_optional_mapping(storyboard_path)
    count = int(payload.get("observation_count") or 0)
    run_count = int(matrix.get("scenario_eval_run_count") or 0)
    observations = [
        dict(observation)
        for observation in payload.get("observations", []) or []
        if isinstance(observation, Mapping)
    ]
    required_run_ids = sorted(
        {
            _string(run.get("scenario_eval_run_id"))
            for run in matrix.get("runs", []) or []
            if isinstance(run, Mapping) and _string(run.get("scenario_eval_run_id"))
        }
    )
    covered_run_ids = sorted(
        {
            _string(observation.get("scenario_eval_run_id") or observation.get("scenarioEvalRunId"))
            for observation in observations
            if _string(observation.get("scenario_eval_run_id") or observation.get("scenarioEvalRunId"))
        }
    )
    missing_run_ids = sorted(set(required_run_ids) - set(covered_run_ids))
    observation_rows_missing_required_fields: List[Dict[str, Any]] = []
    observation_generated_frame_artifacts: Dict[str, Dict[str, Any]] = {}
    missing_observation_generated_frame_paths: List[str] = []
    for index, observation in enumerate(observations):
        missing_fields = [
            field
            for field in ROBOT_POV_OBSERVATION_REQUIRED_FIELDS
            if not _card_field_present(observation.get(field))
        ]
        observation_id = _string(observation.get("observation_id") or observation.get("observationId"))
        run_id = _string(observation.get("scenario_eval_run_id") or observation.get("scenarioEvalRunId"))
        if missing_fields:
            observation_rows_missing_required_fields.append(
                {
                    "index": index,
                    "observation_id": observation_id or None,
                    "scenario_eval_run_id": run_id or None,
                    "missing_fields": missing_fields,
                }
            )
        frame_ref = _string(
            observation.get("generated_frame_path")
            or observation.get("generatedFramePath")
        )
        if frame_ref:
            local_path = _job_local_artifact_path(job_dir, frame_ref)
            if local_path is not None:
                key = observation_id or run_id or f"observation_{index:04d}"
                artifact = _artifact(local_path, base_dir=job_dir)
                observation_generated_frame_artifacts[key] = artifact
                if not artifact["exists"]:
                    missing_observation_generated_frame_paths.append(_relative_to(job_dir, local_path))
    frame_sequences = [
        dict(sequence)
        for sequence in frame_sequence_payload.get("sequences", []) or []
        if isinstance(sequence, Mapping)
    ]
    sequence_count = int(frame_sequence_payload.get("sequence_count") or 0)
    total_frame_count = int(frame_sequence_payload.get("total_frame_count") or 0)
    frame_sequence_run_ids = sorted(
        {
            _string(sequence.get("scenario_eval_run_id") or sequence.get("scenarioEvalRunId"))
            for sequence in frame_sequences
            if _string(sequence.get("scenario_eval_run_id") or sequence.get("scenarioEvalRunId"))
        }
    )
    missing_frame_sequence_run_ids = sorted(set(required_run_ids) - set(frame_sequence_run_ids))
    missing_local_frame_paths: List[str] = []
    empty_frame_sequence_ids: List[str] = []
    for sequence in frame_sequences:
        frame_paths = _string_list(sequence.get("frame_paths") or sequence.get("framePaths"))
        if not frame_paths:
            empty_frame_sequence_ids.append(
                _string(sequence.get("sequence_id") or sequence.get("sequenceId"))
            )
        for frame_path in frame_paths:
            local_path = _job_local_artifact_path(job_dir, frame_path)
            if local_path is not None and not local_path.is_file():
                missing_local_frame_paths.append(_relative_to(job_dir, local_path))
    storyboards = [
        dict(storyboard)
        for storyboard in storyboard_payload.get("storyboards", []) or []
        if isinstance(storyboard, Mapping)
    ]
    storyboard_count = int(storyboard_payload.get("storyboard_count") or 0)
    storyboard_run_ids = sorted(
        {
            _string(storyboard.get("scenario_eval_run_id") or storyboard.get("scenarioEvalRunId"))
            for storyboard in storyboards
            if _string(storyboard.get("scenario_eval_run_id") or storyboard.get("scenarioEvalRunId"))
        }
    )
    missing_storyboard_run_ids = sorted(set(required_run_ids) - set(storyboard_run_ids))
    empty_storyboard_ids: List[str] = []
    missing_storyboard_frame_paths: List[str] = []
    for storyboard in storyboards:
        storyboard_id = _string(storyboard.get("storyboard_id") or storyboard.get("storyboardId"))
        frames = storyboard.get("frames")
        if not isinstance(frames, list) or not frames:
            empty_storyboard_ids.append(storyboard_id)
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            frame_path = _string(frame.get("frame_path") or frame.get("framePath"))
            if not frame_path:
                missing_storyboard_frame_paths.append(
                    f"{storyboard_id or 'storyboard'}:<missing_frame_path>"
                )
                continue
            local_path = _job_local_artifact_path(job_dir, frame_path)
            if local_path is not None and not local_path.is_file():
                missing_storyboard_frame_paths.append(_relative_to(job_dir, local_path))
    blockers = []
    if payload.get("status") != "completed":
        blockers.append("robot_pov_manifest_not_completed")
    if count <= 0:
        blockers.append("robot_pov_observations_empty")
    if run_count > 0 and count < run_count:
        blockers.append("robot_pov_missing_scenario_variation_run_coverage")
    if missing_run_ids:
        blockers.append("robot_pov_missing_required_scenario_eval_run_ids")
    if observation_rows_missing_required_fields:
        blockers.append("robot_pov_observations_missing_required_fields")
    if missing_observation_generated_frame_paths:
        blockers.append("robot_pov_observation_local_generated_frames_missing")
    if not frame_sequence_path.is_file():
        blockers.append("missing_robot_pov_frame_sequence_manifest")
    elif frame_sequence_payload.get("status") != "completed":
        blockers.append("robot_pov_frame_sequence_manifest_not_completed")
    if sequence_count <= 0:
        blockers.append("robot_pov_frame_sequences_empty")
    if total_frame_count <= 0:
        blockers.append("robot_pov_local_render_frames_empty")
    if run_count > 0 and sequence_count < run_count:
        blockers.append("robot_pov_frame_sequence_missing_scenario_variation_run_coverage")
    if missing_frame_sequence_run_ids:
        blockers.append("robot_pov_frame_sequence_missing_required_scenario_eval_run_ids")
    if empty_frame_sequence_ids:
        blockers.append("robot_pov_frame_sequences_missing_frame_paths")
    if missing_local_frame_paths:
        blockers.append("robot_pov_frame_sequence_local_files_missing")
    if not storyboard_path.is_file():
        blockers.append("missing_robot_pov_render_storyboard")
    elif storyboard_payload.get("status") != "completed":
        blockers.append("robot_pov_render_storyboard_not_completed")
    if storyboard_count <= 0:
        blockers.append("robot_pov_render_storyboard_empty")
    if run_count > 0 and storyboard_count < run_count:
        blockers.append("robot_pov_storyboard_missing_scenario_variation_run_coverage")
    if missing_storyboard_run_ids:
        blockers.append("robot_pov_storyboard_missing_required_scenario_eval_run_ids")
    if empty_storyboard_ids:
        blockers.append("robot_pov_storyboards_missing_frames")
    if missing_storyboard_frame_paths:
        blockers.append("robot_pov_storyboard_local_frame_files_missing")
    return _gate(
        "robot_pov_generation",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "artifact": _artifact(path, base_dir=job_dir),
            "scenario_eval_matrix": _artifact(matrix_path, base_dir=job_dir),
            "frame_sequence_manifest": _artifact(frame_sequence_path, base_dir=job_dir),
            "render_storyboard": _artifact(storyboard_path, base_dir=job_dir),
            "observation_count": count,
            "scenario_eval_run_count": run_count,
            "required_scenario_eval_run_ids": required_run_ids,
            "covered_scenario_eval_run_ids": covered_run_ids,
            "missing_scenario_eval_run_ids": missing_run_ids,
            "observation_rows_missing_required_fields": observation_rows_missing_required_fields,
            "observation_generated_frame_artifacts": observation_generated_frame_artifacts,
            "missing_observation_generated_frame_paths": missing_observation_generated_frame_paths,
            "frame_sequence_count": sequence_count,
            "total_frame_count": total_frame_count,
            "frame_sequence_run_ids": frame_sequence_run_ids,
            "missing_frame_sequence_run_ids": missing_frame_sequence_run_ids,
            "empty_frame_sequence_ids": empty_frame_sequence_ids,
            "missing_local_frame_paths": missing_local_frame_paths,
            "storyboard_count": storyboard_count,
            "storyboard_run_ids": storyboard_run_ids,
            "missing_storyboard_run_ids": missing_storyboard_run_ids,
            "empty_storyboard_ids": empty_storyboard_ids,
            "missing_storyboard_frame_paths": missing_storyboard_frame_paths,
        },
    )




def _scenario_eval_suite_gate(*, capture_root: Path, job_dir: Path) -> Dict[str, Any]:
    eval_gate = _robot_eval_dataset_gate(
        capture_root=capture_root,
        job_dir=job_dir,
        gate_id="scenario_eval_suite",
        filename="eval_cards.json",
        count_fields=("eval_card_count", "eval_count"),
        required_card_fields=EVAL_CARD_REQUIRED_FIELDS,
    )
    schedule_path = job_dir / "arena_eval_schedule.json"
    schedule = _read_optional_mapping(schedule_path)
    schedule_count = int(schedule.get("scenario_count") or 0)
    matrix_path = job_dir / "scenario_eval_matrix.json"
    matrix = _read_optional_mapping(matrix_path)
    matrix_status = _string(matrix.get("status"))
    matrix_blockers = _string_list(matrix.get("blockers"))
    run_count = int(matrix.get("scenario_eval_run_count") or 0)
    matrix_runs = [
        dict(run)
        for run in matrix.get("runs", []) or []
        if isinstance(run, Mapping)
    ]
    variation_payload = _read_optional_mapping(
        capture_root / "pipeline" / "simulation_automation" / "scenario_variation_instances.json"
    )
    variation_instance_rows = [
        dict(instance)
        for instance in variation_payload.get("instances", []) or []
        if isinstance(instance, Mapping)
    ]
    matrix_runs_missing_concrete_details = (
        _scenario_eval_runs_missing_concrete_variation_details(
            rows=matrix_runs,
            variation_instance_details=_variation_instance_detail_index(
                variation_instance_rows
            ),
        )
    )
    run_rows_missing_fields = _cards_missing_required_fields(
        matrix_runs,
        (
            "scenario_eval_run_id",
            "task_id",
            "scenario_id",
            "variation_name",
        ),
    )
    required_variation_names = _string_list(matrix.get("required_variation_names"))
    variation_names_covered = _string_list(matrix.get("variation_names_covered"))
    exact_requested_eval_run_filter_count = int(
        _number(matrix.get("requested_scenario_eval_run_filter_count")) or 0
    )
    exact_followup_rerun_scope = exact_requested_eval_run_filter_count > 0
    missing_required_variation_names = _string_list(
        matrix.get("missing_required_variation_names")
    )
    if required_variation_names and not missing_required_variation_names:
        missing_required_variation_names = sorted(
            set(required_variation_names) - set(variation_names_covered)
        )
    missing_required_variations_by_scenario = _missing_required_variations_by_scenario(
        coverage_rows=matrix_runs,
        required_variation_names=required_variation_names,
    )
    if exact_followup_rerun_scope:
        missing_required_variation_names = []
        missing_required_variations_by_scenario = {}
    blockers = list(eval_gate["blockers"])
    if not matrix_path.is_file():
        blockers.append("missing_scenario_eval_matrix")
    elif matrix_status != "completed":
        blockers.append("scenario_eval_matrix_not_completed")
    elif run_count <= 0:
        blockers.append("scenario_eval_matrix_empty")
    blockers.extend(matrix_blockers)
    if run_count > 0 and not matrix_runs:
        blockers.append("scenario_eval_matrix_missing_run_rows")
    if matrix_path.is_file() and run_count != len(matrix_runs):
        blockers.append("scenario_eval_matrix_run_count_mismatch")
    if run_rows_missing_fields:
        blockers.append("scenario_eval_matrix_run_rows_missing_required_fields")
    if missing_required_variation_names:
        blockers.append("scenario_eval_matrix_missing_required_variations")
    if missing_required_variations_by_scenario:
        blockers.append("scenario_eval_matrix_missing_required_variations_per_scenario")
    if matrix_runs_missing_concrete_details:
        blockers.append("scenario_eval_matrix_runs_missing_concrete_variation_details")
    evidence = dict(eval_gate["evidence"])
    evidence["arena_eval_schedule"] = _artifact(schedule_path, base_dir=job_dir)
    evidence["arena_schedule_scenario_count"] = schedule_count
    evidence["scenario_eval_matrix"] = _artifact(matrix_path, base_dir=job_dir)
    evidence["scenario_eval_matrix_status"] = matrix_status or None
    evidence["scenario_eval_matrix_blockers"] = matrix_blockers
    evidence["scenario_eval_run_count"] = run_count
    evidence["scenario_eval_run_row_count"] = len(matrix_runs)
    evidence["scenario_eval_run_rows_missing_required_fields"] = run_rows_missing_fields
    evidence["required_variation_names"] = required_variation_names
    evidence["variation_names_covered"] = variation_names_covered
    evidence["exact_followup_rerun_scope"] = exact_followup_rerun_scope
    evidence["requested_scenario_eval_run_filter_count"] = (
        exact_requested_eval_run_filter_count
    )
    evidence["missing_required_variation_names"] = missing_required_variation_names
    evidence["missing_required_variations_by_scenario"] = missing_required_variations_by_scenario
    evidence["scenario_eval_runs_missing_concrete_details"] = (
        matrix_runs_missing_concrete_details
    )
    return _gate(
        "scenario_eval_suite",
        passed=not blockers,
        blockers=blockers,
        evidence=evidence,
    )


def _failure_labels_gate(job_dir: Path) -> Dict[str, Any]:
    path = job_dir / "failure_labels.json"
    trace_path = job_dir / "normalized_attempt_trace.json"
    policy_trace_path = job_dir / "policy_execution_trace.json"
    payload = _read_optional_mapping(path)
    trace = _read_optional_mapping(trace_path)
    policy_trace = _read_optional_mapping(policy_trace_path)
    status = _string(payload.get("status")).lower()
    audit = build_failure_diagnosis_audit(
        labels_payload=payload,
        trace_payload=trace,
        policy_trace_payload=policy_trace,
    )
    blockers: List[str] = []
    if not path.is_file():
        blockers.append("missing_failure_labels")
    if status.startswith("blocked") or status in {"missing", "not_available"}:
        blockers.append("failure_labels_not_available")
    blockers.extend(_string_list(audit.get("blockers")))
    return _gate(
        "failure_labels",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "artifact": _artifact(path, base_dir=job_dir),
            "normalized_attempt_trace": _artifact(trace_path, base_dir=job_dir),
            "policy_execution_trace": _artifact(policy_trace_path, base_dir=job_dir),
            "status": status,
            "label_count": int(payload.get("label_count") or 0),
            **audit,
        },
    )


def _evaluation_methodology_gate(job_dir: Path) -> Dict[str, Any]:
    path = job_dir / "evaluation_result.json"
    payload = _read_optional_mapping(path)
    matrix_path = job_dir / "scenario_eval_matrix.json"
    matrix = _read_optional_mapping(matrix_path)
    trace_ref = (
        _string(payload.get("normalized_attempt_trace_path"))
        or _string(payload.get("normalizedAttemptTracePath"))
        or "normalized_attempt_trace.json"
    )
    trace_path = _job_local_artifact_path(job_dir, trace_ref) or (job_dir / "normalized_attempt_trace.json")
    trace_payload = _read_optional_mapping(trace_path)
    trace_attempts = [
        item for item in trace_payload.get("attempts", []) or [] if isinstance(item, Mapping)
    ]
    required_run_ids = _scenario_eval_run_ids(matrix.get("runs", []) or [])
    scored_run_ids = _scenario_eval_run_ids(trace_attempts)
    missing_run_ids = sorted(set(required_run_ids) - set(scored_run_ids))
    scorecard = _mapping(payload.get("standard_policy_scorecard"))
    missing = [field for field in SCORECARD_REQUIRED_FIELDS if field not in scorecard]
    invalid = _invalid_scorecard_fields(scorecard)
    blockers = []
    if not path.is_file():
        blockers.append("missing_evaluation_result")
    elif _string(payload.get("status")) not in {"completed", "completed_with_failures"}:
        blockers.append("evaluation_result_not_completed")
    if not trace_path.is_file():
        blockers.append("missing_normalized_attempt_trace")
    elif _string(trace_payload.get("status")) != "completed":
        blockers.append("normalized_attempt_trace_not_completed")
    if missing_run_ids:
        blockers.append("evaluation_scorecard_missing_required_scenario_eval_run_ids")
    if missing:
        blockers.append("standard_policy_scorecard_missing_required_metrics")
    if invalid:
        blockers.append("standard_policy_scorecard_invalid_metric_values")
    return _gate(
        "evaluation_methodology",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "artifact": _artifact(path, base_dir=job_dir),
            "scenario_eval_matrix": _artifact(matrix_path, base_dir=job_dir),
            "normalized_attempt_trace": _artifact(trace_path, base_dir=job_dir),
            "evaluation_status": payload.get("status"),
            "normalized_attempt_trace_status": trace_payload.get("status"),
            "required_scenario_eval_run_ids": required_run_ids,
            "scored_scenario_eval_run_ids": scored_run_ids,
            "missing_scenario_eval_run_ids": missing_run_ids,
            "scorecard_fields_present": sorted(scorecard),
            "missing_scorecard_fields": missing,
            "invalid_scorecard_fields": invalid,
        },
    )


def _policy_interface_gate(*, capture_root: Path, job_dir: Path) -> Dict[str, Any]:
    path = job_dir / "policy_package_manifest.json"
    payload = _read_optional_mapping(path)
    modalities = _mapping(payload.get("modalities"))
    selected = _string_list(payload.get("selected_modalities"))
    missing_modalities = [modality for modality in SUPPORTED_POLICY_MODALITIES if modality not in modalities]
    unknown_selected = [modality for modality in selected if modality not in SUPPORTED_POLICY_MODALITIES]
    selected_statuses: Dict[str, str] = {}
    selected_missing_inputs: Dict[str, List[str]] = {}
    selected_local_ref_artifacts: Dict[str, Dict[str, Dict[str, Any]]] = {}
    selected_missing_local_ref_keys: Dict[str, List[str]] = {}
    invalid_selected: List[str] = []
    for modality in selected:
        modality_payload = _mapping(modalities.get(modality))
        status = _string(modality_payload.get("status")).lower()
        selected_statuses[modality] = status or "missing"
        reference = _mapping(modality_payload.get("reference"))
        explicit_missing = _string_list(modality_payload.get("missing_inputs"))
        computed_missing = (
            _policy_modality_missing_inputs(modality, reference)
            if modality in SUPPORTED_POLICY_MODALITIES
            else []
        )
        local_ref_artifacts, missing_local_ref_keys, missing_local_ref_inputs = (
            _policy_modality_local_reference_audit(
                modality=modality,
                reference=reference,
                capture_root=capture_root,
                job_dir=job_dir,
            )
            if modality in SUPPORTED_POLICY_MODALITIES
            else ({}, [], [])
        )
        if local_ref_artifacts:
            selected_local_ref_artifacts[modality] = local_ref_artifacts
        if missing_local_ref_keys:
            selected_missing_local_ref_keys[modality] = missing_local_ref_keys
        missing_inputs = []
        for item in [*explicit_missing, *computed_missing, *missing_local_ref_inputs]:
            if item and item not in missing_inputs:
                missing_inputs.append(item)
        if missing_inputs:
            selected_missing_inputs[modality] = missing_inputs
        if (
            modality in unknown_selected
            or not modality_payload
            or status in {"", "blocked", "missing", "not_available", "not_selected"}
            or not bool(modality_payload.get("selected"))
            or missing_inputs
        ):
            invalid_selected.append(modality)
    blockers = []
    if not path.is_file():
        blockers.append("missing_policy_package_manifest")
    if missing_modalities:
        blockers.append("policy_interface_missing_supported_modalities")
    if unknown_selected:
        blockers.append("policy_interface_unknown_selected_modalities")
    if not selected:
        blockers.append("policy_package_no_supported_modality_selected")
    if invalid_selected:
        blockers.append("policy_interface_selected_modalities_invalid")
    if payload.get("status") == "blocked":
        blockers.extend(_string_list(payload.get("missing_inputs")))
    return _gate(
        "policy_interface",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "artifact": _artifact(path, base_dir=job_dir),
            "selected_modalities": selected,
            "supported_modalities_present": sorted(modalities),
            "missing_supported_modalities": missing_modalities,
            "unknown_selected_modalities": unknown_selected,
            "invalid_selected_modalities": sorted(set(invalid_selected)),
            "selected_modality_statuses": selected_statuses,
            "selected_modality_missing_inputs": selected_missing_inputs,
            "selected_modality_local_ref_artifacts": selected_local_ref_artifacts,
            "selected_modality_missing_local_ref_keys": selected_missing_local_ref_keys,
        },
    )


def _simulator_plugins_gate(*, capture_root: Path, job_dir: Path) -> Dict[str, Any]:
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    path = automation_dir / "simulator_engine_plugin_registry.json"
    payload = _read_optional_mapping(path)
    plugins = _mapping(payload.get("plugins"))
    world_model_plugins = _mapping(payload.get("world_model_plugins"))
    required_plugins = set(SIMULATOR_FRAMEWORKS)
    required_world_model_plugins = set(WORLD_MODEL_ENGINE_TARGETS)
    present_plugins = set(plugins)
    present_world_model_plugins = set(world_model_plugins)
    missing_required_plugins = sorted(required_plugins - present_plugins)
    missing_required_world_model_plugins = sorted(
        required_world_model_plugins - present_world_model_plugins
    )
    unready_plugins = []
    unready_world_model_plugins = []
    local_input_artifacts: Dict[str, Dict[str, Dict[str, Any]]] = {}
    missing_local_input_artifacts_by_plugin: Dict[str, List[str]] = {}
    world_model_local_input_artifacts: Dict[str, Dict[str, Dict[str, Any]]] = {}
    missing_local_input_artifacts_by_world_model_plugin: Dict[str, List[str]] = {}
    for framework in sorted(required_plugins & present_plugins):
        plugin = _mapping(plugins.get(framework))
        adapter_status = _string(plugin.get("adapter_contract_status"))
        managed_supported = bool(plugin.get("managed_execution_supported"))
        inputs = _mapping(plugin.get("inputs"))
        plugin_local_inputs: Dict[str, Dict[str, Any]] = {}
        missing_input_keys: List[str] = []
        for key, value in sorted(inputs.items()):
            text = _string(value)
            if not text or _external_uri(text):
                continue
            local_path = _automation_local_reference_path(
                text,
                capture_root=capture_root,
                automation_dir=automation_dir,
            )
            if local_path is None:
                continue
            artifact = _artifact(local_path, base_dir=automation_dir)
            plugin_local_inputs[key] = artifact
            if not artifact["exists"]:
                missing_input_keys.append(key)
        if plugin_local_inputs:
            local_input_artifacts[framework] = plugin_local_inputs
        if missing_input_keys:
            missing_local_input_artifacts_by_plugin[framework] = missing_input_keys
        if adapter_status != "ready" or not managed_supported:
            unready_plugins.append(
                {
                    "framework": framework,
                    "adapter_contract_status": adapter_status or None,
                    "managed_execution_supported": managed_supported,
                }
            )
    for engine in sorted(required_world_model_plugins & present_world_model_plugins):
        plugin = _mapping(world_model_plugins.get(engine))
        adapter_status = _string(plugin.get("adapter_contract_status"))
        managed_supported = bool(plugin.get("managed_execution_supported"))
        source_status = _string(plugin.get("source_status")).lower()
        inputs = _mapping(plugin.get("inputs"))
        plugin_local_inputs: Dict[str, Dict[str, Any]] = {}
        missing_input_keys: List[str] = []
        for key, value in sorted(inputs.items()):
            text = _string(value)
            if not text or _external_uri(text):
                continue
            local_path = _automation_local_reference_path(
                text,
                capture_root=capture_root,
                automation_dir=automation_dir,
            )
            if local_path is None:
                continue
            artifact = _artifact(local_path, base_dir=automation_dir)
            plugin_local_inputs[key] = artifact
            optional_missing_source = (
                key in WORLD_MODEL_PLUGIN_OPTIONAL_INPUT_KEYS
                and source_status in {"", "missing", "not_available", "optional_missing"}
            )
            if not artifact["exists"] and not optional_missing_source:
                missing_input_keys.append(key)
        if plugin_local_inputs:
            world_model_local_input_artifacts[engine] = plugin_local_inputs
        if missing_input_keys:
            missing_local_input_artifacts_by_world_model_plugin[engine] = missing_input_keys
        if adapter_status != "ready" or not managed_supported:
            unready_world_model_plugins.append(
                {
                    "engine": engine,
                    "adapter_contract_status": adapter_status or None,
                    "managed_execution_supported": managed_supported,
                }
            )
    blockers = []
    if not path.is_file():
        blockers.append("missing_simulator_engine_plugin_registry")
    if int(payload.get("plugin_count") or 0) <= 0:
        blockers.append("simulator_engine_plugin_registry_empty")
    if missing_required_plugins:
        blockers.append("simulator_engine_plugin_registry_missing_required_engines")
    if missing_required_world_model_plugins:
        blockers.append("simulator_engine_plugin_registry_missing_required_world_model_engines")
    if unready_plugins:
        blockers.append("simulator_engine_plugins_not_ready")
    if unready_world_model_plugins:
        blockers.append("world_model_engine_plugins_not_ready")
    if missing_local_input_artifacts_by_plugin:
        blockers.append("simulator_engine_plugin_registry_missing_local_input_artifacts")
    if missing_local_input_artifacts_by_world_model_plugin:
        blockers.append("world_model_engine_plugin_registry_missing_local_input_artifacts")
    return _gate(
        "simulator_engine_plugins",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "artifact": _artifact(path, base_dir=job_dir),
            "plugin_count": int(payload.get("plugin_count") or 0),
            "world_model_plugin_count": int(payload.get("world_model_plugin_count") or 0),
            "required_plugins": sorted(required_plugins),
            "plugins": sorted(plugins),
            "missing_required_plugins": missing_required_plugins,
            "unready_plugins": unready_plugins,
            "required_world_model_plugins": sorted(required_world_model_plugins),
            "world_model_plugins": sorted(world_model_plugins),
            "missing_required_world_model_plugins": missing_required_world_model_plugins,
            "unready_world_model_plugins": unready_world_model_plugins,
            "local_input_artifacts": local_input_artifacts,
            "missing_local_input_artifacts_by_plugin": missing_local_input_artifacts_by_plugin,
            "world_model_local_input_artifacts": world_model_local_input_artifacts,
            "missing_local_input_artifacts_by_world_model_plugin": (
                missing_local_input_artifacts_by_world_model_plugin
            ),
        },
    )


REPORT_REQUIRED_ARTIFACT_PATH_KEYS = (
    "scenario_eval_matrix",
    "evaluation_result",
    "policy_execution_manifest",
    "policy_execution_trace",
    "deployment_outcome_ledger",
    "prediction_vs_actual_deployment_summary",
    "proof_boundary",
)


def _report_referenced_artifact_audit(
    *,
    report: Mapping[str, Any],
    job_dir: Path,
) -> Dict[str, Any]:
    artifact_paths = _mapping(report.get("artifact_paths"))
    missing_path_keys = [
        key for key in REPORT_REQUIRED_ARTIFACT_PATH_KEYS if not _string(artifact_paths.get(key))
    ]
    local_artifacts: Dict[str, Dict[str, Any]] = {}
    missing_file_keys: List[str] = []
    for key in REPORT_REQUIRED_ARTIFACT_PATH_KEYS:
        value = artifact_paths.get(key)
        path = _job_local_artifact_path(job_dir, value)
        if path is None:
            continue
        artifact = _artifact(path, base_dir=job_dir)
        local_artifacts[key] = artifact
        if not artifact["exists"]:
            missing_file_keys.append(key)

    def _artifact_payload(key: str) -> Dict[str, Any]:
        path = _job_local_artifact_path(job_dir, artifact_paths.get(key))
        return _read_optional_mapping(path) if path is not None else {}

    matrix = _artifact_payload("scenario_eval_matrix")
    evaluation = _artifact_payload("evaluation_result")
    policy_manifest = _artifact_payload("policy_execution_manifest")
    proof_boundary = _artifact_payload("proof_boundary")

    mismatches: List[Dict[str, Any]] = []

    def _add_mismatch(field: str, report_value: Any, artifact_value: Any) -> None:
        if report_value != artifact_value:
            mismatches.append(
                {
                    "field": field,
                    "report_value": report_value,
                    "artifact_value": artifact_value,
                }
            )

    scenario_report = _mapping(report.get("scenario_eval"))
    if matrix:
        _add_mismatch(
            "scenario_eval.status",
            scenario_report.get("status"),
            matrix.get("status"),
        )
        _add_mismatch(
            "scenario_eval.scenario_eval_run_count",
            scenario_report.get("scenario_eval_run_count"),
            matrix.get("scenario_eval_run_count"),
        )
        _add_mismatch(
            "scenario_eval.variation_names_covered",
            sorted(_string_list(scenario_report.get("variation_names_covered"))),
            sorted(_string_list(matrix.get("variation_names_covered"))),
        )
    if evaluation:
        _add_mismatch("evaluation_status", report.get("evaluation_status"), evaluation.get("status"))
        scorecard = _mapping(report.get("evaluator_scores"))
        artifact_scorecard = _mapping(evaluation.get("standard_policy_scorecard"))
        missing_scorecard_fields = [
            field for field in SCORECARD_REQUIRED_FIELDS if field not in scorecard
        ]
        artifact_missing_scorecard_fields = [
            field for field in SCORECARD_REQUIRED_FIELDS if field not in artifact_scorecard
        ]
        if missing_scorecard_fields:
            mismatches.append(
                {
                    "field": "evaluator_scores",
                    "missing_report_fields": missing_scorecard_fields,
                }
            )
        if artifact_missing_scorecard_fields:
            mismatches.append(
                {
                    "field": "evaluation_result.standard_policy_scorecard",
                    "missing_artifact_fields": artifact_missing_scorecard_fields,
                }
            )
    policy_report = _mapping(report.get("policy_interface"))
    if policy_manifest:
        _add_mismatch(
            "policy_interface.policy_execution_status",
            policy_report.get("policy_execution_status"),
            policy_manifest.get("status"),
        )
        _add_mismatch(
            "policy_interface.selected_modalities",
            sorted(_string_list(policy_report.get("selected_modalities"))),
            sorted(_string_list(policy_manifest.get("selected_modalities"))),
        )
        _add_mismatch(
            "policy_interface.robot_policy_execution_proven",
            bool(policy_report.get("robot_policy_execution_proven")),
            bool(policy_manifest.get("robot_policy_execution_proven")),
        )
    proof_report = _mapping(report.get("proof_boundary"))
    if proof_boundary:
        for field in (
            "simulator_execution_proven",
            "robot_policy_execution_proven",
            "rank_fidelity_result_proven",
            "public_claim_upgrade_allowed",
        ):
            _add_mismatch(
                f"proof_boundary.{field}",
                bool(proof_report.get(field)),
                bool(proof_boundary.get(field)),
            )

    return {
        "required_artifact_path_keys": list(REPORT_REQUIRED_ARTIFACT_PATH_KEYS),
        "missing_artifact_path_keys": missing_path_keys,
        "local_artifacts": local_artifacts,
        "missing_artifact_file_keys": missing_file_keys,
        "artifact_mismatches": mismatches,
    }


def _report_generation_gate(job_dir: Path) -> Dict[str, Any]:
    report_path = job_dir / "robot_eval_report.json"
    markdown_path = job_dir / "robot_eval_report.md"
    report = _read_optional_mapping(report_path)
    required_sections = (
        "scenario_eval",
        "policy_interface",
        "evaluator_scores",
        "live_eval_closure",
        "requirement_coverage",
        "proof_boundary",
        "artifact_paths",
    )
    missing_sections = [
        section for section in required_sections if not _card_field_present(report.get(section))
    ]
    flow = _string_list(report.get("neutral_eval_harness_flow"))
    artifact_audit = _report_referenced_artifact_audit(report=report, job_dir=job_dir)
    blockers = []
    if not report_path.is_file():
        blockers.append("missing_robot_eval_report")
    if not markdown_path.is_file():
        blockers.append("missing_robot_eval_report_markdown")
    if report and report.get("status") != "generated":
        blockers.append("robot_eval_report_not_generated")
    if missing_sections:
        blockers.append("robot_eval_report_missing_required_sections")
    if "report_generated" not in flow:
        blockers.append("robot_eval_report_missing_report_generated_flow_step")
    if artifact_audit["missing_artifact_path_keys"]:
        blockers.append("robot_eval_report_missing_required_artifact_paths")
    if artifact_audit["missing_artifact_file_keys"]:
        blockers.append("robot_eval_report_referenced_artifacts_missing")
    if artifact_audit["artifact_mismatches"]:
        blockers.append("robot_eval_report_artifact_mismatches")
    return _gate(
        "report_generation",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "robot_eval_report": _artifact(report_path, base_dir=job_dir),
            "robot_eval_report_markdown": _artifact(markdown_path, base_dir=job_dir),
            "report_status": _string(report.get("status")),
            "required_sections": list(required_sections),
            "missing_sections": missing_sections,
            "neutral_eval_harness_flow": flow,
            "artifact_audit": artifact_audit,
        },
    )


def _live_simulator_gate(*, capture_root: Path, job_dir: Path) -> Dict[str, Any]:
    sim_path = job_dir / "simulator_service_result.json"
    trace_path = job_dir / "normalized_attempt_trace.json"
    matrix_path = job_dir / "scenario_eval_matrix.json"
    job_request_path = job_dir / "job_request.json"
    worker_manifest_path = job_dir / "worker_manifest.json"
    job_run_manifest_path = job_dir / "job_run_manifest.json"
    sim_result = _read_optional_mapping(sim_path)
    trace = _read_optional_mapping(trace_path)
    matrix = _read_optional_mapping(matrix_path)
    job_request = _read_optional_mapping(job_request_path)
    worker_manifest = _read_optional_mapping(worker_manifest_path)
    job_run_manifest = _read_optional_mapping(job_run_manifest_path)
    owner_path = capture_root / "pipeline" / "simulation_automation" / "owner_gpu_simulator_execution_proof_manifest.json"
    owner_proof = _read_optional_mapping(owner_path)
    owner_proof_audit = _owner_gpu_proof_manifest_audit(owner_proof)
    run_count = int(matrix.get("scenario_eval_run_count") or 0)
    covered_run_ids = {
        _string(attempt.get("scenario_eval_run_id"))
        for attempt in trace.get("attempts", []) or []
        if isinstance(attempt, Mapping) and _string(attempt.get("scenario_eval_run_id"))
    }
    required_run_ids = {
        _string(run.get("scenario_eval_run_id"))
        for run in matrix.get("runs", []) or []
        if isinstance(run, Mapping) and _string(run.get("scenario_eval_run_id"))
    }
    simulator_status = _string(sim_result.get("status"))
    trace_status = _string(trace.get("status"))
    non_fixture_simulator = _string(sim_result.get("framework")) != "fixture"
    simulator_proof_claimed = (
        bool(sim_result.get("simulator_execution_proven"))
        or bool(sim_result.get("simulators_run"))
    ) and non_fixture_simulator
    simulator_framework = _string(sim_result.get("framework"))
    expected_simulator_candidates = [
        _string(worker_manifest.get("simulator")),
        _string(job_run_manifest.get("simulator")),
        _string(job_request.get("simulator")),
        _string(job_request.get("requested_simulator")),
        _string(job_request.get("simulator_backend")),
        _string(job_request.get("simulator_preference")),
        simulator_framework,
        _string(owner_proof.get("simulator_backend")),
    ]
    expected_simulator = ""
    for candidate in expected_simulator_candidates:
        normalized = candidate.lower().strip()
        if normalized in LIVE_SIMULATOR_FRAMEWORKS:
            expected_simulator = normalized
            break
        if normalized in {"mujoco_first", "mujoco-first", "default_mujoco"}:
            expected_simulator = "mujoco"
            break
        if normalized in {"isaac_first", "isaac-first", "isaac"}:
            expected_simulator = "isaac_sim"
            break
    if not expected_simulator:
        expected_simulator = "mujoco"
    sim_completed = (
        bool(sim_result.get("simulator_execution_proven"))
        and bool(sim_result.get("simulators_run"))
        and non_fixture_simulator
        and simulator_status == "completed"
    )
    owner_accepted = bool(owner_proof_audit["accepted"])
    service_isaac_asset_proven = (
        simulator_framework in ISAAC_LIVE_SIMULATOR_FRAMEWORKS
        and sim_completed
        and bool(
            sim_result.get("isaac_sim_execution_proven")
            or sim_result.get("isaac_robot_asset_execution_proven")
        )
        and bool(
            sim_result.get("unitree_g1_asset_spawned")
            or sim_result.get("unitree_g1_robot_asset_spawned")
        )
    )
    service_mujoco_asset_proven = (
        simulator_framework in MUJOCO_LIVE_SIMULATOR_FRAMEWORKS
        and sim_completed
        and bool(
            sim_result.get("mujoco_g1_asset_execution_proven")
            or sim_result.get("mujoco_g1_asset_spawned")
            or sim_result.get("unitree_g1_asset_spawned")
            or sim_result.get("unitree_g1_robot_asset_spawned")
        )
    )
    owner_isaac_asset_proven = (
        owner_accepted
        and _string(owner_proof.get("simulator_backend")) in ISAAC_LIVE_SIMULATOR_FRAMEWORKS
        and bool(owner_proof.get("isaac_sim_execution_proven"))
        and bool(owner_proof.get("isaac_robot_asset_execution_proven"))
        and bool(owner_proof.get("unitree_g1_asset_spawned"))
    )
    owner_mujoco_asset_proven = (
        owner_accepted
        and _string(owner_proof.get("simulator_backend")) in MUJOCO_LIVE_SIMULATOR_FRAMEWORKS
        and bool(
            owner_proof.get("mujoco_g1_asset_execution_proven")
            or _mapping(owner_proof.get("robot_asset")).get("mujoco_g1_asset_execution_proven")
        )
        and bool(
            owner_proof.get("mujoco_g1_asset_spawned")
            or _mapping(owner_proof.get("evidence")).get("mujoco_g1_asset_spawned")
        )
    )
    blockers = []
    if not sim_completed and not owner_accepted:
        blockers.append("live_simulator_execution_not_proven")
    if (
        expected_simulator in ISAAC_LIVE_SIMULATOR_FRAMEWORKS
        and (sim_completed or owner_accepted)
        and not (service_isaac_asset_proven or owner_isaac_asset_proven)
    ):
        blockers.append("isaac_sim_unitree_g1_execution_not_proven")
    if (
        expected_simulator in MUJOCO_LIVE_SIMULATOR_FRAMEWORKS
        and (sim_completed or owner_accepted)
        and not (service_mujoco_asset_proven or owner_mujoco_asset_proven)
    ):
        blockers.append("mujoco_g1_execution_not_proven")
    if (
        owner_path.is_file()
        and (
            bool(owner_proof.get("owner_gpu_simulator_execution_proven"))
            or _string(owner_proof.get("status")) == "accepted"
        )
        and owner_proof_audit["blockers"]
    ):
        blockers.extend(_string_list(owner_proof_audit["blockers"]))
    if simulator_proof_claimed and simulator_status != "completed":
        blockers.append("simulator_service_result_not_completed")
    if trace_path.is_file() and trace_status != "completed" and not owner_accepted:
        blockers.append("normalized_attempt_trace_not_completed")
    attempt_count = int(trace.get("attempt_count") or 0)
    if run_count > 0 and attempt_count != run_count:
        blockers.append("simulator_execution_missing_scenario_variation_run_coverage")
    if required_run_ids and not required_run_ids.issubset(covered_run_ids):
        blockers.append("simulator_execution_missing_required_scenario_eval_run_ids")
    required_trace_count = int(trace.get("required_scenario_eval_run_count") or 0)
    if (
        trace.get("scenario_eval_run_coverage_complete") is False
        and (required_run_ids or required_trace_count > 0)
    ):
        blockers.append("simulator_execution_incomplete_scenario_eval_run_coverage")
    return _gate(
        "live_simulator_execution",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "simulator_service_result": _artifact(sim_path, base_dir=job_dir),
            "normalized_attempt_trace": _artifact(trace_path, base_dir=job_dir),
            "scenario_eval_matrix": _artifact(matrix_path, base_dir=job_dir),
            "simulator_status": sim_result.get("status"),
            "normalized_attempt_trace_status": trace_status or None,
            "simulator_framework": sim_result.get("framework"),
            "expected_simulator": expected_simulator,
            "simulators_run": bool(sim_result.get("simulators_run")),
            "simulator_execution_proven": bool(sim_result.get("simulator_execution_proven")),
            "service_isaac_unitree_g1_execution_proven": service_isaac_asset_proven,
            "service_mujoco_unitree_g1_execution_proven": service_mujoco_asset_proven,
            "attempt_count": attempt_count,
            "scenario_eval_run_count": run_count,
            "required_scenario_eval_run_count": required_trace_count,
            "covered_scenario_eval_run_count": len(covered_run_ids),
            "scenario_eval_run_coverage_complete": trace.get(
                "scenario_eval_run_coverage_complete"
            ),
            "attempt_count_matches_matrix_count": trace.get(
                "attempt_count_matches_matrix_count"
            ),
            "scenario_eval_run_id_coverage_exact": trace.get(
                "scenario_eval_run_id_coverage_exact"
            ),
            "owner_gpu_proof_manifest": _artifact(owner_path, base_dir=job_dir),
            "owner_gpu_proof_status": owner_proof.get("status"),
            "owner_isaac_unitree_g1_execution_proven": owner_isaac_asset_proven,
            "owner_mujoco_unitree_g1_execution_proven": owner_mujoco_asset_proven,
            "owner_gpu_proof_audit": owner_proof_audit,
        },
    )


def _live_policy_execution_gate(job_dir: Path) -> Dict[str, Any]:
    manifest_path = job_dir / "policy_execution_manifest.json"
    trace_path = job_dir / "policy_execution_trace.json"
    matrix_path = job_dir / "scenario_eval_matrix.json"
    manifest = _read_optional_mapping(manifest_path)
    trace = _read_optional_mapping(trace_path)
    matrix = _read_optional_mapping(matrix_path)
    policy_result_audit = _policy_execution_result_audit(manifest)
    run_count = int(matrix.get("scenario_eval_run_count") or 0)
    attempts = [
        dict(attempt)
        for attempt in trace.get("attempts", []) or []
        if isinstance(attempt, Mapping)
    ]
    covered_run_ids = {
        _string(attempt.get("scenario_eval_run_id"))
        for attempt in attempts
        if _string(attempt.get("scenario_eval_run_id"))
    }
    required_run_ids = {
        _string(run.get("scenario_eval_run_id"))
        for run in matrix.get("runs", []) or []
        if isinstance(run, Mapping) and _string(run.get("scenario_eval_run_id"))
    }
    attempts_missing_action_or_skill_trace = [
        _string(attempt.get("attempt_id") or attempt.get("attemptId"))
        or _string(attempt.get("scenario_eval_run_id") or attempt.get("scenarioEvalRunId"))
        or f"attempt_{index:04d}"
        for index, attempt in enumerate(attempts, start=1)
        if not (
            isinstance(attempt.get("actions"), list)
            and bool(attempt.get("actions"))
        )
        and not (
            isinstance(attempt.get("skills"), list)
            and bool(attempt.get("skills"))
        )
    ]
    blockers = []
    if not bool(manifest.get("robot_policy_execution_proven")):
        blockers.append("live_policy_execution_not_proven")
    if not bool(trace.get("robot_policy_execution_proven")):
        blockers.append("policy_execution_trace_not_proven")
    if _string(manifest.get("status")) != "completed":
        blockers.append("policy_execution_manifest_not_completed")
    if _string(trace.get("status")) != "completed":
        blockers.append("policy_execution_trace_not_completed")
    if int(trace.get("attempt_count") or 0) <= 0:
        blockers.append("policy_execution_trace_empty")
    if run_count > 0 and int(trace.get("attempt_count") or 0) < run_count:
        blockers.append("policy_execution_missing_scenario_variation_run_coverage")
    if required_run_ids and not required_run_ids.issubset(covered_run_ids):
        blockers.append("policy_execution_missing_required_scenario_eval_run_ids")
    if attempts_missing_action_or_skill_trace:
        blockers.append("policy_execution_attempts_missing_action_or_skill_trace")
    if bool(manifest.get("robot_policy_execution_proven")):
        blockers.extend(_string_list(policy_result_audit["blockers"]))
    return _gate(
        "live_policy_execution",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "policy_execution_manifest": _artifact(manifest_path, base_dir=job_dir),
            "policy_execution_trace": _artifact(trace_path, base_dir=job_dir),
            "scenario_eval_matrix": _artifact(matrix_path, base_dir=job_dir),
            "policy_execution_manifest_status": _string(manifest.get("status")) or None,
            "policy_execution_trace_status": _string(trace.get("status")) or None,
            "attempt_count": int(trace.get("attempt_count") or 0),
            "scenario_eval_run_count": run_count,
            "covered_scenario_eval_run_count": len(covered_run_ids),
            "attempts_missing_action_or_skill_trace": attempts_missing_action_or_skill_trace,
            "policy_execution_result_audit": policy_result_audit,
            "missing_scenario_eval_run_count": int(
                manifest.get("missing_scenario_eval_run_count")
                or trace.get("missing_scenario_eval_run_count")
                or 0
            ),
            "missing_scenario_eval_run_ids": _string_list(
                manifest.get("missing_scenario_eval_run_ids")
                or trace.get("missing_scenario_eval_run_ids")
            ),
            "selected_modalities": manifest.get("selected_modalities") or [],
            "default_test_policy_execution_proven": bool(
                manifest.get("default_test_policy_execution_proven")
            ),
            "robot_team_policy_execution_proven": bool(
                manifest.get("robot_team_policy_execution_proven")
            ),
        },
    )












def build_live_robot_eval_closure_manifest(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    job_request: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    resolved_job_dir = Path(job_dir).resolve()
    ensure_dir(resolved_job_dir)
    request = dict(job_request or _read_optional_mapping(resolved_job_dir / "job_request.json"))
    timestamp = generated_at or utc_now_iso()
    evidence, evidence_sources = _load_live_evidence(
        capture_root=context.capture_root,
        job_dir=resolved_job_dir,
        job_request=request,
    )

    gates: Dict[str, Dict[str, Any]] = {}
    gates["site_capture"] = _site_capture_gate(
        capture_root=context.capture_root,
        base_dir=resolved_job_dir,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
    )
    gates["task_definitions"] = _robot_eval_dataset_gate(
        capture_root=context.capture_root,
        job_dir=resolved_job_dir,
        gate_id="task_definitions",
        filename="task_cards.json",
        count_fields=("task_card_count", "task_count"),
        required_card_fields=TASK_CARD_REQUIRED_FIELDS,
    )
    gates["scenario_library"] = _scenario_library_gate(capture_root=context.capture_root, job_dir=resolved_job_dir)
    gates["robot_pov_generation"] = _robot_pov_gate(resolved_job_dir)
    gates["scenario_eval_suite"] = _scenario_eval_suite_gate(capture_root=context.capture_root, job_dir=resolved_job_dir)
    gates["failure_labels"] = _failure_labels_gate(resolved_job_dir)
    gates["evaluation_methodology"] = _evaluation_methodology_gate(resolved_job_dir)
    gates["policy_interface"] = _policy_interface_gate(
        capture_root=context.capture_root,
        job_dir=resolved_job_dir,
    )
    gates["simulator_engine_plugins"] = _simulator_plugins_gate(capture_root=context.capture_root, job_dir=resolved_job_dir)
    gates["report_generation"] = _report_generation_gate(resolved_job_dir)
    gates["live_evidence_integrity"] = _live_evidence_integrity_gate(evidence)
    gates["webapp_upstream_truth"] = _webapp_upstream_gate(
        capture_root=context.capture_root,
        job_dir=resolved_job_dir,
        job_request=request,
        evidence=evidence,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
    )
    gates["rights_privacy_scope"] = _rights_gate(
        job_request=request,
        evidence=evidence,
        capture_root=context.capture_root,
        job_dir=resolved_job_dir,
    )
    gates["review_acceptance"] = _review_acceptance_gate(
        evidence=evidence,
        capture_root=context.capture_root,
        job_dir=resolved_job_dir,
    )
    gates["signed_delivery_access"] = _signed_delivery_access_gate(evidence)
    gates["real_robot_pov_evidence"] = _real_robot_pov_evidence_gate(
        job_dir=resolved_job_dir,
        evidence=evidence,
    )
    gates["real_world_validation_loop"] = _real_world_validation_loop_gate(resolved_job_dir)
    gates["predicted_vs_actual_calibration"] = _predicted_vs_actual_calibration_gate(
        resolved_job_dir
    )
    gates["safety_contact_physics_readiness"] = _safety_contact_physics_gate(
        evidence=evidence,
        capture_root=context.capture_root,
        job_dir=resolved_job_dir,
    )
    gates["live_simulator_execution"] = _live_simulator_gate(capture_root=context.capture_root, job_dir=resolved_job_dir)
    gates["live_policy_execution"] = _live_policy_execution_gate(resolved_job_dir)

    ordered_gate_ids = [*REPO_LOCAL_GATES, *LIVE_EXTERNAL_GATES]
    repo_local_ready = all(gates[gate_id]["passed"] for gate_id in REPO_LOCAL_GATES)
    live_external_ready = all(gates[gate_id]["passed"] for gate_id in LIVE_EXTERNAL_GATES)
    live_external_proof_ready = all(
        bool(gates[gate_id].get("proof_boolean"))
        for gate_id in LIVE_EXTERNAL_PROOF_GATES
    )
    all_ready = repo_local_ready and live_external_ready and live_external_proof_ready
    requirement_coverage = _requirement_coverage(gates)
    blockers = [
        f"{gate_id}:{blocker}"
        for gate_id in ordered_gate_ids
        for blocker in gates[gate_id].get("blockers", [])
    ]
    if all_ready:
        status = "live_end_to_end_verified"
    elif repo_local_ready and live_external_ready:
        status = "local_artifacts_ready_optional_live_proof_not_claimed"
    elif repo_local_ready:
        status = "local_artifacts_ready_live_external_blocked"
    else:
        status = "blocked"

    proof_boundary = {
        **dict(CLAIM_BOUNDARY),
        "repo_local_artifacts_ready": repo_local_ready,
        "live_external_ready": live_external_ready,
        "live_external_proof_ready": live_external_proof_ready,
        "live_end_to_end_verified": all_ready,
        "review_acceptance_proven": bool(gates["review_acceptance"]["passed"]),
        "rights_privacy_scope_proven": bool(gates["rights_privacy_scope"]["passed"]),
        "signed_delivery_access_proven": bool(gates["signed_delivery_access"]["passed"]),
        "delivery_access_is_deployment_approval": False,
        "package_delivery_is_deployment_approval": False,
        "deployment_approval_proven": False,
        "physical_robot_readiness_proven": False,
        "safety_validation_proven": bool(
            _mapping(gates["safety_contact_physics_readiness"].get("evidence")).get(
                "safety_validated"
            )
        ),
        "simulator_execution_proven": bool(gates["live_simulator_execution"]["passed"]),
        "robot_policy_execution_proven": bool(gates["live_policy_execution"]["passed"]),
        "rank_fidelity_result_proven": all_ready,
        "public_claim_upgrade_allowed": all_ready and bool(gates["rights_privacy_scope"]["passed"]),
    }
    robot_team_beta_readiness = _robot_team_beta_readiness_summary(
        gates=gates,
        job_dir=resolved_job_dir,
        repo_local_ready=repo_local_ready,
        live_external_ready=live_external_ready,
        live_end_to_end_verified=all_ready,
    )
    sc3_protocol = _read_optional_mapping(resolved_job_dir / "sc3_eval_protocol.json")
    sc3_eval_protocol_summary = {
        "path": "sc3_eval_protocol.json",
        "present": bool(sc3_protocol),
        "status": _string(sc3_protocol.get("status")) or "not_available",
        "correlation_claim_status": (
            _string(sc3_protocol.get("correlation_claim_status")) or "correlation_not_measured"
        ),
        "claim_boundary": {
            "sc3_protocol_is_not_a_closure_gate": True,
            "sc3_self_consistency_is_reliability_support_only": True,
            "sc3_protocol_does_not_claim_blueprint_90_percent_accuracy": True,
        },
    }

    manifest = {
        "schema_version": LIVE_ROBOT_EVAL_CLOSURE_SCHEMA_VERSION,
        "generated_at": timestamp,
        "status": status,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "capture_root": str(context.capture_root),
        "job_dir": str(resolved_job_dir),
        "repo_local_artifacts_ready": repo_local_ready,
        "live_external_ready": live_external_ready,
        "live_external_proof_ready": live_external_proof_ready,
        "live_end_to_end_verified": all_ready,
        "gate_order": ordered_gate_ids,
        "gates": gates,
        "requirement_coverage": requirement_coverage,
        "robot_team_beta_readiness": robot_team_beta_readiness,
        "sc3_eval_protocol": sc3_eval_protocol_summary,
        "blockers": blockers,
        "evidence_sources": evidence_sources,
        "proof_boundary": proof_boundary,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    path = Path(output_path) if output_path else resolved_job_dir / "live_eval_closure_manifest.json"
    ensure_dir(path.parent)
    write_json(path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit live robot-eval closure for a job directory"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--job-request")
    parser.add_argument("--output-path")
    args = parser.parse_args(argv)
    request = _read_optional_mapping(Path(args.job_request)) if args.job_request else None
    result = build_live_robot_eval_closure_manifest(
        capture_root=args.capture_root,
        job_dir=args.job_dir,
        job_request=request,
        output_path=args.output_path,
    )
    manifest_path = args.output_path or str(Path(args.job_dir).resolve() / "live_eval_closure_manifest.json")
    print(f"[live-robot-eval-closure] manifest={manifest_path}")
    print(f"[live-robot-eval-closure] status={result['status']}")
    if result["blockers"]:
        print(f"[live-robot-eval-closure] blockers={len(result['blockers'])}")
    return 0 if result["status"] == "live_end_to_end_verified" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
