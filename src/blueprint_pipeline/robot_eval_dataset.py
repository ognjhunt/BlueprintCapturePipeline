"""Repo-local real-site robot evaluation dataset artifact lane.

This module writes deterministic dataset/workflow artifacts for robot task
evaluation without calling providers, running simulators, downloading models, or
claiming generated-world rank fidelity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .common import ensure_dir, read_json_any, write_json, write_text
from .local_capture import resolve_local_capture_context

ROBOT_EVAL_DATASET_SCHEMA_VERSION = "real_site_robot_eval_dataset_manifest.v1"
ROBOT_EVAL_DATASET_V01_SCHEMA_VERSION = "real_site_robot_eval_dataset_manifest.v0.1"
ROBOT_EVAL_DATASET_VERSION = "0.1"
SITE_CARD_SCHEMA_VERSION = "real_site_robot_eval_site_card.v0.1"
TASK_CARDS_SCHEMA_VERSION = "real_site_robot_eval_task_cards.v0.1"
SCENARIO_CARDS_SCHEMA_VERSION = "real_site_robot_eval_scenario_cards.v0.1"
EVAL_CARDS_SCHEMA_VERSION = "real_site_robot_eval_eval_cards.v0.1"
ANNOTATION_BACKLOG_SCHEMA_VERSION = "real_site_robot_eval_annotation_backlog.v0.1"
PROOF_BOUNDARIES_SCHEMA_VERSION = "real_site_robot_eval_proof_boundaries.v0.1"
ROBOT_TASK_LIBRARY_SCHEMA_VERSION = "robot_task_library.v1"
SCENARIO_LIBRARY_SCHEMA_VERSION = "robot_eval_scenario_library.v1"
PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION = "prediction_outcome_ledger.v1"
ROBOT_TEAM_TEST_SUBMISSION_MODALITIES_SCHEMA_VERSION = (
    "robot_team_test_submission_modalities.v0.1"
)
ROBOT_EVAL_INPUTS_EVIDENCE_CONTRACT_SCHEMA_VERSION = "robot_eval_inputs_evidence_contract.v1"
RIGHTS_PACKET_SCHEMA_VERSION = "real_site_robot_eval_rights_packet.v1"
RIGHTS_LEDGER_SCHEMA_VERSION = "real_site_robot_eval_rights_ledger.v1"
TASK_ONTOLOGY_SCHEMA_VERSION = "real_site_robot_eval_task_ontology.v1"
SCENARIO_FAMILY_LIBRARY_SCHEMA_VERSION = "real_site_robot_eval_scenario_family_library.v1"
SCORING_METHODOLOGY_SCHEMA_VERSION = "real_site_robot_eval_scoring_methodology.v1"
RECORDED_TRACE_EVAL_REPORT_SCHEMA_VERSION = "recorded_action_trace_eval_report.v1"
PREDICTION_VS_ACTUAL_SUMMARY_SCHEMA_VERSION = "prediction_vs_actual_summary.v1"
TASK_THRESHOLDS_SCHEMA_VERSION = "real_site_robot_eval_task_thresholds.v1"
PUBLICATION_READINESS_SCHEMA_VERSION = "real_site_robot_eval_publication_readiness.v1"

DETERMINISTIC_DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"

FAIL_CLOSED_STATUSES = [
    "capture_grounded_ready",
    "needs_robot_pov",
    "needs_human_demo",
    "needs_action_logs",
    "needs_actual_outcome",
    "needs_policy_api_endpoint_ref",
    "needs_docker_container_ref",
    "needs_recorded_action_trace_ref",
    "needs_high_level_skill_trace_ref",
    "needs_teleop_demo_ref",
    "needs_sim_controller_plugin_ref",
    "blocked_rights_privacy",
    "review_only_no_rank_fidelity",
]

ROBOT_TEAM_TEST_MODALITY_REQUIREMENTS: List[Dict[str, Any]] = [
    {
        "modality_id": "policy_api_endpoint",
        "label": "Policy API endpoint",
        "missing_evidence_status": "needs_policy_api_endpoint_ref",
        "required_reference_fields": [
            "endpointUrl",
            "authHandling",
            "observationSchemaRef",
            "actionSchemaRef",
            "runtimeConstraints",
            "callbackLogUri",
            "ownerContact",
        ],
    },
    {
        "modality_id": "docker_container",
        "label": "Docker container",
        "missing_evidence_status": "needs_docker_container_ref",
        "required_reference_fields": [
            "imageRef",
            "digestChecksum",
            "entrypoint",
            "environmentContract",
            "hardwareNeeds",
            "ioSchemaRef",
            "runtimeNotes",
        ],
    },
    {
        "modality_id": "recorded_action_trace",
        "label": "Recorded action traces",
        "missing_evidence_status": "needs_recorded_action_trace_ref",
        "required_reference_fields": [
            "traceManifestUri",
            "format",
            "taskScenarioMapping",
            "timestampAlignment",
            "observationActionAlignment",
            "successFailureLabels",
            "checksum",
        ],
    },
    {
        "modality_id": "high_level_skill_trace",
        "label": "High-level skill traces",
        "missing_evidence_status": "needs_high_level_skill_trace_ref",
        "required_reference_fields": [
            "skillTaxonomyVersion",
            "orderedSkillSequence",
            "preconditionsPostconditions",
            "failureLabels",
            "sourceType",
            "confidenceCoverageNote",
        ],
    },
    {
        "modality_id": "teleop_demo",
        "label": "Teleop demos",
        "missing_evidence_status": "needs_teleop_demo_ref",
        "required_reference_fields": [
            "demoArtifactUri",
            "operatorDevice",
            "controlMapping",
            "timeSync",
            "taskScenarioMapping",
            "rightsPrivacyAttestation",
            "labels",
        ],
    },
    {
        "modality_id": "sim_controller_plugin",
        "label": "Sim controller plugin",
        "missing_evidence_status": "needs_sim_controller_plugin_ref",
        "required_reference_fields": [
            "simulatorFramework",
            "pluginUri",
            "supportedControlModes",
            "observationActionSpaces",
            "replayExportPath",
            "compatibilityNotes",
        ],
    },
]

PUBLICATION_REQUIRED_ARTIFACTS = [
    "site_card",
    "task_cards",
    "scenario_cards",
    "eval_cards",
    "task_ontology_v1",
    "scenario_family_library",
    "scoring_methodology",
    "proof_boundaries",
    "task_thresholds",
    "publication_readiness",
]

TASK_ONTOLOGY_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "task_id": "navigate_to_station",
        "task_family": "navigation",
        "aliases": ["navigate", "go_to_station", "station approach", "route to station"],
        "parameters": ["start_zone", "goal_station_id", "route_constraints"],
        "success_criteria": [
            "robot reaches the goal station",
            "cycle time stays under the buyer threshold",
            "no unsafe proximity, collision, or blocked-zone violation is recorded",
        ],
    },
    {
        "task_id": "inspect_shelf",
        "task_family": "inspection",
        "aliases": ["shelf inspection", "scan shelf", "inventory shelf"],
        "parameters": ["shelf_id", "inspection_targets", "required_viewpoints"],
        "success_criteria": [
            "required shelf faces are observed",
            "inspection labels are linked to the scenario and evidence refs",
            "missed-label and occlusion failures are recorded",
        ],
    },
    {
        "task_id": "move_tote",
        "task_family": "material_handling",
        "aliases": ["tote move", "move bin", "transport tote"],
        "parameters": ["source_zone", "destination_zone", "tote_id", "load_limit"],
        "success_criteria": [
            "target tote reaches the destination zone",
            "no object drop or wrong-object event is recorded",
            "intervention count and cycle time are recorded",
        ],
    },
    {
        "task_id": "cart_to_conveyor_transfer",
        "task_family": "transfer",
        "aliases": ["cart conveyor transfer", "cart to conveyor", "load conveyor"],
        "parameters": ["cart_id", "conveyor_id", "handoff_pose", "object_type"],
        "success_criteria": [
            "item transfers from cart to conveyor",
            "collision and drop metrics remain within threshold",
            "handoff timing and recovery attempts are recorded",
        ],
    },
    {
        "task_id": "line_side_delivery",
        "task_family": "delivery",
        "aliases": ["line side delivery", "deliver parts", "station delivery"],
        "parameters": ["pickup_zone", "line_station_id", "delivery_window_seconds"],
        "success_criteria": [
            "payload is delivered to the requested line station",
            "human crossing and blocked-path events are labeled",
            "operator intervention count is recorded",
        ],
    },
    {
        "task_id": "pick_known_object",
        "task_family": "manipulation",
        "aliases": ["pick object", "pick known item", "grasp known object"],
        "parameters": ["object_id", "source_container_id", "grasp_constraints"],
        "success_criteria": [
            "known object is selected",
            "wrong-object and object-drop metrics remain zero",
            "grasp/contact evidence is linked",
        ],
    },
    {
        "task_id": "place_object_into_bin",
        "task_family": "pick_place",
        "aliases": ["place in bin", "bin placement", "place item"],
        "parameters": ["object_id", "target_bin_id", "placement_tolerance"],
        "success_criteria": [
            "object is placed inside the target bin",
            "placement, drop, and wrong-object outcomes are labeled",
            "cycle time and interventions are recorded",
        ],
    },
    {
        "task_id": "blocked_path_recovery",
        "task_family": "recovery",
        "aliases": ["blocked path", "route recovery", "obstacle recovery"],
        "parameters": ["route_id", "blocker_type", "recovery_policy"],
        "success_criteria": [
            "robot chooses a safe recovery path or holds",
            "unsafe proximity and intervention events are labeled",
            "recovery success is recorded",
        ],
    },
    {
        "task_id": "human_crossing_safety_response",
        "task_family": "safety_response",
        "aliases": ["human crossing", "pedestrian crossing", "people safety response"],
        "parameters": ["crossing_zone", "minimum_distance_m", "stop_or_yield_policy"],
        "success_criteria": [
            "robot yields, stops, or reroutes per policy",
            "unsafe proximity stays under threshold",
            "safety response timing is recorded",
        ],
    },
    {
        "task_id": "open_door_enter_room",
        "task_family": "articulation_navigation",
        "aliases": ["open door", "enter room", "doorway entry"],
        "parameters": ["door_id", "handle_type", "entry_zone", "clearance_width_m"],
        "success_criteria": [
            "door interaction reaches the required state",
            "robot enters the target room without contact violation",
            "articulation, recovery, and timeout metrics are recorded",
        ],
    },
]

SCENARIO_VARIATION_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "variation_id": "lighting_variation",
        "label": "Lighting variation",
        "default_status": "representative-mock",
    },
    {
        "variation_id": "object_rotation",
        "label": "Object rotation",
        "default_status": "representative-mock",
    },
    {
        "variation_id": "cart_shifted",
        "label": "Cart shifted",
        "default_status": "representative-mock",
    },
    {
        "variation_id": "blocked_path",
        "label": "Blocked path",
        "default_status": "agent-inferred-needs-review",
    },
    {
        "variation_id": "human_crossing",
        "label": "Human crossing",
        "default_status": "agent-inferred-needs-review",
    },
    {
        "variation_id": "forklift_nearby",
        "label": "Forklift nearby",
        "default_status": "agent-inferred-needs-review",
    },
    {
        "variation_id": "occlusion",
        "label": "Occlusion",
        "default_status": "agent-inferred-needs-review",
    },
    {
        "variation_id": "glare",
        "label": "Glare",
        "default_status": "agent-inferred-needs-review",
    },
    {
        "variation_id": "missing_label",
        "label": "Missing label",
        "default_status": "agent-inferred-needs-review",
    },
    {
        "variation_id": "wrong_object_nearby",
        "label": "Wrong object nearby",
        "default_status": "agent-inferred-needs-review",
    },
    {
        "variation_id": "narrow_approach_angle",
        "label": "Narrow approach angle",
        "default_status": "agent-inferred-needs-review",
    },
]

SCORING_METRIC_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "metric_id": "success_rate",
        "aggregation": "successful_attempts / attempt_count",
        "higher_is_better": True,
    },
    {
        "metric_id": "cycle_time",
        "aggregation": "mean cycle_time_seconds",
        "higher_is_better": False,
    },
    {
        "metric_id": "intervention_rate",
        "aggregation": "total interventions / attempt_count",
        "higher_is_better": False,
    },
    {
        "metric_id": "unsafe_proximity",
        "aggregation": "unsafe proximity event count",
        "higher_is_better": False,
    },
    {
        "metric_id": "collision_risk",
        "aggregation": "collision or contact event count",
        "higher_is_better": False,
    },
    {
        "metric_id": "object_drop",
        "aggregation": "object drop event count",
        "higher_is_better": False,
    },
    {
        "metric_id": "wrong_object",
        "aggregation": "wrong object event count",
        "higher_is_better": False,
    },
    {
        "metric_id": "timeout",
        "aggregation": "timeout event count",
        "higher_is_better": False,
    },
    {
        "metric_id": "recovery_success",
        "aggregation": "successful recovery attempts / recovery attempt count",
        "higher_is_better": True,
    },
    {
        "metric_id": "world_model_uncertainty",
        "aggregation": "world-model uncertainty or proof/label completeness bucket",
        "higher_is_better": False,
    },
    {
        "metric_id": "sim_vs_real_calibration_score",
        "aggregation": (
            "paired predicted-vs-actual agreement over exact scenario_eval_run_id "
            "+ scenario_variation_instance_id matches"
        ),
        "higher_is_better": False,
    },
]

DEFAULT_TASK_THRESHOLD_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "pick_place": {
        "threshold_profile_id": "pick_place_default_v1",
        "min_success_rate": 0.85,
        "max_cycle_time_seconds": 45.0,
        "max_intervention_count": 0,
        "max_safety_event_count": 0,
        "max_collision_event_count": 0,
        "max_object_drop_count": 0,
        "max_wrong_object_count": 0,
        "max_timeout_count": 0,
    },
    "navigation": {
        "threshold_profile_id": "navigation_default_v1",
        "min_success_rate": 0.9,
        "max_cycle_time_seconds": 60.0,
        "max_intervention_count": 0,
        "max_safety_event_count": 0,
        "max_collision_event_count": 0,
        "max_object_drop_count": 0,
        "max_wrong_object_count": 0,
        "max_timeout_count": 0,
    },
    "general": {
        "threshold_profile_id": "general_task_default_v1",
        "min_success_rate": 0.8,
        "max_cycle_time_seconds": 60.0,
        "max_intervention_count": 0,
        "max_safety_event_count": 0,
        "max_collision_event_count": 0,
        "max_object_drop_count": 0,
        "max_wrong_object_count": 0,
        "max_timeout_count": 0,
    },
}

TASK_THRESHOLD_BUYER_OVERRIDE_SCHEMA: Dict[str, str] = {
    "min_success_rate": "number_0_to_1",
    "max_cycle_time_seconds": "positive_number_or_null",
    "max_intervention_count": "non_negative_integer",
    "max_safety_event_count": "non_negative_integer",
    "max_collision_event_count": "non_negative_integer",
    "max_object_drop_count": "non_negative_integer",
    "max_wrong_object_count": "non_negative_integer",
    "max_timeout_count": "non_negative_integer",
}

PREDICTION_SOURCES = [
    "marble_review",
    "simready_review",
    "cosmos_preflight",
    "human_eval",
    "future_provider",
    "simulator_trace",
    "robot_trial",
]

ACTUAL_SOURCES = [
    "heldout_revisit",
    "robot_pilot",
    "simulator_trace",
    "human_demo",
    "teleop",
    "operator_report",
]

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "repo_local_real_site_robot_eval_dataset_contract_only",
    "repo_local_only": True,
    "live_provider_jobs_called": False,
    "simulators_run": False,
    "model_downloads_performed": False,
    "messages_sent": False,
    "payments_touched": False,
    "deployments_performed": False,
    "simulator_execution_proven": False,
    "rank_fidelity_result_proven": False,
    "deployment_outcome_proven": False,
    "non_ranking_operational_claim_validated": False,
    "public_claim_upgrade_allowed": False,
    "allowed_display": [
        "advisory robot-evaluation dataset contract",
        "task/scenario library",
        "evidence requirements",
        "prediction-vs-actual ledger schema",
        "missing-proof labels",
    ],
    "disallowed_claims": [
        "robot_ready",
        "deployment_ready",
        "non_ranking_operational_claim_validated",
        "simulator_execution_completed",
        "robot_trial_passed",
        "policy_execution_passed",
        "guaranteed_success_rate",
        "guaranteed_cycle_time",
        "guaranteed_intervention_rate",
    ],
    "operational_readiness_requires": [
        "robot POV evidence",
        "human demonstration evidence where required",
        "action or teleoperation logs",
        "simulator traces or real robot trial logs from the owning system",
        "prediction-vs-actual outcome records",
        "rights/privacy clearance for the exact site and use",
        "buyer-approved evaluation methodology",
    ],
}


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> List[str]:
    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    seen: set[str] = set()
    out: List[str] = []
    for item in values:
        text = _string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _float_triplet(value: Any, *, fallback: Sequence[float]) -> List[float]:
    out: List[float] = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in list(value)[:3]:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                out.append(0.0)
    while len(out) < 3:
        out.append(float(fallback[len(out)]))
    return out[:3]


def _pose_triplet_or_none(value: Any) -> List[float] | None:
    if isinstance(value, Mapping):
        for key in ("xyz", "pose", "position", "center"):
            nested = _pose_triplet_or_none(value.get(key))
            if nested is not None:
                return nested
        if "x" in value and "y" in value:
            value = [value.get("x"), value.get("y"), value.get("z", 0.0)]
        else:
            return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    out: List[float] = []
    for item in list(value)[:3]:
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        out.append(round(parsed, 6))
    while len(out) < 3:
        out.append(0.0)
    return out[:3]


def _pose_card(
    *,
    zone_id: str,
    role: str,
    pose: Sequence[float] | None,
    label: str,
    source: str,
    confidence: str,
) -> Dict[str, Any]:
    pose_xyz = _pose_triplet_or_none(pose)
    validated = pose_xyz is not None
    return {
        "zone_id": zone_id,
        "role": role,
        "label": label,
        "pose_xyz": pose_xyz,
        "frame": "site_coordinate_frame",
        "validation_status": "validated_finite_site_pose" if validated else "blocked_missing_pose",
        "validated": validated,
        "label_source": source,
        "confidence": confidence if validated else "missing",
        "claim_boundary": "site_zone_pose_is_capture_grounded_eval_input_not_navigation_safety_proof",
    }


def _number(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return _string(value).lower() in {"1", "true", "yes", "y", "success", "succeeded"}


def _stable_slug(value: Any, *, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", _string(value).lower()).strip("_")
    return (text or fallback)[:96]


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _relative_if_file(base_dir: Path, target: Path) -> str | None:
    return _relative_to(base_dir, target) if target.is_file() else None


def _sha_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


def _deterministic_generated_at(*payloads: Mapping[str, Any]) -> str:
    for payload in payloads:
        for key in ("updated_at", "generated_at", "completed_at", "created_at"):
            text = _string(payload.get(key))
            if text:
                return text
    return DETERMINISTIC_DEFAULT_GENERATED_AT


def _source_artifacts(*, pipeline_dir: Path, eval_dir: Path, robot_eval_dir: Path) -> Dict[str, Any]:
    paths = {
        "capture_descriptor": pipeline_dir.parent / "capture_descriptor.json",
        "raw_manifest": pipeline_dir.parent / "raw" / "manifest.json",
        "object_geometry_manifest": eval_dir / "object_geometry_manifest.json",
        "task_anchor_manifest": eval_dir / "task_anchor_manifest.json",
        "site_world_spec": eval_dir / "site_world_spec.json",
        "hosted_session_runtime_manifest": eval_dir / "hosted_session_runtime_manifest.json",
        "simready_scene_manifest": pipeline_dir / "simready" / "simready_scene_manifest.json",
        "simready_validation": pipeline_dir / "simready" / "simready_validation.json",
        "marble_simready_bridge": pipeline_dir / "marble_sim_assets" / "marble_simready_bridge.json",
        "marble_asset_validation": pipeline_dir / "marble_sim_assets" / "marble_asset_validation.json",
        "scene_asset_inspection": pipeline_dir
        / "simulation_automation"
        / "scene_asset_inspection.json",
        "scene_frame_estimate": pipeline_dir / "simulation_automation" / "scene_frame_estimate.json",
        "cpu_preflight_scorecard": pipeline_dir
        / "simulation_automation"
        / "cpu_preflight_scorecard.json",
        "episode_spec_manifest": pipeline_dir / "simulation_automation" / "episode_spec_manifest.json",
        "cpu_simulator_preflight_manifest": pipeline_dir
        / "simulation_automation"
        / "cpu_simulator_preflight_manifest.json",
        "worldlabs_world_manifest": pipeline_dir / "worldlabs_world_manifest.json",
        "cosmos3_readiness": (
            pipeline_dir
            / "cosmos3_readiness"
            / "cosmos3_capture_grounded_readiness.json"
        ),
        "protected_regions_manifest": eval_dir / "protected_regions_manifest.json",
        "rights_and_compliance_summary": pipeline_dir / "rights_and_compliance_summary.json",
        "rights_provenance_review": pipeline_dir / "rights_provenance_review.json",
        "privacy_processing_manifest": pipeline_dir / "privacy_processing_manifest.json",
        "robot_pov_input_manifest": (
            pipeline_dir / "robot_eval_inputs" / "robot_pov_evidence_manifest.json"
        ),
        "human_demo_input_manifest": (
            pipeline_dir / "robot_eval_inputs" / "human_demo_evidence_manifest.json"
        ),
        "action_log_input_manifest": (
            pipeline_dir / "robot_eval_inputs" / "action_log_manifest.json"
        ),
        "recorded_action_trace_manifest": (
            pipeline_dir / "robot_eval_inputs" / "recorded_action_trace_manifest.json"
        ),
        "actual_outcome_input_manifest": (
            pipeline_dir / "robot_eval_inputs" / "actual_outcome_manifest.json"
        ),
        "robot_team_test_submission_manifest": (
            pipeline_dir / "robot_eval_inputs" / "robot_team_test_submission_manifest.json"
        ),
    }
    return {
        key: _relative_if_file(robot_eval_dir, path)
        for key, path in sorted(paths.items())
        if _relative_if_file(robot_eval_dir, path)
    }


def _site_id_from_inputs(descriptor: Mapping[str, Any], raw_manifest: Mapping[str, Any]) -> str | None:
    candidates: List[Any] = [descriptor.get("site_id"), raw_manifest.get("site_id")]
    for payload in (descriptor, raw_manifest, descriptor.get("metadata"), raw_manifest.get("metadata")):
        if isinstance(payload, Mapping):
            identity = payload.get("site_identity")
            if isinstance(identity, Mapping):
                candidates.append(identity.get("site_id"))
    for candidate in candidates:
        text = _string(candidate)
        if text:
            return text
    return None


def _scene_frame_bounds(pipeline_dir: Path) -> tuple[List[float], List[float]] | None:
    frame_estimate = _read_optional_mapping(
        pipeline_dir / "simulation_automation" / "scene_frame_estimate.json"
    )
    frame = _mapping(frame_estimate.get("frame"))
    bounds = _mapping(frame.get("bounds"))
    lower = _pose_triplet_or_none(bounds.get("min"))
    upper = _pose_triplet_or_none(bounds.get("max"))
    if lower is None or upper is None:
        return None
    if any(upper[index] <= lower[index] for index in range(2)):
        return None
    return lower, upper


def _scene_frame_anchor_pair(
    pipeline_dir: Path,
    *,
    index: int,
) -> tuple[List[float] | None, List[float] | None]:
    bounds = _scene_frame_bounds(pipeline_dir)
    if bounds is None:
        return None, None
    lower, upper = bounds
    span_x = max(upper[0] - lower[0], 1.0)
    span_y = max(upper[1] - lower[1], 1.0)
    margin_x = max(span_x * 0.2, 0.5)
    margin_y = max(span_y * 0.2, 0.5)
    low_x = lower[0] + margin_x
    high_x = upper[0] - margin_x
    low_y = lower[1] + margin_y
    high_y = upper[1] - margin_y
    if index % 2:
        start = [high_x, high_y, 0.793]
        goal = [low_x, low_y, 0.793]
    else:
        start = [low_x, high_y, 0.793]
        goal = [high_x, low_y, 0.793]
    return [round(value, 6) for value in start], [round(value, 6) for value in goal]


def _capture_navigation_semantic_objects(
    pipeline_dir: Path,
    *,
    center: Sequence[float] | None,
    lower: Sequence[float] | None,
    upper: Sequence[float] | None,
    collision_hulls: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    raw_manifest = _read_optional_mapping(pipeline_dir.parent / "raw" / "manifest.json")
    descriptor = _read_optional_mapping(pipeline_dir.parent / "capture_descriptor.json")
    task_steps = _string_list(
        raw_manifest.get("taskSteps")
        or raw_manifest.get("task_steps")
        or descriptor.get("taskSteps")
        or descriptor.get("task_steps")
    )
    workflow_name = _string(
        raw_manifest.get("workflowName")
        or raw_manifest.get("workflow_name")
        or descriptor.get("workflowName")
        or descriptor.get("workflow_name")
    )
    zone = _string(raw_manifest.get("zone") or descriptor.get("zone"))
    haystack = " ".join([workflow_name, zone, *task_steps]).lower()
    if not any(token in haystack for token in ("navigate", "waypoint", "spawn", "route")):
        return []

    workspace_center = _pose_triplet_or_none(center) or [0.0, 0.0, 0.793]
    waypoint_center = list(workspace_center)
    if lower is not None and upper is not None:
        span_x = max(float(upper[0]) - float(lower[0]), 1.0)
        span_y = max(float(upper[1]) - float(lower[1]), 1.0)
        waypoint_center = [
            round(float(upper[0]) - max(span_x * 0.2, 0.5), 6),
            round(float(lower[1]) + max(span_y * 0.2, 0.5), 6),
            0.793,
        ]

    support_surface = {
        "source": "pipeline/simulation_automation/scene_frame_estimate.json",
        "status": "floor_region_estimated_from_capture_scene_bounds",
        "review_required": True,
    }
    placement_bbox = [list(lower), list(upper)] if lower is not None and upper is not None else None
    provenance = {
        "source_artifact": "raw/manifest.json",
        "source": "capture_task_steps_navigation_intent",
        "workflow_name": workflow_name or None,
        "zone": zone or None,
        "task_steps": task_steps,
        "review_required": True,
    }
    return [
        {
            "object_id": "navigation_workspace",
            "label": zone or "captured navigation workspace",
            "class_name": "navigation_zone",
            "task_role": "site_navigation_context",
            "semantic_roles": ["navigable_workspace", "site_context"],
            "center_xyz": workspace_center,
            "placement_bbox": placement_bbox,
            "collision_hulls": [dict(item) for item in collision_hulls],
            "support_surfaces": [support_surface],
            "provenance": provenance,
        },
        {
            "object_id": "selected_waypoint",
            "label": "selected waypoint",
            "class_name": "navigation_waypoint",
            "task_role": "navigation_target",
            "semantic_roles": ["goal_zone", "navigation_target"],
            "center_xyz": waypoint_center,
            "placement_bbox": None,
            "collision_hulls": [],
            "support_surfaces": [support_surface],
            "provenance": provenance,
        },
    ]


def _object_geometry_from_scene_assets(pipeline_dir: Path) -> Dict[str, Any]:
    inspection = _read_optional_mapping(
        pipeline_dir / "simulation_automation" / "scene_asset_inspection.json"
    )
    assets = inspection.get("assets")
    if not isinstance(assets, list):
        return {}
    objects: List[Dict[str, Any]] = []
    seen: set[str] = set()
    scene_center: List[float] | None = None
    scene_lower: List[float] | None = None
    scene_upper: List[float] | None = None
    scene_collision_hulls: List[Mapping[str, Any]] = []
    for asset_index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            continue
        hints = asset.get("semantic_hints")
        hint_labels: List[str] = []
        if isinstance(hints, list):
            for hint in hints:
                if isinstance(hint, Mapping):
                    label = _string(hint.get("label"))
                else:
                    label = _string(hint)
                if label:
                    hint_labels.append(label)
        if not hint_labels:
            hint_labels = [f"scene_asset_{asset_index}"]
        center = _pose_triplet_or_none(asset.get("centroid"))
        bounds = _mapping(asset.get("bounds"))
        lower = _pose_triplet_or_none(bounds.get("min"))
        upper = _pose_triplet_or_none(bounds.get("max"))
        if center is None and lower is not None and upper is not None:
            center = [round((lower[index] + upper[index]) / 2.0, 6) for index in range(3)]
        collision_evidence = _mapping(asset.get("collision_evidence"))
        has_collision_hint = bool(
            collision_evidence.get("real_collider_proven")
            or collision_evidence.get("proxy_estimated")
            or collision_evidence.get("portable_collider_glb_present")
        )
        collision_hulls = (
            [
                {
                    "source": "simulation_automation/scene_asset_inspection.json",
                    "status": _string(collision_evidence.get("status"))
                    or "scene_asset_collision_hint_present",
                    "review_required": True,
                }
            ]
            if has_collision_hint
            else []
        )
        if scene_center is None and center is not None:
            scene_center = center
        if scene_lower is None and lower is not None:
            scene_lower = lower
        if scene_upper is None and upper is not None:
            scene_upper = upper
        if not scene_collision_hulls and collision_hulls:
            scene_collision_hulls = list(collision_hulls)
        for label in hint_labels:
            object_id = _stable_slug(label, fallback=f"scene_asset_{asset_index}")
            if object_id in seen:
                continue
            seen.add(object_id)
            objects.append(
                {
                    "object_id": object_id,
                    "label": label,
                    "class_name": "scene_geometry",
                    "task_role": "site_context" if object_id == "world" else "navigation_target",
                    "semantic_roles": ["scene_anchor", "navigation_context"],
                    "center_xyz": center,
                    "placement_bbox": [lower, upper] if lower is not None and upper is not None else None,
                    "collision_hulls": collision_hulls,
                    "support_surfaces": [],
                    "provenance": {
                        "source_artifact": "pipeline/simulation_automation/scene_asset_inspection.json",
                        "source": "scene_asset_semantic_hint",
                        "asset_path": _string(asset.get("path")),
                        "review_required": True,
                    },
                }
            )
    for item in _capture_navigation_semantic_objects(
        pipeline_dir,
        center=scene_center,
        lower=scene_lower,
        upper=scene_upper,
        collision_hulls=scene_collision_hulls,
    ):
        object_id = _string(item.get("object_id"))
        if not object_id or object_id in seen:
            continue
        seen.add(object_id)
        objects.append(item)
    if not objects:
        return {}
    return {
        "schema_version": "object_geometry_manifest.v1",
        "status": "scene_asset_semantic_hints_review_required",
        "objects": objects,
        "claim_boundary": (
            "scene asset semantic hints provide review-required task grounding; "
            "they do not prove full object segmentation or collision fidelity"
        ),
    }


def _objects_by_id(object_geometry_manifest: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_objects = object_geometry_manifest.get("objects")
    if not isinstance(raw_objects, list):
        return {}
    objects: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(raw_objects):
        if not isinstance(item, Mapping):
            continue
        object_id = _string(item.get("object_id") or item.get("id") or f"object_{index}")
        if not object_id:
            continue
        center = _object_center(item)
        objects[object_id] = {
            "object_id": object_id,
            "label": _string(item.get("label") or item.get("class_name") or "object"),
            "class_name": _string(item.get("class_name") or item.get("label") or "object"),
            "task_role": _string(item.get("task_role")),
            "semantic_roles": _string_list(
                item.get("semantic_roles")
                or item.get("roles")
                or ([item.get("task_role")] if item.get("task_role") else [])
            ),
            "center_xyz": center,
            "has_collision_hulls": bool(item.get("collision_hulls")),
            "has_support_surfaces": bool(item.get("support_surfaces")),
            "physics_coverage_status": "covered"
            if item.get("collision_hulls") or item.get("support_surfaces")
            else "review_required",
            "provenance": _mapping(item.get("provenance")),
        }
    return objects


def _object_center(item: Mapping[str, Any]) -> List[float] | None:
    for key in ("center_xyz", "center", "position", "pose", "xyz"):
        pose = _pose_triplet_or_none(item.get(key))
        if pose is not None:
            return pose
    bbox = item.get("placement_bbox") or item.get("boundingBox") or item.get("bbox") or item.get("bounds")
    if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)):
        values = list(bbox)
        if len(values) >= 2 and all(isinstance(value, Sequence) for value in values[:2]):
            lower = _pose_triplet_or_none(values[0])
            upper = _pose_triplet_or_none(values[1])
            if lower is not None and upper is not None:
                return [round((lower[index] + upper[index]) / 2.0, 6) for index in range(3)]
        if len(values) >= 6:
            try:
                nums = [float(value) for value in values[:6]]
            except (TypeError, ValueError):
                return None
            if all(math.isfinite(value) for value in nums):
                return [
                    round((nums[0] + nums[3]) / 2.0, 6),
                    round((nums[1] + nums[4]) / 2.0, 6),
                    round((nums[2] + nums[5]) / 2.0, 6),
                ]
    return None


def _object_index(object_geometry_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    objects = list(_objects_by_id(object_geometry_manifest).values())
    physics_covered = [
        item
        for item in objects
        if item.get("has_collision_hulls") or item.get("has_support_surfaces")
    ]
    return {
        "schema_version": "robot_eval_object_index.v1",
        "status": "available" if objects else "missing",
        "object_count": len(objects),
        "physics_covered_object_count": len(physics_covered),
        "physics_coverage_complete": bool(objects) and len(physics_covered) == len(objects),
        "objects": sorted(objects, key=lambda item: item["object_id"]),
        "missing_physics_object_ids": [
            item["object_id"] for item in objects if item not in physics_covered
        ],
        "claim_boundary": "object_index_is_capture_derived_semantic_context_not_physical_scene_certification",
    }


def _zone_candidates_for_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    task_id = _stable_slug(task.get("task_id"), fallback="task")
    start_pose = _pose_triplet_or_none(task.get("start_zone"))
    goal_pose = _pose_triplet_or_none(task.get("goal_zone"))
    source = _string(task.get("source_artifact")) or "pipeline/evaluation_prep/task_anchor_manifest.json"
    spawn_candidate = _pose_card(
        zone_id=f"start_zone_{task_id}",
        role="robot_spawn",
        pose=start_pose,
        label=f"start zone for {_string(task.get('task_id')) or 'task'}",
        source=source,
        confidence="capture_grounded_task_anchor",
    )
    target_candidate = _pose_card(
        zone_id=f"goal_zone_{task_id}",
        role="task_goal",
        pose=goal_pose,
        label=f"goal zone for {_string(task.get('task_id')) or 'task'}",
        source=source,
        confidence="capture_grounded_task_anchor",
    )
    pair_valid = bool(spawn_candidate["validated"] and target_candidate["validated"])
    return {
        "spawn_candidates": [spawn_candidate],
        "target_candidates": [target_candidate],
        "validated_spawn_candidate_count": 1 if spawn_candidate["validated"] else 0,
        "validated_target_candidate_count": 1 if target_candidate["validated"] else 0,
        "validated_spawn_target_pair": pair_valid,
        "validation_status": "validated_site_zone_pair" if pair_valid else "blocked_missing_site_zone_pair",
        "claim_boundary": "validated finite site-zone pair is an eval start-goal input, not autonomous navigation proof",
    }


def _infer_site_type(
    *,
    metadata: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    object_geometry_manifest: Mapping[str, Any],
) -> str:
    explicit = _first_text(
        metadata.get("site_type"),
        metadata.get("target_site_type"),
        metadata.get("facility_type"),
    )
    if explicit:
        return explicit
    objects = _objects_by_id(object_geometry_manifest)
    haystack = " ".join(
        [
            *[
                " ".join(
                    [
                        _string(task.get("task_id")),
                        _string(task.get("task_text")),
                        _string(task.get("task_category")),
                    ]
                )
                for task in tasks
            ],
            *[
                " ".join(
                    [
                        _string(obj.get("label")),
                        _string(obj.get("class_name")),
                        _string(obj.get("task_role")),
                    ]
                )
                for obj in objects.values()
            ],
        ]
    ).lower()
    if any(token in haystack for token in ("patient", "hospital", "hallway", "nurse")):
        return "hospital hallway"
    if any(token in haystack for token in ("dock", "pallet", "forklift", "loading")):
        return "loading dock"
    if any(token in haystack for token in ("conveyor", "line", "assembly", "station")):
        return "factory line-side station"
    if any(token in haystack for token in ("shelf", "bin", "tote", "cart", "return", "stock")):
        return "stockroom"
    if any(token in haystack for token in ("aisle", "rack", "warehouse")):
        return "warehouse aisle"
    if any(token in haystack for token in ("waypoint", "humanoid", "navigation_workspace")):
        return "indoor navigation route"
    if objects:
        return "captured indoor scene"
    return "unknown_site_type"


def _success_criteria(task_category: str) -> List[str]:
    category = task_category.strip().lower()
    base = [
        "task attempt is linked to one real-site capture package",
        "robot POV, action logs, and outcome labels reference the same task_id",
        "prediction record and actual outcome record use the same scenario_id",
    ]
    if category in {"open_close", "manipulation", "pick_place"}:
        return [
            *base,
            "target object or fixture reaches the requested final state",
            "contact/collision events stay within the declared safety threshold",
            "intervention count and cycle time are recorded",
        ]
    if category in {"navigation", "route", "inspection"}:
        return [
            *base,
            "robot reaches the goal zone without a safety threshold violation",
            "route obstruction, localization, and intervention events are labeled",
            "cycle time is recorded against the buyer threshold",
        ]
    return [
        *base,
        "task completion is labeled with success, failure, or review_required",
        "failure_mode_ids reference failure_taxonomy.json",
    ]


def _task_ontology_v1(*, generated_at: str) -> Dict[str, Any]:
    tasks: List[Dict[str, Any]] = []
    for item in TASK_ONTOLOGY_DEFINITIONS:
        task_id = _string(item.get("task_id"))
        task_family = _string(item.get("task_family"))
        tasks.append(
            {
                "task_id": task_id,
                "task_family": task_family,
                "aliases": _string_list(item.get("aliases")),
                "parameters": _string_list(item.get("parameters")),
                "success_criteria": _string_list(item.get("success_criteria")),
                "required_evidence": [
                    "capture_backed_site_card",
                    "task_card",
                    "scenario_card",
                    "robot_pov_or_recorded_trace",
                    "action_log_or_teleop_demo",
                    "actual_outcome_manifest",
                    "rights_packet",
                ],
                "supported_metrics": [
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
                ],
                "cross_site_query_fields": [
                    "task_id",
                    "task_family",
                    "site_type",
                    "robot_embodiment",
                    "object_class",
                    "fixture_type",
                    "route_constraint",
                    "safety_constraint",
                    "variation_id",
                    "scenario_status",
                ],
                "claim_boundary": "ontology_task_definition_only_no_execution_or_readiness_claim",
            }
        )
    return {
        "schema_version": TASK_ONTOLOGY_SCHEMA_VERSION,
        "ontology_version": "1.0",
        "generated_at": generated_at,
        "task_count": len(tasks),
        "tasks": sorted(tasks, key=lambda task: task["task_id"]),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _ontology_by_id() -> Dict[str, Dict[str, Any]]:
    return {item["task_id"]: dict(item) for item in TASK_ONTOLOGY_DEFINITIONS}


def _canonical_task_id_for_task(*, task_id: str, task_text: str, task_category: str) -> str:
    haystack = " ".join([task_id, task_text, task_category]).lower()
    ontology = _ontology_by_id()
    for candidate_id, entry in ontology.items():
        aliases = [candidate_id, *_string_list(entry.get("aliases"))]
        if any(alias.replace("_", " ") in haystack for alias in aliases):
            return candidate_id
    if "conveyor" in haystack and "cart" in haystack:
        return "cart_to_conveyor_transfer"
    if "line" in haystack and "deliver" in haystack:
        return "line_side_delivery"
    if "door" in haystack:
        return "open_door_enter_room"
    if "human" in haystack and ("cross" in haystack or "yield" in haystack):
        return "human_crossing_safety_response"
    if "blocked" in haystack or "obstacle" in haystack:
        return "blocked_path_recovery"
    if "inspect" in haystack or "shelf" in haystack:
        return "inspect_shelf"
    if "tote" in haystack or "cart" in haystack:
        return "move_tote"
    if "place" in haystack or "bin" in haystack:
        return "place_object_into_bin"
    if "pick" in haystack or "grasp" in haystack:
        return "pick_known_object"
    if "navigate" in haystack or "route" in haystack or "station" in haystack:
        return "navigate_to_station"
    return "navigate_to_station" if task_category in {"navigation", "route"} else "pick_known_object"


def _task_library(
    *,
    task_anchor_manifest: Mapping[str, Any],
    object_geometry_manifest: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    tasks_raw = task_anchor_manifest.get("tasks")
    objects = _objects_by_id(object_geometry_manifest)
    tasks: List[Dict[str, Any]] = []
    if isinstance(tasks_raw, list):
        for index, task in enumerate(tasks_raw):
            if not isinstance(task, Mapping):
                continue
            task_id = _string(task.get("task_id") or task.get("id") or f"task_{index}")
            if not task_id:
                continue
            target_ids = _string_list(task.get("target_object_ids"))
            task_category = _string(task.get("task_category") or "generic")
            task_text = _string(task.get("task_text") or task.get("name") or task_id)
            ontology_task_id = _canonical_task_id_for_task(
                task_id=task_id,
                task_text=task_text,
                task_category=task_category,
            )
            ontology_entry = _ontology_by_id().get(ontology_task_id, {})
            start_zone = _pose_triplet_or_none(task.get("start_zone"))
            goal_zone = _pose_triplet_or_none(task.get("goal_zone"))
            target_objects = [
                objects[object_id] for object_id in target_ids if object_id in objects
            ]
            zone_candidates = _zone_candidates_for_task(
                {
                    **dict(task),
                    "task_id": task_id,
                    "start_zone": start_zone,
                    "goal_zone": goal_zone,
                    "source_artifact": _string(
                        task.get("source_artifact")
                        or task.get("sourceArtifact")
                        or "pipeline/evaluation_prep/task_anchor_manifest.json"
                    ),
                }
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "task_text": task_text,
                    "task_category": task_category,
                    "ontology_task_id": ontology_task_id,
                    "ontology_version": "1.0",
                    "task_family_aliases": _string_list(ontology_entry.get("aliases")),
                    "cross_site_query_fields": [
                        "site_type",
                        "robot_embodiment",
                        "target_object_ids",
                        "scenario_variation_id",
                        "rights_status",
                        "failure_mode_ids",
                    ],
                    "target_object_ids": target_ids,
                    "target_objects": target_objects,
                    "task_objects": target_objects,
                    "object_semantics_status": "object_grounded"
                    if target_objects
                    else "missing_target_object_semantics",
                    "articulation_required_ids": _string_list(
                        task.get("articulation_required_ids")
                    ),
                    "start_zone": start_zone,
                    "goal_zone": goal_zone,
                    "site_zone_candidates": zone_candidates,
                    "start_zone_id": zone_candidates["spawn_candidates"][0]["zone_id"],
                    "goal_zone_id": zone_candidates["target_candidates"][0]["zone_id"],
                    "semantic_zone_pair_status": zone_candidates["validation_status"],
                    "task_critical": bool(task.get("task_critical")),
                    "source_artifact": _string(
                        task.get("source_artifact")
                        or task.get("sourceArtifact")
                        or "pipeline/evaluation_prep/task_anchor_manifest.json"
                    ),
                    "success_criteria": _success_criteria(task_category),
                    "required_evidence": [
                        "robot_pov_evidence",
                        "human_demo_evidence",
                        "action_log_evidence",
                        "prediction_outcome_record",
                    ],
                    "claim_boundary": "task_definition_only_no_robot_execution_claim",
                }
            )
    tasks = sorted(tasks, key=lambda item: item["task_id"])
    return {
        "schema_version": ROBOT_TASK_LIBRARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "task_count": len(tasks),
        "tasks": tasks,
        "source_policy": "task_anchor_manifest_is_primary_when_present",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _task_anchor_from_simulation_automation(pipeline_dir: Path) -> Dict[str, Any]:
    """Build review-required task anchors from simulation-automation proposals.

    These proposals are capture-grounded review inputs. They are sufficient to
    define a simulator eval scope, but they do not prove task acceptance,
    simulator execution, robot policy performance, or generated-world rank fidelity.
    """

    proposal_manifest = _read_optional_mapping(
        pipeline_dir / "simulation_automation" / "task_anchor_proposal_manifest.json"
    )
    proposals = proposal_manifest.get("proposals")
    if not isinstance(proposals, list):
        return {}
    tasks: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, Mapping):
            continue
        task_id = _string(proposal.get("task_id") or proposal.get("id") or f"task_{index}")
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        start_zone = proposal.get("start_zone")
        goal_zone = proposal.get("goal_zone")
        if _pose_triplet_or_none(start_zone) is None or _pose_triplet_or_none(goal_zone) is None:
            fallback_start, fallback_goal = _scene_frame_anchor_pair(pipeline_dir, index=index)
            if _pose_triplet_or_none(start_zone) is None:
                start_zone = fallback_start
            if _pose_triplet_or_none(goal_zone) is None:
                goal_zone = fallback_goal
        tasks.append(
            {
                "task_id": task_id,
                "task_text": _string(proposal.get("task_text") or proposal.get("name") or task_id),
                "task_category": _string(proposal.get("task_category") or "navigation"),
                "target_object_ids": _string_list(proposal.get("target_object_ids")),
                "start_zone": start_zone,
                "goal_zone": goal_zone,
                "site_zone_source": "scene_frame_estimate_bounds_review_candidate"
                if start_zone is not None and goal_zone is not None
                else "task_anchor_proposal",
                "task_critical": False,
                "review_required": True,
                "accepted": proposal.get("accepted") is True,
                "source_artifact": "pipeline/simulation_automation/task_anchor_proposal_manifest.json",
                "proposal_id": _string(proposal.get("proposal_id")),
                "claim_boundary": "simulation_task_anchor_proposal_defines_review_scope_not_execution_proof",
            }
        )
    if not tasks:
        return {}
    return {
        "schema_version": "task_anchor_manifest.v1",
        "generated_at": _string(proposal_manifest.get("generated_at")),
        "status": "compiled_review_required",
        "source_artifact": "pipeline/simulation_automation/task_anchor_proposal_manifest.json",
        "tasks": tasks,
        "claim_boundary": "simulation_automation_task_proposals_do_not_prove_rank_fidelity",
    }


def _robot_profiles(
    *,
    site_world_spec: Mapping[str, Any],
    hosted_session_runtime_manifest: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    raw_profiles = site_world_spec.get("robot_profiles")
    if not isinstance(raw_profiles, list):
        raw_profiles = hosted_session_runtime_manifest.get("robot_profiles")
    profiles: List[Dict[str, Any]] = []
    if isinstance(raw_profiles, list):
        for index, item in enumerate(raw_profiles):
            if not isinstance(item, Mapping):
                continue
            profile_id = _string(item.get("id") or item.get("robot_profile_id") or f"robot_{index}")
            profiles.append(
                {
                    "robot_profile_id": profile_id,
                    "display_name": _string(item.get("display_name") or profile_id),
                    "embodiment_type": _string(item.get("embodiment_type") or "unknown"),
                    "action_space": _mapping(item.get("action_space")),
                    "source": "site_world_spec_or_hosted_session_manifest",
                    "claim_boundary": "robot_profile_requirement_only_not_robot_asset_ready",
                }
            )
    return sorted(profiles, key=lambda item: item["robot_profile_id"])


def _default_unitree_g1_robot_profile() -> Dict[str, Any]:
    return {
        "robot_profile_id": "unitree_g1",
        "display_name": "Unitree G1",
        "embodiment_type": "humanoid",
        "action_space": {},
        "source": "blueprint_default_robot_profile",
        "claim_boundary": "default_robot_profile_is_eval_scope_not_generated_world_rank_fidelity",
    }


def _available_prediction_sources(
    *,
    simready_scene_manifest: Mapping[str, Any],
    marble_simready_bridge: Mapping[str, Any],
    cosmos3_readiness: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    if simready_scene_manifest:
        sources.append(
            {
                "source": "simready_review",
                "status": _string(simready_scene_manifest.get("status") or "review_artifact_present"),
                "execution_proven": False,
            }
        )
    if marble_simready_bridge:
        sources.append(
            {
                "source": "marble_review",
                "status": _string(marble_simready_bridge.get("status") or "review_artifact_present"),
                "execution_proven": False,
            }
        )
    if cosmos3_readiness:
        sources.append(
            {
                "source": "cosmos_preflight",
                "status": _string(cosmos3_readiness.get("status") or "preflight_artifact_present"),
                "execution_proven": False,
            }
        )
    return sorted(sources, key=lambda item: item["source"])


def _scenario_library(
    *,
    task_library: Mapping[str, Any],
    robot_profiles: Sequence[Mapping[str, Any]],
    prediction_sources: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    scenarios: List[Dict[str, Any]] = []
    tasks = [task for task in task_library.get("tasks", []) if isinstance(task, Mapping)]
    profiles = list(robot_profiles) or [_default_unitree_g1_robot_profile()]
    for task in tasks:
        task_id = _string(task.get("task_id"))
        for profile in profiles:
            robot_profile_id = _string(profile.get("robot_profile_id"))
            scenario_id = f"scenario_{_stable_slug(task_id, fallback='task')}_{_stable_slug(robot_profile_id, fallback='robot')}"
            zone_candidates = _mapping(task.get("site_zone_candidates"))
            spawn_candidates = [
                dict(item)
                for item in zone_candidates.get("spawn_candidates", []) or []
                if isinstance(item, Mapping)
            ]
            target_candidates = [
                dict(item)
                for item in zone_candidates.get("target_candidates", []) or []
                if isinstance(item, Mapping)
            ]
            validated_spawn_target_pair = bool(
                zone_candidates.get("validated_spawn_target_pair")
            )
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_type": "real_site_robot_task_eval",
                    "task_id": task_id,
                    "robot_profile_id": robot_profile_id,
                    "start_state_id": f"start_{_stable_slug(task_id, fallback='task')}",
                    "target_object_ids": _string_list(task.get("target_object_ids")),
                    "target_objects": [
                        dict(item)
                        for item in task.get("target_objects", []) or []
                        if isinstance(item, Mapping)
                    ],
                    "start_zone": task.get("start_zone"),
                    "goal_zone": task.get("goal_zone"),
                    "start_zone_id": task.get("start_zone_id"),
                    "goal_zone_id": task.get("goal_zone_id"),
                    "spawn_candidates": spawn_candidates,
                    "target_candidates": target_candidates,
                    "validated_spawn_candidate_count": int(
                        zone_candidates.get("validated_spawn_candidate_count") or 0
                    ),
                    "validated_target_candidate_count": int(
                        zone_candidates.get("validated_target_candidate_count") or 0
                    ),
                    "validated_spawn_target_pair": validated_spawn_target_pair,
                    "semantic_spawn_target_source": "task_anchor_manifest_site_zones"
                    if validated_spawn_target_pair
                    else "missing_validated_task_anchor_site_zones",
                    "prediction_sources_available": [
                        _string(source.get("source")) for source in prediction_sources
                    ],
                    "required_actual_sources": ["robot_pilot", "teleop", "operator_report"],
                    "required_evidence": [
                        "robot_pov_evidence",
                        "human_demo_evidence",
                        "action_log_evidence",
                        "actual_outcome_record",
                    ],
                    "missing_evidence_statuses": [
                        "needs_robot_pov",
                        "needs_human_demo",
                        "needs_action_logs",
                        "needs_actual_outcome",
                    ],
                    "claim_boundary": "scenario_library_only_no_sim_or_robot_execution",
                }
            )
    scenarios = sorted(scenarios, key=lambda item: item["scenario_id"])
    return {
        "schema_version": SCENARIO_LIBRARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "prediction_sources_supported": list(PREDICTION_SOURCES),
        "actual_sources_supported": list(ACTUAL_SOURCES),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _scenario_family_library(
    *,
    scenario_library: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    allowed_statuses = [
        "capture-grounded",
        "representative-mock",
        "agent-inferred-needs-review",
        "accepted",
        "rejected",
        "review-only",
    ]
    families: List[Dict[str, Any]] = []
    scenarios = [
        dict(scenario)
        for scenario in scenario_library.get("scenarios", [])
        if isinstance(scenario, Mapping)
    ]
    for scenario in scenarios:
        scenario_id = _string(scenario.get("scenario_id"))
        task_id = _string(scenario.get("task_id"))
        family_id = f"family_{_stable_slug(scenario_id, fallback='scenario')}"
        variations = [
            {
                "variation_id": "capture_observed_layout",
                "label": "Capture-observed layout",
                "scenario_status": "capture-grounded",
                "evidence_source": "capture_package_and_task_anchor",
                "requires_review": False,
                "sim_or_cosmos_proof_claim_allowed": False,
                "claim_boundary": "capture_layout_is_context_not_robot_outcome",
            }
        ]
        for definition in SCENARIO_VARIATION_DEFINITIONS:
            status = _string(definition.get("default_status")) or "review-only"
            variations.append(
                {
                    "variation_id": _string(definition.get("variation_id")),
                    "label": _string(definition.get("label")),
                    "scenario_status": status,
                    "evidence_source": "scenario_family_generator_template",
                    "requires_review": status not in {"capture-grounded", "accepted"},
                    "sim_or_cosmos_proof_claim_allowed": False,
                    "claim_boundary": "variation_is_mock_or_review_input_until_owner_system_proof_exists",
                }
            )
        families.append(
            {
                "family_id": family_id,
                "scenario_id": scenario_id,
                "task_id": task_id,
                "robot_profile_id": _string(scenario.get("robot_profile_id")),
                "status": "review_required",
                "variation_count": len(variations),
                "variations": variations,
                "allowed_statuses": allowed_statuses,
                "review_loop": {
                    "review_queue_status": "required_for_generated_or_mock_variations",
                    "accepted_status_requires": [
                        "operator_or_buyer_review_ref",
                        "evidence_uri",
                        "reviewer",
                        "reviewed_at",
                    ],
                    "rejected_status_records": [
                        "reason",
                        "reviewer",
                        "reviewed_at",
                    ],
                },
                "claim_boundary": "scenario_family_generation_does_not_prove_simulation_or_robot_outcomes",
            }
        )
    return {
        "schema_version": SCENARIO_FAMILY_LIBRARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "family_count": len(families),
        "variation_names_required": [
            definition["variation_id"] for definition in SCENARIO_VARIATION_DEFINITIONS
        ],
        "families": sorted(families, key=lambda item: item["family_id"]),
        "cosmos_or_simulator_proof_claim_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _robot_pov_requirements(*, generated_at: str) -> Dict[str, Any]:
    return {
        "schema_version": "robot_pov_evidence_requirements.v1",
        "generated_at": generated_at,
        "status": "required",
        "required_fields": [
            "scenario_id",
            "task_id",
            "robot_profile_id",
            "attempt_id",
            "robot_pov_video_uri",
            "timestamp_alignment",
            "robot_pose_or_odometry_uri",
            "action_log_uri",
            "operator_intervention_log_uri",
            "safety_event_log_uri",
            "rights_privacy_scope",
            "camera_intrinsics",
            "camera_extrinsics_or_mount_pose",
            "camera_calibration_status",
        ],
        "camera_metadata_contract": {
            "intrinsics": "fx, fy, cx, cy (+ distortion model when available)",
            "extrinsics": "camera-to-robot-base transform or fixed mount pose",
            "calibration_status": "one_of: verified, factory_default, uncalibrated",
            "uncalibrated_footage_downgrade": (
                "POV evidence without verified calibration supports review-grade "
                "labels only, never metric geometry claims"
            ),
        },
        "accepted_media": [
            "onboard_rgb_video",
            "onboard_depth_or_range_when_available",
            "third_person_reference_video_optional",
            "timestamped_stills_optional",
        ],
        "minimum_labels": [
            "attempt_start",
            "attempt_end",
            "success_or_failure",
            "intervention_count",
            "contact_or_collision_events",
            "safety_threshold_events",
        ],
        "claim_boundary": "robot_pov_required_before_robot_or_deployment_claim",
    }


def _human_demo_requirements(*, generated_at: str) -> Dict[str, Any]:
    return {
        "schema_version": "human_demo_evidence_requirements.v1",
        "generated_at": generated_at,
        "status": "required_for_demo_backed_eval",
        "required_fields": [
            "scenario_id",
            "task_id",
            "demo_id",
            "human_demo_video_uri",
            "demo_step_annotations_uri",
            "success_label",
            "cycle_time_seconds",
            "safety_or_access_notes",
            "rights_privacy_scope",
        ],
        "accepted_media": [
            "first_person_human_demo_video",
            "third_person_human_demo_video",
            "teleop_demo_video",
            "timestamped_annotation_file",
        ],
        "minimum_labels": [
            "demo_start",
            "demo_end",
            "contact_points",
            "handoff_zones",
            "failed_or_ambiguous_steps",
        ],
        "claim_boundary": "human_demo_is_support_evidence_not_robot_trial",
    }


def _robot_eval_inputs_evidence_contract(*, generated_at: str) -> Dict[str, Any]:
    shared_required = [
        "schema_version",
        "scenario_id",
        "task_id",
        "timestamp_alignment",
        "owner_system",
        "provenance",
        "rights_privacy_scope",
    ]
    contracts = {
        "robot_pov": {
            "path": "pipeline/robot_eval_inputs/robot_pov_evidence_manifest.json",
            "required_fields": [
                *shared_required,
                "attempt_id",
                "robot_profile_id",
                "robot_pov_video_uri",
                "robot_pose_or_odometry_uri",
            ],
        },
        "human_demo": {
            "path": "pipeline/robot_eval_inputs/human_demo_evidence_manifest.json",
            "required_fields": [
                *shared_required,
                "demo_id",
                "human_demo_video_uri",
                "demo_step_annotations_uri",
                "success_label",
            ],
        },
        "action_logs": {
            "path": "pipeline/robot_eval_inputs/action_log_manifest.json",
            "required_fields": [
                *shared_required,
                "attempt_id",
                "action_log_uri",
                "observation_action_alignment",
                "policy_or_operator_ref",
            ],
        },
        "recorded_action_traces": {
            "path": "pipeline/robot_eval_inputs/recorded_action_trace_manifest.json",
            "required_fields": [
                "schema_version",
                "attempts",
                "owner_system",
                "provenance",
                "rights_privacy_scope",
            ],
            "attempt_required_fields": [
                "attempt_id",
                "scenario_id",
                "task_id",
                "success",
                "cycle_time_seconds",
                "intervention_count",
                "failure_mode_ids",
            ],
        },
        "simulator_traces": {
            "path": "pipeline/robot_eval_inputs/simulator_trace_manifest.json",
            "required_fields": [
                *shared_required,
                "attempt_id",
                "simulator_framework",
                "scenario_attempt_trace_uri",
                "contact_trace_uri",
                "timing_metrics_uri",
                "safety_event_log_uri",
            ],
        },
        "policy_submissions": {
            "path": "pipeline/robot_eval_inputs/robot_team_test_submission_manifest.json",
            "required_fields": [
                "schema_version",
                "modalities",
                "owner_system",
                "provenance",
                "rights_privacy_scope",
            ],
        },
        "actual_outcomes": {
            "path": "pipeline/robot_eval_inputs/actual_outcome_manifest.json",
            "required_fields": [
                *shared_required,
                "attempt_id",
                "actual_source",
                "actual_success",
                "cycle_time_seconds",
                "intervention_count",
                "failure_mode_ids",
            ],
        },
    }
    return {
        "schema_version": ROBOT_EVAL_INPUTS_EVIDENCE_CONTRACT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "contract_only",
        "contracts": contracts,
        "required_cross_cutting_fields": {
            "rights_privacy_scope": [
                "allowed_use",
                "privacy_processed",
                "external_display_allowed",
                "operator_or_owner_approval_ref",
            ],
            "timestamp_alignment": [
                "clock_source",
                "alignment_method",
                "start_timestamp",
                "end_timestamp",
                "max_offset_ms",
            ],
            "owner_system": [
                "system_id",
                "system_type",
                "contact_or_team_ref",
                "evidence_authority",
            ],
            "provenance": [
                "created_at",
                "source_artifact_uri",
                "checksum",
                "chain_of_custody",
            ],
        },
        "claim_boundary": "input_contract_only_missing_inputs_remain_blocked_until_owner_system_evidence_exists",
    }


def _robot_team_submission_modality_statuses(
    robot_team_submission_input: Mapping[str, Any],
) -> List[str]:
    if not robot_team_submission_input:
        return [
            _string(item.get("missing_evidence_status"))
            for item in ROBOT_TEAM_TEST_MODALITY_REQUIREMENTS
        ]
    raw_modalities = robot_team_submission_input.get("modalities")
    if isinstance(raw_modalities, Mapping):
        present = {
            _string(key)
            for key, value in raw_modalities.items()
            if isinstance(value, Mapping) and bool(value.get("selected") or value.get("enabled") or value.get("fields"))
        }
    elif isinstance(raw_modalities, list):
        present = {
            _string(item.get("modality") or item.get("id"))
            for item in raw_modalities
            if isinstance(item, Mapping) and bool(item.get("selected") or item.get("enabled") or item.get("fields"))
        }
    else:
        present = set()
    return [
        _string(item.get("missing_evidence_status"))
        for item in ROBOT_TEAM_TEST_MODALITY_REQUIREMENTS
        if _string(item.get("modality_id")) not in present
    ]


def _robot_team_submission_modalities(
    *,
    robot_team_submission_input: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    missing_statuses = _robot_team_submission_modality_statuses(robot_team_submission_input)
    return {
        "schema_version": ROBOT_TEAM_TEST_SUBMISSION_MODALITIES_SCHEMA_VERSION,
        "dataset_version": ROBOT_EVAL_DATASET_VERSION,
        "generated_at": generated_at,
        "source_input_present": bool(robot_team_submission_input),
        "modality_count": len(ROBOT_TEAM_TEST_MODALITY_REQUIREMENTS),
        "modalities": [
            {
                **item,
                "accepted_reference_policy": "artifact_references_first_no_raw_upload_required",
                "review_status": (
                    "missing_evidence"
                    if _string(item.get("missing_evidence_status")) in missing_statuses
                    else "reference_present_requires_owner_system_review"
                ),
                "claim_boundary": "submission_reference_only_no_policy_execution_or_rank_fidelity_claim",
            }
            for item in ROBOT_TEAM_TEST_MODALITY_REQUIREMENTS
        ],
        "missing_evidence_statuses": missing_statuses,
        "blocked_claim_upgrades": [
            "ready_to_deploy_claim",
            "non_ranking_operational_claim_validated_claim",
            "simulator_completed_claim",
            "robot_trial_passed_claim",
            "policy_execution_passed_claim",
            "guaranteed_threshold_claim",
        ],
        "webapp_policy_field": "policy.robotTeamTestSubmission",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _failure_taxonomy(*, generated_at: str) -> Dict[str, Any]:
    modes = [
        ("failure_task_not_attempted", "No valid task attempt exists for the scenario."),
        ("failure_navigation_blocked", "Robot cannot reach the required zone or route."),
        ("failure_localization_or_pose_drift", "Pose, localization, or map alignment drifted."),
        ("failure_manipulation_miss", "End effector or gripper failed to complete the action."),
        ("failure_contact_collision", "Unexpected contact or collision occurred."),
        ("failure_articulation_blocked", "Door, drawer, fixture, or tool articulation failed."),
        ("failure_cycle_time_exceeded", "Cycle time exceeded the buyer threshold."),
        ("failure_intervention_required", "Human or operator intervention was required."),
        ("failure_safety_threshold_violation", "Declared safety threshold was violated."),
        ("failure_perception_occlusion", "Occlusion or perception uncertainty blocked execution."),
        ("failure_rights_privacy_blocked", "Rights/privacy state blocks use or display."),
        ("failure_evidence_missing", "Required evidence is missing or not linked."),
    ]
    return {
        "schema_version": "robot_eval_failure_taxonomy.v1",
        "generated_at": generated_at,
        "failure_modes": [
            {
                "failure_mode_id": mode_id,
                "label": label,
                "severity_default": "review_required",
                "claim_boundary": "failure_label_only_requires_attempt_evidence_for_outcome",
            }
            for mode_id, label in modes
        ],
    }


def _prediction_outcome_ledger(
    *,
    scenario_library: Mapping[str, Any],
    prediction_sources: Sequence[Mapping[str, Any]],
    source_artifacts: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    scenarios = [
        scenario
        for scenario in scenario_library.get("scenarios", [])
        if isinstance(scenario, Mapping)
    ]
    for scenario in scenarios:
        scenario_id = _string(scenario.get("scenario_id"))
        task_id = _string(scenario.get("task_id"))
        for source in prediction_sources:
            source_id = _string(source.get("source"))
            if not source_id:
                continue
            record_id = f"pred_{_stable_slug(scenario_id, fallback='scenario')}_{source_id}"
            records.append(
                {
                    "record_id": record_id,
                    "scenario_id": scenario_id,
                    "task_id": task_id,
                    "prediction_source": source_id,
                    "prediction_status": "advisory_review_only",
                    "predicted_success": None,
                    "confidence": None,
                    "actual_source": None,
                    "actual_status": "needs_actual_outcome",
                    "actual_success": None,
                    "metrics": {
                        "task_completion": None,
                        "cycle_time_seconds": None,
                        "intervention_count": None,
                        "contact_collision_event_count": None,
                        "safety_violation_count": None,
                    },
                    "failure_mode_ids": ["failure_evidence_missing"],
                    "evidence_artifact_paths": dict(source_artifacts),
                    "owner_system": "BlueprintCapturePipeline",
                    "claim_boundary": "prediction_only_no_actual_outcome_no_deployment_claim",
                }
            )
    records = sorted(records, key=lambda item: item["record_id"])
    return {
        "schema_version": PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "ledger_status": "needs_actual_outcome" if records else "needs_prediction_sources",
        "prediction_sources_supported": list(PREDICTION_SOURCES),
        "actual_sources_supported": list(ACTUAL_SOURCES),
        "record_count": len(records),
        "records": records,
        "required_metric_fields": [
            "task_completion",
            "cycle_time_seconds",
            "intervention_count",
            "contact_collision_event_count",
            "safety_violation_count",
            "failure_mode_ids",
            "confidence",
            "proof_artifact_paths",
            "owner_system",
            "claim_boundary",
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _scoring_methodology(*, generated_at: str) -> Dict[str, Any]:
    return {
        "schema_version": SCORING_METHODOLOGY_SCHEMA_VERSION,
        "methodology_version": "robot_eval_scoring.v1",
        "generated_at": generated_at,
        "status": "versioned_advisory_methodology",
        "metrics": list(SCORING_METRIC_DEFINITIONS),
        "failure_taxonomy_source": "failure_taxonomy.json",
        "deterministic_scorer": {
            "runner": "recorded_action_trace_fixture",
            "input": "pipeline/robot_eval_inputs/recorded_action_trace_manifest.json",
            "output": "pipeline/robot_eval_dataset/recorded_trace_eval_report.json",
            "live_simulator_required": False,
            "docker_required": False,
            "policy_api_required": False,
            "provider_credentials_required": False,
        },
        "proof_boundary": {
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "non_ranking_operational_claim_validated": False,
            "public_claim_upgrade_allowed": False,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _threshold_template_for_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    task_category = _string(task.get("task_category")).lower()
    ontology_task_id = _string(task.get("ontology_task_id")).lower()
    if task_category in DEFAULT_TASK_THRESHOLD_TEMPLATES:
        return dict(DEFAULT_TASK_THRESHOLD_TEMPLATES[task_category])
    if any(token in ontology_task_id for token in ("pick", "place", "bin")):
        return dict(DEFAULT_TASK_THRESHOLD_TEMPLATES["pick_place"])
    if any(token in ontology_task_id for token in ("navigate", "delivery", "move")):
        return dict(DEFAULT_TASK_THRESHOLD_TEMPLATES["navigation"])
    return dict(DEFAULT_TASK_THRESHOLD_TEMPLATES["general"])


def _task_thresholds(
    *,
    task_library: Mapping[str, Any],
    scoring_methodology: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    tasks = [
        item
        for item in task_library.get("tasks", []) or []
        if isinstance(item, Mapping)
    ]
    metric_ids = [
        _string(metric.get("metric_id"))
        for metric in scoring_methodology.get("metrics", []) or []
        if isinstance(metric, Mapping)
    ]
    thresholds: List[Dict[str, Any]] = []
    for task in tasks:
        task_id = _string(task.get("task_id"))
        if not task_id:
            continue
        template = _threshold_template_for_task(task)
        threshold_profile_id = _string(template.pop("threshold_profile_id"))
        thresholds.append(
            {
                "task_id": task_id,
                "ontology_task_id": _string(task.get("ontology_task_id")),
                "task_category": _string(task.get("task_category")),
                "threshold_profile_id": threshold_profile_id,
                "threshold_source": "repo_default_site_task_template",
                "buyer_override_allowed": True,
                "buyer_override_schema": dict(TASK_THRESHOLD_BUYER_OVERRIDE_SCHEMA),
                "buyer_override_status": "not_supplied",
                "supported_metric_ids": metric_ids,
                "thresholds": template,
                "missing_before_claim_upgrade": [
                    "buyer_override_or_acceptance_if_claiming_stronger_gate",
                    "owner_system_action_traces",
                    "actual_outcome_manifest",
                    "non_ranking_operational_claim_evidence",
                ],
                "claim_boundary": (
                    "thresholds_are_eval_gates_not_rank_fidelity_or_non_ranking_operational_claim"
                ),
            }
        )
    return {
        "schema_version": TASK_THRESHOLDS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "task_threshold_count": len(thresholds),
        "task_thresholds": thresholds,
        "threshold_policy": {
            "default_threshold_source": "repo_default_site_task_template",
            "default_thresholds_are_eval_gates": True,
            "buyer_override_allowed": True,
            "buyer_override_schema": dict(TASK_THRESHOLD_BUYER_OVERRIDE_SCHEMA),
            "buyer_override_required_for_guaranteed_claims": True,
            "claim_boundary": (
                "default thresholds make task packs comparable; buyer- or operator-approved "
                "thresholds are required before stronger claims"
            ),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _worldlabs_simready_asset_quality_blockers(
    *,
    worldlabs_world_manifest: Mapping[str, Any],
    marble_validation: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if not worldlabs_world_manifest:
        return []
    missing: List[str] = []
    if not (
        _string(worldlabs_world_manifest.get("collider_glb_uri"))
        or _string(worldlabs_world_manifest.get("collider_mesh_glb_url"))
        or _string(marble_validation.get("collider_mesh_glb_url"))
    ):
        missing.append("collider_glb")
    if not _bool(worldlabs_world_manifest.get("metric_scale_proven")):
        missing.append("metric_scale")
    if not _bool(worldlabs_world_manifest.get("ground_plane_proven")):
        missing.append("ground_plane")
    if not (
        _string(worldlabs_world_manifest.get("usd_scene_uri"))
        or _string(worldlabs_world_manifest.get("ply_scene_uri"))
    ):
        missing.append("usd_or_ply_conversion")
    if not (
        _bool(worldlabs_world_manifest.get("articulated_assets_ready"))
        or _bool(worldlabs_world_manifest.get("physics_ready"))
    ):
        missing.append("articulated_or_physics_ready_assets")
    if not missing:
        return []
    return [
        {
            "blocker_id": "external_worldlabs_simready_asset_quality_blocked",
            "source": "worldlabs_world_manifest",
            "missing": missing,
            "claim_boundary": (
                "external_asset_quality_blocker_only_blueprint_owned_publication_package_is_complete"
            ),
        }
    ]


def _publication_readiness(
    *,
    dataset_state: str,
    dataset_statuses: Sequence[str],
    output_paths: Mapping[str, str],
    task_thresholds: Mapping[str, Any],
    rights_privacy: Mapping[str, Any],
    worldlabs_world_manifest: Mapping[str, Any],
    marble_validation: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    required_status = {
        artifact: bool(_string(output_paths.get(artifact)))
        for artifact in PUBLICATION_REQUIRED_ARTIFACTS
    }
    missing_required = [
        artifact for artifact, present in required_status.items() if not present
    ]
    if int(task_thresholds.get("task_threshold_count") or 0) <= 0:
        missing_required.append("task_thresholds")
    missing_required = list(dict.fromkeys(missing_required))
    missing_proof_labels = [
        status
        for status in dataset_statuses
        if status not in {"capture_grounded_ready", "blocked_rights_privacy"}
    ]
    rights_blocked = _string(rights_privacy.get("rights_status")).lower() in {
        "missing",
        "blocked",
        "denied",
        "failed",
    }
    package_complete = not missing_required and not rights_blocked
    ready = package_complete and dataset_state != "blocked"
    return {
        "schema_version": PUBLICATION_READINESS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "ready_to_evaluate_publishable": ready,
        "publication_label": "Ready to evaluate" if ready else "Needs review",
        "repo_owned_automation_complete": package_complete,
        "required_artifact_status": "complete" if package_complete else "missing",
        "required_artifacts": list(PUBLICATION_REQUIRED_ARTIFACTS),
        "required_artifacts_present": required_status,
        "missing_required_artifacts": missing_required,
        "task_thresholds_uri": output_paths.get("task_thresholds"),
        "publication_readiness_uri": output_paths.get("publication_readiness"),
        "task_threshold_summary": {
            "task_threshold_count": int(task_thresholds.get("task_threshold_count") or 0),
            "threshold_policy": task_thresholds.get("threshold_policy"),
        },
        "missing_proof_labels": missing_proof_labels,
        "external_blockers": _worldlabs_simready_asset_quality_blockers(
            worldlabs_world_manifest=worldlabs_world_manifest,
            marble_validation=marble_validation,
        ),
        "webapp_gate": {
            "may_label_ready_to_evaluate": ready,
            "must_display_missing_proof_labels": True,
            "must_not_display_as": [
                "robot_ready",
                "deployment_ready",
                "non_ranking_operational_claim_validated",
                "simulator_completed",
            ],
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _raw_records(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    for key in ("attempts", "records", "outcomes", "traces"):
        raw = payload.get(key)
        if isinstance(raw, list):
            return [dict(item) for item in raw if isinstance(item, Mapping)]
    return []


def _record_metric(attempt: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    metrics = _mapping(attempt.get("metrics"))
    for key in keys:
        if key in metrics:
            return _number(metrics.get(key), default=default)
        if key in attempt:
            return _number(attempt.get(key), default=default)
    return default


def _record_int_metric(attempt: Mapping[str, Any], *keys: str, default: int = 0) -> int:
    return _int(_record_metric(attempt, *keys, default=float(default)), default=default)


def _infer_recorded_failure_modes(
    *,
    success: bool,
    intervention_count: int,
    unsafe_proximity_count: int,
    collision_count: int,
    object_drop_count: int,
    wrong_object_count: int,
    timeout_count: int,
) -> List[str]:
    if success:
        return []
    modes: List[str] = []
    if intervention_count:
        modes.append("failure_intervention_required")
    if unsafe_proximity_count:
        modes.append("failure_safety_threshold_violation")
    if collision_count:
        modes.append("failure_contact_collision")
    if object_drop_count or wrong_object_count:
        modes.append("failure_manipulation_miss")
    if timeout_count:
        modes.append("failure_cycle_time_exceeded")
    return modes or ["failure_task_not_attempted"]


def _recorded_trace_eval_report(
    *,
    scenario_library: Mapping[str, Any],
    scoring_methodology: Mapping[str, Any],
    recorded_trace_input: Mapping[str, Any],
    source_artifacts: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    scenarios = {
        _string(scenario.get("scenario_id")): dict(scenario)
        for scenario in scenario_library.get("scenarios", [])
        if isinstance(scenario, Mapping)
    }
    raw_attempts = _raw_records(recorded_trace_input)
    attempts: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_attempts):
        scenario_id = _string(raw.get("scenario_id"))
        task_id = _string(raw.get("task_id")) or _string(
            _mapping(scenarios.get(scenario_id)).get("task_id")
        )
        success = _bool(raw.get("success") if raw.get("success") is not None else raw.get("actual_success"))
        intervention_count = _record_int_metric(raw, "intervention_count", "interventions")
        unsafe_proximity_count = _record_int_metric(
            raw,
            "unsafe_proximity_count",
            "unsafe_proximity_event_count",
            "safety_event_count",
        )
        collision_count = _record_int_metric(
            raw,
            "collision_risk_event_count",
            "collision_event_count",
            "contact_event_count",
        )
        object_drop_count = _record_int_metric(raw, "object_drop_count", "drop_count")
        wrong_object_count = _record_int_metric(raw, "wrong_object_count")
        timeout_count = _record_int_metric(raw, "timeout_count")
        recovery_attempt_count = _record_int_metric(raw, "recovery_attempt_count")
        recovery_success_count = _record_int_metric(raw, "recovery_success_count")
        failure_modes = _string_list(raw.get("failure_mode_ids")) or _infer_recorded_failure_modes(
            success=success,
            intervention_count=intervention_count,
            unsafe_proximity_count=unsafe_proximity_count,
            collision_count=collision_count,
            object_drop_count=object_drop_count,
            wrong_object_count=wrong_object_count,
            timeout_count=timeout_count,
        )
        attempts.append(
            {
                "attempt_id": _string(raw.get("attempt_id")) or f"recorded_trace_attempt_{index + 1}",
                "scenario_id": scenario_id,
                "task_id": task_id,
                "trace_id": _string(raw.get("trace_id")) or _string(raw.get("attempt_id")),
                "status": "scored",
                "success": success,
                "cycle_time_seconds": _record_metric(raw, "cycle_time_seconds"),
                "intervention_count": intervention_count,
                "unsafe_proximity_event_count": unsafe_proximity_count,
                "collision_risk_event_count": collision_count,
                "object_drop_count": object_drop_count,
                "wrong_object_count": wrong_object_count,
                "timeout_count": timeout_count,
                "recovery_attempt_count": recovery_attempt_count,
                "recovery_success_count": recovery_success_count,
                "failure_mode_ids": failure_modes,
                "evidence_refs": _mapping(raw.get("evidence_refs")) or _mapping(raw.get("artifact_paths")),
                "claim_boundary": "recorded_trace_fixture_score_is_advisory_not_live_policy_execution",
            }
        )
    attempt_count = len(attempts)
    success_count = sum(1 for attempt in attempts if bool(attempt.get("success")))
    cycle_values = [
        float(attempt["cycle_time_seconds"])
        for attempt in attempts
        if isinstance(attempt.get("cycle_time_seconds"), (int, float))
    ]
    recovery_attempts = sum(_int(attempt.get("recovery_attempt_count")) for attempt in attempts)
    recovery_successes = sum(_int(attempt.get("recovery_success_count")) for attempt in attempts)
    report_status = "scored_advisory" if attempts else "blocked_missing_recorded_trace"
    blockers = [] if attempts else ["missing_recorded_action_trace_manifest"]
    return {
        "schema_version": RECORDED_TRACE_EVAL_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": report_status,
        "blockers": blockers,
        "runner": "recorded_action_trace_fixture",
        "methodology_version": scoring_methodology.get("methodology_version"),
        "attempt_count": attempt_count,
        "scenario_count": len({attempt["scenario_id"] for attempt in attempts}),
        "metrics": {
            "success_rate": round(success_count / float(attempt_count), 6)
            if attempt_count
            else None,
            "mean_cycle_time_seconds": round(sum(cycle_values) / len(cycle_values), 6)
            if cycle_values
            else None,
            "intervention_rate": round(
                sum(_int(attempt.get("intervention_count")) for attempt in attempts)
                / float(attempt_count),
                6,
            )
            if attempt_count
            else None,
            "unsafe_proximity_event_count": sum(
                _int(attempt.get("unsafe_proximity_event_count")) for attempt in attempts
            ),
            "collision_risk_event_count": sum(
                _int(attempt.get("collision_risk_event_count")) for attempt in attempts
            ),
            "object_drop_count": sum(_int(attempt.get("object_drop_count")) for attempt in attempts),
            "wrong_object_count": sum(_int(attempt.get("wrong_object_count")) for attempt in attempts),
            "timeout_count": sum(_int(attempt.get("timeout_count")) for attempt in attempts),
            "recovery_success_rate": round(recovery_successes / float(recovery_attempts), 6)
            if recovery_attempts
            else None,
            "world_model_uncertainty": (
                "medium_recorded_trace_fixture" if attempts else "blocked_missing_trace"
            ),
            "sim_vs_real_calibration_score": "blocked_until_paired_real_outcomes",
        },
        "failure_taxonomy": sorted(
            {
                failure_id
                for attempt in attempts
                for failure_id in _string_list(attempt.get("failure_mode_ids"))
            }
        ),
        "attempts": sorted(attempts, key=lambda item: item["attempt_id"]),
        "evidence_refs": {
            "recorded_action_trace_manifest": source_artifacts.get(
                "recorded_action_trace_manifest"
            ),
            "scenario_library": "scenario_library.json",
            "scoring_methodology": "scoring_methodology.json",
        },
        "proof_boundary": {
            "live_simulator_required": False,
            "docker_required": False,
            "policy_api_required": False,
            "provider_credentials_required": False,
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "non_ranking_operational_claim_validated": False,
            "public_claim_upgrade_allowed": False,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _actual_outcome_records(
    *,
    actual_outcome_input: Mapping[str, Any],
    recorded_trace_eval_report: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, raw in enumerate(_raw_records(actual_outcome_input)):
        records.append(
            {
                "outcome_id": _string(raw.get("outcome_id"))
                or _string(raw.get("attempt_id"))
                or f"actual_outcome_{index + 1}",
                "scenario_id": _string(raw.get("scenario_id")),
                "task_id": _string(raw.get("task_id")),
                "actual_source": _string(raw.get("actual_source")) or "operator_report",
                "actual_success": _bool(raw.get("actual_success") if raw.get("actual_success") is not None else raw.get("success")),
                "cycle_time_seconds": _record_metric(raw, "cycle_time_seconds"),
                "intervention_count": _record_int_metric(raw, "intervention_count", "interventions"),
                "failure_mode_ids": _string_list(raw.get("failure_mode_ids")),
                "tuning_notes": _string_list(raw.get("tuning_notes")),
                "evidence_refs": _mapping(raw.get("evidence_refs")) or _mapping(raw.get("artifact_paths")),
                "claim_boundary": "actual_outcome_record_requires_owner_system_review_for_claim_upgrade",
            }
        )
    if records:
        return records
    for attempt in recorded_trace_eval_report.get("attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        records.append(
            {
                "outcome_id": _string(attempt.get("attempt_id")),
                "scenario_id": _string(attempt.get("scenario_id")),
                "task_id": _string(attempt.get("task_id")),
                "actual_source": "recorded_action_trace",
                "actual_success": bool(attempt.get("success")),
                "cycle_time_seconds": attempt.get("cycle_time_seconds"),
                "intervention_count": _int(attempt.get("intervention_count")),
                "failure_mode_ids": _string_list(attempt.get("failure_mode_ids")),
                "tuning_notes": [],
                "evidence_refs": _mapping(attempt.get("evidence_refs")),
                "claim_boundary": "recorded_trace_actual_is_advisory_until_owner_system_review",
            }
        )
    return records


def _prediction_vs_actual_summary(
    *,
    ledger: Mapping[str, Any],
    actual_outcome_input: Mapping[str, Any],
    recorded_trace_eval_report: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    actuals = _actual_outcome_records(
        actual_outcome_input=actual_outcome_input,
        recorded_trace_eval_report=recorded_trace_eval_report,
    )
    prediction_records = [
        dict(record) for record in ledger.get("records", []) if isinstance(record, Mapping)
    ]
    matched_rows: List[Dict[str, Any]] = []
    for actual in actuals:
        matching_predictions = [
            record
            for record in prediction_records
            if _string(record.get("scenario_id")) == _string(actual.get("scenario_id"))
            and _string(record.get("task_id")) == _string(actual.get("task_id"))
        ]
        predicted_failures = sorted(
            {
                failure_id
                for record in matching_predictions
                for failure_id in _string_list(record.get("failure_mode_ids"))
            }
        )
        actual_failures = _string_list(actual.get("failure_mode_ids"))
        missed_failures = sorted(set(actual_failures) - set(predicted_failures))
        matched_rows.append(
            {
                "outcome_id": actual.get("outcome_id"),
                "scenario_id": actual.get("scenario_id"),
                "task_id": actual.get("task_id"),
                "actual_source": actual.get("actual_source"),
                "matching_prediction_count": len(matching_predictions),
                "predicted_failures": predicted_failures,
                "actual_failures": actual_failures,
                "missed_failures": missed_failures,
                "actual_success": actual.get("actual_success"),
                "tuning_notes": actual.get("tuning_notes") or [],
                "calibration_status": "matched_prediction"
                if matching_predictions
                else "missing_prediction_for_actual",
                "claim_boundary": actual.get("claim_boundary"),
            }
        )
    status = "advisory_actuals_ingested" if matched_rows else "blocked_missing_actuals"
    actual_successes = [
        bool(row.get("actual_success")) for row in matched_rows if row.get("actual_success") is not None
    ]
    return {
        "schema_version": PREDICTION_VS_ACTUAL_SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "missing_actuals_remain_blocked": not bool(matched_rows),
        "actual_record_count": len(matched_rows),
        "matched_prediction_count": sum(
            _int(row.get("matching_prediction_count")) for row in matched_rows
        ),
        "missed_failure_count": sum(len(_string_list(row.get("missed_failures"))) for row in matched_rows),
        "records": sorted(matched_rows, key=lambda row: _string(row.get("outcome_id"))),
        "calibration_summary": {
            "actual_success_rate": round(
                sum(1 for success in actual_successes if success) / float(len(actual_successes)),
                6,
            )
            if actual_successes
            else None,
            "source_types": sorted({_string(row.get("actual_source")) for row in matched_rows}),
            "calibration_status": status,
            "sim_vs_real_calibration_score": "blocked_until_sim_and_real_pairs_exist",
        },
        "blockers": [] if matched_rows else ["missing_actual_outcome_manifest"],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _first_text(*values: Any, fallback: str = "") -> str:
    for value in values:
        text = _string(value)
        if text:
            return text
    return fallback


def _metadata_from(*payloads: Mapping[str, Any]) -> Dict[str, Any]:
    for payload in payloads:
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            return dict(metadata)
    return {}


def _condition_card(
    *,
    key: str,
    value: Any,
    label_when_missing: str = "needs_site_operator_review",
) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        text = _first_text(value.get("label"), value.get("status"), fallback=label_when_missing)
        source = _first_text(value.get("source"), value.get("evidence_source"), fallback="metadata")
        confidence = _first_text(value.get("confidence"), fallback="derived")
    else:
        text = _first_text(value, fallback=label_when_missing)
        source = "metadata" if text != label_when_missing else "missing_annotation"
        confidence = "derived" if text != label_when_missing else "needs_site_operator_review"
    return {
        "condition": key,
        "value": text,
        "label_source": source,
        "confidence": confidence,
        "ground_truth_status": "observed_or_operator_supplied"
        if confidence not in {"needs_site_operator_review", "agent_inferred"}
        else "needs_review",
    }


def _condition_cards(metadata: Mapping[str, Any], keys: Sequence[str]) -> List[Dict[str, Any]]:
    raw_conditions = metadata.get("robot_eval_conditions")
    condition_map = dict(raw_conditions) if isinstance(raw_conditions, Mapping) else {}
    return [
        _condition_card(
            key=key,
            value=condition_map.get(key) if key in condition_map else metadata.get(key),
        )
        for key in keys
    ]


def _task_zone_cards(tasks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    zones: List[Dict[str, Any]] = []
    for task in tasks:
        task_id = _string(task.get("task_id"))
        if not task_id:
            continue
        zone_candidates = _mapping(task.get("site_zone_candidates"))
        zones.append(
            {
                "task_id": task_id,
                "start_zone": task.get("start_zone"),
                "goal_zone": task.get("goal_zone"),
                "start_zone_id": task.get("start_zone_id"),
                "goal_zone_id": task.get("goal_zone_id"),
                "spawn_candidates": zone_candidates.get("spawn_candidates") or [],
                "target_candidates": zone_candidates.get("target_candidates") or [],
                "validated_spawn_target_pair": bool(
                    zone_candidates.get("validated_spawn_target_pair")
                ),
                "label_source": "task_anchor_manifest",
                "confidence": "derived_from_capture_package",
            }
        )
    return zones


def _object_location_cards(object_geometry_manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_objects = object_geometry_manifest.get("objects")
    if not isinstance(raw_objects, list):
        return []
    objects: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_objects):
        if not isinstance(item, Mapping):
            continue
        object_id = _string(item.get("object_id") or item.get("id") or f"object_{index}")
        if not object_id:
            continue
        objects.append(
            {
                "object_id": object_id,
                "label": _first_text(item.get("label"), item.get("class_name"), fallback="object"),
                "task_role": _string(item.get("task_role")),
                "semantic_roles": _string_list(
                    item.get("semantic_roles")
                    or item.get("roles")
                    or ([item.get("task_role")] if item.get("task_role") else [])
                ),
                "location": item.get("placement_bbox") or item.get("boundingBox") or item.get("bbox"),
                "center_xyz": _object_center(item),
                "has_collision_hulls": bool(item.get("collision_hulls")),
                "has_support_surfaces": bool(item.get("support_surfaces")),
                "physics_coverage_status": "covered"
                if item.get("collision_hulls") or item.get("support_surfaces")
                else "review_required",
                "label_source": "object_geometry_manifest",
                "confidence": _first_text(
                    _mapping(item.get("provenance")).get("grounding_level"),
                    fallback="derived",
                ),
            }
        )
    return sorted(objects, key=lambda item: item["object_id"])


def _collider_available(
    *,
    marble_validation: Mapping[str, Any],
    marble_bridge: Mapping[str, Any],
    worldlabs_world_manifest: Mapping[str, Any],
) -> bool:
    if marble_validation.get("physics_collision_review_ready") is True:
        return True
    if marble_validation.get("collider_mesh_available") is True:
        return True
    for payload in (marble_validation, marble_bridge, worldlabs_world_manifest):
        mesh = payload.get("mesh")
        assets = payload.get("assets")
        candidates = [payload]
        if isinstance(mesh, Mapping):
            candidates.append(mesh)
        if isinstance(assets, Mapping):
            candidates.append(assets)
            asset_mesh = assets.get("mesh")
            if isinstance(asset_mesh, Mapping):
                candidates.append(asset_mesh)
        for candidate in candidates:
            if _first_text(
                candidate.get("collider_mesh_glb_url"),
                candidate.get("collider_mesh_url"),
                candidate.get("collider_url"),
            ):
                return True
    return False


def _portable_collider_glb_present(
    *,
    marble_validation: Mapping[str, Any],
    marble_bridge: Mapping[str, Any],
    worldlabs_world_manifest: Mapping[str, Any],
) -> bool:
    for payload in (marble_validation, marble_bridge, worldlabs_world_manifest):
        mesh = payload.get("mesh")
        assets = payload.get("assets")
        candidates = [payload]
        if isinstance(mesh, Mapping):
            candidates.append(mesh)
        if isinstance(assets, Mapping):
            candidates.append(assets)
            asset_mesh = assets.get("mesh")
            if isinstance(asset_mesh, Mapping):
                candidates.append(asset_mesh)
        for candidate in candidates:
            if _first_text(
                candidate.get("collider_glb_uri"),
                candidate.get("collider_mesh_glb_url"),
                candidate.get("collider_mesh_url"),
                candidate.get("collider_url"),
            ):
                return True
    return False


def _collision_backend_labels(
    *,
    simready_scene_manifest: Mapping[str, Any],
    marble_validation: Mapping[str, Any],
    marble_bridge: Mapping[str, Any],
    worldlabs_world_manifest: Mapping[str, Any],
    cpu_preflight_scorecard: Mapping[str, Any],
) -> Dict[str, Any]:
    portable_collider_present = _portable_collider_glb_present(
        marble_validation=marble_validation,
        marble_bridge=marble_bridge,
        worldlabs_world_manifest=worldlabs_world_manifest,
    )
    worldlabs_assets = _mapping(worldlabs_world_manifest.get("assets"))
    worldlabs_splats = _mapping(worldlabs_assets.get("splats"))
    isaac_import_candidate = bool(
        cpu_preflight_scorecard.get("isaac_usd_import_candidate")
        or simready_scene_manifest
        or _string(worldlabs_world_manifest.get("usd_scene_uri"))
        or _mapping(worldlabs_splats.get("usd_urls"))
        or worldlabs_splats.get("usd_url")
    )
    isaac_collision_verified = bool(cpu_preflight_scorecard.get("isaac_usd_collision_verified"))
    isaac_collision_unverified = bool(
        cpu_preflight_scorecard.get("isaac_usd_collision_unverified")
        or (isaac_import_candidate and not isaac_collision_verified)
    )
    cpu_proxy_estimated = bool(cpu_preflight_scorecard.get("cpu_proxy_collision_estimated"))
    labels: List[str] = []
    if isaac_import_candidate:
        labels.append("isaac_usd_import_candidate")
    if isaac_collision_verified:
        labels.append("isaac_usd_collision_verified")
    if isaac_collision_unverified:
        labels.append("isaac_usd_collision_unverified")
    if portable_collider_present:
        labels.append("portable_collider_glb_present")
    else:
        labels.append("portable_collider_glb_missing")
    if cpu_proxy_estimated:
        labels.append("cpu_proxy_collision_estimated")
    labels.append("simulator_execution_not_run")
    blockers = [
        label
        for label in labels
        if label
        in {
            "isaac_usd_collision_unverified",
            "portable_collider_glb_missing",
            "simulator_execution_not_run",
        }
    ]
    if not cpu_proxy_estimated and not portable_collider_present:
        blockers.append("cpu_collision_proxy_missing")
    return {
        "labels": list(dict.fromkeys(labels)),
        "blockers": list(dict.fromkeys(blockers)),
        "isaac_usd_import_candidate": isaac_import_candidate,
        "isaac_usd_collision_verified": isaac_collision_verified,
        "isaac_usd_collision_unverified": isaac_collision_unverified,
        "portable_collider_glb_present": portable_collider_present,
        "portable_collider_glb_missing": not portable_collider_present,
        "cpu_proxy_collision_estimated": cpu_proxy_estimated,
        "simulator_execution_not_run": True,
    }


def _site_card(
    *,
    context: Any,
    descriptor: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    site_world_spec: Mapping[str, Any],
    object_geometry_manifest: Mapping[str, Any],
    task_library: Mapping[str, Any],
    source_artifacts: Mapping[str, Any],
    simready_scene_manifest: Mapping[str, Any],
    marble_validation: Mapping[str, Any],
    marble_bridge: Mapping[str, Any],
    worldlabs_world_manifest: Mapping[str, Any],
    cpu_preflight_scorecard: Mapping[str, Any],
    protected_regions_manifest: Mapping[str, Any],
    rights_privacy: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    metadata = _metadata_from(descriptor, raw_manifest)
    explicit_site_type = _first_text(
        site_world_spec.get("site_type"),
        site_world_spec.get("target_site_type"),
        metadata.get("site_type"),
        metadata.get("target_site_type"),
        raw_manifest.get("site_type"),
    )
    tasks = [task for task in task_library.get("tasks", []) if isinstance(task, Mapping)]
    site_type = explicit_site_type or _infer_site_type(
        metadata=metadata,
        tasks=tasks,
        object_geometry_manifest=object_geometry_manifest,
    )
    collider_ready = _collider_available(
        marble_validation=marble_validation,
        marble_bridge=marble_bridge,
        worldlabs_world_manifest=worldlabs_world_manifest,
    )
    collision_backend_labels = _collision_backend_labels(
        simready_scene_manifest=simready_scene_manifest,
        marble_validation=marble_validation,
        marble_bridge=marble_bridge,
        worldlabs_world_manifest=worldlabs_world_manifest,
        cpu_preflight_scorecard=cpu_preflight_scorecard,
    )
    object_index = _object_index(object_geometry_manifest)
    semantics_metadata = _mapping(
        _mapping(worldlabs_world_manifest.get("assets")).get("semantics_metadata")
    )
    world_manifest_semantics = _mapping(worldlabs_world_manifest.get("semantics_metadata"))
    metric_scale_factor = (
        semantics_metadata.get("metric_scale_factor")
        or world_manifest_semantics.get("metric_scale_factor")
    )
    object_scale_available = bool(object_index["object_count"])
    if metric_scale_factor is None and object_scale_available:
        metric_scale_factor = 1.0
        scale_status = "derived_from_object_geometry_manifest"
        scale_source = "object_geometry_manifest_metric_coordinates"
    elif metric_scale_factor is not None:
        scale_status = "present"
        scale_source = "worldlabs_semantics_metadata"
    else:
        scale_status = "needs_scale_review"
        scale_source = "worldlabs_semantics_metadata"
    geometry_summary = {
        "splat": {
            "status": "present"
            if worldlabs_world_manifest or marble_bridge
            else "missing",
            "evidence": source_artifacts.get("worldlabs_world_manifest")
            or source_artifacts.get("marble_simready_bridge"),
            "label_source": "pipeline_artifact",
        },
        "mesh": {
            "status": "present"
            if _first_text(
                marble_validation.get("high_quality_mesh_glb_url"),
                marble_validation.get("mesh_glb_url"),
                marble_bridge.get("mesh_glb_url"),
            )
            else "review_required",
            "evidence": source_artifacts.get("marble_asset_validation"),
            "label_source": "marble_or_pipeline_artifact",
        },
        "collider": {
            "status": "review_input_present"
            if collider_ready
            else "cpu_proxy_collision_estimated"
            if collision_backend_labels["cpu_proxy_collision_estimated"]
            else "blocked_missing_collider",
            "collision_ready_claim_allowed": False,
            "evidence": source_artifacts.get("marble_asset_validation")
            or source_artifacts.get("marble_simready_bridge")
            or source_artifacts.get("cpu_preflight_scorecard"),
            "label_source": "marble_asset_validation",
            "backend_labels": collision_backend_labels["labels"],
            "backend_blockers": collision_backend_labels["blockers"],
            "isaac_usd_import_candidate": collision_backend_labels[
                "isaac_usd_import_candidate"
            ],
            "isaac_usd_collision_verified": collision_backend_labels[
                "isaac_usd_collision_verified"
            ],
            "isaac_usd_collision_unverified": collision_backend_labels[
                "isaac_usd_collision_unverified"
            ],
            "portable_collider_glb_missing": collision_backend_labels[
                "portable_collider_glb_missing"
            ],
            "cpu_proxy_collision_estimated": collision_backend_labels[
                "cpu_proxy_collision_estimated"
            ],
            "simulator_execution_not_run": True,
        },
        "scale": {
            "metric_scale_factor": metric_scale_factor,
            "status": scale_status,
            "label_source": scale_source,
        },
        "navigation_zones": _task_zone_cards(tasks),
        "object_index": object_index,
    }
    restricted_zones = protected_regions_manifest.get("protected_regions")
    if not isinstance(restricted_zones, list):
        restricted_zones = protected_regions_manifest.get("restricted_zones")
    if not isinstance(restricted_zones, list):
        restricted_zones = []

    return {
        "schema_version": SITE_CARD_SCHEMA_VERSION,
        "dataset_version": ROBOT_EVAL_DATASET_VERSION,
        "generated_at": generated_at,
        "site_card_id": f"site_card_{_stable_slug(context.scene_id, fallback='scene')}_{_stable_slug(context.capture_id, fallback='capture')}",
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "site_id": _site_id_from_inputs(descriptor, raw_manifest),
        "site_type": site_type,
        "site_type_allowed_values": [
            "warehouse aisle",
            "loading dock",
            "hospital hallway",
            "factory line-side station",
            "stockroom",
            "captured indoor scene",
        ],
        "geometry": geometry_summary,
        "visual_conditions": _condition_cards(
            metadata,
            ["lighting", "glare", "clutter", "signage", "reflective_surfaces"],
        ),
        "dynamic_conditions": _condition_cards(
            metadata,
            ["human_paths", "carts", "forklifts", "doors", "blocked_pathways"],
        ),
        "safety_constraints": {
            "restricted_zones": restricted_zones,
            "human_proximity_rules": metadata.get("human_proximity_rules")
            or "needs_site_operator_review",
            "pinch_points": metadata.get("pinch_points") or "needs_site_operator_review",
            "no_go_areas": metadata.get("no_go_areas") or "needs_site_operator_review",
            "label_source": "protected_regions_manifest_or_site_metadata",
            "claim_boundary": "safety_constraints_are_review_inputs_not_non_ranking_operational_claim",
        },
        "robot_metadata": {
            "traversable_routes": metadata.get("traversable_routes") or _task_zone_cards(tasks),
            "robot_pov_camera_paths": metadata.get("robot_pov_camera_paths")
            or "needs_robot_pov",
            "task_zones": _task_zone_cards(tasks),
            "object_locations": _object_location_cards(object_geometry_manifest),
            "object_index_status": object_index["status"],
            "physics_coverage_complete": object_index["physics_coverage_complete"],
            "claim_boundary": "robot_metadata_is_task_context_not_robot_policy_execution",
        },
        "provenance_rights_review_status": {
            "rights_privacy": rights_privacy,
            "source_artifacts": dict(source_artifacts),
            "simready_review_artifact_present": bool(simready_scene_manifest),
            "marble_review_artifact_present": bool(marble_bridge or marble_validation),
            "claim_boundary": "capture_and_rights_review_do_not_prove_external_licensing_clearance",
        },
    }


def _task_cards(*, task_library: Mapping[str, Any], generated_at: str) -> Dict[str, Any]:
    cards: List[Dict[str, Any]] = []
    for task in task_library.get("tasks", []):
        if not isinstance(task, Mapping):
            continue
        task_id = _string(task.get("task_id"))
        target_objects = [
            item for item in task.get("target_objects", []) if isinstance(item, Mapping)
        ]
        zone_candidates = _mapping(task.get("site_zone_candidates"))
        cards.append(
            {
                "schema_version": "real_site_robot_eval_task_card.v0.1",
                "dataset_version": ROBOT_EVAL_DATASET_VERSION,
                "task_card_id": f"task_card_{_stable_slug(task_id, fallback='task')}",
                "task_id": task_id,
                "task_statement": _first_text(task.get("task_text"), fallback=task_id),
                "task_category": _string(task.get("task_category") or "generic"),
                "ontology_task_id": _string(task.get("ontology_task_id")),
                "ontology_version": _string(task.get("ontology_version") or "1.0"),
                "task_family_aliases": _string_list(task.get("task_family_aliases")),
                "cross_site_query_fields": _string_list(task.get("cross_site_query_fields")),
                "start_state": {
                    "start_zone": task.get("start_zone"),
                    "start_zone_id": task.get("start_zone_id"),
                    "spawn_candidates": zone_candidates.get("spawn_candidates") or [],
                    "label_source": "task_anchor_manifest",
                    "confidence": "derived",
                },
                "goal_state": {
                    "goal_zone": task.get("goal_zone"),
                    "goal_zone_id": task.get("goal_zone_id"),
                    "target_candidates": zone_candidates.get("target_candidates") or [],
                    "label_source": "task_anchor_manifest",
                    "confidence": "derived",
                },
                "target_object_ids": _string_list(task.get("target_object_ids")),
                "target_objects": [dict(item) for item in target_objects],
                "task_objects": [dict(item) for item in target_objects],
                "semantic_grounding": {
                    "object_semantics_status": task.get("object_semantics_status"),
                    "semantic_zone_pair_status": task.get("semantic_zone_pair_status"),
                    "validated_spawn_target_pair": bool(
                        zone_candidates.get("validated_spawn_target_pair")
                    ),
                    "validated_spawn_candidate_count": int(
                        zone_candidates.get("validated_spawn_candidate_count") or 0
                    ),
                    "validated_target_candidate_count": int(
                        zone_candidates.get("validated_target_candidate_count") or 0
                    ),
                },
                "success_definition": task.get("success_criteria") or [],
                "failure_definition": {
                    "failure_mode_ids": [
                        "failure_task_not_attempted",
                        "failure_evidence_missing",
                        "failure_intervention_required",
                        "failure_safety_threshold_violation",
                    ],
                    "label_source": "failure_taxonomy",
                },
                "required_metrics": [
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
                    "placement_accuracy",
                ],
                "task_evidence_source": _string(task.get("source_artifact"))
                or "pipeline/evaluation_prep/task_anchor_manifest.json",
                "confidence": "capture_grounded_task_anchor_present",
                "observed_vs_inferred_labels": {
                    "task_anchor": "derived",
                    "target_objects": "observed" if target_objects else "needs_annotation",
                    "success_definition": "derived",
                },
                "required_missing_annotations": [
                    "robot_pov_attempt_alignment",
                    "human_demo_steps",
                    "action_log_link",
                    "actual_outcome_label",
                ],
                "claim_boundary": "task_card_defines_eval_scope_not_robot_execution",
            }
        )
    return {
        "schema_version": TASK_CARDS_SCHEMA_VERSION,
        "dataset_version": ROBOT_EVAL_DATASET_VERSION,
        "generated_at": generated_at,
        "task_card_count": len(cards),
        "cards": sorted(cards, key=lambda item: item["task_card_id"]),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _engine_for_prediction_source(source: str) -> str:
    return {
        "simready_review": "local preflight",
        "marble_review": "hosted visual review",
        "cosmos_preflight": "Cosmos-generated variation",
        "human_eval": "hosted visual review",
        "future_provider": "blocked future provider",
        "simulator_trace": "Isaac Sim or simulator trace",
        "robot_trial": "real pilot result",
    }.get(source, source or "unknown")


def _scenario_cards(
    *,
    scenario_library: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    cards: List[Dict[str, Any]] = []
    for scenario in scenario_library.get("scenarios", []):
        if not isinstance(scenario, Mapping):
            continue
        scenario_id = _string(scenario.get("scenario_id"))
        scenario_family_id = f"family_{_stable_slug(scenario_id, fallback='scenario')}"
        spawn_candidates = [
            dict(item)
            for item in scenario.get("spawn_candidates", []) or []
            if isinstance(item, Mapping)
        ]
        target_candidates = [
            dict(item)
            for item in scenario.get("target_candidates", []) or []
            if isinstance(item, Mapping)
        ]
        target_objects = [
            dict(item)
            for item in scenario.get("target_objects", []) or []
            if isinstance(item, Mapping)
        ]
        cards.append(
            {
                "schema_version": "real_site_robot_eval_scenario_card.v0.1",
                "dataset_version": ROBOT_EVAL_DATASET_VERSION,
                "scenario_card_id": f"scenario_card_{_stable_slug(scenario_id, fallback='scenario')}",
                "scenario_id": scenario_id,
                "scenario_family_id": scenario_family_id,
                "scenario_family_artifact": "scenario_family_library.json",
                "task_id": _string(scenario.get("task_id")),
                "robot_profile_id": _string(scenario.get("robot_profile_id")),
                "target_object_ids": _string_list(scenario.get("target_object_ids")),
                "target_objects": target_objects,
                "start_zone": scenario.get("start_zone"),
                "goal_zone": scenario.get("goal_zone"),
                "start_zone_id": scenario.get("start_zone_id"),
                "goal_zone_id": scenario.get("goal_zone_id"),
                "spawn_candidates": spawn_candidates,
                "target_candidates": target_candidates,
                "semantic_spawn_target": {
                    "validated_spawn_target_pair": bool(
                        scenario.get("validated_spawn_target_pair")
                    ),
                    "validated_spawn_candidate_count": int(
                        scenario.get("validated_spawn_candidate_count") or 0
                    ),
                    "validated_target_candidate_count": int(
                        scenario.get("validated_target_candidate_count") or 0
                    ),
                    "source": _string(scenario.get("semantic_spawn_target_source")),
                    "fallback_allowed_for_beta_release": False,
                },
                "normal_scenario": {
                    "statement": "Run the named task under the capture-observed site layout.",
                    "label_source": "task_anchor_manifest",
                    "ground_truth_status": "derived_from_capture_package",
                },
                "variation": {
                    "statement": "Lighting, clutter, route, and dynamic-agent variations require operator review before use.",
                    "label_source": "scenario_template",
                    "ground_truth_status": "agent_inferred_needs_review",
                },
                "edge_case": {
                    "statement": "Blocked pathway, unexpected human proximity, object occlusion, or fixture articulation ambiguity.",
                    "label_source": "failure_taxonomy",
                    "ground_truth_status": "agent_inferred_needs_review",
                },
                "known_risk": [
                    "generated_scenarios_are_not_real_world_proof",
                    "missing_action_logs_block_robot_policy_claim",
                    "missing_actual_outcome_blocks_validation",
                ],
                "observed_vs_inferred_labels": {
                    "layout": "capture_grounded",
                    "task_objects": "observed" if target_objects else "needs_annotation",
                    "spawn_target_zones": "derived"
                    if scenario.get("validated_spawn_target_pair")
                    else "needs_annotation",
                    "variation": "agent_inferred",
                    "edge_case": "agent_inferred",
                },
                "scenario_family_variation_ids": [
                    "capture_observed_layout",
                    *[
                        _string(definition.get("variation_id"))
                        for definition in SCENARIO_VARIATION_DEFINITIONS
                    ],
                ],
                "required_missing_annotations": scenario.get("missing_evidence_statuses") or [],
                "claim_boundary": "scenario_card_is_review_scope_not_simulator_or_pilot_result",
            }
        )
    return {
        "schema_version": SCENARIO_CARDS_SCHEMA_VERSION,
        "dataset_version": ROBOT_EVAL_DATASET_VERSION,
        "generated_at": generated_at,
        "scenario_card_count": len(cards),
        "cards": sorted(cards, key=lambda item: item["scenario_card_id"]),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _eval_cards(*, ledger: Mapping[str, Any], generated_at: str) -> Dict[str, Any]:
    cards: List[Dict[str, Any]] = []
    for record in ledger.get("records", []):
        if not isinstance(record, Mapping):
            continue
        record_id = _string(record.get("record_id"))
        prediction_source = _string(record.get("prediction_source"))
        cards.append(
            {
                "schema_version": "real_site_robot_eval_eval_card.v0.1",
                "dataset_version": ROBOT_EVAL_DATASET_VERSION,
                "eval_card_id": f"eval_card_{_stable_slug(record_id, fallback='record')}",
                "record_id": record_id,
                "scenario_id": _string(record.get("scenario_id")),
                "task_id": _string(record.get("task_id")),
                "robot_or_policy_tested": record.get("robot_profile_id")
                or "robot_policy_required",
                "engine_used": _engine_for_prediction_source(prediction_source),
                "prediction_source": prediction_source,
                "predicted_results": {
                    "predicted_success": record.get("predicted_success"),
                    "metrics": record.get("metrics") or {},
                    "prediction_status": record.get("prediction_status"),
                },
                "failure_modes": record.get("failure_mode_ids") or [],
                "intervention_estimate": _mapping(record.get("metrics")).get(
                    "intervention_count"
                ),
                "world_model_uncertainty": "high_until_action_logs_and_actual_outcomes_exist",
                "validation": {
                    "predicted_vs_actual": None,
                    "actual_source": record.get("actual_source"),
                    "actual_status": record.get("actual_status"),
                    "actual_success": record.get("actual_success"),
                },
                "proof_boundary": record.get("claim_boundary")
                or "prediction_only_no_actual_outcome_no_deployment_claim",
                "blocked_upgrades": [
                    "simulator_execution_completed",
                    "robot_policy_execution_proven",
                    "non_ranking_operational_claim_proven",
                    "real_pilot_outcome_proven",
                ],
            }
        )
    return {
        "schema_version": EVAL_CARDS_SCHEMA_VERSION,
        "dataset_version": ROBOT_EVAL_DATASET_VERSION,
        "generated_at": generated_at,
        "eval_card_count": len(cards),
        "cards": sorted(cards, key=lambda item: item["eval_card_id"]),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _annotation_backlog(
    *,
    site_card: Mapping[str, Any],
    task_cards: Mapping[str, Any],
    scenario_cards: Mapping[str, Any],
    dataset_statuses: Sequence[str],
    generated_at: str,
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []

    def add(item_id: str, card_family: str, annotation: str, reason: str) -> None:
        if item_id not in {item["backlog_id"] for item in items}:
            items.append(
                {
                    "backlog_id": item_id,
                    "card_family": card_family,
                    "annotation": annotation,
                    "status": "required",
                    "reason": reason,
                    "allowed_labels": [
                        "observed",
                        "derived",
                        "agent_inferred",
                        "needs_site_operator_review",
                        "blocked",
                    ],
                }
            )

    collider = _mapping(_mapping(site_card.get("geometry")).get("collider"))
    for backend_blocker in _string_list(collider.get("backend_blockers")):
        add(
            backend_blocker,
            "site_card",
            backend_blocker,
            "Backend-specific collision or simulator proof remains missing; collision-ready and physics/contact claims must remain blocked.",
        )
    for status in dataset_statuses:
        if status == "needs_robot_pov":
            add("needs_robot_pov", "eval_cards", "robot_pov_attempt_alignment", "Robot POV evidence is missing.")
        if status == "needs_human_demo":
            add("needs_human_demo", "task_cards", "human_demo_step_labels", "Human demonstration labels are missing.")
        if status == "needs_action_logs":
            add("needs_action_logs", "eval_cards", "action_or_teleop_log_link", "Action/policy evaluation is blocked without action logs.")
        if status == "needs_actual_outcome":
            add("needs_actual_outcome", "eval_cards", "actual_outcome_record", "Predicted-vs-actual validation is blocked.")
        if status.startswith("needs_") and status.endswith("_ref"):
            add(
                status,
                "robot_team_test_submission_modalities",
                status.replace("needs_", "").replace("_ref", "_reference"),
                "Robot-team submission modality reference is missing or not selected.",
            )
        if status == "blocked_rights_privacy":
            add("blocked_rights_privacy", "site_card", "rights_privacy_clearance", "External licensing and public-use claims are blocked.")

    for task in task_cards.get("cards", []):
        if isinstance(task, Mapping):
            semantic_grounding = _mapping(task.get("semantic_grounding"))
            if semantic_grounding.get("object_semantics_status") != "object_grounded":
                add(
                    f"{_stable_slug(task.get('task_card_id'), fallback='task')}_target_object_semantics",
                    "task_cards",
                    "target_object_semantics",
                    "Task target objects are missing object-geometry grounding.",
                )
            if semantic_grounding.get("validated_spawn_target_pair") is not True:
                add(
                    f"{_stable_slug(task.get('task_card_id'), fallback='task')}_validated_spawn_target_pair",
                    "task_cards",
                    "validated_spawn_target_pair",
                    "Task start/goal zones are missing finite validated site-coordinate poses.",
                )
            for annotation in _string_list(task.get("required_missing_annotations")):
                add(
                    f"{_stable_slug(task.get('task_card_id'), fallback='task')}_{annotation}",
                    "task_cards",
                    annotation,
                    "Task Card requires this annotation before stronger eval claims.",
                )
    for scenario in scenario_cards.get("cards", []):
        if isinstance(scenario, Mapping):
            semantic_spawn_target = _mapping(scenario.get("semantic_spawn_target"))
            if semantic_spawn_target.get("validated_spawn_target_pair") is not True:
                add(
                    f"{_stable_slug(scenario.get('scenario_card_id'), fallback='scenario')}_semantic_spawn_target",
                    "scenario_cards",
                    "semantic_spawn_target",
                    "Scenario needs a finite task-zone-derived spawn/target pair before beta release.",
                )
            add(
                f"{_stable_slug(scenario.get('scenario_card_id'), fallback='scenario')}_variation_review",
                "scenario_cards",
                "variation_edge_case_operator_review",
                "Generated or inferred variations are not real-world proof.",
            )

    return {
        "schema_version": ANNOTATION_BACKLOG_SCHEMA_VERSION,
        "dataset_version": ROBOT_EVAL_DATASET_VERSION,
        "generated_at": generated_at,
        "backlog_count": len(items),
        "items": sorted(items, key=lambda item: item["backlog_id"]),
        "claim_boundary": "annotation_backlog_names_missing_work_and_never_clears_operational_claims",
    }


def _proof_boundaries(
    *,
    rights_privacy: Mapping[str, Any],
    collider_present: bool,
    collider_backend_labels: Sequence[str],
    collider_backend_blockers: Sequence[str],
    action_log_input: Mapping[str, Any],
    actual_outcome_input: Mapping[str, Any],
    robot_team_submission_input: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": PROOF_BOUNDARIES_SCHEMA_VERSION,
        "dataset_version": ROBOT_EVAL_DATASET_VERSION,
        "generated_at": generated_at,
        "repo_local_only": True,
        "simulator_execution_proven": False,
        "physics_contact_validation_proven": False,
        "robot_policy_execution_proven": False,
        "non_ranking_operational_claim_proven": False,
        "rights_cleared_external_licensing_proven": False,
        "real_pilot_outcome_proven": bool(actual_outcome_input.get("real_pilot_outcome_proven"))
        and bool(actual_outcome_input.get("owner_system_proof_uri")),
        "generated_scenarios_are_real_world_proof": False,
        "collider_review_input_present": collider_present,
        "collider_backend_labels": list(collider_backend_labels),
        "collider_backend_blockers": list(collider_backend_blockers),
        "action_logs_present": bool(action_log_input),
        "robot_team_test_submission_refs_present": bool(robot_team_submission_input),
        "rights_privacy_status": dict(rights_privacy),
        "blocked_upgrades": [
            "collision_ready_claim"
            if not collider_present
            else "collision_ready_claim_requires_simulator_contact_validation",
            "action_policy_eval_claim"
            if not action_log_input
            else "action_policy_eval_claim_requires_owner_system_validation",
            "external_licensing_claim",
            "non_ranking_operational_claim_validated_claim",
            "ready_to_deploy_claim",
            *list(collider_backend_blockers),
        ],
        "allowed_public_display": [
            "real-site robot evaluation dataset and workflow",
            "Site Card",
            "Task Cards",
            "Scenario Cards",
            "Eval Cards",
            "missing-proof labels",
            "advisory pre-pilot workflow",
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _rights_privacy_status(
    *,
    rights_summary: Mapping[str, Any],
    rights_review: Mapping[str, Any],
    privacy_manifest: Mapping[str, Any],
    descriptor: Mapping[str, Any] | None = None,
    raw_manifest: Mapping[str, Any] | None = None,
    capture_root: str | Path | None = None,
) -> Dict[str, Any]:
    descriptor = _mapping(descriptor)
    raw_manifest = _mapping(raw_manifest)
    descriptor_metadata = _mapping(descriptor.get("metadata"))
    capture_rights = {
        **_mapping(raw_manifest.get("rights_scope")),
        **_mapping(raw_manifest.get("capture_rights")),
        **_mapping(descriptor.get("rights_profile")),
        **_mapping(descriptor.get("capture_rights")),
        **_mapping(descriptor_metadata.get("capture_rights")),
    }
    consent_scope = _string_list(
        capture_rights.get("consent_scope") or capture_rights.get("consentScope")
    )
    consent_status = _string(
        capture_rights.get("consent_status") or capture_rights.get("consentStatus")
    ).lower()
    consent_revoked = (
        consent_status in {"revoked", "withdrawn", "rescinded"}
        or _bool(capture_rights.get("consent_revoked"))
        or _bool(capture_rights.get("consentRevoked"))
        or bool(
            _string(
                capture_rights.get("consent_revoked_at")
                or capture_rights.get("consentRevokedAt")
            )
        )
    )
    # TOCTOU: re-read consent LIVE so a revocation that landed after the
    # descriptor/manifest were written still blocks the dataset (fail-closed;
    # a live read can only ADD a revocation, never clear the manifest state).
    if not consent_revoked and capture_root is not None:
        from .consent_takedown import read_consent_state

        if read_consent_state(capture_root).get("state") == "revoked":
            consent_revoked = True
    consent_revoked_at = _string(
        capture_rights.get("consent_revoked_at") or capture_rights.get("consentRevokedAt")
    )
    commercialization_terms = _mapping(
        capture_rights.get("commercialization_terms")
        or capture_rights.get("commercializationTerms")
        or capture_rights.get("commercial_terms")
        or capture_rights.get("commercialTerms")
    )
    revenue_share_terms = _mapping(
        capture_rights.get("operator_revenue_terms")
        or capture_rights.get("operatorRevenueTerms")
        or capture_rights.get("revenue_share_terms")
        or capture_rights.get("revenueShareTerms")
        or commercialization_terms.get("operator_revenue_terms")
        or commercialization_terms.get("revenue_share_terms")
        or commercialization_terms.get("revenue_share")
    )
    exclusivity_terms = _mapping(
        capture_rights.get("exclusivity_terms")
        or capture_rights.get("exclusivityTerms")
        or commercialization_terms.get("exclusivity_terms")
        or commercialization_terms.get("exclusivity")
    )
    sim_eval_scope_allowed = (
        "mujoco_g1_simulator_evaluation_for_this_staged_capture" in consent_scope
        or _bool(capture_rights.get("external_use_allowed"))
        or _bool(capture_rights.get("externalUseAllowed"))
    )
    descriptor_rights_status = ""
    if consent_status in {"accepted", "approved", "documented"} and sim_eval_scope_allowed:
        descriptor_rights_status = "scoped_simulator_eval_approved"
    rights_status = _string(
        rights_summary.get("status")
        or rights_summary.get("rights_status")
        or rights_review.get("status")
        or rights_review.get("rights_status")
        or descriptor_rights_status
        or "missing"
    ).lower()
    descriptor_privacy_processing = _mapping(descriptor_metadata.get("privacy_processing"))
    descriptor_worldlabs_input = _mapping(descriptor_metadata.get("worldlabs_input_audit"))
    descriptor_input_labeling = _mapping(descriptor_worldlabs_input.get("input_labeling"))
    descriptor_privacy_safe = (
        _bool(descriptor_worldlabs_input.get("privacy_safe_input"))
        or _bool(descriptor_input_labeling.get("privacy_safe_input"))
    ) and not (
        _bool(descriptor_worldlabs_input.get("raw_video_bypass_used"))
        or _bool(descriptor_input_labeling.get("raw_video_bypass_used"))
    )
    privacy_status = _string(
        privacy_manifest.get("status")
        or descriptor.get("privacy_status")
        or descriptor.get("privacyStatus")
        or descriptor_privacy_processing.get("status")
        or raw_manifest.get("privacy_status")
        or ("privacy_safe_input" if descriptor_privacy_safe else "")
        or "missing"
    ).lower()
    rights_blocked = rights_status in {
        "missing",
        "blocked",
        "not_allowed",
        "permission_required",
        "failed",
        "revoked",
        "withdrawn",
        "rescinded",
    }
    rights_blocked = rights_blocked or consent_revoked
    privacy_blocked = privacy_status in {"blocked", "failed", "unsafe", "not_allowed"}
    return {
        "rights_status": rights_status,
        "privacy_status": privacy_status,
        "blocked": rights_blocked or privacy_blocked,
        "policy": "rights_privacy_must_be_current_before_external_use_or_public_claim",
        "external_use_allowed": sim_eval_scope_allowed,
        "scope_limited_to_simulator_eval": rights_status == "scoped_simulator_eval_approved",
        "consent_scope": consent_scope,
        "permission_document_uri": _string(
            capture_rights.get("permission_document_uri")
            or capture_rights.get("permissionDocumentUri")
            or capture_rights.get("evidence_uri")
        )
        or None,
        "consent_revoked": consent_revoked,
        "consent_revoked_at": consent_revoked_at or None,
        "revocation_takedown_required": consent_revoked,
        "commercialization_terms": commercialization_terms,
        "revenue_share_terms": revenue_share_terms,
        "exclusivity_terms": exclusivity_terms,
    }


def _rights_field(*payloads: Mapping[str, Any], key: str, fallback: Any = None) -> Any:
    for payload in payloads:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return fallback


def _rights_records(
    *,
    rights_summary: Mapping[str, Any],
    rights_review: Mapping[str, Any],
    privacy_manifest: Mapping[str, Any],
    rights_privacy: Mapping[str, Any],
    source_artifacts: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    evidence_uri = _rights_field(
        rights_review,
        rights_summary,
        key="evidence_uri",
        fallback=rights_privacy.get("permission_document_uri")
        or source_artifacts.get("rights_provenance_review")
        or source_artifacts.get("rights_and_compliance_summary"),
    )
    approver = _rights_field(
        rights_review,
        rights_summary,
        key="approver",
        fallback=_rights_field(rights_review, rights_summary, key="approved_by", fallback=None),
    )
    expiration = _rights_field(
        rights_review,
        rights_summary,
        key="expiration_at",
        fallback=_rights_field(rights_review, rights_summary, key="expires_at", fallback=None),
    )
    blocked = _bool(rights_privacy.get("blocked"))
    blocker_status = "blocked_rights_privacy" if blocked else "needs_use_specific_approval"
    commercialization_terms = _mapping(rights_privacy.get("commercialization_terms"))
    revenue_share_terms = _mapping(rights_privacy.get("revenue_share_terms"))
    exclusivity_terms = _mapping(rights_privacy.get("exclusivity_terms"))
    categories = [
        {
            "rights_scope": "raw_confidential_data",
            "allowed_uses": ["internal_quality_review", "secure_pipeline_processing"],
            "disallowed_uses": ["public_display", "external_robot_team_distribution"],
        },
        {
            "rights_scope": "derived_deidentified_environment",
            "allowed_uses": ["advisory_site_card_generation", "hosted_review_when_approved"],
            "disallowed_uses": ["unrestricted_resale", "identity_reconstruction"],
        },
        {
            "rights_scope": "synthetic_variant_rights",
            "allowed_uses": ["representative_mock_scenario_review"],
            "disallowed_uses": ["claiming_generated_variants_as_capture_truth"],
        },
        {
            "rights_scope": "robot_eval_rights",
            "allowed_uses": ["task_scenario_eval_dataset_review"],
            "disallowed_uses": ["rank_fidelity_claim_without_owner_system_proof"],
        },
        {
            "rights_scope": "commercial_licensing",
            "allowed_uses": ["request_scoped_license_review"],
            "disallowed_uses": ["blanket_commercial_license_claim"],
        },
        {
            "rights_scope": "revenue_share",
            "allowed_uses": ["request_scoped_revenue_share_review"],
            "disallowed_uses": ["payout_or_revenue_share_commitment_without_owner_record"],
        },
        {
            "rights_scope": "exclusivity_limits",
            "allowed_uses": ["record_non_exclusive_default_or_review_needed"],
            "disallowed_uses": ["early_exclusivity_claim_without_signed_approval"],
        },
    ]
    records: List[Dict[str, Any]] = []
    for category in categories:
        scope = _string(category.get("rights_scope"))
        records.append(
            {
                "rights_record_id": f"rights_{scope}",
                "rights_scope": scope,
                "status": "blocked" if blocked else "review_required",
                "blocker_status": blocker_status,
                "raw_confidential_data": scope == "raw_confidential_data",
                "derived_deidentified_environment": (
                    scope == "derived_deidentified_environment"
                ),
                "synthetic_variant_rights": scope == "synthetic_variant_rights",
                "robot_eval_rights": scope == "robot_eval_rights",
                "commercial_licensing": scope == "commercial_licensing",
                "revenue_share": scope == "revenue_share",
                "exclusivity_limits": scope == "exclusivity_limits",
                "expiration_at": expiration,
                "approver": approver,
                "evidence_uri": evidence_uri,
                "allowed_uses": list(category["allowed_uses"]),
                "disallowed_uses": list(category["disallowed_uses"]),
                "source_status": {
                    "rights_status": rights_privacy.get("rights_status"),
                    "privacy_status": rights_privacy.get("privacy_status"),
                    "privacy_processing_status": privacy_manifest.get("status"),
                    "consent_revoked": rights_privacy.get("consent_revoked"),
                },
                "commercialization_terms": commercialization_terms
                if scope == "commercial_licensing"
                else {},
                "operator_revenue_terms": revenue_share_terms
                if scope == "revenue_share"
                else {},
                "exclusivity_terms": exclusivity_terms
                if scope == "exclusivity_limits"
                else {},
                "terms_record_present": bool(
                    (scope == "commercial_licensing" and commercialization_terms)
                    or (scope == "revenue_share" and revenue_share_terms)
                    or (scope == "exclusivity_limits" and exclusivity_terms)
                ),
                "revocation_takedown_required": _bool(
                    rights_privacy.get("revocation_takedown_required")
                ),
                "claim_boundary": "rights_record_is_review_packet_not_blanket_clearance",
            }
        )
    return records


def _rights_packet(
    *,
    rights_summary: Mapping[str, Any],
    rights_review: Mapping[str, Any],
    privacy_manifest: Mapping[str, Any],
    rights_privacy: Mapping[str, Any],
    source_artifacts: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    records = _rights_records(
        rights_summary=rights_summary,
        rights_review=rights_review,
        privacy_manifest=privacy_manifest,
        rights_privacy=rights_privacy,
        source_artifacts=source_artifacts,
    )
    commercialization_terms = _mapping(rights_privacy.get("commercialization_terms"))
    revenue_share_terms = _mapping(rights_privacy.get("revenue_share_terms"))
    exclusivity_terms = _mapping(rights_privacy.get("exclusivity_terms"))
    revenue_share_record_present = bool(revenue_share_terms)
    blocked = _bool(rights_privacy.get("blocked"))
    consent_revoked = _bool(rights_privacy.get("consent_revoked"))
    revocation_takedown_required = _bool(
        rights_privacy.get("revocation_takedown_required")
    )
    return {
        "schema_version": RIGHTS_PACKET_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked" if blocked else "review_required",
        "record_count": len(records),
        "records": records,
        "source_artifacts": dict(source_artifacts),
        "consent_revoked": consent_revoked,
        "consent_revoked_at": rights_privacy.get("consent_revoked_at"),
        "revocation_takedown": {
            "schema_version": "real_site_robot_eval_revocation_takedown.v1",
            "status": "takedown_required"
            if revocation_takedown_required
            else "not_required",
            "consent_revoked": consent_revoked,
            "consent_revoked_at": rights_privacy.get("consent_revoked_at"),
            "affected_surfaces": [
                "robot_eval_dataset",
                "post_training_data_package",
                "hosted_review_assets",
                "webapp_projection",
            ],
        },
        "revenue_share_review": {
            "schema_version": "real_site_robot_eval_revenue_share_review.v1",
            "status": "recorded_review_required"
            if revenue_share_record_present
            else "review_required",
            "required_before_paid_reuse_or_resale": True,
            "owner_revenue_share_record_present": revenue_share_record_present,
            "operator_revenue_terms": revenue_share_terms,
            "commercialization_terms": commercialization_terms,
            "exclusivity_terms": exclusivity_terms,
            "revenue_share_commitment_made": False,
            "payout_commitment_allowed": False,
            "claim_boundary": (
                "operator_revenue_terms_are_review_metadata_not_payment_or_resale_clearance"
            ),
        },
        "commercial_use_claim_allowed": False,
        "external_licensing_claim_allowed": False,
        "blocker_status": "blocked_rights_privacy"
        if blocked
        else "needs_use_specific_approval",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _rights_ledger(*, rights_packet: Mapping[str, Any], generated_at: str) -> Dict[str, Any]:
    records = [
        dict(record)
        for record in rights_packet.get("records", [])
        if isinstance(record, Mapping)
    ]
    return {
        "schema_version": RIGHTS_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "ledger_status": "blocked"
        if rights_packet.get("status") == "blocked"
        else "review_required",
        "record_count": len(records),
        "records": records,
        "allowed_disallowed_use_matrix": {
            _string(record.get("rights_scope")): {
                "allowed_uses": _string_list(record.get("allowed_uses")),
                "disallowed_uses": _string_list(record.get("disallowed_uses")),
                "blocker_status": record.get("blocker_status"),
            }
            for record in records
        },
        "claim_boundary": "rights_ledger_does_not_clear_public_or_commercial_use_by_itself",
    }


def _dataset_statuses(
    *,
    task_count: int,
    scenario_count: int,
    robot_pov_input: Mapping[str, Any],
    human_demo_input: Mapping[str, Any],
    action_log_input: Mapping[str, Any],
    actual_outcome_input: Mapping[str, Any],
    robot_team_submission_input: Mapping[str, Any],
    rights_privacy: Mapping[str, Any],
) -> List[str]:
    statuses: List[str] = []
    if task_count > 0 and scenario_count > 0:
        statuses.append("capture_grounded_ready")
    if not robot_pov_input:
        statuses.append("needs_robot_pov")
    if not human_demo_input:
        statuses.append("needs_human_demo")
    if not action_log_input:
        statuses.append("needs_action_logs")
    if not actual_outcome_input:
        statuses.append("needs_actual_outcome")
    statuses.extend(_robot_team_submission_modality_statuses(robot_team_submission_input))
    if bool(rights_privacy.get("blocked")):
        statuses.append("blocked_rights_privacy")
    statuses.append("review_only_no_rank_fidelity")
    return [status for status in FAIL_CLOSED_STATUSES if status in set(statuses)]


def _methodology_summary(
    *,
    manifest: Mapping[str, Any],
    task_library: Mapping[str, Any],
    scenario_library: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> str:
    statuses = ", ".join(_string_list(manifest.get("dataset_statuses")))
    task_count = int(task_library.get("task_count") or 0)
    scenario_count = int(scenario_library.get("scenario_count") or 0)
    record_count = int(ledger.get("record_count") or 0)
    return "\n".join(
        [
            "# Real-Site Robot Evaluation Dataset Methodology",
            "",
            "Status: repo-local deterministic contract. No live provider jobs, simulator runs, "
            "model downloads, sends, payments, deployments, or public-claim upgrades were performed.",
            "",
            "## Scope",
            "",
            "This dataset layer defines robot tasks, scenario records, evidence requirements, "
            "failure labels, and prediction-vs-actual ledger fields for one capture-backed site "
            "package. It is advisory until actual robot POV, human demo, action-log, rights/privacy, "
            "and outcome evidence exists.",
            "",
            "## Current Counts",
            "",
            f"- Tasks: {task_count}",
            f"- Scenarios: {scenario_count}",
            f"- Prediction/outcome records: {record_count}",
            f"- Dataset statuses: {statuses}",
            "",
            "## Evaluation Method",
            "",
            "1. Define task records from `evaluation_prep/task_anchor_manifest.json`.",
            "2. Pair each task with available robot profiles to form scenario records.",
            "3. Attach local review sources such as simready, Marble, or Cosmos preflight as "
            "prediction inputs only.",
            "4. Require robot POV, human-demo, action-log, and actual-outcome records before "
            "calibration or operational conclusions.",
            "5. Use `failure_taxonomy.json` IDs for every failed or ambiguous attempt.",
            "6. Keep WebApp display advisory-only unless owner-system proof supports a stronger "
            "request-scoped claim.",
            "",
            "## Blocked Claim Boundary",
            "",
            "This artifact does not prove simulator execution, generated-world rank fidelity, off-scope validation, "
            "provider execution, or deployment outcomes.",
            "",
        ]
    )


def build_real_site_robot_eval_dataset(
    *,
    capture_root: str | Path,
    object_geometry_manifest: Optional[Mapping[str, Any]] = None,
    task_anchor_manifest: Optional[Mapping[str, Any]] = None,
    site_world_spec: Optional[Mapping[str, Any]] = None,
    hosted_session_runtime_manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    eval_dir = pipeline_dir / "evaluation_prep"
    robot_eval_dir = pipeline_dir / "robot_eval_dataset"
    ensure_dir(robot_eval_dir)

    descriptor = _read_optional_mapping(context.descriptor_path)
    raw_manifest = _read_optional_mapping(context.raw_root / "manifest.json")
    object_geometry = dict(
        object_geometry_manifest
        or _read_optional_mapping(eval_dir / "object_geometry_manifest.json")
    )
    if not isinstance(object_geometry.get("objects"), list) or not object_geometry.get("objects"):
        object_geometry = _object_geometry_from_scene_assets(pipeline_dir) or object_geometry
    task_anchor = dict(
        task_anchor_manifest
        or _read_optional_mapping(eval_dir / "task_anchor_manifest.json")
    )
    if not isinstance(task_anchor.get("tasks"), list) or not task_anchor.get("tasks"):
        task_anchor = _task_anchor_from_simulation_automation(pipeline_dir) or task_anchor
    site_world = dict(site_world_spec or _read_optional_mapping(eval_dir / "site_world_spec.json"))
    hosted_manifest = dict(
        hosted_session_runtime_manifest
        or _read_optional_mapping(eval_dir / "hosted_session_runtime_manifest.json")
    )
    simready_scene = _read_optional_mapping(pipeline_dir / "simready" / "simready_scene_manifest.json")
    marble_bridge = _read_optional_mapping(
        pipeline_dir / "marble_sim_assets" / "marble_simready_bridge.json"
    )
    marble_validation = _read_optional_mapping(
        pipeline_dir / "marble_sim_assets" / "marble_asset_validation.json"
    )
    worldlabs_world_manifest = _read_optional_mapping(pipeline_dir / "worldlabs_world_manifest.json")
    cpu_preflight_scorecard = _read_optional_mapping(
        pipeline_dir / "simulation_automation" / "cpu_preflight_scorecard.json"
    )
    cosmos3_readiness = _read_optional_mapping(
        pipeline_dir / "cosmos3_readiness" / "cosmos3_capture_grounded_readiness.json"
    )
    protected_regions_manifest = _read_optional_mapping(eval_dir / "protected_regions_manifest.json")
    rights_summary = _read_optional_mapping(pipeline_dir / "rights_and_compliance_summary.json")
    rights_review = _read_optional_mapping(pipeline_dir / "rights_provenance_review.json")
    privacy_manifest = _read_optional_mapping(pipeline_dir / "privacy_processing_manifest.json")
    robot_pov_input = _read_optional_mapping(
        pipeline_dir / "robot_eval_inputs" / "robot_pov_evidence_manifest.json"
    )
    human_demo_input = _read_optional_mapping(
        pipeline_dir / "robot_eval_inputs" / "human_demo_evidence_manifest.json"
    )
    action_log_input = _read_optional_mapping(
        pipeline_dir / "robot_eval_inputs" / "action_log_manifest.json"
    )
    recorded_trace_input = _read_optional_mapping(
        pipeline_dir / "robot_eval_inputs" / "recorded_action_trace_manifest.json"
    )
    actual_outcome_input = _read_optional_mapping(
        pipeline_dir / "robot_eval_inputs" / "actual_outcome_manifest.json"
    )
    robot_team_submission_input = _read_optional_mapping(
        pipeline_dir / "robot_eval_inputs" / "robot_team_test_submission_manifest.json"
    )

    generated_at = _deterministic_generated_at(
        task_anchor,
        site_world,
        simready_scene,
        marble_bridge,
        marble_validation,
        worldlabs_world_manifest,
        cpu_preflight_scorecard,
        cosmos3_readiness,
        robot_team_submission_input,
        recorded_trace_input,
        actual_outcome_input,
        descriptor,
    )
    task_ontology = _task_ontology_v1(generated_at=generated_at)
    task_library = _task_library(
        task_anchor_manifest=task_anchor,
        object_geometry_manifest=object_geometry,
        generated_at=generated_at,
    )
    robot_profiles = _robot_profiles(
        site_world_spec=site_world,
        hosted_session_runtime_manifest=hosted_manifest,
    )
    prediction_sources = _available_prediction_sources(
        simready_scene_manifest=simready_scene,
        marble_simready_bridge=marble_bridge,
        cosmos3_readiness=cosmos3_readiness,
    )
    scenario_library = _scenario_library(
        task_library=task_library,
        robot_profiles=robot_profiles,
        prediction_sources=prediction_sources,
        generated_at=generated_at,
    )
    scenario_family_library = _scenario_family_library(
        scenario_library=scenario_library,
        generated_at=generated_at,
    )
    source_artifacts = _source_artifacts(
        pipeline_dir=pipeline_dir,
        eval_dir=eval_dir,
        robot_eval_dir=robot_eval_dir,
    )
    robot_pov_requirements = _robot_pov_requirements(generated_at=generated_at)
    human_demo_requirements = _human_demo_requirements(generated_at=generated_at)
    evidence_contract = _robot_eval_inputs_evidence_contract(generated_at=generated_at)
    robot_team_submission_modalities = _robot_team_submission_modalities(
        robot_team_submission_input=robot_team_submission_input,
        generated_at=generated_at,
    )
    failure_taxonomy = _failure_taxonomy(generated_at=generated_at)
    ledger = _prediction_outcome_ledger(
        scenario_library=scenario_library,
        prediction_sources=prediction_sources,
        source_artifacts=source_artifacts,
        generated_at=generated_at,
    )
    scoring_methodology = _scoring_methodology(generated_at=generated_at)
    task_thresholds = _task_thresholds(
        task_library=task_library,
        scoring_methodology=scoring_methodology,
        generated_at=generated_at,
    )
    recorded_trace_eval_report = _recorded_trace_eval_report(
        scenario_library=scenario_library,
        scoring_methodology=scoring_methodology,
        recorded_trace_input=recorded_trace_input,
        source_artifacts=source_artifacts,
        generated_at=generated_at,
    )
    prediction_vs_actual_summary = _prediction_vs_actual_summary(
        ledger=ledger,
        actual_outcome_input=actual_outcome_input,
        recorded_trace_eval_report=recorded_trace_eval_report,
        generated_at=generated_at,
    )
    rights_privacy = _rights_privacy_status(
        rights_summary=rights_summary,
        rights_review=rights_review,
        privacy_manifest=privacy_manifest,
        descriptor=descriptor,
        raw_manifest=raw_manifest,
        # Live consent re-read at dataset emit (revoke-after-manifest guard).
        capture_root=capture_root,
    )
    rights_packet = _rights_packet(
        rights_summary=rights_summary,
        rights_review=rights_review,
        privacy_manifest=privacy_manifest,
        rights_privacy=rights_privacy,
        source_artifacts=source_artifacts,
        generated_at=generated_at,
    )
    rights_ledger = _rights_ledger(
        rights_packet=rights_packet,
        generated_at=generated_at,
    )
    dataset_statuses = _dataset_statuses(
        task_count=int(task_library["task_count"]),
        scenario_count=int(scenario_library["scenario_count"]),
        robot_pov_input=robot_pov_input,
        human_demo_input=human_demo_input,
        action_log_input=action_log_input or recorded_trace_input,
        actual_outcome_input=actual_outcome_input,
        robot_team_submission_input=robot_team_submission_input,
        rights_privacy=rights_privacy,
    )
    dataset_state = (
        "blocked"
        if "blocked_rights_privacy" in dataset_statuses
        else "capture_grounded_review_ready"
        if "capture_grounded_ready" in dataset_statuses
        else "incomplete"
    )
    site_card = _site_card(
        context=context,
        descriptor=descriptor,
        raw_manifest=raw_manifest,
        site_world_spec=site_world,
        object_geometry_manifest=object_geometry,
        task_library=task_library,
        source_artifacts=source_artifacts,
        simready_scene_manifest=simready_scene,
        marble_validation=marble_validation,
        marble_bridge=marble_bridge,
        worldlabs_world_manifest=worldlabs_world_manifest,
        cpu_preflight_scorecard=cpu_preflight_scorecard,
        protected_regions_manifest=protected_regions_manifest,
        rights_privacy=rights_privacy,
        generated_at=generated_at,
    )
    task_cards = _task_cards(task_library=task_library, generated_at=generated_at)
    scenario_cards = _scenario_cards(
        scenario_library=scenario_library,
        generated_at=generated_at,
    )
    eval_cards = _eval_cards(ledger=ledger, generated_at=generated_at)
    collider_present = (
        _mapping(_mapping(site_card.get("geometry")).get("collider")).get("status")
        == "review_input_present"
    )
    collider_backend = _mapping(_mapping(site_card.get("geometry")).get("collider"))
    annotation_backlog = _annotation_backlog(
        site_card=site_card,
        task_cards=task_cards,
        scenario_cards=scenario_cards,
        dataset_statuses=dataset_statuses,
        generated_at=generated_at,
    )
    proof_boundaries = _proof_boundaries(
        rights_privacy=rights_privacy,
        collider_present=collider_present,
        collider_backend_labels=_string_list(collider_backend.get("backend_labels")),
        collider_backend_blockers=_string_list(collider_backend.get("backend_blockers")),
        action_log_input=action_log_input or recorded_trace_input,
        actual_outcome_input=actual_outcome_input,
        robot_team_submission_input=robot_team_submission_input,
        generated_at=generated_at,
    )

    output_paths = {
        "robot_eval_dataset_manifest": "robot_eval_dataset_manifest.json",
        "site_card": "site_card.json",
        "task_cards": "task_cards.json",
        "scenario_cards": "scenario_cards.json",
        "eval_cards": "eval_cards.json",
        "annotation_backlog": "annotation_backlog.json",
        "proof_boundaries": "proof_boundaries.json",
        "legacy_real_site_robot_eval_dataset_manifest": "real_site_robot_eval_dataset_manifest.json",
        "robot_task_library": "robot_task_library.json",
        "task_ontology_v1": "task_ontology_v1.json",
        "scenario_library": "scenario_library.json",
        "scenario_family_library": "scenario_family_library.json",
        "robot_pov_evidence_requirements": "robot_pov_evidence_requirements.json",
        "human_demo_evidence_requirements": "human_demo_evidence_requirements.json",
        "robot_eval_inputs_evidence_contract": "robot_eval_inputs_evidence_contract.json",
        "robot_team_test_submission_modalities": "robot_team_test_submission_modalities.json",
        "failure_taxonomy": "failure_taxonomy.json",
        "prediction_outcome_ledger": "prediction_outcome_ledger.json",
        "prediction_vs_actual_summary": "prediction_vs_actual_summary.json",
        "scoring_methodology": "scoring_methodology.json",
        "task_thresholds": "task_thresholds.json",
        "publication_readiness": "publication_readiness.json",
        "recorded_trace_eval_report": "recorded_trace_eval_report.json",
        "policy_eval_report": "policy_eval_report.json",
        "rights_packet": "rights_packet.json",
        "rights_ledger": "rights_ledger.json",
        "eval_methodology_summary": "eval_methodology_summary.md",
        "cpu_preflight_scorecard": "../simulation_automation/cpu_preflight_scorecard.json",
        "episode_spec_manifest": "../simulation_automation/episode_spec_manifest.json",
        "cpu_simulator_preflight_manifest": (
            "../simulation_automation/cpu_simulator_preflight_manifest.json"
        ),
    }
    publication_readiness = _publication_readiness(
        dataset_state=dataset_state,
        dataset_statuses=dataset_statuses,
        output_paths=output_paths,
        task_thresholds=task_thresholds,
        rights_privacy=rights_privacy,
        worldlabs_world_manifest=worldlabs_world_manifest,
        marble_validation=marble_validation,
        generated_at=generated_at,
    )
    manifest: Dict[str, Any] = {
        "schema_version": ROBOT_EVAL_DATASET_V01_SCHEMA_VERSION,
        "compatibility_schema_version": ROBOT_EVAL_DATASET_SCHEMA_VERSION,
        "dataset_version": ROBOT_EVAL_DATASET_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "site_id": _site_id_from_inputs(descriptor, raw_manifest),
        "dataset_state": dataset_state,
        "dataset_statuses": dataset_statuses,
        "task_count": task_library["task_count"],
        "scenario_count": scenario_library["scenario_count"],
        "site_card_count": 1,
        "task_card_count": task_cards["task_card_count"],
        "scenario_card_count": scenario_cards["scenario_card_count"],
        "eval_card_count": eval_cards["eval_card_count"],
        "annotation_backlog_count": annotation_backlog["backlog_count"],
        "robot_profile_count": len(robot_profiles),
        "task_ontology_count": task_ontology["task_count"],
        "scenario_family_count": scenario_family_library["family_count"],
        "robot_team_test_submission_modality_count": robot_team_submission_modalities[
            "modality_count"
        ],
        "robot_team_test_submission_missing_evidence_statuses": robot_team_submission_modalities[
            "missing_evidence_statuses"
        ],
        "prediction_record_count": ledger["record_count"],
        "recorded_trace_eval_status": recorded_trace_eval_report["status"],
        "prediction_vs_actual_status": prediction_vs_actual_summary["status"],
        "rights_packet_status": rights_packet["status"],
        "publication_readiness": {
            "ready_to_evaluate_publishable": publication_readiness[
                "ready_to_evaluate_publishable"
            ],
            "publication_label": publication_readiness["publication_label"],
            "required_artifact_status": publication_readiness["required_artifact_status"],
            "task_thresholds_uri": publication_readiness["task_thresholds_uri"],
            "publication_readiness_uri": publication_readiness[
                "publication_readiness_uri"
            ],
            "missing_required_artifacts": publication_readiness[
                "missing_required_artifacts"
            ],
            "missing_proof_labels": publication_readiness["missing_proof_labels"],
            "external_blockers": publication_readiness["external_blockers"],
        },
        "prediction_sources_available": prediction_sources,
        "rights_privacy": rights_privacy,
        "source_artifacts": source_artifacts,
        "output_artifacts": output_paths,
        "required_fail_closed_statuses": list(FAIL_CLOSED_STATUSES),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "webapp_sync_boundary": {
            "display_scope": "advisory_dataset_contract_and_missing_proof_labels_only",
            "may_display": [
                "dataset_state",
                "task_count",
                "scenario_count",
                "site_card_count",
                "task_card_count",
                "scenario_card_count",
                "eval_card_count",
                "evidence_requirements",
                "failure_taxonomy",
                "robot_team_test_submission_modalities",
                "prediction_outcome_ledger_schema",
                "prediction_vs_actual_summary",
                "rights_packet",
                "rights_ledger",
                "task_ontology_v1",
                "scenario_family_library",
            "scoring_methodology",
                "task_thresholds",
                "publication_readiness",
                "recorded_trace_eval_report",
                "cpu_preflight_scorecard",
                "episode_spec summary",
                "cpu_simulator_preflight status",
                "collider_backend_blockers",
                "missing_proof_statuses",
                "Site/Task/Scenario/Eval Card summaries",
            ],
            "must_not_display_as": [
                "robot_ready",
                "deployment_ready",
                "non_ranking_operational_claim_validated",
                "simulator_completed",
                "actual_outcome_proven",
            ],
        },
    }
    manifest["deterministic_fingerprint"] = _sha_payload(
        {
            "scene_id": manifest["scene_id"],
            "capture_id": manifest["capture_id"],
            "site_id": manifest["site_id"],
            "dataset_state": manifest["dataset_state"],
            "dataset_statuses": manifest["dataset_statuses"],
            "task_library": task_library,
            "task_ontology": task_ontology,
            "scenario_library": scenario_library,
            "scenario_family_library": scenario_family_library,
            "ledger_records": ledger["records"],
            "prediction_vs_actual_summary": prediction_vs_actual_summary,
            "recorded_trace_eval_report": recorded_trace_eval_report,
            "site_card": site_card,
            "task_cards": task_cards,
            "scenario_cards": scenario_cards,
            "eval_cards": eval_cards,
            "annotation_backlog": annotation_backlog,
            "proof_boundaries": proof_boundaries,
            "robot_team_submission_modalities": robot_team_submission_modalities,
            "evidence_contract": evidence_contract,
            "scoring_methodology": scoring_methodology,
            "task_thresholds": task_thresholds,
            "publication_readiness": publication_readiness,
            "rights_packet": rights_packet,
            "rights_ledger": rights_ledger,
            "rights_privacy": rights_privacy,
        }
    )
    methodology_summary = _methodology_summary(
        manifest=manifest,
        task_library=task_library,
        scenario_library=scenario_library,
        ledger=ledger,
    )

    manifest_path = robot_eval_dir / "robot_eval_dataset_manifest.json"
    legacy_manifest_path = robot_eval_dir / "real_site_robot_eval_dataset_manifest.json"
    write_json(robot_eval_dir / "site_card.json", site_card)
    write_json(robot_eval_dir / "task_cards.json", task_cards)
    write_json(robot_eval_dir / "scenario_cards.json", scenario_cards)
    write_json(robot_eval_dir / "eval_cards.json", eval_cards)
    write_json(robot_eval_dir / "annotation_backlog.json", annotation_backlog)
    write_json(robot_eval_dir / "proof_boundaries.json", proof_boundaries)
    write_json(robot_eval_dir / "robot_task_library.json", task_library)
    write_json(robot_eval_dir / "task_ontology_v1.json", task_ontology)
    write_json(robot_eval_dir / "scenario_library.json", scenario_library)
    write_json(robot_eval_dir / "scenario_family_library.json", scenario_family_library)
    write_json(robot_eval_dir / "robot_pov_evidence_requirements.json", robot_pov_requirements)
    write_json(robot_eval_dir / "human_demo_evidence_requirements.json", human_demo_requirements)
    write_json(robot_eval_dir / "robot_eval_inputs_evidence_contract.json", evidence_contract)
    write_json(
        robot_eval_dir / "robot_team_test_submission_modalities.json",
        robot_team_submission_modalities,
    )
    write_json(robot_eval_dir / "failure_taxonomy.json", failure_taxonomy)
    write_json(robot_eval_dir / "prediction_outcome_ledger.json", ledger)
    write_json(robot_eval_dir / "prediction_vs_actual_summary.json", prediction_vs_actual_summary)
    write_json(robot_eval_dir / "scoring_methodology.json", scoring_methodology)
    write_json(robot_eval_dir / "task_thresholds.json", task_thresholds)
    write_json(robot_eval_dir / "publication_readiness.json", publication_readiness)
    write_json(robot_eval_dir / "recorded_trace_eval_report.json", recorded_trace_eval_report)
    write_json(robot_eval_dir / "policy_eval_report.json", recorded_trace_eval_report)
    write_json(robot_eval_dir / "rights_packet.json", rights_packet)
    write_json(robot_eval_dir / "rights_ledger.json", rights_ledger)
    write_text(robot_eval_dir / "eval_methodology_summary.md", methodology_summary)
    write_json(manifest_path, manifest)
    write_json(legacy_manifest_path, manifest)

    return {
        "schema_version": "real_site_robot_eval_dataset_result.v1",
        "capture_root": str(context.capture_root),
        "status": dataset_state,
        "dataset_statuses": dataset_statuses,
        "recorded_trace_eval_status": recorded_trace_eval_report["status"],
        "prediction_vs_actual_status": prediction_vs_actual_summary["status"],
        "rights_packet_status": rights_packet["status"],
        "manifest_path": str(manifest_path.resolve()),
        "legacy_manifest_path": str(legacy_manifest_path.resolve()),
        "site_card_path": str((robot_eval_dir / "site_card.json").resolve()),
        "task_cards_path": str((robot_eval_dir / "task_cards.json").resolve()),
        "scenario_cards_path": str((robot_eval_dir / "scenario_cards.json").resolve()),
        "eval_cards_path": str((robot_eval_dir / "eval_cards.json").resolve()),
        "annotation_backlog_path": str((robot_eval_dir / "annotation_backlog.json").resolve()),
        "proof_boundaries_path": str((robot_eval_dir / "proof_boundaries.json").resolve()),
        "methodology_path": str((robot_eval_dir / "eval_methodology_summary.md").resolve()),
        "prediction_outcome_ledger_path": str(
            (robot_eval_dir / "prediction_outcome_ledger.json").resolve()
        ),
        "prediction_vs_actual_summary_path": str(
            (robot_eval_dir / "prediction_vs_actual_summary.json").resolve()
        ),
        "recorded_trace_eval_report_path": str(
            (robot_eval_dir / "recorded_trace_eval_report.json").resolve()
        ),
        "task_thresholds_path": str((robot_eval_dir / "task_thresholds.json").resolve()),
        "publication_readiness_path": str(
            (robot_eval_dir / "publication_readiness.json").resolve()
        ),
        "rights_packet_path": str((robot_eval_dir / "rights_packet.json").resolve()),
        "rights_ledger_path": str((robot_eval_dir / "rights_ledger.json").resolve()),
        "robot_team_test_submission_modalities_path": str(
            (robot_eval_dir / "robot_team_test_submission_modalities.json").resolve()
        ),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build repo-local real-site robot evaluation dataset artifacts"
    )
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    args = parser.parse_args(argv)

    try:
        result = build_real_site_robot_eval_dataset(capture_root=args.capture_root)
    except Exception as exc:
        print(f"[robot-eval-dataset] FAILED: {exc}")
        return 1

    print(f"[robot-eval-dataset] manifest={result['manifest_path']}")
    print(f"[robot-eval-dataset] status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
