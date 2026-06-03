"""Repo-local real-site robot evaluation dataset artifact lane.

This module writes deterministic dataset/workflow artifacts for robot task
evaluation without calling providers, running simulators, downloading models, or
claiming deployment readiness.
"""

from __future__ import annotations

import argparse
import json
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

DETERMINISTIC_DEFAULT_GENERATED_AT = "1970-01-01T00:00:00+00:00"

FAIL_CLOSED_STATUSES = [
    "capture_grounded_ready",
    "needs_robot_pov",
    "needs_human_demo",
    "needs_action_logs",
    "needs_actual_outcome",
    "blocked_rights_privacy",
    "review_only_no_robot_readiness",
]

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
    "robot_readiness_proven": False,
    "deployment_outcome_proven": False,
    "safety_validated": False,
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
        "safety_validated",
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
        "actual_outcome_input_manifest": (
            pipeline_dir / "robot_eval_inputs" / "actual_outcome_manifest.json"
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
        objects[object_id] = {
            "object_id": object_id,
            "label": _string(item.get("label") or item.get("class_name") or "object"),
            "task_role": _string(item.get("task_role")),
            "has_collision_hulls": bool(item.get("collision_hulls")),
            "has_support_surfaces": bool(item.get("support_surfaces")),
            "provenance": _mapping(item.get("provenance")),
        }
    return objects


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
            tasks.append(
                {
                    "task_id": task_id,
                    "task_text": _string(task.get("task_text") or task.get("name") or task_id),
                    "task_category": task_category,
                    "target_object_ids": target_ids,
                    "target_objects": [
                        objects[object_id] for object_id in target_ids if object_id in objects
                    ],
                    "articulation_required_ids": _string_list(
                        task.get("articulation_required_ids")
                    ),
                    "start_zone": _float_triplet(task.get("start_zone"), fallback=(0.0, 0.0, 0.0)),
                    "goal_zone": _float_triplet(task.get("goal_zone"), fallback=(0.0, 0.0, 0.0)),
                    "task_critical": bool(task.get("task_critical")),
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
    profiles = list(robot_profiles) or [
        {
            "robot_profile_id": "robot_profile_required",
            "display_name": "Robot profile required",
            "embodiment_type": "unknown",
        }
    ]
    for task in tasks:
        task_id = _string(task.get("task_id"))
        for profile in profiles:
            robot_profile_id = _string(profile.get("robot_profile_id"))
            scenario_id = f"scenario_{_stable_slug(task_id, fallback='task')}_{_stable_slug(robot_profile_id, fallback='robot')}"
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_type": "real_site_robot_task_eval",
                    "task_id": task_id,
                    "robot_profile_id": robot_profile_id,
                    "start_state_id": f"start_{_stable_slug(task_id, fallback='task')}",
                    "target_object_ids": _string_list(task.get("target_object_ids")),
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
        ],
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
        zones.append(
            {
                "task_id": task_id,
                "start_zone": task.get("start_zone"),
                "goal_zone": task.get("goal_zone"),
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
                "location": item.get("placement_bbox") or item.get("boundingBox") or item.get("bbox"),
                "has_collision_hulls": bool(item.get("collision_hulls")),
                "has_support_surfaces": bool(item.get("support_surfaces")),
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
    protected_regions_manifest: Mapping[str, Any],
    rights_privacy: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    metadata = _metadata_from(descriptor, raw_manifest)
    site_type = _first_text(
        site_world_spec.get("site_type"),
        site_world_spec.get("target_site_type"),
        metadata.get("site_type"),
        metadata.get("target_site_type"),
        raw_manifest.get("site_type"),
        fallback="unknown_site_type",
    )
    tasks = [task for task in task_library.get("tasks", []) if isinstance(task, Mapping)]
    collider_ready = _collider_available(
        marble_validation=marble_validation,
        marble_bridge=marble_bridge,
        worldlabs_world_manifest=worldlabs_world_manifest,
    )
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
            "status": "review_input_present" if collider_ready else "blocked_missing_collider",
            "collision_ready_claim_allowed": False,
            "evidence": source_artifacts.get("marble_asset_validation")
            or source_artifacts.get("marble_simready_bridge"),
            "label_source": "marble_asset_validation",
        },
        "scale": {
            "metric_scale_factor": _mapping(
                _mapping(worldlabs_world_manifest.get("assets")).get("semantics_metadata")
            ).get("metric_scale_factor")
            or _mapping(worldlabs_world_manifest.get("semantics_metadata")).get(
                "metric_scale_factor"
            ),
            "status": "present"
            if (
                _mapping(_mapping(worldlabs_world_manifest.get("assets")).get("semantics_metadata"))
                .get("metric_scale_factor")
                is not None
                or _mapping(worldlabs_world_manifest.get("semantics_metadata")).get(
                    "metric_scale_factor"
                )
                is not None
            )
            else "needs_scale_review",
            "label_source": "worldlabs_semantics_metadata",
        },
        "navigation_zones": _task_zone_cards(tasks),
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
            "claim_boundary": "safety_constraints_are_review_inputs_not_safety_validation",
        },
        "robot_metadata": {
            "traversable_routes": metadata.get("traversable_routes") or _task_zone_cards(tasks),
            "robot_pov_camera_paths": metadata.get("robot_pov_camera_paths")
            or "needs_robot_pov",
            "task_zones": _task_zone_cards(tasks),
            "object_locations": _object_location_cards(object_geometry_manifest),
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
        cards.append(
            {
                "schema_version": "real_site_robot_eval_task_card.v0.1",
                "dataset_version": ROBOT_EVAL_DATASET_VERSION,
                "task_card_id": f"task_card_{_stable_slug(task_id, fallback='task')}",
                "task_id": task_id,
                "task_statement": _first_text(task.get("task_text"), fallback=task_id),
                "task_category": _string(task.get("task_category") or "generic"),
                "start_state": {
                    "start_zone": task.get("start_zone"),
                    "label_source": "task_anchor_manifest",
                    "confidence": "derived",
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
                    "cycle_time_seconds",
                    "placement_accuracy",
                    "intervention_rate",
                    "recovery_success",
                ],
                "task_evidence_source": "pipeline/evaluation_prep/task_anchor_manifest.json",
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
        cards.append(
            {
                "schema_version": "real_site_robot_eval_scenario_card.v0.1",
                "dataset_version": ROBOT_EVAL_DATASET_VERSION,
                "scenario_card_id": f"scenario_card_{_stable_slug(scenario_id, fallback='scenario')}",
                "scenario_id": scenario_id,
                "task_id": _string(scenario.get("task_id")),
                "robot_profile_id": _string(scenario.get("robot_profile_id")),
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
                    "variation": "agent_inferred",
                    "edge_case": "agent_inferred",
                },
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
                "uncertainty": "high_until_action_logs_and_actual_outcomes_exist",
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
                    "safety_validation_proven",
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

    if _mapping(_mapping(site_card.get("geometry")).get("collider")).get("status") == "blocked_missing_collider":
        add(
            "missing_collider_blocks_collision_ready_claim",
            "site_card",
            "collider_mesh_or_collision_proxy_review",
            "No collider evidence is present; collision-ready and physics/contact claims must remain blocked.",
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
        if status == "blocked_rights_privacy":
            add("blocked_rights_privacy", "site_card", "rights_privacy_clearance", "External licensing and public-use claims are blocked.")

    for task in task_cards.get("cards", []):
        if isinstance(task, Mapping):
            for annotation in _string_list(task.get("required_missing_annotations")):
                add(
                    f"{_stable_slug(task.get('task_card_id'), fallback='task')}_{annotation}",
                    "task_cards",
                    annotation,
                    "Task Card requires this annotation before stronger eval claims.",
                )
    for scenario in scenario_cards.get("cards", []):
        if isinstance(scenario, Mapping):
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
    action_log_input: Mapping[str, Any],
    actual_outcome_input: Mapping[str, Any],
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
        "safety_validation_proven": False,
        "rights_cleared_external_licensing_proven": False,
        "real_pilot_outcome_proven": bool(actual_outcome_input.get("real_pilot_outcome_proven"))
        and bool(actual_outcome_input.get("owner_system_proof_uri")),
        "generated_scenarios_are_real_world_proof": False,
        "collider_review_input_present": collider_present,
        "action_logs_present": bool(action_log_input),
        "rights_privacy_status": dict(rights_privacy),
        "blocked_upgrades": [
            "collision_ready_claim"
            if not collider_present
            else "collision_ready_claim_requires_simulator_contact_validation",
            "action_policy_eval_claim"
            if not action_log_input
            else "action_policy_eval_claim_requires_owner_system_validation",
            "external_licensing_claim",
            "safety_validated_claim",
            "ready_to_deploy_claim",
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
) -> Dict[str, Any]:
    rights_status = _string(
        rights_summary.get("status")
        or rights_summary.get("rights_status")
        or rights_review.get("status")
        or rights_review.get("rights_status")
        or "missing"
    ).lower()
    privacy_status = _string(privacy_manifest.get("status") or "missing").lower()
    rights_blocked = rights_status in {
        "missing",
        "blocked",
        "not_allowed",
        "permission_required",
        "failed",
    }
    privacy_blocked = privacy_status in {"blocked", "failed", "unsafe", "not_allowed"}
    return {
        "rights_status": rights_status,
        "privacy_status": privacy_status,
        "blocked": rights_blocked or privacy_blocked,
        "policy": "rights_privacy_must_be_current_before_external_use_or_public_claim",
    }


def _dataset_statuses(
    *,
    task_count: int,
    scenario_count: int,
    robot_pov_input: Mapping[str, Any],
    human_demo_input: Mapping[str, Any],
    action_log_input: Mapping[str, Any],
    actual_outcome_input: Mapping[str, Any],
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
    if bool(rights_privacy.get("blocked")):
        statuses.append("blocked_rights_privacy")
    statuses.append("review_only_no_robot_readiness")
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
            "This artifact does not prove simulator execution, robot readiness, safety validation, "
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
    task_anchor = dict(
        task_anchor_manifest
        or _read_optional_mapping(eval_dir / "task_anchor_manifest.json")
    )
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
    actual_outcome_input = _read_optional_mapping(
        pipeline_dir / "robot_eval_inputs" / "actual_outcome_manifest.json"
    )

    generated_at = _deterministic_generated_at(
        task_anchor,
        site_world,
        simready_scene,
        marble_bridge,
        marble_validation,
        worldlabs_world_manifest,
        cosmos3_readiness,
        descriptor,
    )
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
    source_artifacts = _source_artifacts(
        pipeline_dir=pipeline_dir,
        eval_dir=eval_dir,
        robot_eval_dir=robot_eval_dir,
    )
    robot_pov_requirements = _robot_pov_requirements(generated_at=generated_at)
    human_demo_requirements = _human_demo_requirements(generated_at=generated_at)
    failure_taxonomy = _failure_taxonomy(generated_at=generated_at)
    ledger = _prediction_outcome_ledger(
        scenario_library=scenario_library,
        prediction_sources=prediction_sources,
        source_artifacts=source_artifacts,
        generated_at=generated_at,
    )
    rights_privacy = _rights_privacy_status(
        rights_summary=rights_summary,
        rights_review=rights_review,
        privacy_manifest=privacy_manifest,
    )
    dataset_statuses = _dataset_statuses(
        task_count=int(task_library["task_count"]),
        scenario_count=int(scenario_library["scenario_count"]),
        robot_pov_input=robot_pov_input,
        human_demo_input=human_demo_input,
        action_log_input=action_log_input,
        actual_outcome_input=actual_outcome_input,
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
        action_log_input=action_log_input,
        actual_outcome_input=actual_outcome_input,
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
        "scenario_library": "scenario_library.json",
        "robot_pov_evidence_requirements": "robot_pov_evidence_requirements.json",
        "human_demo_evidence_requirements": "human_demo_evidence_requirements.json",
        "failure_taxonomy": "failure_taxonomy.json",
        "prediction_outcome_ledger": "prediction_outcome_ledger.json",
        "eval_methodology_summary": "eval_methodology_summary.md",
    }
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
        "prediction_record_count": ledger["record_count"],
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
                "prediction_outcome_ledger_schema",
                "missing_proof_statuses",
                "Site/Task/Scenario/Eval Card summaries",
            ],
            "must_not_display_as": [
                "robot_ready",
                "deployment_ready",
                "safety_validated",
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
            "scenario_library": scenario_library,
            "ledger_records": ledger["records"],
            "site_card": site_card,
            "task_cards": task_cards,
            "scenario_cards": scenario_cards,
            "eval_cards": eval_cards,
            "annotation_backlog": annotation_backlog,
            "proof_boundaries": proof_boundaries,
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
    write_json(robot_eval_dir / "scenario_library.json", scenario_library)
    write_json(robot_eval_dir / "robot_pov_evidence_requirements.json", robot_pov_requirements)
    write_json(robot_eval_dir / "human_demo_evidence_requirements.json", human_demo_requirements)
    write_json(robot_eval_dir / "failure_taxonomy.json", failure_taxonomy)
    write_json(robot_eval_dir / "prediction_outcome_ledger.json", ledger)
    write_text(robot_eval_dir / "eval_methodology_summary.md", methodology_summary)
    write_json(manifest_path, manifest)
    write_json(legacy_manifest_path, manifest)

    return {
        "schema_version": "real_site_robot_eval_dataset_result.v1",
        "capture_root": str(context.capture_root),
        "status": dataset_state,
        "dataset_statuses": dataset_statuses,
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
