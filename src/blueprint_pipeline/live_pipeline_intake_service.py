"""Authenticated HTTP intake for live WebApp robot-eval job requests.

The service is a thin wrapper around ``build_live_pipeline_input_intake``. It
accepts a WebApp ``robot_eval_job_request.v1`` payload or queue envelope, accepts
job-specific policy packages, real robot POV evidence, deployment outcomes, and live closure evidence,
stages validated files into the configured control-plane paths, and optionally
runs a configured trigger command. It does not execute simulator/provider work
or promote proof claims.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .live_pipeline_control_plane import (
    CONTROL_PLANE_OUTPUT_PATH_ENV,
    WEBAPP_JOB_REQUEST_QUEUE_CONTRACT,
    WEBAPP_JOB_REQUEST_SCHEMA_VERSION,
)
from .live_pipeline_input_intake import build_live_pipeline_input_intake


DEFAULT_MANIFEST_PATH = (
    "/var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json"
)
INTAKE_TOKEN_ENV = "BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN"
INTAKE_WORK_DIR_ENV = "BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR"
INTAKE_TRIGGER_ENV = "BLUEPRINT_LIVE_PIPELINE_INTAKE_TRIGGER_COMMAND"
INTAKE_ALLOW_TRIGGER_ENV = "BLUEPRINT_ALLOW_LIVE_PIPELINE_INTAKE_TRIGGER"
INTAKE_OVERWRITE_ENV = "BLUEPRINT_LIVE_PIPELINE_INTAKE_OVERWRITE"
INTAKE_ALLOW_PER_REQUEST_CAPTURE_ROOT_ENV = (
    "BLUEPRINT_LIVE_PIPELINE_ALLOW_PER_REQUEST_CAPTURE_ROOT"
)
INTAKE_CAPTURE_ROOT_BY_SITE_ENV = "BLUEPRINT_LIVE_PIPELINE_CAPTURE_ROOT_BY_SITE_JSON"
INTAKE_REQUIRE_SIGNED_REQUEST_ENV = (
    "BLUEPRINT_LIVE_PIPELINE_INTAKE_REQUIRE_SIGNED_REQUEST"
)
INTAKE_SIGNING_SECRET_ENV = "BLUEPRINT_LIVE_PIPELINE_INTAKE_SIGNING_SECRET"
INTAKE_NONCE_WINDOW_SECONDS_ENV = (
    "BLUEPRINT_LIVE_PIPELINE_INTAKE_NONCE_WINDOW_SECONDS"
)
DEFAULT_INTAKE_NONCE_WINDOW_SECONDS = 300
INTAKE_SCHEMA_VERSION = "blueprint_live_pipeline_intake_service.v1"
CAPTURE_HANDOFF_SOURCE_KIND = "capture_pipeline_handoff"

# In-memory replay guard: maps a seen nonce to the request timestamp it arrived
# with. Entries older than the bounded window are pruned on each check, so a
# replayed nonce inside the window is rejected while a fresh one passes.
_SEEN_INTAKE_NONCES: Dict[str, float] = {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "on"}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _manifest_path() -> Path:
    return Path(os.getenv(CONTROL_PLANE_OUTPUT_PATH_ENV) or DEFAULT_MANIFEST_PATH).expanduser()


def _work_dir(manifest_path: Path) -> Path:
    configured = _string(os.getenv(INTAKE_WORK_DIR_ENV))
    if configured:
        return Path(configured).expanduser()
    return manifest_path.parent / "incoming_webapp_job_requests"


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return stem[:120] or "webapp-job-request"


def _request_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("queue_contract") == WEBAPP_JOB_REQUEST_QUEUE_CONTRACT:
        return _mapping(payload.get("job_request"))
    if payload.get("schema_version") == WEBAPP_JOB_REQUEST_SCHEMA_VERSION:
        return payload
    return {}


def _candidate_path(payload: Mapping[str, Any], work_dir: Path) -> Path:
    request = _request_from_payload(payload)
    job_id = _string(request.get("job_id") or payload.get("job_id"))
    digest = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return work_dir / f"{_safe_stem(job_id or digest)}-{digest}.json"


def _capture_handoff_candidate_path(payload: Mapping[str, Any], work_dir: Path) -> Path:
    scene_id = _string(payload.get("scene_id") or payload.get("sceneId"))
    capture_id = _string(payload.get("capture_id") or payload.get("captureId"))
    digest = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    stem = "-".join(part for part in (scene_id, capture_id, digest) if part)
    return work_dir / "capture_handoffs" / f"{_safe_stem(stem or digest)}.json"


def _closure_candidate_path(payload: Mapping[str, Any], work_dir: Path) -> Path:
    job_id = _string(
        payload.get("job_id")
        or payload.get("jobId")
        or payload.get("robot_eval_job_id")
        or payload.get("robotEvalJobId")
    )
    digest = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return work_dir / "live_closure_evidence" / f"{_safe_stem(job_id or digest)}-{digest}.json"


def _deployment_outcome_candidate_path(payload: Mapping[str, Any], work_dir: Path) -> Path:
    job_id = _string(
        payload.get("job_id")
        or payload.get("jobId")
        or payload.get("robot_eval_job_id")
        or payload.get("robotEvalJobId")
    )
    digest = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return work_dir / "deployment_outcomes" / f"{_safe_stem(job_id or digest)}-{digest}.json"


def _policy_package_candidate_path(payload: Mapping[str, Any], work_dir: Path) -> Path:
    job_id = _string(
        payload.get("job_id")
        or payload.get("jobId")
        or payload.get("robot_eval_job_id")
        or payload.get("robotEvalJobId")
    )
    digest = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return work_dir / "policy_packages" / f"{_safe_stem(job_id or digest)}-{digest}.json"


def _read_mapping_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _first_string(*values: Any) -> str:
    for value in values:
        text = _string(value)
        if text:
            return text
    return ""


def _list_from_payload(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _cards_from_file(path: Path) -> list[Mapping[str, Any]]:
    payload = read_json_any(path)
    if isinstance(payload, Mapping):
        cards = payload.get("cards")
        return [card for card in _list_from_payload(cards) if isinstance(card, Mapping)]
    return [card for card in _list_from_payload(payload) if isinstance(card, Mapping)]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_timestamp(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _string(value)
    if not text:
        return None
    normalized = text.removesuffix("Z") + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _capture_upload_complete_freshness(capture_root: Path) -> Dict[str, Any]:
    path = capture_root / "raw" / "capture_upload_complete.json"
    if not path.is_file():
        return {"present": False}
    payload = _read_mapping_file(path)
    timestamp = None
    timestamp_source = None
    for key in (
        "capture_upload_completed_at",
        "captureUploadCompletedAt",
        "upload_completed_at",
        "uploadCompletedAt",
        "completed_at",
        "completedAt",
        "uploaded_at",
        "uploadedAt",
        "generated_at",
        "generatedAt",
        "timestamp",
    ):
        timestamp = _parse_timestamp(payload.get(key))
        if timestamp is not None:
            timestamp_source = key
            break
    if timestamp is None:
        timestamp = path.stat().st_mtime
        timestamp_source = "file_mtime"
    return {
        "present": True,
        "path": str(path),
        "sha256": _file_sha256(path),
        "timestamp": timestamp,
        "timestamp_source": timestamp_source,
    }


def _capture_root_ids(capture_root: Path) -> Dict[str, str]:
    descriptor = _read_mapping_file(capture_root / "capture_descriptor.json")
    upload_complete = _read_mapping_file(capture_root / "raw" / "capture_upload_complete.json")
    parts = list(capture_root.parts)
    scene_from_path = ""
    capture_from_path = capture_root.name
    if "scenes" in parts and "captures" in parts:
        scene_index = parts.index("scenes")
        capture_index = parts.index("captures")
        if scene_index + 1 < len(parts):
            scene_from_path = parts[scene_index + 1]
        if capture_index + 1 < len(parts):
            capture_from_path = parts[capture_index + 1]
    return {
        "scene_id": _first_string(
            descriptor.get("scene_id"),
            descriptor.get("sceneId"),
            upload_complete.get("scene_id"),
            upload_complete.get("sceneId"),
            scene_from_path,
        ),
        "capture_id": _first_string(
            descriptor.get("capture_id"),
            descriptor.get("captureId"),
            upload_complete.get("capture_id"),
            upload_complete.get("captureId"),
            capture_from_path,
        ),
    }


def _capture_root_map() -> Dict[str, Path]:
    raw = _string(os.getenv(INTAKE_CAPTURE_ROOT_BY_SITE_ENV))
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    roots: Dict[str, Path] = {}
    for key, value in payload.items():
        text_key = _string(key)
        text_value = _string(value)
        if text_key and text_value:
            roots[text_key] = Path(text_value).expanduser().resolve()
    return roots


def _capture_root_from_handoff_payload(
    *,
    payload: Mapping[str, Any],
    manifest_capture_root: Path | None,
) -> Path | None:
    explicit = _first_string(payload.get("capture_root"), payload.get("captureRoot"))
    if explicit:
        return Path(explicit).expanduser().resolve()
    roots = _capture_root_map()
    lookup_keys = [
        _first_string(payload.get("site_submission_id"), payload.get("siteSubmissionId")),
        _first_string(payload.get("buyer_request_id"), payload.get("buyerRequestId")),
        _first_string(payload.get("capture_job_id"), payload.get("captureJobId")),
        _first_string(payload.get("scene_id"), payload.get("sceneId")),
        _first_string(payload.get("capture_id"), payload.get("captureId")),
        _first_string(payload.get("site_slug"), payload.get("siteSlug")),
    ]
    for key in lookup_keys:
        if key and key in roots:
            return roots[key]
    return manifest_capture_root


def _select_dataset_task(capture_root: Path) -> tuple[Dict[str, Any] | None, list[str]]:
    dataset_dir = capture_root / "pipeline" / "robot_eval_dataset"
    task_cards_path = dataset_dir / "task_cards.json"
    scenario_cards_path = dataset_dir / "scenario_cards.json"
    upload_freshness = _capture_upload_complete_freshness(capture_root)
    blockers: list[str] = []
    if not task_cards_path.is_file():
        blockers.append("robot_eval_task_cards_missing")
    if not scenario_cards_path.is_file():
        blockers.append("robot_eval_scenario_cards_missing")
    if blockers:
        return None, blockers
    task_cards = _cards_from_file(task_cards_path)
    scenario_cards = _cards_from_file(scenario_cards_path)
    if upload_freshness.get("present"):
        upload_timestamp = float(upload_freshness.get("timestamp") or 0.0)
        stale_paths = [
            path.name
            for path in (task_cards_path, scenario_cards_path)
            if path.stat().st_mtime + 0.001 < upload_timestamp
        ]
        if stale_paths:
            blockers.append("robot_eval_dataset_stale_for_capture_upload_complete")
    if not task_cards:
        blockers.append("robot_eval_task_cards_empty")
    if not scenario_cards:
        blockers.append("robot_eval_scenario_cards_empty")
    if blockers:
        return None, blockers
    for task in task_cards:
        task_id = _string(task.get("task_id") or task.get("taskId"))
        if not task_id:
            continue
        scenario = next(
            (
                card
                for card in scenario_cards
                if _string(card.get("task_id") or card.get("taskId")) == task_id
                and _string(card.get("scenario_id") or card.get("scenarioId"))
            ),
            None,
        )
        if scenario is None:
            continue
        return {
            "task_id": task_id,
            "scenario_id": _string(scenario.get("scenario_id") or scenario.get("scenarioId")),
            "task_cards_uri": str(task_cards_path),
            "scenario_cards_uri": str(scenario_cards_path),
            "task_cards_sha256": _file_sha256(task_cards_path),
            "scenario_cards_sha256": _file_sha256(scenario_cards_path),
            "capture_upload_complete": upload_freshness,
            "dataset_fresh_for_capture_upload_complete": not upload_freshness.get("present")
            or "robot_eval_dataset_stale_for_capture_upload_complete" not in blockers,
            "task_card_count": len(task_cards),
            "scenario_card_count": len(scenario_cards),
        }, []
    return None, ["robot_eval_no_task_scenario_pair"]


def _capture_handoff_requests_robot_eval(payload: Mapping[str, Any]) -> bool:
    requested_lanes = {
        _string(item)
        for item in _list_from_payload(payload.get("requested_lanes") or payload.get("requestedLanes"))
        if _string(item)
    }
    requested_outputs = {
        _string(item)
        for item in _list_from_payload(
            payload.get("requested_outputs") or payload.get("requestedOutputs")
        )
        if _string(item)
    }
    return (
        payload.get("robot_eval_dataset_requested") is True
        or payload.get("robotEvalDatasetRequested") is True
        or "robot_eval_dataset" in requested_lanes
        or "task_evaluation_run" in requested_lanes
        or "robot_eval_dataset" in requested_outputs
        or "task_evaluation_run" in requested_outputs
    )


def _capture_handoff_to_webapp_request(
    *,
    payload: Mapping[str, Any],
    capture_root: Path,
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    capture_ids = _capture_root_ids(capture_root)
    handoff_scene_id = _first_string(payload.get("scene_id"), payload.get("sceneId"))
    handoff_capture_id = _first_string(payload.get("capture_id"), payload.get("captureId"))
    site_submission_id = _first_string(
        payload.get("site_submission_id"),
        payload.get("siteSubmissionId"),
    )
    buyer_request_id = _first_string(payload.get("buyer_request_id"), payload.get("buyerRequestId"))
    capture_job_id = _first_string(payload.get("capture_job_id"), payload.get("captureJobId"))
    pipeline_handoff_uri = _first_string(
        payload.get("pipeline_handoff_uri"),
        payload.get("pipelineHandoffUri"),
    )
    capture_descriptor_uri = _first_string(
        payload.get("capture_descriptor_uri"),
        payload.get("captureDescriptorUri"),
    )
    blockers: list[str] = []
    if not _capture_handoff_requests_robot_eval(payload):
        blockers.append("capture_handoff_robot_eval_not_requested")
    if handoff_scene_id and capture_ids["scene_id"] and handoff_scene_id != capture_ids["scene_id"]:
        blockers.append("capture_handoff_scene_id_mismatch")
    if handoff_capture_id and capture_ids["capture_id"] and handoff_capture_id != capture_ids["capture_id"]:
        blockers.append("capture_handoff_capture_id_mismatch")
    for field, value in (
        ("site_submission_id", site_submission_id),
        ("buyer_request_id", buyer_request_id),
        ("capture_job_id", capture_job_id),
    ):
        if not value:
            blockers.append(f"capture_handoff_missing_{field}")
    dataset_selection, dataset_blockers = _select_dataset_task(capture_root)
    blockers.extend(dataset_blockers)
    if blockers:
        return None, {
            "status": "blocked",
            "ready": False,
            "scene_id": handoff_scene_id or capture_ids["scene_id"],
            "capture_id": handoff_capture_id or capture_ids["capture_id"],
            "blockers": blockers,
        }
    assert dataset_selection is not None
    identity_digest_material = {
        "handoff_payload": dict(payload),
        "dataset_selection": dataset_selection,
    }
    digest = sha256(
        json.dumps(identity_digest_material, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    scene_id = handoff_scene_id or capture_ids["scene_id"]
    capture_id = handoff_capture_id or capture_ids["capture_id"]
    job_id = _safe_stem(f"capture-handoff-{scene_id}-{capture_id}-{digest}")
    request = {
        "schema_version": WEBAPP_JOB_REQUEST_SCHEMA_VERSION,
        "job_id": job_id,
        "request_id": job_id,
        "buyer_request_id": buyer_request_id,
        "site_package": {
            "capture_root": str(capture_root),
            "scene_id": scene_id,
            "capture_id": capture_id,
            "site_submission_id": site_submission_id,
            "buyer_request_id": buyer_request_id,
            "capture_job_id": capture_job_id,
            "pipeline_prefix": str(capture_root / "pipeline"),
            "package_uri": str(
                capture_root
                / "pipeline"
                / "robot_eval_dataset"
                / "robot_eval_dataset_manifest.json"
            ),
            "pipeline_handoff_uri": pipeline_handoff_uri or None,
            "capture_descriptor_uri": capture_descriptor_uri or None,
        },
        "owner_system": {
            "name": "BlueprintCapturePipelineIntake",
            "request_id": job_id,
            "buyer_request_id": buyer_request_id,
            "site_submission_id": site_submission_id,
            "capture_job_id": capture_job_id,
            "capture_id": capture_id,
        },
        "source": {
            "system": "BlueprintCapture",
            "source_kind": CAPTURE_HANDOFF_SOURCE_KIND,
            "pipeline_handoff_uri": pipeline_handoff_uri or None,
            "capture_descriptor_uri": capture_descriptor_uri or None,
            "selection_state": {
                "source_kind": CAPTURE_HANDOFF_SOURCE_KIND,
                "scene_id": scene_id,
                "capture_id": capture_id,
                "site_submission_id": site_submission_id,
                "buyer_request_id": buyer_request_id,
                "capture_job_id": capture_job_id,
                "task_id": dataset_selection["task_id"],
                "scenario_id": dataset_selection["scenario_id"],
                "dataset_selection": dataset_selection,
            },
        },
        "source_kind": CAPTURE_HANDOFF_SOURCE_KIND,
        "requested_tasks": [
            {
                "task_id": dataset_selection["task_id"],
                "scenario_ids": [dataset_selection["scenario_id"]],
            }
        ],
        "robot_profile": {"robot_profile_id": "unitree_g1_humanoid"},
        "simulator_preference": {"framework": "mujoco"},
        "policy_package": {
            "policy_api_endpoint": {},
            "docker_container": {},
            "recorded_action_trace": {},
            "high_level_skill_trace": {
                "ordered_skill_sequence": ["walk_to_target"],
                "skill_taxonomy_version": "blueprint_default_test_policy.v1",
                "source_type": "capture_handoff_default_sim_only_policy",
                "confidence_coverage_note": (
                    "Capture handoff synthesized sim-only beta request; does not prove "
                    "robot-team policy execution."
                ),
            },
            "teleop_demo": {},
            "sim_controller_plugin": {},
        },
        "proof_boundary": {
            "capture_handoff_driven_request": True,
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    envelope = {
        "queue_contract": WEBAPP_JOB_REQUEST_QUEUE_CONTRACT,
        "status": "queued_for_pipeline",
        "job_id": job_id,
        "source_kind": CAPTURE_HANDOFF_SOURCE_KIND,
        "capture_handoff": {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "pipeline_handoff_uri": pipeline_handoff_uri or None,
            "capture_descriptor_uri": capture_descriptor_uri or None,
            "robot_eval_dataset_requested": True,
        },
        "job_request": request,
    }
    return envelope, {
        "status": "ready",
        "ready": True,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "job_id": job_id,
        "dataset_selection": dataset_selection,
        "blockers": [],
    }


def _real_robot_pov_candidate_path(payload: Mapping[str, Any], work_dir: Path) -> Path:
    job_id = _string(
        payload.get("job_id")
        or payload.get("jobId")
        or payload.get("robot_eval_job_id")
        or payload.get("robotEvalJobId")
    )
    digest = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return work_dir / "real_robot_pov" / f"{_safe_stem(job_id or digest)}-{digest}.json"


def _redacted_intake_response(
    *,
    candidate_path: Path,
    intake: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> Dict[str, Any]:
    webapp = _mapping(intake.get("webapp_job_request"))
    staging = _mapping(intake.get("webapp_staging"))
    staged_inputs = _mapping(intake.get("staged_inputs"))
    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "status": intake.get("status"),
        "accepted": intake.get("status") == "staged_for_control_plane",
        "generated_at": utc_now_iso(),
        "candidate": {
            "path": str(candidate_path),
            "sha256": webapp.get("sha256"),
        },
        "webapp_job_request": {
            "status": webapp.get("status"),
            "job_id": webapp.get("job_id"),
            "fields_present": webapp.get("fields_present"),
            "missing_fields": webapp.get("missing_fields"),
            "capture_root_matches_control_plane": webapp.get(
                "request_capture_root_matches_control_plane"
            ),
            "blockers": webapp.get("blockers", []),
        },
        "webapp_staging": {
            "status": staging.get("status"),
            "performed": bool(staging.get("performed")),
            "target_path": staging.get("target_path"),
            "blockers": staging.get("blockers", []),
        },
        "staged_inputs": {
            "status": staged_inputs.get("status"),
            "performed": bool(staged_inputs.get("performed")),
            "path": staged_inputs.get("path"),
            "blockers": staged_inputs.get("blockers", []),
        },
        "input_blockers": list(intake.get("input_blockers") or []),
        "trigger": dict(trigger),
        "proof_boundary": {
            "intake_performs_simulator_execution": False,
            "intake_sets_proof_booleans": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _redacted_policy_package_response(
    *,
    candidate_path: Path,
    intake: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> Dict[str, Any]:
    policy = _mapping(intake.get("policy_package"))
    staging = _mapping(intake.get("policy_package_staging"))
    staged_inputs = _mapping(intake.get("staged_inputs"))
    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "status": intake.get("status"),
        "accepted": intake.get("status") == "staged_for_control_plane",
        "generated_at": utc_now_iso(),
        "candidate": {
            "path": str(candidate_path),
            "sha256": policy.get("sha256"),
        },
        "policy_package": {
            "status": policy.get("status"),
            "job_id": policy.get("job_id"),
            "selected_modalities": policy.get("selected_modalities"),
            "blockers": policy.get("blockers", []),
        },
        "policy_package_staging": {
            "status": staging.get("status"),
            "performed": bool(staging.get("performed")),
            "target_path": staging.get("target_path"),
            "blockers": staging.get("blockers", []),
        },
        "staged_inputs": {
            "status": staged_inputs.get("status"),
            "performed": bool(staged_inputs.get("performed")),
            "path": staged_inputs.get("path"),
            "blockers": staged_inputs.get("blockers", []),
        },
        "input_blockers": list(intake.get("input_blockers") or []),
        "trigger": dict(trigger),
        "proof_boundary": {
            "intake_performs_policy_execution": False,
            "intake_sets_proof_booleans": False,
            "robot_policy_execution_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _redacted_real_robot_pov_response(
    *,
    candidate_path: Path,
    intake: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> Dict[str, Any]:
    pov = _mapping(intake.get("real_robot_pov"))
    staging = _mapping(intake.get("real_robot_pov_staging"))
    staged_inputs = _mapping(intake.get("staged_inputs"))
    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "status": intake.get("status"),
        "accepted": intake.get("status") == "staged_for_control_plane",
        "generated_at": utc_now_iso(),
        "candidate": {
            "path": str(candidate_path),
            "sha256": pov.get("sha256"),
        },
        "real_robot_pov": {
            "status": pov.get("status"),
            "job_id": pov.get("job_id"),
            "record_count": pov.get("record_count"),
            "record_ids": pov.get("record_ids"),
            "exact_key_record_count": pov.get("exact_key_record_count"),
            "camera_video_record_count": pov.get("camera_video_record_count"),
            "action_log_record_count": pov.get("action_log_record_count"),
            "timestamp_alignment_record_count": pov.get(
                "timestamp_alignment_record_count"
            ),
            "evidence_record_count": pov.get("evidence_record_count"),
            "missing_exact_key_record_ids": pov.get("missing_exact_key_record_ids"),
            "missing_camera_video_record_ids": pov.get(
                "missing_camera_video_record_ids"
            ),
            "missing_action_log_record_ids": pov.get("missing_action_log_record_ids"),
            "missing_timestamp_alignment_record_ids": pov.get(
                "missing_timestamp_alignment_record_ids"
            ),
            "missing_evidence_record_ids": pov.get("missing_evidence_record_ids"),
            "blockers": pov.get("blockers", []),
        },
        "real_robot_pov_staging": {
            "status": staging.get("status"),
            "performed": bool(staging.get("performed")),
            "target_path": staging.get("target_path"),
            "blockers": staging.get("blockers", []),
        },
        "staged_inputs": {
            "status": staged_inputs.get("status"),
            "performed": bool(staged_inputs.get("performed")),
            "path": staged_inputs.get("path"),
            "blockers": staged_inputs.get("blockers", []),
        },
        "input_blockers": list(intake.get("input_blockers") or []),
        "trigger": dict(trigger),
        "proof_boundary": {
            "intake_performs_robot_execution": False,
            "intake_sets_proof_booleans": False,
            "robot_pov_evidence_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def stage_capture_handoff_for_control_plane(
    *,
    payload: Mapping[str, Any],
    capture_root: str | Path,
    manifest_path: str | Path,
    work_dir: str | Path | None = None,
    overwrite: bool = False,
    staged_inputs_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Convert a capture handoff into a control-plane inbox request.

    This is the non-HTTP form of ``/api/live-pipeline/capture-handoffs`` used by
    the Pub/Sub handoff listener. It stages input pointers only; it does not run
    simulator/provider work or promote proof booleans.
    """

    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    resolved_capture_root = Path(capture_root).expanduser().resolve()
    resolved_work_dir = (
        Path(work_dir).expanduser().resolve()
        if work_dir
        else _work_dir(resolved_manifest_path).resolve()
    )
    ensure_dir(resolved_work_dir)
    handoff_path = _capture_handoff_candidate_path(payload, resolved_work_dir)
    write_json(handoff_path, dict(payload))
    envelope, handoff_audit = _capture_handoff_to_webapp_request(
        payload=payload,
        capture_root=resolved_capture_root,
    )
    if envelope is None:
        return {
            "schema_version": INTAKE_SCHEMA_VERSION,
            "status": "blocked",
            "accepted": False,
            "generated_at": utc_now_iso(),
            "candidate": {"path": str(handoff_path)},
            "capture_handoff": handoff_audit,
            "input_blockers": [
                f"capture_handoff:{blocker}"
                for blocker in handoff_audit.get("blockers", [])
            ],
            "proof_boundary": {
                "capture_handoff_converted_to_job_request": False,
                "intake_performs_robot_execution": False,
                "intake_sets_proof_booleans": False,
                "simulator_execution_proven": False,
                "rank_fidelity_result_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        }

    request_path = _candidate_path(envelope, resolved_work_dir)
    write_json(request_path, envelope)
    intake = build_live_pipeline_input_intake(
        manifest_path=resolved_manifest_path,
        webapp_job_request=request_path,
        stage_webapp_request=True,
        overwrite=overwrite,
        allow_request_capture_root=True,
        staged_inputs_path=staged_inputs_path,
    )
    response = _redacted_intake_response(
        candidate_path=request_path,
        intake=intake,
        trigger={
            "status": "not_run",
            "performed": False,
            "reason": "non_http_pubsub_staging_helper",
        },
    )
    response["capture_handoff"] = {
        **handoff_audit,
        "candidate_path": str(handoff_path),
        "webapp_job_request_candidate_path": str(request_path),
        "converted_to_job_request": True,
    }
    response["proof_boundary"] = {
        **_mapping(response.get("proof_boundary")),
        "capture_handoff_converted_to_job_request": True,
        "capture_handoff_endpoint_directly_runs_simulator": False,
        "pubsub_listener_directly_runs_control_plane": False,
    }
    return response


def _redacted_deployment_outcome_response(
    *,
    candidate_path: Path,
    intake: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> Dict[str, Any]:
    outcomes = _mapping(intake.get("deployment_outcomes"))
    staging = _mapping(intake.get("deployment_outcomes_staging"))
    staged_inputs = _mapping(intake.get("staged_inputs"))
    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "status": intake.get("status"),
        "accepted": intake.get("status") == "staged_for_control_plane",
        "generated_at": utc_now_iso(),
        "candidate": {
            "path": str(candidate_path),
            "sha256": outcomes.get("sha256"),
        },
        "deployment_outcomes": {
            "status": outcomes.get("status"),
            "job_id": outcomes.get("job_id"),
            "record_count": outcomes.get("record_count"),
            "record_ids": outcomes.get("record_ids"),
            "owner_evidence_ready": bool(outcomes.get("owner_evidence_ready")),
            "owner_evidence_record_count": outcomes.get("owner_evidence_record_count"),
            "missing_owner_evidence_record_ids": outcomes.get(
                "missing_owner_evidence_record_ids"
            ),
            "blockers": outcomes.get("blockers", []),
        },
        "deployment_outcomes_staging": {
            "status": staging.get("status"),
            "performed": bool(staging.get("performed")),
            "target_path": staging.get("target_path"),
            "blockers": staging.get("blockers", []),
        },
        "staged_inputs": {
            "status": staged_inputs.get("status"),
            "performed": bool(staged_inputs.get("performed")),
            "path": staged_inputs.get("path"),
            "blockers": staged_inputs.get("blockers", []),
        },
        "input_blockers": list(intake.get("input_blockers") or []),
        "trigger": dict(trigger),
        "proof_boundary": {
            "intake_performs_simulator_execution": False,
            "intake_sets_proof_booleans": False,
            "real_world_outcome_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _redacted_closure_evidence_response(
    *,
    candidate_path: Path,
    intake: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> Dict[str, Any]:
    evidence = _mapping(intake.get("live_closure_evidence"))
    staging = _mapping(intake.get("live_closure_evidence_staging"))
    staged_inputs = _mapping(intake.get("staged_inputs"))
    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "status": intake.get("status"),
        "accepted": intake.get("status") == "staged_for_control_plane",
        "generated_at": utc_now_iso(),
        "candidate": {
            "path": str(candidate_path),
            "sha256": evidence.get("sha256"),
        },
        "live_closure_evidence": {
            "status": evidence.get("status"),
            "job_id": evidence.get("job_id"),
            "sections": evidence.get("sections"),
            "blockers": evidence.get("blockers", []),
        },
        "live_closure_evidence_staging": {
            "status": staging.get("status"),
            "performed": bool(staging.get("performed")),
            "target_path": staging.get("target_path"),
            "blockers": staging.get("blockers", []),
        },
        "staged_inputs": {
            "status": staged_inputs.get("status"),
            "performed": bool(staged_inputs.get("performed")),
            "path": staged_inputs.get("path"),
            "blockers": staged_inputs.get("blockers", []),
        },
        "input_blockers": list(intake.get("input_blockers") or []),
        "trigger": dict(trigger),
        "proof_boundary": {
            "intake_performs_simulator_execution": False,
            "intake_sets_proof_booleans": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _trigger_control_plane() -> Dict[str, Any]:
    command = _string(os.getenv(INTAKE_TRIGGER_ENV))
    allowed = _truthy(os.getenv(INTAKE_ALLOW_TRIGGER_ENV))
    if not command:
        return {
            "status": "not_configured",
            "performed": False,
            "allowed": allowed,
            "command_configured": False,
        }
    if not allowed:
        return {
            "status": "blocked",
            "performed": False,
            "allowed": False,
            "command_configured": True,
            "blockers": [f"missing_env_{INTAKE_ALLOW_TRIGGER_ENV}"],
        }
    completed = subprocess.run(
        command,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "status": "triggered" if completed.returncode == 0 else "failed",
        "performed": completed.returncode == 0,
        "allowed": True,
        "command_configured": True,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _intake_nonce_window_seconds() -> float:
    raw = _string(os.getenv(INTAKE_NONCE_WINDOW_SECONDS_ENV))
    if not raw:
        return float(DEFAULT_INTAKE_NONCE_WINDOW_SECONDS)
    try:
        value = float(raw)
    except ValueError:
        return float(DEFAULT_INTAKE_NONCE_WINDOW_SECONDS)
    return value if value > 0 else float(DEFAULT_INTAKE_NONCE_WINDOW_SECONDS)


def _parse_intake_timestamp(value: Any) -> float | None:
    text = _string(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return _parse_timestamp(text)


def _prune_seen_nonces(now: float, window: float) -> None:
    expired = [
        nonce
        for nonce, seen_at in _SEEN_INTAKE_NONCES.items()
        if now - seen_at > window
    ]
    for nonce in expired:
        _SEEN_INTAKE_NONCES.pop(nonce, None)


def _reset_intake_replay_cache() -> None:
    """Clear the in-memory replay guard (used by tests and on process boundaries)."""

    _SEEN_INTAKE_NONCES.clear()


def _enforce_intake_replay_protection(
    *,
    timestamp_header: str | None,
    nonce_header: str | None,
    signature_header: str | None,
) -> None:
    """Reject expired or replayed intake requests using a nonce + bounded window.

    Backward-compatible: when signed requests are not required and the caller sends
    no timestamp/nonce headers, the check is a no-op so the plain-bearer happy path
    keeps working. Once a request carries replay headers (or the operator opts in
    via env), the timestamp must be inside the window, the nonce must be unused, and
    (if a signing secret is configured) an HMAC signature over ``timestamp.nonce``
    must match with a constant-time compare.
    """

    required = _truthy(os.getenv(INTAKE_REQUIRE_SIGNED_REQUEST_ENV))
    provided_ts = _string(timestamp_header)
    provided_nonce = _string(nonce_header)
    if not required and not provided_ts and not provided_nonce:
        return
    if not provided_ts or not provided_nonce:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="intake request requires timestamp and nonce",
        )
    timestamp = _parse_intake_timestamp(provided_ts)
    if timestamp is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="intake request timestamp is invalid",
        )
    window = _intake_nonce_window_seconds()
    now = datetime.now(timezone.utc).timestamp()
    if abs(now - timestamp) > window:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="intake request timestamp outside allowed window",
        )
    signing_secret = _string(os.getenv(INTAKE_SIGNING_SECRET_ENV))
    if signing_secret:
        expected_signature = hmac.new(
            signing_secret.encode("utf-8"),
            f"{provided_ts}.{provided_nonce}".encode("utf-8"),
            sha256,
        ).hexdigest()
        provided_signature = _string(signature_header)
        if not provided_signature or not hmac.compare_digest(
            provided_signature, expected_signature
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="intake request signature invalid",
            )
    _prune_seen_nonces(now, window)
    if provided_nonce in _SEEN_INTAKE_NONCES:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="intake request nonce already used",
        )
    _SEEN_INTAKE_NONCES[provided_nonce] = now


def _require_token(
    authorization: str | None = Header(default=None),
    x_blueprint_intake_token: str | None = Header(default=None),
    x_blueprint_intake_timestamp: str | None = Header(default=None),
    x_blueprint_intake_nonce: str | None = Header(default=None),
    x_blueprint_intake_signature: str | None = Header(default=None),
) -> None:
    expected = _string(os.getenv(INTAKE_TOKEN_ENV))
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{INTAKE_TOKEN_ENV} is not configured",
        )
    provided = _string(x_blueprint_intake_token)
    if not provided and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            provided = _string(token)
    if not hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid intake token",
        )
    _enforce_intake_replay_protection(
        timestamp_header=x_blueprint_intake_timestamp,
        nonce_header=x_blueprint_intake_nonce,
        signature_header=x_blueprint_intake_signature,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Blueprint Live Pipeline Intake", version=INTAKE_SCHEMA_VERSION)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        manifest_path = _manifest_path()
        return {
            "ok": True,
            "schema_version": INTAKE_SCHEMA_VERSION,
            "manifest_path": str(manifest_path),
            "manifest_exists": manifest_path.is_file(),
            "token_configured": bool(_string(os.getenv(INTAKE_TOKEN_ENV))),
            "trigger_configured": bool(_string(os.getenv(INTAKE_TRIGGER_ENV))),
            "signed_request_required": _truthy(
                os.getenv(INTAKE_REQUIRE_SIGNED_REQUEST_ENV)
            ),
            "request_signing_configured": bool(
                _string(os.getenv(INTAKE_SIGNING_SECRET_ENV))
            ),
            "per_request_capture_root_enabled": _truthy(
                os.getenv(INTAKE_ALLOW_PER_REQUEST_CAPTURE_ROOT_ENV)
            ),
            "endpoints": [
                "/api/live-pipeline/job-requests",
                "/api/live-pipeline/capture-handoffs",
                "/api/live-pipeline/policy-packages",
                "/api/live-pipeline/real-robot-pov",
                "/api/live-pipeline/deployment-outcomes",
                "/api/live-pipeline/live-closure-evidence",
                "/api/live-pipeline/intake-audit",
            ],
            "proof_boundary": {
                "service_is_intake_only": True,
                "simulator_execution_proven": False,
                "rank_fidelity_result_proven": False,
            },
        }

    @app.post("/api/live-pipeline/job-requests", dependencies=[Depends(_require_token)])
    async def intake_job_request(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        work_dir = _work_dir(manifest_path).resolve()
        ensure_dir(work_dir)
        candidate_path = _candidate_path(payload, work_dir)
        write_json(candidate_path, dict(payload))
        intake = build_live_pipeline_input_intake(
            manifest_path=manifest_path,
            webapp_job_request=candidate_path,
            stage_webapp_request=True,
            overwrite=_truthy(os.getenv(INTAKE_OVERWRITE_ENV)),
            allow_request_capture_root=_truthy(
                os.getenv(INTAKE_ALLOW_PER_REQUEST_CAPTURE_ROOT_ENV)
            ),
        )
        trigger = (
            _trigger_control_plane()
            if intake.get("status") == "staged_for_control_plane"
            else {
                "status": "not_run",
                "performed": False,
                "reason": "intake_not_staged_for_control_plane",
            }
        )
        response = _redacted_intake_response(
            candidate_path=candidate_path,
            intake=intake,
            trigger=trigger,
        )
        if intake.get("input_blockers"):
            return JSONResponse(status_code=202, content=response)
        return response

    @app.post("/api/live-pipeline/capture-handoffs", dependencies=[Depends(_require_token)])
    async def intake_capture_handoff(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        manifest_payload = read_json_any(manifest_path)
        if not isinstance(manifest_payload, Mapping):
            raise HTTPException(status_code=503, detail="control-plane manifest is not JSON object")
        manifest_capture_root_text = _string(manifest_payload.get("capture_root"))
        manifest_capture_root = (
            Path(manifest_capture_root_text).resolve()
            if manifest_capture_root_text
            else None
        )
        capture_root = _capture_root_from_handoff_payload(
            payload=payload,
            manifest_capture_root=manifest_capture_root,
        )
        if capture_root is None:
            raise HTTPException(status_code=503, detail="control-plane capture_root missing")
        if not capture_root.is_dir():
            raise HTTPException(
                status_code=503,
                detail=f"capture_root missing for handoff: {capture_root}",
            )
        work_dir = _work_dir(manifest_path).resolve()
        ensure_dir(work_dir)
        handoff_path = _capture_handoff_candidate_path(payload, work_dir)
        write_json(handoff_path, dict(payload))
        envelope, handoff_audit = _capture_handoff_to_webapp_request(
            payload=payload,
            capture_root=capture_root,
        )
        if envelope is None:
            return JSONResponse(
                status_code=202,
                content={
                    "schema_version": INTAKE_SCHEMA_VERSION,
                    "status": "blocked",
                    "accepted": False,
                    "generated_at": utc_now_iso(),
                    "candidate": {"path": str(handoff_path)},
                    "capture_handoff": handoff_audit,
                    "input_blockers": [
                        f"capture_handoff:{blocker}"
                        for blocker in handoff_audit.get("blockers", [])
                    ],
                    "trigger": {
                        "status": "not_run",
                        "performed": False,
                        "reason": "capture_handoff_not_ready",
                    },
                    "proof_boundary": {
                        "capture_handoff_converted_to_job_request": False,
                        "intake_performs_robot_execution": False,
                        "intake_sets_proof_booleans": False,
                        "simulator_execution_proven": False,
                        "rank_fidelity_result_proven": False,
                        "public_claim_upgrade_allowed": False,
                    },
                },
            )
        request_path = _candidate_path(envelope, work_dir)
        write_json(request_path, envelope)
        intake = build_live_pipeline_input_intake(
            manifest_path=manifest_path,
            webapp_job_request=request_path,
            stage_webapp_request=True,
            overwrite=_truthy(os.getenv(INTAKE_OVERWRITE_ENV)),
            allow_request_capture_root=True,
        )
        trigger = (
            _trigger_control_plane()
            if intake.get("status") == "staged_for_control_plane"
            else {
                "status": "not_run",
                "performed": False,
                "reason": "intake_not_staged_for_control_plane",
            }
        )
        response = _redacted_intake_response(
            candidate_path=request_path,
            intake=intake,
            trigger=trigger,
        )
        response["capture_handoff"] = {
            **handoff_audit,
            "candidate_path": str(handoff_path),
            "webapp_job_request_candidate_path": str(request_path),
            "converted_to_job_request": True,
        }
        response["proof_boundary"] = {
            **_mapping(response.get("proof_boundary")),
            "capture_handoff_converted_to_job_request": True,
            "capture_handoff_endpoint_directly_runs_simulator": False,
        }
        if intake.get("input_blockers"):
            return JSONResponse(status_code=202, content=response)
        return response

    @app.post(
        "/api/live-pipeline/policy-packages",
        dependencies=[Depends(_require_token)],
    )
    async def intake_policy_package(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        work_dir = _work_dir(manifest_path).resolve()
        ensure_dir(work_dir)
        candidate_path = _policy_package_candidate_path(payload, work_dir)
        write_json(candidate_path, dict(payload))
        intake = build_live_pipeline_input_intake(
            manifest_path=manifest_path,
            policy_package=candidate_path,
            stage_policy_package=True,
            overwrite=_truthy(os.getenv(INTAKE_OVERWRITE_ENV)),
        )
        trigger = (
            _trigger_control_plane()
            if intake.get("status") == "staged_for_control_plane"
            else {
                "status": "not_run",
                "performed": False,
                "reason": "intake_not_staged_for_control_plane",
            }
        )
        response = _redacted_policy_package_response(
            candidate_path=candidate_path,
            intake=intake,
            trigger=trigger,
        )
        if intake.get("input_blockers"):
            return JSONResponse(status_code=202, content=response)
        return response

    @app.post(
        "/api/live-pipeline/real-robot-pov",
        dependencies=[Depends(_require_token)],
    )
    async def intake_real_robot_pov(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        work_dir = _work_dir(manifest_path).resolve()
        ensure_dir(work_dir)
        candidate_path = _real_robot_pov_candidate_path(payload, work_dir)
        write_json(candidate_path, dict(payload))
        intake = build_live_pipeline_input_intake(
            manifest_path=manifest_path,
            real_robot_pov=candidate_path,
            stage_real_robot_pov=True,
            overwrite=_truthy(os.getenv(INTAKE_OVERWRITE_ENV)),
        )
        trigger = (
            _trigger_control_plane()
            if intake.get("status") == "staged_for_control_plane"
            else {
                "status": "not_run",
                "performed": False,
                "reason": "intake_not_staged_for_control_plane",
            }
        )
        response = _redacted_real_robot_pov_response(
            candidate_path=candidate_path,
            intake=intake,
            trigger=trigger,
        )
        if intake.get("input_blockers"):
            return JSONResponse(status_code=202, content=response)
        return response

    @app.post(
        "/api/live-pipeline/deployment-outcomes",
        dependencies=[Depends(_require_token)],
    )
    async def intake_deployment_outcomes(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        work_dir = _work_dir(manifest_path).resolve()
        ensure_dir(work_dir)
        candidate_path = _deployment_outcome_candidate_path(payload, work_dir)
        write_json(candidate_path, dict(payload))
        intake = build_live_pipeline_input_intake(
            manifest_path=manifest_path,
            deployment_outcomes=candidate_path,
            stage_deployment_outcomes=True,
            overwrite=_truthy(os.getenv(INTAKE_OVERWRITE_ENV)),
        )
        trigger = (
            _trigger_control_plane()
            if intake.get("status") == "staged_for_control_plane"
            else {
                "status": "not_run",
                "performed": False,
                "reason": "intake_not_staged_for_control_plane",
            }
        )
        response = _redacted_deployment_outcome_response(
            candidate_path=candidate_path,
            intake=intake,
            trigger=trigger,
        )
        if intake.get("input_blockers"):
            return JSONResponse(status_code=202, content=response)
        return response

    @app.post(
        "/api/live-pipeline/live-closure-evidence",
        dependencies=[Depends(_require_token)],
    )
    async def intake_live_closure_evidence(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        manifest_path = _manifest_path().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        work_dir = _work_dir(manifest_path).resolve()
        ensure_dir(work_dir)
        candidate_path = _closure_candidate_path(payload, work_dir)
        write_json(candidate_path, dict(payload))
        intake = build_live_pipeline_input_intake(
            manifest_path=manifest_path,
            live_closure_evidence=candidate_path,
            stage_live_closure_evidence=True,
            overwrite=_truthy(os.getenv(INTAKE_OVERWRITE_ENV)),
        )
        trigger = (
            _trigger_control_plane()
            if intake.get("status") == "staged_for_control_plane"
            else {
                "status": "not_run",
                "performed": False,
                "reason": "intake_not_staged_for_control_plane",
            }
        )
        response = _redacted_closure_evidence_response(
            candidate_path=candidate_path,
            intake=intake,
            trigger=trigger,
        )
        if intake.get("input_blockers"):
            return JSONResponse(status_code=202, content=response)
        return response

    @app.get("/api/live-pipeline/intake-audit", dependencies=[Depends(_require_token)])
    def latest_intake_audit() -> Dict[str, Any]:
        manifest_path = _manifest_path().resolve()
        audit_path = manifest_path.parent / "live_pipeline_input_intake_audit.json"
        if not audit_path.is_file():
            raise HTTPException(status_code=404, detail="intake audit not found")
        payload = read_json_any(audit_path)
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=500, detail="intake audit is not a JSON object")
        return {
            "schema_version": INTAKE_SCHEMA_VERSION,
            "audit_path": str(audit_path),
            "status": payload.get("status"),
            "input_blockers": list(payload.get("input_blockers") or []),
            "webapp_job_request": _mapping(payload.get("webapp_job_request")),
            "webapp_staging": _mapping(payload.get("webapp_staging")),
            "policy_package": _mapping(payload.get("policy_package")),
            "policy_package_staging": _mapping(payload.get("policy_package_staging")),
            "real_robot_pov": _mapping(payload.get("real_robot_pov")),
            "real_robot_pov_staging": _mapping(payload.get("real_robot_pov_staging")),
            "deployment_outcomes": _mapping(payload.get("deployment_outcomes")),
            "deployment_outcomes_staging": _mapping(
                payload.get("deployment_outcomes_staging")
            ),
            "live_closure_evidence": _mapping(payload.get("live_closure_evidence")),
            "live_closure_evidence_staging": _mapping(
                payload.get("live_closure_evidence_staging")
            ),
            "staged_inputs": _mapping(payload.get("staged_inputs")),
            "proof_boundary": payload.get("proof_boundary"),
        }

    return app


app = create_app()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the live Pipeline intake HTTP service.")
    parser.add_argument("--host", default=os.getenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8765")))
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run("blueprint_pipeline.live_pipeline_intake_service:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
