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
import json
import os
import re
import subprocess
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
INTAKE_SCHEMA_VERSION = "blueprint_live_pipeline_intake_service.v1"


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
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


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


def _require_token(
    authorization: str | None = Header(default=None),
    x_blueprint_intake_token: str | None = Header(default=None),
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
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid intake token",
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
            "endpoints": [
                "/api/live-pipeline/job-requests",
                "/api/live-pipeline/policy-packages",
                "/api/live-pipeline/real-robot-pov",
                "/api/live-pipeline/deployment-outcomes",
                "/api/live-pipeline/live-closure-evidence",
                "/api/live-pipeline/intake-audit",
            ],
            "proof_boundary": {
                "service_is_intake_only": True,
                "simulator_execution_proven": False,
                "robot_readiness_proven": False,
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
