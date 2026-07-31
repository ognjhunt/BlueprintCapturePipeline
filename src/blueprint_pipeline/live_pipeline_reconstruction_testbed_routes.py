"""Reconstruction and testbed routes for the live Pipeline intake facade.

The HTTP facade owns authentication and service configuration. This module
keeps reconstruction authorization and Pipeline-owned testbed science behind
that facade without allowing ``create_app`` to become an unbounded monolith.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from fastapi import Depends, FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .capture_upload_intake import CAPTURE_UPLOAD_STORE_ROOT_ENV
from .core.security_controls import strict_identifier
from .reconstruction_control_plane import (
    ReconstructionControlPlaneError,
    authorize_reconstruction_plan,
    execute_reconstruction_plan,
    inspect_reconstruction_plan,
    load_reconstruction_compilation_inputs,
    prepare_reconstruction_plan,
    resolve_reconstruction_capture_build,
)
from .site_task_testbed_compilation_contract import (
    validate_testbed_compilation_submission,
)
from .site_task_testbed_compiler import (
    SiteTaskTestbedCompilerError,
    build_pipeline_owned_compilation_support,
    compile_site_task_testbed,
    write_testbed_decision_evidence_request,
    write_testbed_version,
)
from .site_task_testbed_webapp_sync import (
    TESTBED_WEBAPP_SYNC_REQUIRED_ENV,
    sync_site_task_testbed_to_webapp,
)
from .task_candidate_control_plane import (
    TaskCandidateControlPlaneError,
    load_latest_task_candidate_decision_result,
)
from .task_candidate_discovery import compile_approved_task_decision_request
from .task_evaluation_supervisor import (
    capture_supervisor_execution_options_from_env,
    refresh_capture_reconstruction_execution_readiness,
)


def _string(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "on"}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def register_reconstruction_testbed_routes(
    app: FastAPI,
    *,
    require_admission: Callable[..., Any],
    manifest_path_provider: Callable[[], Path],
    work_dir_provider: Callable[[Path], Path],
) -> None:
    """Register authenticated reconstruction and immutable-testbed endpoints."""

    def reconstruction_root(manifest_path: Path) -> Path:
        return (
            work_dir_provider(manifest_path).expanduser().resolve() / "reconstruction_control_plane"
        )

    def maintained_testbed_root(manifest_path: Path) -> Path:
        return (
            work_dir_provider(manifest_path).expanduser().resolve()
            / "maintained_site_task_testbeds"
        )

    async def refresh_reconstruction_readiness(
        *, manifest_path: Path, capture_store_root: Path, plan_id: str
    ) -> dict[str, Any]:
        try:
            options = capture_supervisor_execution_options_from_env()
            capture_build_path = resolve_reconstruction_capture_build(
                state_root=reconstruction_root(manifest_path),
                capture_store_root=capture_store_root,
                plan_id=plan_id,
            )
            inspection = inspect_reconstruction_plan(
                state_root=reconstruction_root(manifest_path),
                plan_id=plan_id,
            )
            return await run_in_threadpool(
                refresh_capture_reconstruction_execution_readiness,
                capture_root=capture_build_path,
                control_plane_inspection=inspection,
                **options,
            )
        except ReconstructionControlPlaneError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=503,
                detail="reconstruction supervisor readiness refresh failed closed",
            ) from exc

    @app.post(
        "/api/live-pipeline/reconstructions/plan",
        dependencies=[Depends(require_admission)],
    )
    async def plan_reconstruction(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        if payload.get("schema_version") != "reconstruction_plan_submission.v1":
            raise HTTPException(
                status_code=422,
                detail="reconstruction plan schema version mismatch",
            )
        claims = payload.get("requested_claim_types")
        if not isinstance(claims, list):
            raise HTTPException(
                status_code=422,
                detail="requested claim types must be a list",
            )
        manifest_path = manifest_path_provider().resolve()
        store_root_text = _string(os.getenv(CAPTURE_UPLOAD_STORE_ROOT_ENV))
        if not manifest_path.is_file() or not store_root_text:
            raise HTTPException(
                status_code=503,
                detail="reconstruction control plane is not configured",
            )
        try:
            result = await run_in_threadpool(
                prepare_reconstruction_plan,
                state_root=reconstruction_root(manifest_path),
                capture_store_root=Path(store_root_text).expanduser().resolve(),
                capture_session_id=str(payload.get("capture_session_id") or ""),
                intake_id=str(payload.get("intake_id") or ""),
                requested_claim_types=[str(row) for row in claims],
                idempotency_key=str(payload.get("idempotency_key") or ""),
            )
            readiness = await refresh_reconstruction_readiness(
                manifest_path=manifest_path,
                capture_store_root=Path(store_root_text).expanduser().resolve(),
                plan_id=str(result["plan_id"]),
            )
            return result | {"task_evaluation_supervisor_readiness": readiness}
        except ReconstructionControlPlaneError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc

    @app.post(
        "/api/live-pipeline/reconstructions/{plan_id}/authorize",
        dependencies=[Depends(require_admission)],
    )
    async def authorize_reconstruction(plan_id: str, request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        if payload.get("schema_version") != "reconstruction_authorization_submission.v1":
            raise HTTPException(
                status_code=422,
                detail="reconstruction authorization schema version mismatch",
            )
        references = payload.get("authorized_adapter_references")
        actor = payload.get("actor")
        if not isinstance(references, list) or not isinstance(actor, Mapping):
            raise HTTPException(
                status_code=422,
                detail="authorization references or actor invalid",
            )
        manifest_path = manifest_path_provider().resolve()
        store_root_text = _string(os.getenv(CAPTURE_UPLOAD_STORE_ROOT_ENV))
        if not store_root_text:
            raise HTTPException(
                status_code=503,
                detail="capture intake store is not configured",
            )
        try:
            result = await run_in_threadpool(
                authorize_reconstruction_plan,
                state_root=reconstruction_root(manifest_path),
                plan_id=plan_id,
                reconstruction_plan_digest=str(payload.get("reconstruction_plan_digest") or ""),
                authorized_adapter_references=[str(row) for row in references],
                actor=dict(actor),
                idempotency_key=str(payload.get("idempotency_key") or ""),
            )
            readiness = await refresh_reconstruction_readiness(
                manifest_path=manifest_path,
                capture_store_root=Path(store_root_text).expanduser().resolve(),
                plan_id=plan_id,
            )
            return result | {"task_evaluation_supervisor_readiness": readiness}
        except ReconstructionControlPlaneError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc

    @app.post(
        "/api/live-pipeline/reconstructions/{plan_id}/execute",
        dependencies=[Depends(require_admission)],
    )
    async def execute_reconstruction(plan_id: str) -> Dict[str, Any]:
        manifest_path = manifest_path_provider().resolve()
        store_root_text = _string(os.getenv(CAPTURE_UPLOAD_STORE_ROOT_ENV))
        if not store_root_text:
            raise HTTPException(
                status_code=503,
                detail="capture intake store is not configured",
            )
        try:
            result = await run_in_threadpool(
                execute_reconstruction_plan,
                state_root=reconstruction_root(manifest_path),
                capture_store_root=Path(store_root_text).expanduser().resolve(),
                plan_id=plan_id,
            )
            readiness = await refresh_reconstruction_readiness(
                manifest_path=manifest_path,
                capture_store_root=Path(store_root_text).expanduser().resolve(),
                plan_id=plan_id,
            )
            return result | {"task_evaluation_supervisor_readiness": readiness}
        except ReconstructionControlPlaneError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc

    @app.get(
        "/api/live-pipeline/reconstructions/{plan_id}",
        dependencies=[Depends(require_admission)],
    )
    async def inspect_reconstruction(plan_id: str) -> Dict[str, Any]:
        manifest_path = manifest_path_provider().resolve()
        try:
            return inspect_reconstruction_plan(
                state_root=reconstruction_root(manifest_path),
                plan_id=plan_id,
            )
        except ReconstructionControlPlaneError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc

    @app.post(
        "/api/live-pipeline/testbeds/compile",
        dependencies=[Depends(require_admission)],
    )
    async def compile_testbed(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        if payload.get("schema_version") != "site_task_testbed_compilation_submission.v2":
            raise HTTPException(
                status_code=422,
                detail=("schema_version:must_be:site_task_testbed_compilation_submission.v2"),
            )
        client_owned_reconstruction_fields = sorted(
            {
                "capture_intake_envelope",
                "capture_qa_report",
                "reconstruction_plan",
                "reconstruction_results",
                "simready_decision",
                "robot_placement_result",
                "artifact_references",
                "supported_condition_ranges",
                "previous_testbed",
            }.intersection(payload)
        )
        if client_owned_reconstruction_fields:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Pipeline-owned scientific inputs forbidden:"
                    + ",".join(client_owned_reconstruction_fields)
                ),
            )
        try:
            payload = validate_testbed_compilation_submission(payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        manifest_path = manifest_path_provider().resolve()
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"control-plane manifest missing: {manifest_path}",
            )
        try:
            session_id = strict_identifier(
                payload.get("capture_session_id"),
                field="capture_session_id",
                max_length=192,
            )
            intake_id = strict_identifier(
                payload.get("intake_id"),
                field="intake_id",
                max_length=192,
            )
            authoritative = load_latest_task_candidate_decision_result(
                state_root=(
                    work_dir_provider(manifest_path).expanduser().resolve()
                    / "task_candidate_control_plane"
                ),
                capture_session_id=session_id,
            )
            approved = _mapping(authoritative.get("approved_task_definition"))
            if authoritative.get("pipeline_approval_status") != "approved" or not approved:
                raise SiteTaskTestbedCompilerError(["authoritative_task_decision:not_approved"])
            if authoritative.get("capture_session_id") != session_id:
                raise SiteTaskTestbedCompilerError(["authoritative_task_decision:session_mismatch"])
            if authoritative.get("intake_id") != intake_id:
                raise SiteTaskTestbedCompilerError(["authoritative_task_decision:intake_mismatch"])
            if payload.get("approved_task_digest") != approved.get("approved_task_digest"):
                raise SiteTaskTestbedCompilerError(["approved_task_digest:authoritative_mismatch"])
            store_root_text = _string(os.getenv(CAPTURE_UPLOAD_STORE_ROOT_ENV))
            if not store_root_text:
                raise ReconstructionControlPlaneError(
                    "capture intake store is not configured",
                    status_code=503,
                )
            reconstruction = load_reconstruction_compilation_inputs(
                state_root=reconstruction_root(manifest_path),
                capture_store_root=Path(store_root_text).expanduser().resolve(),
                plan_id=str(payload.get("reconstruction_plan_id") or ""),
                execution_result_digest=str(
                    payload.get("reconstruction_execution_result_digest") or ""
                ),
            )
            context = _mapping(reconstruction.get("context"))
            if (
                context.get("capture_session_id") != session_id
                or context.get("intake_id") != intake_id
            ):
                raise SiteTaskTestbedCompilerError(["authoritative_reconstruction:source_mismatch"])
            request_constraints = payload.get("decision_request_constraints")
            if request_constraints is not None and not isinstance(request_constraints, Mapping):
                raise SiteTaskTestbedCompilerError(["decision_request_constraints:not_object"])
            constraints = dict(request_constraints or {})
            requested_claim_types = sorted(
                {
                    _string(row.get("claim_type"))
                    for row in constraints.get("claims", [])
                    if isinstance(row, Mapping) and _string(row.get("claim_type"))
                }
            )
            planned_claim_types = sorted(
                {
                    _string(row)
                    for row in _mapping(reconstruction.get("reconstruction_plan")).get(
                        "requested_claim_types", []
                    )
                    if _string(row)
                }
            )
            if requested_claim_types and requested_claim_types != planned_claim_types:
                raise SiteTaskTestbedCompilerError(
                    ["decision_request_constraints:claim_types_reconstruction_plan_mismatch"]
                )
            support = build_pipeline_owned_compilation_support(
                testbed_id=str(payload.get("testbed_id") or ""),
                version=str(payload.get("version") or ""),
                approved_task_definition=approved,
                capture_qa_report=_mapping(reconstruction.get("capture_qa_report")),
                reconstruction_plan=_mapping(reconstruction.get("reconstruction_plan")),
                robot_binding=_mapping(payload.get("robot_binding")),
            )
            testbed = compile_site_task_testbed(
                testbed_id=str(payload.get("testbed_id") or ""),
                version=str(payload.get("version") or ""),
                capture_intake_envelope=_mapping(reconstruction.get("capture_intake_envelope")),
                capture_qa_report=_mapping(reconstruction.get("capture_qa_report")),
                approved_task_definition=approved,
                reconstruction_plan=_mapping(reconstruction.get("reconstruction_plan")),
                reconstruction_results=[
                    dict(row)
                    for row in reconstruction.get("reconstruction_results", [])
                    if isinstance(row, Mapping)
                ],
                simready_decision=support["simready_decision"],
                robot_placement_result=support["robot_placement_result"],
                artifact_references=support["artifact_references"],
                supported_condition_ranges=support["supported_condition_ranges"],
                previous_testbed=None,
                pipeline_owned_support_artifacts=support["pipeline_owned_support_artifacts"],
            )
            compilation = write_testbed_version(
                output_root=maintained_testbed_root(manifest_path),
                testbed=testbed,
            )
            decision_evidence_request = None
            request_write = None
            if request_constraints is not None:
                decision_evidence_request = compile_approved_task_decision_request(
                    approved,
                    testbed=testbed,
                    request_id=str(constraints.get("request_id") or ""),
                    decision_id=str(constraints.get("decision_id") or ""),
                    candidates=[
                        dict(row)
                        for row in constraints.get("candidates", [])
                        if isinstance(row, Mapping)
                    ],
                    claims=[
                        dict(row)
                        for row in constraints.get("claims", [])
                        if isinstance(row, Mapping)
                    ],
                    budget=_mapping(constraints.get("budget")),
                    deadline=str(constraints.get("deadline") or ""),
                    permitted_evidence_methods=[
                        str(row) for row in constraints.get("permitted_evidence_methods", [])
                    ],
                    restrictions=_mapping(constraints.get("restrictions")),
                    requested_result_audience=str(
                        constraints.get("requested_result_audience") or ""
                    ),
                    caller_identity="pipeline:testbed-compiler",
                    idempotency_key=str(constraints.get("idempotency_key") or ""),
                    proposed_evaluator_identities=[],
                )
                request_write = write_testbed_decision_evidence_request(
                    output_root=maintained_testbed_root(manifest_path),
                    testbed=testbed,
                    request=decision_evidence_request,
                )
        except TaskCandidateControlPlaneError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
        except ReconstructionControlPlaneError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
        except (SiteTaskTestbedCompilerError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        webapp_sync = sync_site_task_testbed_to_webapp(
            capture_session_id=session_id,
            intake_id=intake_id,
            approved_task_digest=approved["approved_task_digest"],
            testbed=testbed,
            decision_evidence_request=decision_evidence_request,
        )
        if (
            _truthy(os.getenv(TESTBED_WEBAPP_SYNC_REQUIRED_ENV))
            and webapp_sync["status"] != "succeeded"
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"testbed WebApp sync required:{webapp_sync['status']}:"
                    f"{webapp_sync.get('reason', 'unknown')}"
                ),
            )
        return {
            "schema_version": "site_task_testbed_compilation_response.v1",
            "status": "testbed_ready",
            "capture_session_id": session_id,
            "intake_id": intake_id,
            "testbed_id": testbed["testbed_id"],
            "version": testbed["version"],
            "testbed_digest": testbed["testbed_digest"],
            "already_exists": compilation["already_exists"],
            "artifact_reference": {
                "uri": (
                    f"testbed://{testbed['testbed_id']}/{testbed['version']}/"
                    f"{testbed['testbed_digest'].removeprefix('sha256:')}.json"
                ),
                "digest": testbed["testbed_digest"],
            },
            "testbed": testbed,
            "decision_evidence_request": decision_evidence_request,
            "decision_evidence_request_artifact": request_write,
            "webapp_sync": webapp_sync,
            "proof_boundary": testbed["proof_boundary"],
        }
