"""Build a fail-closed proof matrix for realistic robot-readiness rehearsal runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import ensure_dir, optional_read_json, utc_now_iso, write_json, write_text
from .g1_controlled_proof_setup import DEFAULT_ROBOT_MAKE_MODEL, DEFAULT_ROBOT_PROFILE_ID


REALISTIC_READINESS_REHEARSAL_SCHEMA_VERSION = "realistic_readiness_rehearsal.v1"
DEFAULT_LOCAL_MUJOCO_RELATIVE = (
    "pipeline/realistic_readiness_rehearsal/mujoco_g1_walk_to_target_run/"
    "mujoco_g1_local_smoke_manifest.json"
)
DEFAULT_ROBOT_PROOF_TARGET = "physical Unitree G1 controlled run"
LIVE_PRODUCT_PROOF_GATES = (
    "physical_robot_readiness",
    "safety_validation",
    "real_robot_pov",
    "robot_team_policy_performance",
    "production_runpod_worker_execution",
    "customer_through_website_testing_ready",
)
PROOF_GATE_LABELS = {
    "physical_robot_readiness": "Physical G1 readiness",
    "safety_validation": "Safety validation",
    "real_robot_pov": "Real robot POV",
    "robot_team_policy_performance": "Robot-team policy performance",
    "production_runpod_worker_execution": "Production RunPod worker execution",
    "customer_through_website_testing_ready": "Customer-through-website readiness",
}
EXTERNAL_INPUTS_BY_PROOF_GATE = {
    "physical_robot_readiness": [
        "real Unitree G1 run package for the same job/request",
        "operator and hardware-owner attestation",
        "physical robot action logs and outcome ledger",
    ],
    "safety_validation": [
        "reviewed safety/contact/threshold evidence",
        "accepted safety thresholds and stop conditions",
        "real-world or controlled-test safety outcome record",
    ],
    "real_robot_pov": [
        "physical robot camera video references",
        "timestamp alignment between frames and robot actions",
        "owner/operator evidence attestation",
    ],
    "robot_team_policy_performance": [
        "robot-team owner acceptance or review",
        "policy metrics tied to the same scenario variation",
        "physical-run policy execution trace",
    ],
    "production_runpod_worker_execution": [
        "published provider-fetchable worker image and manifest URI",
        "RunPod API execution result",
        "active-pod before/after and shutdown proof",
    ],
    "customer_through_website_testing_ready": [
        "accepted production WebApp robot-eval request",
        "pipelineForward or durable sync success tied to the same job/request",
        "customer-visible request status within proven claim boundaries",
    ],
}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _default_robot() -> dict[str, str]:
    return {
        "make_model": DEFAULT_ROBOT_MAKE_MODEL,
        "robot_profile_id": DEFAULT_ROBOT_PROFILE_ID,
        "proof_target": DEFAULT_ROBOT_PROOF_TARGET,
    }


def _append_unique(target: list[str], values: Iterable[Any]) -> None:
    for value in values:
        text = _string(value)
        if text and text not in target:
            target.append(text)


def _optional_artifact(name: str, path: Path, *, description: str) -> dict[str, Any]:
    payload = optional_read_json(path)
    return {
        "name": name,
        "path": str(path),
        "exists": payload is not None,
        "status": _string(payload.get("status")) if payload else None,
        "description": description,
    }


def _runpod_live_execution_proof_score(payload: Mapping[str, Any] | None) -> tuple[int, int, int, int]:
    if not payload:
        return (-1, -1, -1, -1)
    return (
        1 if payload.get("production_runpod_worker_execution_proven") is True else 0,
        1 if payload.get("shutdown_or_termination_proof") is True else 0,
        1 if payload.get("status") == "runpod_live_proof_collected" else 0,
        1 if not _as_list(payload.get("blockers")) else 0,
    )


def _select_runpod_live_execution_proof(root: Path) -> tuple[Path, dict[str, Any] | None]:
    default_path = root / "pipeline" / "g1_controlled_proof_setup" / "runpod_live_execution_proof.json"
    candidates = [default_path]
    signed_dir = root / "pipeline" / "g1_controlled_proof_setup" / "signed_runpod_io"
    if signed_dir.is_dir():
        candidates.extend(sorted(signed_dir.glob("runpod_live_execution_proof.*.json")))

    best_path = default_path
    best_payload = optional_read_json(default_path)
    best_score = _runpod_live_execution_proof_score(best_payload)
    for candidate in candidates[1:]:
        payload = optional_read_json(candidate)
        score = _runpod_live_execution_proof_score(payload)
        if score > best_score or (score == best_score and str(candidate) > str(best_path)):
            best_path = candidate
            best_payload = payload
            best_score = score
    return best_path, best_payload


def _webapp_route_proof_score(payload: Mapping[str, Any] | None) -> tuple[int, int, int, int, int, str]:
    if not payload:
        return (-1, -1, -1, -1, -1, "")
    boundary = _as_mapping(payload.get("proof_boundary"))
    webapp_route = _as_mapping(payload.get("webapp_route"))
    pipeline_forward = _as_mapping(payload.get("pipeline_forward"))
    pipeline_intake = _as_mapping(payload.get("pipeline_intake"))
    blockers = _as_list(pipeline_intake.get("input_blockers"))
    return (
        1 if payload.get("status") == "forwarded_to_pipeline_intake" else 0,
        1 if boundary.get("pipeline_intake_staged_request_proven") is True else 0,
        1 if pipeline_forward.get("accepted") is True else 0,
        1 if webapp_route.get("full_production_webapp_deployment_proven") is True else 0,
        1 if not blockers else 0,
        _string(payload.get("generated_at")),
    )


def _select_webapp_route_forwarding_proof(root: Path) -> tuple[Path, dict[str, Any] | None]:
    proof_dir = root / "pipeline" / "webapp_route_forwarding_proof"
    default_path = proof_dir / "webapp_route_forwarding_proof.json"
    candidates = [default_path]
    if proof_dir.is_dir():
        for candidate in sorted(proof_dir.glob("webapp_route_forwarding_proof*.json")):
            if candidate not in candidates:
                candidates.append(candidate)

    best_path = default_path
    best_payload = optional_read_json(default_path)
    best_score = _webapp_route_proof_score(best_payload)
    for candidate in candidates[1:]:
        payload = optional_read_json(candidate)
        score = _webapp_route_proof_score(payload)
        if score > best_score or (score == best_score and str(candidate) > str(best_path)):
            best_path = candidate
            best_payload = payload
            best_score = score
    return best_path, best_payload


def _find_primary_job_id(capture_root: Path, requested: str | None) -> str | None:
    if requested:
        return requested
    jobs_root = capture_root / "pipeline" / "robot_eval_jobs"
    if not jobs_root.is_dir():
        return None
    job_dirs = sorted(path for path in jobs_root.iterdir() if path.is_dir())
    for job_dir in job_dirs:
        if (job_dir / "runpod_provider_adapter_result.json").is_file():
            return job_dir.name
    for job_dir in job_dirs:
        if (job_dir / "job_request.json").is_file():
            return job_dir.name
    return job_dirs[0].name if job_dirs else None


def _proof_item(
    *,
    proven: bool,
    status: str,
    evidence: Iterable[str],
    blockers: Iterable[str],
    required_evidence: Iterable[str],
    claim_boundary: str,
) -> dict[str, Any]:
    evidence_list: list[str] = []
    blockers_list: list[str] = []
    required_list: list[str] = []
    _append_unique(evidence_list, evidence)
    _append_unique(blockers_list, blockers)
    _append_unique(required_list, required_evidence)
    return {
        "proven": proven,
        "status": status,
        "evidence": evidence_list,
        "blockers": blockers_list,
        "required_evidence": required_list,
        "claim_boundary": claim_boundary,
    }


def _current_proof_state(requested_proof_matrix: Mapping[str, Any]) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    proven_gate_ids: list[str] = []
    not_proven_gate_ids: list[str] = []
    next_external_inputs: list[str] = []
    plain_language: list[str] = []
    for gate_id in LIVE_PRODUCT_PROOF_GATES:
        item = _as_mapping(requested_proof_matrix.get(gate_id))
        label = PROOF_GATE_LABELS.get(gate_id, gate_id)
        proven = item.get("proven") is True
        blockers = [_string(blocker) for blocker in _as_list(item.get("blockers")) if _string(blocker)]
        evidence = [_string(path) for path in _as_list(item.get("evidence")) if _string(path)]
        required_evidence = [
            _string(requirement)
            for requirement in _as_list(item.get("required_evidence"))
            if _string(requirement)
        ]
        if proven:
            proven_gate_ids.append(gate_id)
        else:
            not_proven_gate_ids.append(gate_id)
            _append_unique(next_external_inputs, EXTERNAL_INPUTS_BY_PROOF_GATE.get(gate_id, []))
        status = _string(item.get("status")) or "missing"
        plain_status = "proven" if proven else f"not proven ({status})"
        plain_language.append(f"{label}: {plain_status}")
        gates[gate_id] = {
            "label": label,
            "proven": proven,
            "status": status,
            "blockers": blockers,
            "evidence": evidence,
            "required_evidence": required_evidence,
            "claim_boundary": _string(item.get("claim_boundary")),
        }
    return {
        "gate_ids": list(LIVE_PRODUCT_PROOF_GATES),
        "live_product_gate_count": len(LIVE_PRODUCT_PROOF_GATES),
        "proven_gate_count": len(proven_gate_ids),
        "remaining_gate_count": len(not_proven_gate_ids),
        "all_live_product_gates_proven": not not_proven_gate_ids,
        "proven": proven_gate_ids,
        "not_proven": not_proven_gate_ids,
        "plain_language": plain_language,
        "next_external_inputs": next_external_inputs,
        "gates": gates,
    }


def _existing_artifacts(paths: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for raw_path in paths:
        path_text = _string(raw_path)
        if path_text and Path(path_text).exists():
            result.append(path_text)
    return result


G1_READY_FOR_LIVE_STAGING_STATUS = "ready_for_live_input_staging"


def _ready_g1_assembly_artifacts(
    *,
    assembly_manifest: Mapping[str, Any] | None,
    assembly_path: Path,
    artifact_keys: Iterable[str],
) -> tuple[bool, list[str], list[str]]:
    evidence: list[str] = []
    blockers: list[str] = []
    if not assembly_manifest:
        return False, evidence, ["missing_g1_controlled_run_evidence_assembly"]
    _append_unique(evidence, _existing_artifacts([assembly_path]))
    assembly_status = _string(assembly_manifest.get("status"))
    if assembly_status != G1_READY_FOR_LIVE_STAGING_STATUS:
        blockers.append("g1_controlled_run_evidence_assembly_not_ready")
    _append_unique(blockers, assembly_manifest.get("blockers") or [])
    _append_unique(blockers, assembly_manifest.get("file_blockers") or [])
    _append_unique(blockers, assembly_manifest.get("config_blockers") or [])
    _append_unique(blockers, assembly_manifest.get("content_blockers") or [])

    artifacts = _as_mapping(assembly_manifest.get("artifacts"))
    for artifact_key in artifact_keys:
        path_text = _string(artifacts.get(artifact_key))
        if not path_text:
            blockers.append(f"missing_g1_assembly_artifact_ref:{artifact_key}")
            continue
        path = Path(path_text)
        if not path.exists():
            blockers.append(f"missing_g1_assembly_artifact:{artifact_key}")
            continue
        _append_unique(evidence, [str(path)])
        payload = optional_read_json(path)
        if not payload:
            blockers.append(f"invalid_g1_assembly_artifact_json:{artifact_key}")
            continue
        payload_status = _string(payload.get("status"))
        if payload_status != G1_READY_FOR_LIVE_STAGING_STATUS:
            blockers.append(f"g1_assembly_artifact_not_ready:{artifact_key}:{payload_status}")
        _append_unique(blockers, payload.get("blockers") or [])
    return not blockers, evidence, blockers


def _worker_rehearsal_job_blockers(
    worker_runtime_manifest: Mapping[str, Any],
    worker_preflight_detail: Mapping[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    _append_unique(blockers, worker_runtime_manifest.get("blockers") or [])
    _append_unique(blockers, worker_runtime_manifest.get("runtime_preflight_blockers") or [])
    if worker_preflight_detail:
        _append_unique(blockers, worker_preflight_detail.get("blockers") or [])
    job_status = _string(worker_runtime_manifest.get("job_status"))
    if job_status and job_status != "completed":
        blockers.append(f"worker_job_status:{job_status}")
    job_dir = _string(worker_runtime_manifest.get("job_dir"))
    if not job_dir:
        return blockers
    job_root = Path(job_dir)
    for relative in (
        "job_run_manifest.json",
        "blocked_manifest.json",
        "job_validation.json",
        "simulator_service_result.json",
    ):
        payload = optional_read_json(job_root / relative)
        if payload:
            _append_unique(blockers, payload.get("blockers") or [])
    return blockers


def _container_rehearsal_blockers(
    runtime_manifest: Mapping[str, Any] | None,
    preflight_detail: Mapping[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    if not runtime_manifest:
        return blockers
    _append_unique(blockers, runtime_manifest.get("blockers") or [])
    _append_unique(blockers, runtime_manifest.get("runtime_preflight_blockers") or [])
    if preflight_detail:
        _append_unique(blockers, preflight_detail.get("blockers") or [])
    job_status = _string(runtime_manifest.get("job_status"))
    if job_status and job_status != "completed":
        blockers.append(f"worker_job_status:{job_status}")
    return blockers


def _container_rehearsal_summary(
    *,
    output_root: Path,
    runtime_manifest: Mapping[str, Any] | None,
    preflight_detail: Mapping[str, Any] | None,
    image_manifest: Mapping[str, Any] | None,
    runtime_manifest_path: Path,
    preflight_path: Path,
    preflight_detail_path: Path,
    image_manifest_path: Path,
) -> dict[str, Any]:
    blockers = _container_rehearsal_blockers(runtime_manifest, preflight_detail)
    return {
        "status": runtime_manifest.get("status") if runtime_manifest else "missing",
        "performed": bool(runtime_manifest),
        "claim_boundary": (
            "Local container-image worker rehearsal proves only that this local image can start "
            "the worker entrypoint and execute runtime preflight checks. It is not a published "
            "provider-fetchable image, production RunPod execution, physical robot proof, or "
            "customer readiness proof."
        ),
        "container_runtime": "local_docker_colima_or_equivalent",
        "image_ref": image_manifest.get("image_ref") if image_manifest else None,
        "image_id": image_manifest.get("image_id") if image_manifest else None,
        "image_architecture": image_manifest.get("architecture") if image_manifest else None,
        "image_os": image_manifest.get("os") if image_manifest else None,
        "entrypoint": image_manifest.get("entrypoint") if image_manifest else None,
        "image_manifest": str(image_manifest_path),
        "runtime_manifest": str(runtime_manifest_path),
        "runtime_preflight": str(preflight_path),
        "runtime_preflight_detail": str(preflight_detail_path),
        "runtime_preflight_status": runtime_manifest.get("runtime_preflight_status")
        if runtime_manifest
        else None,
        "runtime_preflight_detail_status": preflight_detail.get("status")
        if preflight_detail
        else None,
        "runtime_preflight_executed": bool(
            preflight_detail
            and _as_mapping(preflight_detail.get("proof_boundary")).get(
                "runtime_preflight_executed"
            )
        ),
        "nvidia_smi_required": bool(
            preflight_detail
            and _as_mapping(preflight_detail.get("requirements")).get("require_nvidia_smi")
        ),
        "egl_render_required": bool(
            preflight_detail
            and _as_mapping(preflight_detail.get("requirements")).get("require_egl_render")
        ),
        "artifact_upload_status": _as_mapping(runtime_manifest.get("artifact_upload")).get("status")
        if runtime_manifest
        else None,
        "blockers": blockers,
        "local_artifact_root": str(output_root / "container_worker_image_rehearsal_artifact_output"),
        "proof_boundary": {
            "local_container_image_entrypoint_started": bool(runtime_manifest),
            "runtime_preflight_executed": bool(
                preflight_detail
                and _as_mapping(preflight_detail.get("proof_boundary")).get(
                    "runtime_preflight_executed"
                )
            ),
            "runtime_preflight_passed": bool(
                preflight_detail and preflight_detail.get("status") == "passed"
            ),
            "provider_fetchable_image_proven": False,
            "production_runpod_execution_proven": False,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "safety_validated": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _proof_status(
    *,
    proven: bool,
    proof_class: str,
    evidence: Iterable[str],
    blockers: Iterable[str],
    missing_inputs: Iterable[str],
    mujoco_blocker: bool,
    external_blocker: bool,
    claim_boundary: str,
) -> dict[str, Any]:
    evidence_list: list[str] = []
    blockers_list: list[str] = []
    missing_list: list[str] = []
    _append_unique(evidence_list, evidence)
    _append_unique(blockers_list, blockers)
    _append_unique(missing_list, missing_inputs)
    return {
        "proven": proven,
        "proof_class": proof_class,
        "evidence": evidence_list,
        "blockers": blockers_list,
        "missing_inputs": missing_list,
        "mujoco_blocker": mujoco_blocker,
        "external_blocker": external_blocker,
        "claim_boundary": claim_boundary,
    }


def _mujoco_evidence(mujoco_manifest_path: Path, mujoco_manifest: Mapping[str, Any]) -> list[str]:
    artifacts = _as_mapping(mujoco_manifest.get("artifacts"))
    frame_paths = _as_list(artifacts.get("frames"))
    evidence = [
        str(mujoco_manifest_path),
        artifacts.get("scene_trace"),
        artifacts.get("spawn_trace"),
        artifacts.get("policy_trace"),
        artifacts.get("sim_robot_pov_evidence"),
    ]
    evidence.extend(frame_paths)
    return _existing_artifacts(evidence)


def _runpod_blockers(
    provider_setup: Mapping[str, Any] | None,
    runpod_result: Mapping[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    if provider_setup:
        _append_unique(blockers, provider_setup.get("blockers") or [])
        proof_boundary = _as_mapping(provider_setup.get("proof_boundary"))
        if proof_boundary.get("provider_inputs_uploaded") is not True:
            blockers.append("provider_inputs_not_uploaded")
        if proof_boundary.get("image_ref_published_proven") is not True:
            blockers.append("worker_image_ref_not_published")
    else:
        blockers.append("provider_input_setup_manifest_missing")

    if runpod_result:
        _append_unique(blockers, runpod_result.get("blockers") or [])
        if runpod_result.get("api_call_performed") is not True:
            blockers.append("runpod_api_call_not_performed")
        if runpod_result.get("status") not in {
            "serverless_run_submitted",
            "on_demand_pod_created",
            "completed",
        }:
            blockers.append(f"runpod_status:{_string(runpod_result.get('status')) or 'unknown'}")
    else:
        blockers.append("runpod_provider_adapter_result_missing")
    return blockers


def _webapp_route_proof_item(
    *,
    proof_path: Path,
    proof: Mapping[str, Any] | None,
) -> dict[str, Any]:
    required_evidence = [
        "production WebApp URL for the customer-facing /sites route",
        "accepted production /api/robot-eval/job-requests response",
        "pipelineForward or durable sync success tied to the same job/request",
        "customer-visible request status that stays within proven claim boundaries",
    ]
    claim_boundary = (
        "Local route rehearsal and prepared scripts do not prove customer-through-"
        "website readiness."
    )
    if not proof:
        return _proof_item(
            proven=False,
            status="not_ready_missing_production_webapp_route_proof",
            evidence=[],
            blockers=[
                "missing_BLUEPRINT_WEBAPP_PRODUCTION_URL",
                "missing_production_webapp_robot_eval_request_acceptance",
                "missing_production_webapp_to_pipeline_sync_success",
            ],
            required_evidence=required_evidence,
            claim_boundary=claim_boundary,
        )

    boundary = _as_mapping(proof.get("proof_boundary"))
    webapp_route = _as_mapping(proof.get("webapp_route"))
    pipeline_intake = _as_mapping(proof.get("pipeline_intake"))
    pipeline_forward = _as_mapping(proof.get("pipeline_forward"))
    blockers: list[str] = []
    if webapp_route.get("full_production_webapp_deployment_proven") is not True:
        blockers.append("production_webapp_route_not_proven")
    if pipeline_forward.get("accepted") is not True:
        blockers.append("production_webapp_pipeline_forward_not_accepted")
    _append_unique(blockers, pipeline_intake.get("input_blockers") or [])
    if boundary.get("pipeline_intake_staged_request_proven") is not True:
        blockers.append("production_pipeline_intake_not_staged")
    if boundary.get("full_webapp_db_persistence_proven") is not True:
        blockers.append("production_webapp_db_persistence_not_proven")
    proven = (
        boundary.get("production_live_webapp_forwarding_proven") is True
        and boundary.get("pipeline_intake_staged_request_proven") is True
        and boundary.get("full_webapp_db_persistence_proven") is True
    )
    return _proof_item(
        proven=proven,
        status="production_webapp_route_forwarded_to_pipeline"
        if proven
        else "not_ready_production_webapp_route_blocked",
        evidence=_existing_artifacts([proof_path]),
        blockers=blockers,
        required_evidence=required_evidence,
        claim_boundary=claim_boundary,
    )


def _controlled_proof_setup_summary(
    setup_manifest: Mapping[str, Any] | None,
    path: Path,
    assembly_manifest: Mapping[str, Any] | None,
    assembly_path: Path,
) -> dict[str, Any]:
    default_robot = _as_mapping(setup_manifest.get("default_robot")) if setup_manifest else {}
    if not default_robot:
        default_robot = _default_robot()
    field_run_kit = _as_mapping(setup_manifest.get("field_run_capture_kit")) if setup_manifest else {}
    field_run_artifacts = _as_mapping(field_run_kit.get("artifacts"))
    return {
        "status": _string(setup_manifest.get("status")) if setup_manifest else "missing",
        "path": str(path),
        "exists": setup_manifest is not None,
        "artifacts": _as_mapping(setup_manifest.get("artifacts")) if setup_manifest else {},
        "field_run_capture_kit": {
            "status": _string(field_run_kit.get("status")) if field_run_kit else "missing",
            "path": _string(field_run_artifacts.get("manifest")),
            "exists": bool(field_run_kit),
            "evidence_dir": _string(field_run_kit.get("evidence_dir"))
            or _string(field_run_artifacts.get("evidence_dir")),
            "capture_script": _string(field_run_artifacts.get("capture_script")),
            "config": _string(field_run_artifacts.get("config")),
            "proof_boundary": _as_mapping(field_run_kit.get("proof_boundary")),
        },
        "assembled_evidence": {
            "status": _string(assembly_manifest.get("status")) if assembly_manifest else "missing",
            "path": str(assembly_path),
            "exists": assembly_manifest is not None,
            "blockers": _as_list(assembly_manifest.get("blockers")) if assembly_manifest else [],
        },
        "default_robot": default_robot,
        "required_to_prove": _as_mapping(setup_manifest.get("required_to_prove"))
        if setup_manifest
        else {},
        "proof_boundary": {
            "setup_packet_is_not_proof": True,
            "physical_robot_readiness_proven": False,
            "safety_validated": False,
            "real_robot_pov_evidence_proven": False,
            "robot_team_policy_performance_proven": False,
            "production_runpod_worker_execution_proven": False,
            "customer_through_website_testing_ready": False,
        },
    }


def _build_markdown(manifest: Mapping[str, Any]) -> str:
    matrix = _as_mapping(manifest.get("requested_proof_matrix"))
    current_state = _as_mapping(manifest.get("current_proof_state"))
    lines = [
        "# Realistic Readiness Rehearsal",
        "",
        f"- Status: `{manifest.get('status')}`",
        f"- Capture root: `{manifest.get('capture_root')}`",
        f"- Primary job: `{manifest.get('primary_job_id')}`",
        (
            f"- Live-product gates proven: `{current_state.get('proven_gate_count', 0)}/"
            f"{current_state.get('live_product_gate_count', 0)}`"
        ),
        "",
        "## Current Proof State",
        "",
    ]
    for line in _as_list(current_state.get("plain_language")):
        lines.append(f"- {line}")
    lines.extend(
        [
            "",
            "## Proof Matrix",
            "",
        ]
    )
    for name, item_raw in matrix.items():
        item = _as_mapping(item_raw)
        lines.append(f"- `{name}`: `{item.get('status')}`")
        blockers = _as_list(item.get("blockers"))
        if blockers:
            lines.append(f"  - blockers: {', '.join(_string(blocker) for blocker in blockers)}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This rehearsal can prove simulator-side MuJoCo/Unitree G1 execution only when "
                "the attached simulator artifacts say so. It does not upgrade physical robot, "
                "real robot POV, safety, robot-team policy, or public-readiness claims. Production "
                "RunPod and WebApp claims require their own live proof artifacts."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _external_input_requirements(
    *,
    capture_root: Path,
    primary_job_id: str | None,
    requested_proof_matrix: Mapping[str, Any],
    controlled_proof_setup: Mapping[str, Any],
) -> dict[str, Any]:
    job_id = primary_job_id or "<job_id>"
    records = [
        {
            "input_id": "physical_robot_run_package",
            "schema_version": "physical_robot_run_package.v1",
            "status": "missing",
            "unblocks": ["physical_robot_readiness"],
            "required_fields": [
                "job_id",
                "robot_make_model",
                "robot_serial_or_fleet_id",
                "site_or_lab_location_id",
                "run_id",
                "operator_attestation",
                "hardware_owner_attestation",
                "start_time_utc",
                "end_time_utc",
                "task_id",
                "scenario_variation_id",
                "action_log_refs",
                "outcome_ledger_ref",
            ],
            "required_file_refs": [
                "physical robot run manifest",
                "robot action log",
                "operator/hardware-owner attestation",
                "deployment or controlled-test outcome ledger",
            ],
            "claim_boundary": "Required before physical_robot_readiness_proven may become true.",
        },
        {
            "input_id": "real_robot_pov_manifest",
            "schema_version": "real_robot_pov_manifest.v1",
            "status": "missing",
            "unblocks": ["real_robot_pov"],
            "required_fields": [
                "job_id",
                "run_id",
                "scenario_variation_id",
                "robot_camera_video_refs",
                "camera_mount_or_sensor_ids",
                "action_log_refs",
                "timestamp_alignment",
                "owner_evidence_refs",
                "operator_attestation",
            ],
            "required_file_refs": [
                "physical robot camera video",
                "robot action log",
                "timestamp alignment table",
                "owner/operator attestation",
            ],
            "claim_boundary": "Simulator POV frames cannot satisfy this input.",
        },
        {
            "input_id": "reviewed_safety_validation_package",
            "schema_version": "reviewed_safety_validation_package.v1",
            "status": "missing",
            "unblocks": ["safety_validation"],
            "required_fields": [
                "job_id",
                "robot_id",
                "task_id",
                "scenario_variation_id",
                "reviewer_id",
                "accepted_safety_thresholds",
                "stop_conditions",
                "contact_or_collision_log_refs",
                "physics_or_hardware_validation_refs",
                "review_decision",
                "review_timestamp_utc",
            ],
            "required_file_refs": [
                "reviewed safety case",
                "contact/collision logs",
                "threshold and stop-condition record",
                "review decision or signoff",
            ],
            "claim_boundary": "Rendered or kinematic simulation artifacts are review inputs, not safety validation.",
        },
        {
            "input_id": "robot_team_policy_package",
            "schema_version": "robot_team_policy_package.v1",
            "status": "missing",
            "unblocks": ["robot_team_policy_performance"],
            "supported_modalities": [
                "api_endpoint",
                "docker_container",
                "recorded_action_trace",
                "high_level_skill_trace",
                "teleop_demo",
                "sim_controller_plugin",
            ],
            "required_fields": [
                "job_id",
                "policy_id",
                "policy_owner",
                "modality",
                "scenario_variation_ids",
                "execution_trace_refs",
                "metric_refs",
                "owner_acceptance_or_review",
            ],
            "required_file_refs": [
                "non-default policy package body",
                "policy execution trace",
                "metrics tied to scenario variation",
                "robot-team owner acceptance or review metadata",
            ],
            "claim_boundary": "The default walk_to_target smoke policy is not a robot-team policy.",
        },
        {
            "input_id": "production_runpod_worker_execution_package",
            "schema_version": "production_runpod_worker_execution_package.v1",
            "status": "missing",
            "unblocks": ["production_runpod_worker_execution"],
            "required_fields": [
                "job_id",
                "region",
                "active_pod_count_before",
                "active_pod_count_after",
                "worker_image_ref",
                "worker_image_digest_or_versioned_tag",
                "worker_manifest_uri",
                "capture_root_bundle_uri",
                "artifact_output_uri",
                "runpod_request_id_or_pod_id",
                "runpod_status",
                "shutdown_or_termination_proof",
                "spend_usd_estimate",
            ],
            "required_file_refs": [
                "provider-fetchable worker manifest",
                "provider-fetchable capture bundle",
                "provider-writeable output artifact root",
                "RunPod result manifest",
                "pod termination/shutdown proof",
            ],
            "claim_boundary": "Dry-run, local worker rehearsal, or blocked adapter results do not prove production RunPod execution.",
        },
        {
            "input_id": "production_webapp_forwarding_sync_package",
            "schema_version": "production_webapp_forwarding_sync_package.v1",
            "status": "missing",
            "unblocks": ["customer_through_website_testing_ready"],
            "required_fields": [
                "site_submission_id",
                "buyer_request_id",
                "capture_job_id",
                "webapp_request_id",
                "pipeline_intake_request_id",
                "sync_status",
                "production_forward_url",
                "request_timestamp_utc",
                "response_status_code",
            ],
            "required_file_refs": [
                "production WebApp request proof",
                "Pipeline intake proof",
                "WebApp sync result",
                "request/response audit without secrets",
            ],
            "claim_boundary": "Local route proof is useful rehearsal evidence, not production customer-through-website proof.",
        },
    ]
    missing_records: list[dict[str, Any]] = []
    satisfied_records: list[dict[str, Any]] = []
    for record in records:
        proof_ids = [_string(item) for item in _as_list(record.get("unblocks"))]
        satisfied = bool(proof_ids) and all(
            _as_mapping(requested_proof_matrix.get(proof_id)).get("proven") is True
            for proof_id in proof_ids
        )
        if satisfied:
            record["status"] = "satisfied"
            satisfied_records.append(record)
        else:
            record["status"] = "missing"
            missing_records.append(record)
    return {
        "schema_version": "realistic_readiness_external_input_packet.v1",
        "generated_at": utc_now_iso(),
        "status": "missing_external_inputs" if missing_records else "external_inputs_satisfied",
        "capture_root": str(capture_root),
        "job_id": job_id,
        "requested_proof_statuses": {
            name: _as_mapping(item).get("status") for name, item in requested_proof_matrix.items()
        },
        "controlled_proof_setup": controlled_proof_setup,
        "missing_inputs": missing_records,
        "satisfied_inputs": satisfied_records,
        "claim_boundary": {
            "external_input_packet_is_not_proof": True,
            "physical_robot_readiness_proven": False,
            "safety_validated": False,
            "real_robot_pov_evidence_proven": False,
            "robot_team_policy_performance_proven": False,
            "production_runpod_worker_execution_proven": False,
            "customer_through_website_testing_ready": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _external_input_markdown(packet: Mapping[str, Any]) -> str:
    lines = [
        "# Realistic Readiness External Input Packet",
        "",
        f"- Status: `{packet.get('status')}`",
        f"- Capture root: `{packet.get('capture_root')}`",
        f"- Job ID: `{packet.get('job_id')}`",
        "",
        "## Missing Inputs",
        "",
    ]
    for record_raw in _as_list(packet.get("missing_inputs")):
        record = _as_mapping(record_raw)
        lines.append(f"### {record.get('input_id')}")
        lines.append(f"- Schema: `{record.get('schema_version')}`")
        lines.append(f"- Unblocks: `{', '.join(_string(item) for item in _as_list(record.get('unblocks')))}`")
        lines.append(f"- Boundary: {record.get('claim_boundary')}")
        lines.append("- Required fields:")
        for field in _as_list(record.get("required_fields")):
            lines.append(f"  - `{field}`")
        refs = _as_list(record.get("required_file_refs"))
        if refs:
            lines.append("- Required file refs:")
            for ref in refs:
                lines.append(f"  - {ref}")
        lines.append("")
    return "\n".join(lines)


def build_realistic_readiness_rehearsal(
    *,
    capture_root: str | Path,
    output_dir: str | Path | None = None,
    local_mujoco_manifest_path: str | Path | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    root = Path(capture_root).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve() if output_dir else root / "pipeline" / "realistic_readiness_rehearsal"
    ensure_dir(output_root)

    primary_job_id = _find_primary_job_id(root, job_id)
    mujoco_manifest_path = (
        Path(local_mujoco_manifest_path).expanduser().resolve()
        if local_mujoco_manifest_path
        else root / DEFAULT_LOCAL_MUJOCO_RELATIVE
    )
    mujoco_manifest = optional_read_json(mujoco_manifest_path)
    mujoco_claims = _as_mapping(mujoco_manifest.get("claim_boundary")) if mujoco_manifest else {}
    mujoco_complete = bool(
        mujoco_manifest
        and mujoco_manifest.get("status") == "complete"
        and mujoco_manifest.get("simulator_backend") == "mujoco"
        and mujoco_claims.get("local_cpu_mujoco_execution_proven") is True
        and mujoco_claims.get("mujoco_g1_asset_execution_proven") is True
    )
    default_policy_complete = bool(
        mujoco_complete and mujoco_manifest and mujoco_manifest.get("default_sim_policy_execution_proven")
    )
    sim_pov_complete = bool(
        mujoco_complete and mujoco_manifest and mujoco_manifest.get("sim_robot_pov_evidence_proven")
    )
    mujoco_evidence = _mujoco_evidence(mujoco_manifest_path, mujoco_manifest or {})

    job_root = root / "pipeline" / "robot_eval_jobs" / primary_job_id if primary_job_id else None
    provider_setup_path = (
        root
        / "pipeline"
        / "robot_eval_provider_inputs"
        / primary_job_id
        / "provider_input_setup_manifest.json"
        if primary_job_id
        else None
    )
    runpod_result_path = (
        job_root / "runpod_provider_adapter_result.json" if job_root is not None else None
    )
    provider_setup = optional_read_json(provider_setup_path) if provider_setup_path else None
    runpod_result = optional_read_json(runpod_result_path) if runpod_result_path else None
    controlled_proof_setup_path = (
        root / "pipeline" / "g1_controlled_proof_setup" / "g1_controlled_proof_setup_manifest.json"
    )
    g1_field_run_capture_kit_path = (
        root
        / "pipeline"
        / "g1_controlled_proof_setup"
        / "field_run_capture_kit"
        / "g1_field_run_capture_kit_manifest.json"
    )
    g1_evidence_assembly_path = (
        root
        / "pipeline"
        / "g1_controlled_proof_setup"
        / "assembled_live_inputs"
        / "g1_controlled_run_evidence_assembly_manifest.json"
    )
    controlled_proof_setup_manifest = optional_read_json(controlled_proof_setup_path)
    g1_evidence_assembly_manifest = optional_read_json(g1_evidence_assembly_path)
    controlled_setup_artifacts = (
        _as_mapping(controlled_proof_setup_manifest.get("artifacts"))
        if controlled_proof_setup_manifest
        else {}
    )
    official_policy_candidate_path = Path(
        _string(controlled_setup_artifacts.get("official_g1_policy_candidate"))
        or str(
            root
            / "pipeline"
            / "g1_controlled_proof_setup"
            / "official_unitree_g1_policy_candidate.json"
        )
    )
    official_policy_candidate = optional_read_json(official_policy_candidate_path)
    official_policy_execution_path = (
        root
        / "pipeline"
        / "g1_controlled_proof_setup"
        / "official_unitree_g1_policy_execution"
        / "official_unitree_g1_policy_execution_manifest.json"
    )
    official_policy_execution = optional_read_json(official_policy_execution_path)
    runpod_live_execution_proof_path, runpod_live_execution_proof = (
        _select_runpod_live_execution_proof(root)
    )
    controlled_proof_setup = _controlled_proof_setup_summary(
        controlled_proof_setup_manifest,
        controlled_proof_setup_path,
        g1_evidence_assembly_manifest,
        g1_evidence_assembly_path,
    )
    default_robot = _as_mapping(controlled_proof_setup.get("default_robot")) or _default_robot()
    production_handoff_path = root / "pipeline" / "production_handoff_readiness_manifest.json"
    provider_preview_qa_path = root / "pipeline" / "provider_preview_qa_manifest.json"
    webapp_route_proof_path, webapp_route_proof = _select_webapp_route_forwarding_proof(root)
    live_closure_path = job_root / "live_eval_closure_manifest.json" if job_root is not None else None
    worker_rehearsal_root = output_root / "same_entrypoint_worker_rehearsal"
    worker_runtime_manifest_path = worker_rehearsal_root / "worker_runtime_manifest.json"
    worker_preflight_path = worker_rehearsal_root / "worker_runtime_preflight.json"
    worker_preflight_detail_path = worker_rehearsal_root / "worker_runtime_preflight_detail.json"
    container_rehearsal_root = output_root / "container_worker_image_rehearsal"
    container_runtime_manifest_path = container_rehearsal_root / "worker_runtime_manifest.json"
    container_preflight_path = container_rehearsal_root / "worker_runtime_preflight.json"
    container_preflight_detail_path = container_rehearsal_root / "worker_runtime_preflight_detail.json"
    container_image_manifest_path = container_rehearsal_root / "container_image_manifest.json"
    local_container_cleanup_path = output_root / "local_container_runtime_cleanup_manifest.json"

    production_handoff = optional_read_json(production_handoff_path)
    provider_preview_qa = optional_read_json(provider_preview_qa_path)
    live_closure = optional_read_json(live_closure_path) if live_closure_path else None
    worker_runtime_manifest = optional_read_json(worker_runtime_manifest_path)
    worker_preflight_detail = optional_read_json(worker_preflight_detail_path)
    container_runtime_manifest = optional_read_json(container_runtime_manifest_path)
    container_preflight_detail = optional_read_json(container_preflight_detail_path)
    container_image_manifest = optional_read_json(container_image_manifest_path)
    runpod_blockers = _runpod_blockers(provider_setup, runpod_result)
    if runpod_live_execution_proof:
        _append_unique(runpod_blockers, runpod_live_execution_proof.get("blockers") or [])
    production_runpod_worker_execution_proven = bool(
        runpod_live_execution_proof
        and runpod_live_execution_proof.get("production_runpod_worker_execution_proven") is True
    )
    provider_setup_boundary = _as_mapping(provider_setup.get("proof_boundary")) if provider_setup else {}
    published_worker_image_ref_proven = bool(
        production_runpod_worker_execution_proven
        or provider_setup_boundary.get("image_ref_published_proven") is True
    )
    if production_runpod_worker_execution_proven:
        runpod_blockers = []
    official_policy_candidate_selected = bool(
        official_policy_candidate
        and official_policy_candidate.get("status") == "candidate_selected_execution_required"
    )
    official_policy_execution_complete = bool(
        official_policy_execution
        and official_policy_execution.get("status") == "completed"
        and _as_mapping(official_policy_execution.get("proof_boundary")).get(
            "non_default_policy_execution_trace_proven"
        )
        is True
        and _as_mapping(official_policy_execution.get("proof_boundary")).get(
            "policy_metrics_tied_to_scenario_variation"
        )
        is True
    )
    policy_evidence = _existing_artifacts(
        [official_policy_candidate_path, official_policy_execution_path]
    )
    if official_policy_execution_complete:
        policy_blockers = ["missing_robot_team_owner_acceptance_or_review"]
        policy_status = (
            "not_proven_official_unitree_g1_policy_executed_owner_acceptance_required"
        )
    elif official_policy_candidate_selected:
        policy_blockers = [
            "official_unitree_g1_policy_candidate_not_executed",
            "missing_non_default_policy_execution_trace",
            "missing_policy_metrics_tied_to_scenario_variation",
            "missing_robot_team_owner_acceptance_or_review",
        ]
        policy_status = "not_proven_official_unitree_g1_candidate_selected_but_not_executed"
    else:
        policy_blockers = [
            "missing_robot_team_policy_package",
            "missing_non_default_policy_execution_bundle",
            "default_walk_to_target_smoke_policy_is_not_robot_team_policy",
        ]
        policy_status = "not_proven_default_smoke_policy_only"

    physical_g1_ready, physical_g1_evidence, physical_g1_blockers = _ready_g1_assembly_artifacts(
        assembly_manifest=g1_evidence_assembly_manifest,
        assembly_path=g1_evidence_assembly_path,
        artifact_keys=[
            "physical_robot_run_manifest",
            "deployment_outcome_manifest",
            "live_closure_evidence",
        ],
    )
    safety_g1_ready, safety_g1_evidence, safety_g1_blockers = _ready_g1_assembly_artifacts(
        assembly_manifest=g1_evidence_assembly_manifest,
        assembly_path=g1_evidence_assembly_path,
        artifact_keys=["reviewed_safety_validation_package", "live_closure_evidence"],
    )
    real_pov_g1_ready, real_pov_g1_evidence, real_pov_g1_blockers = (
        _ready_g1_assembly_artifacts(
            assembly_manifest=g1_evidence_assembly_manifest,
            assembly_path=g1_evidence_assembly_path,
            artifact_keys=["real_robot_pov_manifest", "live_closure_evidence"],
        )
    )
    policy_g1_ready, policy_g1_evidence, policy_g1_blockers = _ready_g1_assembly_artifacts(
        assembly_manifest=g1_evidence_assembly_manifest,
        assembly_path=g1_evidence_assembly_path,
        artifact_keys=["robot_team_policy_package", "live_closure_evidence"],
    )
    include_g1_assembly_blockers = g1_evidence_assembly_manifest is not None

    simulator_blockers: list[str] = []
    if not mujoco_complete:
        simulator_blockers.append("local_mujoco_rehearsal_missing_or_incomplete")
    if not default_policy_complete:
        simulator_blockers.append("default_walk_to_target_policy_trace_missing_or_incomplete")
    if not sim_pov_complete:
        simulator_blockers.append("sim_robot_pov_frames_missing_or_incomplete")

    requested_proof_matrix = {
        "mujoco_unitree_g1_simulator_rehearsal": _proof_item(
            proven=not simulator_blockers,
            status="proven" if not simulator_blockers else "blocked",
            evidence=mujoco_evidence,
            blockers=simulator_blockers,
            required_evidence=[
                "MuJoCo scene load trace",
                "Unitree G1 spawn trace",
                "default walk_to_target policy execution trace",
                "simulator POV frame artifact",
            ],
            claim_boundary=(
                "Simulator-side rehearsal only; proves MuJoCo/Unitree G1 artifact execution "
                "when complete, not physical robot readiness."
            ),
        ),
        "physical_robot_readiness": _proof_item(
            proven=physical_g1_ready,
            status="proven" if physical_g1_ready else "not_proven_missing_physical_robot_run",
            evidence=physical_g1_evidence if physical_g1_evidence else mujoco_evidence,
            blockers=[]
            if physical_g1_ready
            else [
                "missing_physical_robot_run_manifest",
                "missing_hardware_operator_attestation",
                "missing_real_world_outcome_or_deployment_trial",
                *(physical_g1_blockers if include_g1_assembly_blockers else []),
            ],
            required_evidence=[
                "physical Unitree G1 or target robot run identifier",
                "operator/hardware owner attestation",
                "robot action logs from the physical run",
                "real-world outcome ledger tied to the same job/request",
            ],
            claim_boundary="Simulation output cannot prove physical robot readiness.",
        ),
        "safety_validation": _proof_item(
            proven=safety_g1_ready,
            status="proven"
            if safety_g1_ready
            else "not_proven_missing_safety_contact_physics_evidence",
            evidence=safety_g1_evidence if safety_g1_evidence else mujoco_evidence,
            blockers=[]
            if safety_g1_ready
            else [
                "missing_reviewed_safety_case",
                "missing_physical_robot_safety_validation_record",
                "missing_contact_dynamics_validation_logs",
                "missing_owner_accepted_safety_thresholds",
                *(safety_g1_blockers if include_g1_assembly_blockers else []),
            ],
            required_evidence=[
                "reviewed safety case for the target site/task/robot",
                "contact and collision logs or reviewed physics validation",
                "operator-approved safety thresholds and stop conditions",
                "real-world or controlled-test safety outcome record",
            ],
            claim_boundary="Rendered frames and kinematic simulation are not safety validation.",
        ),
        "real_robot_pov": _proof_item(
            proven=real_pov_g1_ready,
            status="proven"
            if real_pov_g1_ready
            else "not_proven_missing_real_robot_camera_action_logs",
            evidence=real_pov_g1_evidence if real_pov_g1_evidence else mujoco_evidence,
            blockers=[]
            if real_pov_g1_ready
            else [
                "missing_real_robot_pov_manifest",
                "missing_physical_robot_camera_video_refs",
                "missing_timestamp_alignment_to_robot_actions",
                "sim_robot_pov_is_virtual_camera_only",
                *(real_pov_g1_blockers if include_g1_assembly_blockers else []),
            ],
            required_evidence=[
                "real_robot_pov_manifest.v1 for the same job and variation",
                "physical robot camera video references",
                "timestamp alignment between frames and robot actions",
                "owner evidence/operator attestation",
            ],
            claim_boundary="MuJoCo camera frames are simulator POV, not physical robot POV.",
        ),
        "robot_team_policy_performance": _proof_item(
            proven=policy_g1_ready,
            status="proven" if policy_g1_ready else policy_status,
            evidence=policy_g1_evidence if policy_g1_evidence else policy_evidence,
            blockers=[]
            if policy_g1_ready
            else [
                *policy_blockers,
                *(policy_g1_blockers if include_g1_assembly_blockers else []),
            ],
            required_evidence=[
                "robot_team_policy_package.v1 or supported direct policy package",
                "non-default policy execution trace",
                "policy metrics tied to the same scenario variation",
                "robot-team owner acceptance or review metadata",
            ],
            claim_boundary=(
                "The default smoke policy checks pipeline execution only; it does not prove "
                "robot-team policy performance."
            ),
        ),
        "production_runpod_worker_execution": _proof_item(
            proven=production_runpod_worker_execution_proven,
            status="proven" if production_runpod_worker_execution_proven else "not_run_provider_gates_blocked",
            evidence=_existing_artifacts(
                [provider_setup_path, runpod_result_path, runpod_live_execution_proof_path]
            ),
            blockers=runpod_blockers,
            required_evidence=[
                "versioned provider-fetchable worker image",
                "provider-fetchable worker manifest URI",
                "provider-writeable artifact output URI",
                "successful RunPod API submission/completion result",
                "active pod count proof before and after launch",
            ],
            claim_boundary=(
                "Production RunPod worker execution requires a submitted RunPod pod, clean "
                "worker runtime manifest, and shutdown proof; it still does not prove physical "
                "robot readiness."
            ),
        ),
        "customer_through_website_testing_ready": _webapp_route_proof_item(
            proof_path=webapp_route_proof_path,
            proof=webapp_route_proof,
        ),
    }
    customer_website_proven = bool(
        _as_mapping(
            requested_proof_matrix.get("customer_through_website_testing_ready")
        ).get("proven")
    )

    production_blockers: list[str] = []
    if production_handoff:
        _append_unique(production_blockers, production_handoff.get("blockers") or [])
    if provider_preview_qa:
        _append_unique(production_blockers, provider_preview_qa.get("blockers") or [])
    if live_closure:
        _append_unique(production_blockers, live_closure.get("blockers") or [])
    if customer_website_proven:
        closed_website_blockers = {
            "production_webapp_route_not_proven",
            "production_webapp_pipeline_forward_not_accepted",
            "production_pipeline_intake_not_staged",
            "production_webapp_db_persistence_not_proven",
            "production_live_webapp_forwarding_not_proven",
        }
        production_blockers = [
            blocker
            for blocker in production_blockers
            if _string(blocker) not in closed_website_blockers
        ]

    external_blockers: list[str] = []
    for item in requested_proof_matrix.values():
        _append_unique(external_blockers, _as_list(item.get("blockers")))
    _append_unique(external_blockers, production_blockers)

    runpod_api_key_present = bool(
        os.environ.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY_FILE")
    )
    worker_rehearsal_performed = bool(worker_runtime_manifest)
    worker_rehearsal_blockers = (
        _worker_rehearsal_job_blockers(worker_runtime_manifest, worker_preflight_detail)
        if worker_runtime_manifest
        else []
    )
    container_worker_summary = _container_rehearsal_summary(
        output_root=output_root,
        runtime_manifest=container_runtime_manifest,
        preflight_detail=container_preflight_detail,
        image_manifest=container_image_manifest,
        runtime_manifest_path=container_runtime_manifest_path,
        preflight_path=container_preflight_path,
        preflight_detail_path=container_preflight_detail_path,
        image_manifest_path=container_image_manifest_path,
    )
    container_worker_blockers = _as_list(container_worker_summary.get("blockers"))
    container_mujoco_blockers = {
        "python_import_mujoco_failed",
        "blank_model_or_scene_load_failed",
        "short_rollout_smoke_failed",
        "egl_context_when_rendering_failed",
        "egl_context_when_rendering_not_attempted",
    }
    container_worker_mujoco_blocked = any(
        _string(blocker) in container_mujoco_blockers for blocker in container_worker_blockers
    )
    manifest_path = output_root / "realistic_readiness_rehearsal_manifest.json"
    proof_matrix_path = output_root / "realistic_readiness_proof_matrix.json"
    report_path = output_root / "realistic_readiness_rehearsal_report.md"
    external_input_packet_path = output_root / "realistic_readiness_external_input_packet.json"
    external_input_packet_md_path = output_root / "realistic_readiness_external_input_packet.md"
    evidence_gap_audit_path = output_root / "realistic_readiness_evidence_gap_audit.json"
    proof_boundary = {
        "mujoco_unitree_g1_simulator_rehearsal_proven": not simulator_blockers,
        "local_mujoco_execution_proven": bool(
            mujoco_claims.get("local_cpu_mujoco_execution_proven")
        ),
        "mujoco_g1_asset_execution_proven": bool(
            mujoco_claims.get("mujoco_g1_asset_execution_proven")
        ),
        "default_smoke_policy_execution_proven": default_policy_complete,
        "sim_robot_pov_evidence_proven": sim_pov_complete,
        "same_entrypoint_worker_rehearsal_performed": worker_rehearsal_performed,
        "same_entrypoint_worker_rehearsal_completed": bool(
            worker_runtime_manifest and worker_runtime_manifest.get("status") == "completed"
        ),
        "same_entrypoint_worker_runtime_preflight_executed": bool(
            worker_preflight_detail
            and _as_mapping(worker_preflight_detail.get("proof_boundary")).get(
                "runtime_preflight_executed"
            )
        ),
        "container_worker_image_rehearsal_performed": bool(container_runtime_manifest),
        "container_worker_image_runtime_preflight_executed": bool(
            container_preflight_detail
            and _as_mapping(container_preflight_detail.get("proof_boundary")).get(
                "runtime_preflight_executed"
            )
        ),
        "container_worker_image_runtime_preflight_passed": bool(
            container_preflight_detail and container_preflight_detail.get("status") == "passed"
        ),
        "container_worker_image_gpu_inventory_passed": bool(
            container_preflight_detail
            and not any(
                _string(blocker) in {"nvidia_smi_unavailable", "nvidia_smi_failed"}
                for blocker in container_worker_blockers
            )
        ),
        "container_worker_image_is_local_only": bool(container_runtime_manifest),
        "published_worker_image_ref_proven": published_worker_image_ref_proven,
        "physical_robot_readiness_proven": physical_g1_ready,
        "safety_validated": safety_g1_ready,
        "real_robot_pov_evidence_proven": real_pov_g1_ready,
        "robot_team_policy_performance_proven": policy_g1_ready,
        "production_runpod_worker_execution_proven": production_runpod_worker_execution_proven,
        "runpod_api_call_performed": bool(runpod_result and runpod_result.get("api_call_performed")),
        "runpod_side_effects_may_have_occurred": bool(
            runpod_result and runpod_result.get("runpod_side_effects_may_have_occurred")
        ),
        "runpod_live_execution_api_call_performed": bool(
            runpod_live_execution_proof
            and runpod_live_execution_proof.get("api_call_performed")
        ),
        "runpod_live_execution_side_effects_may_have_occurred": bool(
            runpod_live_execution_proof
            and runpod_live_execution_proof.get("runpod_side_effects_may_have_occurred")
        ),
        "runpod_shutdown_or_termination_proof": bool(
            runpod_live_execution_proof
            and runpod_live_execution_proof.get("shutdown_or_termination_proof")
        ),
        "runpod_api_key_present_in_env": runpod_api_key_present,
        "raw_secrets_persisted": False,
        "customer_through_website_testing_ready": customer_website_proven,
        "public_claim_upgrade_allowed": False,
    }
    current_proof_state = _current_proof_state(requested_proof_matrix)
    status = (
        "simulator_rehearsal_completed_external_evidence_blocked"
        if not simulator_blockers
        else "blocked_simulator_rehearsal_incomplete"
    )
    manifest: dict[str, Any] = {
        "schema_version": REALISTIC_READINESS_REHEARSAL_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "capture_root": str(root),
        "output_dir": str(output_root),
        "primary_job_id": primary_job_id,
        "default_robot": default_robot,
        "requested_claims_can_all_be_proven_by_this_test": False,
        "input_artifacts": [
            _optional_artifact(
                "local_mujoco_g1_smoke_manifest",
                mujoco_manifest_path,
                description="Fresh local MuJoCo Unitree G1 walk_to_target rehearsal output.",
            ),
            _optional_artifact(
                "g1_controlled_proof_setup",
                controlled_proof_setup_path,
                description=(
                    "Fillable Unitree G1 physical-run, safety, POV, policy, RunPod, and WebApp "
                    "proof packet."
                ),
            ),
            _optional_artifact(
                "g1_field_run_capture_kit",
                g1_field_run_capture_kit_path,
                description=(
                    "Operator-side physical Unitree G1 capture script, config, and evidence "
                    "checklist."
                ),
            ),
            _optional_artifact(
                "official_unitree_g1_policy_candidate",
                official_policy_candidate_path,
                description=(
                    "Official Unitree G1 policy/controller candidate selection; execution "
                    "traces and metrics are still required for performance proof."
                ),
            ),
            _optional_artifact(
                "official_unitree_g1_policy_execution",
                official_policy_execution_path,
                description=(
                    "Headless MuJoCo execution of the official Unitree G1 pretrained policy "
                    "with action trace and scenario-tied metrics."
                ),
            ),
            _optional_artifact(
                "g1_controlled_run_evidence_assembly",
                g1_evidence_assembly_path,
                description=(
                    "Assembled physical G1 evidence manifests and exact blockers for staging."
                ),
            ),
            _optional_artifact(
                "provider_input_setup_manifest",
                provider_setup_path or output_root / "missing_provider_input_setup_manifest.json",
                description="Provider-fetchable worker input setup and upload evidence.",
            ),
            _optional_artifact(
                "runpod_provider_adapter_result",
                runpod_result_path or output_root / "missing_runpod_provider_adapter_result.json",
                description="RunPod adapter dry/live result and spend side-effect boundary.",
            ),
            _optional_artifact(
                "runpod_live_execution_proof",
                runpod_live_execution_proof_path,
                description=(
                    "RunPod active-pod before/after and stop-proof collector output; not "
                    "worker execution proof by itself."
                ),
            ),
            _optional_artifact(
                "production_handoff_readiness_manifest",
                production_handoff_path,
                description="Production handoff readiness gate.",
            ),
            _optional_artifact(
                "provider_preview_qa_manifest",
                provider_preview_qa_path,
                description="Provider preview QA gate.",
            ),
            _optional_artifact(
                "webapp_route_forwarding_proof",
                webapp_route_proof_path,
                description="Local WebApp route/request proof artifact.",
            ),
            _optional_artifact(
                "live_eval_closure_manifest",
                live_closure_path or output_root / "missing_live_eval_closure_manifest.json",
                description="Live robot-eval closure gate for real policy, real POV, safety, and outcomes.",
            ),
            _optional_artifact(
                "same_entrypoint_worker_runtime_manifest",
                worker_runtime_manifest_path,
                description=(
                    "Local same-entrypoint worker rehearsal output; not production RunPod proof."
                ),
            ),
            _optional_artifact(
                "same_entrypoint_worker_runtime_preflight_detail",
                worker_preflight_detail_path,
                description="Machine-readable MuJoCo worker runtime preflight detail output.",
            ),
            _optional_artifact(
                "container_worker_image_manifest",
                container_image_manifest_path,
                description="Local Docker image inspect summary for the containerized worker rehearsal.",
            ),
            _optional_artifact(
                "container_worker_image_runtime_manifest",
                container_runtime_manifest_path,
                description=(
                    "Local containerized worker-image rehearsal output; not production RunPod proof."
                ),
            ),
            _optional_artifact(
                "container_worker_image_runtime_preflight_detail",
                container_preflight_detail_path,
                description=(
                    "Machine-readable MuJoCo runtime preflight detail from the local worker image."
                ),
            ),
            _optional_artifact(
                "local_container_runtime_cleanup_manifest",
                local_container_cleanup_path,
                description="Local Colima/Docker cleanup evidence after container rehearsal.",
            ),
        ],
        "simulator_rehearsal": {
            "status": "complete" if not simulator_blockers else "blocked",
            "simulator_backend": "mujoco" if mujoco_manifest else None,
            "robot_asset": _as_mapping(mujoco_manifest.get("robot_asset")) if mujoco_manifest else {},
            "policy_id": mujoco_manifest.get("policy_id") if mujoco_manifest else None,
            "policy_semantics": mujoco_manifest.get("policy_semantics") if mujoco_manifest else None,
            "frame_count": len(_as_list(_as_mapping(mujoco_manifest.get("artifacts")).get("frames")))
            if mujoco_manifest
            else 0,
            "artifacts": _as_mapping(mujoco_manifest.get("artifacts")) if mujoco_manifest else {},
            "blockers": simulator_blockers,
        },
        "g1_controlled_proof_setup": controlled_proof_setup,
        "same_entrypoint_worker_rehearsal": {
            "status": worker_runtime_manifest.get("status") if worker_runtime_manifest else "missing",
            "performed": worker_rehearsal_performed,
            "claim_boundary": (
                "Local same-entrypoint worker rehearsal exercises the worker code path only; "
                "it is not production RunPod execution and does not prove robot readiness."
            ),
            "runtime_manifest": str(worker_runtime_manifest_path),
            "job_status": worker_runtime_manifest.get("job_status") if worker_runtime_manifest else None,
            "job_dir": worker_runtime_manifest.get("job_dir") if worker_runtime_manifest else None,
            "artifact_upload_status": _as_mapping(
                worker_runtime_manifest.get("artifact_upload")
            ).get("status")
            if worker_runtime_manifest
            else None,
            "runtime_preflight_status": worker_runtime_manifest.get("runtime_preflight_status")
            if worker_runtime_manifest
            else None,
            "runtime_preflight": str(worker_preflight_path),
            "runtime_preflight_detail": str(worker_preflight_detail_path),
            "runtime_preflight_detail_status": worker_preflight_detail.get("status")
            if worker_preflight_detail
            else None,
            "blockers": worker_rehearsal_blockers,
        },
        "container_worker_image_rehearsal": container_worker_summary,
        "runpod_spend_boundary": {
            "api_key_present_in_env": runpod_api_key_present,
            "api_call_performed": bool(runpod_result and runpod_result.get("api_call_performed")),
            "runpod_side_effects_may_have_occurred": bool(
                runpod_result and runpod_result.get("runpod_side_effects_may_have_occurred")
            ),
            "active_pod_count_before": runpod_result.get("active_pod_count_before")
            if runpod_result
            else None,
            "active_pod_count_after": runpod_result.get("active_pod_count_after")
            if runpod_result
            else None,
            "active_pod_count_verified": bool(
                runpod_result
                and runpod_result.get("active_pod_count_before") is not None
                and runpod_result.get("active_pod_count_after") is not None
            ),
            "live_execution_proof": {
                "status": runpod_live_execution_proof.get("status")
                if runpod_live_execution_proof
                else "missing",
                "path": str(runpod_live_execution_proof_path),
                "exists": runpod_live_execution_proof is not None,
                "blockers": _as_list(runpod_live_execution_proof.get("blockers"))
                if runpod_live_execution_proof
                else [],
                "api_call_performed": bool(
                    runpod_live_execution_proof
                    and runpod_live_execution_proof.get("api_call_performed")
                ),
                "runpod_side_effects_may_have_occurred": bool(
                    runpod_live_execution_proof
                    and runpod_live_execution_proof.get("runpod_side_effects_may_have_occurred")
                ),
                "active_pod_count_before": runpod_live_execution_proof.get(
                    "active_pod_count_before"
                )
                if runpod_live_execution_proof
                else None,
                "active_pod_count_after": runpod_live_execution_proof.get(
                    "active_pod_count_after"
                )
                if runpod_live_execution_proof
                else None,
                "shutdown_or_termination_proof": bool(
                    runpod_live_execution_proof
                    and runpod_live_execution_proof.get("shutdown_or_termination_proof")
                ),
                "runtime_manifest_worker_completed": bool(
                    runpod_live_execution_proof
                    and runpod_live_execution_proof.get("runtime_manifest_worker_completed")
                ),
                "production_runpod_worker_execution_proven": bool(
                    runpod_live_execution_proof
                    and runpod_live_execution_proof.get(
                        "production_runpod_worker_execution_proven"
                    )
                ),
                "simulator_execution_proven": bool(
                    runpod_live_execution_proof
                    and runpod_live_execution_proof.get("simulator_execution_proven")
                ),
            },
            "raw_secrets_persisted": False,
            "reason_live_run_not_attempted": None
            if (
                runpod_live_execution_proof
                and runpod_live_execution_proof.get("api_call_performed")
            )
            else "provider_gates_blocked_before_runpod_spend",
        },
        "current_proof_state": current_proof_state,
        "requested_proof_matrix": requested_proof_matrix,
        "proof_boundary": proof_boundary,
        "external_inputs_required_to_upgrade_blocked_claims": current_proof_state[
            "next_external_inputs"
        ],
        "non_mujoco_external_blockers": external_blockers,
        "artifacts": {
            "manifest": str(manifest_path),
            "proof_matrix": str(proof_matrix_path),
            "report": str(report_path),
            "local_mujoco_manifest": str(mujoco_manifest_path),
            "external_input_packet": str(external_input_packet_path),
            "external_input_packet_markdown": str(external_input_packet_md_path),
            "evidence_gap_audit": str(evidence_gap_audit_path),
            "same_entrypoint_worker_runtime_manifest": str(worker_runtime_manifest_path),
            "container_worker_image_runtime_manifest": str(container_runtime_manifest_path),
            "container_worker_image_runtime_preflight_detail": str(
                container_preflight_detail_path
            ),
            "container_worker_image_manifest": str(container_image_manifest_path),
            "local_container_runtime_cleanup": str(local_container_cleanup_path),
            "g1_controlled_proof_setup": str(controlled_proof_setup_path),
            "g1_field_run_capture_kit": str(g1_field_run_capture_kit_path),
            "g1_controlled_run_evidence_assembly": str(g1_evidence_assembly_path),
            "official_unitree_g1_policy_execution": str(official_policy_execution_path),
            "runpod_live_execution_proof": str(runpod_live_execution_proof_path),
        },
    }
    evidence_gap_audit = {
        "schema_version": "realistic_readiness_evidence_gap_audit.v1",
        "generated_at": utc_now_iso(),
        "status": "open_external_gaps"
        if current_proof_state["remaining_gate_count"]
        else "all_live_product_gates_proven",
        "capture_root": str(root),
        "claim_boundary": {
            "audit_is_not_proof_by_itself": True,
            "simulator_outputs_do_not_prove_physical_robot_readiness": True,
            "simulator_pov_frames_do_not_prove_real_robot_pov": True,
            "default_smoke_policy_is_not_robot_team_policy": True,
            "local_or_container_worker_rehearsal_is_not_production_runpod_execution": True,
            "public_claim_upgrade_allowed": False,
        },
        "requirements": {
            "mujoco_unitree_g1_simulator_rehearsal": _proof_status(
                proven=not simulator_blockers,
                proof_class="simulator",
                evidence=mujoco_evidence,
                blockers=simulator_blockers,
                missing_inputs=[] if not simulator_blockers else ["complete_mujoco_g1_smoke_run"],
                mujoco_blocker=bool(simulator_blockers),
                external_blocker=False,
                claim_boundary="Proves only MuJoCo/Unitree G1 simulator execution for this sample.",
            ),
            "physical_robot_readiness": _proof_status(
                proven=physical_g1_ready,
                proof_class="external_hardware",
                evidence=physical_g1_evidence,
                blockers=_as_list(requested_proof_matrix["physical_robot_readiness"]["blockers"]),
                missing_inputs=[]
                if physical_g1_ready
                else [
                    "physical_robot_run_manifest",
                    "hardware_operator_attestation",
                    "robot_action_logs",
                    "real_world_outcome_ledger",
                ],
                mujoco_blocker=False,
                external_blocker=not physical_g1_ready,
                claim_boundary="Requires a physical robot run; MuJoCo cannot prove this claim.",
            ),
            "safety_validation": _proof_status(
                proven=safety_g1_ready,
                proof_class="external_safety_review",
                evidence=safety_g1_evidence,
                blockers=_as_list(requested_proof_matrix["safety_validation"]["blockers"]),
                missing_inputs=[]
                if safety_g1_ready
                else [
                    "reviewed_safety_case",
                    "contact_or_collision_logs",
                    "accepted_thresholds_and_stop_conditions",
                    "review_decision_or_operator_signoff",
                ],
                mujoco_blocker=False,
                external_blocker=not safety_g1_ready,
                claim_boundary="Requires reviewed safety/contact evidence; rendered frames are not safety validation.",
            ),
            "real_robot_pov": _proof_status(
                proven=real_pov_g1_ready,
                proof_class="external_hardware_sensor_evidence",
                evidence=real_pov_g1_evidence,
                blockers=_as_list(requested_proof_matrix["real_robot_pov"]["blockers"]),
                missing_inputs=[]
                if real_pov_g1_ready
                else [
                    "real_robot_pov_manifest",
                    "physical_robot_camera_video_refs",
                    "timestamp_alignment_to_action_logs",
                    "operator_or_owner_attestation",
                ],
                mujoco_blocker=False,
                external_blocker=not real_pov_g1_ready,
                claim_boundary="Requires physical robot camera/action evidence; MuJoCo POV is virtual.",
            ),
            "robot_team_policy_performance": _proof_status(
                proven=policy_g1_ready,
                proof_class="external_policy_package",
                evidence=policy_g1_evidence if policy_g1_ready else policy_evidence,
                blockers=_as_list(
                    requested_proof_matrix["robot_team_policy_performance"]["blockers"]
                ),
                missing_inputs=[]
                if policy_g1_ready
                else [
                    "non_default_robot_team_policy_package",
                    "policy_owner_or_team_provenance",
                    "policy_execution_trace",
                    "policy_metrics_and_acceptance_review",
                ],
                mujoco_blocker=False,
                external_blocker=not policy_g1_ready,
                claim_boundary="The default walk_to_target smoke policy is not robot-team policy proof.",
            ),
            "production_runpod_worker_execution": _proof_status(
                proven=production_runpod_worker_execution_proven,
                proof_class="external_provider_execution",
                evidence=_existing_artifacts(
                    [
                        provider_setup_path,
                        runpod_result_path,
                        container_runtime_manifest_path,
                        runpod_live_execution_proof_path,
                    ]
                ),
                blockers=_as_list(
                    requested_proof_matrix["production_runpod_worker_execution"]["blockers"]
                ),
                missing_inputs=[]
                if production_runpod_worker_execution_proven
                else [
                    "published_provider_fetchable_worker_image_ref",
                    "provider_fetchable_worker_manifest_uri",
                    "provider_writeable_artifact_output_uri",
                    "runpod_api_credentials_in_environment",
                    "active_pod_count_before_and_after",
                    "pod_termination_or_shutdown_proof",
                ],
                mujoco_blocker=False,
                external_blocker=not production_runpod_worker_execution_proven,
                claim_boundary=(
                    "Production RunPod proof establishes worker execution only; it does not "
                    "prove physical robot readiness."
                    if production_runpod_worker_execution_proven
                    else "Local container worker evidence does not prove production RunPod execution."
                ),
            ),
            "local_container_worker_image_rehearsal": _proof_status(
                proven=bool(
                    container_preflight_detail and container_preflight_detail.get("status") == "passed"
                ),
                proof_class="local_container_runtime_preflight",
                evidence=_existing_artifacts(
                    [
                        container_image_manifest_path,
                        container_runtime_manifest_path,
                        container_preflight_path,
                        container_preflight_detail_path,
                    ]
                ),
                blockers=container_worker_blockers,
                missing_inputs=["nvidia_smi_visible_gpu_runtime"]
                if "nvidia_smi_unavailable" in container_worker_blockers
                else [],
                mujoco_blocker=container_worker_mujoco_blocked,
                external_blocker=bool(container_worker_blockers)
                and not container_worker_mujoco_blocked,
                claim_boundary=(
                    "Local image rehearsal is useful worker-entrypoint evidence only; it is not "
                    "provider-fetchable image proof or production RunPod proof."
                ),
            ),
        },
        "conclusion": {
            "mujoco_related_work_ambiguous": bool(simulator_blockers or container_worker_mujoco_blocked),
            "remaining_blockers_are_non_mujoco_external_blockers": bool(
                current_proof_state["remaining_gate_count"]
                and not simulator_blockers
                and not container_worker_mujoco_blocked
            ),
            "live_product_gates_proven": current_proof_state["proven_gate_count"],
            "live_product_gate_count": current_proof_state["live_product_gate_count"],
            "remaining_live_product_gates": current_proof_state["not_proven"],
            "customer_through_website_testing_ready": customer_website_proven,
            "reason": (
                "All live-product gates are proven by the current evidence set."
                if not current_proof_state["remaining_gate_count"]
                else (
                    "MuJoCo simulator artifacts are complete, but these live-product gates "
                    "still require external evidence: "
                    f"{', '.join(current_proof_state['not_proven'])}."
                )
            ),
        },
    }
    external_packet = _external_input_requirements(
        capture_root=root,
        primary_job_id=primary_job_id,
        requested_proof_matrix=requested_proof_matrix,
        controlled_proof_setup=controlled_proof_setup,
    )
    write_json(proof_matrix_path, requested_proof_matrix)
    write_json(external_input_packet_path, external_packet)
    write_text(external_input_packet_md_path, _external_input_markdown(external_packet))
    write_json(evidence_gap_audit_path, evidence_gap_audit)
    write_json(manifest_path, manifest)
    write_text(report_path, _build_markdown(manifest))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--local-mujoco-manifest", type=Path, default=None)
    parser.add_argument("--job-id", default=None)
    args = parser.parse_args(argv)
    manifest = build_realistic_readiness_rehearsal(
        capture_root=args.capture_root,
        output_dir=args.output_dir,
        local_mujoco_manifest_path=args.local_mujoco_manifest,
        job_id=args.job_id,
    )
    print(manifest["artifacts"]["manifest"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
