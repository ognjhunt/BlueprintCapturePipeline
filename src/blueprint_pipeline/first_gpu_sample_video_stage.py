"""Stage a single collected video as a first-GPU E2E sample capture."""

from __future__ import annotations

import argparse
import os
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Sequence

from .common import PipelineError, ensure_dir, read_json_any, utc_now_iso, write_json
from .first_gpu_candidate_audit import build_first_gpu_candidate_audit
from .first_gpu_sample_video_preflight import (
    DEFAULT_MAX_DURATION_SECONDS,
    DEFAULT_MAX_SIZE_BYTES,
    build_first_gpu_sample_video_preflight,
)
from .preflight_capture import build_capture_preflight_report
from .simulation_automation import build_simulation_automation


FIRST_GPU_SAMPLE_VIDEO_STAGE_SCHEMA_VERSION = "first_gpu_sample_video_stage.v1"
LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION = "blueprint_live_pipeline_staged_inputs.v1"
WEBAPP_JOB_REQUEST_QUEUE_CONTRACT = "robot_eval_job_request_inbox.v1"
WEBAPP_JOB_REQUEST_SCHEMA_VERSION = "robot_eval_job_request.v1"
LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND = "local_first_gpu_rehearsal_request"
REQUESTED_OUTPUTS = [
    "qualification",
    "preview_simulation",
    "robot_eval_dataset",
    "task_evaluation_run",
]
VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v"}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _safe_id(value: str, *, field: str) -> str:
    text = _string(value)
    if not text:
        raise PipelineError(f"{field} is required")
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise PipelineError(f"{field} must be a path-safe identifier, got: {text}")
    return text


def _optional_id(value: str | None) -> str | None:
    text = _string(value)
    return text or None


def _copy_or_link_video(*, source: Path, target: Path, mode: str) -> None:
    ensure_dir(target.parent)
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "link":
        os.symlink(source, target)
    else:
        raise PipelineError(f"Unsupported staging mode: {mode}")


def _remove_existing_capture_root(capture_root: Path) -> None:
    if capture_root.is_symlink() or capture_root.is_file():
        capture_root.unlink()
    elif capture_root.is_dir():
        shutil.rmtree(capture_root)


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_webapp_ids(
    *,
    site_submission_id: str | None,
    request_id: str | None,
    buyer_request_id: str | None,
    capture_job_id: str | None,
) -> None:
    values = {
        "site_submission_id": site_submission_id,
        "request_id": request_id,
        "buyer_request_id": buyer_request_id,
        "capture_job_id": capture_job_id,
    }
    missing = [field for field, value in values.items() if not _string(value)]
    if missing:
        raise PipelineError(
            "Local WebApp rehearsal request staging requires real upstream IDs; "
            f"missing: {', '.join(missing)}"
        )


def _manifest_payload(
    *,
    scene_id: str,
    capture_id: str,
    video_name: str,
    workflow_name: str,
    task_steps: list[str],
    zone: str,
    owner: str,
    site_submission_id: str | None,
    request_id: str | None,
    buyer_request_id: str | None,
    capture_job_id: str | None,
    derived_scene_generation_allowed: bool = False,
    data_licensing_allowed: bool = False,
    capture_contributor_payout_eligible: bool = False,
    consent_status: str = "unknown",
    permission_document_uri: str | None = None,
    consent_scope: Sequence[str] = (),
    consent_notes: Sequence[str] = (),
) -> Dict[str, Any]:
    rights_scope = {
        "derived_scene_generation_allowed": bool(derived_scene_generation_allowed),
        "data_licensing_allowed": bool(data_licensing_allowed),
        "capture_contributor_payout_eligible": bool(
            capture_contributor_payout_eligible
        ),
        "consent_status": _string(consent_status) or "unknown",
        "permission_document_uri": _optional_id(permission_document_uri),
        "consent_scope": [_string(item) for item in consent_scope if _string(item)],
        "consent_notes": [_string(item) for item in consent_notes if _string(item)],
    }
    owner_approval = {
        "status": (
            "documented"
            if rights_scope["consent_status"] == "documented"
            and rights_scope["permission_document_uri"]
            else "not_documented"
        ),
        "approved_scope": list(rights_scope["consent_scope"]),
        "permission_document_uri": rights_scope["permission_document_uri"],
        "derived_scene_generation_allowed": rights_scope["derived_scene_generation_allowed"],
        "data_licensing_allowed": rights_scope["data_licensing_allowed"],
        "capture_contributor_payout_eligible": rights_scope[
            "capture_contributor_payout_eligible"
        ],
        "customer_publication_allowed": False,
        "generated_world_rank_fidelity_claim_allowed": False,
        "proof_boundary": (
            "Owner approval on a staged sample scopes simulator/world-model smoke work only; "
            "it does not prove unrestricted licensing, customer publication, or generated-world rank fidelity."
        ),
    }
    return {
        "schema_version": "capture_raw_manifest.v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "video_uri": video_name,
        "requested_outputs": list(REQUESTED_OUTPUTS),
        "capture_source": "operator_staged_sample_video",
        "capture_modality": "single_walkthrough_video",
        "evidence_tier": "pre_screen_video",
        "capture_capabilities": {
            "walkthrough_video": True,
            "camera_pose": False,
            "intrinsics": False,
            "depth": False,
            "motion": False,
            "manual_stage": True,
        },
        "workflowName": workflow_name,
        "taskSteps": task_steps,
        "zone": zone,
        "owner": owner,
        "site_submission_id": site_submission_id,
        "request_id": request_id,
        "buyer_request_id": buyer_request_id,
        "capture_job_id": capture_job_id,
        "capture_rights": rights_scope,
        "owner_approval": owner_approval,
        "proof_boundary": (
            "Single-video staging preserves raw walkthrough truth only. Missing pose, depth, "
            "intrinsics, WebApp upstream IDs, policy packages, and GPU proof must remain blocked "
            "unless supplied by their authoritative systems."
        ),
    }


def _write_local_webapp_rehearsal_request(
    *,
    capture_root: Path,
    scene_id: str,
    capture_id: str,
    workflow_name: str,
    task_steps: Sequence[str],
    zone: str,
    owner: str,
    site_submission_id: str,
    request_id: str,
    buyer_request_id: str,
    capture_job_id: str,
    job_id: str | None,
) -> Dict[str, Any]:
    selected_job_id = _safe_id(
        job_id or f"local-first-gpu-{scene_id}-{capture_id}",
        field="local_webapp_job_id",
    )
    request_path = (
        capture_root
        / "pipeline"
        / "robot_eval_job_requests"
        / "local_rehearsal"
        / f"{selected_job_id}.json"
    )
    staged_inputs_path = capture_root / "pipeline" / "live_pipeline_staged_inputs.json"
    generated_at = utc_now_iso()
    job_request = {
        "schema_version": WEBAPP_JOB_REQUEST_SCHEMA_VERSION,
        "job_id": selected_job_id,
        "source_kind": LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND,
        "source": {
            "source_kind": LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND,
            "site_submission_id": site_submission_id,
            "request_id": request_id,
            "buyer_request_id": buyer_request_id,
            "capture_job_id": capture_job_id,
        },
        "site_package": {
            "capture_root": str(capture_root.resolve()),
            "site_submission_id": site_submission_id,
            "capture_job_id": capture_job_id,
            "buyer_request_id": buyer_request_id,
            "scene_id": scene_id,
            "capture_id": capture_id,
        },
        "owner_system": {
            "request_id": request_id,
            "buyer_request_id": buyer_request_id,
            "site_submission_id": site_submission_id,
            "capture_job_id": capture_job_id,
        },
        "request_context": {
            "workflowName": workflow_name,
            "taskSteps": list(task_steps),
            "zone": zone,
            "owner": owner,
        },
        "proof_boundary": {
            "local_rehearsal_only": True,
            "webapp_forwarding_proven": False,
            "webapp_request_submitted_by_webapp": False,
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    envelope = {
        "queue_contract": WEBAPP_JOB_REQUEST_QUEUE_CONTRACT,
        "source_kind": LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND,
        "local_rehearsal_only": True,
        "status": "queued_for_pipeline_local_rehearsal",
        "generated_at": generated_at,
        "job_id": selected_job_id,
        "job_request": job_request,
        "proof_boundary": {
            "local_rehearsal_only": True,
            "webapp_forwarding_proven": False,
            "webapp_request_submitted_by_webapp": False,
        },
    }
    write_json(request_path, envelope)
    staged_inputs = {
        "schema_version": LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_kind": LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND,
        "local_rehearsal_only": True,
        "configured_capture_root": str(capture_root.resolve()),
        "webapp_request": {
            "ready": True,
            "staged": True,
            "source_kind": LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND,
            "job_id": selected_job_id,
            "path": str(request_path),
            "target_path": str(request_path),
            "sha256": _sha_file(request_path),
        },
        "proof_boundary": {
            "staged_inputs_are_pointers_only": True,
            "local_rehearsal_only": True,
            "real_webapp_forwarding_proven": False,
            "webapp_request_submitted_by_webapp": False,
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    write_json(staged_inputs_path, staged_inputs)
    return {
        "status": "staged",
        "source_kind": LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND,
        "job_id": selected_job_id,
        "request_path": str(request_path),
        "staged_inputs_path": str(staged_inputs_path),
        "sha256": staged_inputs["webapp_request"]["sha256"],
        "proof_boundary": staged_inputs["proof_boundary"],
    }


def stage_first_gpu_sample_video(
    *,
    source_video: str | Path,
    storage_root: str | Path,
    scene_id: str,
    capture_id: str,
    bucket: str = "local-blueprint",
    mode: str = "copy",
    force: bool = False,
    workflow_name: str = "First GPU sample walkthrough",
    task_steps: Sequence[str] = (),
    zone: str = "sample-zone",
    owner: str = "operator",
    site_submission_id: str | None = None,
    request_id: str | None = None,
    buyer_request_id: str | None = None,
    capture_job_id: str | None = None,
    derived_scene_generation_allowed: bool = False,
    data_licensing_allowed: bool = False,
    capture_contributor_payout_eligible: bool = False,
    consent_status: str = "unknown",
    permission_document_uri: str | None = None,
    consent_scope: Sequence[str] = (),
    consent_notes: Sequence[str] = (),
    stage_local_webapp_rehearsal_request: bool = False,
    local_webapp_job_id: str | None = None,
    require_source_video_preflight: bool = False,
    max_video_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
    max_video_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    scene_assets: Sequence[str | Path] = (),
    run_simulation_automation: bool = False,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    source = Path(source_video).expanduser().resolve()
    if not source.is_file():
        raise PipelineError(f"Source video is missing: {source}")
    suffix = source.suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise PipelineError(
            f"Source video must use one of {sorted(VIDEO_SUFFIXES)}, got: {source.name}"
        )
    safe_scene_id = _safe_id(scene_id, field="scene_id")
    safe_capture_id = _safe_id(capture_id, field="capture_id")
    resolved_bucket = _safe_id(bucket, field="bucket")
    steps = [_string(step) for step in task_steps if _string(step)] or [
        "load captured scene",
        "spawn robot at proposed start pose",
        "attempt the selected task trace",
    ]
    source_video_preflight = build_first_gpu_sample_video_preflight(
        source_videos=[source],
        max_duration_seconds=max_video_duration_seconds,
        max_size_bytes=max_video_size_bytes,
        require_probe=require_source_video_preflight,
    )
    if require_source_video_preflight and source_video_preflight.get("status") != "ready":
        candidate_blockers: list[str] = []
        for candidate in source_video_preflight.get("candidates") or []:
            candidate_blockers.extend(candidate.get("staging_blockers") or [])
            candidate_blockers.extend(candidate.get("worldlabs_blockers") or [])
        blockers = source_video_preflight.get("blockers") or candidate_blockers
        raise PipelineError(
            "Source video failed strict first-GPU preflight: "
            + ", ".join(str(item) for item in blockers)
        )

    capture_root = (
        Path(storage_root).expanduser().resolve()
        / resolved_bucket
        / "scenes"
        / safe_scene_id
        / "captures"
        / safe_capture_id
    )
    if capture_root.exists() or capture_root.is_symlink():
        if not force:
            raise PipelineError(
                f"Capture root already exists: {capture_root}. Re-run with --force to replace it."
            )
        _remove_existing_capture_root(capture_root)

    raw_root = capture_root / "raw"
    video_name = f"walkthrough{suffix}"
    _copy_or_link_video(source=source, target=raw_root / video_name, mode=mode)
    source_video_preflight_path = capture_root / "pipeline" / "source_video_preflight_manifest.json"
    write_json(source_video_preflight_path, source_video_preflight)
    generated_at = utc_now_iso()
    site_submission = _optional_id(site_submission_id)
    request = _optional_id(request_id)
    buyer_request = _optional_id(buyer_request_id)
    capture_job = _optional_id(capture_job_id)
    if stage_local_webapp_rehearsal_request:
        _require_webapp_ids(
            site_submission_id=site_submission,
            request_id=request,
            buyer_request_id=buyer_request,
            capture_job_id=capture_job,
        )

    manifest = _manifest_payload(
        scene_id=safe_scene_id,
        capture_id=safe_capture_id,
        video_name=video_name,
        workflow_name=workflow_name,
        task_steps=steps,
        zone=zone,
        owner=owner,
        site_submission_id=site_submission,
        request_id=request,
        buyer_request_id=buyer_request,
        capture_job_id=capture_job,
        derived_scene_generation_allowed=derived_scene_generation_allowed,
        data_licensing_allowed=data_licensing_allowed,
        capture_contributor_payout_eligible=capture_contributor_payout_eligible,
        consent_status=consent_status,
        permission_document_uri=permission_document_uri,
        consent_scope=consent_scope,
        consent_notes=consent_notes,
    )
    source_video_metadata = {
        "source_path": str(source),
        "raw_video_path": str(raw_root / video_name),
        "raw_video_uri": f"gs://{resolved_bucket}/scenes/{safe_scene_id}/captures/{safe_capture_id}/raw/{video_name}",
        "sha256": _sha_file(raw_root / video_name),
        "staging_mode": mode,
    }
    manifest["source_video"] = dict(source_video_metadata)
    capture_context = {
        **manifest,
        "schema_version": "v1",
        "generated_at": generated_at,
        "captured_at": generated_at,
        "special_task_type": "open_capture",
        "task_hypothesis_status": "accepted",
    }
    intake_packet = {
        "schema_version": "v1",
        "workflowName": workflow_name,
        "taskSteps": steps,
        "zone": zone,
        "owner": owner,
        "capture_rights": dict(manifest["capture_rights"]),
        "owner_approval": dict(manifest["owner_approval"]),
        "source_video": dict(source_video_metadata),
        "proof_boundary": manifest["proof_boundary"],
        "successCriteria": [
            "scene load trace exists",
            "spawn pose trace exists",
            "action or policy trace exists",
        ],
    }
    task_hypothesis = {
        "schema_version": "v1",
        "status": "accepted",
        "workflowName": workflow_name,
        "taskSteps": steps,
        "targetKPI": "first_gpu_smoke",
    }
    completion = {
        "schema_version": "v1",
        "scene_id": safe_scene_id,
        "capture_id": safe_capture_id,
        "raw_prefix": f"scenes/{safe_scene_id}/captures/{safe_capture_id}/raw",
        "raw_video_uri": source_video_metadata["raw_video_uri"],
        "source_video_sha256": source_video_metadata["sha256"],
        "capture_rights": dict(manifest["capture_rights"]),
        "owner_approval": dict(manifest["owner_approval"]),
        "proof_boundary": manifest["proof_boundary"],
        "completed_at": generated_at,
    }

    write_json(raw_root / "manifest.json", manifest)
    write_json(raw_root / "capture_context.json", capture_context)
    write_json(raw_root / "intake_packet.json", intake_packet)
    write_json(raw_root / "task_hypothesis.json", task_hypothesis)
    write_json(raw_root / "capture_upload_complete.json", completion)

    local_webapp_rehearsal: Dict[str, Any] | None = None
    if stage_local_webapp_rehearsal_request:
        local_webapp_rehearsal = _write_local_webapp_rehearsal_request(
            capture_root=capture_root,
            scene_id=safe_scene_id,
            capture_id=safe_capture_id,
            workflow_name=workflow_name,
            task_steps=steps,
            zone=zone,
            owner=owner,
            site_submission_id=site_submission or "",
            request_id=request or "",
            buyer_request_id=buyer_request or "",
            capture_job_id=capture_job or "",
            job_id=local_webapp_job_id,
        )

    preflight = build_capture_preflight_report(capture_root)
    candidate_audit_path = capture_root / "pipeline" / "first_gpu_candidate_audit_manifest.json"
    candidate_audit = build_first_gpu_candidate_audit(
        capture_roots=[capture_root],
        output_path=candidate_audit_path,
    )
    simulation_automation_result: Dict[str, Any] | None = None
    gpu_handoff_packet_path = (
        capture_root / "pipeline" / "simulation_automation" / "gpu_handoff_packet.json"
    )
    gpu_handoff_packet: Dict[str, Any] = {}
    if run_simulation_automation:
        simulation_automation_result = build_simulation_automation(
            capture_root=capture_root,
            scene_assets=scene_assets,
        )
        if gpu_handoff_packet_path.is_file():
            gpu_handoff_packet = read_json_any(gpu_handoff_packet_path)

    result = {
        "schema_version": FIRST_GPU_SAMPLE_VIDEO_STAGE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_video": str(source),
        "capture_root": str(capture_root),
        "raw_video_path": str(raw_root / video_name),
        "mode": mode,
        "requested_outputs": list(REQUESTED_OUTPUTS),
        "webapp_upstream_ids_supplied": {
            "site_submission_id": bool(site_submission),
            "request_id": bool(request),
            "buyer_request_id": bool(buyer_request),
            "capture_job_id": bool(capture_job),
        },
        "capture_rights": dict(manifest["capture_rights"]),
        "preflight_status": preflight.get("status"),
        "preflight_missing_required_inputs": preflight.get("missing_required_inputs") or [],
        "candidate_audit_path": str(candidate_audit_path),
        "candidate_audit_status": candidate_audit.get("status"),
        "candidate_audit_blockers": candidate_audit.get("blockers") or [],
        "source_video_preflight_path": str(source_video_preflight_path),
        "source_video_preflight_status": source_video_preflight.get("status"),
        "source_video_ready_for_worldlabs_first_clip": bool(
            source_video_preflight.get("ready_for_worldlabs_first_clip_count")
        ),
        "source_video_preflight_blockers": source_video_preflight.get("blockers") or [],
        "source_video_preflight_candidates": source_video_preflight.get("candidates") or [],
        "single_video_limitations": [
            "walkthrough video only; camera pose, intrinsics, depth, motion, and policy package are not supplied",
            "local staging does not create WebApp upstream IDs or staged WebApp request truth",
            "GPU handoff may still block on scene asset, spawn pose, or owner GPU proof requirements",
        ],
        "local_webapp_rehearsal_request": local_webapp_rehearsal
        or {
            "status": "not_requested",
            "source_kind": LOCAL_WEBAPP_REHEARSAL_SOURCE_KIND,
            "request_path": None,
            "staged_inputs_path": None,
        },
        "simulation_automation_run": bool(run_simulation_automation),
        "scene_asset_inputs": [str(Path(item).expanduser()) for item in scene_assets],
        "simulation_automation_status": (
            simulation_automation_result.get("status") if simulation_automation_result else None
        ),
        "simulation_automation_manifest_path": (
            simulation_automation_result.get("manifest_path")
            if simulation_automation_result
            else None
        ),
        "gpu_handoff_packet_path": str(gpu_handoff_packet_path),
        "gpu_handoff_status": gpu_handoff_packet.get("status") if gpu_handoff_packet else None,
        "gpu_handoff_ready_for_owner_gpu_preflight": bool(
            gpu_handoff_packet.get("ready_for_owner_gpu_preflight")
        )
        if gpu_handoff_packet
        else False,
        "gpu_handoff_blockers": gpu_handoff_packet.get("blockers") if gpu_handoff_packet else [],
        "gpu_handoff_hard_preflight_blockers": (
            gpu_handoff_packet.get("hard_preflight_blockers") if gpu_handoff_packet else []
        ),
        "gpu_handoff_pre_gpu_blocker_details": (
            gpu_handoff_packet.get("pre_gpu_blocker_details") if gpu_handoff_packet else []
        ),
        "gpu_handoff_spawn_validation_summary": (
            gpu_handoff_packet.get("spawn_validation_summary") if gpu_handoff_packet else {}
        ),
        "next_commands": {
            "preflight": f"blueprint-preflight-capture --capture-root {capture_root}",
            "candidate_audit": (
                f"blueprint-audit-first-gpu-candidates --capture-root {capture_root} "
                f"--output {candidate_audit_path}"
            ),
            "source_video_preflight": (
                "blueprint-audit-first-gpu-sample-video "
                f"--source-video {source} --require-probe "
                f"--output {source_video_preflight_path}"
            ),
            "simulation_automation": (
                f"blueprint-run-simulation-automation --capture-root {capture_root}"
            ),
            "run_packet": (
                f"blueprint-build-first-gpu-run-packet --capture-root {capture_root} "
                f"--webapp-site-slug {safe_scene_id}"
            ),
            "local_rehearsal_readiness": (
                f"blueprint-audit-first-gpu-e2e-readiness --capture-root {capture_root} "
                f"--webapp-site-slug {safe_scene_id} "
                f"--webapp-staged-inputs {capture_root / 'pipeline' / 'live_pipeline_staged_inputs.json'} "
                "--allow-local-webapp-rehearsal"
            ),
        },
        "claim_boundary": {
            "artifact_purpose": "single_video_first_gpu_sample_staging",
            "raw_capture_truth_preserved": True,
            "live_provider_calls_performed": False,
            "webapp_requests_submitted": False,
            "local_webapp_rehearsal_request_written": bool(local_webapp_rehearsal),
            "real_webapp_forwarding_proven": False,
            "simulation_automation_artifacts_written": bool(run_simulation_automation),
            "simulator_execution_performed": False,
            "gpu_provisioning_performed": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    if output_path:
        write_json(Path(output_path).expanduser(), result)
    else:
        write_json(capture_root / "pipeline" / "first_gpu_sample_video_stage_manifest.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage one collected walkthrough video as a first-GPU sample capture"
    )
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--bucket", default="local-blueprint")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--capture-id", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true", help="Copy the source video into raw/")
    mode.add_argument("--link", action="store_true", help="Symlink the source video into raw/")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workflow-name", default="First GPU sample walkthrough")
    parser.add_argument("--task-step", action="append", default=[])
    parser.add_argument("--zone", default="sample-zone")
    parser.add_argument("--owner", default="operator")
    parser.add_argument("--site-submission-id", default=None)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--buyer-request-id", default=None)
    parser.add_argument("--capture-job-id", default=None)
    parser.add_argument(
        "--allow-derived-scene-generation",
        action="store_true",
        help=(
            "Mark this staged sample as explicitly allowed for derived scene generation. "
            "Omit to keep rights fail-closed."
        ),
    )
    parser.add_argument(
        "--allow-data-licensing",
        action="store_true",
        help="Mark this staged sample as explicitly allowed for downstream data licensing.",
    )
    parser.add_argument(
        "--capture-contributor-payout-eligible",
        action="store_true",
        help="Mark the staged sample as payout-eligible under the supplied rights scope.",
    )
    parser.add_argument("--consent-status", default="unknown")
    parser.add_argument("--permission-document-uri", default=None)
    parser.add_argument("--consent-scope", action="append", default=[])
    parser.add_argument("--consent-note", action="append", default=[])
    parser.add_argument(
        "--require-source-video-preflight",
        action="store_true",
        help=(
            "Require ffprobe-backed source video suitability before writing the staged capture. "
            "Use this for the real first-GPU sample."
        ),
    )
    parser.add_argument(
        "--max-video-duration-seconds",
        type=float,
        default=DEFAULT_MAX_DURATION_SECONDS,
    )
    parser.add_argument("--max-video-size-bytes", type=int, default=DEFAULT_MAX_SIZE_BYTES)
    parser.add_argument(
        "--stage-local-webapp-rehearsal-request",
        action="store_true",
        help=(
            "Write a local rehearsal robot_eval_job_request and staged-input pointer. "
            "Requires real upstream IDs and is not live WebApp forwarding proof."
        ),
    )
    parser.add_argument("--local-webapp-job-id", default=None)
    parser.add_argument(
        "--scene-asset",
        action="append",
        default=[],
        help=(
            "Optional local scene asset path for simulation automation bounds checks. "
            "Repeat for multiple assets."
        ),
    )
    parser.add_argument(
        "--run-simulation-automation",
        action="store_true",
        help=(
            "After staging, write local simulation automation and GPU handoff artifacts. "
            "This does not run a simulator or provision a GPU."
        ),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    try:
        result = stage_first_gpu_sample_video(
            source_video=args.source_video,
            storage_root=args.storage_root,
            bucket=args.bucket,
            scene_id=args.scene_id,
            capture_id=args.capture_id,
            mode="link" if args.link else "copy",
            force=args.force,
            workflow_name=args.workflow_name,
            task_steps=args.task_step,
            zone=args.zone,
            owner=args.owner,
            site_submission_id=args.site_submission_id,
            request_id=args.request_id,
            buyer_request_id=args.buyer_request_id,
            capture_job_id=args.capture_job_id,
            derived_scene_generation_allowed=args.allow_derived_scene_generation,
            data_licensing_allowed=args.allow_data_licensing,
            capture_contributor_payout_eligible=args.capture_contributor_payout_eligible,
            consent_status=args.consent_status,
            permission_document_uri=args.permission_document_uri,
            consent_scope=args.consent_scope,
            consent_notes=args.consent_note,
            require_source_video_preflight=args.require_source_video_preflight,
            max_video_duration_seconds=args.max_video_duration_seconds,
            max_video_size_bytes=args.max_video_size_bytes,
            stage_local_webapp_rehearsal_request=args.stage_local_webapp_rehearsal_request,
            local_webapp_job_id=args.local_webapp_job_id,
            scene_assets=args.scene_asset,
            run_simulation_automation=args.run_simulation_automation,
            output_path=args.output,
        )
    except Exception as exc:
        print(f"[first-gpu-sample-stage] FAILED: {exc}")
        return 1

    print(f"[first-gpu-sample-stage] capture_root={result['capture_root']}")
    print(f"[first-gpu-sample-stage] preflight_status={result['preflight_status']}")
    print(
        "[first-gpu-sample-stage] source_video_preflight_status="
        + str(result.get("source_video_preflight_status"))
    )
    print(f"[first-gpu-sample-stage] candidate_audit_status={result['candidate_audit_status']}")
    blockers = result.get("candidate_audit_blockers") or []
    if blockers:
        print("[first-gpu-sample-stage] candidate_audit_blockers=" + ",".join(blockers))
    local_rehearsal = result.get("local_webapp_rehearsal_request") or {}
    if local_rehearsal.get("status") == "staged":
        print(
            "[first-gpu-sample-stage] local_webapp_rehearsal_staged_inputs="
            + str(local_rehearsal.get("staged_inputs_path"))
        )
    if result.get("simulation_automation_run"):
        print(
            "[first-gpu-sample-stage] simulation_automation_status="
            + str(result.get("simulation_automation_status"))
        )
        gpu_blockers = result.get("gpu_handoff_blockers") or []
        if gpu_blockers:
            print("[first-gpu-sample-stage] gpu_handoff_blockers=" + ",".join(gpu_blockers))
        hard_blockers = result.get("gpu_handoff_hard_preflight_blockers") or []
        if hard_blockers:
            print(
                "[first-gpu-sample-stage] gpu_handoff_hard_preflight_blockers="
                + ",".join(hard_blockers)
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
