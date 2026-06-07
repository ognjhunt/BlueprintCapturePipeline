"""Validate and optionally stage live external inputs for the control plane.

The intake command is a preflight for real external handoffs. It can inspect a
WebApp ``robot_eval_job_request.v1`` file and an owner-system Arena result
directory against the current live control-plane manifest. It never runs live
simulators, calls providers, uploads storage, or promotes proof claims.
"""

from __future__ import annotations

import argparse
import json
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .live_pipeline_control_plane import (
    ARENA_RESULT_ARTIFACT_NAMES,
    JOB_REQUEST_INBOX_ENV,
    LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION,
    LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION,
    WEBAPP_JOB_REQUEST_QUEUE_CONTRACT,
    WEBAPP_JOB_REQUEST_SCHEMA_VERSION,
    WEBAPP_UPSTREAM_REQUIRED_FIELDS,
)


LIVE_PIPELINE_INPUT_INTAKE_SCHEMA_VERSION = "blueprint_live_pipeline_input_intake.v1"


def _string(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_mapping(path: Path) -> Dict[str, Any]:
    payload = read_json_any(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return dict(payload)


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any] | None:
    if payload.get("queue_contract") == WEBAPP_JOB_REQUEST_QUEUE_CONTRACT:
        request = payload.get("job_request")
        if isinstance(request, Mapping) and request.get("schema_version") == WEBAPP_JOB_REQUEST_SCHEMA_VERSION:
            return dict(request)
        return None
    if payload.get("schema_version") == WEBAPP_JOB_REQUEST_SCHEMA_VERSION:
        return dict(payload)
    return None


def _field_value(request: Mapping[str, Any], field: str) -> str | None:
    source = _mapping(request.get("source"))
    for candidate in (request, source):
        value = _string(candidate.get(field))
        if value:
            return value
    if field == "request_id":
        owner_system = _mapping(request.get("owner_system"))
        value = _string(owner_system.get("request_id"))
        if value:
            return value
    return None


def _path_matches(value: str | None, expected: Path | None) -> bool:
    if not value or expected is None:
        return False
    try:
        return Path(value).resolve() == expected.resolve()
    except (OSError, RuntimeError):
        return False


def _load_control_plane_manifest(path: Path) -> Dict[str, Any]:
    manifest = _read_mapping(path)
    if manifest.get("schema_version") != LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION:
        raise ValueError(f"Expected {LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION} at {path}")
    return manifest


def _audit_webapp_request(
    *,
    request_path: Path | None,
    expected_capture_root: Path | None,
    configured_inbox: Path | None,
) -> Dict[str, Any]:
    if request_path is None:
        return {
            "status": "not_provided",
            "ready": False,
            "path": None,
            "blockers": ["webapp_job_request_not_provided"],
        }
    if not request_path.is_file():
        return {
            "status": "blocked",
            "ready": False,
            "path": str(request_path),
            "blockers": ["webapp_job_request_missing"],
        }
    try:
        payload = _read_mapping(request_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "blocked",
            "ready": False,
            "path": str(request_path),
            "blockers": [f"webapp_job_request_read_failed:{type(exc).__name__}"],
        }
    request = _request_from_payload(payload)
    if request is None:
        return {
            "status": "blocked",
            "ready": False,
            "path": str(request_path),
            "blockers": ["not_robot_eval_job_request_v1_or_queue_envelope"],
            "sha256": _sha_file(request_path),
        }
    site_package = _mapping(request.get("site_package"))
    request_capture_root = _string(site_package.get("capture_root")) or None
    fields_present = {
        field: bool(_field_value(request, field))
        for field in WEBAPP_UPSTREAM_REQUIRED_FIELDS
    }
    missing_fields = [
        field for field, present in fields_present.items() if not present
    ]
    capture_root_matches = _path_matches(request_capture_root, expected_capture_root)
    blockers: List[str] = []
    if missing_fields:
        blockers.append("missing_required_webapp_ids")
    if not capture_root_matches:
        blockers.append("request_capture_root_does_not_match_control_plane")
    job_id = _string(request.get("job_id")) or request_path.stem
    return {
        "status": "ready" if not blockers else "blocked",
        "ready": not blockers,
        "path": str(request_path),
        "sha256": _sha_file(request_path),
        "job_id": job_id,
        "schema_version": request.get("schema_version"),
        "fields_present": fields_present,
        "missing_fields": missing_fields,
        "request_capture_root_configured": bool(request_capture_root),
        "request_capture_root_matches_control_plane": capture_root_matches,
        "configured_capture_root": str(expected_capture_root) if expected_capture_root else None,
        "configured_inbox": str(configured_inbox) if configured_inbox else None,
        "blockers": blockers,
        "metadata_only": True,
        "proof_boundary": (
            "Valid WebApp request metadata proves handoff shape only; the control plane still "
            "owns scheduling and proof-boundary enforcement."
        ),
    }


def _audit_arena_results(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {
            "status": "not_provided",
            "ready": False,
            "arena_results_dir": None,
            "blockers": ["arena_results_dir_not_provided"],
            "json_artifact_count": 0,
            "recognized_artifacts": [],
        }
    if not path.is_dir():
        return {
            "status": "blocked",
            "ready": False,
            "arena_results_dir": str(path),
            "blockers": ["arena_results_dir_missing"],
            "json_artifact_count": 0,
            "recognized_artifacts": [],
        }
    json_artifacts = sorted(item for item in path.rglob("*.json") if item.is_file())
    recognized_names = set(ARENA_RESULT_ARTIFACT_NAMES)
    recognized = [
        str(item.relative_to(path))
        for item in json_artifacts
        if item.name in recognized_names
    ]
    blockers: List[str] = []
    if not json_artifacts:
        blockers.append("arena_results_dir_has_no_json_artifacts")
    return {
        "status": "ready_for_ingest" if not blockers else "blocked",
        "ready": not blockers,
        "arena_results_dir": str(path),
        "blockers": blockers,
        "json_artifact_count": len(json_artifacts),
        "recognized_artifacts": recognized,
        "artifact_sample": [str(item.relative_to(path)) for item in json_artifacts[:20]],
        "truncated_artifact_sample": len(json_artifacts) > 20,
        "proof_boundary": (
            "Arena result artifacts are ingest inputs only; they are not simulator execution, "
            "robot policy, contact, safety, or readiness proof by themselves."
        ),
    }


def _stage_webapp_request(
    *,
    request_path: Path,
    audit: Mapping[str, Any],
    inbox: Path | None,
    overwrite: bool,
) -> Dict[str, Any]:
    if not audit.get("ready"):
        return {
            "status": "blocked",
            "performed": False,
            "blockers": ["webapp_request_not_ready_for_staging"],
        }
    if inbox is None:
        return {
            "status": "blocked",
            "performed": False,
            "blockers": [f"missing_env_or_manifest_{JOB_REQUEST_INBOX_ENV}"],
        }
    job_id = _string(audit.get("job_id")) or request_path.stem
    target = inbox / f"{job_id}.json"
    ensure_dir(inbox)
    blockers: List[str] = []
    if target.exists() and not overwrite:
        blockers.append("target_request_already_exists")
    if blockers:
        return {
            "status": "blocked",
            "performed": False,
            "target_path": str(target),
            "blockers": blockers,
        }
    shutil.copy2(request_path, target)
    return {
        "status": "staged",
        "performed": True,
        "target_path": str(target),
        "sha256": _sha_file(target),
        "blockers": [],
        "proof_boundary": "staging copies an input request only and does not process the job",
    }


def _write_staged_inputs(
    *,
    path: Path,
    manifest_path: Path,
    capture_root: Path | None,
    webapp_audit: Mapping[str, Any],
    webapp_staging: Mapping[str, Any],
    arena_audit: Mapping[str, Any],
    stage_arena_results: bool,
) -> Dict[str, Any]:
    arena_ready = bool(arena_audit.get("ready"))
    webapp_ready = bool(webapp_audit.get("ready"))
    webapp_staged = bool(webapp_staging.get("performed"))
    blockers: List[str] = []
    if stage_arena_results and not arena_ready:
        blockers.append("arena_results_not_ready_for_staging")
    if webapp_staging.get("status") == "blocked":
        blockers.append("webapp_request_not_staged")
    if blockers:
        return {
            "status": "blocked",
            "performed": False,
            "path": str(path),
            "blockers": blockers,
        }
    if not stage_arena_results and not webapp_staged:
        return {
            "status": "not_requested",
            "performed": False,
            "path": str(path),
            "blockers": [],
        }
    payload = {
        "schema_version": LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "source_intake_manifest_path": str(manifest_path),
        "configured_capture_root": str(capture_root) if capture_root else None,
        "webapp_request": {
            "ready": webapp_ready and webapp_staged,
            "staged": webapp_staged,
            "job_id": webapp_audit.get("job_id"),
            "path": webapp_audit.get("path"),
            "target_path": webapp_staging.get("target_path"),
            "sha256": webapp_staging.get("sha256") or webapp_audit.get("sha256"),
        },
        "arena_results": {
            "ready": arena_ready if stage_arena_results else False,
            "arena_results_dir": arena_audit.get("arena_results_dir")
            if stage_arena_results
            else None,
            "json_artifact_count": arena_audit.get("json_artifact_count", 0)
            if stage_arena_results
            else 0,
            "recognized_artifacts": arena_audit.get("recognized_artifacts", [])
            if stage_arena_results
            else [],
        },
        "proof_boundary": {
            "staged_inputs_are_pointers_only": True,
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    ensure_dir(path.parent)
    write_json(path, payload)
    return {
        "status": "staged",
        "performed": True,
        "path": str(path),
        "blockers": [],
        "arena_results_staged": bool(stage_arena_results and arena_ready),
        "webapp_request_staged": webapp_staged,
    }


def build_live_pipeline_input_intake(
    *,
    manifest_path: str | Path,
    webapp_job_request: str | Path | None = None,
    arena_results_dir: str | Path | None = None,
    stage_webapp_request: bool = False,
    stage_arena_results: bool = False,
    overwrite: bool = False,
    output_path: str | Path | None = None,
    staged_inputs_path: str | Path | None = None,
) -> Dict[str, Any]:
    resolved_manifest_path = Path(manifest_path).resolve()
    manifest = _load_control_plane_manifest(resolved_manifest_path)
    capture_root = Path(manifest["capture_root"]).resolve() if manifest.get("capture_root") else None
    inbox = Path(manifest["job_request_inbox"]).resolve() if manifest.get("job_request_inbox") else None
    request_path = Path(webapp_job_request).resolve() if webapp_job_request else None
    results_path = Path(arena_results_dir).resolve() if arena_results_dir else None
    generated_at = utc_now_iso()

    webapp_audit = _audit_webapp_request(
        request_path=request_path,
        expected_capture_root=capture_root,
        configured_inbox=inbox,
    )
    arena_audit = _audit_arena_results(results_path)
    staging = (
        _stage_webapp_request(
            request_path=request_path or Path(),
            audit=webapp_audit,
            inbox=inbox,
            overwrite=overwrite,
        )
        if stage_webapp_request
        else {
            "status": "not_requested",
            "performed": False,
            "blockers": [],
        }
    )
    input_blockers: List[str] = []
    if webapp_job_request and not webapp_audit.get("ready"):
        input_blockers.extend(f"webapp:{blocker}" for blocker in webapp_audit.get("blockers", []))
    if arena_results_dir and not arena_audit.get("ready"):
        input_blockers.extend(f"arena:{blocker}" for blocker in arena_audit.get("blockers", []))
    if staging.get("blockers"):
        input_blockers.extend(f"staging:{blocker}" for blocker in staging.get("blockers", []))

    status = "ready_for_control_plane"
    if input_blockers:
        status = "blocked"
    elif not webapp_job_request and not arena_results_dir:
        status = "waiting_for_inputs"
    elif stage_webapp_request and staging.get("performed"):
        status = "staged_for_control_plane"
    elif stage_arena_results and arena_audit.get("ready"):
        status = "staged_for_control_plane"

    if output_path:
        path = Path(output_path).resolve()
    else:
        path = resolved_manifest_path.parent / "live_pipeline_input_intake_audit.json"
    staged_path = (
        Path(staged_inputs_path).resolve()
        if staged_inputs_path
        else resolved_manifest_path.parent / "live_pipeline_staged_inputs.json"
    )
    staged_inputs = _write_staged_inputs(
        path=staged_path,
        manifest_path=path,
        capture_root=capture_root,
        webapp_audit=webapp_audit,
        webapp_staging=staging,
        arena_audit=arena_audit,
        stage_arena_results=stage_arena_results,
    )
    if staged_inputs.get("blockers"):
        input_blockers.extend(
            f"staged_inputs:{blocker}" for blocker in staged_inputs.get("blockers", [])
        )
        status = "blocked"

    intake = {
        "schema_version": LIVE_PIPELINE_INPUT_INTAKE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "manifest_path": str(resolved_manifest_path),
        "configured_capture_root": str(capture_root) if capture_root else None,
        "configured_job_request_inbox": str(inbox) if inbox else None,
        "webapp_job_request": webapp_audit,
        "arena_results": arena_audit,
        "webapp_staging": staging,
        "staged_inputs": staged_inputs,
        "input_blockers": input_blockers,
        "next_steps": [
            "Run blueprint-run-live-pipeline-control-plane after staging a WebApp request.",
            "Run blueprint-run-live-pipeline-control-plane after staging owner Arena artifacts.",
            "Run blueprint-audit-live-pipeline-proof-boundary after the control-plane pass.",
        ],
        "proof_boundary": {
            "intake_performs_live_actions": False,
            "webapp_truth_proven": bool(webapp_audit.get("ready")),
            "arena_results_ready_for_ingest": bool(arena_audit.get("ready")),
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    ensure_dir(path.parent)
    intake["output_path"] = str(path)
    write_json(path, intake)
    return intake


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and optionally stage live WebApp/Arena inputs for the control plane."
    )
    parser.add_argument(
        "--manifest-path",
        default="/var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json",
    )
    parser.add_argument("--webapp-job-request")
    parser.add_argument("--arena-results-dir")
    parser.add_argument("--stage-webapp-request", action="store_true")
    parser.add_argument("--stage-arena-results", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-path")
    parser.add_argument("--staged-inputs-path")
    args = parser.parse_args(argv)
    result = build_live_pipeline_input_intake(
        manifest_path=args.manifest_path,
        webapp_job_request=args.webapp_job_request,
        arena_results_dir=args.arena_results_dir,
        stage_webapp_request=args.stage_webapp_request,
        stage_arena_results=args.stage_arena_results,
        overwrite=args.overwrite,
        output_path=args.output_path,
        staged_inputs_path=args.staged_inputs_path,
    )
    print(f"[live-pipeline-input-intake] audit={result['output_path']}")
    print(f"[live-pipeline-input-intake] status={result['status']}")
    if result["input_blockers"]:
        print(f"[live-pipeline-input-intake] blockers={len(result['input_blockers'])}")
    return 0 if not result["input_blockers"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
