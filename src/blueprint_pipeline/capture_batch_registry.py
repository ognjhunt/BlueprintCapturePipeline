"""Site/capture batch registry for production handoff lanes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context


SITE_CAPTURE_BATCH_REGISTRY_SCHEMA_VERSION = "site_capture_batch_registry.v1"

STAGE_ORDER = (
    "privacy",
    "worldlabs",
    "materialization",
    "cpu_preflight",
    "gpu_handoff",
    "eval_result",
    "data_package_export",
)

COMPLETE_STATUSES = {
    "ready",
    "passed",
    "completed",
    "complete",
    "person_removed",
    "no_people_detected",
    "face_anonymized_fallback",
    "full_frame_redacted_local_proof",
    "export_ready_review_required",
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _complete(status: Any) -> bool:
    return _string(status).lower() in COMPLETE_STATUSES


def _status(
    status: str,
    *,
    artifact: Path | None = None,
    detail: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": status}
    if artifact is not None:
        payload["artifact_path"] = str(artifact)
        payload["artifact_exists"] = artifact.is_file()
    if detail:
        payload.update(dict(detail))
    return payload


def _latest_job_dir(pipeline_dir: Path) -> Path | None:
    jobs_root = pipeline_dir / "robot_eval_jobs"
    if not jobs_root.is_dir():
        return None
    jobs = sorted(
        (path for path in jobs_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return jobs[0] if jobs else None


def _stage_statuses(capture_root: Path) -> Dict[str, Dict[str, Any]]:
    context = resolve_local_capture_context(capture_root)
    pipeline = context.pipeline_root
    automation = pipeline / "simulation_automation"
    latest_job = _latest_job_dir(pipeline)

    privacy_path = pipeline / "privacy_processing_manifest.json"
    privacy = _read_optional_mapping(privacy_path)
    worldlabs_request_path = pipeline / "worldlabs_request_manifest.json"
    operation_path = pipeline / "worldlabs_operation_manifest.json"
    world_path = pipeline / "worldlabs_world_manifest.json"
    materialization_path = pipeline / "worldlabs_assets" / "materialized_assets_manifest.json"
    materialization = _read_optional_mapping(materialization_path)
    cpu_path = automation / "cpu_preflight_manifest.json"
    cpu = _read_optional_mapping(cpu_path)
    gpu_path = automation / "gpu_handoff_packet.json"
    gpu = _read_optional_mapping(gpu_path)
    eval_path = latest_job / "evaluation_result.json" if latest_job else None
    eval_result = _read_optional_mapping(eval_path) if eval_path else {}
    package_path = (
        latest_job / "post_training_data_package_export_manifest.json" if latest_job else None
    )
    data_package = _read_optional_mapping(package_path) if package_path else {}

    statuses: Dict[str, Dict[str, Any]] = {}
    statuses["privacy"] = _status(
        "complete" if _complete(privacy.get("status")) else "blocked" if privacy else "pending",
        artifact=privacy_path,
        detail={"raw_status": privacy.get("status")},
    )
    statuses["worldlabs"] = _status(
        "complete"
        if operation_path.is_file() and world_path.is_file()
        else "ready_for_generation"
        if worldlabs_request_path.is_file()
        else "pending",
        artifact=world_path,
        detail={
            "request_manifest_exists": worldlabs_request_path.is_file(),
            "operation_manifest_exists": operation_path.is_file(),
        },
    )
    statuses["materialization"] = _status(
        "complete"
        if _string(materialization.get("status")) in {"complete", "partial", "complete_with_download_failures"}
        else "blocked"
        if materialization
        else "pending",
        artifact=materialization_path,
        detail={"raw_status": materialization.get("status")},
    )
    statuses["cpu_preflight"] = _status(
        "complete"
        if bool(cpu.get("ready_for_owner_gpu_preflight"))
        else "blocked"
        if cpu
        else "pending",
        artifact=cpu_path,
        detail={"raw_status": cpu.get("status")},
    )
    gpu_blockers = [str(item) for item in gpu.get("blockers") or []]
    statuses["gpu_handoff"] = _status(
        "ready_except_owner_gpu"
        if gpu.get("status") == "ready_for_owner_gpu_preflight_handoff"
        and gpu_blockers == ["owner_gpu_simulator_execution_not_run"]
        else "complete"
        if gpu.get("owner_gpu_simulator_execution_proven")
        else "blocked"
        if gpu
        else "pending",
        artifact=gpu_path,
        detail={"raw_status": gpu.get("status"), "blockers": gpu_blockers},
    )
    statuses["eval_result"] = _status(
        "complete"
        if _complete(eval_result.get("status"))
        or _string(eval_result.get("status")) == "completed_with_failures"
        else "blocked"
        if eval_result
        else "pending",
        artifact=eval_path,
        detail={"raw_status": eval_result.get("status")},
    )
    statuses["data_package_export"] = _status(
        "complete"
        if _complete(data_package.get("status"))
        else "blocked"
        if data_package
        else "pending",
        artifact=package_path,
        detail={"raw_status": data_package.get("status")},
    )
    return statuses


def _site_id(context: Any, descriptor: Mapping[str, Any], raw: Mapping[str, Any]) -> str:
    for payload in (
        _mapping(descriptor.get("metadata")).get("site_identity"),
        descriptor.get("site_identity"),
        raw.get("site_identity"),
    ):
        value = _string(_mapping(payload).get("site_id"))
        if value:
            return value
    return context.scene_id


def _existing_capture(
    existing: Mapping[str, Any],
    *,
    site_id: str,
    capture_id: str,
) -> Dict[str, Any]:
    captures = _mapping(_mapping(_mapping(existing.get("sites")).get(site_id)).get("captures"))
    return _mapping(captures.get(capture_id))


def _attempts(
    *,
    previous: Mapping[str, Any],
    statuses: Mapping[str, Mapping[str, Any]],
    retry_stage: str | None,
) -> Dict[str, Dict[str, Any]]:
    previous_attempts = _mapping(previous.get("attempts"))
    out: Dict[str, Dict[str, Any]] = {}
    for stage in STAGE_ORDER:
        previous_stage = _mapping(previous_attempts.get(stage))
        prior_count = int(previous_stage.get("attempt_count") or 1)
        count = prior_count + 1 if retry_stage == stage else prior_count
        out[stage] = {
            "attempt_count": count,
            "last_status": statuses[stage]["status"],
            "retryable": statuses[stage]["status"] in {"blocked", "pending", "ready_for_generation"},
        }
    return out


def _resume(statuses: Mapping[str, Mapping[str, Any]], retry_stage: str | None) -> Dict[str, Any]:
    if retry_stage:
        return {"resume_from_stage": retry_stage, "next_stage": retry_stage}
    for stage in STAGE_ORDER:
        if statuses[stage]["status"] in {"blocked", "pending", "ready_for_generation", "queued_for_retry"}:
            return {"resume_from_stage": stage, "next_stage": stage}
    if statuses["gpu_handoff"]["status"] == "ready_except_owner_gpu":
        return {
            "resume_from_stage": "owner_gpu_simulator_execution",
            "next_stage": "owner_gpu_simulator_execution",
        }
    return {"resume_from_stage": None, "next_stage": None}


def _quarantine_capture(
    *,
    raw_root: str | Path,
    error: BaseException,
    capture_id: str | None,
    site_id: str | None,
) -> Dict[str, Any]:
    """Record why a single malformed capture was skipped from the registry build.

    Mirrors the per-request isolation used by the robot-eval job inbox runner: one
    bad capture is quarantined with a bounded reason instead of aborting the whole
    batch, so the good captures still make it into the registry.
    """

    return {
        "capture_root": str(raw_root),
        "capture_id": capture_id,
        "site_id": site_id,
        "status": "skipped",
        "error_type": type(error).__name__,
        "error": str(error),
        "reason": "skipped_after_capture_build_error",
    }


def update_capture_batch_registry(
    *,
    capture_roots: Sequence[str | Path],
    registry_path: str | Path,
    resume: bool = True,
    retry_stage: str | None = None,
) -> Dict[str, Any]:
    if retry_stage is not None and retry_stage not in STAGE_ORDER:
        raise ValueError(f"retry_stage must be one of {', '.join(STAGE_ORDER)}")
    output_path = Path(registry_path)
    existing = _read_optional_mapping(output_path) if resume else {}
    generated_at = utc_now_iso()
    registry: Dict[str, Any] = {
        "schema_version": SITE_CAPTURE_BATCH_REGISTRY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "resume_enabled": resume,
        "retry_stage": retry_stage,
        "stage_order": list(STAGE_ORDER),
        "sites": {},
        "skipped_captures": [],
        "skipped_capture_count": 0,
        "skipped_capture_ids": [],
    }
    skipped_captures: List[Dict[str, Any]] = []
    for raw_root in capture_roots:
        resolved_capture_id: str | None = None
        resolved_site_id: str | None = None
        try:
            context = resolve_local_capture_context(raw_root)
            resolved_capture_id = context.capture_id
            descriptor = _read_optional_mapping(context.descriptor_path)
            raw_manifest = _read_optional_mapping(context.raw_root / "manifest.json")
            site_id = _site_id(context, descriptor, raw_manifest)
            resolved_site_id = site_id
            statuses = _stage_statuses(context.capture_root)
            if retry_stage:
                previous_status = statuses[retry_stage]["status"]
                statuses[retry_stage] = {
                    **statuses[retry_stage],
                    "status": "queued_for_retry",
                    "previous_status": previous_status,
                }
            previous = _existing_capture(
                existing, site_id=site_id, capture_id=context.capture_id
            )
            attempts = _attempts(
                previous=previous, statuses=statuses, retry_stage=retry_stage
            )
            capture_entry = {
                "capture_id": context.capture_id,
                "capture_root": str(context.capture_root),
                "stage_statuses": statuses,
                "attempts": attempts,
                "resume": _resume(statuses, retry_stage),
            }
        except Exception as exc:  # noqa: BLE001 - isolate one malformed capture from the batch
            skipped_captures.append(
                _quarantine_capture(
                    raw_root=raw_root,
                    error=exc,
                    capture_id=resolved_capture_id,
                    site_id=resolved_site_id,
                )
            )
            continue
        site = registry["sites"].setdefault(
            site_id,
            {
                "site_id": site_id,
                "scene_id": context.scene_id,
                "captures": {},
            },
        )
        site["captures"][context.capture_id] = capture_entry
    registry["skipped_captures"] = skipped_captures
    registry["skipped_capture_count"] = len(skipped_captures)
    registry["skipped_capture_ids"] = [
        entry["capture_id"] or entry["capture_root"] for entry in skipped_captures
    ]
    ensure_dir(output_path.parent)
    write_json(output_path, registry)
    return registry


def _discover_capture_roots(storage_root: Path) -> List[Path]:
    return sorted(path.parent for path in storage_root.glob("**/capture_descriptor.json"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or update a site/capture batch registry")
    parser.add_argument("--capture-root", action="append", default=[])
    parser.add_argument("--storage-root")
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-stage", choices=STAGE_ORDER)
    args = parser.parse_args(argv)
    capture_roots = [Path(path) for path in args.capture_root]
    if args.storage_root:
        capture_roots.extend(_discover_capture_roots(Path(args.storage_root)))
    if not capture_roots:
        raise SystemExit("provide at least one --capture-root or --storage-root")
    registry = update_capture_batch_registry(
        capture_roots=capture_roots,
        registry_path=args.registry_path,
        resume=not args.no_resume,
        retry_stage=args.retry_stage,
    )
    print(f"[capture-batch-registry] registry={args.registry_path}")
    print(f"[capture-batch-registry] site_count={len(registry['sites'])}")
    print(
        "[capture-batch-registry] skipped_capture_count="
        f"{registry['skipped_capture_count']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
