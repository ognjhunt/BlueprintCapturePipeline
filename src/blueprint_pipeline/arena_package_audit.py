"""Deterministic artifact and proof-boundary audit for Arena packages."""

from __future__ import annotations

import argparse
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context


ARENA_PACKAGE_AUDIT_SCHEMA_VERSION = "arena_package_proof_boundary_audit.v1"

REQUIRED_ARTIFACTS = (
    "arena_eval_schedule.json",
    "arena_eval_retry_queue.json",
    "arena_eval_cost_ledger.json",
    "arena_eval_resume_manifest.json",
    "policy_adapter_manifest.json",
    "arena_result_ingest_ledger.json",
    "arena_artifact_checksums.json",
    "arena_eval_metrics.json",
    "normalized_attempt_trace.json",
    "normalized_attempt_trace.jsonl",
    "failure_labels.json",
    "failure_labels.jsonl",
    "clips_manifest.json",
    "rollout_vision_labels.json",
    "review_resolution_ledger.json",
    "accepted_failure_labels.json",
    "prediction_outcome_ledger.json",
    "calibration_report.json",
    "breakage_library.json",
    "arena_rerun_plan.json",
    "arena_rerun_lineage.json",
    "customer_handoff_report.md",
    "customer_handoff_report.json",
    "entitlement_check.json",
    "retention_policy.json",
    "egress_estimate.json",
    "delivery_manifest.json",
    "signed_access_manifest.json",
    "live_operator_ledger.json",
    "dataset_card.json",
    "license_manifest.json",
    "optional_export_manifest.json",
    "package_index.json",
    "checksums.json",
    "archive_manifest.json",
    "post_training_data_package_export_manifest.json",
    "arena_result_ingest_run_manifest.json",
)

JOB_ARTIFACTS = (
    "job_request.json",
    "simulator_service_result.json",
    "evaluation_result.json",
    "live_eval_closure_manifest.json",
    "proof_boundary.json",
    "job_run_manifest.json",
)

FORBIDDEN_PROOF_TRUE_FIELDS = (
    "simulator_execution_proven",
    "robot_policy_execution_proven",
    "rank_fidelity_result_proven",
    "physics_contact_validated",
    "non_ranking_operational_claim_validated",
    "public_claim_upgrade_allowed",
)

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "arena_package_artifact_assertion_and_proof_boundary_audit",
    "repo_local_only": True,
    "simulator_execution_proven": False,
    "robot_policy_execution_proven": False,
    "rank_fidelity_result_proven": False,
    "physics_contact_validated": False,
    "non_ranking_operational_claim_validated": False,
    "public_claim_upgrade_allowed": False,
}


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
    out: List[str] = []
    seen: set[str] = set()
    for item in values:
        text = _string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(base_dir: Path, relative_path: str) -> Dict[str, Any]:
    path = base_dir / relative_path
    return {
        "path": relative_path,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": _sha_file(path) if path.is_file() else None,
    }


def _latest_arena_job_dir(pipeline_dir: Path) -> Path | None:
    jobs_root = pipeline_dir / "robot_eval_jobs"
    if not jobs_root.is_dir():
        return None
    candidates = [
        path
        for path in jobs_root.iterdir()
        if path.is_dir() and (path / "arena_result_ingest_run_manifest.json").is_file()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _resolve_package_dir(capture_root: Path, package_dir: str | Path | None) -> Path:
    if package_dir:
        return Path(package_dir).resolve()
    context = resolve_local_capture_context(capture_root)
    latest_job = _latest_arena_job_dir(context.pipeline_root)
    if latest_job:
        return latest_job
    return context.pipeline_root / "arena_eval_package"


def _closure_allows_proof_field(package_dir: Path, field: str) -> bool:
    closure = _read_optional_mapping(package_dir / "live_eval_closure_manifest.json")
    closure_boundary = _mapping(closure.get("proof_boundary"))
    return (
        closure.get("status") == "live_end_to_end_verified"
        and bool(closure.get("live_end_to_end_verified"))
        and bool(closure_boundary.get(field))
    )


def _proof_field_violations(package_dir: Path, relative_paths: Sequence[str]) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    for relative_path in relative_paths:
        path = package_dir / relative_path
        if path.suffix != ".json":
            continue
        try:
            payload = _read_optional_mapping(path)
        except ValueError:
            continue
        if not payload:
            continue
        for field in FORBIDDEN_PROOF_TRUE_FIELDS:
            if bool(payload.get(field)):
                if _closure_allows_proof_field(package_dir, field):
                    continue
                violations.append(
                    {
                        "artifact": relative_path,
                        "field": field,
                        "value": True,
                        "reason": "proof_boolean_true_without_live_eval_closure",
                    }
                )
            boundary = _mapping(payload.get("claim_boundary") or payload.get("proof_boundary"))
            if bool(boundary.get(field)):
                if _closure_allows_proof_field(package_dir, field):
                    continue
                violations.append(
                    {
                        "artifact": relative_path,
                        "field": f"claim_boundary.{field}",
                        "value": True,
                        "reason": "claim_boundary_upgraded_without_live_eval_closure",
                    }
                )
    return violations


def build_arena_package_proof_boundary_audit(
    *,
    capture_root: str | Path,
    package_dir: str | Path | None = None,
    expected_scenario_count: int = 500,
    require_job_artifacts: bool = False,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    resolved_package_dir = _resolve_package_dir(context.capture_root, package_dir)
    generated_at = utc_now_iso()
    required = list(REQUIRED_ARTIFACTS)
    if require_job_artifacts:
        required.extend(JOB_ARTIFACTS)

    artifact_assertions = {
        name: _artifact(resolved_package_dir, name)
        for name in required
    }
    blockers = [
        f"missing_artifact:{name}"
        for name, artifact in artifact_assertions.items()
        if not artifact["exists"]
    ]

    schedule = _read_optional_mapping(resolved_package_dir / "arena_eval_schedule.json")
    trace = _read_optional_mapping(resolved_package_dir / "normalized_attempt_trace.json")
    labels = _read_optional_mapping(resolved_package_dir / "failure_labels.json")
    clips = _read_optional_mapping(resolved_package_dir / "clips_manifest.json")
    package_export = _read_optional_mapping(
        resolved_package_dir / "post_training_data_package_export_manifest.json"
    )
    archive = _read_optional_mapping(resolved_package_dir / "archive_manifest.json")
    delivery = _read_optional_mapping(resolved_package_dir / "delivery_manifest.json")
    signed_access = _read_optional_mapping(resolved_package_dir / "signed_access_manifest.json")
    operators = _read_optional_mapping(resolved_package_dir / "live_operator_ledger.json")
    review = _read_optional_mapping(resolved_package_dir / "review_resolution_ledger.json")
    rerun = _read_optional_mapping(resolved_package_dir / "arena_rerun_plan.json")

    if int(schedule.get("scenario_count") or 0) != expected_scenario_count:
        blockers.append("arena_schedule_scenario_count_mismatch")
    if int(schedule.get("shard_count") or 0) <= 0:
        blockers.append("arena_schedule_shards_missing")
    if int(trace.get("attempt_count") or 0) <= 0:
        blockers.append("normalized_attempt_trace_empty")
    if int(clips.get("clip_count") or 0) != int(trace.get("attempt_count") or -1):
        blockers.append("clip_count_does_not_match_attempt_count")
    if not labels and int(trace.get("attempt_count") or 0) > 0:
        blockers.append("failure_labels_manifest_missing_or_invalid")
    if package_export.get("status") != "export_ready_review_required":
        blockers.append("post_training_data_package_not_export_ready_review_required")
    if not _mapping(archive.get("archive")).get("exists"):
        blockers.append("post_training_data_package_archive_missing")
    if delivery.get("status") != "local_delivery_bundle_ready":
        blockers.append("local_delivery_bundle_not_ready")
    if not signed_access:
        blockers.append("signed_access_manifest_missing")
    if not operators:
        blockers.append("live_operator_ledger_missing")
    if review.get("status") not in {
        "accepted_labels_ready",
        "review_required",
        "no_review_required",
    }:
        blockers.append("review_resolution_status_unexpected")
    if rerun.get("status") not in {
        "reruns_queued",
        "no_eligible_reruns",
        "blocked_cost_budget_exhausted",
    }:
        blockers.append("rerun_plan_status_unexpected")

    proof_paths = [
        *REQUIRED_ARTIFACTS,
        *JOB_ARTIFACTS,
    ]
    proof_violations = _proof_field_violations(resolved_package_dir, proof_paths)
    blockers.extend(
        f"proof_boundary_violation:{item['artifact']}:{item['field']}"
        for item in proof_violations
    )

    external_blockers = []
    if signed_access.get("status") == "blocked":
        external_blockers.append(
            {
                "system": "storage_delivery",
                "status": "blocked_or_not_requested",
                "blockers": _string_list(signed_access.get("blockers")),
                "next_input_needed": (
                    "Set BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD=true, pass "
                    "--allow-delivery-upload, and provide a delivery command when live upload "
                    "or signed URLs are required."
                ),
            }
        )
    if operators.get("status") in {"blocked", "not_requested"}:
        external_blockers.append(
            {
                "system": "live_agents_codex_operators",
                "status": operators.get("status"),
                "blockers": _string_list(operators.get("blockers")),
                "next_input_needed": (
                    "Set the live operator env gates, provide SDK dependencies and credentials, "
                    "and pass the live operator CLI flags when real SDK execution is required."
                ),
            }
        )

    audit = {
        "schema_version": ARENA_PACKAGE_AUDIT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed" if not blockers else "blocked",
        "capture_root": str(context.capture_root),
        "package_dir": str(resolved_package_dir),
        "expected_scenario_count": expected_scenario_count,
        "blockers": blockers,
        "artifact_assertions": artifact_assertions,
        "summary": {
            "scenario_count": int(schedule.get("scenario_count") or 0),
            "shard_count": int(schedule.get("shard_count") or 0),
            "attempt_count": int(trace.get("attempt_count") or 0),
            "failure_label_count": int(labels.get("label_count") or 0),
            "clip_count": int(clips.get("clip_count") or 0),
            "package_status": package_export.get("status"),
            "delivery_status": delivery.get("status"),
            "signed_access_status": signed_access.get("status"),
            "operator_status": operators.get("status"),
            "review_status": review.get("status"),
            "rerun_status": rerun.get("status"),
        },
        "proof_boundary_violations": proof_violations,
        "external_blockers": external_blockers,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    path = Path(output_path) if output_path else resolved_package_dir / "arena_package_proof_boundary_audit.json"
    ensure_dir(path.parent)
    write_json(path, audit)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit Arena package artifacts and proof-boundary booleans"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--package-dir")
    parser.add_argument("--expected-scenario-count", type=int, default=500)
    parser.add_argument("--require-job-artifacts", action="store_true")
    parser.add_argument("--output-path")
    args = parser.parse_args(argv)
    result = build_arena_package_proof_boundary_audit(
        capture_root=args.capture_root,
        package_dir=args.package_dir,
        expected_scenario_count=args.expected_scenario_count,
        require_job_artifacts=args.require_job_artifacts,
        output_path=args.output_path,
    )
    manifest_path = args.output_path or str(
        Path(result["package_dir"]) / "arena_package_proof_boundary_audit.json"
    )
    print(f"[arena-package-audit] manifest={manifest_path}")
    print(f"[arena-package-audit] status={result['status']}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
