#!/usr/bin/env python3
"""Build the sim-only beta release readiness go/no-go report.

This gate is stricter than the local sim-only beta gate. It requires local
post-upload autonomy proof plus production-style WebApp-to-Pipeline forwarding
and deployment evidence. It does not execute live provider calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.common import ensure_dir, read_json_any, utc_now_iso, write_json  # noqa: E402


SCHEMA_VERSION = "blueprint.sim_only_beta_release_gate_report.v1"
READY_FORWARDING_STATUSES = {
    "ready_for_required_forwarding_with_probe",
}
READY_DEPLOYMENT_STATUSES = {"passed", "ready", "healthy", "verified"}


def _default_webapp_repo() -> Path:
    return ROOT.parent / "Blueprint-WebApp"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _load_mapping(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, "missing"
    payload = read_json_any(path)
    if not isinstance(payload, Mapping):
        return {}, "not_json_object"
    return dict(payload), None


def _string(value: Any) -> str:
    return str(value or "").strip()


def _bool(value: Any) -> bool:
    return value is True


def _origin_is_localhost(origin: str) -> bool:
    text = origin.lower()
    return "localhost" in text or "127.0.0.1" in text or "[::1]" in text


def _gate(gate_id: str, *, passed: bool, blockers: list[str], evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": gate_id,
        "status": "passed" if passed else "blocked",
        "blockers": blockers,
        "evidence": evidence,
    }


def _local_sim_only_gate(local_gate: Mapping[str, Any], load_error: str | None, path: Path) -> dict[str, Any]:
    blockers: list[str] = []
    if load_error:
        blockers.append(f"local_sim_only_gate_report_{load_error}")
    if local_gate.get("status") != "passed":
        blockers.append("local_sim_only_gate_not_passed")
    if local_gate.get("blockers"):
        blockers.append("local_sim_only_gate_has_blockers")

    proof_boundary = _mapping(local_gate.get("proof_boundary"))
    batch_closure = _mapping(local_gate.get("batch_closure"))
    robot_team_closure = _mapping(local_gate.get("robot_team_grade_closure"))
    visual_coverage = _mapping(batch_closure.get("visual_coverage"))
    scenario_eval_matrix = _mapping(local_gate.get("scenario_eval_matrix"))
    required_true = {
        "local_mujoco_simulator_execution_proven": proof_boundary.get(
            "local_mujoco_simulator_execution_proven"
        ),
        "semantic_spawn_target_coverage_complete": scenario_eval_matrix.get(
            "semantic_spawn_target_coverage_complete"
        ),
        "scenario_eval_run_coverage_complete": batch_closure.get(
            "scenario_eval_run_coverage_complete"
        ),
        "scenario_eval_run_id_coverage_exact": batch_closure.get(
            "scenario_eval_run_id_coverage_exact"
        ),
        "metric_coverage_complete": batch_closure.get("metric_coverage_complete"),
        "machine_trace_package_complete": batch_closure.get("machine_trace_package_complete"),
        "failure_label_coverage_complete": batch_closure.get("failure_label_coverage_complete"),
        "visual_review_coverage_complete": batch_closure.get(
            "visual_review_coverage_complete"
        ),
        "visual_recording_coverage_complete": visual_coverage.get(
            "all_required_runs_have_visual_recording"
        ),
        "visual_files_complete": visual_coverage.get("all_video_files_complete"),
        "sim_only_beta_core_complete": robot_team_closure.get("sim_only_beta_core_complete"),
    }
    for key, value in required_true.items():
        if value is not True:
            blockers.append(f"{key}_not_true")

    return _gate(
        "local_sim_only_post_upload_autonomy",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "path": str(path),
            "status": local_gate.get("status"),
            "attempt_count": batch_closure.get("attempt_count"),
            "required_true": required_true,
            "scenario_eval_matrix": scenario_eval_matrix,
            "proof_boundary": proof_boundary,
        },
    )


def _forwarding_gate(
    forwarding_report: Mapping[str, Any],
    load_error: str | None,
    path: Path,
    *,
    require_non_local_endpoint: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    if load_error:
        blockers.append(f"forwarding_preflight_report_{load_error}")

    status = _string(forwarding_report.get("status"))
    if status not in READY_FORWARDING_STATUSES:
        blockers.append("forwarding_preflight_not_ready_with_probe")
    if forwarding_report.get("blockers"):
        blockers.append("forwarding_preflight_has_blockers")
    if forwarding_report.get("warnings"):
        blockers.append("forwarding_preflight_has_warnings")
    if forwarding_report.get("forwarding_required") is not True:
        blockers.append("forwarding_not_required_in_report")
    if forwarding_report.get("endpoint_configured") is not True:
        blockers.append("forwarding_endpoint_not_configured")

    configured_env = _mapping(forwarding_report.get("configured_env"))
    forward_token = _mapping(configured_env.get("forward_token"))
    if forward_token.get("configured") is not True:
        blockers.append("forwarding_token_not_configured")

    forward_url = _mapping(configured_env.get("forward_url"))
    origin = _string(forward_url.get("origin"))
    if require_non_local_endpoint and origin and _origin_is_localhost(origin):
        blockers.append("forwarding_endpoint_is_localhost")

    probe = _mapping(forwarding_report.get("probe"))
    if probe.get("requested") is not True or probe.get("attempted") is not True:
        blockers.append("forwarding_probe_not_attempted")
    if probe.get("status") != "reachable" or probe.get("http_status") != 200:
        blockers.append("forwarding_probe_not_reachable")
    if probe.get("audit_status") != "staged_for_control_plane":
        blockers.append("forwarding_probe_audit_not_staged_for_control_plane")
    if probe.get("input_blockers_count") not in (0, None):
        blockers.append("forwarding_probe_input_blockers_present")

    proof_boundary = _mapping(forwarding_report.get("proof_boundary"))
    if proof_boundary.get("command_is_read_only") is not True:
        blockers.append("forwarding_preflight_not_read_only")

    return _gate(
        "production_webapp_to_pipeline_forwarding",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "path": str(path),
            "status": status,
            "origin": origin,
            "blocker_count": len(forwarding_report.get("blockers") or []),
            "warning_count": len(forwarding_report.get("warnings") or []),
            "probe": {
                "status": probe.get("status"),
                "http_status": probe.get("http_status"),
                "audit_status": probe.get("audit_status"),
                "webapp_staging_performed": probe.get("webapp_staging_performed"),
            },
        },
    )


def _route_proof_gate(
    route_proof: Mapping[str, Any],
    load_error: str | None,
    path: Path | None,
    capture_root: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    if path is None:
        blockers.append("production_route_forwarding_proof_path_missing")
    elif load_error:
        blockers.append(f"production_route_forwarding_proof_{load_error}")

    expected_capture_root = str(capture_root)
    proof_capture_root = _string(route_proof.get("capture_root"))
    if proof_capture_root != expected_capture_root:
        blockers.append("production_route_capture_root_mismatch")

    job_request = _mapping(route_proof.get("job_request"))
    site_package = _mapping(job_request.get("site_package"))
    job_capture_root = _string(site_package.get("capture_root"))
    if job_capture_root and job_capture_root != expected_capture_root:
        blockers.append("production_route_job_request_capture_root_mismatch")

    if route_proof.get("status") != "forwarded_to_pipeline_intake":
        blockers.append("production_route_forwarding_not_forwarded_to_pipeline_intake")

    pipeline_intake = _mapping(route_proof.get("pipeline_intake"))
    if pipeline_intake.get("accepted") is not True:
        blockers.append("production_route_pipeline_intake_not_accepted")
    if pipeline_intake.get("status") != "staged_for_control_plane":
        blockers.append("production_route_pipeline_intake_not_staged")
    if pipeline_intake.get("input_blockers"):
        blockers.append("production_route_pipeline_intake_has_blockers")

    proof_boundary = _mapping(route_proof.get("proof_boundary"))
    if proof_boundary.get("production_live_webapp_forwarding_proven") is not True:
        blockers.append("production_live_webapp_forwarding_not_proven")
    for field in (
        "simulator_execution_proven",
        "robot_policy_execution_proven",
        "real_robot_pov_evidence_proven",
        "safety_validated",
        "customer_delivery_readiness_proven",
        "public_claim_upgrade_allowed",
    ):
        if proof_boundary.get(field) is True:
            blockers.append(f"production_route_overclaimed_{field}")

    return _gate(
        "production_route_forwarding_proof",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "path": str(path) if path else None,
            "status": route_proof.get("status"),
            "pipeline_intake": pipeline_intake,
            "proof_boundary": proof_boundary,
        },
    )


def _deployment_gate(deployment_proof: Mapping[str, Any], load_error: str | None, path: Path | None) -> dict[str, Any]:
    blockers: list[str] = []
    if path is None:
        blockers.append("production_deployment_proof_path_missing")
    elif load_error:
        blockers.append(f"production_deployment_proof_{load_error}")

    status = _string(deployment_proof.get("status")).lower()
    if status not in READY_DEPLOYMENT_STATUSES:
        blockers.append("production_deployment_status_not_ready")
    for field in (
        "production_deployment_proven",
        "webapp_health_ready",
        "pipeline_intake_health_ready",
        "git_parity_proven",
    ):
        if deployment_proof.get(field) is not True:
            blockers.append(f"{field}_not_true")

    return _gate(
        "production_deployment_parity",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "path": str(path) if path else None,
            "status": deployment_proof.get("status"),
            "webapp_url": deployment_proof.get("webapp_url"),
            "pipeline_intake_url": deployment_proof.get("pipeline_intake_url"),
        },
    )


def build_release_gate_report(
    *,
    capture_root: Path,
    local_gate_report_path: Path,
    forwarding_preflight_report_path: Path,
    production_route_forwarding_proof_path: Path | None,
    production_deployment_proof_path: Path | None,
    require_non_local_forwarding_endpoint: bool = True,
) -> dict[str, Any]:
    local_gate, local_error = _load_mapping(local_gate_report_path)
    forwarding_report, forwarding_error = _load_mapping(forwarding_preflight_report_path)
    route_proof, route_error = (
        _load_mapping(production_route_forwarding_proof_path)
        if production_route_forwarding_proof_path is not None
        else ({}, "missing")
    )
    deployment_proof, deployment_error = (
        _load_mapping(production_deployment_proof_path)
        if production_deployment_proof_path is not None
        else ({}, "missing")
    )

    gates = [
        _local_sim_only_gate(local_gate, local_error, local_gate_report_path),
        _forwarding_gate(
            forwarding_report,
            forwarding_error,
            forwarding_preflight_report_path,
            require_non_local_endpoint=require_non_local_forwarding_endpoint,
        ),
        _route_proof_gate(route_proof, route_error, production_route_forwarding_proof_path, capture_root),
        _deployment_gate(deployment_proof, deployment_error, production_deployment_proof_path),
    ]
    blockers = [
        f"{gate['id']}:{blocker}"
        for gate in gates
        for blocker in gate.get("blockers", [])
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "passed" if not blockers else "blocked",
        "ready_for_beta_release": not blockers,
        "capture_root": str(capture_root),
        "blockers": blockers,
        "gates": gates,
        "proof_boundary": {
            "local_sim_only_post_upload_autonomy_checked": True,
            "production_forwarding_checked": True,
            "production_route_forwarding_checked": True,
            "production_deployment_parity_checked": True,
            "physical_robot_readiness_required_for_this_gate": False,
            "remote_cloud_provider_execution_required_for_this_gate": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--webapp-repo", type=Path, default=_default_webapp_repo())
    parser.add_argument("--local-gate-report", type=Path)
    parser.add_argument("--forwarding-preflight-report", type=Path)
    parser.add_argument("--production-route-forwarding-proof", type=Path)
    parser.add_argument("--production-deployment-proof", type=Path)
    parser.add_argument("--allow-local-forwarding-endpoint", action="store_true")
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args(argv)

    capture_root = args.capture_root.resolve()
    webapp_repo = args.webapp_repo.resolve()
    local_gate_report = (
        args.local_gate_report
        or capture_root
        / "pipeline"
        / "live_pipeline_control_plane"
        / "sim_only_beta_local_gate"
        / "sim_only_beta_local_gate_report.json"
    ).resolve()
    forwarding_report = (
        args.forwarding_preflight_report
        or webapp_repo
        / "output"
        / "pipeline"
        / "robot_eval_job_requests"
        / "forwarding_preflight.json"
    ).resolve()
    output_path = (
        args.output_path
        or capture_root
        / "pipeline"
        / "live_pipeline_control_plane"
        / "sim_only_beta_release_gate_report.json"
    ).resolve()

    report = build_release_gate_report(
        capture_root=capture_root,
        local_gate_report_path=local_gate_report,
        forwarding_preflight_report_path=forwarding_report,
        production_route_forwarding_proof_path=(
            args.production_route_forwarding_proof.resolve()
            if args.production_route_forwarding_proof
            else None
        ),
        production_deployment_proof_path=(
            args.production_deployment_proof.resolve()
            if args.production_deployment_proof
            else None
        ),
        require_non_local_forwarding_endpoint=not args.allow_local_forwarding_endpoint,
    )
    ensure_dir(output_path.parent)
    write_json(output_path, report)
    print(f"[sim-only-beta-release-gate] report={output_path}")
    print(f"[sim-only-beta-release-gate] status={report['status']}")
    if report["blockers"]:
        print(f"[sim-only-beta-release-gate] blockers={len(report['blockers'])}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
