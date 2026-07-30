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

from blueprint_pipeline.buyer_claim_ceiling import build_buyer_claim_ceiling  # noqa: E402
from blueprint_pipeline.common import ensure_dir, read_json_any, utc_now_iso, write_json  # noqa: E402


SCHEMA_VERSION = "blueprint.sim_only_beta_release_gate_report.v1"
DEPLOYMENT_PROOF_SCHEMA_VERSION = "blueprint.sim_only_beta_deployment_parity_proof.v2"
READY_FORWARDING_STATUSES = {
    "ready_for_required_forwarding_with_probe",
}
READY_DEPLOYMENT_STATUSES = {"passed", "ready", "healthy", "verified"}
NON_BLOCKING_FORWARDING_WARNINGS = {
    "capture_root_override_not_configured",
}
BUYER_CLAIM_COPY_FIELDS = {
    "buyer_facing_copy",
    "buyer_copy",
    "marketing_copy",
    "report_copy",
    "public_copy",
    "public_claims",
    "claim_copy",
    "copy_claims",
}


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


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = _string(value)
        return [text] if text else []
    if not isinstance(value, Sequence):
        return []
    return [_string(item) for item in value if _string(item)]


def _bool(value: Any) -> bool:
    return value is True


def _origin_is_localhost(origin: str) -> bool:
    text = origin.lower()
    return "localhost" in text or "127.0.0.1" in text or "[::1]" in text


def _capture_root_identity(value: str) -> dict[str, str] | None:
    path = _string(value)
    if not path:
        return None
    parts = [part for part in Path(path).parts if part not in {"/", ""}]
    for index, part in enumerate(parts):
        if part == "scenes" and index + 3 < len(parts) and parts[index + 2] == "captures":
            return {"scene_id": parts[index + 1], "capture_id": parts[index + 3]}
    return None


def _same_capture_root_or_identity(left: str, right: str) -> bool:
    if _string(left) == _string(right):
        return True
    left_identity = _capture_root_identity(left)
    right_identity = _capture_root_identity(right)
    return bool(left_identity and right_identity and left_identity == right_identity)


def _gate(
    gate_id: str, *, passed: bool, blockers: list[str], evidence: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "status": "passed" if passed else "blocked",
        "blockers": blockers,
        "evidence": evidence,
    }


def _nested_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        mapped = _mapping(value)
        if mapped:
            return mapped
    return {}


def _claim_copy_inputs_from_surface(
    surface: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    copy_inputs: dict[str, Any] = {}
    for key, value in surface.items():
        if key in BUYER_CLAIM_COPY_FIELDS:
            copy_inputs[f"{source}.{key}"] = value
        elif isinstance(value, Mapping):
            nested = _claim_copy_inputs_from_surface(value, source=f"{source}.{key}")
            copy_inputs.update(nested)
    return copy_inputs


def _sim_only_beta_requirement_summary(
    local_gate: Mapping[str, Any],
    robot_team_closure: Mapping[str, Any],
) -> tuple[bool, list[str], dict[str, list[str]], bool]:
    requirements = [
        dict(item)
        for item in robot_team_closure.get("requirements") or []
        if isinstance(item, Mapping)
    ]
    explicit_blocked_ids = [
        *_string_list(local_gate.get("sim_only_beta_blocked_requirement_ids")),
        *_string_list(robot_team_closure.get("sim_only_beta_blocked_requirement_ids")),
    ]
    requirement_blocked_ids = [
        _string(requirement.get("requirement_id"))
        for requirement in requirements
        if requirement.get("sim_only_beta_required") is True
        and requirement.get("passed") is not True
        and _string(requirement.get("requirement_id"))
    ]
    blocked_ids = sorted({*explicit_blocked_ids, *requirement_blocked_ids})
    closure_blockers = _mapping(robot_team_closure.get("sim_only_beta_requirement_blockers"))
    blockers_by_requirement = {
        requirement_id: _string_list(closure_blockers.get(requirement_id))
        for requirement_id in blocked_ids
    }
    for requirement in requirements:
        requirement_id = _string(requirement.get("requirement_id"))
        if requirement_id in blocked_ids and not blockers_by_requirement.get(requirement_id):
            blockers_by_requirement[requirement_id] = _string_list(requirement.get("blockers"))

    explicit_satisfied = local_gate.get("sim_only_beta_requirements_satisfied")
    if explicit_satisfied is None:
        explicit_satisfied = robot_team_closure.get("sim_only_beta_requirements_satisfied")
    details_present = bool(
        requirements
        or explicit_blocked_ids
        or explicit_satisfied is not None
        or robot_team_closure.get("sim_only_beta_core_complete") is not None
    )
    if explicit_satisfied is not None:
        satisfied = explicit_satisfied is True and not blocked_ids
    elif requirements or explicit_blocked_ids:
        satisfied = not blocked_ids
    else:
        satisfied = robot_team_closure.get("sim_only_beta_core_complete") is True
    return satisfied, blocked_ids, blockers_by_requirement, details_present


def _local_sim_only_gate(
    local_gate: Mapping[str, Any], load_error: str | None, path: Path
) -> dict[str, Any]:
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
    visual_review = _mapping(batch_closure.get("visual_review"))
    scenario_eval_matrix = _mapping(local_gate.get("scenario_eval_matrix"))
    (
        sim_only_requirements_satisfied,
        sim_only_blocked_requirement_ids,
        sim_only_requirement_blockers,
        sim_only_requirement_details_present,
    ) = _sim_only_beta_requirement_summary(local_gate, robot_team_closure)
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
        "visual_review_coverage_complete": batch_closure.get("visual_review_coverage_complete"),
        "visual_recording_coverage_complete": visual_coverage.get(
            "all_required_runs_have_visual_recording"
        ),
        "visual_files_complete": visual_coverage.get("all_video_files_complete"),
    }
    for key, value in required_true.items():
        if value is not True:
            blockers.append(f"{key}_not_true")
    if sim_only_blocked_requirement_ids:
        blockers.extend(
            f"sim_only_beta_requirement_{requirement_id}_not_complete"
            for requirement_id in sim_only_blocked_requirement_ids
        )
    elif not sim_only_requirements_satisfied:
        if sim_only_requirement_details_present:
            blockers.append("sim_only_beta_requirements_not_satisfied_without_requirement_ids")
        else:
            blockers.append("sim_only_beta_core_completion_not_true_without_requirement_details")

    return _gate(
        "local_sim_only_post_upload_autonomy",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "path": str(path),
            "status": local_gate.get("status"),
            "attempt_count": batch_closure.get("attempt_count"),
            "required_true": {
                **required_true,
                "sim_only_beta_requirements_satisfied": sim_only_requirements_satisfied,
            },
            "scenario_eval_matrix": scenario_eval_matrix,
            "visual_review_accepted_count": visual_review.get("accepted_review_count"),
            "sim_only_beta_core_complete": robot_team_closure.get("sim_only_beta_core_complete"),
            "sim_only_beta_blocked_requirement_ids": sim_only_blocked_requirement_ids,
            "sim_only_beta_requirement_blockers": sim_only_requirement_blockers,
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
    warnings = [
        _string(warning)
        for warning in forwarding_report.get("warnings", []) or []
        if _string(warning)
    ]
    blocking_warnings = [
        warning for warning in warnings if warning not in NON_BLOCKING_FORWARDING_WARNINGS
    ]
    if blocking_warnings:
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
            "warning_count": len(warnings),
            "blocking_warnings": blocking_warnings,
            "non_blocking_warnings": [
                warning for warning in warnings if warning in NON_BLOCKING_FORWARDING_WARNINGS
            ],
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
    if not _same_capture_root_or_identity(proof_capture_root, expected_capture_root):
        blockers.append("production_route_capture_root_mismatch")

    job_request = _mapping(route_proof.get("job_request"))
    site_package = _mapping(job_request.get("site_package"))
    job_capture_root = _string(site_package.get("capture_root"))
    if job_capture_root and not _same_capture_root_or_identity(
        job_capture_root, expected_capture_root
    ):
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
        "non_ranking_operational_claim_validated",
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
            "expected_capture_identity": _capture_root_identity(expected_capture_root),
            "proof_capture_identity": _capture_root_identity(proof_capture_root),
            "pipeline_intake": pipeline_intake,
            "proof_boundary": proof_boundary,
        },
    )


def _deployment_gate(
    deployment_proof: Mapping[str, Any], load_error: str | None, path: Path | None
) -> dict[str, Any]:
    blockers: list[str] = []
    if path is None:
        blockers.append("production_deployment_proof_path_missing")
    elif load_error:
        blockers.append(f"production_deployment_proof_{load_error}")

    status = _string(deployment_proof.get("status")).lower()
    if status not in READY_DEPLOYMENT_STATUSES:
        blockers.append("production_deployment_status_not_ready")
    if deployment_proof.get("schema_version") != DEPLOYMENT_PROOF_SCHEMA_VERSION:
        blockers.append("production_deployment_proof_schema_not_supported")
    if deployment_proof.get("deployment_environment") != "production":
        blockers.append("production_deployment_environment_not_production")
    for field in (
        "deployment_proven",
        "production_deployment_proven",
        "webapp_health_ready",
        "pipeline_intake_health_ready",
        "webapp_deployment_identity_ready",
        "pipeline_deployment_identity_ready",
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
            "schema_version": deployment_proof.get("schema_version"),
            "status": deployment_proof.get("status"),
            "deployment_environment": deployment_proof.get("deployment_environment"),
            "deployment_proven": deployment_proof.get("deployment_proven"),
            "webapp_url": deployment_proof.get("webapp_url"),
            "pipeline_intake_url": deployment_proof.get("pipeline_intake_url"),
            "webapp_health_ready": deployment_proof.get("webapp_health_ready"),
            "pipeline_intake_health_ready": deployment_proof.get("pipeline_intake_health_ready"),
            "webapp_deployment_identity_ready": deployment_proof.get(
                "webapp_deployment_identity_ready"
            ),
            "pipeline_deployment_identity_ready": deployment_proof.get(
                "pipeline_deployment_identity_ready"
            ),
            "git_parity_proven": deployment_proof.get("git_parity_proven"),
        },
    )


def _buyer_claim_ceiling_gate(
    *,
    local_gate: Mapping[str, Any],
    route_proof: Mapping[str, Any],
    deployment_proof: Mapping[str, Any],
) -> dict[str, Any]:
    live_closure = _nested_mapping(
        local_gate.get("live_eval_closure"),
        local_gate.get("live_eval_closure_manifest"),
        local_gate.get("live_robot_eval_closure"),
        route_proof.get("live_eval_closure"),
        route_proof.get("live_eval_closure_manifest"),
        route_proof.get("live_robot_eval_closure"),
    )
    live_boundary = _mapping(live_closure.get("proof_boundary"))
    task_eval_run_report = _nested_mapping(
        local_gate.get("task_eval_run_report"),
        route_proof.get("task_eval_run_report"),
        live_closure.get("task_eval_run_report"),
    )
    success_claim_ledger = _nested_mapping(
        task_eval_run_report.get("success_claim_ledger"),
        local_gate.get("success_claim_ledger"),
        route_proof.get("success_claim_ledger"),
        live_closure.get("success_claim_ledger"),
    )
    copy_inputs: dict[str, Any] = {}
    for source, surface in (
        ("local_gate", local_gate),
        ("route_proof", route_proof),
        ("deployment_proof", deployment_proof),
        ("task_eval_run_report", task_eval_run_report),
        ("live_closure", live_closure),
    ):
        copy_inputs.update(_claim_copy_inputs_from_surface(surface, source=source))

    buyer_claim_ceiling = build_buyer_claim_ceiling(
        success_claim_ledger=success_claim_ledger,
        proof_boundary={
            "live_simulator_execution_proven": (
                live_boundary.get("live_simulator_execution_proven") is True
                or live_boundary.get("simulator_execution_proven") is True
            ),
            "live_policy_execution_proven": (
                live_boundary.get("live_policy_execution_proven") is True
                or live_boundary.get("robot_policy_execution_proven") is True
            ),
        },
        live_closure=live_closure,
        buyer_copy_inputs=copy_inputs,
    )
    blockers = _string_list(buyer_claim_ceiling.get("blockers"))
    return _gate(
        "buyer_claim_ceiling_and_copy",
        passed=not blockers,
        blockers=blockers,
        evidence={
            "buyer_claim_ceiling": buyer_claim_ceiling,
            "copy_input_sources": sorted(copy_inputs),
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
        _route_proof_gate(
            route_proof, route_error, production_route_forwarding_proof_path, capture_root
        ),
        _deployment_gate(deployment_proof, deployment_error, production_deployment_proof_path),
        _buyer_claim_ceiling_gate(
            local_gate=local_gate,
            route_proof=route_proof,
            deployment_proof=deployment_proof,
        ),
    ]
    blockers = [f"{gate['id']}:{blocker}" for gate in gates for blocker in gate.get("blockers", [])]
    local_gate_evidence = _mapping(gates[0].get("evidence"))
    local_gate_proof_boundary = _mapping(local_gate_evidence.get("proof_boundary"))
    local_scenario_eval_matrix = _mapping(local_gate_evidence.get("scenario_eval_matrix"))
    deployment_gate_evidence = _mapping(gates[3].get("evidence"))
    simulator_execution_proven = (
        local_gate_proof_boundary.get("simulator_execution_proven") is True
        or local_gate_proof_boundary.get("local_mujoco_simulator_execution_proven") is True
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "passed" if not blockers else "blocked",
        "ready_for_beta_release": not blockers,
        "capture_root": str(capture_root),
        "blockers": blockers,
        "scenario_eval_run_count": local_scenario_eval_matrix.get("scenario_eval_run_count"),
        "visual_review_accepted_count": local_gate_evidence.get("visual_review_accepted_count"),
        "webapp_health_ready": deployment_gate_evidence.get("webapp_health_ready"),
        "pipeline_intake_health_ready": deployment_gate_evidence.get(
            "pipeline_intake_health_ready"
        ),
        "git_parity_proven": deployment_gate_evidence.get("git_parity_proven"),
        "simulator_execution_proven": simulator_execution_proven,
        "public_claim_upgrade_allowed": False,
        "buyer_claim_ceiling": gates[4]["evidence"]["buyer_claim_ceiling"],
        "gates": gates,
        "proof_boundary": {
            "local_sim_only_post_upload_autonomy_checked": True,
            "production_forwarding_checked": True,
            "production_route_forwarding_checked": True,
            "production_deployment_parity_checked": True,
            "simulator_execution_proven": simulator_execution_proven,
            "generated_world_rank_fidelity_required_for_this_gate": False,
            "remote_cloud_provider_execution_required_for_this_gate": False,
            "public_claim_upgrade_allowed": False,
            "buyer_facing_claim_ceiling_pinned_to_highest_truthful_claim": True,
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
            args.production_deployment_proof.resolve() if args.production_deployment_proof else None
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
