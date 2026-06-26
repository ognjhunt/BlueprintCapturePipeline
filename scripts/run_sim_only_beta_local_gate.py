#!/usr/bin/env python3
"""Run a local sim-only beta gate across WebApp forwarding and Pipeline intake.

The gate uses a synthetic local intake token, starts the real Pipeline intake
HTTP service, routes a WebApp-built robot-eval request into it, lets the live
control plane consume the staged inbox, and verifies the resulting MuJoCo
sim-only closure artifacts. It is intentionally local: it does not prove
production deployment, cloud provider execution, generated-world rank fidelity, or
customer delivery.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.common import ensure_dir, read_json_any, utc_now_iso, write_json  # noqa: E402


DEFAULT_TOKEN = "local-sim-only-beta-forwarding-token"
WAM_HANDOFF_ARTIFACTS = {
    "policy_ranking_scorecard": "policy_ranking_scorecard.json",
    "candidate_selection_report": "candidate_selection_report.json",
    "wam_eval_claim_boundary": "wam_eval_claim_boundary.json",
}


def _repo_root() -> Path:
    return ROOT


def _default_webapp_repo() -> Path:
    return _repo_root().parent / "Blueprint-WebApp"


def _default_mujoco_g1_root() -> Path:
    return _repo_root() / "output" / "external_assets" / "mujoco_menagerie" / "unitree_g1"


def _run(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"[sim-only-beta-local-gate] cwd={cwd}")
    print(f"[sim-only-beta-local-gate] $ {' '.join(cmd)}")
    return subprocess.run(
        list(cmd),
        cwd=cwd,
        check=True,
        env=dict(env) if env is not None else None,
        text=True,
        timeout=timeout_seconds,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _health_ready(url: str) -> bool:
    try:
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=1.0) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError):
        return False


def _wait_for_health(url: str, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _health_ready(url):
            return
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for intake service health: {url}")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = _string(value)
        return [text] if text else []
    if not isinstance(value, Sequence):
        return []
    return [_string(item) for item in value if _string(item)]


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = read_json_any(path)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Expected JSON object at {path}")
    return dict(payload)


def _require(condition: bool, blocker: str, blockers: list[str]) -> None:
    if not condition:
        blockers.append(blocker)


def _sim_only_beta_requirement_summary(
    robot_team_closure: Mapping[str, Any],
) -> tuple[list[str], dict[str, list[str]], bool]:
    requirements = [
        dict(item)
        for item in robot_team_closure.get("requirements") or []
        if isinstance(item, Mapping)
    ]
    explicit_blocked_ids = _string_list(
        robot_team_closure.get("sim_only_beta_blocked_requirement_ids")
    )
    requirement_blocked_ids = [
        _string(requirement.get("requirement_id"))
        for requirement in requirements
        if requirement.get("sim_only_beta_required") is True
        and requirement.get("passed") is not True
        and _string(requirement.get("requirement_id"))
    ]
    blocked_ids = sorted({*explicit_blocked_ids, *requirement_blocked_ids})
    blockers_by_requirement = {
        requirement_id: _string_list(requirement.get("blockers"))
        for requirement in requirements
        for requirement_id in [_string(requirement.get("requirement_id"))]
        if requirement_id in blocked_ids
    }
    details_present = bool(requirements or explicit_blocked_ids)
    return blocked_ids, blockers_by_requirement, details_present


def _sim_only_beta_core_blockers(
    robot_team_closure: Mapping[str, Any],
) -> tuple[list[str], list[str], dict[str, list[str]], bool]:
    blocked_ids, blockers_by_requirement, details_present = (
        _sim_only_beta_requirement_summary(robot_team_closure)
    )
    if blocked_ids:
        blockers = [
            f"sim_only_beta_requirement_{requirement_id}_not_complete"
            for requirement_id in blocked_ids
        ]
    elif details_present:
        blockers = []
    elif robot_team_closure.get("sim_only_beta_core_complete") is not True:
        blockers = ["sim_only_beta_core_completion_not_true_without_requirement_details"]
    else:
        blockers = []
    return blockers, blocked_ids, blockers_by_requirement, details_present


def _load_optional_mapping(path: Path, blockers: list[str], blocker_prefix: str) -> dict[str, Any]:
    if not path.is_file():
        blockers.append(f"{blocker_prefix}_missing")
        return {}
    try:
        payload = read_json_any(path)
    except (OSError, ValueError) as exc:
        blockers.append(f"{blocker_prefix}_unreadable:{type(exc).__name__}")
        return {}
    if not isinstance(payload, Mapping):
        blockers.append(f"{blocker_prefix}_not_json_object")
        return {}
    return dict(payload)


def _not_true(value: Any) -> bool:
    return value is not True


def _validate_wam_handoff_artifacts(
    *,
    job_root: Path,
    blockers: list[str],
) -> dict[str, Any]:
    payloads: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {"required_artifacts": dict(WAM_HANDOFF_ARTIFACTS)}
    for artifact_key, filename in WAM_HANDOFF_ARTIFACTS.items():
        path = job_root / filename
        payload = _load_optional_mapping(
            path,
            blockers,
            f"wam_handoff_artifact_{artifact_key}",
        )
        payloads[artifact_key] = payload
        summary[artifact_key] = {
            "path": filename,
            "present": bool(payload),
            "status": payload.get("status"),
        }

    scorecard = payloads.get("policy_ranking_scorecard") or {}
    scorecard_boundary = _mapping(scorecard.get("claim_boundary"))
    ranking_confidence = _mapping(scorecard.get("ranking_confidence"))
    scorecard_status = _string(scorecard.get("status"))
    guarded_scorecard_statuses = {
        "blocked_inconclusive_ranking",
        "completed_ambiguous_ranking",
        "completed_visual_review_required",
        "completed_low_confidence_ranking",
    }
    if scorecard:
        _require(
            scorecard_boundary.get("policy_ranking_is_evaluator_bounded") is True,
            "policy_ranking_scorecard_boundary_not_evaluator_bounded",
            blockers,
        )
        _require(
            scorecard_boundary.get("policy_ranking_is_not_evaluation_readiness") is True,
            "policy_ranking_scorecard_boundary_allows_evaluation_readiness",
            blockers,
        )
        _require(
            scorecard_boundary.get("rank_fidelity_result_proven") is False,
            "policy_ranking_scorecard_boundary_upgrades_rank_fidelity",
            blockers,
        )
        _require(
            scorecard_boundary.get("public_claim_upgrade_allowed") is False,
            "policy_ranking_scorecard_boundary_allows_public_claim_upgrade",
            blockers,
        )
        _require(
            _not_true(scorecard_boundary.get("deployment_approval_proven")),
            "policy_ranking_scorecard_boundary_claims_deployment_approval",
            blockers,
        )
        _require(
            _not_true(scorecard_boundary.get("physical_robot_readiness_proven")),
            "policy_ranking_scorecard_boundary_claims_physical_robot_readiness",
            blockers,
        )
        _require(
            _not_true(scorecard_boundary.get("safety_validation_proven")),
            "policy_ranking_scorecard_boundary_claims_safety_validation",
            blockers,
        )
        if scorecard_status in guarded_scorecard_statuses:
            _require(
                scorecard.get("top_policy_id") in (None, ""),
                "policy_ranking_scorecard_claims_winner_despite_blocked_or_ambiguous_status",
                blockers,
            )

    candidate_report = payloads.get("candidate_selection_report") or {}
    candidate_boundary = _mapping(candidate_report.get("claim_boundary"))
    candidate_status = _string(candidate_report.get("status"))
    if candidate_report:
        _require(
            candidate_boundary.get("do_not_use_as_rank_fidelity_result") is True,
            "candidate_selection_report_missing_rank_fidelity_guard",
            blockers,
        )
        _require(
            candidate_boundary.get("rank_fidelity_result_claimed") is False,
            "candidate_selection_report_claims_rank_fidelity",
            blockers,
        )
        _require(
            candidate_boundary.get("accepted_anchor_success_claimed") is False,
            "candidate_selection_report_claims_accepted_anchor_success",
            blockers,
        )
        _require(
            _not_true(candidate_boundary.get("deployment_approval_proven")),
            "candidate_selection_report_claims_deployment_approval",
            blockers,
        )
        _require(
            _not_true(candidate_boundary.get("physical_robot_readiness_proven")),
            "candidate_selection_report_claims_physical_robot_readiness",
            blockers,
        )
        _require(
            _not_true(candidate_boundary.get("safety_validation_proven")),
            "candidate_selection_report_claims_safety_validation",
            blockers,
        )
        if candidate_status != "clear_winner":
            _require(
                candidate_report.get("top_policy_id") in (None, ""),
                "candidate_selection_report_claims_winner_despite_blocked_or_ambiguous_status",
                blockers,
            )

    claim_boundary = payloads.get("wam_eval_claim_boundary") or {}
    if claim_boundary:
        _require(
            claim_boundary.get("primary_proof_target")
            == "policy_comparison_within_configured_evaluator",
            "wam_eval_claim_boundary_primary_target_not_policy_comparison",
            blockers,
        )
        _require(
            claim_boundary.get("policy_ranking_is_evaluator_bounded") is True,
            "wam_eval_claim_boundary_not_evaluator_bounded",
            blockers,
        )
        _require(
            claim_boundary.get("policy_ranking_is_not_evaluation_readiness") is True,
            "wam_eval_claim_boundary_allows_evaluation_readiness",
            blockers,
        )
        _require(
            claim_boundary.get("rank_fidelity_result_proven") is False,
            "wam_eval_claim_boundary_upgrades_rank_fidelity",
            blockers,
        )
        _require(
            claim_boundary.get("public_claim_upgrade_allowed") is False,
            "wam_eval_claim_boundary_allows_public_claim_upgrade",
            blockers,
        )
        _require(
            claim_boundary.get("simulator_execution_proven") is False,
            "wam_eval_claim_boundary_claims_simulator_execution",
            blockers,
        )
        _require(
            claim_boundary.get("robot_policy_execution_proven") is False,
            "wam_eval_claim_boundary_claims_robot_policy_execution",
            blockers,
        )
        _require(
            claim_boundary.get("real_world_outcome_proven") is False,
            "wam_eval_claim_boundary_claims_real_world_outcome",
            blockers,
        )
        _require(
            _not_true(claim_boundary.get("deployment_approval_proven")),
            "wam_eval_claim_boundary_claims_deployment_approval",
            blockers,
        )
        _require(
            _not_true(claim_boundary.get("physical_robot_readiness_proven")),
            "wam_eval_claim_boundary_claims_physical_robot_readiness",
            blockers,
        )
        _require(
            _not_true(claim_boundary.get("safety_validation_proven")),
            "wam_eval_claim_boundary_claims_safety_validation",
            blockers,
        )

    summary["policy_ranking"] = {
        "status": scorecard_status or None,
        "top_policy_id": scorecard.get("top_policy_id"),
        "evaluator_top_policy_id": scorecard.get("evaluator_top_policy_id"),
        "single_best_policy_claimed": bool(scorecard.get("single_best_policy_claimed")),
        "comparison_blockers": _string_list(scorecard.get("comparison_blockers")),
        "ranking_confidence": ranking_confidence,
    }
    summary["candidate_selection"] = {
        "status": candidate_status or None,
        "top_policy_id": candidate_report.get("top_policy_id"),
        "evaluator_top_policy_id": candidate_report.get("evaluator_top_policy_id"),
        "tie_or_ambiguity_status": candidate_report.get("tie_or_ambiguity_status"),
        "candidate_shortlist_count": len(
            [
                item
                for item in candidate_report.get("candidate_shortlist") or []
                if isinstance(item, Mapping)
            ]
        ),
    }
    summary["claim_boundary"] = {
        "primary_proof_target": claim_boundary.get("primary_proof_target"),
        "policy_ranking_is_evaluator_bounded": claim_boundary.get(
            "policy_ranking_is_evaluator_bounded"
        ),
        "policy_ranking_is_not_evaluation_readiness": claim_boundary.get(
            "policy_ranking_is_not_evaluation_readiness"
        ),
        "rank_fidelity_result_proven": claim_boundary.get("rank_fidelity_result_proven"),
        "public_claim_upgrade_allowed": claim_boundary.get("public_claim_upgrade_allowed"),
    }
    return summary


def _validate_sim_only_outputs(*, capture_root: Path, proof_path: Path) -> dict[str, Any]:
    proof = _load_mapping(proof_path)
    pipeline_intake = _mapping(proof.get("pipeline_intake"))
    proof_boundary = _mapping(proof.get("proof_boundary"))
    blockers: list[str] = []
    _require(proof.get("status") == "forwarded_to_pipeline_intake", "route_forwarding_not_proven", blockers)
    _require(pipeline_intake.get("accepted") is True, "pipeline_intake_did_not_accept_request", blockers)
    _require(
        pipeline_intake.get("status") == "staged_for_control_plane",
        "pipeline_intake_not_staged_for_control_plane",
        blockers,
    )
    _require(
        proof_boundary.get("local_webapp_route_forwarding_proven") is True,
        "local_webapp_route_forwarding_boundary_false",
        blockers,
    )
    _require(
        proof_boundary.get("pipeline_intake_staged_request_proven") is True,
        "pipeline_intake_staged_boundary_false",
        blockers,
    )
    _require(
        proof_boundary.get("simulator_execution_proven") is False,
        "webapp_route_proof_overclaimed_simulator_execution",
        blockers,
    )

    inbox_manifest = _load_mapping(capture_root / "pipeline" / "robot_eval_job_requests" / "inbox_run_manifest.json")
    _require(inbox_manifest.get("status") == "completed", "inbox_run_not_completed", blockers)
    _require(int(inbox_manifest.get("processed_count") or 0) >= 1, "inbox_run_processed_zero_requests", blockers)

    jobs = [item for item in inbox_manifest.get("jobs") or [] if isinstance(item, Mapping)]
    proof_job_request = _mapping(proof.get("job_request"))
    expected_job_id = str(proof_job_request.get("job_id") or proof.get("job_id") or "").strip()
    if not jobs:
        blockers.append("inbox_run_missing_job_records")
        job_id = ""
        job_root = None
    else:
        if expected_job_id:
            matched_job = next(
                (
                    dict(item)
                    for item in jobs
                    if str(item.get("job_id") or "").strip() == expected_job_id
                ),
                None,
            )
            if matched_job is None:
                blockers.append("inbox_run_missing_route_proof_job_record")
                matched_job = dict(jobs[0])
        else:
            matched_job = dict(jobs[0])
        job = matched_job
        job_id = str(job.get("job_id") or "").strip()
        job_root = capture_root / "pipeline" / "robot_eval_jobs" / job_id if job_id else None
        _require(job.get("status") == "simulator_command_completed", "job_status_not_simulator_command_completed", blockers)

    job_run_manifest: dict[str, Any] = {}
    simulator_result: dict[str, Any] = {}
    scenario_eval_matrix: dict[str, Any] = {}
    batch_closure: dict[str, Any] = {}
    robot_team_closure: dict[str, Any] = {}
    sim_only_beta_core_blockers: list[str] = []
    sim_only_beta_blocked_requirement_ids: list[str] = []
    sim_only_beta_requirement_blockers: dict[str, list[str]] = {}
    sim_only_beta_requirement_details_present = False
    wam_handoff_blockers: list[str] = []
    wam_handoff_artifacts: dict[str, Any] = {
        "required_artifacts": dict(WAM_HANDOFF_ARTIFACTS)
    }
    if job_root is None or not job_root.is_dir():
        blockers.append("job_root_missing")
    else:
        job_run_manifest = _load_mapping(job_root / "job_run_manifest.json")
        scenario_eval_matrix = _load_mapping(job_root / "scenario_eval_matrix.json")
        simulator_result = _load_mapping(job_root / "simulator_service_result.json")
        batch_closure = _load_mapping(job_root / "simulator_command_batch_closure_manifest.json")
        robot_team_closure = _load_mapping(job_root / "robot_team_grade_eval_closure_manifest.json")

        _require(
            job_run_manifest.get("status") == "simulator_command_completed",
            "job_run_manifest_not_simulator_command_completed",
            blockers,
        )
        _require(
            job_run_manifest.get("simulator_execution_proven") is True,
            "job_run_manifest_simulator_execution_not_proven",
            blockers,
        )
        _require(simulator_result.get("status") == "completed", "simulator_service_not_completed", blockers)
        _require(
            simulator_result.get("simulator_execution_proven") is True,
            "simulator_service_execution_not_proven",
            blockers,
        )
        _require(batch_closure.get("batch_execution_status") == "completed", "batch_execution_not_completed", blockers)
        _require(
            batch_closure.get("scenario_eval_run_coverage_complete") is True,
            "scenario_eval_run_coverage_incomplete",
            blockers,
        )
        _require(
            batch_closure.get("scenario_eval_run_id_coverage_exact") is True,
            "scenario_eval_run_id_coverage_not_exact",
            blockers,
        )
        _require(batch_closure.get("metric_coverage_complete") is True, "metric_coverage_incomplete", blockers)
        _require(
            batch_closure.get("machine_trace_package_complete") is True,
            "machine_trace_package_incomplete",
            blockers,
        )
        _require(
            scenario_eval_matrix.get("semantic_spawn_target_coverage_complete") is True,
            "semantic_spawn_target_coverage_incomplete",
            blockers,
        )
        _require(
            int(scenario_eval_matrix.get("deterministic_fallback_spawn_target_run_count") or 0)
            == 0,
            "deterministic_spawn_target_fallback_used",
            blockers,
        )
        _require(
            batch_closure.get("failure_label_coverage_complete") is True,
            "failure_label_coverage_incomplete",
            blockers,
        )
        _require(
            batch_closure.get("visual_review_coverage_complete") is True,
            "visual_review_coverage_incomplete",
            blockers,
        )
        visual_coverage = _mapping(batch_closure.get("visual_coverage"))
        _require(
            visual_coverage.get("all_required_runs_have_visual_recording") is True,
            "visual_recording_coverage_incomplete",
            blockers,
        )
        _require(
            visual_coverage.get("all_video_files_complete") is True,
            "visual_files_incomplete",
            blockers,
        )
        (
            sim_only_beta_core_blockers,
            sim_only_beta_blocked_requirement_ids,
            sim_only_beta_requirement_blockers,
            sim_only_beta_requirement_details_present,
        ) = _sim_only_beta_core_blockers(robot_team_closure)
        blockers.extend(sim_only_beta_core_blockers)
        wam_handoff_artifacts = _validate_wam_handoff_artifacts(
            job_root=job_root,
            blockers=wam_handoff_blockers,
        )
        blockers.extend(wam_handoff_blockers)

    sim_only_beta_requirements_satisfied = not sim_only_beta_core_blockers
    simulator_only_requirement_blockers = set(sim_only_beta_core_blockers) | set(
        wam_handoff_blockers
    )
    simulator_execution_blockers = [
        blocker
        for blocker in blockers
        if blocker not in simulator_only_requirement_blockers
    ]
    simulator_execution_proven = not simulator_execution_blockers

    robot_team_grade_closure = {
        "status": robot_team_closure.get("status"),
        "sim_only_beta_core_complete": robot_team_closure.get("sim_only_beta_core_complete"),
        "sim_only_beta_requirements_satisfied": sim_only_beta_requirements_satisfied,
        "sim_only_beta_requirement_details_present": sim_only_beta_requirement_details_present,
        "sim_only_beta_blocked_requirement_ids": sim_only_beta_blocked_requirement_ids,
        "sim_only_beta_requirement_blockers": sim_only_beta_requirement_blockers,
        "robot_team_grade_evaluation_complete": robot_team_closure.get(
            "robot_team_grade_evaluation_complete"
        ),
        "evaluation_readiness_complete": robot_team_closure.get("evaluation_readiness_complete"),
        "blocked_requirement_ids": robot_team_closure.get("blocked_requirement_ids"),
    }
    if robot_team_closure.get("requirements") is not None:
        robot_team_grade_closure["requirements"] = robot_team_closure.get(
            "requirements"
        )
    status = "passed" if not blockers else "blocked"
    return {
        "schema_version": "blueprint.sim_only_beta_local_gate_report.v1",
        "generated_at": utc_now_iso(),
        "status": status,
        "blockers": blockers,
        "capture_root": str(capture_root),
        "route_forwarding_proof_path": str(proof_path),
        "inbox_run_manifest_path": str(
            capture_root / "pipeline" / "robot_eval_job_requests" / "inbox_run_manifest.json"
        ),
        "job_id": job_id,
        "route_proof_job_id": expected_job_id or None,
        "simulator_execution_proven": simulator_execution_proven,
        "sim_only_beta_requirements_satisfied": sim_only_beta_requirements_satisfied,
        "sim_only_beta_blocked_requirement_ids": sim_only_beta_blocked_requirement_ids,
        "wam_handoff_artifacts_satisfied": not wam_handoff_blockers,
        "wam_handoff_blockers": wam_handoff_blockers,
        "public_claim_upgrade_allowed": False,
        "job_run_manifest": {
            "status": job_run_manifest.get("status"),
            "simulator_execution_proven": job_run_manifest.get("simulator_execution_proven"),
        },
        "simulator_service_result": {
            "status": simulator_result.get("status"),
            "simulator_execution_proven": simulator_result.get("simulator_execution_proven"),
        },
        "scenario_eval_matrix": {
            "status": scenario_eval_matrix.get("status"),
            "scenario_eval_run_count": scenario_eval_matrix.get("scenario_eval_run_count"),
            "semantic_spawn_target_coverage_complete": scenario_eval_matrix.get(
                "semantic_spawn_target_coverage_complete"
            ),
            "deterministic_fallback_spawn_target_run_count": scenario_eval_matrix.get(
                "deterministic_fallback_spawn_target_run_count"
            ),
            "fallback_spawn_target_run_ids": scenario_eval_matrix.get(
                "fallback_spawn_target_run_ids"
            ),
        },
        "batch_closure": {
            "status": batch_closure.get("status"),
            "batch_execution_status": batch_closure.get("batch_execution_status"),
            "attempt_count": batch_closure.get("attempt_count"),
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
            "visual_review": _mapping(batch_closure.get("visual_review")),
            "visual_coverage": _mapping(batch_closure.get("visual_coverage")),
            "robot_team_grade_package_complete": batch_closure.get("robot_team_grade_package_complete"),
            "robot_team_grade_blockers": batch_closure.get("robot_team_grade_blockers"),
        },
        "robot_team_grade_closure": robot_team_grade_closure,
        "wam_handoff_artifacts": wam_handoff_artifacts,
        "proof_boundary": {
            "local_webapp_route_forwarding_proven": True,
            "pipeline_intake_staged_request_proven": True,
            "local_control_plane_processed_staged_request": True,
            "local_mujoco_simulator_execution_proven": simulator_execution_proven,
            "simulator_execution_proven": simulator_execution_proven,
            "production_live_webapp_forwarding_proven": False,
            "production_deployment_proven": False,
            "remote_cloud_provider_execution_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--webapp-repo", type=Path, default=_default_webapp_repo())
    parser.add_argument("--mujoco-g1-root", type=Path, default=_default_mujoco_g1_root())
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--health-timeout-seconds", type=int, default=15)
    parser.add_argument("--command-timeout-seconds", type=int, default=2400)
    parser.add_argument("--simulator-timeout-seconds", type=int, default=1800)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args(argv)

    capture_root = args.capture_root.resolve()
    webapp_repo = args.webapp_repo.resolve()
    mujoco_g1_root = args.mujoco_g1_root.resolve()
    if not capture_root.is_dir():
        raise SystemExit(f"capture root does not exist: {capture_root}")
    if not webapp_repo.is_dir():
        raise SystemExit(f"WebApp repo does not exist: {webapp_repo}")
    if not mujoco_g1_root.is_dir():
        raise SystemExit(f"MuJoCo G1 root does not exist: {mujoco_g1_root}")

    gate_dir = capture_root / "pipeline" / "live_pipeline_control_plane" / "sim_only_beta_local_gate"
    ensure_dir(gate_dir)
    inbox_dir = capture_root / "pipeline" / "robot_eval_job_requests" / "intake_inbox"
    manifest_path = gate_dir / "live_pipeline_control_plane_manifest.json"
    processed_manifest_path = gate_dir / "live_pipeline_control_plane_manifest.processed.json"
    audit_path = gate_dir / "live_pipeline_input_intake_audit.json"
    staged_inputs_path = gate_dir / "live_pipeline_staged_inputs.json"
    proof_path = gate_dir / "local_beta_route_forwarding_proof.json"
    report_path = (args.output_path or gate_dir / "sim_only_beta_local_gate_report.json").resolve()

    beta_env = {
        **os.environ,
        "BLUEPRINT_SIM_ONLY_BETA_DEFAULT_TASK_EVAL": "true",
        "BLUEPRINT_SIM_ONLY_BETA_AUTONOMY": "true",
        "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION": "true",
        "BLUEPRINT_MUJOCO_G1_MODEL_ROOT": str(mujoco_g1_root),
    }
    _run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.simulation_automation",
            "--capture-root",
            str(capture_root),
        ],
        cwd=_repo_root(),
        env=beta_env,
        timeout_seconds=args.command_timeout_seconds,
    )
    _run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.robot_eval_dataset",
            "--capture-root",
            str(capture_root),
        ],
        cwd=_repo_root(),
        env=beta_env,
        timeout_seconds=args.command_timeout_seconds,
    )

    _run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.live_pipeline_control_plane",
            "--capture-root",
            str(capture_root),
            "--job-request-inbox",
            str(inbox_dir),
            "--no-process-inbox",
            "--no-load-env-files",
            "--output-path",
            str(manifest_path),
        ],
        cwd=_repo_root(),
        timeout_seconds=args.command_timeout_seconds,
    )
    _run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.live_pipeline_input_intake",
            "--manifest-path",
            str(manifest_path),
            "--output-path",
            str(audit_path),
            "--staged-inputs-path",
            str(staged_inputs_path),
        ],
        cwd=_repo_root(),
        timeout_seconds=args.command_timeout_seconds,
    )

    port = args.port or _free_port()
    forward_url = f"http://127.0.0.1:{port}/api/live-pipeline/job-requests"
    intake_env = {
        **os.environ,
        "BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN": args.token,
        "BLUEPRINT_LIVE_PIPELINE_INTAKE_OVERWRITE": "true",
        "BLUEPRINT_CONTROL_PLANE_OUTPUT_PATH": str(manifest_path),
        "BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR": str(gate_dir / "incoming"),
        "PORT": str(port),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "blueprint_pipeline.live_pipeline_intake_service"],
        cwd=_repo_root(),
        env=intake_env,
        text=True,
    )
    try:
        _wait_for_health(f"http://127.0.0.1:{port}/health", timeout_seconds=args.health_timeout_seconds)
        forward_env = {
            **os.environ,
            "ROBOT_EVAL_JOB_REQUEST_FORWARD_URL": forward_url,
            "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN": args.token,
            "ROBOT_EVAL_JOB_REQUEST_FORWARD_REQUIRED": "true",
        }
        _run(
            [
                "npm",
                "run",
                "pipeline:forwarding:preflight",
                "--",
                "--require-forwarding",
                "--probe-intake-audit",
            ],
            cwd=webapp_repo,
            env=forward_env,
            timeout_seconds=args.command_timeout_seconds,
        )
        _run(
            [
                "npx",
                "tsx",
                "scripts/pipeline/run-first-gpu-webapp-route-forwarding-proof.ts",
                "--capture-root",
                str(capture_root),
                "--output",
                str(proof_path),
                "--forward-url",
                forward_url,
                "--site-slug",
                "sim-only-beta-local-gate",
            ],
            cwd=webapp_repo,
            env={**os.environ, "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN": args.token},
            timeout_seconds=args.command_timeout_seconds,
        )
        _run(
            [
                "npm",
                "run",
                "pipeline:forwarding:preflight",
                "--",
                "--require-forwarding",
                "--probe-intake-audit",
            ],
            cwd=webapp_repo,
            env=forward_env,
            timeout_seconds=args.command_timeout_seconds,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    _run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.live_pipeline_control_plane",
            "--capture-root",
            str(capture_root),
            "--job-request-inbox",
            str(inbox_dir),
            "--no-load-env-files",
            "--simulator",
            "mujoco",
            "--evaluation-substrate",
            "fixture_wam",
            "--allow-simulator-execution",
            "--allow-simulator",
            "mujoco",
            "--timeout-seconds",
            str(args.simulator_timeout_seconds),
            "--output-path",
            str(processed_manifest_path),
        ],
        cwd=_repo_root(),
        env=beta_env,
        timeout_seconds=args.command_timeout_seconds,
    )

    report = _validate_sim_only_outputs(capture_root=capture_root, proof_path=proof_path)
    write_json(report_path, report)
    print(f"[sim-only-beta-local-gate] report={report_path}")
    print(f"[sim-only-beta-local-gate] status={report['status']}")
    if report["blockers"]:
        print(f"[sim-only-beta-local-gate] blockers={len(report['blockers'])}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
