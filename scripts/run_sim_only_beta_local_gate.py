#!/usr/bin/env python3
"""Run a local sim-only beta gate across WebApp forwarding and Pipeline intake.

The gate uses a synthetic local intake token, starts the real Pipeline intake
HTTP service, routes a WebApp-built robot-eval request into it, lets the live
control plane consume the staged inbox, and verifies the resulting MuJoCo
sim-only closure artifacts. It is intentionally local: it does not prove
production deployment, cloud provider execution, physical robot readiness, or
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


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = read_json_any(path)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Expected JSON object at {path}")
    return dict(payload)


def _require(condition: bool, blocker: str, blockers: list[str]) -> None:
    if not condition:
        blockers.append(blocker)


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
        _require(
            robot_team_closure.get("sim_only_beta_core_complete") is True,
            "sim_only_beta_core_not_complete",
            blockers,
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
        "robot_team_grade_closure": {
            "status": robot_team_closure.get("status"),
            "sim_only_beta_core_complete": robot_team_closure.get("sim_only_beta_core_complete"),
            "robot_team_grade_evaluation_complete": robot_team_closure.get(
                "robot_team_grade_evaluation_complete"
            ),
            "deployment_readiness_complete": robot_team_closure.get("deployment_readiness_complete"),
            "blocked_requirement_ids": robot_team_closure.get("blocked_requirement_ids"),
        },
        "proof_boundary": {
            "local_webapp_route_forwarding_proven": True,
            "pipeline_intake_staged_request_proven": True,
            "local_control_plane_processed_staged_request": True,
            "local_mujoco_simulator_execution_proven": not blockers,
            "production_live_webapp_forwarding_proven": False,
            "production_deployment_proven": False,
            "remote_cloud_provider_execution_proven": False,
            "physical_robot_readiness_proven": False,
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
