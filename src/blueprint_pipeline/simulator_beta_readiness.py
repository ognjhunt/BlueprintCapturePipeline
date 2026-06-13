"""Build a simulator-only beta readiness verdict for Unitree G1 MuJoCo runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import ensure_dir, optional_read_json, utc_now_iso, write_json


SIMULATOR_BETA_READINESS_SCHEMA_VERSION = "simulator_beta_readiness.v1"
DEFAULT_OUTPUT_RELATIVE = "pipeline/sim_only_beta_rehearsal/simulator_beta_readiness"
DEFAULT_MUJOCO_OUTPUT_RELATIVE = (
    "pipeline/sim_only_beta_rehearsal/mujoco_g1_command/mujoco_g1_simulator_output.json"
)
DEFAULT_POLICY_EXECUTION_RELATIVE = (
    "pipeline/sim_only_beta_rehearsal/official_unitree_g1_policy_execution/"
    "official_unitree_g1_policy_execution_manifest.json"
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _append_unique(target: list[str], values: Iterable[Any]) -> None:
    for value in values:
        text = _string(value)
        if text and text not in target:
            target.append(text)


def _existing_paths(paths: Iterable[Any]) -> list[str]:
    existing: list[str] = []
    for value in paths:
        text = _string(value)
        if text and Path(text).exists():
            existing.append(text)
    return existing


def _image_nonblank(path: Path) -> bool:
    try:
        from PIL import Image

        image = Image.open(path).convert("L")
        extrema = image.getextrema()
    except Exception:
        return False
    return bool(extrema and extrema[0] != extrema[1])


def _frame_evidence(frame_paths: Iterable[Any]) -> tuple[list[str], list[str]]:
    evidence: list[str] = []
    blockers: list[str] = []
    for raw_path in frame_paths:
        path_text = _string(raw_path)
        if not path_text:
            continue
        path = Path(path_text)
        if not path.is_file():
            blockers.append(f"missing_sim_frame:{path_text}")
            continue
        evidence.append(str(path))
        if not _image_nonblank(path):
            blockers.append(f"blank_or_unreadable_sim_frame:{path_text}")
    return evidence, blockers


def _proof_item(
    *,
    proven: bool,
    status: str,
    evidence: Iterable[Any],
    blockers: Iterable[Any],
    claim_boundary: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_list: list[str] = []
    blocker_list: list[str] = []
    _append_unique(evidence_list, evidence)
    _append_unique(blocker_list, blockers)
    item: dict[str, Any] = {
        "proven": proven,
        "status": status,
        "evidence": evidence_list,
        "blockers": blocker_list,
        "claim_boundary": claim_boundary,
    }
    if details:
        item["details"] = dict(details)
    return item


def _select_runpod_live_execution_proof(root: Path) -> tuple[Path, dict[str, Any] | None]:
    default_path = root / "pipeline" / "g1_controlled_proof_setup" / "runpod_live_execution_proof.json"
    candidates = [default_path]
    signed_dir = root / "pipeline" / "g1_controlled_proof_setup" / "signed_runpod_io"
    if signed_dir.is_dir():
        candidates.extend(sorted(signed_dir.glob("runpod_live_execution_proof.*.json")))
    job_root = root / "pipeline" / "robot_eval_jobs"
    if job_root.is_dir():
        candidates.extend(sorted(job_root.glob("*/runpod_live_execution_proof.json")))
        candidates.extend(sorted(job_root.glob("*/runpod_live_execution_proof.*.json")))

    def score(payload: Mapping[str, Any] | None) -> tuple[int, int, int, int]:
        if not payload:
            return (-1, -1, -1, -1)
        return (
            1 if payload.get("production_runpod_worker_execution_proven") is True else 0,
            1 if payload.get("simulator_execution_proven") is True else 0,
            1 if payload.get("shutdown_or_termination_proof") is True else 0,
            1 if not _as_list(payload.get("blockers")) else 0,
        )

    best_path = default_path
    best_payload = optional_read_json(default_path)
    best_score = score(best_payload)
    for candidate in candidates[1:]:
        payload = optional_read_json(candidate)
        candidate_score = score(payload)
        if candidate_score > best_score or (
            candidate_score == best_score and str(candidate) > str(best_path)
        ):
            best_path = candidate
            best_payload = payload
            best_score = candidate_score
    return best_path, best_payload


def _select_webapp_route_forwarding_proof(root: Path) -> tuple[Path, dict[str, Any] | None]:
    proof_dir = root / "pipeline" / "webapp_route_forwarding_proof"
    default_path = proof_dir / "webapp_route_forwarding_proof.json"
    candidates = [default_path]
    if proof_dir.is_dir():
        for candidate in sorted(proof_dir.glob("webapp_route_forwarding_proof*.json")):
            if candidate not in candidates:
                candidates.append(candidate)

    def score(payload: Mapping[str, Any] | None) -> tuple[int, int, int, int, str]:
        if not payload:
            return (-1, -1, -1, -1, "")
        boundary = _mapping(payload.get("proof_boundary"))
        webapp_route = _mapping(payload.get("webapp_route"))
        pipeline_forward = _mapping(payload.get("pipeline_forward"))
        pipeline_intake = _mapping(payload.get("pipeline_intake"))
        return (
            1 if payload.get("status") == "forwarded_to_pipeline_intake" else 0,
            1 if boundary.get("production_live_webapp_forwarding_proven") is True else 0,
            1 if webapp_route.get("full_production_webapp_deployment_proven") is True else 0,
            1 if pipeline_forward.get("accepted") is True and pipeline_intake.get("accepted") is True else 0,
            _string(payload.get("generated_at")),
        )

    best_path = default_path
    best_payload = optional_read_json(default_path)
    best_score = score(best_payload)
    for candidate in candidates[1:]:
        payload = optional_read_json(candidate)
        candidate_score = score(payload)
        if candidate_score > best_score or (
            candidate_score == best_score and str(candidate) > str(best_path)
        ):
            best_path = candidate
            best_payload = payload
            best_score = candidate_score
    return best_path, best_payload


def _mujoco_gate(path: Path, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return _proof_item(
            proven=False,
            status="missing_mujoco_g1_simulator_output",
            evidence=[],
            blockers=[f"missing_file:{path}"],
            claim_boundary="Simulator beta requires a completed MuJoCo Unitree G1 run.",
        )
    artifacts = _mapping(payload.get("artifact_paths"))
    frames, frame_blockers = _frame_evidence(_as_list(artifacts.get("frames")))
    blockers: list[str] = []
    if payload.get("status") != "completed":
        blockers.append(f"mujoco_status:{_string(payload.get('status')) or 'missing'}")
    for field in (
        "scene_loaded",
        "unitree_g1_asset_spawned",
        "mujoco_g1_asset_execution_proven",
        "default_sim_policy_execution_proven",
        "sim_robot_pov_evidence_proven",
    ):
        if payload.get(field) is not True:
            blockers.append(f"{field}_not_true")
    _append_unique(blockers, frame_blockers)
    attempt = _mapping((_as_list(payload.get("attempts")) or [{}])[0])
    metrics = _mapping(attempt.get("metrics"))
    if attempt.get("success") is not True:
        blockers.append("mujoco_attempt_not_successful")
    if int(metrics.get("simulated_step_count") or 0) <= 0:
        blockers.append("mujoco_simulated_step_count_missing")
    evidence = _existing_paths(
        [
            path,
            artifacts.get("scene_trace"),
            artifacts.get("spawn_trace"),
            artifacts.get("policy_trace"),
            artifacts.get("sim_robot_pov_evidence"),
            artifacts.get("artifact_manifest"),
        ]
    )
    evidence.extend(frames)
    return _proof_item(
        proven=not blockers,
        status="proven" if not blockers else "blocked",
        evidence=evidence,
        blockers=blockers,
        claim_boundary=(
            "Proves simulator-side Unitree G1 scene load, spawn, default walk_to_target "
            "execution, and simulator POV frames. It does not claim physical robot readiness."
        ),
        details={
            "simulator_backend": payload.get("simulator_backend"),
            "mujoco_version": payload.get("mujoco_version"),
            "simulated_step_count": metrics.get("simulated_step_count"),
            "frame_count": len(frames),
        },
    )


def _official_policy_gate(path: Path, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return _proof_item(
            proven=False,
            status="missing_official_unitree_g1_policy_execution",
            evidence=[],
            blockers=[f"missing_file:{path}"],
            claim_boundary="Simulator beta requires a completed official Unitree G1 policy run.",
        )
    boundary = _mapping(payload.get("proof_boundary"))
    execution = _mapping(payload.get("execution"))
    metrics = _mapping(payload.get("metrics"))
    trace_path = Path(_string(execution.get("trace_path")))
    blockers: list[str] = []
    if payload.get("status") != "completed":
        blockers.append(f"official_policy_status:{_string(payload.get('status')) or 'missing'}")
    if boundary.get("non_default_policy_execution_trace_proven") is not True:
        blockers.append("non_default_policy_execution_trace_not_proven")
    if boundary.get("policy_metrics_tied_to_scenario_variation") is not True:
        blockers.append("policy_metrics_not_tied_to_scenario")
    if metrics.get("finite_state") is not True:
        blockers.append("official_policy_finite_state_not_true")
    if metrics.get("finite_actions") is not True:
        blockers.append("official_policy_finite_actions_not_true")
    if not trace_path.is_file():
        blockers.append(f"missing_policy_trace:{trace_path}")
        trace_rows = 0
    else:
        trace_rows = len(trace_path.read_text(encoding="utf-8").splitlines())
        if trace_rows <= 0:
            blockers.append("empty_policy_execution_trace")
    return _proof_item(
        proven=not blockers,
        status="proven" if not blockers else "blocked",
        evidence=_existing_paths([path, execution.get("trace_path"), execution.get("metrics_path")]),
        blockers=blockers,
        claim_boundary=(
            "Proves the official Unitree RL Gym G1 policy executes headlessly in MuJoCo "
            "with finite actions and metrics. It is not physical policy approval."
        ),
        details={
            "policy_id": payload.get("policy_id"),
            "pinned_commit": _mapping(payload.get("source_repository")).get("pinned_commit"),
            "sim_time_s": metrics.get("sim_time_s"),
            "steps": metrics.get("steps"),
            "control_updates": metrics.get("control_updates"),
            "trace_rows": trace_rows,
        },
    )


def _runpod_gate(path: Path, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return _proof_item(
            proven=False,
            status="missing_runpod_live_execution_proof",
            evidence=[],
            blockers=[f"missing_file:{path}"],
            claim_boundary="Simulator beta requires production RunPod worker proof or an explicit no-RunPod scope.",
        )
    blockers = [str(blocker) for blocker in _as_list(payload.get("blockers")) if _string(blocker)]
    if payload.get("production_runpod_worker_execution_proven") is not True:
        blockers.append("production_runpod_worker_execution_not_proven")
    if payload.get("simulator_execution_proven") is not True:
        blockers.append("runpod_simulator_execution_not_proven")
    if payload.get("shutdown_or_termination_proof") is not True:
        blockers.append("runpod_shutdown_proof_missing")
    return _proof_item(
        proven=not blockers,
        status="proven" if not blockers else "blocked",
        evidence=_existing_paths([path, payload.get("runtime_manifest_path")]),
        blockers=blockers,
        claim_boundary=(
            "Proves production RunPod worker execution and shutdown accounting for simulator "
            "beta; it does not claim a physical robot run."
        ),
        details={
            "active_pod_count_before": payload.get("active_pod_count_before"),
            "active_pod_count_after": payload.get("active_pod_count_after"),
            "api_call_performed": payload.get("api_call_performed"),
            "shutdown_or_termination_proof": payload.get("shutdown_or_termination_proof"),
        },
    )


def _webapp_gate(path: Path, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return _proof_item(
            proven=False,
            status="missing_webapp_route_forwarding_proof",
            evidence=[],
            blockers=[f"missing_file:{path}"],
            claim_boundary="Simulator beta requires a production WebApp request forwarded into Pipeline.",
        )
    boundary = _mapping(payload.get("proof_boundary"))
    webapp_route = _mapping(payload.get("webapp_route"))
    pipeline_forward = _mapping(payload.get("pipeline_forward"))
    pipeline_intake = _mapping(payload.get("pipeline_intake"))
    blockers: list[str] = []
    if payload.get("status") != "forwarded_to_pipeline_intake":
        blockers.append(f"webapp_route_status:{_string(payload.get('status')) or 'missing'}")
    if boundary.get("production_live_webapp_forwarding_proven") is not True:
        blockers.append("production_live_webapp_forwarding_not_proven")
    if webapp_route.get("full_production_webapp_deployment_proven") is not True:
        blockers.append("production_webapp_deployment_not_proven")
    if pipeline_forward.get("accepted") is not True:
        blockers.append("pipeline_forward_not_accepted")
    if pipeline_intake.get("accepted") is not True:
        blockers.append("pipeline_intake_not_accepted")
    _append_unique(blockers, pipeline_intake.get("input_blockers") or [])
    return _proof_item(
        proven=not blockers,
        status="proven" if not blockers else "blocked",
        evidence=_existing_paths([path]),
        blockers=blockers,
        claim_boundary=(
            "Proves the customer-facing request path can stage a simulator job request. "
            "It does not claim physical robot readiness."
        ),
        details={
            "http_status": webapp_route.get("http_status"),
            "pipeline_status": pipeline_forward.get("pipeline_status"),
            "pipeline_intake_status": pipeline_intake.get("status"),
        },
    )


def build_simulator_beta_readiness(
    *,
    capture_root: str | Path,
    output_dir: str | Path | None = None,
    mujoco_output_path: str | Path | None = None,
    official_policy_execution_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(capture_root).expanduser().resolve()
    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root / DEFAULT_OUTPUT_RELATIVE
    )
    ensure_dir(out_dir)
    mujoco_path = (
        Path(mujoco_output_path).expanduser().resolve()
        if mujoco_output_path
        else root / DEFAULT_MUJOCO_OUTPUT_RELATIVE
    )
    policy_path = (
        Path(official_policy_execution_path).expanduser().resolve()
        if official_policy_execution_path
        else root / DEFAULT_POLICY_EXECUTION_RELATIVE
    )
    runpod_path, runpod_payload = _select_runpod_live_execution_proof(root)
    webapp_path, webapp_payload = _select_webapp_route_forwarding_proof(root)
    gates = {
        "site_capture_mujoco_g1_run": _mujoco_gate(mujoco_path, optional_read_json(mujoco_path)),
        "official_unitree_g1_policy_execution": _official_policy_gate(
            policy_path,
            optional_read_json(policy_path),
        ),
        "production_runpod_worker_execution": _runpod_gate(runpod_path, runpod_payload),
        "customer_website_to_pipeline_request": _webapp_gate(webapp_path, webapp_payload),
    }
    blocking_gate_ids = [
        gate_id for gate_id, gate in gates.items() if gate.get("proven") is not True
    ]
    out_of_scope = {
        "physical_robot_readiness": "out_of_scope_for_simulator_beta",
        "real_robot_pov": "out_of_scope_for_simulator_beta",
        "physical_safety_validation": "out_of_scope_for_simulator_beta",
        "physical_robot_team_policy_acceptance": "out_of_scope_for_simulator_beta",
    }
    status = "ready_for_simulator_beta" if not blocking_gate_ids else "blocked_simulator_beta"
    manifest_path = out_dir / "simulator_beta_readiness_manifest.json"
    manifest = {
        "schema_version": SIMULATOR_BETA_READINESS_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "ready_for_simulator_beta": not blocking_gate_ids,
        "capture_root": str(root),
        "scope": "simulator_only",
        "default_robot": {
            "make_model": "Unitree G1",
            "robot_profile_id": "unitree_g1_humanoid",
            "simulator_backend": "mujoco",
        },
        "blocking_gate_ids": blocking_gate_ids,
        "gates": gates,
        "out_of_scope_gates": out_of_scope,
        "claim_boundary": {
            "simulator_beta_only": True,
            "physical_robot_readiness_claimed": False,
            "real_robot_pov_claimed": False,
            "physical_safety_validation_claimed": False,
            "customer_can_request_simulator_job": gates[
                "customer_website_to_pipeline_request"
            ].get("proven")
            is True,
            "production_runpod_worker_execution_proven": gates[
                "production_runpod_worker_execution"
            ].get("proven")
            is True,
            "public_claim_upgrade_allowed": False,
        },
        "artifacts": {
            "manifest": str(manifest_path),
            "mujoco_output": str(mujoco_path),
            "official_policy_execution": str(policy_path),
            "runpod_live_execution_proof": str(runpod_path),
            "webapp_route_forwarding_proof": str(webapp_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--mujoco-output")
    parser.add_argument("--official-policy-execution")
    args = parser.parse_args(argv)
    manifest = build_simulator_beta_readiness(
        capture_root=args.capture_root,
        output_dir=args.output_dir,
        mujoco_output_path=args.mujoco_output,
        official_policy_execution_path=args.official_policy_execution,
    )
    print(manifest["artifacts"]["manifest"])
    return 0 if manifest["ready_for_simulator_beta"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
