"""Proof-boundary audit for live pipeline control-plane artifacts.

This verifier reads the always-on control-plane manifest, the generated
external-input packet, and the setup manifest. It does not perform live actions.
Its job is to prove that the control-plane artifacts are internally consistent,
that they do not leak secrets or overclaim proof, and that any remaining live
readiness blockers are represented as external inputs rather than hidden
success.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .live_pipeline_control_plane import (
    LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION,
    LIVE_PIPELINE_EXTERNAL_INPUT_PACKET_SCHEMA_VERSION,
    LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION,
)


LIVE_PIPELINE_PROOF_AUDIT_SCHEMA_VERSION = "blueprint_live_pipeline_proof_audit.v1"

FORBIDDEN_TRUE_FIELDS = (
    "simulator_execution_proven",
    "robot_policy_execution_proven",
    "robot_readiness_proven",
    "physics_contact_validated",
    "safety_validated",
    "training_completed",
    "public_claim_upgrade_allowed",
)

CORE_EXTERNAL_INPUT_IDS = (
    "webapp_upstream_truth",
    "isaac_lab_arena_owner_evidence",
    "real_robot_pov_evidence",
    "live_robot_eval_closure_evidence",
    "real_world_deployment_outcomes",
    "predicted_vs_actual_exact_match_keys",
    "real_world_deployment_outcome_owner_evidence",
    "robot_team_policy_package",
)

CORE_GOAL_REQUIREMENTS = (
    "arena_result_ingest",
    "500_scenario_scheduler",
    "policy_adapters",
    "clips",
    "vision_labeling",
    "review_resolution",
    "dataset_packaging",
    "customer_handoff_report",
    "storage_delivery",
    "rerun_loop",
    "live_agents_operator",
    "live_codex_operator",
    "webapp_upstream_truth",
    "owner_arena_evidence",
    "real_robot_pov_evidence",
    "live_robot_eval_closure_evidence",
    "real_world_deployment_outcomes",
    "predicted_vs_actual_exact_match_keys",
    "real_world_deployment_outcome_owner_evidence",
    "robot_team_policy_package",
)

BLOCKER_PACKET_REQUIRED_FIELDS = (
    "id",
    "owner",
    "required_input",
    "safe_proof_command",
    "retry_condition",
    "resume_target",
    "disallowed_workaround",
)


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_mapping(path: Path) -> Dict[str, Any]:
    payload = read_json_any(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return dict(payload)


def _artifact(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "size_bytes": 0}
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def _path_from_value(value: Any) -> Path | None:
    text = str(value or "").strip()
    return Path(text) if text else None


def _proof_violations(payload: Mapping[str, Any], *, artifact_name: str) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    for field in FORBIDDEN_TRUE_FIELDS:
        if bool(payload.get(field)):
            violations.append(
                {
                    "artifact": artifact_name,
                    "field": field,
                    "value": True,
                    "reason": "forbidden_proof_boolean_true",
                }
            )
    for boundary_name in ("control_plane_boundary", "claim_boundary", "proof_boundary"):
        boundary = _mapping(payload.get(boundary_name))
        for field in FORBIDDEN_TRUE_FIELDS:
            if bool(boundary.get(field)):
                violations.append(
                    {
                        "artifact": artifact_name,
                        "field": f"{boundary_name}.{field}",
                        "value": True,
                        "reason": "forbidden_boundary_boolean_true",
                    }
                )
    return violations


def _required_input_ids(packet: Mapping[str, Any]) -> List[str]:
    inputs = packet.get("required_inputs")
    if not isinstance(inputs, list):
        return []
    return [
        str(item.get("id"))
        for item in inputs
        if isinstance(item, Mapping) and str(item.get("id") or "").strip()
    ]


def _enablement_input_ids(packet: Mapping[str, Any]) -> List[str]:
    inputs = packet.get("enablement_inputs")
    if not isinstance(inputs, list):
        return []
    return [
        str(item.get("id"))
        for item in inputs
        if isinstance(item, Mapping) and str(item.get("id") or "").strip()
    ]


def _blocker_packet_audit(packet: Mapping[str, Any]) -> Dict[str, Any]:
    missing_packet_ids: List[str] = []
    invalid_packet_fields: Dict[str, List[str]] = {}
    for section_name in ("required_inputs", "enablement_inputs"):
        inputs = packet.get(section_name)
        if not isinstance(inputs, list):
            continue
        for item in inputs:
            if not isinstance(item, Mapping):
                continue
            input_id = str(item.get("id") or "").strip()
            if not input_id:
                continue
            blocker_packet = item.get("blocker_packet")
            if not isinstance(blocker_packet, Mapping):
                missing_packet_ids.append(input_id)
                continue
            missing_fields = [
                field
                for field in BLOCKER_PACKET_REQUIRED_FIELDS
                if not str(blocker_packet.get(field) or "").strip()
            ]
            if str(blocker_packet.get("id") or "").strip() != input_id:
                missing_fields.append("id_matches_input")
            if missing_fields:
                invalid_packet_fields[input_id] = missing_fields
    return {
        "status": (
            "passed"
            if not missing_packet_ids and not invalid_packet_fields
            else "failed"
        ),
        "missing_blocker_packet_input_ids": missing_packet_ids,
        "invalid_blocker_packet_fields": invalid_packet_fields,
        "required_fields": list(BLOCKER_PACKET_REQUIRED_FIELDS),
        "proof_boundary": (
            "Blocker packets are operational resume instructions, not evidence that external "
            "inputs are complete."
        ),
    }


def _setup_section(setup_manifest: Mapping[str, Any], name: str) -> Dict[str, Any]:
    sections = _mapping(setup_manifest.get("sections"))
    return _mapping(sections.get(name))


def _goal_requirement_audit(
    *,
    manifest: Mapping[str, Any],
    packet: Mapping[str, Any],
    setup_manifest: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    required_inputs = set(_required_input_ids(packet))
    enablement_inputs = set(_enablement_input_ids(packet))
    arena_section = _setup_section(setup_manifest, "real_arena_execution")
    package_dir = _mapping(packet.get("configured_paths")).get("package_dir")
    package_audit = _mapping(_setup_section(setup_manifest, "local_deterministic_lane").get("package_audit"))
    package_audit_status = package_audit.get("audit_status")
    local_package_ready = package_audit_status == "passed" or not bool(package_audit.get("blockers"))
    webapp_ready = bool(manifest.get("effective_webapp_upstream_truth_ready")) and (
        "webapp_upstream_truth" not in required_inputs
    )
    arena_ready = bool(arena_section.get("ready")) and (
        "isaac_lab_arena_owner_evidence" not in required_inputs
    )
    real_robot_pov_ready = "real_robot_pov_evidence" not in required_inputs
    live_agents_ready = (
        bool(_setup_section(setup_manifest, "live_agents_operator").get("ready"))
        and "live_agents_operator" not in enablement_inputs
    )
    live_codex_ready = (
        bool(_setup_section(setup_manifest, "live_codex_operator").get("ready"))
        and "live_codex_operator" not in enablement_inputs
    )
    vision_ready = (
        bool(_setup_section(setup_manifest, "rollout_vision_labeling").get("ready"))
        and "rollout_vision_labeling" not in enablement_inputs
    )
    delivery_ready = (
        bool(_setup_section(setup_manifest, "delivery_upload").get("ready"))
        and "delivery_upload" not in enablement_inputs
    )
    live_closure_evidence_ready = "live_robot_eval_closure_evidence" not in required_inputs
    deployment_outcomes_ready = "real_world_deployment_outcomes" not in required_inputs
    prediction_match_keys_ready = "predicted_vs_actual_exact_match_keys" not in required_inputs
    deployment_outcome_owner_evidence_ready = (
        deployment_outcomes_ready
        and prediction_match_keys_ready
        and "real_world_deployment_outcome_owner_evidence" not in required_inputs
    )
    policy_package_ready = "robot_team_policy_package" not in required_inputs

    package_lane = {
        "status": "ready" if local_package_ready else "not_proven_in_current_control_plane",
        "evidence": package_dir or package_audit.get("artifact", {}).get("path"),
        "proof_boundary": "package artifacts prove local ingest/package surfaces only",
    }
    return {
        "arena_result_ingest": dict(package_lane),
        "500_scenario_scheduler": dict(package_lane),
        "policy_adapters": dict(package_lane),
        "clips": dict(package_lane),
        "review_resolution": dict(package_lane),
        "dataset_packaging": dict(package_lane),
        "customer_handoff_report": dict(package_lane),
        "rerun_loop": dict(package_lane),
        "vision_labeling": {
            "status": "ready" if vision_ready else "enablement_missing_or_not_configured",
            "proof_boundary": "model labels remain review-required support evidence",
        },
        "storage_delivery": {
            "status": "ready" if delivery_ready else "enablement_missing_or_not_configured",
            "proof_boundary": "delivery artifacts do not prove robot readiness",
        },
        "live_agents_operator": {
            "status": "ready" if live_agents_ready else "enablement_missing_or_not_configured",
            "proof_boundary": "agents cannot directly upgrade proof booleans",
        },
        "live_codex_operator": {
            "status": "ready" if live_codex_ready else "enablement_missing_or_not_configured",
            "proof_boundary": "codex operators cannot directly upgrade proof booleans",
        },
        "webapp_upstream_truth": {
            "status": "ready" if webapp_ready else "external_input_missing",
            "proof_boundary": "requires real capture/job IDs from WebApp artifacts",
        },
        "owner_arena_evidence": {
            "status": "ready" if arena_ready else "external_input_missing",
            "arena_section_status": arena_section.get("status"),
            "proof_boundary": "requires owner-system Arena command or result artifacts",
        },
        "real_robot_pov_evidence": {
            "status": "ready" if real_robot_pov_ready else "external_input_missing",
            "proof_boundary": (
                "requires real robot camera/action evidence aligned to every scenario eval run"
            ),
        },
        "live_robot_eval_closure_evidence": {
            "status": "ready" if live_closure_evidence_ready else "external_input_missing",
            "proof_boundary": (
                "requires job-specific review, delivery, rights/privacy, and "
                "safety/contact/physics evidence for live closure"
            ),
        },
        "real_world_deployment_outcomes": {
            "status": "ready" if deployment_outcomes_ready else "external_input_missing",
            "proof_boundary": (
                "requires job-specific actual pilot or deployment outcomes for "
                "predicted-vs-actual calibration"
            ),
        },
        "predicted_vs_actual_exact_match_keys": {
            "status": (
                "ready" if prediction_match_keys_ready else "external_input_missing"
            ),
            "proof_boundary": (
                "requires scenario_eval_run_id and scenario_variation_instance_id "
                "before staged outcomes can be joined to predictions"
            ),
        },
        "real_world_deployment_outcome_owner_evidence": {
            "status": (
                "ready"
                if deployment_outcome_owner_evidence_ready
                else "external_input_missing"
            ),
            "proof_boundary": (
                "requires owner evidence on every actual outcome record before live "
                "real-world outcome proof can pass"
            ),
        },
        "robot_team_policy_package": {
            "status": "ready" if policy_package_ready else "external_input_missing",
            "proof_boundary": "requires one supported robot-team policy or trace modality",
        },
    }


def _staged_inputs_audit(
    *,
    staged_info: Mapping[str, Any],
    staged_manifest: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    status = str(staged_info.get("status") or "not_configured")
    return {
        "status": status,
        "artifact_exists": bool(artifact.get("exists")),
        "arena_results_ready": bool(staged_info.get("arena_results_ready")),
        "webapp_request_ready": bool(staged_info.get("webapp_request_ready")),
        "live_closure_evidence_ready": bool(
            staged_info.get("live_closure_evidence_ready")
        ),
        "live_closure_evidence_job_id": staged_info.get("live_closure_evidence_job_id"),
        "deployment_outcomes_ready": bool(staged_info.get("deployment_outcomes_ready")),
        "deployment_outcomes_owner_evidence_ready": bool(
            staged_info.get("deployment_outcomes_owner_evidence_ready")
        ),
        "deployment_outcomes_records_ready_for_calibration": bool(
            staged_info.get("deployment_outcomes_records_ready_for_calibration")
        ),
        "deployment_outcomes_prediction_match_keys_ready": bool(
            staged_info.get("deployment_outcomes_prediction_match_keys_ready")
        ),
        "deployment_outcomes_job_id": staged_info.get("deployment_outcomes_job_id"),
        "deployment_outcome_record_count": int(
            staged_info.get("deployment_outcome_record_count") or 0
        ),
        "deployment_outcome_prediction_match_key_record_count": int(
            staged_info.get("deployment_outcome_prediction_match_key_record_count") or 0
        ),
        "deployment_outcome_missing_prediction_match_key_record_ids": list(
            staged_info.get("deployment_outcome_missing_prediction_match_key_record_ids") or []
        ),
        "deployment_outcome_owner_evidence_record_count": int(
            staged_info.get("deployment_outcome_owner_evidence_record_count") or 0
        ),
        "deployment_outcome_missing_owner_evidence_record_ids": list(
            staged_info.get("deployment_outcome_missing_owner_evidence_record_ids") or []
        ),
        "policy_package_ready": bool(staged_info.get("policy_package_ready")),
        "policy_package_job_id": staged_info.get("policy_package_job_id"),
        "policy_package_selected_modalities": list(
            staged_info.get("policy_package_selected_modalities") or []
        ),
        "real_robot_pov_ready": bool(staged_info.get("real_robot_pov_ready")),
        "real_robot_pov_job_id": staged_info.get("real_robot_pov_job_id"),
        "real_robot_pov_record_count": int(
            staged_info.get("real_robot_pov_record_count") or 0
        ),
        "real_robot_pov_exact_key_record_count": int(
            staged_info.get("real_robot_pov_exact_key_record_count") or 0
        ),
        "real_robot_pov_camera_video_record_count": int(
            staged_info.get("real_robot_pov_camera_video_record_count") or 0
        ),
        "real_robot_pov_action_log_record_count": int(
            staged_info.get("real_robot_pov_action_log_record_count") or 0
        ),
        "real_robot_pov_timestamp_alignment_record_count": int(
            staged_info.get("real_robot_pov_timestamp_alignment_record_count") or 0
        ),
        "real_robot_pov_evidence_record_count": int(
            staged_info.get("real_robot_pov_evidence_record_count") or 0
        ),
        "real_robot_pov_missing_exact_key_record_ids": list(
            staged_info.get("real_robot_pov_missing_exact_key_record_ids") or []
        ),
        "real_robot_pov_missing_evidence_record_ids": list(
            staged_info.get("real_robot_pov_missing_evidence_record_ids") or []
        ),
        "blockers": list(staged_info.get("blockers") or []),
        "schema_version": staged_manifest.get("schema_version"),
        "proof_boundary": "staged inputs are validated pointers only, not proof claims",
    }


def _audit_status(
    *,
    internal_blockers: Sequence[str],
    required_input_ids: Sequence[str],
    require_live_ready: bool,
) -> str:
    if (
        require_live_ready
        and required_input_ids
        and set(internal_blockers) <= {"required_live_inputs_missing"}
    ):
        return "failed_live_ready_required"
    if internal_blockers:
        return "failed"
    if required_input_ids:
        return "failed_live_ready_required" if require_live_ready else "passed_external_inputs_blocked"
    return "passed_live_ready_inputs_present"


def build_live_pipeline_proof_audit(
    *,
    manifest_path: str | Path,
    output_path: str | Path | None = None,
    require_live_ready: bool = False,
) -> Dict[str, Any]:
    resolved_manifest_path = Path(manifest_path).resolve()
    generated_at = utc_now_iso()
    internal_blockers: List[str] = []
    artifacts: Dict[str, Any] = {
        "control_plane_manifest": _artifact(resolved_manifest_path),
    }
    manifest = _read_mapping(resolved_manifest_path)
    if manifest.get("schema_version") != LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION:
        internal_blockers.append("control_plane_manifest_schema_mismatch")
    if bool(manifest.get("secrets_leaked")):
        internal_blockers.append("control_plane_manifest_reports_secret_leak")

    packet_info = _mapping(manifest.get("external_input_packet"))
    staged_info = _mapping(manifest.get("staged_inputs"))
    packet_path = _path_from_value(packet_info.get("path"))
    packet_markdown_path = _path_from_value(packet_info.get("markdown_path"))
    setup_manifest_path = _path_from_value(manifest.get("setup_manifest_path"))
    staged_inputs_path = _path_from_value(staged_info.get("path"))
    artifacts["external_input_packet"] = _artifact(packet_path)
    artifacts["external_input_packet_markdown"] = _artifact(packet_markdown_path)
    artifacts["setup_manifest"] = _artifact(setup_manifest_path)
    artifacts["staged_inputs"] = _artifact(staged_inputs_path)

    packet: Dict[str, Any] = {}
    setup_manifest: Dict[str, Any] = {}
    staged_manifest: Dict[str, Any] = {}
    if not artifacts["external_input_packet"]["exists"]:
        internal_blockers.append("external_input_packet_missing")
    else:
        packet = _read_mapping(packet_path or Path())
        if packet.get("schema_version") != LIVE_PIPELINE_EXTERNAL_INPUT_PACKET_SCHEMA_VERSION:
            internal_blockers.append("external_input_packet_schema_mismatch")
        if bool(packet.get("secrets_leaked")):
            internal_blockers.append("external_input_packet_reports_secret_leak")
        if packet.get("status") != packet_info.get("status"):
            internal_blockers.append("external_input_packet_status_mismatch")
    if not artifacts["external_input_packet_markdown"]["exists"]:
        internal_blockers.append("external_input_packet_markdown_missing")
    if not artifacts["setup_manifest"]["exists"]:
        internal_blockers.append("setup_manifest_missing")
    else:
        setup_manifest = _read_mapping(setup_manifest_path or Path())
    staged_status = str(staged_info.get("status") or "not_configured")
    if staged_status == "ready" and not artifacts["staged_inputs"]["exists"]:
        internal_blockers.append("staged_inputs_ready_but_missing")
    if staged_status == "blocked":
        internal_blockers.append("staged_inputs_blocked")
    if artifacts["staged_inputs"]["exists"]:
        staged_manifest = _read_mapping(staged_inputs_path or Path())
        if staged_manifest.get("schema_version") != LIVE_PIPELINE_STAGED_INPUTS_SCHEMA_VERSION:
            internal_blockers.append("staged_inputs_schema_mismatch")

    proof_violations = []
    proof_violations.extend(_proof_violations(manifest, artifact_name="control_plane_manifest"))
    if packet:
        proof_violations.extend(_proof_violations(packet, artifact_name="external_input_packet"))
    if setup_manifest:
        proof_violations.extend(_proof_violations(setup_manifest, artifact_name="setup_manifest"))
    if staged_manifest:
        proof_violations.extend(_proof_violations(staged_manifest, artifact_name="staged_inputs"))
    if proof_violations:
        internal_blockers.append("forbidden_proof_boundary_upgrade")

    blocker_packet_audit = _blocker_packet_audit(packet) if packet else {
        "status": "not_available",
        "missing_blocker_packet_input_ids": [],
        "invalid_blocker_packet_fields": {},
        "required_fields": list(BLOCKER_PACKET_REQUIRED_FIELDS),
        "proof_boundary": "external input packet was not available for blocker packet audit",
    }
    if blocker_packet_audit["missing_blocker_packet_input_ids"]:
        internal_blockers.append("external_input_packet_missing_blocker_packets")
    if blocker_packet_audit["invalid_blocker_packet_fields"]:
        internal_blockers.append("external_input_packet_invalid_blocker_packets")

    required_input_ids = _required_input_ids(packet)
    enablement_input_ids = _enablement_input_ids(packet)
    for input_id in required_input_ids:
        if input_id not in CORE_EXTERNAL_INPUT_IDS:
            internal_blockers.append(f"unexpected_required_input:{input_id}")

    deployment_outcomes_ready = "real_world_deployment_outcomes" not in required_input_ids
    prediction_match_keys_ready = "predicted_vs_actual_exact_match_keys" not in required_input_ids
    live_readiness = {
        "webapp_upstream_truth_ready": bool(
            manifest.get("effective_webapp_upstream_truth_ready")
        )
        and "webapp_upstream_truth" not in required_input_ids,
        "owner_arena_evidence_ready": "isaac_lab_arena_owner_evidence" not in required_input_ids,
        "real_robot_pov_evidence_ready": "real_robot_pov_evidence" not in required_input_ids,
        "live_closure_evidence_ready": (
            "live_robot_eval_closure_evidence" not in required_input_ids
        ),
        "deployment_outcomes_ready": deployment_outcomes_ready,
        "deployment_outcomes_prediction_match_keys_ready": prediction_match_keys_ready,
        "deployment_outcomes_owner_evidence_ready": (
            deployment_outcomes_ready
            and prediction_match_keys_ready
            and "real_world_deployment_outcome_owner_evidence" not in required_input_ids
        ),
        "policy_package_ready": "robot_team_policy_package" not in required_input_ids,
        "required_input_ids": required_input_ids,
        "enablement_input_ids": enablement_input_ids,
        "next_inputs_needed": list(manifest.get("next_inputs_needed") or []),
    }
    external_blockers = [
        input_id
        for input_id in CORE_EXTERNAL_INPUT_IDS
        if input_id in required_input_ids
    ]
    if require_live_ready and external_blockers:
        internal_blockers.append("required_live_inputs_missing")

    audit = {
        "schema_version": LIVE_PIPELINE_PROOF_AUDIT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": _audit_status(
            internal_blockers=internal_blockers,
            required_input_ids=required_input_ids,
            require_live_ready=require_live_ready,
        ),
        "require_live_ready": require_live_ready,
        "manifest_status": manifest.get("status"),
        "setup_status": manifest.get("setup_status"),
        "packet_status": packet.get("status"),
        "artifacts": artifacts,
        "internal_blockers": internal_blockers,
        "external_blockers": external_blockers,
        "proof_violations": proof_violations,
        "blocker_packet_audit": blocker_packet_audit,
        "live_readiness": live_readiness,
        "staged_inputs_audit": _staged_inputs_audit(
            staged_info=staged_info,
            staged_manifest=staged_manifest,
            artifact=artifacts["staged_inputs"],
        ),
        "goal_requirement_audit": _goal_requirement_audit(
            manifest=manifest,
            packet=packet,
            setup_manifest=setup_manifest,
        ),
        "proof_boundary": {
            "audit_performs_live_actions": False,
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    if output_path:
        path = Path(output_path).resolve()
    else:
        path = resolved_manifest_path.parent / "live_pipeline_proof_boundary_audit.json"
    ensure_dir(path.parent)
    audit["output_path"] = str(path)
    write_json(path, audit)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit live control-plane manifests for proof-boundary integrity."
    )
    parser.add_argument(
        "--manifest-path",
        default="/var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json",
    )
    parser.add_argument("--output-path")
    parser.add_argument("--require-live-ready", action="store_true")
    args = parser.parse_args(argv)
    result = build_live_pipeline_proof_audit(
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        require_live_ready=args.require_live_ready,
    )
    print(f"[live-pipeline-proof-audit] audit={result['output_path']}")
    print(f"[live-pipeline-proof-audit] status={result['status']}")
    if result["internal_blockers"]:
        print(f"[live-pipeline-proof-audit] internal_blockers={len(result['internal_blockers'])}")
    if result["external_blockers"]:
        print(f"[live-pipeline-proof-audit] external_blockers={len(result['external_blockers'])}")
    return 0 if not result["internal_blockers"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
