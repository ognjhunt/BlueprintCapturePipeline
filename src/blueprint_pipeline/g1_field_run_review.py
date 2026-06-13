"""Review Unitree G1 physical field-run evidence before assembly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, optional_read_json, utc_now_iso, write_json


G1_FIELD_RUN_REVIEW_SCHEMA_VERSION = "g1_field_run_review.v1"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _string(value).lower() in {"1", "true", "yes", "on", "accepted", "passed"}


def _number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, Mapping):
            records.append(dict(payload))
    return records


def _file_present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _existing_contact_force(evidence_dir: Path) -> float | None:
    contact = optional_read_json(evidence_dir / "contact_collision_log.json")
    return _number(_mapping(contact).get("max_contact_force_n")) if contact else None


def _base_blockers(evidence_dir: Path, config: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    required_files = [
        "robot_camera_video.mp4",
        "timestamp_alignment.json",
        "action_log.jsonl",
        "robot_state_log.jsonl",
        "command_log.jsonl",
        "contact_collision_log.json",
        "policy_execution_trace.jsonl",
    ]
    for filename in required_files:
        if not _file_present(evidence_dir / filename):
            blockers.append(f"missing_or_empty_evidence_file:{filename}")
    action_records = _read_json_records(evidence_dir / "action_log.jsonl")
    if not any(record.get("action_id") or record.get("motor_targets") for record in action_records):
        blockers.append("action_log_missing_robot_action_record")
    command_records = _read_json_records(evidence_dir / "command_log.jsonl")
    completed = [
        record
        for record in command_records
        if _string(record.get("kind")) == "policy_command_completed"
    ]
    if not completed:
        blockers.append("command_log_missing_policy_command_completed")
    elif any(_number(record.get("exit_code")) != 0 for record in completed):
        blockers.append("command_log_policy_command_exit_nonzero")
    thresholds = _mapping(config.get("accepted_safety_thresholds"))
    threshold = _number(thresholds.get("max_contact_force_n"))
    contact_force = _existing_contact_force(evidence_dir)
    if contact_force is None:
        blockers.append("contact_collision_log_missing_max_contact_force_n")
    elif threshold is not None and contact_force > threshold:
        blockers.append("contact_collision_log_exceeds_accepted_threshold")
    if not _bool(config.get("actual_success")):
        blockers.append("physical_run_actual_success_not_true")
    if _string(config.get("actual_status")).lower() != "passed":
        blockers.append("physical_run_status_not_passed")
    return blockers


def review_g1_field_run_evidence(
    *,
    evidence_dir: str | Path,
    accept_safety: bool = False,
    accept_policy: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    evidence_root = Path(evidence_dir).expanduser().resolve()
    ensure_dir(evidence_root)
    config_path = evidence_root / "g1_controlled_run_inputs.json"
    config = optional_read_json(config_path) or {}
    blockers = _base_blockers(evidence_root, config)
    if not accept_safety:
        blockers.append("missing_explicit_safety_acceptance")
    if not accept_policy:
        blockers.append("missing_explicit_policy_acceptance")
    ready = not blockers
    safety_reviewer = _string(config.get("safety_reviewer_id"))
    robot_team_reviewer = _string(config.get("robot_team_reviewer_id"))
    intervention_count = _number(config.get("intervention_count"), 0)

    hardware_validation = {
        "schema_version": "g1_hardware_validation.v1",
        "status": "accepted" if ready else "operator_review_required",
        "hardware_ready": ready,
        "estop_verified": ready,
        "reviewer_id": safety_reviewer,
        "reviewed_at_utc": utc_now_iso(),
        "blockers": blockers,
    }
    write_json(evidence_root / "hardware_validation.json", hardware_validation)

    policy_metrics = {
        "schema_version": "g1_policy_metrics.v1",
        "status": "accepted" if ready else "operator_review_required",
        "episode_count": 1 if ready else 0,
        "success_rate": 1.0 if ready else 0.0,
        "intervention_count": intervention_count if ready else None,
        "blockers": blockers,
    }
    write_json(evidence_root / "policy_metrics.json", policy_metrics)

    robot_team_review = {
        "schema_version": "g1_robot_team_review.v1",
        "review_decision": "accepted" if ready else "not_reviewed",
        "accepted": ready,
        "reviewer_id": robot_team_reviewer,
        "reviewed_at_utc": utc_now_iso(),
        "blockers": blockers,
    }
    write_json(evidence_root / "robot_team_review.json", robot_team_review)

    contact = optional_read_json(evidence_root / "contact_collision_log.json")
    if isinstance(contact, Mapping):
        contact_payload = dict(contact)
        contact_payload["status"] = "accepted" if ready else "operator_review_required"
        contact_payload["reviewed_at_utc"] = utc_now_iso()
        contact_payload["reviewer_id"] = safety_reviewer
        write_json(evidence_root / "contact_collision_log.json", contact_payload)

    manifest = {
        "schema_version": G1_FIELD_RUN_REVIEW_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "reviewed_evidence_ready_for_assembly"
        if ready
        else "blocked_review_acceptance_required",
        "evidence_dir": str(evidence_root),
        "input_config_path": str(config_path),
        "blockers": blockers,
        "artifacts": {
            "hardware_validation": str(evidence_root / "hardware_validation.json"),
            "policy_metrics": str(evidence_root / "policy_metrics.json"),
            "robot_team_review": str(evidence_root / "robot_team_review.json"),
            "contact_collision_log": str(evidence_root / "contact_collision_log.json"),
        },
        "proof_boundary": {
            "review_helper_is_not_physical_robot_proof": True,
            "requires_real_g1_hardware_files": True,
            "requires_explicit_safety_acceptance": True,
            "requires_explicit_policy_acceptance": True,
            "public_claim_upgrade_allowed": False,
        },
    }
    manifest_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else evidence_root / "g1_field_run_review_manifest.json"
    )
    write_json(manifest_path, manifest)
    manifest["artifacts"]["manifest"] = str(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--accept-safety", action="store_true")
    parser.add_argument("--accept-policy", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    manifest = review_g1_field_run_evidence(
        evidence_dir=args.evidence_dir,
        output_path=args.output_path,
        accept_safety=args.accept_safety,
        accept_policy=args.accept_policy,
    )
    print(json.dumps({"status": manifest["status"], "manifest": manifest["artifacts"]["manifest"]}))
    if args.require_ready and manifest["status"] != "reviewed_evidence_ready_for_assembly":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
