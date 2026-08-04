"""Fail-closed release gate from Arena controls to the frozen candidate schedule.

This module never launches a simulator, contacts a policy server, or allocates
paid compute. It compiles worker-produced receipts into one digest-bound
authorization for the already-approved 88 logical jobs only after native
controls, parity, materialization, and both media-complete policy dry-runs pass.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_founder_sim_protocol import (
    ALTERNATIVE_ID,
    BASELINE_ID,
    FounderSimProtocolError,
    admit_founder_sim_execution,
    build_founder_sim_protocol,
)
from .adp_isaac_lab_arena_request import build_arena_worker_request
from .adp_isaac_lab_arena_materialization import (
    ADMISSION_SCHEMA_VERSION as MATERIALIZED_ADMISSION_SCHEMA_VERSION,
)
from .adp_isaac_lab_arena_materialization import SCHEMA_VERSION as MATERIALIZATION_SCHEMA_VERSION
from .decision_evidence_contracts import canonical_digest
from .common import write_json


SCHEMA_VERSION = "adp_arena_candidate_execution_gate.v1"
CONTROL_SCHEMA_VERSION = "adp_arena_native_control_receipt.v1"
POLICY_DRY_RUN_SCHEMA_VERSION = "adp_arena_policy_dry_run_receipt.v1"

ZERO_CONTROL = "arena_zero_action_negative"
POSITIVE_CONTROL = "arena_replay_or_scripted_positive"
PARITY_CONTROL = "arena_droid_camera_action_reset_parity"
REQUIRED_CONTROLS = (ZERO_CONTROL, POSITIVE_CONTROL, PARITY_CONTROL)
REQUIRED_CANDIDATES = (BASELINE_ID, ALTERNATIVE_ID)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _digest_matches(receipt: Mapping[str, Any], *, field: str) -> bool:
    return _is_digest(receipt.get(field)) and receipt.get(field) == canonical_digest(
        receipt, digest_field=field
    )


def _media_blockers(receipt: Mapping[str, Any], *, prefix: str) -> list[str]:
    blockers: list[str] = []
    media = _mapping(receipt.get("visual_evidence"))
    requirements = {
        "lossless_policy_input_images": "lossless_policy_input_images_missing",
        "terminal_image": "terminal_image_missing",
        "frame_manifest": "frame_manifest_missing",
        "review_video": "review_video_missing",
        "independent_grader_provenance": "independent_grader_provenance_missing",
    }
    for key, blocker in requirements.items():
        if media.get(key) is not True:
            blockers.append(f"{prefix}_{blocker}")
    if not _is_digest(media.get("frame_manifest_digest")):
        blockers.append(f"{prefix}_frame_manifest_digest_invalid")
    if media.get("policy_self_graded") is not False:
        blockers.append(f"{prefix}_policy_self_grading_not_rejected")
    return blockers


def _control_blockers(
    receipt: Mapping[str, Any],
    *,
    control_id: str,
    protocol_digest: str,
    materialization_digest: str,
) -> list[str]:
    prefix = f"arena_control_{control_id}"
    blockers: list[str] = []
    if receipt.get("schema_version") != CONTROL_SCHEMA_VERSION:
        blockers.append(f"{prefix}_schema_invalid")
    if not _digest_matches(receipt, field="control_receipt_digest"):
        blockers.append(f"{prefix}_receipt_digest_invalid")
    if receipt.get("status") != "completed":
        blockers.append(f"{prefix}_not_completed")
    if receipt.get("control_id") != control_id:
        blockers.append(f"{prefix}_identity_mismatch")
    if receipt.get("protocol_digest") != protocol_digest:
        blockers.append(f"{prefix}_protocol_digest_mismatch")
    if receipt.get("materialization_digest") != materialization_digest:
        blockers.append(f"{prefix}_materialization_digest_mismatch")
    if receipt.get("candidate_policy_queried") is not False:
        blockers.append(f"{prefix}_candidate_policy_query_not_rejected")
    if control_id == ZERO_CONTROL:
        if receipt.get("task_success") is not False:
            blockers.append(f"{prefix}_unexpected_success")
        blockers.extend(_media_blockers(receipt, prefix=prefix))
    elif control_id == POSITIVE_CONTROL:
        if receipt.get("task_success") is not True:
            blockers.append(f"{prefix}_success_not_proven")
        if not _is_digest(receipt.get("action_fixture_digest")):
            blockers.append(f"{prefix}_action_fixture_digest_invalid")
        blockers.extend(_media_blockers(receipt, prefix=prefix))
    else:
        parity = _mapping(receipt.get("parity"))
        for key in (
            "camera_schema_matches",
            "action_schema_matches",
            "reset_replay_matches",
            "termination_matches",
        ):
            if parity.get(key) is not True:
                blockers.append(f"{prefix}_{key}_not_proven")
    return blockers


def _policy_dry_run_blockers(
    receipt: Mapping[str, Any],
    *,
    candidate_id: str,
    protocol_digest: str,
    materialization_digest: str,
    checkpoint_inventory_digest: str,
) -> list[str]:
    role = "baseline" if candidate_id == BASELINE_ID else "alternative"
    prefix = f"arena_policy_dry_run_{role}"
    blockers: list[str] = []
    if receipt.get("schema_version") != POLICY_DRY_RUN_SCHEMA_VERSION:
        blockers.append(f"{prefix}_schema_invalid")
    if not _digest_matches(receipt, field="policy_dry_run_receipt_digest"):
        blockers.append(f"{prefix}_receipt_digest_invalid")
    if receipt.get("status") != "completed":
        blockers.append(f"{prefix}_not_completed")
    if receipt.get("candidate_id") != candidate_id:
        blockers.append(f"{prefix}_candidate_identity_mismatch")
    if receipt.get("protocol_digest") != protocol_digest:
        blockers.append(f"{prefix}_protocol_digest_mismatch")
    if receipt.get("materialization_digest") != materialization_digest:
        blockers.append(f"{prefix}_materialization_digest_mismatch")
    if receipt.get("checkpoint_inventory_digest") != checkpoint_inventory_digest:
        blockers.append(f"{prefix}_checkpoint_inventory_digest_mismatch")
    if receipt.get("candidate_policy_queried") is not True:
        blockers.append(f"{prefix}_candidate_policy_query_not_proven")
    if receipt.get("episode_count") != 1:
        blockers.append(f"{prefix}_episode_count_not_one")
    if receipt.get("outcome_ignored_for_decision") is not True:
        blockers.append(f"{prefix}_outcome_not_ignored")
    if receipt.get("production_schedule_trial_id") is not None:
        blockers.append(f"{prefix}_production_trial_id_forbidden")
    blockers.extend(_media_blockers(receipt, prefix=prefix))
    return blockers


def build_candidate_execution_gate(
    *,
    founder_execution_admission: Mapping[str, Any],
    materialized_worker_admission: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
    control_receipts: Mapping[str, Mapping[str, Any]],
    policy_dry_run_receipts: Mapping[str, Mapping[str, Any]],
    worker_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit exactly the frozen jobs, but never authorize spend by itself."""

    protocol = build_founder_sim_protocol()
    canonical_request = build_arena_worker_request(protocol)
    blockers: list[str] = []

    approval = founder_execution_admission.get("approval")
    if not isinstance(approval, Mapping):
        blockers.append("arena_gate_founder_execution_admission_missing")
    else:
        try:
            canonical_admission = admit_founder_sim_execution(protocol, approval)
        except FounderSimProtocolError as exc:
            blockers.extend(f"arena_gate_{item}" for item in exc.blockers)
        else:
            if dict(founder_execution_admission) != canonical_admission:
                blockers.append("arena_gate_founder_execution_admission_not_canonical")

    materialization = _mapping(materialization_receipt)
    materialization_digest = str(materialization.get("materialization_digest") or "")
    if not materialization:
        blockers.append("arena_gate_materialization_receipt_missing")
    else:
        if materialization.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
            blockers.append("arena_gate_materialization_schema_invalid")
        if not _digest_matches(materialization, field="materialization_digest"):
            blockers.append("arena_gate_materialization_digest_invalid")
        if materialization.get("status") != "verified_from_local_worker_bytes":
            blockers.append("arena_gate_materialization_receipt_not_verified")
        if materialization.get("protocol_digest") != protocol["protocol_digest"]:
            blockers.append("arena_gate_materialization_protocol_digest_mismatch")
        if materialization.get("candidate_jobs_authorized") is not False:
            blockers.append("arena_gate_materialization_preauthorized_candidate_jobs")
    materialized_admission = _mapping(materialized_worker_admission)
    if not materialized_admission:
        blockers.append("arena_gate_materialized_worker_admission_missing")
    else:
        if materialized_admission.get("schema_version") != MATERIALIZED_ADMISSION_SCHEMA_VERSION:
            blockers.append("arena_gate_materialized_worker_admission_schema_invalid")
        if not _digest_matches(materialized_admission, field="admission_digest"):
            blockers.append("arena_gate_materialized_worker_admission_digest_invalid")
        if materialized_admission.get("status") != "materialized_pending_native_controls":
            blockers.append("arena_gate_materialized_worker_admission_invalid")
        if materialized_admission.get("protocol_digest") != protocol["protocol_digest"]:
            blockers.append("arena_gate_materialized_worker_protocol_digest_mismatch")
        if materialized_admission.get("materialization_digest") != materialization_digest:
            blockers.append("arena_gate_materialized_worker_digest_mismatch")
        if materialized_admission.get("native_control_canaries_authorized") is not True:
            blockers.append("arena_gate_native_controls_not_authorized")
        if materialized_admission.get("candidate_jobs_authorized") is not False:
            blockers.append("arena_gate_materialized_worker_preauthorized_candidate_jobs")

    if set(control_receipts) != set(REQUIRED_CONTROLS):
        blockers.append("arena_gate_control_receipts_not_exact")
    else:
        for control_id in REQUIRED_CONTROLS:
            blockers.extend(
                _control_blockers(
                    control_receipts[control_id],
                    control_id=control_id,
                    protocol_digest=protocol["protocol_digest"],
                    materialization_digest=materialization_digest,
                )
            )

    candidate_bindings = _mapping(materialization.get("candidate_bindings"))
    checkpoint_digests = {
        BASELINE_ID: _mapping(candidate_bindings.get("baseline")).get(
            "checkpoint_inventory_digest"
        ),
        ALTERNATIVE_ID: _mapping(candidate_bindings.get("alternative")).get(
            "checkpoint_inventory_digest"
        ),
    }
    if set(policy_dry_run_receipts) != set(REQUIRED_CANDIDATES):
        blockers.append("arena_gate_policy_dry_run_receipts_not_exact")
    else:
        for candidate_id in REQUIRED_CANDIDATES:
            blockers.extend(
                _policy_dry_run_blockers(
                    policy_dry_run_receipts[candidate_id],
                    candidate_id=candidate_id,
                    protocol_digest=protocol["protocol_digest"],
                    materialization_digest=materialization_digest,
                    checkpoint_inventory_digest=str(checkpoint_digests[candidate_id] or ""),
                )
            )

    if dict(worker_request) != canonical_request:
        blockers.append("arena_gate_worker_request_not_canonical")
    if canonical_request["job_count"] != protocol["schedule"]["total_trial_budget"]:
        blockers.append("arena_gate_schedule_job_count_mismatch")
    if canonical_request["job_count"] != 88:
        blockers.append("arena_gate_schedule_not_eighty_eight_jobs")

    unique_blockers = sorted(set(blockers))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_schedule_admitted" if not unique_blockers else "blocked",
        "protocol_id": protocol["protocol_id"],
        "protocol_digest": protocol["protocol_digest"],
        "schedule_digest": protocol["schedule"]["schedule_digest"],
        "worker_request_digest": canonical_request["worker_request_digest"],
        "materialization_digest": materialization_digest or None,
        "control_ids": list(REQUIRED_CONTROLS),
        "policy_dry_run_candidate_ids": list(REQUIRED_CANDIDATES),
        "authorized_trial_ids": (
            [row["trial_id"] for row in canonical_request["jobs"]]
            if not unique_blockers
            else []
        ),
        "authorized_trial_count": canonical_request["job_count"] if not unique_blockers else 0,
        "candidate_jobs_authorized": not unique_blockers,
        "paid_compute_authorized": False,
        "separate_paid_resource_admission_required": True,
        "production_simulation_started": False,
        "physical_execution_authorized": False,
        "blockers": unique_blockers,
    }
    result["gate_digest"] = canonical_digest(result, digest_field="gate_digest")
    return result


def _read_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return {}
    value = json.loads(source.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--founder-execution-admission")
    parser.add_argument("--materialized-worker-admission")
    parser.add_argument("--materialization-receipt")
    parser.add_argument("--control-receipts")
    parser.add_argument("--policy-dry-run-receipts")
    parser.add_argument("--worker-request")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_candidate_execution_gate(
        founder_execution_admission=_read_json(args.founder_execution_admission),
        materialized_worker_admission=_read_json(args.materialized_worker_admission),
        materialization_receipt=_read_json(args.materialization_receipt),
        control_receipts=_read_json(args.control_receipts),
        policy_dry_run_receipts=_read_json(args.policy_dry_run_receipts),
        worker_request=(
            _read_json(args.worker_request)
            if args.worker_request
            else build_arena_worker_request()
        ),
    )
    write_json(Path(args.output).expanduser().resolve(), result)
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "candidate_schedule_admitted" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__: Sequence[str] = (
    "CONTROL_SCHEMA_VERSION",
    "PARITY_CONTROL",
    "POLICY_DRY_RUN_SCHEMA_VERSION",
    "POSITIVE_CONTROL",
    "REQUIRED_CANDIDATES",
    "REQUIRED_CONTROLS",
    "SCHEMA_VERSION",
    "ZERO_CONTROL",
    "build_candidate_execution_gate",
    "main",
)
