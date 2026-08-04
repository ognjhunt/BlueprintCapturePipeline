from __future__ import annotations

import copy
import json
from pathlib import Path

from blueprint_pipeline.adp_arena_candidate_execution_gate import (
    CONTROL_SCHEMA_VERSION,
    PARITY_CONTROL,
    POLICY_DRY_RUN_SCHEMA_VERSION,
    POSITIVE_CONTROL,
    ZERO_CONTROL,
    build_candidate_execution_gate,
    main,
)
from blueprint_pipeline.adp_founder_sim_protocol import (
    ALTERNATIVE_ID,
    APPROVAL_SCHEMA_VERSION,
    BASELINE_ID,
    PROTOCOL_ID,
    admit_founder_sim_execution,
    build_founder_sim_protocol,
)
from blueprint_pipeline.adp_isaac_lab_arena_request import build_arena_worker_request
from blueprint_pipeline.adp_isaac_lab_arena_materialization import (
    ADMISSION_SCHEMA_VERSION as MATERIALIZED_ADMISSION_SCHEMA_VERSION,
)
from blueprint_pipeline.adp_isaac_lab_arena_materialization import (
    SCHEMA_VERSION as MATERIALIZATION_SCHEMA_VERSION,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _media() -> dict[str, object]:
    return {
        "lossless_policy_input_images": True,
        "terminal_image": True,
        "frame_manifest": True,
        "frame_manifest_digest": "sha256:" + "a" * 64,
        "review_video": True,
        "independent_grader_provenance": True,
        "policy_self_graded": False,
    }


def _inputs() -> dict[str, object]:
    protocol = build_founder_sim_protocol()
    approval = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "approved": True,
        "approver_role": "blueprint_founder_sim_owner",
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": protocol["protocol_digest"],
    }
    founder_admission = admit_founder_sim_execution(protocol, approval)
    materialization_digest = "sha256:" + "b" * 64
    materialization = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "status": "verified_from_local_worker_bytes",
        "protocol_digest": protocol["protocol_digest"],
        "materialization_digest": materialization_digest,
        "candidate_jobs_authorized": False,
        "candidate_bindings": {
            "baseline": {
                "candidate_id": BASELINE_ID,
                "checkpoint_inventory_digest": "sha256:" + "c" * 64,
            },
            "alternative": {
                "candidate_id": ALTERNATIVE_ID,
                "checkpoint_inventory_digest": "sha256:" + "d" * 64,
            },
        },
    }
    materialization["materialization_digest"] = canonical_digest(
        materialization, digest_field="materialization_digest"
    )
    materialization_digest = materialization["materialization_digest"]
    materialized_admission = {
        "schema_version": MATERIALIZED_ADMISSION_SCHEMA_VERSION,
        "status": "materialized_pending_native_controls",
        "protocol_digest": protocol["protocol_digest"],
        "materialization_digest": materialization_digest,
        "native_control_canaries_authorized": True,
        "candidate_jobs_authorized": False,
    }
    materialized_admission["admission_digest"] = canonical_digest(
        materialized_admission, digest_field="admission_digest"
    )
    base_control = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "status": "completed",
        "protocol_digest": protocol["protocol_digest"],
        "materialization_digest": materialization_digest,
        "candidate_policy_queried": False,
    }
    controls = {
        ZERO_CONTROL: {
            **base_control,
            "control_id": ZERO_CONTROL,
            "task_success": False,
            "visual_evidence": _media(),
        },
        POSITIVE_CONTROL: {
            **base_control,
            "control_id": POSITIVE_CONTROL,
            "task_success": True,
            "action_fixture_digest": "sha256:" + "e" * 64,
            "visual_evidence": _media(),
        },
        PARITY_CONTROL: {
            **base_control,
            "control_id": PARITY_CONTROL,
            "parity": {
                "camera_schema_matches": True,
                "action_schema_matches": True,
                "reset_replay_matches": True,
                "termination_matches": True,
            },
        },
    }
    for receipt in controls.values():
        receipt["control_receipt_digest"] = canonical_digest(
            receipt, digest_field="control_receipt_digest"
        )
    dry_runs = {}
    for candidate_id, checkpoint_digest in (
        (BASELINE_ID, "sha256:" + "c" * 64),
        (ALTERNATIVE_ID, "sha256:" + "d" * 64),
    ):
        dry_runs[candidate_id] = {
            "schema_version": POLICY_DRY_RUN_SCHEMA_VERSION,
            "status": "completed",
            "protocol_digest": protocol["protocol_digest"],
            "materialization_digest": materialization_digest,
            "candidate_id": candidate_id,
            "checkpoint_inventory_digest": checkpoint_digest,
            "candidate_policy_queried": True,
            "episode_count": 1,
            "outcome_ignored_for_decision": True,
            "production_schedule_trial_id": None,
            "visual_evidence": _media(),
        }
        dry_runs[candidate_id]["policy_dry_run_receipt_digest"] = canonical_digest(
            dry_runs[candidate_id], digest_field="policy_dry_run_receipt_digest"
        )
    return {
        "founder_execution_admission": founder_admission,
        "materialized_worker_admission": materialized_admission,
        "materialization_receipt": materialization,
        "control_receipts": controls,
        "policy_dry_run_receipts": dry_runs,
        "worker_request": build_arena_worker_request(protocol),
    }


def test_gate_releases_exact_frozen_schedule_only_after_all_precursors() -> None:
    result = build_candidate_execution_gate(**_inputs())

    assert result["status"] == "candidate_schedule_admitted"
    assert result["candidate_jobs_authorized"] is True
    assert result["authorized_trial_count"] == 88
    assert len(set(result["authorized_trial_ids"])) == 88
    assert result["paid_compute_authorized"] is False
    assert result["separate_paid_resource_admission_required"] is True
    assert result["production_simulation_started"] is False
    assert result["physical_execution_authorized"] is False


def test_gate_rejects_missing_control_and_dry_run_media() -> None:
    inputs = _inputs()
    inputs["control_receipts"].pop(POSITIVE_CONTROL)
    inputs["policy_dry_run_receipts"][BASELINE_ID]["visual_evidence"][
        "lossless_policy_input_images"
    ] = False

    result = build_candidate_execution_gate(**inputs)

    assert result["status"] == "blocked"
    assert result["authorized_trial_count"] == 0
    assert "arena_gate_control_receipts_not_exact" in result["blockers"]
    assert (
        "arena_policy_dry_run_baseline_lossless_policy_input_images_missing"
        in result["blockers"]
    )


def test_gate_rejects_policy_self_grading_checkpoint_drift_and_outcome_reuse() -> None:
    inputs = _inputs()
    receipt = inputs["policy_dry_run_receipts"][ALTERNATIVE_ID]
    receipt["checkpoint_inventory_digest"] = "sha256:" + "0" * 64
    receipt["outcome_ignored_for_decision"] = False
    receipt["visual_evidence"]["policy_self_graded"] = True

    result = build_candidate_execution_gate(**inputs)

    assert result["candidate_jobs_authorized"] is False
    assert "arena_policy_dry_run_alternative_checkpoint_inventory_digest_mismatch" in result[
        "blockers"
    ]
    assert "arena_policy_dry_run_alternative_outcome_not_ignored" in result["blockers"]
    assert "arena_policy_dry_run_alternative_policy_self_grading_not_rejected" in result[
        "blockers"
    ]


def test_gate_rejects_any_worker_request_change() -> None:
    inputs = _inputs()
    inputs["worker_request"] = copy.deepcopy(inputs["worker_request"])
    inputs["worker_request"]["jobs"][0]["rollout"]["seed"] += 1

    result = build_candidate_execution_gate(**inputs)

    assert result["status"] == "blocked"
    assert result["authorized_trial_ids"] == []
    assert "arena_gate_worker_request_not_canonical" in result["blockers"]


def test_cli_writes_typed_blocker_artifact_when_precursors_are_absent(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "gate.json"

    assert main(["--output", str(output)]) == 2

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert result["candidate_jobs_authorized"] is False
    assert result["authorized_trial_ids"] == []
    assert "arena_gate_founder_execution_admission_missing" in result["blockers"]
    assert "arena_gate_control_receipts_not_exact" in result["blockers"]
    assert "arena_gate_policy_dry_run_receipts_not_exact" in result["blockers"]
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
