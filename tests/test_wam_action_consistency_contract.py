from __future__ import annotations

import copy
import hashlib
import json

import pytest

from blueprint_pipeline.wam_action_consistency_contract import (
    cross_step_action_motion_replay_blockers,
    strict_action_consistency_blockers,
)


def _contract() -> dict:
    return {
        "commanded_action_sha256": "a" * 64,
        "commanded_action_vector": [0.1, -0.2],
        "action_dimension": 2,
        "action_unit": "per_dimension",
        "action_units": ["latent", "latent"],
        "action_timing": {
            "step_index": 1,
            "sim_time_s": 0.02,
            "control_hz": 50.0,
            "sample_period_seconds": 0.02,
            "unit": "s",
        },
        "controller_fk_state_sha256": "b" * 64,
        "generated_state_sha256": "c" * 64,
        "generated_motion_sha256": "d" * 64,
    }


def _check() -> dict:
    recovered = [0.1, -0.2]
    contract = _contract()
    return {
        "commanded_action_sha256": contract["commanded_action_sha256"],
        "recovered_action": recovered,
        "recovered_action_sha256": hashlib.sha256(
            json.dumps(recovered, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "per_dimension_error": [0.0, 0.0],
        "per_dimension_uncertainty": [0.01, 0.01],
        "threshold": {"max_abs_error": 0.05, "unit": "per_dimension"},
        "calibration_identity": {"calibration_id": "cal-1", "sha256": "e" * 64},
        "action_timing": contract["action_timing"],
        "action_units": contract["action_units"],
        "controller_fk_state_sha256": contract["controller_fk_state_sha256"],
        "generated_state_sha256": contract["generated_state_sha256"],
        "generated_motion_sha256": contract["generated_motion_sha256"],
        "scorer_runtime_id": "strict-scorer-runtime-1",
        "provider_output_replay_used": False,
        "forward_consistent": True,
        "inverse_consistent": True,
        "forward_result": {"passed": True, "method": "calibrated-forward-model"},
        "inverse_result": {"passed": True, "method": "calibrated-inverse-model"},
        "evidence_refs": ["generated_motion.mp4", "controller_state.json"],
        "termination_chunk": {
            "step_index": 1,
            "commanded_action_sha256": contract["commanded_action_sha256"],
            "generated_motion_sha256": contract["generated_motion_sha256"],
        },
    }


def test_complete_strict_action_consistency_contract_passes() -> None:
    assert strict_action_consistency_blockers(_check(), _contract()) == []


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        ("boolean_only", "wam_consistency_recovered_action_missing_wrong_dim_or_nonfinite"),
        ("command_checksum", "wam_consistency_commanded_action_sha256_mismatch"),
        ("recovered_checksum", "wam_consistency_recovered_action_sha256_mismatch"),
        ("error", "wam_consistency_per_dimension_error_mismatch"),
        ("uncertainty", "wam_consistency_uncertainty_missing_wrong_dim_or_nonfinite"),
        ("threshold", "wam_consistency_numeric_threshold_exceeded"),
        ("calibration", "wam_consistency_calibration_identity_missing_or_invalid"),
        ("timing", "wam_consistency_action_timing_missing_or_invalid"),
        ("units", "wam_consistency_action_units_missing_or_mismatch"),
        ("controller_state", "wam_consistency_controller_fk_state_sha256_missing_or_mismatch"),
        ("generated_state", "wam_consistency_generated_state_sha256_missing_or_mismatch"),
        ("generated_motion", "wam_consistency_generated_motion_sha256_missing_or_mismatch"),
        ("runtime", "wam_consistency_scorer_runtime_id_missing"),
        ("replay", "wam_consistency_replay_or_replay_status_missing"),
        ("forward", "wam_consistency_forward_result_missing_or_invalid"),
        ("inverse", "wam_consistency_inverse_result_missing_or_invalid"),
        ("termination", "wam_consistency_termination_chunk_missing_or_invalid"),
        ("evidence", "wam_consistency_evidence_refs_missing"),
    ],
)
def test_strict_contract_fails_closed_for_each_required_field(
    mutation: str, blocker: str
) -> None:
    check = copy.deepcopy(_check())
    if mutation == "boolean_only":
        check = {"forward_consistent": True, "inverse_consistent": True}
    elif mutation == "command_checksum":
        check["commanded_action_sha256"] = "f" * 64
    elif mutation == "recovered_checksum":
        check["recovered_action_sha256"] = "f" * 64
    elif mutation == "error":
        check["per_dimension_error"] = [0.2, 0.0]
    elif mutation == "uncertainty":
        check["per_dimension_uncertainty"] = [0.01, float("nan")]
    elif mutation == "threshold":
        check["recovered_action"] = [0.2, -0.2]
        check["recovered_action_sha256"] = hashlib.sha256(
            json.dumps(check["recovered_action"], separators=(",", ":")).encode()
        ).hexdigest()
        check["per_dimension_error"] = [0.1, 0.0]
    elif mutation == "calibration":
        check["calibration_identity"] = {}
    elif mutation == "timing":
        check["action_timing"]["control_hz"] = 49.0
    elif mutation == "units":
        check["action_units"] = ["rad", "rad"]
    elif mutation == "controller_state":
        check["controller_fk_state_sha256"] = "f" * 64
    elif mutation == "generated_state":
        check["generated_state_sha256"] = "f" * 64
    elif mutation == "generated_motion":
        check["generated_motion_sha256"] = "f" * 64
    elif mutation == "runtime":
        check["scorer_runtime_id"] = ""
    elif mutation == "replay":
        check["provider_output_replay_used"] = True
    elif mutation == "forward":
        check["forward_result"] = {"passed": False, "method": "forward"}
    elif mutation == "inverse":
        check["inverse_result"] = {}
    elif mutation == "termination":
        check["termination_chunk"]["generated_motion_sha256"] = "f" * 64
    elif mutation == "evidence":
        check["evidence_refs"] = []
    assert blocker in strict_action_consistency_blockers(check, _contract())


def test_different_actions_cannot_reuse_motion_or_recovered_action() -> None:
    first = {
        "commanded_action_sha256": "a" * 64,
        "generated_motion_sha256": "c" * 64,
        "recovered_action_sha256": "d" * 64,
    }
    second = {
        **first,
        "commanded_action_sha256": "b" * 64,
    }
    blockers = cross_step_action_motion_replay_blockers([first, second])
    assert "wam_consistency_generated_motion_reused_for_different_action:step_1" in blockers
    assert "wam_consistency_recovered_action_reused_for_different_command:step_1" in blockers
