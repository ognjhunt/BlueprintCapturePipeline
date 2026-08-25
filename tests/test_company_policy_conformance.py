from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from blueprint_pipeline.company_policy_conformance import (
    BLOCKER_CONFORMANCE_FAILED,
    CompanyPolicyConformanceError,
    build_conformance_probe,
    evaluate_conformance_response,
)
from blueprint_pipeline.company_policy_container_contract import (
    SCHEMA_VERSION,
    validate_company_policy_container_contract,
)


def _contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_id": "acme_widget_grasp_v3",
        "company_id": "acme_robotics",
        "display_name": "ACME Widget Grasp v3",
        "checkpoint_identity": {
            "repository": "https://models.acme.example/widget-grasp",
            "revision": "2026.08.1",
        },
        "claim_ceiling": "development_only",
        "rights": {
            "license": "ACME Evaluation License 2026-08",
            "rights_provenance": "acme_msa_2026_07_appendix_b",
            "provider_use_status": "permitted_on_rented_gpu_for_this_evaluation",
            "redistribution_status": "no_redistribution_weights_stay_in_container",
            "rights_ready": True,
        },
        "container": {
            "image": "registry.acme.example/widget-grasp@sha256:" + "b" * 64,
            "serve_command": ["python", "-m", "acme_policy.serve"],
            "port": 8600,
            "handshake_kind": "http_json_v1",
            "credential_files": [],
            "gpu_required": True,
        },
        "observation_schema": {
            "cameras": [
                {"name": "exterior_image_1_left", "width": 320, "height": 180},
                {"name": "wrist_image_left", "width": 224, "height": 224},
            ],
            "state_keys": ["joint_position", "gripper_position"],
        },
        "action_schema": {
            "action_space_id": "acme_joint_velocity_v1",
            "chunk_rows": 15,
            "channels": [
                *(
                    {
                        "name": f"joint_velocity_{index}",
                        "kind": "bounded_continuous",
                        "command_interval": [-1.0, 1.0],
                        "raw_accepted_bounds": [-1.0, 1.0],
                        "executed_semantics": (
                            "normalized velocity mapped to bounded position delta"
                        ),
                    }
                    for index in range(7)
                ),
                {
                    "name": "gripper",
                    "kind": "threshold_scalar",
                    "command_interval": [0.0, 1.0],
                    "raw_accepted_bounds": [-0.25, 1.25],
                    "executed_semantics": (
                        "clip_to_command_interval_then_threshold_at_0.5"
                    ),
                },
            ],
        },
    }


def _conformant_chunk(rows: int = 15, width: int = 8) -> np.ndarray:
    return np.zeros((rows, width), dtype=float)


def test_probe_matches_the_declared_observation_schema() -> None:
    """Zero images sized exactly per camera; zero state; no network."""

    probe = build_conformance_probe(_contract())

    images = probe["observation"]["images"]
    assert set(images) == {"exterior_image_1_left", "wrist_image_left"}
    # (height, width, channels): declared 320x180 means an array of 180 rows.
    assert images["exterior_image_1_left"].shape == (180, 320, 3)
    assert images["wrist_image_left"].shape == (224, 224, 3)
    for image in images.values():
        assert image.dtype == np.uint8
        assert int(image.sum()) == 0
    assert probe["observation"]["state"] == {
        "joint_position": 0.0,
        "gripper_position": 0.0,
    }
    assert probe["endpoint"] == {"host": "127.0.0.1", "port": 8600}
    assert probe["expected_response"] == {"chunk_rows": 15, "chunk_width": 8}
    assert probe["network_performed"] is False
    # Bound to the sealed contract identity, not merely to a policy id.
    normalized = validate_company_policy_container_contract(_contract())
    assert probe["contract_digest"] == normalized["contract_digest"]


def test_a_conformant_chunk_passes_and_reports_per_channel_envelopes() -> None:
    chunk = _conformant_chunk()
    # A realistic overshoot inside the declared raw envelope: reported, never
    # refused (the pi05 gripper lesson, generalized to declared data).
    chunk[:, 7] = 1.02

    receipt = evaluate_conformance_response(_contract(), chunk)

    assert receipt["status"] == "conformant"
    assert receipt["chunk_rows"] == 15
    assert receipt["chunk_width"] == 8
    applied = receipt["bounds_receipt"]["channel_contracts_applied"]
    assert [channel["name"] for channel in applied][-1] == "gripper"
    gripper = applied[-1]
    assert gripper["rows_outside_command_interval"] == 15
    assert gripper["max_command_interval_overshoot"] == pytest.approx(0.02)
    assert applied[0]["rows_outside_command_interval"] == 0
    assert receipt["bounds_receipt"]["raw_candidate_clipping_permitted"] is False
    assert receipt["network_performed"] is False


@pytest.mark.parametrize(
    ("chunk", "detail"),
    [
        (np.zeros((15, 7)), "chunk_width"),
        (np.zeros((15, 9)), "chunk_width"),
        (np.zeros((10, 8)), "chunk_rows"),
        (np.zeros((16, 8)), "chunk_rows"),
        (np.zeros(8), "chunk_not_2d"),
    ],
)
def test_wrong_shapes_refuse_with_the_conformance_prefix(
    chunk: np.ndarray, detail: str
) -> None:
    with pytest.raises(CompanyPolicyConformanceError) as excinfo:
        evaluate_conformance_response(_contract(), chunk)
    assert any(
        error.startswith(f"{BLOCKER_CONFORMANCE_FAILED}:{detail}")
        for error in excinfo.value.errors
    ), excinfo.value.errors


def test_an_out_of_envelope_channel_refuses_and_names_the_channel() -> None:
    chunk = _conformant_chunk()
    chunk[3, 7] = 2.0  # beyond the gripper's declared raw envelope of 1.25

    with pytest.raises(CompanyPolicyConformanceError) as excinfo:
        evaluate_conformance_response(_contract(), chunk)
    assert any(
        error.startswith(
            f"{BLOCKER_CONFORMANCE_FAILED}:"
            "candidate_action_channel_bounds_invalid:gripper"
        )
        for error in excinfo.value.errors
    ), excinfo.value.errors


def test_nonfinite_responses_refuse() -> None:
    chunk = _conformant_chunk()
    chunk[0, 0] = np.nan
    with pytest.raises(CompanyPolicyConformanceError) as excinfo:
        evaluate_conformance_response(_contract(), chunk)
    assert all(
        error.startswith(BLOCKER_CONFORMANCE_FAILED)
        for error in excinfo.value.errors
    )


def test_conformance_never_runs_on_an_inadmissible_contract() -> None:
    """Both halves re-validate: an unadmitted mapping cannot be probed."""

    from blueprint_pipeline.company_policy_container_contract import (
        CompanyPolicyContractError,
    )

    broken = _contract()
    broken["container"]["image"] = "registry.acme.example/widget-grasp:latest"
    with pytest.raises(CompanyPolicyContractError):
        build_conformance_probe(broken)
    with pytest.raises(CompanyPolicyContractError):
        evaluate_conformance_response(broken, _conformant_chunk())
