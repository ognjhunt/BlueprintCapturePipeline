from __future__ import annotations

from copy import deepcopy

import pytest

from blueprint_pipeline.wam_conditioning_fidelity import (
    ConditioningFidelityThresholds,
    assess_wam_conditioning_fidelity,
    validate_conditioning_fidelity_certificate,
)


def _evidence() -> dict:
    conditions = {
        "recorded": {
            "action_sha256": "1" * 64,
            "vendor_native_action": True,
        },
        "no_motion": {
            "action_sha256": "2" * 64,
            "valid_identity_rotation": True,
            "explicit_gripper_hold": True,
        },
        "shuffled": {
            "action_sha256": "3" * 64,
            "real_action_permutation": True,
        },
    }
    attestations = []
    for seed in range(4):
        for condition, control in conditions.items():
            attestations.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "requested_action_sha256": control["action_sha256"],
                    "parsed_action_sha256": control["action_sha256"],
                    "applied_action_sha256": control["action_sha256"],
                    "parsed_action_shape": [16, 10],
                    "attestation_location": "inside_model_preprocess",
                    "effective_parameters_sha256": "4" * 64,
                    "output_sha256": f"{seed + 5:x}" * 64,
                }
            )
    return {
        "schema_version": "wam_conditioning_fidelity_evidence.v1",
        "backend": {
            "backend_id": "oscar_purpose_built_wam",
            "source_revision": "a" * 64,
            "model_revision": "b" * 64,
        },
        "vendor_reference": {
            "asset_id": "vendor/reference/example-1",
            "asset_sha256": "c" * 64,
            "license": "Vendor-Test-License",
        },
        "action_contract": {
            "shape": [16, 10],
            "effective_parameters_sha256": "4" * 64,
        },
        "controls": conditions,
        "server_action_attestations": attestations,
        "causal_views": [
            {
                "view_id": "primary",
                "seed_comparisons": [
                    {
                        "seed": seed,
                        "cross_seed_noise": 0.1,
                        "active_vs_no_motion_distance": 0.2,
                        "active_vs_shuffled_distance": 0.15,
                    }
                    for seed in range(4)
                ],
            }
        ],
    }


def test_conditioning_certificate_requires_server_attestation_and_paired_causality() -> None:
    certificate = assess_wam_conditioning_fidelity(
        _evidence(), thresholds=ConditioningFidelityThresholds()
    )

    assert certificate["status"] == "passed"
    assert certificate["server_side_action_attestation_passed"] is True
    assert certificate["view_results"][0]["passing_seed_count"] == 4
    assert certificate["blockers"] == []
    assert len(certificate["manifest_sha256"]) == 64
    assert validate_conditioning_fidelity_certificate(
        certificate, backend_id="oscar_purpose_built_wam"
    )["status"] == "passed"


def test_client_request_echo_cannot_substitute_for_inference_path_attestation() -> None:
    evidence = _evidence()
    evidence["server_action_attestations"][0]["attestation_location"] = "client_request_echo"

    certificate = assess_wam_conditioning_fidelity(
        evidence, thresholds=ConditioningFidelityThresholds()
    )

    assert certificate["status"] == "failed"
    assert "conditioning_attestation_not_inside_inference_path" in certificate["blockers"]


def test_motion_without_action_control_separation_fails() -> None:
    evidence = _evidence()
    for row in evidence["causal_views"][0]["seed_comparisons"]:
        row["active_vs_no_motion_distance"] = 0.05
        row["active_vs_shuffled_distance"] = 0.04

    certificate = assess_wam_conditioning_fidelity(
        evidence, thresholds=ConditioningFidelityThresholds()
    )

    assert certificate["status"] == "failed"
    assert "conditioning_causal_view_failed:primary" in certificate["blockers"]


def test_underpowered_two_seed_canary_fails() -> None:
    evidence = _evidence()
    evidence["server_action_attestations"] = [
        row for row in evidence["server_action_attestations"] if row["seed"] < 2
    ]
    evidence["causal_views"][0]["seed_comparisons"] = evidence["causal_views"][0][
        "seed_comparisons"
    ][:2]

    certificate = assess_wam_conditioning_fidelity(
        evidence, thresholds=ConditioningFidelityThresholds()
    )

    assert certificate["status"] == "failed"
    assert "conditioning_seed_count_below_threshold" in certificate["blockers"]


def test_tampered_or_wrong_backend_certificate_is_rejected() -> None:
    certificate = assess_wam_conditioning_fidelity(
        _evidence(), thresholds=ConditioningFidelityThresholds()
    )
    tampered = deepcopy(certificate)
    tampered["seed_count"] = 99

    with pytest.raises(ValueError, match="digest_invalid"):
        validate_conditioning_fidelity_certificate(
            tampered, backend_id="oscar_purpose_built_wam"
        )
    with pytest.raises(ValueError, match="backend_mismatch"):
        validate_conditioning_fidelity_certificate(certificate, backend_id="native_cosmos")
