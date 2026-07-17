from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.wam_eval_substrate import (
    build_wam_eval_claim_boundary,
    build_wam_evaluation_request,
    build_evaluation_substrate_registry,
    is_classical_sim_evaluation_substrate,
    is_wam_evaluation_substrate,
    legacy_simulator_substrate,
    normalize_evaluation_substrate,
    requested_evaluation_substrate,
    write_evaluation_substrate_registry,
)


def test_evaluation_substrate_registry_preserves_wam_and_sim_aliases(tmp_path: Path) -> None:
    registry = build_evaluation_substrate_registry(generated_at="2026-06-20T00:00:00+00:00")

    assert registry["schema_version"] == "evaluation_substrate_registry.v1"
    assert registry["default_primary_substrate"] == "classical_sim_mujoco"
    assert registry["default_primary_substrate_is_learned_model"] is False
    assert registry["preferred_configured_learned_wam_substrate"] == "cosmos3_wam"
    assert registry["fixture_wam_is_not_default_learned_backend"] is True
    assert set(registry["supported_substrates"]) == {
        "fixture_wam",
        "cosmos3_wam",
        "oscar_wam",
        "classical_sim_mujoco",
        "classical_sim_isaac",
        "recorded_trace",
    }
    assert registry["entries"]["fixture_wam"]["local_available"] is True
    assert registry["entries"]["fixture_wam"]["learned_model_backend"] is False
    assert registry["entries"]["fixture_wam"]["deterministic_fixture"] is True
    assert registry["entries"]["fixture_wam"]["proof_ceiling"] == (
        "deterministic_fixture_not_learned_model_backend"
    )
    assert registry["entries"]["cosmos3_wam"]["live_provider_required"] is True
    assert registry["entries"]["cosmos3_wam"]["learned_model_backend"] is True
    assert registry["entries"]["cosmos3_wam"]["backbone"] == "Cosmos3-Nano"
    assert registry["entries"]["cosmos3_wam"]["model_id"] == "nvidia/Cosmos3-Nano"
    assert registry["entries"]["cosmos3_wam"]["adapter_id"] == "deepinfra_cosmos3_nano_api"
    # R115: DeepInfra is the DEFAULT provider; the cosmos3 backbone stays swappable.
    cosmos3 = registry["entries"]["cosmos3_wam"]
    assert cosmos3["provider_backend_swappable"] is True
    assert cosmos3["default_provider"] == "deepinfra"
    assert set(cosmos3["supported_providers"]) == {"deepinfra", "runpod", "vast"}
    assert "<deepinfra|runpod|vast>" in cosmos3["command_surface"]
    assert registry["entries"]["classical_sim_mujoco"]["family"] == "classical_simulation"
    assert registry["contract"]["generated_rollouts_are_model_derived_support_artifacts"] is True
    assert registry["contract"]["deterministic_fixture_is_not_learned_model_backend"] is True
    assert registry["contract"]["learned_wam_requires_provider_execution_manifest"] is True
    assert (
        registry["contract"]["customer_specific_srcc_requires_real_world_validation_rollouts"]
        is True
    )

    path_payload = write_evaluation_substrate_registry(
        tmp_path,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert (tmp_path / "evaluation_substrate_registry.json").is_file()
    assert path_payload == registry


def test_requested_evaluation_substrate_accepts_legacy_and_wam_fields() -> None:
    assert normalize_evaluation_substrate("mujoco") == "classical_sim_mujoco"
    assert normalize_evaluation_substrate("isaac_sim") == "classical_sim_isaac"
    assert normalize_evaluation_substrate("cosmos_3") == "cosmos3_wam"
    assert normalize_evaluation_substrate("", default="fixture_wam") == "fixture_wam"
    assert legacy_simulator_substrate("mujoco") == "classical_sim_mujoco"
    assert is_wam_evaluation_substrate("fixture") is True
    assert is_wam_evaluation_substrate("mujoco") is False
    assert is_classical_sim_evaluation_substrate("isaac") is True
    assert is_classical_sim_evaluation_substrate("fixture") is False

    request = {
        "execution_request": {
            "wam_evaluation": {
                "substrate": "oscar",
            }
        }
    }
    assert requested_evaluation_substrate(request) == "oscar_wam"
    assert requested_evaluation_substrate(request, explicit="fixture") == "fixture_wam"

    with pytest.raises(ValueError):
        normalize_evaluation_substrate("hardwired_private_model")


def test_wam_request_and_claim_boundary_normalize_policy_ids() -> None:
    string_policy = build_wam_evaluation_request(
        job_id="job-1",
        substrate="fixture_wam",
        policy_ids="policy-from-string",
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert string_policy["policy_ids"] == ["policy-from-string"]

    non_sequence_policy = build_wam_evaluation_request(
        job_id="job-1",
        substrate="fixture_wam",
        policy_ids=object(),
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert non_sequence_policy["policy_ids"] == []

    boundary = build_wam_eval_claim_boundary(
        substrate="cosmos3_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert boundary["evaluation_substrate"] == "cosmos3_wam"
    assert boundary["fixture_wam_is_deterministic_local_test_substrate"] is False
    assert boundary["fixture_wam_is_not_learned_model_backend"] is False
    assert boundary["learned_model_backend_executed"] is False
    assert boundary["learned_model_backend_requires_provider_execution_manifest"] is True
    assert boundary["generated_rollouts_are_raw_capture_evidence"] is False


def test_fixture_claim_boundary_is_required_and_blocks_correlation_claims() -> None:
    from blueprint_pipeline.wam_eval_substrate import FixtureClaimBoundaryError

    boundary = build_wam_eval_claim_boundary(
        substrate="fixture_wam",
        generated_at="2026-07-02T00:00:00+00:00",
    )
    assert boundary["fixture_evaluator_only"] is True
    assert boundary["fixture_provenance_required_in_downstream_artifacts"] is True
    assert boundary["correlation_metrics_blocked_for_fixture_runs"] is True
    assert boundary["unlabeled_predicted_success_blocked_for_fixture_runs"] is True
    assert boundary["spearman_pearson_mmrv_status"] == (
        "blocked_fixture_evaluator_only_no_correlation_claims"
    )

    model_boundary = build_wam_eval_claim_boundary(
        substrate="cosmos3_wam",
        generated_at="2026-07-02T00:00:00+00:00",
    )
    assert model_boundary["fixture_evaluator_only"] is False
    assert model_boundary["spearman_pearson_mmrv_status"] == (
        "not_measured_until_real_anchors_exist"
    )

    with pytest.raises(FixtureClaimBoundaryError):
        build_wam_eval_claim_boundary(
            substrate="fixture_wam",
            generated_at="2026-07-02T00:00:00+00:00",
            fixture_evaluator_only=False,
        )

    request = build_wam_evaluation_request(
        job_id="job-1",
        substrate="fixture_wam",
        generated_at="2026-07-02T00:00:00+00:00",
    )
    assert request["fixture_evaluator_only"] is True
    assert request["claim_boundary"]["fixture_evaluator_only"] is True

    model_request = build_wam_evaluation_request(
        job_id="job-1",
        substrate="cosmos3_wam",
        generated_at="2026-07-02T00:00:00+00:00",
    )
    assert model_request["fixture_evaluator_only"] is False


def test_enforce_fixture_claim_boundary_stamps_provenance_and_fails_closed() -> None:
    from blueprint_pipeline.wam_eval_substrate import (
        FixtureClaimBoundaryError,
        enforce_fixture_claim_boundary,
        fixture_claim_boundary_violations,
    )

    stamped = enforce_fixture_claim_boundary(
        {
            "rollouts": [
                {"policy_id": "policy-1", "predicted_success": True},
            ]
        },
        substrate="fixture_wam",
    )
    assert stamped["fixture_evaluator_only"] is True
    assert stamped["rollouts"][0]["fixture_evaluator_only"] is True

    with pytest.raises(FixtureClaimBoundaryError):
        enforce_fixture_claim_boundary(
            {"mmrv": 0.12},
            substrate="fixture_wam",
        )

    with pytest.raises(FixtureClaimBoundaryError):
        enforce_fixture_claim_boundary(
            {"metrics": {"spearman": 0.9}},
            substrate="fixture_wam",
        )

    with pytest.raises(FixtureClaimBoundaryError):
        enforce_fixture_claim_boundary(
            {"rollouts": [{"metrics": {"pearson": 0.8}}]},
            substrate="fixture_wam",
        )

    violations = fixture_claim_boundary_violations(
        {
            "predicted_success": True,
            "rollouts": [{"predicted_success": True}],
        },
        fixture_evaluator_only=True,
    )
    assert "fixture_run_emits_unlabeled_predicted_success:top_level" in violations
    assert "fixture_run_emits_unlabeled_predicted_success:rollouts[0]" in violations
    assert (
        fixture_claim_boundary_violations({"mmrv": 0.5}, fixture_evaluator_only=False)
        == []
    )

    model_payload = enforce_fixture_claim_boundary(
        {"mmrv": None, "rollouts": [{"predicted_success": True}]},
        substrate="cosmos3_wam",
    )
    assert model_payload["fixture_evaluator_only"] is False
    assert model_payload["rollouts"][0]["fixture_evaluator_only"] is False
