from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.wam_eval_substrate import (
    build_evaluation_substrate_registry,
    normalize_evaluation_substrate,
    requested_evaluation_substrate,
    write_evaluation_substrate_registry,
)


def test_evaluation_substrate_registry_preserves_wam_and_sim_aliases(tmp_path: Path) -> None:
    registry = build_evaluation_substrate_registry(generated_at="2026-06-20T00:00:00+00:00")

    assert registry["schema_version"] == "evaluation_substrate_registry.v1"
    assert registry["default_primary_substrate"] == "fixture_wam"
    assert set(registry["supported_substrates"]) == {
        "fixture_wam",
        "cosmos3_wam",
        "oscar_wam",
        "classical_sim_mujoco",
        "classical_sim_isaac",
        "recorded_trace",
    }
    assert registry["entries"]["fixture_wam"]["local_available"] is True
    assert registry["entries"]["cosmos3_wam"]["live_provider_required"] is True
    assert registry["entries"]["classical_sim_mujoco"]["family"] == "classical_simulation"
    assert registry["contract"]["generated_rollouts_are_model_derived_support_artifacts"] is True
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
