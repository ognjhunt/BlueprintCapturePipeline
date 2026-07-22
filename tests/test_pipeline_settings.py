from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.capture_orchestrator import PipelineConfig
from blueprint_pipeline.core.pipeline_settings import PipelineSettings
from blueprint_pipeline.pipeline_settings import PipelineSettings as CompatibilitySettings


def test_pipeline_settings_compatibility_import_is_canonical_type() -> None:
    assert CompatibilitySettings is PipelineSettings


def test_pipeline_settings_parse_once_with_typed_values() -> None:
    settings = PipelineSettings.from_env(
        {
            "BLUEPRINT_ENV": "production",
            "GCS_ROOT": "/srv/blueprint-gcs",
            "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION": "true",
            "BLUEPRINT_ALLOW_GPU_PROVISIONING": "false",
            "BLUEPRINT_ALLOW_COSMOS_TRAINING": "on",
            "BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS": "0",
            "BLUEPRINT_ALLOW_AGENTS_SDK_JOB_ORCHESTRATION": "yes",
            "BLUEPRINT_SIM_ONLY_BETA_AUTONOMY": "1",
            "BLUEPRINT_SIM_ONLY_BETA_DEFAULT_TASK_EVAL": "yes",
        }
    )
    assert settings.environment == "production"
    assert settings.gcs_root == Path("/srv/blueprint-gcs")
    assert settings.allow_simulator_execution is True
    assert settings.allow_gpu_provisioning is False
    assert settings.allow_cosmos_training is True
    assert settings.allow_live_agents_sdk_operators is False
    assert settings.allow_legacy_agents_sdk_job_orchestration is True
    assert settings.sim_only_beta_autonomy is True
    assert settings.sim_only_beta_default_task_eval is True


def test_pipeline_settings_reject_ambiguous_truthy_strings() -> None:
    with pytest.raises(
        ValueError,
        match="invalid_boolean_environment_value:BLUEPRINT_ALLOW_GPU_PROVISIONING",
    ):
        PipelineSettings.from_env({"BLUEPRINT_ALLOW_GPU_PROVISIONING": "enabled"})


def test_pipeline_settings_require_environment_for_mutating_cli_gates() -> None:
    settings = PipelineSettings.from_env({})
    with pytest.raises(
        ValueError,
        match=(
            "cli_admission_missing_environment_approval:"
            "BLUEPRINT_ALLOW_GPU_PROVISIONING,"
            "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION,"
            "BLUEPRINT_ALLOW_COSMOS_TRAINING,"
            "BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS"
        ),
    ):
        settings.validate_cli_admission(
            allow_gpu_provisioning=True,
            allow_simulator_execution=True,
            allow_cosmos_training=True,
            allow_live_agents_sdk_operator=True,
        )

    admitted = PipelineSettings.from_env(
        {
            "BLUEPRINT_ALLOW_GPU_PROVISIONING": "true",
            "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION": "true",
            "BLUEPRINT_ALLOW_COSMOS_TRAINING": "true",
            "BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS": "true",
        }
    )
    admitted.validate_cli_admission(
        allow_gpu_provisioning=True,
        allow_simulator_execution=True,
        allow_cosmos_training=True,
        allow_live_agents_sdk_operator=True,
    )


def test_pipeline_config_reads_gcs_root_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCS_ROOT", "/first")
    first = PipelineConfig()
    monkeypatch.setenv("GCS_ROOT", "/second")
    second = PipelineConfig()
    assert first.gcs_root == Path("/first")
    assert second.gcs_root == Path("/second")
