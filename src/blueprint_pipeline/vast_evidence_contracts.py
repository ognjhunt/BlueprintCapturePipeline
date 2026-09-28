"""Cold, dependency-free identities shared by Vast evidence validators."""

from __future__ import annotations


VAST_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION = "vast_provider_adapter_result.v1"
VAST_TEARDOWN_SCHEMA_VERSION = "vast_teardown_manifest.v1"
SCENE_CONFIGURATION_DIAGNOSTIC_RESULT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_diagnostic_vast_result.v1"
)
VAST_PROVIDER_ZERO_API_CALL = [
    "blueprint_pipeline.gpu_render_providers.VastRenderProvider.billable_inventory",
    "name_prefix=",
]
VAST_PROVIDER_ZERO_LEGACY_COMMAND = ["vastai", "show", "instances", "--raw"]


def valid_vast_provider_zero_api_call(value: object) -> bool:
    """Accept the hardened in-process API seam and retained legacy receipts."""

    return value in (VAST_PROVIDER_ZERO_API_CALL, VAST_PROVIDER_ZERO_LEGACY_COMMAND)


__all__ = [
    "SCENE_CONFIGURATION_DIAGNOSTIC_RESULT_SCHEMA_VERSION",
    "VAST_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION",
    "VAST_PROVIDER_ZERO_API_CALL",
    "VAST_PROVIDER_ZERO_LEGACY_COMMAND",
    "VAST_TEARDOWN_SCHEMA_VERSION",
    "valid_vast_provider_zero_api_call",
]
