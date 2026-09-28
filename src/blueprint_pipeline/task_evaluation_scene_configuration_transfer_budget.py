"""Canonical byte ceilings for scene-configuration provider transport."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def scene_configuration_provider_transfer_byte_budget(
    receipt: Mapping[str, Any],
    *,
    provisioning_download_overhead_bytes: int,
    artifixer_pinned_wheel_download_floor_bytes: int,
    provider_output_upload_minimum_bytes: int,
    provider_output_upload_bundle_multiplier: int,
    error_factory: Callable[[str], Exception],
) -> tuple[int, int]:
    """Price the exact bundle plus fail-closed download/upload ceilings."""

    bundle = receipt.get("bundle_size_bytes")
    if not isinstance(bundle, int) or isinstance(bundle, bool) or bundle <= 0:
        raise error_factory("scene_configuration_provider_transfer_budget_inputs_invalid")
    if provisioning_download_overhead_bytes < (
        4 * artifixer_pinned_wheel_download_floor_bytes
    ):
        raise error_factory("scene_configuration_provider_transfer_budget_underdeclared")
    download = bundle + provisioning_download_overhead_bytes
    if (
        provider_output_upload_minimum_bytes < 1_000_000_000
        or provider_output_upload_bundle_multiplier < 2
    ):
        raise error_factory("scene_configuration_provider_transfer_budget_underdeclared")
    return download, max(
        provider_output_upload_bundle_multiplier * bundle,
        provider_output_upload_minimum_bytes,
    )


__all__ = ["scene_configuration_provider_transfer_byte_budget"]
