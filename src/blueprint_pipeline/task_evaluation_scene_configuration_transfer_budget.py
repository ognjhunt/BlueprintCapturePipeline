"""Canonical byte ceilings for scene-configuration provider transport."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

#: Floor for the provider's output archive once the bundle states its
#: toolchain bytes. The output is the run's evidence: stage results, the
#: configured scene and its capsules, never the toolchain it ran. Completed
#: scene-configuration runs retained 0.55 GB (website, 2026-09-21) and 4.54 GB
#: (full six-stage chain with a 1.93 GB pretraining capsule, 2026-09-15) in
#: total, so this floor is about 1.8x the largest.
SCENE_OUTPUT_ARCHIVE_FLOOR_BYTES = 8_000_000_000


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
    # The multiplier covers what a run can return relative to what it was sent,
    # and the toolchain is never returned. capture-447ae4f4 (2026-11-01): with
    # ~6 GB of the 6.1 GB bundle being toolchain, 2x the whole bundle reserved
    # 34.6 GB of control-plane disk and refused the launch before allocation.
    # Only a bundle that states its toolchain archive size gets this sizing;
    # older receipts keep the whole-bundle ceiling.
    # It only ever lowers the ceiling: a small bundle keeps its smaller one.
    whole = max(provider_output_upload_bundle_multiplier * bundle, provider_output_upload_minimum_bytes)
    archive = receipt.get("toolchain_archive")
    fragment = archive.get("fragment_size_bytes") if isinstance(archive, Mapping) else None
    if type(fragment) is int and 0 < fragment < bundle:
        returnable = max(provider_output_upload_bundle_multiplier * (bundle - fragment),
                         provider_output_upload_minimum_bytes, SCENE_OUTPUT_ARCHIVE_FLOOR_BYTES)
        return download, min(whole, returnable)
    return download, whole


__all__ = ["SCENE_OUTPUT_ARCHIVE_FLOOR_BYTES", "scene_configuration_provider_transfer_byte_budget"]
