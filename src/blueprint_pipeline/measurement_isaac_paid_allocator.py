"""Narrow canonical-allocator dispatch for the paid Isaac measurement canary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .gpu_render_providers import get_render_provider
from .measurement_isaac_vast_canary import run_measurement_isaac_vast_canary
from .paid_resource_admission import PaidResourceAdmissionGrant


def add_measurement_isaac_allocator_arguments(parser: Any) -> None:
    parser.add_argument("--measurement-isaac-runtime-release")
    parser.add_argument("--measurement-isaac-bundle-receipt")


def run_measurement_isaac_from_canonical_allocator(
    *,
    args: Any,
    bundle_receipt: Mapping[str, Any],
    resolved_urls: Mapping[str, str],
    adapter_path: Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant,
    load_json: Any,
) -> dict[str, Any]:
    return run_measurement_isaac_vast_canary(
        bound_request=load_json(args.bound_request_out),
        bundle_receipt=bundle_receipt,
        preflight=load_json(args.preflight_bundle),
        job_dir=adapter_path.parent / "measurement_isaac_vast_canary",
        input_bundle_get_url=resolved_urls["provider_bundle_url"],
        output_put_url=resolved_urls["provider_output_put_url"],
        output_get_url=resolved_urls["provider_output_get_url"],
        provider=get_render_provider(args.provider),
        paid_resource_admission_grant=paid_resource_admission_grant,
    )


__all__ = [
    "add_measurement_isaac_allocator_arguments",
    "run_measurement_isaac_from_canonical_allocator",
]
