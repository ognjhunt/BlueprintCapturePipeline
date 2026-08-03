"""Canonical-allocator dispatch for the paid DLO-Lab CUDA canary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .gpu_render_providers import get_render_provider
from .measurement_dlo_lab_vast_canary import run_measurement_dlo_lab_vast_canary
from .paid_resource_admission import PaidResourceAdmissionGrant


def add_measurement_dlo_lab_allocator_arguments(parser: Any) -> None:
    parser.add_argument("--measurement-dlo-lab-runtime-release")
    parser.add_argument("--measurement-dlo-lab-bundle-receipt")


def add_measurement_chrono_dem_allocator_arguments(parser: Any) -> None:
    parser.add_argument("--measurement-chrono-dem-runtime-release")
    parser.add_argument("--measurement-chrono-dem-bundle-receipt")
    parser.add_argument("--measurement-chrono-dem-object-store-staging-manifest")


def add_measurement_allocator_arguments(parser: Any) -> None:
    from .measurement_isaac_paid_allocator import add_measurement_isaac_allocator_arguments

    add_measurement_isaac_allocator_arguments(parser)
    add_measurement_dlo_lab_allocator_arguments(parser)
    add_measurement_chrono_dem_allocator_arguments(parser)


def run_measurement_dlo_lab_from_canonical_allocator(
    *,
    args: Any,
    bundle_receipt: Mapping[str, Any],
    resolved_urls: Mapping[str, str],
    adapter_path: Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant,
    load_json: Any,
) -> dict[str, Any]:
    return run_measurement_dlo_lab_vast_canary(
        bound_request=load_json(args.bound_request_out),
        bundle_receipt=bundle_receipt,
        preflight=load_json(args.preflight_bundle),
        job_dir=adapter_path.parent / "measurement_dlo_lab_vast_canary",
        input_bundle_get_url=resolved_urls["provider_bundle_url"],
        output_put_url=resolved_urls["provider_output_put_url"],
        output_get_url=resolved_urls["provider_output_get_url"],
        provider=get_render_provider(args.provider),
        paid_resource_admission_grant=paid_resource_admission_grant,
    )


def run_measurement_canary_from_canonical_allocator(
    *, operation: str, **kwargs: Any
) -> dict[str, Any]:
    if operation == "measurement_dlo_lab_canary":
        return run_measurement_dlo_lab_from_canonical_allocator(**kwargs)
    if operation == "measurement_isaac_canary":
        from .measurement_isaac_paid_allocator import (
            run_measurement_isaac_from_canonical_allocator,
        )

        return run_measurement_isaac_from_canonical_allocator(**kwargs)
    if operation == "measurement_chrono_dem_canary":
        from .measurement_chrono_dem_vast_canary import (
            run_measurement_chrono_dem_vast_canary,
        )

        args = kwargs["args"]
        adapter_path = kwargs["adapter_path"]
        resolved_urls = kwargs["resolved_urls"]
        return run_measurement_chrono_dem_vast_canary(
            bound_request=kwargs["load_json"](args.bound_request_out),
            bundle_receipt=kwargs["bundle_receipt"],
            preflight=kwargs["load_json"](args.preflight_bundle),
            job_dir=adapter_path.parent / "measurement_chrono_dem_vast_canary",
            input_bundle_get_url=resolved_urls["provider_bundle_url"],
            output_put_url=resolved_urls["provider_output_put_url"],
            output_get_url=resolved_urls["provider_output_get_url"],
            provider=get_render_provider(args.provider),
            paid_resource_admission_grant=kwargs["paid_resource_admission_grant"],
        )
    raise ValueError("measurement_paid_allocator_operation_unsupported")


__all__ = [
    "add_measurement_allocator_arguments",
    "add_measurement_chrono_dem_allocator_arguments",
    "add_measurement_dlo_lab_allocator_arguments",
    "run_measurement_canary_from_canonical_allocator",
    "run_measurement_dlo_lab_from_canonical_allocator",
]
