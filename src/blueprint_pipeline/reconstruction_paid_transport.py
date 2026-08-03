"""Receipt and file-backed URL preparation for reconstruction-family paid canaries."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .measurement_isaac_vast_bundle import (
    MeasurementIsaacVastBundleError,
    validate_measurement_isaac_physx_input_bundle_receipt,
)
from .measurement_dlo_lab_vast_bundle import (
    MeasurementDloLabVastBundleError,
    validate_measurement_dlo_lab_input_bundle_receipt,
)
from .measurement_chrono_dem_vast_bundle import (
    MeasurementChronoDemVastBundleError,
    validate_measurement_chrono_dem_input_bundle_receipt,
)
from .paid_resource_transport import resolve_paid_transport_urls
from .reconstruction_gpu_operation_bundle import (
    ReconstructionGpuOperationBundleError,
    validate_reconstruction_gpu_operation_bundle_receipt,
)
from .reconstruction_isaac_worker_bundle import (
    IsaacWorkerBundleError,
    validate_isaac_verification_worker_bundle_receipt,
)


def prepare_reconstruction_paid_transport(
    *,
    args: Any,
    admission: Mapping[str, Any],
    load_json: Callable[[str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    operation = str(admission.get("operation") or "worker_smoke")
    blockers: list[str] = []
    urls = [
        ("provider_output_put_url", args.provider_output_put_url_file),
        ("provider_output_get_url", args.provider_output_get_url_file),
    ]
    receipt: dict[str, Any] = {}
    isaac_operations = {
        "isaac_canary",
        "provider_nurec_isaac_canary",
        "external_scene_isaac_canary",
    }
    scientific = operation in {
        "pose_canary",
        "trainer_canary",
        "measurement_isaac_canary",
        "measurement_dlo_lab_canary",
        "measurement_chrono_dem_canary",
    } or operation in isaac_operations
    if scientific:
        urls.append(("provider_bundle_url", getattr(args, "provider_bundle_url_file", None)))
        if operation not in {
            "measurement_isaac_canary",
            "measurement_dlo_lab_canary",
            "measurement_chrono_dem_canary",
        }:
            urls.append(
                (
                    "operation_receipt_get_url",
                    getattr(args, "reconstruction_operation_receipt_url_file", None),
                )
            )
        receipt_path = (
            getattr(args, "measurement_isaac_bundle_receipt", None)
            if operation == "measurement_isaac_canary"
            else getattr(args, "measurement_dlo_lab_bundle_receipt", None)
            if operation == "measurement_dlo_lab_canary"
            else getattr(args, "measurement_chrono_dem_bundle_receipt", None)
            if operation == "measurement_chrono_dem_canary"
            else getattr(args, "reconstruction_operation_bundle_receipt", None)
        )
        if not receipt_path:
            blockers.append("reconstruction_operation_bundle_receipt_missing")
        else:
            try:
                raw = load_json(receipt_path)
                if operation in isaac_operations:
                    receipt = validate_isaac_verification_worker_bundle_receipt(raw)
                elif operation == "measurement_isaac_canary":
                    receipt = validate_measurement_isaac_physx_input_bundle_receipt(raw)
                elif operation == "measurement_dlo_lab_canary":
                    receipt = validate_measurement_dlo_lab_input_bundle_receipt(raw)
                elif operation == "measurement_chrono_dem_canary":
                    receipt = validate_measurement_chrono_dem_input_bundle_receipt(raw)
                else:
                    receipt = validate_reconstruction_gpu_operation_bundle_receipt(raw)
            except (
                OSError,
                TypeError,
                ValueError,
                ReconstructionGpuOperationBundleError,
                IsaacWorkerBundleError,
                MeasurementIsaacVastBundleError,
                MeasurementDloLabVastBundleError,
                MeasurementChronoDemVastBundleError,
            ):
                blockers.append("reconstruction_operation_bundle_receipt_invalid")
            else:
                bindings = (
                    (
                        ("operation_request_digest", "isaac_verification_request_digest"),
                        ("operation_input_bundle_digest", "bundle_digest"),
                        ("worker_image_digest", "runtime_container_image_digest"),
                        ("source_commit_sha", "source_commit_sha"),
                    )
                    if operation in isaac_operations
                    else (
                        ("operation_request_digest", "bundle_manifest_digest"),
                        ("operation_input_bundle_digest", "input_bundle_digest"),
                        ("worker_image_digest", "runtime_image_digest"),
                        ("source_commit_sha", "source_commit_sha"),
                    )
                    if operation
                    in {
                        "measurement_isaac_canary",
                        "measurement_dlo_lab_canary",
                        "measurement_chrono_dem_canary",
                    }
                    else (
                        ("operation", "operation"),
                        ("operation_request_digest", "operation_request_digest"),
                        ("operation_input_bundle_digest", "operation_input_bundle_digest"),
                        ("worker_image_digest", "worker_image_digest"),
                        ("source_commit_sha", "source_commit_sha"),
                    )
                )
                for request_key, receipt_key in bindings:
                    if admission.get(request_key) != receipt.get(receipt_key):
                        blockers.append(f"reconstruction_operation_bundle_{request_key}_mismatch")
    resolved, url_blockers = resolve_paid_transport_urls(urls, blocker_prefix="reconstruction")
    blockers.extend(url_blockers)
    return receipt, resolved, blockers


__all__ = ["prepare_reconstruction_paid_transport"]
