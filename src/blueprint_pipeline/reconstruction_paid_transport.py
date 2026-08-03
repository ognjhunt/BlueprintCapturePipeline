"""Receipt and file-backed URL preparation for reconstruction-family paid canaries."""

from __future__ import annotations

from pathlib import Path
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
from .canonical_3dgs_transport import (
    Canonical3DGSTransportError,
    validate_canonical_3dgs_transport_receipt,
)
from .reconstruction_isaac_worker_bundle import (
    IsaacWorkerBundleError,
    validate_isaac_verification_worker_bundle_receipt,
)


def _chrono_transport_manifest_blockers(
    manifest: Mapping[str, Any], *, args: Any, receipt: Mapping[str, Any]
) -> list[str]:
    blockers: list[str] = []
    object_store = manifest.get("object_store")
    object_store = object_store if isinstance(object_store, Mapping) else {}
    round_trip = manifest.get("signed_output_round_trip")
    round_trip = round_trip if isinstance(round_trip, Mapping) else {}
    if (
        manifest.get("schema_version") != "wam_provider_object_store_staging.v1"
        or manifest.get("status") != "completed"
        or manifest.get("raw_secret_values_recorded") is not False
    ):
        blockers.append("measurement_chrono_dem_transport_manifest_invalid")
    expected_bundle_sha256 = str(receipt.get("input_bundle_digest") or "").removeprefix(
        "sha256:"
    )
    if (
        not expected_bundle_sha256
        or manifest.get("bundle_sha256") != expected_bundle_sha256
    ):
        blockers.append("measurement_chrono_dem_transport_bundle_digest_mismatch")
    if (
        object_store.get("output_content_type") != "application/json"
        or not str(manifest.get("output_key") or "").endswith(".json")
    ):
        blockers.append("measurement_chrono_dem_transport_content_type_mismatch")
    if (
        round_trip.get("status") != "passed"
        or round_trip.get("blockers") != []
        or not all(
            isinstance(round_trip.get(key), Mapping)
            and round_trip[key].get("status") == "passed"
            for key in ("put", "get", "cleanup")
        )
    ):
        blockers.append("measurement_chrono_dem_signed_json_round_trip_not_passed")
    file_bindings = (
        ("provider_bundle_url_file", "provider_bundle_url_file"),
        ("provider_output_put_url_file", "provider_output_put_url_file"),
        ("provider_output_get_url_file", "provider_output_get_url_file"),
    )
    for manifest_key, argument_key in file_bindings:
        row = manifest.get(manifest_key)
        row = row if isinstance(row, Mapping) else {}
        expected_path = str(getattr(args, argument_key, None) or "")
        observed_path = str(row.get("path") or "")
        try:
            paths_match = bool(
                expected_path
                and observed_path
                and Path(expected_path).expanduser().resolve()
                == Path(observed_path).expanduser().resolve()
            )
        except OSError:
            paths_match = False
        if (
            not paths_match
            or row.get("present") is not True
            or row.get("mode_is_0600") is not True
        ):
            blockers.append(f"measurement_chrono_dem_transport_{manifest_key}_mismatch")
    return blockers


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
                elif admission.get("execution_adapter_id") == "canonical_splatfacto_vast_v1":
                    receipt = validate_canonical_3dgs_transport_receipt(raw)
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
                Canonical3DGSTransportError,
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
                        ("operation_request_digest", "canonical_3dgs_execution_plan_digest"),
                        ("operation_input_bundle_digest", "transport_bundle_digest"),
                        ("source_commit_sha", "source_commit_sha"),
                    )
                    if admission.get("execution_adapter_id")
                    == "canonical_splatfacto_vast_v1"
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
        if operation == "measurement_chrono_dem_canary":
            staging_path = getattr(
                args, "measurement_chrono_dem_object_store_staging_manifest", None
            )
            if not staging_path:
                blockers.append("measurement_chrono_dem_transport_manifest_missing")
            else:
                try:
                    staging_manifest = load_json(staging_path)
                except (OSError, TypeError, ValueError):
                    blockers.append("measurement_chrono_dem_transport_manifest_invalid")
                else:
                    blockers.extend(
                        _chrono_transport_manifest_blockers(
                            staging_manifest, args=args, receipt=receipt
                        )
                    )
    resolved, url_blockers = resolve_paid_transport_urls(urls, blocker_prefix="reconstruction")
    blockers.extend(url_blockers)
    return receipt, resolved, blockers


__all__ = ["prepare_reconstruction_paid_transport"]
