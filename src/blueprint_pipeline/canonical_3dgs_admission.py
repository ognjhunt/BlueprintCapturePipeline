"""Fail-closed execution admission for one canonical 3DGS worker arm.

Transporting a campaign is not authority to upload it, spend money, or run a
licensed trainer. This contract binds those decisions to one plan, one arm,
one transported byte set, one allocated worker, a watchdog deadline, and an
explicit spend ceiling before the worker command can start.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .canonical_3dgs_transport import validate_canonical_3dgs_transport_receipt
from .decision_evidence_contracts import canonical_digest, canonical_json


ADMISSION_SCHEMA = "canonical_3dgs_worker_admission.v1"
ARM_PLATFORMS = {
    "postshot-primary": "windows",
    "splatfacto-comparison": "linux",
}
MAX_TTL_SECONDS = 14_400


class Canonical3DGSAdmissionError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _positive_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise Canonical3DGSAdmissionError(
                ["canonical_3dgs_admission_immutable_conflict"]
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != payload:
                raise Canonical3DGSAdmissionError(
                    ["canonical_3dgs_admission_immutable_conflict"]
                )
    finally:
        temporary.unlink(missing_ok=True)


def build_canonical_3dgs_worker_admission(
    *,
    transport_receipt: Mapping[str, Any],
    arm_id: str,
    worker_platform: str,
    allocation_binding_digest: str,
    trainer_runtime_digest: str,
    trainer_runtime_version: str,
    authority_id: str,
    max_spend_usd: float,
    hard_ttl_seconds: int,
    provider_upload_authorized: bool,
    paid_compute_authorized: bool,
    watchdog_armed: bool,
    provider_zero_before_allocation: bool,
    timestamp: str,
) -> dict[str, Any]:
    transport = validate_canonical_3dgs_transport_receipt(transport_receipt)
    blockers: list[str] = []
    if arm_id not in ARM_PLATFORMS:
        blockers.append("canonical_3dgs_admission_arm_invalid")
    elif worker_platform != ARM_PLATFORMS[arm_id]:
        blockers.append("canonical_3dgs_admission_platform_mismatch")
    if not _digest(allocation_binding_digest):
        blockers.append("canonical_3dgs_admission_allocation_binding_invalid")
    if not _digest(trainer_runtime_digest):
        blockers.append("canonical_3dgs_admission_trainer_runtime_digest_invalid")
    if not str(trainer_runtime_version or "").strip():
        blockers.append("canonical_3dgs_admission_trainer_runtime_version_missing")
    if not str(authority_id or "").strip():
        blockers.append("canonical_3dgs_admission_authority_missing")
    if not _positive_number(max_spend_usd):
        blockers.append("canonical_3dgs_admission_spend_ceiling_invalid")
    if (
        isinstance(hard_ttl_seconds, bool)
        or not isinstance(hard_ttl_seconds, int)
        or not 1 <= hard_ttl_seconds <= MAX_TTL_SECONDS
    ):
        blockers.append("canonical_3dgs_admission_ttl_invalid")
    for value, code in (
        (provider_upload_authorized, "canonical_3dgs_admission_upload_authority_missing"),
        (paid_compute_authorized, "canonical_3dgs_admission_paid_authority_missing"),
        (watchdog_armed, "canonical_3dgs_admission_watchdog_not_armed"),
        (
            provider_zero_before_allocation,
            "canonical_3dgs_admission_provider_zero_not_proven",
        ),
    ):
        if value is not True:
            blockers.append(code)
    if not str(timestamp or "").strip():
        blockers.append("canonical_3dgs_admission_timestamp_missing")
    result = {
        "schema_version": ADMISSION_SCHEMA,
        "status": "admitted" if not blockers else "blocked",
        "arm_id": arm_id,
        "worker_platform": worker_platform,
        "canonical_3dgs_execution_plan_digest": transport[
            "canonical_3dgs_execution_plan_digest"
        ],
        "worker_python_package_digest": transport[
            "worker_python_package_digest"
        ],
        "colmap_training_dataset_digest": transport[
            "colmap_training_dataset_digest"
        ],
        "transport_bundle_digest": transport["transport_bundle_digest"],
        "transport_receipt_digest": transport["receipt_digest"],
        "allocation_binding_digest": allocation_binding_digest,
        "trainer_runtime_digest": trainer_runtime_digest,
        "trainer_runtime_version": str(trainer_runtime_version),
        "authority_id": str(authority_id),
        "max_spend_usd": float(max_spend_usd),
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "provider_upload_authorized": provider_upload_authorized is True,
        "paid_compute_authorized": paid_compute_authorized is True,
        "watchdog_armed_before_execution": watchdog_armed is True,
        "provider_zero_verified_before_allocation": provider_zero_before_allocation
        is True,
        "provider_zero_required_after_execution": True,
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
        "quality_or_scientific_result_inferred": False,
        "proof_effect": "worker_execution_authority_only",
        "timestamp": str(timestamp),
    }
    result["canonical_3dgs_worker_admission_digest"] = canonical_digest(
        result, digest_field="canonical_3dgs_worker_admission_digest"
    )
    return result


def require_canonical_3dgs_worker_admission(
    value: Mapping[str, Any],
    *,
    arm_id: str,
    plan_digest: str,
    dataset_digest: str,
    transport_bundle_digest: str,
    worker_package_digest: str,
) -> dict[str, Any]:
    admission = json.loads(canonical_json(dict(value)))
    errors: list[str] = []
    if admission.get("schema_version") != ADMISSION_SCHEMA:
        errors.append("canonical_3dgs_worker_admission_schema_invalid")
    if admission.get("canonical_3dgs_worker_admission_digest") != canonical_digest(
        admission, digest_field="canonical_3dgs_worker_admission_digest"
    ):
        errors.append("canonical_3dgs_worker_admission_digest_mismatch")
    if admission.get("status") != "admitted" or admission.get("blockers") != []:
        errors.append("canonical_3dgs_worker_admission_not_admitted")
    expected = {
        "arm_id": arm_id,
        "worker_platform": ARM_PLATFORMS.get(arm_id),
        "canonical_3dgs_execution_plan_digest": plan_digest,
        "colmap_training_dataset_digest": dataset_digest,
        "transport_bundle_digest": transport_bundle_digest,
        "worker_python_package_digest": worker_package_digest,
    }
    for key, expected_value in expected.items():
        if admission.get(key) != expected_value:
            errors.append(f"canonical_3dgs_worker_admission_binding_mismatch:{key}")
    for key in (
        "provider_upload_authorized",
        "paid_compute_authorized",
        "watchdog_armed_before_execution",
        "provider_zero_verified_before_allocation",
        "provider_zero_required_after_execution",
    ):
        if admission.get(key) is not True:
            errors.append(f"canonical_3dgs_worker_admission_control_missing:{key}")
    if admission.get("retry_cap") != 0:
        errors.append("canonical_3dgs_worker_admission_retry_cap_invalid")
    if not _digest(admission.get("allocation_binding_digest")):
        errors.append("canonical_3dgs_worker_admission_allocation_binding_invalid")
    if not _digest(admission.get("trainer_runtime_digest")):
        errors.append("canonical_3dgs_worker_admission_trainer_runtime_digest_invalid")
    if not str(admission.get("trainer_runtime_version") or "").strip():
        errors.append("canonical_3dgs_worker_admission_trainer_runtime_version_missing")
    if not _positive_number(admission.get("max_spend_usd")):
        errors.append("canonical_3dgs_worker_admission_spend_ceiling_invalid")
    ttl = admission.get("hard_ttl_seconds")
    if isinstance(ttl, bool) or not isinstance(ttl, int) or not 1 <= ttl <= MAX_TTL_SECONDS:
        errors.append("canonical_3dgs_worker_admission_ttl_invalid")
    if not str(admission.get("authority_id") or "").strip():
        errors.append("canonical_3dgs_worker_admission_authority_missing")
    if errors:
        raise Canonical3DGSAdmissionError(errors)
    return admission


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport-receipt", required=True)
    parser.add_argument("--arm", choices=tuple(ARM_PLATFORMS), required=True)
    parser.add_argument("--worker-platform", choices=("windows", "linux"), required=True)
    parser.add_argument("--allocation-binding-digest", required=True)
    parser.add_argument("--trainer-runtime-digest", required=True)
    parser.add_argument("--trainer-runtime-version", required=True)
    parser.add_argument("--authority-id", required=True)
    parser.add_argument("--max-spend-usd", type=float, required=True)
    parser.add_argument("--hard-ttl-seconds", type=int, required=True)
    parser.add_argument("--provider-upload-authorized", action="store_true")
    parser.add_argument("--paid-compute-authorized", action="store_true")
    parser.add_argument("--watchdog-armed", action="store_true")
    parser.add_argument("--provider-zero-before-allocation", action="store_true")
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    try:
        transport = json.loads(
            Path(arguments.transport_receipt).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise Canonical3DGSAdmissionError(
            ["canonical_3dgs_admission_transport_receipt_invalid"]
        ) from exc
    admission = build_canonical_3dgs_worker_admission(
        transport_receipt=transport,
        arm_id=arguments.arm,
        worker_platform=arguments.worker_platform,
        allocation_binding_digest=arguments.allocation_binding_digest,
        trainer_runtime_digest=arguments.trainer_runtime_digest,
        trainer_runtime_version=arguments.trainer_runtime_version,
        authority_id=arguments.authority_id,
        max_spend_usd=arguments.max_spend_usd,
        hard_ttl_seconds=arguments.hard_ttl_seconds,
        provider_upload_authorized=arguments.provider_upload_authorized,
        paid_compute_authorized=arguments.paid_compute_authorized,
        watchdog_armed=arguments.watchdog_armed,
        provider_zero_before_allocation=arguments.provider_zero_before_allocation,
        timestamp=arguments.timestamp,
    )
    output = Path(arguments.output)
    _write_immutable_json(output, admission)
    print(canonical_json(admission))
    return 0 if admission["status"] == "admitted" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADMISSION_SCHEMA",
    "Canonical3DGSAdmissionError",
    "build_canonical_3dgs_worker_admission",
    "require_canonical_3dgs_worker_admission",
]
