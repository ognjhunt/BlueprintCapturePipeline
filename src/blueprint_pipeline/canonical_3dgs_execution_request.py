"""Compile an immutable, non-authorizing request for canonical 3DGS workers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .canonical_3dgs_pipeline import PLAN_SCHEMA
from .canonical_3dgs_transport import validate_canonical_3dgs_transport_receipt
from .decision_evidence_contracts import canonical_digest, canonical_json


REQUEST_SCHEMA = "canonical_3dgs_execution_request.v1"


def build_canonical_3dgs_execution_request(
    *, plan: Mapping[str, Any], transport_receipt: Mapping[str, Any], timestamp: str
) -> dict[str, Any]:
    """Bind retained bytes and name every missing fact without inventing authority."""

    source = json.loads(canonical_json(dict(plan)))
    transport = validate_canonical_3dgs_transport_receipt(transport_receipt)
    errors: list[str] = []
    if source.get("schema_version") != PLAN_SCHEMA or source.get(
        "canonical_3dgs_execution_plan_digest"
    ) != canonical_digest(source, digest_field="canonical_3dgs_execution_plan_digest"):
        errors.append("canonical_3dgs_execution_request_plan_invalid")
    bindings = {
        "canonical_3dgs_execution_plan_digest": source.get(
            "canonical_3dgs_execution_plan_digest"
        ),
        "source_commit_sha": source.get("source_commit_sha"),
        "worker_python_package_digest": source.get("worker_python_package_digest"),
        "colmap_training_dataset_digest": source.get("colmap_training_dataset_digest"),
        "frozen_split_digest": source.get("frozen_split_digest"),
    }
    for key, expected in bindings.items():
        if transport.get(key) != expected:
            errors.append(f"canonical_3dgs_execution_request_transport_mismatch:{key}")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    common = [
        "explicit_paid_execution_authority_id_missing",
        "provider_upload_authorization_missing",
        "per_arm_max_spend_usd_missing",
        "per_arm_hard_ttl_seconds_missing",
        "independent_watchdog_handoff_missing",
        "provider_zero_preflight_missing",
        "paid_resource_allocator_execute_ready_admission_missing",
    ]
    arms = [
        {
            "arm_id": "postshot-primary",
            "worker_platform": "windows",
            "worker_image_digest": None,
            "trainer_runtime_digest": None,
            "trainer_runtime_version": None,
            "allocator_operation": "trainer_canary",
            "retry_cap": 0,
            "blockers": sorted(
                common
                + [
                    "paid_resource_allocator_windows_trainer_adapter_not_qualified",
                    "postshot_cli_binary_digest_and_version_missing",
                    "postshot_license_and_runtime_credentials_missing",
                ]
            ),
        },
        {
            "arm_id": "splatfacto-comparison",
            "worker_platform": "linux",
            "worker_image_digest": None,
            "trainer_runtime_digest": None,
            "trainer_runtime_version": "nerfstudio=1.1.5;gsplat=1.4.0",
            "allocator_operation": "trainer_canary",
            "retry_cap": 0,
            "blockers": sorted(
                common
                + [
                    "linux_cuda_worker_image_digest_missing",
                    "observed_nerfstudio_gsplat_runtime_digest_missing",
                ]
            ),
        },
    ]
    result = {
        "schema_version": REQUEST_SCHEMA,
        "status": "admission_ready_missing_authority",
        "source_profile": source.get("source_profile"),
        "source_capture_digest": source.get("source_capture_digest"),
        "source_commit_sha": source.get("source_commit_sha"),
        "canonical_3dgs_source_admission_digest": source.get(
            "canonical_3dgs_source_admission_digest"
        ),
        "canonical_3dgs_execution_plan_digest": source[
            "canonical_3dgs_execution_plan_digest"
        ],
        "worker_python_package_digest": source["worker_python_package_digest"],
        "colmap_training_dataset_digest": source["colmap_training_dataset_digest"],
        "frozen_split_digest": source["frozen_split_digest"],
        "transport_bundle_digest": transport["transport_bundle_digest"],
        "transport_receipt_digest": transport["receipt_digest"],
        "arms": arms,
        "provider_launch_entrypoint": "python -m blueprint_pipeline.paid_resource_allocator gpu-canary",
        "provider_specific_launcher_permitted": False,
        "paid_execution_authorized": False,
        "provider_upload_authorized": False,
        "worker_retry_cap": 0,
        "quality_winner": None,
        "candidate_generated": False,
        "proof_effect": "none",
        "claim_ceiling": "admission_ready_request_only",
        "timestamp": timestamp,
    }
    result["canonical_3dgs_execution_request_digest"] = canonical_digest(
        result, digest_field="canonical_3dgs_execution_request_digest"
    )
    return result


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(value)) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError("canonical_3dgs_execution_request_immutable_conflict")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--transport-receipt", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    result = build_canonical_3dgs_execution_request(
        plan=json.loads(Path(arguments.plan).read_text()),
        transport_receipt=json.loads(Path(arguments.transport_receipt).read_text()),
        timestamp=arguments.timestamp,
    )
    _write_immutable(Path(arguments.output).resolve(), result)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REQUEST_SCHEMA", "build_canonical_3dgs_execution_request"]
