"""Run one admitted reconstruction operation inside the pinned GPU image."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .reconstruction_gpu_operation_bundle import (
    ReconstructionGpuOperationBundleError,
    validate_reconstruction_gpu_operation_bundle_receipt,
)
from .reconstruction_gpu_operation_output import (
    ReconstructionGpuOperationOutputError,
    compile_reconstruction_gpu_operation_output_bundle,
)
from .reconstruction_gpu_operation_worker import (
    ReconstructionGpuOperationWorkerError,
    execute_reconstruction_gpu_operation_bundle,
)
from .safe_outbound_http import (
    SafeOutboundHttpError,
    download_file,
    presigned_transfer_policy,
    upload_file,
)


SCHEMA_VERSION = "reconstruction_gpu_operation_bootstrap.v1"
MAX_INPUT_BUNDLE_BYTES = 96 * 1024**3
MAX_RECEIPT_BYTES = 64 * 1024**2
MAX_OUTPUT_BUNDLE_BYTES = 96 * 1024**3
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_IMAGE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")


class ReconstructionGpuOperationBootstrapError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _required_env(environment: Mapping[str, str], name: str) -> str:
    value = str(environment.get(name) or "")
    if not value:
        raise ReconstructionGpuOperationBootstrapError(
            [f"reconstruction_operation_bootstrap_env_missing:{name}"]
        )
    return value


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionGpuOperationBootstrapError(
            ["reconstruction_operation_bootstrap_receipt_invalid"]
        ) from exc
    if not isinstance(value, Mapping):
        raise ReconstructionGpuOperationBootstrapError(
            ["reconstruction_operation_bootstrap_receipt_invalid"]
        )
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def run_reconstruction_gpu_operation_bootstrap(
    *,
    environment: Mapping[str, str],
    work_root: str | Path,
) -> dict[str, Any]:
    """Download exact inputs, execute one typed operation, and upload all outputs."""

    operation = _required_env(environment, "BLUEPRINT_RECONSTRUCTION_OPERATION")
    input_url = _required_env(
        environment, "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_GET_URL"
    )
    receipt_url = _required_env(
        environment, "BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_GET_URL"
    )
    output_url = _required_env(
        environment, "BLUEPRINT_RECONSTRUCTION_OUTPUT_BUNDLE_PUT_URL"
    )
    input_digest = _required_env(
        environment, "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_DIGEST"
    )
    receipt_file_digest = _required_env(
        environment, "BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_FILE_DIGEST"
    )
    request_digest = _required_env(
        environment, "BLUEPRINT_RECONSTRUCTION_OPERATION_REQUEST_DIGEST"
    )
    worker_image = _required_env(environment, "BLUEPRINT_CONTAINER_IMAGE_DIGEST")
    source_commit = _required_env(environment, "BLUEPRINT_SOURCE_COMMIT")
    if (
        operation not in {"pose_canary", "trainer_canary"}
        or _DIGEST.fullmatch(input_digest) is None
        or _DIGEST.fullmatch(receipt_file_digest) is None
        or _DIGEST.fullmatch(request_digest) is None
        or _IMAGE.fullmatch(worker_image) is None
        or _COMMIT.fullmatch(source_commit) is None
    ):
        raise ReconstructionGpuOperationBootstrapError(
            ["reconstruction_operation_bootstrap_binding_invalid"]
        )
    root = Path(work_root)
    if root.is_symlink():
        raise ReconstructionGpuOperationBootstrapError(
            ["reconstruction_operation_bootstrap_root_symlink_forbidden"]
        )
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    input_path = root / "reconstruction_gpu_operation_input.zip"
    receipt_path = root / "reconstruction_gpu_operation_bundle_receipt.json"
    output_path = root / "reconstruction_gpu_operation_output.zip"
    try:
        input_transfer = download_file(
            input_url,
            output_path=input_path,
            expected_sha256=input_digest,
            max_bytes=MAX_INPUT_BUNDLE_BYTES,
            timeout_seconds=3600,
            policy=presigned_transfer_policy(input_url),
        )
        receipt_transfer = download_file(
            receipt_url,
            output_path=receipt_path,
            expected_sha256=receipt_file_digest,
            max_bytes=MAX_RECEIPT_BYTES,
            timeout_seconds=300,
            policy=presigned_transfer_policy(receipt_url),
        )
        receipt = validate_reconstruction_gpu_operation_bundle_receipt(
            _load_object(receipt_path)
        )
    except (SafeOutboundHttpError, ReconstructionGpuOperationBundleError) as exc:
        raise ReconstructionGpuOperationBootstrapError(
            ["reconstruction_operation_bootstrap_input_invalid"]
        ) from exc
    if (
        receipt.get("operation") != operation
        or receipt.get("operation_input_bundle_digest") != input_digest
        or receipt.get("operation_request_digest") != request_digest
        or receipt.get("worker_image_digest") != worker_image
        or receipt.get("source_commit_sha") != source_commit
    ):
        raise ReconstructionGpuOperationBootstrapError(
            ["reconstruction_operation_bootstrap_receipt_binding_mismatch"]
        )
    try:
        result = execute_reconstruction_gpu_operation_bundle(
            bundle_path=input_path,
            bundle_receipt=receipt,
            materialization_root=root / "materialized",
            output_root=root / "operation_outputs",
        )
        materialized = (
            root / "materialized" / input_digest.removeprefix("sha256:")
        )
        operation_request = _load_object(materialized / "operation_request.json")
        output_receipt = compile_reconstruction_gpu_operation_output_bundle(
            operation=operation,
            operation_request=operation_request,
            runtime_result=result,
            operation_output_root=root / "operation_outputs",
            output_path=output_path,
        )
        output_transfer = upload_file(
            output_url,
            input_path=output_path,
            expected_sha256=output_receipt["operation_output_bundle_digest"],
            max_bytes=MAX_OUTPUT_BUNDLE_BYTES,
            timeout_seconds=3600,
            policy=presigned_transfer_policy(output_url, max_response_bytes=1024 * 1024),
            content_type="application/zip",
        )
    except (
        ReconstructionGpuOperationWorkerError,
        ReconstructionGpuOperationOutputError,
        SafeOutboundHttpError,
    ) as exc:
        raise ReconstructionGpuOperationBootstrapError(
            ["reconstruction_operation_bootstrap_execution_failed"]
        ) from exc
    result_digest = result.get("pose_estimation_result_digest") or result.get(
        "reconstruction_training_result_digest"
    )
    bootstrap = {
        "schema_version": SCHEMA_VERSION,
        "status": "output_uploaded",
        "operation": operation,
        "operation_request_digest": request_digest,
        "operation_input_bundle_digest": input_digest,
        "operation_input_bundle_bytes": input_transfer.transferred_bytes,
        "operation_receipt_file_digest": receipt_transfer.sha256,
        "runtime_result_digest": result_digest,
        "operation_output_bundle_digest": output_transfer.sha256,
        "operation_output_bundle_bytes": output_transfer.transferred_bytes,
        "worker_image_digest": worker_image,
        "source_commit_sha": source_commit,
        "hidden_heldout_observations_accessed": False,
        "scientific_qualification_inferred": False,
        "raw_secret_values_recorded": False,
        "proof_effect": "none",
        "claim_ceiling": "provider_output_transport_only",
    }
    bootstrap["bootstrap_receipt_digest"] = canonical_digest(
        bootstrap, digest_field="bootstrap_receipt_digest"
    )
    write_json(root / "reconstruction_gpu_operation_bootstrap.v1.json", bootstrap)
    return bootstrap


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="blueprint_reconstruction_operation_"
    ) as work_root:
        result = run_reconstruction_gpu_operation_bootstrap(
            environment=os.environ,
            work_root=work_root,
        )
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "status": result["status"],
                "operation": result["operation"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ReconstructionGpuOperationBootstrapError",
    "run_reconstruction_gpu_operation_bootstrap",
]


if __name__ == "__main__":
    raise SystemExit(main())
