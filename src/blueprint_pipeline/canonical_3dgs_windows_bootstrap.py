"""Execute and return one canonical Postshot arm on an admitted Windows host."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from .canonical_3dgs_admission import build_canonical_3dgs_worker_admission
from .canonical_3dgs_transport import (
    extract_canonical_3dgs_transport_bundle,
    validate_canonical_3dgs_transport_receipt,
)
from .canonical_3dgs_vast_output import compile_canonical_3dgs_vast_output_bundle
from .canonical_3dgs_worker import main as run_worker_main
from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .safe_outbound_http import (
    download_file_observed,
    presigned_transfer_policy,
    upload_file,
)


class Canonical3DGSWindowsBootstrapError(ValueError):
    pass


def _publish_progress(
    *, environment: Mapping[str, str], root: Path, stage: str, state: str
) -> None:
    url = str(environment.get("BLUEPRINT_RECONSTRUCTION_PROGRESS_PUT_URL") or "")
    if not url:
        return
    progress: dict[str, Any] = {
        "schema_version": "capture_reconstruction_worker_progress.v1",
        "stage": stage,
        "state": state,
        "capture_digest": environment.get("BLUEPRINT_RECONSTRUCTION_CAPTURE_DIGEST"),
        "observed_at": utc_now_iso(),
        "scientific_qualification_inferred": False,
    }
    progress["progress_digest"] = canonical_digest(
        progress, digest_field="progress_digest"
    )
    path = root / "worker_progress.json"
    write_json(path, progress)
    upload_file(
        url,
        input_path=path,
        expected_sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        max_bytes=64 * 1024,
        timeout_seconds=60,
        policy=presigned_transfer_policy(url),
        content_type="application/json",
    )


def _required(environment: Mapping[str, str], name: str) -> str:
    value = str(environment.get(name) or "")
    if not value:
        raise Canonical3DGSWindowsBootstrapError(
            f"canonical_windows_env_missing:{name}"
        )
    return value


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Canonical3DGSWindowsBootstrapError(
            "canonical_windows_json_not_object"
        )
    return dict(value)


def run_canonical_3dgs_windows_bootstrap(
    *, environment: Mapping[str, str], work_root: str | Path
) -> dict[str, Any]:
    """Download, revalidate, train, package, and upload one Postshot result."""

    root = Path(work_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bundle_url = _required(
        environment, "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_GET_URL"
    )
    receipt_url = _required(
        environment, "BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_GET_URL"
    )
    output_url = _required(
        environment, "BLUEPRINT_RECONSTRUCTION_OUTPUT_BUNDLE_PUT_URL"
    )
    bundle_path = root / "canonical_3dgs_transport.zip"
    receipt_path = root / "canonical_3dgs_transport_receipt.json"
    receipt_transfer = download_file_observed(
        receipt_url,
        output_path=receipt_path,
        max_bytes=8 * 1024**2,
        timeout_seconds=300,
        policy=presigned_transfer_policy(receipt_url),
    )
    if receipt_transfer.sha256 != _required(
        environment, "BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_FILE_DIGEST"
    ):
        raise Canonical3DGSWindowsBootstrapError(
            "canonical_windows_transport_receipt_file_digest_mismatch"
        )
    transport = validate_canonical_3dgs_transport_receipt(_json(receipt_path))
    bundle_transfer = download_file_observed(
        bundle_url,
        output_path=bundle_path,
        max_bytes=int(transport["transport_bundle_bytes"]),
        timeout_seconds=900,
        policy=presigned_transfer_policy(bundle_url),
    )
    if (
        bundle_transfer.sha256 != transport["transport_bundle_digest"]
        or bundle_transfer.sha256
        != _required(environment, "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_DIGEST")
    ):
        raise Canonical3DGSWindowsBootstrapError(
            "canonical_windows_transport_bundle_digest_mismatch"
        )

    allocator = json.loads(
        base64.b64decode(
            _required(environment, "BLUEPRINT_CANONICAL_ALLOCATOR_ADMISSION_B64"),
            validate=True,
        )
    )
    if not isinstance(allocator, Mapping):
        raise Canonical3DGSWindowsBootstrapError(
            "canonical_windows_allocator_admission_not_object"
        )
    image = _required(environment, "BLUEPRINT_WORKER_IMAGE_DIGEST")
    source_commit = _required(environment, "BLUEPRINT_SOURCE_COMMIT")
    admission = build_canonical_3dgs_worker_admission(
        transport_receipt=transport,
        arm_id="postshot-primary",
        worker_platform="windows",
        paid_allocator_admission=allocator,
        worker_image_digest=image,
        trainer_runtime_digest=_required(
            environment, "BLUEPRINT_POSTSHOT_RUNTIME_DIGEST"
        ),
        trainer_runtime_version=_required(
            environment, "BLUEPRINT_POSTSHOT_RUNTIME_VERSION"
        ),
        authority_id=_required(environment, "BLUEPRINT_CANONICAL_AUTHORITY_ID"),
        max_spend_usd=float(
            _required(environment, "BLUEPRINT_CANONICAL_MAX_SPEND_USD")
        ),
        hard_ttl_seconds=int(
            _required(environment, "BLUEPRINT_CANONICAL_HARD_TTL_SECONDS")
        ),
        provider_upload_authorized=True,
        paid_compute_authorized=True,
        watchdog_armed=True,
        provider_zero_before_allocation=True,
        timestamp=utc_now_iso(),
    )
    admission_path = root / "canonical_3dgs_worker_admission.json"
    write_json(admission_path, admission)
    extraction = extract_canonical_3dgs_transport_bundle(
        bundle_path=bundle_path,
        receipt=transport,
        output_root=root / "materialized",
    )
    _publish_progress(
        environment=environment,
        root=root,
        stage="postshot_import",
        state="input_materialized",
    )
    materialized = (
        root
        / "materialized"
        / extraction["transport_bundle_digest"].removeprefix("sha256:")
    )
    result_root = root / "results" / "postshot-primary"
    result_root.mkdir(parents=True, exist_ok=True)
    write_json(result_root / "paid_allocator_admission.json", allocator)
    write_json(result_root / "canonical_3dgs_transport_receipt.json", transport)
    write_json(result_root / "canonical_3dgs_worker_admission.json", admission)
    plan = _json(materialized / "campaign/canonical_3dgs_execution_plan.json")
    sparse_root = materialized / "campaign/dataset/sparse/0"
    camera_snapshot_root = result_root / "camera_metadata"
    camera_snapshot_root.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        source = sparse_root / name
        if not source.is_file() or source.is_symlink():
            raise Canonical3DGSWindowsBootstrapError(
                f"canonical_windows_camera_metadata_missing:{name}"
            )
        shutil.copyfile(source, camera_snapshot_root / name)
    write_json(
        result_root / "postshot_coordinate_binding.json",
        {
            "schema_version": "postshot_coordinate_binding.v1",
            "source_capture_digest": plan.get("source_capture_digest"),
            "colmap_training_dataset_digest": plan.get(
                "colmap_training_dataset_digest"
            ),
            "world_frame": plan.get("world_frame"),
            "coordinate_frame_declaration": plan.get(
                "coordinate_frame_declaration"
            ),
            "postshot_import_axes": "colmap_opencv_preserved",
            "postshot_recenter_disabled": True,
            "metric_scale_status": plan.get("metric_scale_status"),
            "metric_physical_qualification": (
                "proven"
                if plan.get("metric_scale_status")
                == "independently_validated_metric"
                else "fail_closed_unproven"
            ),
        },
    )
    worker_receipt_path = result_root / "worker_receipt.json"
    _publish_progress(
        environment=environment, root=root, stage="training", state="running"
    )
    exit_code = run_worker_main(
        [
            "--arm",
            "postshot-primary",
            "--plan",
            str(materialized / "campaign/canonical_3dgs_execution_plan.json"),
            "--dataset-root",
            str(materialized / "campaign/dataset"),
            "--output-root",
            str(result_root),
            "--receipt",
            str(worker_receipt_path),
            "--transport-receipt",
            str(receipt_path),
            "--admission",
            str(admission_path),
        ]
    )
    if exit_code != 0:
        raise Canonical3DGSWindowsBootstrapError(
            "canonical_windows_postshot_execution_failed"
        )
    _publish_progress(
        environment=environment, root=root, stage="training", state="completed"
    )
    worker_receipt = _json(worker_receipt_path)
    output_path = root / "canonical_3dgs_postshot_output.zip"
    output_receipt = compile_canonical_3dgs_vast_output_bundle(
        result_root=result_root,
        worker_receipt=worker_receipt,
        output_path=output_path,
        worker_image_digest=image,
        source_commit_sha=source_commit,
        arm_id="postshot-primary",
    )
    _publish_progress(
        environment=environment, root=root, stage="export", state="completed"
    )
    transfer = upload_file(
        output_url,
        input_path=output_path,
        expected_sha256=output_receipt["operation_output_bundle_digest"],
        max_bytes=96 * 1024**3,
        timeout_seconds=900,
        policy=presigned_transfer_policy(output_url),
        content_type="application/zip",
    )
    result = {
        "schema_version": "canonical_3dgs_windows_bootstrap.v1",
        "status": "uploaded",
        "arm_id": "postshot-primary",
        "canonical_3dgs_worker_receipt_digest": worker_receipt[
            "canonical_3dgs_worker_receipt_digest"
        ],
        "operation_output_bundle_digest": transfer.sha256,
        "provider_zero_verified": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "appearance_asset_candidate_only",
    }
    write_json(root / "canonical_3dgs_windows_bootstrap.json", result)
    return result


__all__ = [
    "Canonical3DGSWindowsBootstrapError",
    "run_canonical_3dgs_windows_bootstrap",
]
