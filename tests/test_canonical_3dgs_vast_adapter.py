from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import zipfile

import jsonschema
import pytest

from blueprint_pipeline import canonical_3dgs_vast_output as vast_output
from blueprint_pipeline import reconstruction_vast_operation as vast_operation
from blueprint_pipeline.canonical_3dgs_vast_output import (
    MANIFEST_MEMBER,
    compile_canonical_3dgs_vast_output_bundle,
    validate_canonical_3dgs_vast_output_bundle,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_gpu_operation_output import (
    ReconstructionGpuOperationOutputError,
)
from blueprint_pipeline.reconstruction_gpu_admission import (
    CANONICAL_SPLATFACTO_VAST_ADAPTER_ID,
    build_reconstruction_gpu_canary_admission,
    build_reconstruction_gpu_canary_request,
)


SOURCE = "a" * 40
IMAGE = "dromni/nerfstudio@sha256:" + "b" * 64
TRANSPORT = "sha256:" + "4" * 64
DATASET = "sha256:" + "5" * 64
PACKAGE = "sha256:" + "6" * 64


def _write_splat(path: Path) -> None:
    properties = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        + "".join(f"property float {name}\n" for name in properties)
        + "end_header\n"
    )
    path.write_bytes(
        header.encode("ascii")
        + struct.pack("<14f", 0, 0, 1, 0, 0, 0, 1, -3, -3, -3, 1, 0, 0, 0)
    )


def _request() -> dict:
    return build_reconstruction_gpu_canary_request(
        {
            "schema_version": "reconstruction_gpu_canary_request.v1",
            "operation": "trainer_canary",
            "capture_profile": "public_provider_sample",
            "source_commit_sha": SOURCE,
            "worker_image_digest": IMAGE,
            "worker_stack_manifest_digest": "sha256:" + "1" * 64,
            "deterministic_configuration_digest": "sha256:" + "2" * 64,
            "operation_request_digest": "sha256:" + "3" * 64,
            "operation_input_bundle_digest": "sha256:" + "4" * 64,
            "reconstruction_dataset_digest": "sha256:" + "5" * 64,
            "frozen_split_digest": "sha256:" + "6" * 64,
            "calibration_digest": "sha256:" + "7" * 64,
            "expected_runtime_result_schema": "canonical_3dgs_vast_runtime_result.v1",
            "requested_execution_adapter_id": CANONICAL_SPLATFACTO_VAST_ADAPTER_ID,
            "candidate_may_read_hidden_heldout": False,
            "trainer_may_grade_heldout": False,
            "max_spend_usd": 10.0,
            "hard_ttl_seconds": 7200,
            "retry_cap": 0,
            "authority_id": "explicit-user-authority",
            "proof_effect": "none",
        }
    )


def _preflight() -> dict:
    value = {
        "schema_version": "reconstruction_gpu_provider_preflight.v1",
        "status": "verified",
        "provider": "vast",
        "observed_at_epoch": 100.0,
        "provider_api_verified": True,
        "provider_inventory_verified_zero": True,
        "conflicting_owner_present": False,
        "watchdog": {"status": "armed", "independent_process": True},
        "single_gpu_available": True,
        "gpu_memory_bytes": 24 * 1024**3,
        "container_disk_bytes": 100 * 1024**3,
        "on_demand_price_usd_per_hour": 1.0,
    }
    value["preflight_digest"] = canonical_digest(value, digest_field="preflight_digest")
    return value


def _write_worker_controls(root: Path, splat: Path) -> dict:
    transport = {
        "schema_version": "canonical_3dgs_transport_bundle.v1",
        "status": "compiled",
        "transport_bundle_digest": TRANSPORT,
        "transport_bundle_bytes": 1024,
        "transport_manifest_digest": "sha256:" + "7" * 64,
        "canonical_3dgs_execution_plan_digest": "sha256:" + "3" * 64,
        "worker_python_package_digest": PACKAGE,
        "colmap_training_dataset_digest": DATASET,
        "source_capture_digest": "sha256:" + "8" * 64,
        "frozen_split_digest": "sha256:" + "9" * 64,
        "source_commit_sha": SOURCE,
        "dataset_members": [
            {
                "archive_path": "campaign/dataset/images/frame.png",
                "digest": "sha256:" + "a" * 64,
                "bytes": 1,
            }
        ],
        "dataset_member_count": 1,
        "hidden_heldout_pixels_included": False,
        "raw_secret_values_included": False,
        "provider_allocation_performed": False,
        "paid_execution_authorized_by_bundle": False,
        "proof_effect": "none",
    }
    transport["receipt_digest"] = canonical_digest(
        transport, digest_field="receipt_digest"
    )
    allocator = {
        "status": "execute_ready",
        "execution_adapter_id": "canonical_splatfacto_vast_v1",
        "operation_request_digest": "sha256:" + "3" * 64,
        "operation_input_bundle_digest": TRANSPORT,
        "reconstruction_dataset_digest": DATASET,
        "worker_image_digest": IMAGE,
        "source_commit_sha": SOURCE,
        "max_spend_usd": 10.0,
        "hard_ttl_seconds": 7200,
        "authority_id": "explicit-user-authority",
    }
    allocator["admission_digest"] = canonical_digest(
        allocator, digest_field="admission_digest"
    )
    admission = {
        "schema_version": "canonical_3dgs_worker_admission.v1",
        "status": "admitted",
        "blockers": [],
        "arm_id": "splatfacto-comparison",
        "worker_platform": "linux",
        "canonical_3dgs_execution_plan_digest": "sha256:" + "3" * 64,
        "colmap_training_dataset_digest": DATASET,
        "transport_bundle_digest": TRANSPORT,
        "worker_python_package_digest": PACKAGE,
        "provider_upload_authorized": True,
        "paid_compute_authorized": True,
        "watchdog_armed_before_execution": True,
        "provider_zero_verified_before_allocation": True,
        "provider_zero_required_after_execution": True,
        "retry_cap": 0,
        "allocation_binding_digest": allocator["admission_digest"],
        "paid_allocator_admission_digest": allocator["admission_digest"],
        "worker_image_digest": IMAGE,
        "trainer_runtime_digest": "sha256:" + "d" * 64,
        "trainer_runtime_version": "nerfstudio-1.1.5+gsplat-1.4.0",
        "max_spend_usd": 10.0,
        "hard_ttl_seconds": 7200,
        "authority_id": "explicit-user-authority",
        "timestamp": "2026-08-03T05:00:00Z",
        "expires_at": "2026-08-03T07:00:00Z",
    }
    admission["canonical_3dgs_worker_admission_digest"] = canonical_digest(
        admission, digest_field="canonical_3dgs_worker_admission_digest"
    )
    receipt = {
        "exit_code": 0,
        "timestamp": "2026-08-03T06:00:00Z",
        "canonical_3dgs_execution_plan_digest": "sha256:" + "3" * 64,
        "transport_bundle_digest": TRANSPORT,
        "transport_receipt_digest": transport["receipt_digest"],
        "canonical_3dgs_worker_admission_digest": admission[
            "canonical_3dgs_worker_admission_digest"
        ],
        "allocation_binding_digest": admission["allocation_binding_digest"],
        "provider_zero_required_after_execution": True,
        "runtime_identity": {
            "worker_image_digest": IMAGE,
            "source_commit_sha_bound_by_plan": SOURCE,
            "worker_python_package_digest": PACKAGE,
            "trainer_runtime_digest": admission["trainer_runtime_digest"],
            "trainer_runtime_version": admission["trainer_runtime_version"],
        },
        "artifacts": [
            {
                "kind": "standard_3dgs_ply",
                "relative_path": "candidate.ply",
                "digest": "sha256:" + hashlib.sha256(splat.read_bytes()).hexdigest(),
            },
            {"kind": "training_log", "relative_path": "training.log"},
        ],
    }
    receipt["canonical_3dgs_worker_receipt_digest"] = canonical_digest(
        receipt, digest_field="canonical_3dgs_worker_receipt_digest"
    )
    for name, value in (
        ("canonical_3dgs_transport_receipt.json", transport),
        ("canonical_3dgs_worker_admission.json", admission),
        ("paid_allocator_admission.json", allocator),
        ("worker_receipt.json", receipt),
    ):
        (root / name).write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    return receipt


def test_canonical_vast_bootstrap_uses_container_python3_entrypoint() -> None:
    script = vast_operation._bootstrap_script(canonical_splatfacto=True)

    assert "python3 - <<'PY'" in script
    assert "python3 -m pip install" in script
    assert "python3 -m blueprint_pipeline.canonical_3dgs_vast_bootstrap" in script
    assert "\npython -" not in script
    assert "\npython -m" not in script
    assert script.index("INPUT_RECEIPT_GET_URL") < script.index("INPUT_BUNDLE_GET_URL")
    assert "transport_bundle_bytes" in script
    assert "BLUEPRINT_CANONICAL_MAX_INPUT_BYTES" in script
    assert "canonical_download_stream_oversized" in script
    assert "archive.read(wheel_member)" not in script


def test_specialized_request_cannot_be_qualified_by_generic_boolean() -> None:
    blocked, bound = build_reconstruction_gpu_canary_admission(
        request=_request(),
        preflight=_preflight(),
        provider="vast",
        expected_source_commit=SOURCE,
        checkout_source_commit=SOURCE,
        checkout_clean=True,
        max_spend_usd=10.0,
        hard_ttl_seconds=7200,
        retry_cap=0,
        authority_id="explicit-user-authority",
        execute=True,
        execution_adapter_qualified=True,
        observed_now_epoch=100.0,
    )
    assert blocked["status"] == "blocked"
    assert "reconstruction_gpu_requested_execution_adapter_unavailable" in blocked["blockers"]
    assert bound["provider_mutation_authorized"] is False

    admitted, bound = build_reconstruction_gpu_canary_admission(
        request=_request(),
        preflight=_preflight(),
        provider="vast",
        expected_source_commit=SOURCE,
        checkout_source_commit=SOURCE,
        checkout_clean=True,
        max_spend_usd=10.0,
        hard_ttl_seconds=7200,
        retry_cap=0,
        authority_id="explicit-user-authority",
        execute=True,
        execution_adapter_id=CANONICAL_SPLATFACTO_VAST_ADAPTER_ID,
        observed_now_epoch=100.0,
    )
    assert admitted["status"] == "execute_ready"
    assert admitted["execution_adapter_id"] == CANONICAL_SPLATFACTO_VAST_ADAPTER_ID
    assert bound["execution_adapter_id"] == CANONICAL_SPLATFACTO_VAST_ADAPTER_ID


def test_canonical_vast_output_independently_decodes_standard_ply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "result"
    root.mkdir()
    splat = root / "candidate.ply"
    _write_splat(splat)
    log = root / "training.log"
    log.write_text("complete\n", encoding="utf-8")
    receipt = _write_worker_controls(root, splat)
    output = tmp_path / "output.zip"
    compiled = compile_canonical_3dgs_vast_output_bundle(
        result_root=root,
        worker_receipt=receipt,
        output_path=output,
        worker_image_digest=IMAGE,
        source_commit_sha=SOURCE,
    )
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(
            archive.read("canonical_3dgs_vast_output_manifest.json")
        )
    schema = json.loads(
        Path("docs/schemas/canonical_3dgs_vast_output_bundle.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(manifest)
    original_read = zipfile.ZipFile.read

    def manifest_only_read(archive, member, *args, **kwargs):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        assert name == MANIFEST_MEMBER, "artifact members must be streamed"
        return original_read(archive, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", manifest_only_read)
    validated, runtime = validate_canonical_3dgs_vast_output_bundle(
        bundle_path=output,
        expected_operation="trainer_canary",
        expected_operation_request_digest="sha256:" + "3" * 64,
        expected_transport_bundle_digest=TRANSPORT,
        expected_reconstruction_dataset_digest=DATASET,
        expected_allocator_admission_digest=receipt["allocation_binding_digest"],
        expected_worker_image_digest=IMAGE,
        expected_source_commit_sha=SOURCE,
    )
    assert validated["operation_output_bundle_digest"] == compiled[
        "operation_output_bundle_digest"
    ]
    assert validated["gaussian_count"] == 1
    assert runtime["status"] == "succeeded"


def test_canonical_vast_output_rejects_tampering(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionGpuOperationOutputError):
        validate_canonical_3dgs_vast_output_bundle(
            bundle_path=tmp_path / "missing.zip",
            expected_operation="trainer_canary",
            expected_operation_request_digest="sha256:" + "3" * 64,
            expected_transport_bundle_digest=TRANSPORT,
            expected_reconstruction_dataset_digest=DATASET,
            expected_allocator_admission_digest="sha256:" + "c" * 64,
            expected_worker_image_digest=IMAGE,
            expected_source_commit_sha=SOURCE,
        )


def test_canonical_vast_output_rejects_self_consistent_tampered_worker_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "result"
    root.mkdir()
    splat = root / "candidate.ply"
    _write_splat(splat)
    (root / "training.log").write_text("complete\n", encoding="utf-8")
    receipt = _write_worker_controls(root, splat)
    output = tmp_path / "output.zip"
    compile_canonical_3dgs_vast_output_bundle(
        result_root=root,
        worker_receipt=receipt,
        output_path=output,
        worker_image_digest=IMAGE,
        source_commit_sha=SOURCE,
    )
    with zipfile.ZipFile(output) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    worker = json.loads(payloads["results/worker_receipt.json"])
    worker["transport_bundle_digest"] = "sha256:" + "e" * 64
    worker["canonical_3dgs_worker_receipt_digest"] = canonical_digest(
        worker, digest_field="canonical_3dgs_worker_receipt_digest"
    )
    worker_bytes = (
        json.dumps(worker, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    payloads["results/worker_receipt.json"] = worker_bytes
    manifest = json.loads(payloads[MANIFEST_MEMBER])
    manifest["worker_receipt_digest"] = worker[
        "canonical_3dgs_worker_receipt_digest"
    ]
    worker_row = next(
        row
        for row in manifest["members"]
        if row["archive_path"] == "results/worker_receipt.json"
    )
    worker_row["digest"] = "sha256:" + hashlib.sha256(worker_bytes).hexdigest()
    worker_row["bytes"] = len(worker_bytes)
    manifest["output_bundle_receipt_digest"] = canonical_digest(
        manifest, digest_field="output_bundle_receipt_digest"
    )
    payloads[MANIFEST_MEMBER] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)

    with pytest.raises(ReconstructionGpuOperationOutputError):
        validate_canonical_3dgs_vast_output_bundle(
            bundle_path=tampered,
            expected_operation="trainer_canary",
            expected_operation_request_digest="sha256:" + "3" * 64,
            expected_transport_bundle_digest=TRANSPORT,
            expected_reconstruction_dataset_digest=DATASET,
            expected_allocator_admission_digest=receipt["allocation_binding_digest"],
            expected_worker_image_digest=IMAGE,
            expected_source_commit_sha=SOURCE,
        )


def test_canonical_vast_output_caps_manifest_before_json_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "oversized-manifest.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(MANIFEST_MEMBER, b"{" + b" " * 64 + b"}")
    monkeypatch.setattr(vast_output, "MAX_MANIFEST_BYTES", 16)

    with pytest.raises(ReconstructionGpuOperationOutputError):
        validate_canonical_3dgs_vast_output_bundle(
            bundle_path=bundle,
            expected_operation="trainer_canary",
            expected_operation_request_digest="sha256:" + "3" * 64,
            expected_transport_bundle_digest=TRANSPORT,
            expected_reconstruction_dataset_digest=DATASET,
            expected_allocator_admission_digest="sha256:" + "c" * 64,
            expected_worker_image_digest=IMAGE,
            expected_source_commit_sha=SOURCE,
        )
