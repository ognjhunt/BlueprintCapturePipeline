from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import reconstruction_gpu_operation_bootstrap as bootstrap
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.safe_outbound_http import SafeHttpFileTransfer


SHA = "a" * 40
REQUEST_DIGEST = "sha256:" + "1" * 64
MANIFEST_DIGEST = "sha256:" + "2" * 64
IMAGE = "registry.example/reconstruction@sha256:" + "b" * 64


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _receipt(*, input_digest: str) -> dict:
    value = {
        "schema_version": "reconstruction_gpu_operation_bundle.v1",
        "status": "compiled",
        "operation": "pose_canary",
        "operation_request_digest": REQUEST_DIGEST,
        "operation_input_bundle_digest": input_digest,
        "bundle_manifest_digest": MANIFEST_DIGEST,
        "worker_image_digest": IMAGE,
        "source_commit_sha": SHA,
        "artifact_members": [
            {
                "archive_path": "inputs/frame.png",
                "digest": "sha256:" + "3" * 64,
                "bytes": 1,
            }
        ],
        "artifact_member_count": 1,
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
        "raw_secret_values_included": False,
        "provider_allocation_performed": False,
        "paid_execution_authorized_by_bundle": False,
        "proof_effect": "none",
        "claim_ceiling": "candidate_operation_input_only",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def _environment(*, input_digest: str, receipt_file_digest: str) -> dict[str, str]:
    return {
        "BLUEPRINT_RECONSTRUCTION_OPERATION": "pose_canary",
        "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_GET_URL": (
            "https://objects.example/input.zip?signature=input-secret"
        ),
        "BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_GET_URL": (
            "https://objects.example/receipt.json?signature=receipt-secret"
        ),
        "BLUEPRINT_RECONSTRUCTION_OUTPUT_BUNDLE_PUT_URL": (
            "https://objects.example/output.zip?signature=output-secret"
        ),
        "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_DIGEST": input_digest,
        "BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_FILE_DIGEST": receipt_file_digest,
        "BLUEPRINT_RECONSTRUCTION_OPERATION_REQUEST_DIGEST": REQUEST_DIGEST,
        "BLUEPRINT_CONTAINER_IMAGE_DIGEST": IMAGE,
        "BLUEPRINT_SOURCE_COMMIT": SHA,
    }


def test_bootstrap_binds_exact_downloads_dispatch_and_complete_output_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_bytes = b"operation-input-zip"
    input_digest = _digest(input_bytes)
    receipt = _receipt(input_digest=input_digest)
    receipt_bytes = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    receipt_file_digest = _digest(receipt_bytes)
    uploads: list[dict[str, object]] = []

    def fake_download(url, *, output_path, expected_sha256, **_kwargs):
        payload = receipt_bytes if "receipt.json" in url else input_bytes
        assert _digest(payload) == expected_sha256
        Path(output_path).write_bytes(payload)
        return SafeHttpFileTransfer(
            status=200,
            transferred_bytes=len(payload),
            sha256=expected_sha256,
            host="objects.example",
        )

    runtime_result = {"pose_estimation_result_digest": "sha256:" + "4" * 64}

    def fake_execute(**kwargs):
        assert kwargs["bundle_receipt"] == receipt
        assert Path(kwargs["bundle_path"]).read_bytes() == input_bytes
        materialized = Path(kwargs["materialization_root"]) / input_digest[7:]
        materialized.mkdir(parents=True)
        (materialized / "operation_request.json").write_text(
            json.dumps({"fixture": True}), encoding="utf-8"
        )
        return runtime_result

    output_bytes = b"complete-output-bundle"

    def fake_compile(**kwargs):
        assert kwargs["operation"] == "pose_canary"
        assert kwargs["runtime_result"] == runtime_result
        Path(kwargs["output_path"]).write_bytes(output_bytes)
        return {"operation_output_bundle_digest": _digest(output_bytes)}

    def fake_upload(url, *, input_path, expected_sha256, **_kwargs):
        assert Path(input_path).read_bytes() == output_bytes
        assert expected_sha256 == _digest(output_bytes)
        uploads.append({"url_host": "objects.example", "digest": expected_sha256})
        return SafeHttpFileTransfer(
            status=200,
            transferred_bytes=len(output_bytes),
            sha256=expected_sha256,
            host="objects.example",
        )

    monkeypatch.setattr(bootstrap, "download_file", fake_download)
    monkeypatch.setattr(
        bootstrap, "execute_reconstruction_gpu_operation_bundle", fake_execute
    )
    monkeypatch.setattr(
        bootstrap, "compile_reconstruction_gpu_operation_output_bundle", fake_compile
    )
    monkeypatch.setattr(bootstrap, "upload_file", fake_upload)

    result = bootstrap.run_reconstruction_gpu_operation_bootstrap(
        environment=_environment(
            input_digest=input_digest,
            receipt_file_digest=receipt_file_digest,
        ),
        work_root=tmp_path / "worker",
    )
    assert result["status"] == "output_uploaded"
    assert result["operation_output_bundle_digest"] == _digest(output_bytes)
    assert result["scientific_qualification_inferred"] is False
    assert result["proof_effect"] == "none"
    assert result["bootstrap_receipt_digest"] == canonical_digest(
        result, digest_field="bootstrap_receipt_digest"
    )
    assert uploads == [
        {"url_host": "objects.example", "digest": _digest(output_bytes)}
    ]
    encoded = json.dumps(result)
    assert "input-secret" not in encoded
    assert "receipt-secret" not in encoded
    assert "output-secret" not in encoded


def test_bootstrap_rejects_receipt_binding_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_bytes = b"operation-input-zip"
    input_digest = _digest(input_bytes)
    receipt = _receipt(input_digest=input_digest)
    receipt["operation_request_digest"] = "sha256:" + "9" * 64
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_bytes = json.dumps(receipt).encode("utf-8")

    def fake_download(url, *, output_path, expected_sha256, **_kwargs):
        payload = receipt_bytes if "receipt.json" in url else input_bytes
        Path(output_path).write_bytes(payload)
        return SafeHttpFileTransfer(
            status=200,
            transferred_bytes=len(payload),
            sha256=expected_sha256,
            host="objects.example",
        )

    monkeypatch.setattr(bootstrap, "download_file", fake_download)
    monkeypatch.setattr(
        bootstrap,
        "execute_reconstruction_gpu_operation_bundle",
        lambda **_kwargs: pytest.fail("dispatch must not run"),
    )
    with pytest.raises(
        bootstrap.ReconstructionGpuOperationBootstrapError,
        match="receipt_binding_mismatch",
    ):
        bootstrap.run_reconstruction_gpu_operation_bootstrap(
            environment=_environment(
                input_digest=input_digest,
                receipt_file_digest=_digest(receipt_bytes),
            ),
            work_root=tmp_path / "worker",
        )
