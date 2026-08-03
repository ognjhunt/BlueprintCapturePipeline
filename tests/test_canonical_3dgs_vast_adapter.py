from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import zipfile

import jsonschema
import pytest

from blueprint_pipeline.canonical_3dgs_vast_output import (
    compile_canonical_3dgs_vast_output_bundle,
    validate_canonical_3dgs_vast_output_bundle,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_gpu_admission import (
    CANONICAL_SPLATFACTO_VAST_ADAPTER_ID,
    build_reconstruction_gpu_canary_admission,
    build_reconstruction_gpu_canary_request,
)


SOURCE = "a" * 40
IMAGE = "dromni/nerfstudio@sha256:" + "b" * 64


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


def test_canonical_vast_output_independently_decodes_standard_ply(tmp_path: Path) -> None:
    root = tmp_path / "result"
    root.mkdir()
    splat = root / "candidate.ply"
    _write_splat(splat)
    log = root / "training.log"
    log.write_text("complete\n", encoding="utf-8")
    receipt = {
        "exit_code": 0,
        "canonical_3dgs_execution_plan_digest": "sha256:" + "3" * 64,
        "transport_bundle_digest": "sha256:" + "4" * 64,
        "runtime_identity": {
            "worker_image_digest": IMAGE,
            "source_commit_sha_bound_by_plan": SOURCE,
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
    validated, runtime = validate_canonical_3dgs_vast_output_bundle(
        bundle_path=output,
        expected_operation="trainer_canary",
        expected_operation_request_digest="sha256:" + "3" * 64,
        expected_worker_image_digest=IMAGE,
        expected_source_commit_sha=SOURCE,
    )
    assert validated["operation_output_bundle_digest"] == compiled[
        "operation_output_bundle_digest"
    ]
    assert validated["gaussian_count"] == 1
    assert runtime["status"] == "succeeded"


def test_canonical_vast_output_rejects_tampering(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        validate_canonical_3dgs_vast_output_bundle(
            bundle_path=tmp_path / "missing.zip",
            expected_operation="trainer_canary",
            expected_operation_request_digest="sha256:" + "3" * 64,
            expected_worker_image_digest=IMAGE,
            expected_source_commit_sha=SOURCE,
        )
