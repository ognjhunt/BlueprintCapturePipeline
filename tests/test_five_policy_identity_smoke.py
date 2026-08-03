from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from blueprint_pipeline.five_policy_identity_smoke import (
    build_input_bundle,
    canonical_gcs_rows,
    extract_input_bundle,
    gcs_generation_manifest_sha256,
    run_identity_smoke,
    validate_registry,
)
from blueprint_pipeline.openpi_policy_ranking_gpu_admission import (
    MENAGERIE_REVISION,
    OPENPI_REVISION,
    build_openpi_policy_ranking_gpu_admission,
)
from blueprint_pipeline.openpi_policy_ranking_runpod import _validate_output_archive
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


REGISTRY = (
    Path(__file__).parents[1]
    / "docs/evidence/five_policy_proveit_registry_2026-08-03.json"
)


def test_registry_and_input_bundle_are_exactly_five_and_digest_bound(tmp_path: Path) -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert validate_registry(registry) == []

    bundle = tmp_path / "input.zip"
    receipt = build_input_bundle(registry_path=REGISTRY, output_zip=bundle)
    assert receipt["manifest"]["candidate_count"] == 5
    assert receipt["manifest"]["query_count_per_candidate"] == 1

    extracted = extract_input_bundle(
        bundle_path=bundle,
        expected_bundle_sha256=receipt["bundle_sha256"],
        output_dir=tmp_path / "extracted",
    )
    assert validate_registry(extracted["registry"]) == []
    assert extracted["manifest"]["registry_sha256"] == receipt["manifest"]["registry_sha256"]


def test_gcs_generation_manifest_is_order_independent() -> None:
    objects = [
        {
            "name": "checkpoints/example/b",
            "generation": "2",
            "metageneration": "1",
            "size": "2",
            "md5Hash": "bbb=",
            "crc32c": "b=",
            "etag": "etag-b",
        },
        {
            "name": "checkpoints/example/a",
            "generation": "1",
            "metageneration": "1",
            "size": "1",
            "md5Hash": "aaa=",
            "crc32c": "a=",
            "etag": "etag-a",
        },
    ]
    forward = canonical_gcs_rows({"items": objects})
    reverse = canonical_gcs_rows({"items": list(reversed(objects))})
    assert forward == reverse
    assert gcs_generation_manifest_sha256(forward) == gcs_generation_manifest_sha256(reverse)


class _FakePolicy:
    def __init__(self, rows: int, value: float) -> None:
        self.rows = rows
        self.value = value

    def infer(self, observation: dict[str, object]) -> dict[str, np.ndarray]:
        assert observation["observation/exterior_image_1_left"].shape == (224, 224, 3)
        assert observation["observation/wrist_image_left"].shape == (224, 224, 3)
        return {"actions": np.full((self.rows, 8), self.value, dtype=np.float32)}


def test_five_real_identity_receipt_mechanics_with_fake_runtime(tmp_path: Path) -> None:
    bundle = tmp_path / "input.zip"
    receipt = build_input_bundle(registry_path=REGISTRY, output_zip=bundle)
    extracted = extract_input_bundle(
        bundle_path=bundle,
        expected_bundle_sha256=receipt["bundle_sha256"],
        output_dir=tmp_path / "extracted",
    )
    registry = extracted["registry"]
    metadata_by_uri: dict[str, dict[str, object]] = {}
    checkpoint_by_uri: dict[str, Path] = {}
    for index, candidate in enumerate(registry["direct_droid_execution_cohort"]):
        checkpoint = tmp_path / f"checkpoint-{index}"
        checkpoint.mkdir()
        payload = f"checkpoint-{index}".encode()
        (checkpoint / "weights.bin").write_bytes(payload)
        md5 = base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode()
        row = {
            "name": candidate["checkpoint_uri"].removeprefix("gs://openpi-assets/")
            + "/weights.bin",
            "generation": str(index + 1),
            "metageneration": "1",
            "size": str(len(payload)),
            "md5Hash": md5,
            "crc32c": "test=",
            "etag": f"etag-{index}",
        }
        rows = canonical_gcs_rows({"items": [row]})
        candidate["checkpoint_object_count"] = 1
        candidate["checkpoint_size_bytes"] = len(payload)
        candidate["gcs_generation_manifest_sha256"] = gcs_generation_manifest_sha256(rows)
        metadata_by_uri[candidate["checkpoint_uri"]] = {"items": [row]}
        checkpoint_by_uri[candidate["checkpoint_uri"]] = checkpoint
    registry["direct_cohort_total_checkpoint_bytes"] = sum(
        row["checkpoint_size_bytes"] for row in registry["direct_droid_execution_cohort"]
    )

    rows_by_config = {
        row["config_name"]: row["native_action_chunk_rows"]
        for row in registry["direct_droid_execution_cohort"]
    }
    result = run_identity_smoke(
        extracted=extracted,
        output_dir=tmp_path / "output",
        metadata_fetcher=lambda uri: metadata_by_uri[uri],
        checkpoint_downloader=lambda uri: checkpoint_by_uri[uri],
        policy_loader=lambda config, _checkpoint: _FakePolicy(
            rows_by_config[config], float(list(rows_by_config).index(config) + 1)
        ),
        require_gpu=False,
    )
    assert result["status"] == "completed"
    assert result["completed_identity_query_count"] == 5
    assert {row["candidate_id"] for row in result["query_receipts"]} == set(rows_by_config)
    assert all(row["fresh_infer_call_count"] == 1 for row in result["query_receipts"])
    assert result["claim_boundary"]["actions_executed"] is False


def test_gpu_admission_accepts_five_policy_identity_smoke_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "input.zip"
    input_receipt = build_input_bundle(registry_path=REGISTRY, output_zip=bundle)
    source_commit = "a" * 40
    release = {
        "schema_version": "openpi_policy_ranking_gpu_release.v1",
        "status": "passed",
        "resolved_digest_ref": "docker.io/example/five-policy@sha256:" + "b" * 64,
        "source_commit": source_commit,
        "runnable_platform": "linux/amd64",
        "openpi_revision": OPENPI_REVISION,
        "menagerie_revision": MENAGERIE_REVISION,
        "checkpoint_bytes_embedded": 0,
        "interiorgs_assets_embedded": False,
    }
    preflight = {
        "schema_version": "openpi_policy_ranking_provider_preflight.v2",
        "status": "verified",
        "provider": "vast",
        "observed_at_epoch": time.time(),
        "provider_api_verified": True,
        "provider_inventory_verified_zero": True,
        "single_gpu_available": True,
        "gpu_type_id": "NVIDIA RTX A6000",
        "gpu_memory_bytes": 48 * 1024**3,
        "on_demand_price_usd_per_hour": 0.75,
        "container_disk_bytes": 100 * 1024**3,
    }
    spend = {
        "paid_mutation_authorized": True,
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
        "watchdog_armed_before_allocation": True,
        "hard_ttl_seconds": 3600,
        "max_spend_usd": 5.0,
        "physical_robot_endpoint_access_allowed": False,
    }
    admission = build_openpi_policy_ranking_gpu_admission(
        release=release,
        input_bundle=input_receipt,
        preflight=preflight,
        spend=spend,
        expected_source_commit=source_commit,
    )
    assert admission["status"] == "admitted", admission["blockers"]
    assert admission["execution_mode"] == "five_policy_identity_smoke"
    assert admission["checkpoint_size_bytes"] == 58_138_199_882


def test_output_validator_accepts_five_real_identity_receipts() -> None:
    receipts = []
    for index in range(5):
        receipt = {
            "schema_version": "five_policy_identity_query_receipt.v1",
            "candidate_id": f"policy-{index}",
            "status": "completed",
            "fresh_infer_call_count": 1,
            "fixture_or_fake": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        receipts.append(receipt)
    manifest = {
        "schema_version": "five_policy_identity_smoke_result.v1",
        "status": "completed",
        "blockers": [],
        "expected_candidate_count": 5,
        "completed_identity_query_count": 5,
        "query_receipts": receipts,
        "claim_boundary": {
            "actions_executed": False,
            "policy_ranking": False,
            "task_success": False,
            "physical_robot_execution": False,
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "five_policy_identity_smoke_result.json",
            json.dumps(manifest, sort_keys=True),
        )
    validation = _validate_output_archive(stream.getvalue())
    assert validation["status"] == "completed", validation["blockers"]
    assert validation["execution_mode"] == "five_policy_identity_smoke"
    assert validation["identity_query_count"] == 5
