from __future__ import annotations

from blueprint_pipeline.groot_oscar_runpod_carrier_volume import (
    CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
    DEFAULT_MODEL_CACHE_ROOT,
    DEFAULT_RUNTIME_ARCHIVE_PATH,
    DEFAULT_RUNTIME_MANIFEST_PATH,
    DEFAULT_RUNTIME_ROOT,
    RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION,
    build_runtime_bundle_manifest,
    canonical_json_sha256,
    runtime_bootstrap_shell_prefix,
    verify_carrier_volume_admission,
)


SOURCE_REF = "docker.io/blueprint/release@sha256:" + "1" * 64
CARRIER_REF = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:" + "2" * 64


def _admission() -> dict:
    return {
        "schema_version": CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
        "status": "verified",
        "carrier_image_ref": CARRIER_REF,
        "network_volume": {
            "id": "volume123",
            "data_center_id": "EUR-IS-1",
            "size_gib": 120,
        },
        "runtime_bundle": {
            "manifest_schema_version": RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "source_release_image_ref": SOURCE_REF,
            "root": DEFAULT_RUNTIME_ROOT,
            "archive_path": DEFAULT_RUNTIME_ARCHIVE_PATH,
            "manifest_path": DEFAULT_RUNTIME_MANIFEST_PATH,
            "archive_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
        },
        "model_cache": {
            "status": "verified",
            "root": DEFAULT_MODEL_CACHE_ROOT,
            "manifest_sha256": "5" * 64,
        },
        "s3_transfer_verification": {
            "upload_completed": True,
            "full_redownload_sha256_verified": True,
            "provider_volume_id": "volume123",
            "data_center_id": "EUR-IS-1",
        },
        "raw_secret_values_recorded": False,
    }


def test_runtime_manifest_requires_digest_pinned_source_and_carrier() -> None:
    manifest = build_runtime_bundle_manifest(
        source_release_image_ref=SOURCE_REF,
        carrier_image_ref=CARRIER_REF,
        archive_sha256="a" * 64,
        archive_size_bytes=1234,
        healthcheck_argv=(
            ("/opt/gr00t-venv/bin/python", "--version"),
            ("/opt/oscar-venv/bin/python", "--version"),
        ),
        generated_at="2026-07-15T12:00:00Z",
    )

    assert manifest["status"] == "complete"
    assert manifest["archive"]["format"] == "tar.gz"
    assert "opt/gr00t-venv" in manifest["archive"]["member_roots"]
    assert len(canonical_json_sha256(manifest)) == 64
    assert "semantic task success" in manifest["claim_boundary"]

    blocked = build_runtime_bundle_manifest(
        source_release_image_ref="docker.io/blueprint/release:latest",
        carrier_image_ref="pytorch/pytorch:latest",
        archive_sha256="bad",
        archive_size_bytes=0,
        healthcheck_argv=(("python", "--version"),),
        generated_at="now",
    )
    assert blocked["status"] == "blocked"
    assert "runtime_source_release_image_not_digest_pinned" in blocked["blockers"]
    assert "runtime_carrier_image_not_digest_pinned" in blocked["blockers"]
    assert "runtime_healthcheck_executable_outside_opt" in blocked["blockers"]


def test_carrier_volume_admission_binds_runtime_models_s3_and_volume() -> None:
    verified = verify_carrier_volume_admission(
        _admission(), expected_carrier_image_ref=CARRIER_REF
    )

    assert verified["status"] == "verified"
    assert verified["network_volume_id"] == "volume123"
    assert verified["data_center_id"] == "EUR-IS-1"
    assert verified["runtime_archive_sha256"] == "3" * 64
    assert verified["model_manifest_sha256"] == "5" * 64
    assert "provider attachment" in verified["claim_boundary"]

    bad = _admission()
    bad["network_volume"]["size_gib"] = 50
    bad["s3_transfer_verification"]["provider_volume_id"] = "another-volume"
    bad["model_cache"]["status"] = "prepared"
    blocked = verify_carrier_volume_admission(bad, expected_carrier_image_ref=CARRIER_REF)

    assert blocked["status"] == "blocked"
    assert "carrier_network_volume_below_120_gib" in blocked["blockers"]
    assert "carrier_volume_s3_volume_binding_mismatch" in blocked["blockers"]
    assert "carrier_model_cache_not_verified" in blocked["blockers"]


def test_runtime_bootstrap_is_hash_gated_path_safe_and_observable() -> None:
    script = runtime_bootstrap_shell_prefix()

    assert "BLUEPRINT_RUNPOD_CARRIER_RUNTIME_BOOTSTRAP_STARTED" in script
    assert "runtime_manifest_sha256_mismatch" in script
    assert "runtime_archive_sha256_mismatch" in script
    assert "runtime_archive_member_outside_allowlist" in script
    assert "runtime_archive_link_outside_allowlist" in script
    assert 'archive.extractall(path="/", filter=lambda member, _path: member)' in script
    assert "model_cache_manifest_sha256_mismatch" in script
    assert "model_cache_declared_file_sha256_mismatch" in script
    assert "all_declared_model_file_sha256_verified" in script
    assert "runtime_wbc_dynamic_linkage_failed" in script
    assert "BLUEPRINT_OSCAR_WAM_CHECKPOINT" in script
    assert "runpod_carrier_runtime_bootstrap_blocked.zip" in script
    assert "BLUEPRINT_RUNPOD_CARRIER_RUNTIME_BOOTSTRAP_READY" in script
