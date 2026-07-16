from __future__ import annotations

import pytest

from blueprint_pipeline.groot_oscar_runpod_carrier_volume import (
    CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
    DEFAULT_MODEL_CACHE_ROOT,
    DEFAULT_RUNTIME_ARCHIVE_PATH,
    DEFAULT_RUNTIME_MANIFEST_PATH,
    DEFAULT_RUNTIME_ROOT,
    LEGACY_CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
    RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION,
    RUNTIME_CARRIER_ENV,
    RUNTIME_SOURCE_RELEASE_VERIFICATION_SCHEMA_VERSION,
    build_runtime_bundle_manifest,
    canonical_json_sha256,
    is_nvidia_driver_soname,
    runtime_bootstrap_shell_prefix,
    verify_carrier_volume_admission,
    verify_runtime_source_release_evidence,
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
        "runtime_source_release": {
            "schema_version": RUNTIME_SOURCE_RELEASE_VERIFICATION_SCHEMA_VERSION,
            "status": "verified",
            "release_image_ref": SOURCE_REF,
            "source_commit": "a" * 40,
            "thin_release_contract_sha256": "6" * 64,
            "models_externalized": True,
        },
        "model_cache": {
            "status": "verified",
            "root": DEFAULT_MODEL_CACHE_ROOT,
            "manifest_sha256": "5" * 64,
            "manifest_digest": "sha256:" + "7" * 64,
        },
        "s3_transfer_verification": {
            "upload_completed": True,
            "full_redownload_sha256_verified": True,
            "provider_volume_id": "volume123",
            "data_center_id": "EUR-IS-1",
        },
        "raw_secret_values_recorded": False,
    }


def _release_evidence() -> dict:
    return {
        "schema_version": "groot_oscar_thin_remote_build_result.v1",
        "status": "completed",
        "blockers": [],
        "release_image_ref": SOURCE_REF,
        "resolved_digest_ref": SOURCE_REF,
        "runnable_platform": "linux/amd64",
        "required_cuda_version": "12.8",
        "source_commit": "a" * 40,
        "thin_release_contract_status": "passed",
        "thin_release_contract": {
            "schema_version": "groot_oscar_thin_release_image_contract.v1",
            "status": "passed",
            "blockers": [],
            "release_image_ref": SOURCE_REF,
            "models_externalized": True,
            "release_delta_budget_passed": True,
        },
        "models_embedded": False,
        "raw_secret_values_recorded": False,
    }


def test_runtime_source_release_evidence_requires_exact_thin_release() -> None:
    verified = verify_runtime_source_release_evidence(
        _release_evidence(), expected_release_image_ref=SOURCE_REF
    )
    assert verified["status"] == "verified"
    assert verified["release_image_ref"] == SOURCE_REF
    assert verified["models_externalized"] is True
    assert len(verified["thin_release_contract_sha256"]) == 64

    sealed = _release_evidence()
    sealed["release_image_ref"] = "docker.io/blueprint/sealed@sha256:" + "9" * 64
    sealed["resolved_digest_ref"] = sealed["release_image_ref"]
    sealed["thin_release_contract"]["release_image_ref"] = sealed["release_image_ref"]
    sealed["thin_release_contract"]["models_externalized"] = False
    sealed["models_embedded"] = True
    blocked = verify_runtime_source_release_evidence(sealed, expected_release_image_ref=SOURCE_REF)
    assert blocked["status"] == "blocked"
    assert "runtime_source_release_ref_mismatch" in blocked["blockers"]
    assert "runtime_source_release_models_embedded" in blocked["blockers"]
    assert "runtime_source_thin_contract_models_not_externalized" in blocked["blockers"]


def test_runtime_manifest_requires_digest_pinned_source_and_carrier() -> None:
    manifest = build_runtime_bundle_manifest(
        source_release_image_ref=SOURCE_REF,
        carrier_image_ref=CARRIER_REF,
        archive_sha256="a" * 64,
        archive_size_bytes=1234,
        healthcheck_argv=(
            ("/opt/gr00t-venv/bin/python", "--version"),
            ("/opt/oscar-venv/bin/python", "--version"),
            ("/isaac-sim/python.sh", "-c", "import isaacsim"),
        ),
        generated_at="2026-07-15T12:00:00Z",
        gpu_driver_deferred_sonames=("libnvidia-ml.so.1", "libcuda.so.1"),
    )

    assert manifest["status"] == "complete"
    assert manifest["archive"]["format"] == "tar.gz"
    assert manifest["runtime_env"] == RUNTIME_CARRIER_ENV
    assert "isaac-sim" in manifest["archive"]["member_roots"]
    assert "opt/gr00t-venv" in manifest["archive"]["member_roots"]
    assert "opt/runpod-serverless-venv" in manifest["archive"]["member_roots"]
    assert manifest["healthcheck_argv"][-1][0] == "/isaac-sim/python.sh"
    assert manifest["gpu_driver_deferred_sonames"] == [
        "libcuda.so.1",
        "libnvidia-ml.so.1",
    ]
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

    blocked_driver = build_runtime_bundle_manifest(
        source_release_image_ref=SOURCE_REF,
        carrier_image_ref=CARRIER_REF,
        archive_sha256="a" * 64,
        archive_size_bytes=1234,
        healthcheck_argv=(("/opt/gr00t-venv/bin/python", "--version"),),
        generated_at="2026-07-15T12:00:00Z",
        gpu_driver_deferred_sonames=("libcuda_runtime.so.1",),
    )
    assert blocked_driver["status"] == "blocked"
    assert "runtime_gpu_driver_deferred_soname_invalid" in blocked_driver["blockers"]


@pytest.mark.parametrize(
    "soname",
    (
        "libcuda.so.1",
        "libnvidia-ml.so.1",
        "libnvidia-ptxjitcompiler.so.1",
        "libnvcuvid.so.1",
        "libnvoptix.so.1",
        "libGLX_nvidia.so.0",
    ),
)
def test_nvidia_driver_soname_allowlist_accepts_only_host_injected_families(
    soname: str,
) -> None:
    assert is_nvidia_driver_soname(soname) is True


@pytest.mark.parametrize(
    "soname",
    (
        "libcuda_runtime.so.1",
        "libcudart.so.12",
        "libOpenCL.so.1",
        "/usr/lib/libcuda.so.1",
        "libcuda.so.1\nunsafe",
    ),
)
def test_nvidia_driver_soname_allowlist_rejects_carrier_and_unsafe_libraries(
    soname: str,
) -> None:
    assert is_nvidia_driver_soname(soname) is False


def test_carrier_volume_admission_binds_runtime_models_s3_and_volume() -> None:
    verified = verify_carrier_volume_admission(_admission(), expected_carrier_image_ref=CARRIER_REF)

    assert verified["status"] == "verified"
    assert verified["network_volume_id"] == "volume123"
    assert verified["data_center_id"] == "EUR-IS-1"
    assert verified["runtime_archive_sha256"] == "3" * 64
    assert verified["model_manifest_sha256"] == "5" * 64
    assert verified["model_manifest_digest"] == "sha256:" + "7" * 64
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


def test_carrier_volume_admission_migrates_v2_and_requires_v3_content_digest() -> None:
    legacy = _admission()
    legacy["schema_version"] = LEGACY_CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION
    legacy["model_cache"].pop("manifest_digest")

    verified = verify_carrier_volume_admission(legacy, expected_carrier_image_ref=CARRIER_REF)
    assert verified["status"] == "verified"
    assert verified["schema_version"] == LEGACY_CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION
    assert verified["model_manifest_digest"] is None
    assert verified["requires_external_model_manifest_digest_binding"] is True

    missing = _admission()
    missing["model_cache"].pop("manifest_digest")
    blocked = verify_carrier_volume_admission(missing, expected_carrier_image_ref=CARRIER_REF)
    assert blocked["status"] == "blocked"
    assert "carrier_model_cache_content_digest_missing_from_v3" in blocked["blockers"]

    malformed = _admission()
    malformed["model_cache"]["manifest_digest"] = "sha256:bad"
    blocked = verify_carrier_volume_admission(malformed, expected_carrier_image_ref=CARRIER_REF)
    assert blocked["status"] == "blocked"
    assert "carrier_model_cache_content_digest_invalid" in blocked["blockers"]


def test_runtime_bootstrap_is_hash_gated_path_safe_and_observable() -> None:
    script = runtime_bootstrap_shell_prefix()

    assert "BLUEPRINT_RUNPOD_CARRIER_RUNTIME_BOOTSTRAP_STARTED" in script
    assert "runtime_manifest_sha256_mismatch" in script
    assert "runtime_archive_sha256_mismatch" in script
    assert "runtime_manifest_env_mismatch" in script
    assert 'export PYTHONPATH=/opt/wbc:/opt/OSCAR' in script
    assert (
        "export LD_LIBRARY_PATH=/opt/wbc/gear_sonic_deploy/thirdparty_runtime/lib:"
        "/opt/onnxruntime/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
        in script
    )
    assert "runtime_archive_member_outside_allowlist" in script
    assert "runtime_archive_link_outside_allowlist" in script
    assert 'archive.extractall(path="/", filter=lambda member, _path: member)' in script
    assert "model_cache_manifest_sha256_mismatch" in script
    assert "model_cache_declared_file_sha256_mismatch" in script
    assert "all_declared_model_file_sha256_verified" in script
    assert "runtime_wbc_dynamic_linkage_failed" in script
    assert "runtime_gpu_driver_soname_unresolved" in script
    assert "ctypes.CDLL(soname)" in script
    assert "gpu_driver_deferred_sonames_resolved" in script
    assert 'manifest.get("gpu_driver_deferred_sonames", [])' in script
    assert "/opt/onnxruntime/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu" in script
    assert "/usr/local/nvidia/lib:/usr/local/nvidia/lib64" in script
    assert "export BLUEPRINT_GROOT_OSCAR_OSCAR_REPO=/opt/OSCAR" in script
    assert "export BLUEPRINT_ISAAC_PYTHON=/isaac-sim/python.sh" in script
    assert "BLUEPRINT_OSCAR_WAM_CHECKPOINT" in script
    assert "runpod_carrier_runtime_bootstrap_blocked.zip" in script
    assert 'or os.environ.get("BLUEPRINT_ARTIFACT_OUTPUT_URI")' in script
    assert 'export BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT="/opt/gr00t"' in script
    assert 'export BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT="/opt/wbc"' in script
    assert script.index("BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT") < script.index(
        "BLUEPRINT_RUNPOD_CARRIER_RUNTIME_BOOTSTRAP_READY"
    )
    assert "BLUEPRINT_RUNPOD_CARRIER_RUNTIME_BOOTSTRAP_READY" in script
