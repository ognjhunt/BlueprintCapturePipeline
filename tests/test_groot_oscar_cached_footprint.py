from blueprint_pipeline.groot_oscar_cached_footprint import (
    build_cached_footprint_audit,
)


REF = "registry.example/blueprint/release@sha256:" + "a" * 64


def _models(size: int = 14 * 1024**3) -> dict:
    return {
        "status": "passed",
        "model_manifest_digest": "sha256:" + "b" * 64,
        "verified_size_bytes": size,
        "checks": {"models_cached_offline": True},
    }


def test_cached_worker_target_uses_local_image_size_plus_verified_models() -> None:
    result = build_cached_footprint_audit(
        image_evidence={
            "resolved_digest_ref": REF,
            "local_uncompressed_size_bytes": 14 * 1024**3,
        },
        model_cache_verification=_models(),
        expected_release_ref=REF,
    )
    assert result["status"] == "target_met"
    assert result["total_cached_worker_footprint_bytes"] == 28 * 1024**3
    assert result["target_met"] is True


def test_cached_worker_audit_reports_measured_above_target() -> None:
    result = build_cached_footprint_audit(
        image_evidence={
            "resolved_digest_ref": REF,
            "local_uncompressed_size_bytes": 20 * 1024**3,
        },
        model_cache_verification=_models(),
        expected_release_ref=REF,
    )
    assert result["status"] == "measured_above_target"
    assert result["blockers"] == []
    assert result["target_met"] is False


def test_cached_worker_audit_refuses_registry_compressed_size_substitution() -> None:
    result = build_cached_footprint_audit(
        image_evidence={
            "resolved_digest_ref": REF,
            "total_compressed_size_bytes": 10 * 1024**3,
        },
        model_cache_verification=_models(),
        expected_release_ref=REF,
    )
    assert result["status"] == "blocked"
    assert "local_uncompressed_worker_image_size_missing" in result["blockers"]


def test_cached_worker_audit_requires_exact_release_and_verified_models() -> None:
    result = build_cached_footprint_audit(
        image_evidence={
            "resolved_digest_ref": "registry.example/other@sha256:" + "c" * 64,
            "local_uncompressed_size_bytes": 10,
        },
        model_cache_verification={
            "status": "blocked",
            "verified_size_bytes": 10,
            "checks": {"models_cached_offline": False},
        },
        expected_release_ref=REF,
    )
    assert result["status"] == "blocked"
    assert "cached_worker_release_ref_mismatch" in result["blockers"]
    assert "external_model_cache_not_verified" in result["blockers"]
