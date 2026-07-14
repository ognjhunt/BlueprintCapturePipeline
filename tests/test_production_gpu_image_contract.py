from blueprint_pipeline.production_gpu_image_contract import build_image_serving_contract


IMAGE = "docker.io/blueprint/worker@sha256:" + "a" * 64


def test_current_large_release_is_active_worker_only() -> None:
    result = build_image_serving_contract(
        {
            "resolved_digest_ref": IMAGE,
            "total_compressed_size_bytes": 47_101_357_226,
            "largest_layer_size_bytes": 14_083_497_680,
        },
        expected_image_ref=IMAGE,
        models_externalized_with_immutable_manifest=False,
    )

    assert result["status"] == "active_worker_only"
    assert result["scale_to_zero_eligible"] is False
    assert result["customer_serving_mode"] == "preloaded_active_worker_warm_pool"
    assert result["scale_to_zero_checks"]["total_compressed_budget"] is False
    assert result["scale_to_zero_checks"]["largest_layer_budget"] is False


def test_small_externalized_release_is_only_a_scale_to_zero_candidate() -> None:
    result = build_image_serving_contract(
        {
            "resolved_digest_ref": IMAGE,
            "total_compressed_size_bytes": 6 * 1024**3,
            "largest_layer_size_bytes": 1024**3,
        },
        expected_image_ref=IMAGE,
        models_externalized_with_immutable_manifest=True,
    )

    assert result["status"] == "scale_to_zero_candidate"
    assert result["scale_to_zero_eligible"] is True
    assert result["claim_boundary"]["size_budget_is_not_live_startup_proof"] is True


def test_registry_release_mismatch_blocks_all_serving_classification() -> None:
    result = build_image_serving_contract(
        {
            "resolved_digest_ref": "docker.io/blueprint/worker@sha256:" + "b" * 64,
            "total_compressed_size_bytes": 1,
            "largest_layer_size_bytes": 1,
        },
        expected_image_ref=IMAGE,
        models_externalized_with_immutable_manifest=True,
    )

    assert result["status"] == "blocked"
    assert "registry_diagnostic_exact_release_mismatch" in result["blockers"]
