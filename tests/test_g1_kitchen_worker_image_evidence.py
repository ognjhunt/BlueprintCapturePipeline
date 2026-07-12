from __future__ import annotations

from blueprint_pipeline.g1_kitchen_worker_image_evidence import (
    assemble_worker_image_runtime_evidence,
    validate_worker_image_runtime_evidence,
)


def _evidence() -> dict:
    digest = "sha256:" + "a" * 64
    return {
        "schema_version": "g1_kitchen_worker_image_runtime_evidence.v1",
        "status": "passed",
        "image_digest": digest,
        "source_commit": "abc1234",
        "source_dirty_patch_sha256": "b" * 64,
        "runtime_metadata": {
            "image_family": "isaac-eval-worker",
            "simulator_family": "isaac_sim",
            "simulator_major_version": 6,
            "source_commit": "abc1234",
            "source_dirty_patch_sha256": "b" * 64,
            "blueprint_pipeline_imported": True,
            "configured_g1_asset_binding_valid": True,
            "configured_g1_usd_exists": False,
            "g1_asset_resolution_deferred_to_runtime": True,
            "build_time_healthcheck_passed": True,
        },
        "fast_canary": {
            "status": "passed",
            "image_digest": digest,
            "provider_allocation_id": "pod-1",
            "launch_nonce": "nonce-1",
        },
        "review_canary": {
            "status": "passed",
            "image_digest": digest,
            "provider_allocation_id": "pod-1",
            "launch_nonce": "nonce-1",
            "width": 640,
            "height": 480,
        },
        "teardown": {"api_confirmed": True, "terminal_state": "not_found"},
        "final_inventory": {"api_confirmed": True, "live_resource_count": 0},
    }


def test_worker_image_evidence_requires_exact_source_digest_and_same_allocation() -> None:
    evidence = _evidence()
    assert (
        validate_worker_image_runtime_evidence(
            evidence,
            expected_image_digest=evidence["image_digest"],
            expected_source_commit="abc1234",
            expected_dirty_patch_sha256="b" * 64,
        )["status"]
        == "passed"
    )
    evidence["review_canary"]["provider_allocation_id"] = "pod-2"
    result = validate_worker_image_runtime_evidence(evidence)
    assert result["status"] == "blocked"
    assert (
        "worker_image_evidence_same_allocation_provider_allocation_id_mismatch"
        in result["blockers"]
    )


def test_worker_image_evidence_unknown_inventory_and_old_source_fail_closed() -> None:
    evidence = _evidence()
    evidence["final_inventory"]["api_confirmed"] = False
    result = validate_worker_image_runtime_evidence(
        evidence,
        expected_source_commit="different",
    )
    assert "worker_image_evidence_final_inventory_not_zero" in result["blockers"]
    assert "worker_image_evidence_source_commit_mismatch" in result["blockers"]


def test_assembler_derives_pass_only_from_all_exact_live_inputs() -> None:
    expected = _evidence()
    result = assemble_worker_image_runtime_evidence(
        image_digest=expected["image_digest"],
        source_commit=expected["source_commit"],
        source_dirty_patch_sha256=expected["source_dirty_patch_sha256"],
        build_healthcheck={
            "status": "passed",
            "runtime_metadata": {
                **expected["runtime_metadata"],
            },
        },
        fast_canary=expected["fast_canary"],
        review_canary=expected["review_canary"],
        teardown=expected["teardown"],
        final_inventory=expected["final_inventory"],
    )
    assert result["status"] == "passed"
    assert result["blockers"] == []
