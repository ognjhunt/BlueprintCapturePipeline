from __future__ import annotations

import pytest

from blueprint_pipeline.groot_oscar_release_hardening import (
    DiskAdmission,
    build_regional_mirror_plan,
    build_layer_report,
    evaluate_release_slos,
    record_startup_milestone,
    syft_registry_scan_command,
    validate_provenance_digest,
    validate_registry_mirror_equivalence,
    validate_spdx_document,
)


DIGEST = "sha256:" + "a" * 64
REF = "docker.io/example/worker@" + DIGEST


def test_registry_scan_can_never_fall_back_to_docker_daemon():
    command = syft_registry_scan_command(REF, "sbom.spdx.json")
    assert command[1] == "registry:" + REF
    assert "docker:" not in " ".join(command)
    with pytest.raises(ValueError):
        syft_registry_scan_command("docker.io/example/worker:latest", "x.json")


def test_disk_admission_accounts_for_build_scan_and_reserve():
    blocked = DiskAdmission(1, 100, 200, reserve_bytes=10).evidence()
    assert blocked["status"] == "blocked"
    passed = DiskAdmission(1000, 100, 200, reserve_bytes=10).evidence()
    assert passed["status"] == "passed"
    assert passed["scan_source"] == "registry_digest"


def test_sbom_and_provenance_are_fail_closed():
    assert validate_spdx_document({})
    assert (
        validate_spdx_document(
            {"spdxVersion": "SPDX-2.3", "documentNamespace": "urn:x", "packages": [{}]}
        )
        == []
    )
    assert validate_provenance_digest({"subject": DIGEST}, DIGEST) == []
    assert validate_provenance_digest({"subject": "other"}, DIGEST)


def test_registry_mirror_requires_per_platform_digest_equivalence():
    source = {"manifest_digest": DIGEST, "platform_digests": {"linux/amd64": DIGEST}}
    same = {"manifest_digest": "sha256:" + "b" * 64, "platform_digests": {"linux/amd64": DIGEST}}
    assert validate_registry_mirror_equivalence(source, same)["status"] == "passed"
    same["platform_digests"]["linux/amd64"] = "sha256:" + "c" * 64
    assert validate_registry_mirror_equivalence(source, same)["status"] == "blocked"


def test_regional_mirror_plan_is_digest_bound_and_costed_without_compute():
    plan = build_regional_mirror_plan(
        source_digest_ref=REF,
        project_id="project-1",
        repository="workers",
        locations=("us-east1", "us-central1"),
        compressed_size_bytes=46 * 1024**3,
        storage_usd_per_gb_month=0.10,
    )
    assert plan["status"] == "planned_not_executed"
    assert plan["estimated_storage_usd_per_month"] == 9.2
    assert plan["idle_paid_compute_required"] is False
    assert all(item["copy_command"][2] == REF for item in plan["destinations"])
    with pytest.raises(ValueError):
        build_regional_mirror_plan(
            source_digest_ref="example:latest",
            project_id="p",
            repository="r",
            locations=("us-east1",),
            compressed_size_bytes=1,
            storage_usd_per_gb_month=0.1,
        )


def test_layer_report_orders_size_and_preserves_offline_rule():
    report = build_layer_report(
        [
            {"digest": "a", "size_bytes": 10, "created_by": "small"},
            {"digest": "b", "size_bytes": 100, "created_by": "large"},
        ]
    )
    assert report["largest_layers"][0]["digest"] == "b"
    assert report["optimization_rules"]["offline_execution_required"] is True


def test_release_slo_requires_complete_ordered_timing():
    names = (
        "vm_allocation",
        "driver_ready",
        "container_runtime_ready",
        "image_pull",
        "container_start",
        "health",
        "isaac_startup",
        "policy_ready",
        "first_simulator_step",
        "first_learned_action",
        "first_frame",
        "artifact_upload",
    )
    timing = {name: index * 10 for index, name in enumerate(names)}
    assert evaluate_release_slos(timing)["status"] == "passed"
    timing["policy_ready"] = 400
    assert "cached_worker_ready_slo_missed" in evaluate_release_slos(timing)["blockers"]


def test_startup_milestones_are_append_only_and_monotonic():
    timing = record_startup_milestone({}, "vm_allocation", 2.0)
    timing = record_startup_milestone(timing, "driver_ready", 5.0)
    assert timing == {"vm_allocation": 2.0, "driver_ready": 5.0}
    with pytest.raises(ValueError, match="duplicate"):
        record_startup_milestone(timing, "driver_ready", 6.0)
    with pytest.raises(ValueError, match="out_of_order"):
        record_startup_milestone(timing, "image_pull", 7.0)
