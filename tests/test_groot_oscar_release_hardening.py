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
    assert report["status"] == "passed"
    assert report["optimization_rules"]["offline_execution_required"] is True


def test_layer_report_accepts_real_registry_diagnostic_size_field():
    report = build_layer_report(
        [
            {"digest": "sha256:a", "size": 10, "created_by": "small"},
            {"digest": "sha256:b", "size": 100, "created_by": "large"},
        ]
    )
    assert report["status"] == "passed"
    assert report["total_compressed_size_bytes"] == 110
    assert report["largest_layer_size_bytes"] == 100
    assert report["largest_layers"][0]["digest"] == "sha256:b"


def test_layer_report_blocks_release_closure_growth():
    total_blocked = build_layer_report(
        [{"digest": "a", "size_bytes": 101, "created_by": "large"}],
        max_total_compressed_bytes=100,
        max_layer_bytes=200,
    )
    assert total_blocked["status"] == "blocked"
    assert "image_total_compressed_size_budget_exceeded" in total_blocked["blockers"]

    layer_blocked = build_layer_report(
        [{"digest": "a", "size_bytes": 101, "created_by": "large"}],
        max_total_compressed_bytes=200,
        max_layer_bytes=100,
    )
    assert "image_largest_layer_size_budget_exceeded" in layer_blocked["blockers"]


def test_layer_report_attributes_the_measured_runtime_duplication():
    report = build_layer_report(
        [
            {"digest": "base", "size_bytes": 10, "created_by": "COPY . /isaac-sim/"},
            {
                "digest": "oscar",
                "size_bytes": 4,
                "created_by": "uv venv /opt/oscar-venv --python 3.10",
            },
            {
                "digest": "wbc",
                "size_bytes": 7,
                "created_by": "scripts/install_deps.sh && just build",
            },
            {
                "digest": "groot",
                "size_bytes": 9,
                "created_by": "uv venv /opt/gr00t-venv --python 3.10",
            },
            {
                "digest": "models",
                "size_bytes": 14,
                "created_by": "snapshot_download SONIC_CHECKPOINT_REPO",
            },
        ]
    )
    assert report["compressed_bytes_by_build_role"] == {
        "isaac_sim_base": 10,
        "oscar_python_cuda_runtime": 4,
        "wbc_build_and_cuda_toolchain": 7,
        "groot_python_cuda_runtime": 9,
        "sealed_model_checkpoints": 14,
        "other": 0,
    }
    candidates = {row["role"]: row for row in report["measured_optimization_candidates"]}
    assert candidates["wbc_build_and_cuda_toolchain"]["runtime_gpu_abi_test_required"] is True


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
