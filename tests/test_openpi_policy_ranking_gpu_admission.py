from blueprint_pipeline.openpi_policy_ranking_gpu_admission import (
    build_openpi_policy_ranking_gpu_admission,
    collect_openpi_policy_ranking_runpod_preflight,
    collect_openpi_policy_ranking_vast_preflight,
)
from blueprint_pipeline.new_site_diagnostic_canary_gpu import (
    INPUT_RECEIPT_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION,
)
from blueprint_pipeline.openpi_current_reference_gpu_bundle import (
    INPUT_RECEIPT_SCHEMA_VERSION as CURRENT_REFERENCE_RECEIPT_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION as CURRENT_REFERENCE_SCHEMA_VERSION,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


def _inputs():
    release = {
        "schema_version": "openpi_policy_ranking_gpu_release.v1",
        "status": "passed",
        "source_commit": "a" * 40,
        "resolved_digest_ref": "ghcr.io/ognjhunt/blueprint-openpi-policy-ranking@sha256:"
        + "b" * 64,
        "runnable_platform": "linux/amd64",
        "openpi_revision": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
        "menagerie_revision": "71f066ad0be9cd271f7ed58c030243ef157af9f4",
        "checkpoint_bytes_embedded": 0,
        "interiorgs_assets_embedded": False,
    }
    bundle = {
        "schema_version": "openpi_policy_ranking_gpu_input_bundle_receipt.v1",
        "bundle_sha256": "c" * 64,
        "manifest": {
            "schema_version": "openpi_policy_ranking_gpu_input_bundle.v2",
            "raw_3dgs_included": False,
            "redistribution_authorized": False,
            "purpose": "private_internal_noncommercial_research_gpu_execution",
            "background_sha256": "d" * 64,
            "scene_count": 2,
            "scenes": [
                {
                    "source_scene_id": "captured",
                    "source_scene_kind": "captured_3dgs",
                    "background_sha256": "d" * 64,
                },
                {
                    "source_scene_id": "warehouse",
                    "source_scene_kind": "controlled_nvidia_usd",
                    "background_sha256": "e" * 64,
                },
            ],
        },
    }
    preflight = {
        "schema_version": "openpi_policy_ranking_runpod_preflight.v1",
        "status": "verified",
        "provider": "runpod",
        "provider_api_verified": True,
        "observed_at_epoch": 1000.0,
        "provider_inventory_verified_zero": True,
        "single_gpu_available": True,
        "gpu_memory_bytes": 24 * 1024**3,
        "gpu_type_id": "NVIDIA RTX 4090",
        "on_demand_price_usd_per_hour": 0.5,
        "container_disk_bytes": 100 * 1024**3,
    }
    spend = {
        "paid_mutation_authorized": True,
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
        "watchdog_armed_before_allocation": True,
        "hard_ttl_seconds": 3600,
        "max_spend_usd": 1.0,
        "physical_robot_endpoint_access_allowed": False,
    }
    return release, bundle, preflight, spend


def test_openpi_gpu_admission_passes_exact_contract() -> None:
    release, bundle, preflight, spend = _inputs()
    result = build_openpi_policy_ranking_gpu_admission(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit="a" * 40,
        observed_now_epoch=1001.0,
    )
    assert result["status"] == "admitted"
    assert result["shared_paid_lane_admission"]["status"] == "admitted"


def test_openpi_gpu_admission_accepts_one_arm_label_free_canary_receipt() -> None:
    release, _bundle, preflight, spend = _inputs()
    manifest = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "experiment_id": "diagnostic_v6",
        "protocol_filename": "protocol.json",
        "protocol_file_sha256": "f" * 64,
        "protocol_sha256": "1" * 64,
        "arm_id": "skeleton_only",
        "scene_id": "interiorgs_0787",
        "task_instruction": "Pick up the spray can and place it inside the marked tray.",
        "policy_id": "pi05_droid_jointpos_polaris",
        "variant": "center",
        "background_filename": "captured_site_background.png",
        "background_sha256": "d" * 64,
        "background_size_bytes": 123,
        "raw_3dgs_included": False,
        "redistribution_authorized": False,
        "label_free": True,
        "purpose": "private_internal_noncommercial_new_site_diagnostic_canary",
        "initial_observation_source": "mujoco_hybrid_camera_render",
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    bundle = {
        "schema_version": INPUT_RECEIPT_SCHEMA_VERSION,
        "bundle_sha256": "c" * 64,
        "manifest": manifest,
    }

    result = build_openpi_policy_ranking_gpu_admission(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit="a" * 40,
        observed_now_epoch=1001.0,
    )

    assert result["status"] == "admitted"
    assert result["probe_kind"] == "new-site-diagnostic-canary"
    assert result["execution_mode"] == "new_site_diagnostic_canary"


def test_openpi_gpu_admission_accepts_exact_current_reference_source_overlay() -> None:
    release, _bundle, preflight, spend = _inputs()
    runtime_commit = "d" * 40
    manifest = {
        "schema_version": CURRENT_REFERENCE_SCHEMA_VERSION,
        "purpose": "label_free_current_reference_real_policy_identity_canary",
        "runtime_source": {
            "repository": "https://github.com/ognjhunt/BlueprintCapturePipeline",
            "commit": runtime_commit,
            "archive_url": (
                "https://codeload.github.com/ognjhunt/BlueprintCapturePipeline/tar.gz/"
                + runtime_commit
            ),
            "archive_sha256": "e" * 64,
            "overlay_required": True,
        },
        "image_source_commit": release["source_commit"],
        "policy_ids": ["pi05_droid", "pi0_droid", "pi0_fast_droid"],
        "requests_per_policy": 1,
        "raw_3dgs_included": False,
        "redistribution_authorized": False,
        "label_free": True,
        "confirmation_eligible": False,
        "physical_outcome_included": False,
        "checkpoint_weights_included": False,
        "files": [
            {"path": f"file-{index}", "sha256": "f" * 64, "size_bytes": index}
            for index in range(11)
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    bundle = {
        "schema_version": CURRENT_REFERENCE_RECEIPT_SCHEMA_VERSION,
        "bundle_sha256": "c" * 64,
        "manifest": manifest,
    }
    result = build_openpi_policy_ranking_gpu_admission(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit=runtime_commit,
        observed_now_epoch=1001.0,
    )
    assert result["status"] == "admitted"
    assert result["probe_kind"] == "openpi-current-reference-policy-canary"
    assert result["runtime_source_overlay_required"] is True
    assert result["source_commit"] == "a" * 40
    assert result["runtime_source_commit"] == runtime_commit

    requery_manifest = {
        **manifest,
        "purpose": "label_free_current_reference_same_policy_requery",
        "policy_ids": ["pi05_droid"],
        "observation_schema": "openpi_current_reference_generated_observation.v1",
        "same_candidate_policy_id": "pi05_droid",
    }
    requery_manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in requery_manifest.items() if key != "manifest_sha256"}
    )
    requery = build_openpi_policy_ranking_gpu_admission(
        release=release,
        input_bundle={**bundle, "manifest": requery_manifest},
        preflight=preflight,
        spend=spend,
        expected_source_commit=runtime_commit,
        observed_now_epoch=1001.0,
    )
    assert requery["status"] == "admitted", requery["blockers"]

    manifest["runtime_source"]["archive_url"] = (
        "https://codeload.github.com/ognjhunt/BlueprintCapturePipeline/tar.gz/" + "9" * 40
    )
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    blocked = build_openpi_policy_ranking_gpu_admission(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit=runtime_commit,
        observed_now_epoch=1001.0,
    )
    assert "openpi_gpu_runtime_source_archive_invalid" in blocked["blockers"]


def test_openpi_gpu_admission_blocks_rights_robot_and_budget_regressions() -> None:
    release, bundle, preflight, spend = _inputs()
    bundle["manifest"]["raw_3dgs_included"] = True
    spend["physical_robot_endpoint_access_allowed"] = True
    spend["max_spend_usd"] = 0.1
    result = build_openpi_policy_ranking_gpu_admission(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit="a" * 40,
        observed_now_epoch=1001.0,
    )
    assert result["status"] == "blocked"
    assert "openpi_gpu_input_bundle_contains_raw_3dgs" in result["blockers"]
    assert "openpi_gpu_physical_robot_endpoint_not_forbidden" in result["blockers"]
    assert "openpi_gpu_ttl_cost_exceeds_max_spend" in result["blockers"]


def test_openpi_gpu_admission_accepts_four_hour_two_scene_watchdog_window() -> None:
    release, bundle, preflight, spend = _inputs()
    spend["hard_ttl_seconds"] = 14_400
    spend["max_spend_usd"] = 2.0
    admitted = build_openpi_policy_ranking_gpu_admission(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit="a" * 40,
        observed_now_epoch=1001.0,
    )
    assert admitted["status"] == "admitted"

    spend["hard_ttl_seconds"] = 14_401
    blocked = build_openpi_policy_ranking_gpu_admission(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit="a" * 40,
        observed_now_epoch=1001.0,
    )
    assert "openpi_gpu_ttl_invalid" in blocked["blockers"]


def test_openpi_preflight_selects_verified_capacity_without_mutation() -> None:
    inventories = []

    def inventory(prefix):
        inventories.append(prefix)
        return {"api_confirmed": True, "live_resource_count": 0, "resources": []}

    result = collect_openpi_policy_ranking_runpod_preflight(
        name_prefix="blueprint-openpi-ranking-",
        gpu_type_ids=("NVIDIA A40",),
        container_disk_bytes=100 * 1024**3,
        capacity_probe=lambda request: {
            "status": "available",
            "viable_gpu_types": [
                {
                    "gpu_type_id": request["gpuTypeIds"][0],
                    "memory_in_gb": 48,
                    "single_gpu_offer_available": True,
                    "on_demand_price_usd_per_hour": 0.44,
                }
            ],
        },
        inventory_probe=inventory,
        clock=lambda: 1000.0,
    )
    assert result["status"] == "verified"
    assert result["gpu_type_id"] == "NVIDIA A40"
    assert result["provider_mutations_performed"] == 0
    assert inventories == ["blueprint-openpi-ranking-", ""]


def test_openpi_vast_preflight_reserves_frozen_rate_ceiling() -> None:
    result = collect_openpi_policy_ranking_vast_preflight(
        name_prefix="blueprint-openpi-ranking-",
        container_disk_bytes=100 * 1024**3,
        max_hourly_rate_usd=0.75,
        capacity_probe=lambda request: {
            "status": "available",
            "selected_offer": {
                "ask_contract_id": 123,
                "gpu_type_id": "A40",
                "gpu_ram_mb": 46_068,
                "num_gpus": 1,
                "on_demand_price_usd_per_hour": 0.28,
            },
            "selection_policy": request,
        },
        inventory_probe=lambda _prefix: {
            "api_confirmed": True,
            "live_resource_count": 0,
            "resources": [],
        },
        clock=lambda: 1000.0,
    )
    assert result["status"] == "verified"
    assert result["provider"] == "vast"
    assert result["selected_offer_price_usd_per_hour"] == 0.28
    assert result["on_demand_price_usd_per_hour"] == 0.75
    release, bundle, _runpod_preflight, spend = _inputs()
    spend["hard_ttl_seconds"] = 14_400
    spend["max_spend_usd"] = 3.0
    admission = build_openpi_policy_ranking_gpu_admission(
        release=release,
        input_bundle=bundle,
        preflight=result,
        spend=spend,
        expected_source_commit="a" * 40,
        observed_now_epoch=1001.0,
    )
    assert admission["status"] == "admitted"
    assert admission["provider_resource_class"] == "gpu_render"


def test_openpi_vast_preflight_allows_one_existing_resource_under_two_gpu_ceiling() -> None:
    result = collect_openpi_policy_ranking_vast_preflight(
        name_prefix="blueprint-openpi-ranking-",
        container_disk_bytes=100 * 1024**3,
        max_existing_live_resources=1,
        capacity_probe=lambda _request: {
            "status": "available",
            "selected_offer": {
                "ask_contract_id": 123,
                "gpu_type_id": "A40",
                "gpu_ram_mb": 46_068,
                "num_gpus": 1,
                "on_demand_price_usd_per_hour": 0.28,
            },
        },
        inventory_probe=lambda prefix: {
            "api_confirmed": True,
            "live_resource_count": 0 if prefix else 1,
            "resources": [],
        },
        clock=lambda: 1000.0,
    )

    assert result["status"] == "verified"
    assert result["provider_inventory_verified_zero"] is False
    assert result["provider_inventory_within_concurrency_limit"] is True
    assert result["maximum_existing_live_resources"] == 1


def test_openpi_vast_preflight_blocks_two_existing_resources_under_two_gpu_ceiling() -> None:
    result = collect_openpi_policy_ranking_vast_preflight(
        name_prefix="blueprint-openpi-ranking-",
        container_disk_bytes=100 * 1024**3,
        max_existing_live_resources=1,
        capacity_probe=lambda _request: {
            "status": "available",
            "selected_offer": {
                "ask_contract_id": 123,
                "gpu_type_id": "A40",
                "gpu_ram_mb": 46_068,
                "num_gpus": 1,
                "on_demand_price_usd_per_hour": 0.28,
            },
        },
        inventory_probe=lambda _prefix: {
            "api_confirmed": True,
            "live_resource_count": 2,
            "resources": [],
        },
        clock=lambda: 1000.0,
    )

    assert result["status"] == "blocked"
    assert result["provider_inventory_within_concurrency_limit"] is False
    assert result["blockers"] == ["openpi_gpu_preflight_billable_inventory_exceeds_concurrency"]
