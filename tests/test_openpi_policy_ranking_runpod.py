import json
import time
from pathlib import Path

from blueprint_pipeline.openpi_policy_ranking_runpod import (
    INPUT_SECRET_URL_ENV,
    OUTPUT_SECRET_PUT_URL_ENV,
    build_openpi_policy_ranking_provider_request,
    run_openpi_policy_ranking_campaign,
    shape_openpi_policy_ranking_request_without_mutation,
)


def _inputs():
    release = {
        "schema_version": "openpi_policy_ranking_gpu_release.v1",
        "status": "passed",
        "source_commit": "a" * 40,
        "resolved_digest_ref": "ghcr.io/example/openpi@sha256:" + "b" * 64,
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
        "observed_at_epoch": time.time(),
        "provider_inventory_verified_zero": True,
        "single_gpu_available": True,
        "gpu_memory_bytes": 48 * 1024**3,
        "gpu_type_id": "NVIDIA A40",
        "on_demand_price_usd_per_hour": 0.44,
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


def test_openpi_request_shape_is_redacted_and_one_gpu(tmp_path: Path) -> None:
    release, bundle, preflight, spend = _inputs()
    prepared = build_openpi_policy_ranking_provider_request(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit="a" * 40,
        job_id="openpi-test",
    )
    input_url = "https://storage.example/input?x-goog-signature=input-secret"
    output_url = "https://storage.example/output?x-goog-signature=output-secret"
    output = tmp_path / "adapter.json"
    result = shape_openpi_policy_ranking_request_without_mutation(
        prepared=prepared,
        output_path=output,
        input_secret_url=input_url,
        output_secret_put_url=output_url,
        pod_name="blueprint-openpi-ranking-test",
    )
    assert result["status"] == "dry_run_ready"
    body = result["runpod_request"]["on_demand_pod"]["body"]
    assert body["gpuCount"] == 1
    assert body["containerDiskInGb"] == 100
    assert body["dockerEntrypoint"][-2:] == [
        "blueprint_pipeline.openpi_policy_ranking_gpu_bootstrap",
        "run",
    ]
    assert body["env"][INPUT_SECRET_URL_ENV] == "<redacted:secret-env>"
    assert body["env"][OUTPUT_SECRET_PUT_URL_ENV] == "<redacted:secret-env>"
    persisted = output.read_text(encoding="utf-8")
    assert input_url not in persisted
    assert output_url not in persisted
    assert "input-secret" not in persisted
    assert "output-secret" not in persisted
    request = json.loads((tmp_path / "openpi_provider_launch_request.json").read_text())
    assert input_url not in json.dumps(request)


def test_openpi_campaign_dry_run_stays_mutation_free(tmp_path: Path) -> None:
    release, bundle, preflight, _spend = _inputs()
    release_path = tmp_path / "release.json"
    bundle_path = tmp_path / "bundle.json"
    preflight_path = tmp_path / "preflight.json"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    result = run_openpi_policy_ranking_campaign(
        release_evidence=release_path,
        input_bundle_receipt=bundle_path,
        preflight_bundle=preflight_path,
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "adapter.json",
        input_secret_url_file=tmp_path / "unused-input-url",
        output_secret_put_url_file=tmp_path / "unused-output-url",
        pod_name="blueprint-groot-oscar-canary-openpi-ranking-test",
        expected_source_commit="a" * 40,
        execute=False,
        hard_ttl_seconds=3600,
        max_spend_usd=1.0,
    )

    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0
    assert result["watchdog_process_started"] is False
    assert result["budget_reservation_created"] is False
