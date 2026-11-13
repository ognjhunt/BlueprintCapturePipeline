import json
import io
import time
import zipfile
import os
from pathlib import Path
from urllib.error import URLError

from blueprint_pipeline import openpi_policy_ranking_runpod as runpod_module
from blueprint_pipeline import paid_provider_lane_lease as lease_module
from blueprint_pipeline.new_site_diagnostic_canary_gpu import (
    INPUT_RECEIPT_SCHEMA_VERSION as CANARY_INPUT_RECEIPT_SCHEMA_VERSION,
)
from blueprint_pipeline.openpi_policy_ranking_runpod import (
    EXECUTION_MODE_ENV,
    GENERIC_OUTPUT_SECRET_URL_ENV,
    INPUT_SECRET_URL_ENV,
    INPUT_SHA256_ENV,
    OUTPUT_SECRET_PUT_URL_ENV,
    _build_vast_launch_request,
    _monitor_openpi_output_and_teardown,
    _validate_output_archive,
    run_openpi_policy_ranking_campaign,
)
from blueprint_pipeline.openpi_policy_ranking_gpu_bootstrap import POLICY_IDS
from blueprint_pipeline.openpi_policy_ranking_gpu_job import FROZEN_VARIANTS
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256



def _completed_output_archive() -> bytes:
    scene_rows = [
        {"scene_id": "captured", "scene_kind": "captured_3dgs"},
        {"scene_id": "warehouse", "scene_kind": "controlled_nvidia_usd"},
    ]
    policy_runs = []
    episode_payloads = {}
    for policy_id in POLICY_IDS:
        records = []
        for scene in scene_rows:
            for variant_id, _offset in FROZEN_VARIANTS:
                episode = {"status": "completed", "policy_id": policy_id}
                episode["manifest_sha256"] = canonical_sha256(episode)
                records.append(
                    {
                        "scene_id": scene["scene_id"],
                        "scene_kind": scene["scene_kind"],
                        "variant_id": variant_id,
                        "episode_manifest_sha256": episode["manifest_sha256"],
                    }
                )
                episode_payloads[
                    f"{policy_id}/{scene['scene_id']}/{variant_id}/franka_droid_closed_loop.json"
                ] = episode
        policy_runs.append(
            {"policy_id": policy_id, "status": "completed", "episode_records": records}
        )
    manifest = {
        "schema_version": "openpi_policy_ranking_gpu_job.v1",
        "status": "completed",
        "inputs": {"scenes": scene_rows, "policy_ids": list(POLICY_IDS)},
        "policy_runs": policy_runs,
        "rankings": {"captured": {"status": "completed"}, "warehouse": {"status": "completed"}},
        "blockers": [],
        "claim_boundary": {
            "site_specific_physical_success_proven": False,
            "physical_robot_endpoint_contacted": False,
            "physical_robot_operated": False,
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "openpi_policy_ranking_gpu_job.json", json.dumps(manifest)
        )
        for name, episode in episode_payloads.items():
            archive.writestr(name, json.dumps(episode))
    return buffer.getvalue()


def test_completed_output_archive_binds_all_24_episodes() -> None:
    result = _validate_output_archive(_completed_output_archive())
    assert result["status"] == "completed"
    assert result["campaign_status"] == "completed"
    assert result["episode_record_count"] == 24
    assert result["scene_ids"] == ["captured", "warehouse"]


def test_completed_canary_output_requires_individual_camera_media() -> None:
    manifest = {
        "schema_version": "new_site_diagnostic_canary_gpu.v1",
        "status": "completed",
        "arm_id": "skeleton_only",
        "protocol_sha256": "a" * 64,
        "canary": {
            "status": "passed",
            "label_free": True,
            "model_invoked": True,
            "freeze_bindings": {"scene_id": "interiorgs_0787"},
        },
        "claim_boundary": {
            "ranking_accuracy": False,
            "physical_success": False,
            "captured_site_transfer_validation": False,
            "phase_b_confirmation": False,
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("new_site_diagnostic_canary_gpu.json", json.dumps(manifest))
        archive.writestr("loop/query_000_external_skeleton.mp4", b"external")
        archive.writestr("loop/query_000_wrist_skeleton.mp4", b"wrist")

    result = _validate_output_archive(buffer.getvalue())

    assert result["status"] == "completed"
    assert result["execution_mode"] == "new_site_diagnostic_canary"
    assert result["episode_record_count"] == 0
    assert result["individual_camera_media_present"] == {
        "external": True,
        "wrist": True,
    }


def test_monitor_collects_output_then_proves_provider_and_budget_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    archive = _completed_output_archive()

    class Response:
        status = 200
        body = archive

    class Provider:
        def capacity_preflight(self, **_kwargs) -> dict:
            return {"available": True}

        def billable_inventory(self, *, name_prefix: str) -> dict:
            assert name_prefix == ""
            return {"api_confirmed": True, "live_resource_count": 0, "resources": []}

    monkeypatch.setattr(
        "blueprint_pipeline.openpi_policy_ranking_runpod.safe_http_request",
        lambda *_args, **_kwargs: Response(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.openpi_policy_ranking_runpod.terminate_canary_resources",
        lambda **_kwargs: {"provider_absence_confirmed": True},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.openpi_policy_ranking_runpod.write_owner_teardown_cancel_request",
        lambda **_kwargs: {"status": "requested"},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.openpi_policy_ranking_runpod._wait_for_watchdog_terminal",
        lambda _root: {
            "control_plane_terminal": True,
            "campaign_budget_settlement": {"status": "settled", "charged_usd": 0.1},
        },
    )

    result = _monitor_openpi_output_and_teardown(
        root=tmp_path,
        output_secret_get_url="https://storage.example/output?signature=secret",
        provider=Provider(),
        armed={"status": "armed"},
        pod_id="pod-openpi",
        provider_name="runpod",
        deadline_epoch=time.time() + 120,
        poll_interval_seconds=0.01,
    )

    assert result["status"] == "completed"
    assert result["provider_absence_confirmed"] is True
    assert result["control_plane_terminal"] is True
    assert result["continuing_spend"] is False
    assert (tmp_path / "openpi_policy_ranking_provider_output.zip").is_file()


def test_monitor_retries_transient_url_error_before_collecting_output(
    tmp_path: Path, monkeypatch
) -> None:
    archive = _completed_output_archive()
    attempts = 0

    class Response:
        status = 200
        body = archive

    class Provider:
        def billable_inventory(self, *, name_prefix: str) -> dict:
            assert name_prefix == ""
            return {"api_confirmed": True, "live_resource_count": 0, "resources": []}

    def request(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise URLError("synthetic transient DNS failure")
        return Response()

    monkeypatch.setattr(runpod_module, "safe_http_request", request)
    monkeypatch.setattr(
        runpod_module,
        "terminate_canary_resources",
        lambda **_kwargs: {"provider_absence_confirmed": True},
    )
    monkeypatch.setattr(
        runpod_module,
        "write_owner_teardown_cancel_request",
        lambda **_kwargs: {"status": "requested"},
    )
    monkeypatch.setattr(
        runpod_module,
        "_wait_for_watchdog_terminal",
        lambda _root: {
            "control_plane_terminal": True,
            "campaign_budget_settlement": {"status": "settled"},
        },
    )

    result = _monitor_openpi_output_and_teardown(
        root=tmp_path,
        output_secret_get_url="https://storage.example/output?signature=secret",
        provider=Provider(),
        armed={"status": "armed"},
        pod_id="pod-openpi",
        provider_name="vast",
        deadline_epoch=time.time() + 120,
        poll_interval_seconds=0.01,
    )

    assert attempts == 2
    assert result["status"] == "completed"
    assert result["continuing_spend"] is False


def test_monitor_returns_watchdog_control_after_bounded_transient_url_errors(
    tmp_path: Path, monkeypatch
) -> None:
    attempts = 0

    def request(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise URLError("synthetic persistent DNS failure")

    monkeypatch.setattr(runpod_module, "safe_http_request", request)

    result = _monitor_openpi_output_and_teardown(
        root=tmp_path,
        output_secret_get_url="https://storage.example/output?signature=secret",
        provider=object(),
        armed={"status": "armed"},
        pod_id="pod-openpi",
        provider_name="vast",
        deadline_epoch=time.time() + 120,
        poll_interval_seconds=0.01,
    )

    assert attempts == runpod_module.MAX_CONSECUTIVE_TRANSIENT_OUTPUT_ERRORS
    assert result["status"] == "monitor_failed_watchdog_retained"
    assert result["transient_error_attempts"] == attempts
    assert result["continuing_spend"] is True


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


def test_vast_launch_request_uses_frozen_floor_and_args_entrypoint(
    tmp_path: Path,
) -> None:
    release, bundle, _runpod_preflight, _spend = _inputs()
    input_url = "https://storage.example/input?x-goog-signature=input-secret"
    output_url = "https://storage.example/output?x-goog-signature=output-secret"
    preflight = {
        "provider": "vast",
        "gpu_memory_bytes": 46_068_000_000,
        "on_demand_price_usd_per_hour": 0.75,
        "container_disk_bytes": 100 * 1024**3,
        "capacity_request": {
            "min_gpu_ram_mb": 45_000,
            "min_reliability": 0.98,
            "preferred_gpu_keywords": [
                "A40",
                "RTX A6000",
                "RTX 6000 Ada",
                "L40",
            ],
        },
    }

    request = _build_vast_launch_request(
        provider=runpod_module.get_render_provider("vast"),
        root=tmp_path,
        pod_name="blueprint-groot-oscar-canary-openpi-ranking-vast-shape",
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        input_secret_url=input_url,
        output_secret_put_url=output_url,
    )

    payload = request["create_payload"]
    assert request["provider"] == "vast"
    assert request["image"] == release["resolved_digest_ref"]
    assert request["vast_launch_mode"] == "args"
    assert request["entrypoint_override"] == "bash"
    assert request["disk"] == 100
    assert request["min_gpu_ram_mb"] == 45_000
    assert request["max_hourly_rate_usd"] == 0.75
    assert request["min_reliability"] == 0.98
    assert request["preferred_gpu_keywords"][0] == "A40"
    assert payload["onstart"] == "bash"
    assert "openpi_policy_ranking_gpu_bootstrap run" in payload["args_str"]
    assert payload["env"][INPUT_SECRET_URL_ENV] == input_url
    assert payload["env"][INPUT_SHA256_ENV] == bundle["bundle_sha256"]
    assert payload["env"][EXECUTION_MODE_ENV] == "full_campaign"
    assert payload["env"][OUTPUT_SECRET_PUT_URL_ENV] == output_url
    assert payload["env"][GENERIC_OUTPUT_SECRET_URL_ENV] == output_url
    # Request construction is in-memory only.  The returned provider-native
    # request is passed directly to launch and never persisted with raw URLs.
    assert list(tmp_path.iterdir()) == []


def test_vast_launch_request_routes_current_canary_receipt_to_canary_mode(
    tmp_path: Path,
) -> None:
    release, bundle, _runpod_preflight, _spend = _inputs()
    bundle["schema_version"] = CANARY_INPUT_RECEIPT_SCHEMA_VERSION
    preflight = {
        "provider": "vast",
        "gpu_memory_bytes": 46_068_000_000,
        "on_demand_price_usd_per_hour": 0.75,
        "container_disk_bytes": 100 * 1024**3,
        "capacity_request": {"min_gpu_ram_mb": 45_000},
    }

    request = _build_vast_launch_request(
        provider=runpod_module.get_render_provider("vast"),
        root=tmp_path,
        pod_name="blueprint-groot-oscar-canary-openpi-ranking-vast-canary-shape",
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        input_secret_url="https://storage.example/input?signature=secret",
        output_secret_put_url="https://storage.example/output?signature=secret",
    )

    assert request["create_payload"]["env"][EXECUTION_MODE_ENV] == (
        "new_site_diagnostic_canary"
    )


def test_execute_no_create_keeps_watchdog_in_control_until_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    release, bundle, preflight, _spend = _inputs()
    vast_preflight = {
        **preflight,
        "schema_version": "openpi_policy_ranking_provider_preflight.v2",
        "provider": "vast",
        "gpu_type_id": "A40",
        "gpu_memory_bytes": 46_068_000_000,
        "on_demand_price_usd_per_hour": 0.3,
        "capacity_request": {
            "min_reliability": 0.98,
            "preferred_gpu_keywords": ["A40"],
        },
    }
    paths = {}
    for name, value in (("release", release), ("bundle", bundle), ("preflight", preflight)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    for name in ("input", "output-put", "output-get"):
        path = tmp_path / f"{name}.url"
        path.write_text(f"https://storage.example/{name}?signature=secret", encoding="utf-8")
        os.chmod(path, 0o600)
        paths[name] = path
    pending_dir = tmp_path / "pending"
    lease_dir = tmp_path / "leases"
    monkeypatch.setenv("BLUEPRINT_PENDING_TEARDOWN_DIR", str(pending_dir))
    monkeypatch.setenv("BLUEPRINT_PAID_PROVIDER_LANE_LEASE_DIR", str(lease_dir))

    class Provider:
        def capacity_preflight(self, _request=None, **_kwargs) -> dict:
            return {"available": True}

        def billable_inventory(self, *, name_prefix: str) -> dict:
            return {
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
                "name_prefix": name_prefix,
            }

        def build_request(self, spec, _job_dir) -> dict:
            assert spec.image == release["resolved_digest_ref"]
            assert spec.vast_launch_mode == "args"
            return {"create_payload": {"env": spec.env}}

        def launch(self, *_args, **_kwargs) -> dict:
            return {
                "status": "blocked",
                "blockers": ["synthetic_no_create"],
                "allocation_created": False,
            }

    class Process:
        pid = 99_999

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(runpod_module, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        runpod_module,
        "collect_openpi_policy_ranking_vast_preflight",
        lambda **_kwargs: vast_preflight,
    )
    monkeypatch.setattr(runpod_module.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(
        runpod_module,
        "_wait_for_watchdog",
        lambda **kwargs: {
            "status": "armed",
            "independent_process": True,
            "pid": kwargs["process"].pid,
            "pod_name_prefix": kwargs["prefix"],
            "deadline_epoch": kwargs["deadline"],
        },
    )
    monkeypatch.setattr(
        lease_module,
        "_pid_is_alive",
        lambda pid: pid in {os.getpid(), Process.pid},
    )
    monkeypatch.setattr(
        runpod_module,
        "_handoff_cleanup_to_watchdog",
        lambda **_kwargs: {
            "provider_absence_confirmed": True,
            "control_plane_terminal": True,
            "continuing_spend": False,
        },
    )
    adapter_path = tmp_path / "launch" / "adapter.json"
    result = run_openpi_policy_ranking_campaign(
        release_evidence=paths["release"],
        input_bundle_receipt=paths["bundle"],
        preflight_bundle=paths["preflight"],
        admission_out=tmp_path / "launch" / "admission.json",
        bound_request_out=tmp_path / "launch" / "bound.json",
        adapter_output=adapter_path,
        input_secret_url_file=paths["input"],
        output_secret_put_url_file=paths["output-put"],
        output_secret_get_url_file=paths["output-get"],
        pod_name="blueprint-groot-oscar-canary-openpi-ranking-no-create",
        expected_source_commit="a" * 40,
        execute=True,
        hard_ttl_seconds=3600,
        max_spend_usd=1.0,
        campaign_budget_ledger=tmp_path / "campaign-budget.json",
        campaign_initial_spent_usd=0.0,
        campaign_initial_used_gpu_seconds=0,
        campaign_wall_cap_seconds=7200,
    )

    assert result["status"] == "failed"
    assert result["continuing_spend"] is False
    receipt_path = adapter_path.parent / "provider_lane_handoff_receipt.json"
    assert oct(receipt_path.stat().st_mode & 0o777) == "0o600"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["provider_lane_release_mode"] == "watchdog_direct_compute"
    assert receipt["campaign_kind"] == "openpi_policy_ranking"
    pending = json.loads(Path(receipt["pod_pending_teardown_record"]).read_text())
    assert pending["status"] == "open"
    assert pending["lane"] == "openpi_policy_ranking_gpu_canary"
    assert pending["provider"] == "vast"
