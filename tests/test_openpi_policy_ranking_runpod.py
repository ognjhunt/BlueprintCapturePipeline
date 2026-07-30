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
from blueprint_pipeline.openpi_current_reference_gpu_bundle import (
    INPUT_RECEIPT_SCHEMA_VERSION as CURRENT_REFERENCE_RECEIPT_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION as CURRENT_REFERENCE_SCHEMA_VERSION,
)
from blueprint_pipeline.openpi_policy_ranking_runpod import (
    EXECUTION_MODE_ENV,
    EXECUTION_MODE_CURRENT_REFERENCE_POLICY_CANARY,
    GENERIC_OUTPUT_SECRET_URL_ENV,
    INPUT_SECRET_URL_ENV,
    INPUT_SHA256_ENV,
    OUTPUT_SECRET_PUT_URL_ENV,
    _build_vast_launch_request,
    _monitor_openpi_output_and_teardown,
    _runtime_source_bootstrap_script,
    _validate_output_archive,
    build_openpi_policy_ranking_provider_request,
    run_openpi_policy_ranking_campaign,
    shape_openpi_policy_ranking_request_without_mutation,
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
        archive.writestr("openpi_policy_ranking_gpu_job.json", json.dumps(manifest))
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


def _current_reference_bundle(*, image_source_commit: str, runtime_commit: str):
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
        "image_source_commit": image_source_commit,
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
    return {
        "schema_version": CURRENT_REFERENCE_RECEIPT_SCHEMA_VERSION,
        "bundle_sha256": "c" * 64,
        "manifest": manifest,
    }


def test_current_reference_runtime_source_skips_only_frozen_gstack_symlinks() -> None:
    script = _runtime_source_bootstrap_script(EXECUTION_MODE_CURRENT_REFERENCE_POLICY_CANARY)

    assert '".agents/skills/gstack"' in script
    assert '".claude/skills/gstack"' in script
    assert "if member.issym():" in script
    assert 'raise SystemExit("runtime_source_archive_symlink_not_allowlisted")' in script
    assert "if member.islnk() or not (member.isdir() or member.isfile()):" in script
    assert script.count("skipped_symlinks") == 2


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
    assert body["volumeInGb"] == 80
    assert body["dockerEntrypoint"][-2:] == [
        "blueprint_pipeline.openpi_policy_ranking_gpu_bootstrap",
        "run",
    ]
    assert body["env"][INPUT_SECRET_URL_ENV] == "<redacted:secret-env>"
    assert body["env"][OUTPUT_SECRET_PUT_URL_ENV] == "<redacted:secret-env>"
    assert body["env"][EXECUTION_MODE_ENV] == "full_campaign"
    persisted = output.read_text(encoding="utf-8")
    assert input_url not in persisted
    assert output_url not in persisted
    assert "input-secret" not in persisted
    assert "output-secret" not in persisted
    request = json.loads((tmp_path / "openpi_provider_launch_request.json").read_text())
    assert input_url not in json.dumps(request)


def test_current_reference_request_uses_hash_bound_source_overlay() -> None:
    release, _bundle, preflight, spend = _inputs()
    runtime_commit = "d" * 40
    bundle = _current_reference_bundle(
        image_source_commit=release["source_commit"], runtime_commit=runtime_commit
    )
    prepared = build_openpi_policy_ranking_provider_request(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit=runtime_commit,
        job_id="openpi-current-reference-test",
    )
    assert prepared["status"] == "admitted"
    shape = prepared["bound_request"]["provider_request_shape"]
    assert shape["docker_entrypoint"] == ["bash", "-lc"]
    assert len(shape["docker_start_cmd"]) == 1
    assert "runtime_source_archive_sha256_mismatch" in shape["docker_start_cmd"][0]
    assert shape["runtime_source"] == {
        "overlay_required": True,
        "commit": runtime_commit,
        "archive_sha256": "e" * 64,
        "archive_url_is_exact_commit_codeload": True,
        "image_source_commit": release["source_commit"],
    }
    assert shape["gpu"]["gpu_count"] == 1
    assert shape["limits"]["external_watchdog_ttl_required"] is True


def test_current_reference_terminal_output_requires_all_three_native_actions() -> None:
    policy_ids = ("pi0_droid", "pi0_fast_droid", "pi05_droid")
    manifest = {
        "schema_version": "openpi_current_reference_policy_canary.v1",
        "status": "completed",
        "frozen_policy_order": list(policy_ids),
        "requests_per_policy": 1,
        "policy_results": [
            {"policy_id": policy_id, "status": "completed"} for policy_id in policy_ids
        ],
        "blockers": [],
        "wam_called": False,
        "judge_called": False,
        "physical_outcome_accessed": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "openpi_current_reference_policy_canary.json",
            json.dumps(manifest),
        )
        for policy_id in policy_ids:
            archive.writestr(f"{policy_id}_native_action.npy", b"native-action")
            archive.writestr(f"{policy_id}_policy_receipt.json", b"{}")
    valid = _validate_output_archive(buffer.getvalue())
    assert valid["status"] == "completed"
    assert valid["execution_mode"] == "current_reference_policy_identity_canary"

    missing = io.BytesIO()
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr(
            "openpi_current_reference_policy_canary.json",
            json.dumps(manifest),
        )
    blocked = _validate_output_archive(missing.getvalue())
    assert blocked["status"] == "blocked"
    assert any("policy_artifacts_missing" in item for item in blocked["blockers"])


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

    assert request["create_payload"]["env"][EXECUTION_MODE_ENV] == ("new_site_diagnostic_canary")


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
        provider_name="runpod",
    )

    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0
    assert result["watchdog_process_started"] is False
    assert result["budget_reservation_created"] is False


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
        "run_runpod_provider_adapter",
        lambda **_kwargs: {
            "status": "blocked",
            "blockers": ["synthetic_no_create"],
            "runpod_side_effects_may_have_occurred": False,
        },
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
