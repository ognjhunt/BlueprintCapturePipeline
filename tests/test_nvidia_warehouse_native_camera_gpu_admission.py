from __future__ import annotations

import hashlib
import io
import json
import os
import time
import urllib.error
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import nvidia_warehouse_native_camera_gpu_admission as camera_gpu
from blueprint_pipeline import paid_provider_lane_lease as lease_module
from blueprint_pipeline.g1_kitchen_bundle_compatibility import (
    CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
)
from blueprint_pipeline.isaac_worker_image_manifest import (
    SCHEMA_VERSION as ISAAC_IMAGE_MANIFEST_SCHEMA_VERSION,
)
from blueprint_pipeline.nvidia_warehouse_native_camera_gpu_admission import (
    RELEASE_SCHEMA_VERSION,
    build_native_camera_gpu_admission,
    build_native_camera_gpu_provider_request,
    build_native_camera_gpu_release_evidence,
    validate_native_camera_gpu_output_archive,
)
from blueprint_pipeline.nvidia_warehouse_native_camera_gpu_bundle import (
    BUNDLE_SCHEMA_VERSION,
    INPUT_SECRET_URL_ENV,
    INPUT_SHA256_ENV,
    OUTPUT_SECRET_PUT_URL_ENV,
    RECEIPT_SCHEMA_VERSION,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


def _inputs():
    release = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": "passed",
        "source_commit": "a" * 40,
        "resolved_digest_ref": "docker.io/example/isaac@sha256:" + "b" * 64,
        "runnable_platform": "linux/amd64",
        "isaac_sim_major_version": 6,
        "source_dirty_patch_sha256": CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
    }
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_commit": "a" * 40,
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "purpose": "private_internal_nvidia_warehouse_native_camera_canary",
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    bundle = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "bundle_sha256": "c" * 64,
        "manifest": manifest,
    }
    bundle["receipt_sha256"] = canonical_sha256(bundle)
    preflight = {
        "schema_version": "openpi_policy_ranking_provider_preflight.v2",
        "status": "verified",
        "provider": "vast",
        "provider_api_verified": True,
        "observed_at_epoch": time.time(),
        "blockers": [],
        "provider_inventory_verified_zero": True,
        "attempt_billable_inventory": {
            "api_confirmed": True,
            "live_resource_count": 0,
            "resources": [],
        },
        "single_gpu_available": True,
        "gpu_memory_bytes": 24 * 1024**3,
        "gpu_type_id": "RTX 4090",
        "on_demand_price_usd_per_hour": 0.5,
        "container_disk_bytes": 100 * 1024**3,
    }
    spend = {
        "paid_mutation_authorized": True,
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
        "watchdog_armed_before_allocation": True,
        "hard_ttl_seconds": 1200,
        "max_spend_usd": 1.0,
        "physical_robot_endpoint_access_allowed": False,
    }
    return release, bundle, preflight, spend


def test_native_camera_release_binds_registry_config_to_exact_clean_source() -> None:
    image_manifest = {
        "schema_version": ISAAC_IMAGE_MANIFEST_SCHEMA_VERSION,
        "status": "completed",
        "resolved_digest_ref": "docker.io/example/isaac@sha256:" + "b" * 64,
        "runnable_platform": "linux/amd64",
        "raw_secret_values_recorded": False,
        "worker_build_identity": {
            "status": "verified",
            "blockers": [],
            "source_commit": "a" * 40,
            "source_dirty_patch_sha256": CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
            "worker_image_family": "isaac-eval-worker",
            "isaac_sim_major_version": 6,
        },
    }

    result = build_native_camera_gpu_release_evidence(
        image_manifest=image_manifest,
        expected_source_commit="a" * 40,
    )

    assert result["status"] == "passed"
    assert result["source_commit"] == "a" * 40
    assert result["source_dirty_patch_sha256"] == (
        CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256
    )
    assert result["resolved_digest_ref"].endswith("b" * 64)


def test_native_camera_release_rejects_dirty_or_mismatched_registry_config() -> None:
    image_manifest = {
        "schema_version": ISAAC_IMAGE_MANIFEST_SCHEMA_VERSION,
        "status": "completed",
        "resolved_digest_ref": "docker.io/example/isaac@sha256:" + "b" * 64,
        "runnable_platform": "linux/amd64",
        "raw_secret_values_recorded": False,
        "worker_build_identity": {
            "status": "verified",
            "source_commit": "f" * 40,
            "source_dirty_patch_sha256": "d" * 64,
            "worker_image_family": "isaac-eval-worker",
            "isaac_sim_major_version": 6,
        },
    }

    result = build_native_camera_gpu_release_evidence(
        image_manifest=image_manifest,
        expected_source_commit="a" * 40,
    )

    assert result["status"] == "blocked"
    assert "native_camera_gpu_release_source_commit_mismatch" in result["blockers"]
    assert "native_camera_gpu_release_dirty_overlay_forbidden" in result["blockers"]


def test_native_camera_gpu_admission_passes_exact_label_free_contract() -> None:
    release, bundle, preflight, spend = _inputs()
    result = build_native_camera_gpu_admission(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit="a" * 40,
    )

    assert result["status"] == "admitted"
    assert result["limits"]["one_resource"] is True
    assert result["claim_boundary"]["policy_wam_loop_proven"] is False


def test_native_camera_gpu_admission_allows_authorized_second_global_gpu() -> None:
    release, bundle, preflight, spend = _inputs()
    preflight.update(
        {
            "provider_inventory_verified_zero": False,
            "provider_inventory_below_global_ceiling": True,
            "maximum_concurrent_paid_gpus_global": 2,
            "global_paid_gpu_inventory": {
                "status": "verified",
                "total_live_paid_gpus_observed": 1,
            },
        }
    )

    result = build_native_camera_gpu_admission(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit="a" * 40,
    )

    assert result["status"] == "admitted"
    assert result["limits"]["one_resource"] is True
    assert result["limits"]["maximum_concurrent_paid_gpus_global"] == 2


def test_native_camera_concurrency_preflight_blocks_at_global_ceiling() -> None:
    _release, _bundle, preflight, _spend = _inputs()
    preflight["provider_inventory_verified_zero"] = False
    preflight["blockers"] = ["openpi_gpu_preflight_billable_inventory_not_zero"]
    global_inventory = {
        "status": "verified",
        "total_live_paid_gpus_observed": 2,
        "blockers": [],
    }

    result = camera_gpu._concurrency_aware_native_preflight(
        preflight=preflight,
        global_inventory=global_inventory,
        maximum_concurrent_paid_gpus_global=2,
    )

    assert result["status"] == "blocked"
    assert result["provider_inventory_below_global_ceiling"] is False
    assert "native_camera_global_paid_gpu_ceiling_reached_or_unverified" in result[
        "blockers"
    ]


def test_native_camera_gpu_request_binds_exact_worker_and_redacts_secrets() -> None:
    release, bundle, preflight, spend = _inputs()
    preflight["excluded_machine_ids"] = [43326]

    prepared = build_native_camera_gpu_provider_request(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit="a" * 40,
        job_id="blueprint-native-warehouse-camera-v1",
        launcher_source_commit="d" * 40,
    )

    assert prepared["status"] == "admitted"
    request = prepared["bound_request"]
    shape = request["provider_request_shape"]
    assert request["schema_version"] == "nvidia_warehouse_native_camera_gpu_request.v2"
    assert request["provider"] == "vast"
    assert request["input_bundle_sha256"] == "c" * 64
    assert request["launcher_source_commit"] == "d" * 40
    assert shape["docker_entrypoint"] == ["bash"]
    assert shape["docker_start_cmd"] == [
        "-lc",
        "exec /isaac-sim/python.sh -m "
        "blueprint_pipeline.nvidia_warehouse_native_camera_gpu_bundle "
        "worker --workspace /workspace/native-camera-canary",
    ]
    environment = shape["environment"]
    assert environment["secret_env_var_names"] == [
        INPUT_SECRET_URL_ENV,
        OUTPUT_SECRET_PUT_URL_ENV,
    ]
    assert environment["plaintext_env_values"] == {INPUT_SHA256_ENV: "c" * 64}
    assert environment["secret_values_in_artifact"] is False
    assert shape["gpu"]["gpu_count"] == 1
    assert shape["limits"]["attempt_inventory_zero_required_before_launch"] is True
    assert shape["limits"]["global_inventory_below_ceiling_required_before_launch"] is True
    assert shape["limits"]["owned_resource_absence_required_after_launch"] is True
    assert shape["limits"]["maximum_concurrent_paid_gpus_global"] == 1
    assert shape["limits"]["excluded_machine_ids"] == [43326]
    assert shape["output_contract"] == {
        "individual_external_camera_frame_required": True,
        "individual_wrist_camera_frame_required": True,
        "camera_canary_result_required": True,
        "upload_before_shutdown_required": True,
    }
    declared = request["manifest_sha256"]
    payload = dict(request)
    payload.pop("manifest_sha256")
    assert declared == canonical_sha256(payload)


def test_native_camera_gpu_output_requires_exact_four_frame_evidence() -> None:
    frame_bytes = {
        f"runtime/{view}_{phase}.png": f"{view}-{phase}".encode()
        for view in ("external", "wrist")
        for phase in ("initial", "commanded")
    }
    views = {}
    for view in ("external", "wrist"):
        frames = {}
        for phase in ("initial", "commanded"):
            relative = f"runtime/{view}_{phase}.png"
            frames[phase] = {
                "relative_path": relative,
                "sha256": hashlib.sha256(frame_bytes[relative]).hexdigest(),
                "resolution": [640, 480],
            }
        views[view] = {"frames": frames}
    result = {
        "schema_version": "nvidia_warehouse_native_camera_canary_result.v1",
        "status": "passed",
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "paid_policy_or_wam_model_invoked": False,
        "assessment": {"views": views},
        "claim_boundary": {
            "native_scene_and_camera_technical_canary_only": True,
            "policy_wam_loop_proven": False,
            "ranking_accuracy": False,
            "physical_success": False,
            "captured_site_transfer_validation": False,
            "phase_b_confirmation": False,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("native_camera_canary_result.json", json.dumps(result))
        for name, content in frame_bytes.items():
            archive.writestr(name, content)

    validation = validate_native_camera_gpu_output_archive(buffer.getvalue())

    assert validation["status"] == "completed"
    assert validation["canary_status"] == "passed"
    assert validation["archive_member_count"] == 5


def test_native_camera_gpu_output_rejects_missing_wrist_failure_media() -> None:
    result = {
        "schema_version": "nvidia_warehouse_native_camera_canary_result.v1",
        "status": "failed",
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "paid_policy_or_wam_model_invoked": False,
        "assessment": {"views": {}},
        "claim_boundary": {
            "native_scene_and_camera_technical_canary_only": True,
            "policy_wam_loop_proven": False,
            "ranking_accuracy": False,
            "physical_success": False,
            "captured_site_transfer_validation": False,
            "phase_b_confirmation": False,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("native_camera_canary_result.json", json.dumps(result))

    validation = validate_native_camera_gpu_output_archive(buffer.getvalue())

    assert validation["status"] == "blocked"
    assert "native_camera_output_frame_missing:wrist:initial" in validation["blockers"]


def test_native_camera_monitor_tears_down_before_admitting_policy_wam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response:
        status = 200
        body = b"synthetic-terminal-output"

    class Provider:
        def billable_inventory(self, *, name_prefix: str) -> dict:
            assert name_prefix == ""
            return {
                "api_confirmed": True,
                "live_resource_count": 1,
                "resources": [
                    {
                        "instance_id": "other-task-1",
                        "name": "blueprint-disjoint-reference-task",
                    }
                ],
            }

    monkeypatch.setattr(camera_gpu, "safe_http_request", lambda *_a, **_k: Response())
    monkeypatch.setattr(
        camera_gpu,
        "validate_native_camera_gpu_output_archive",
        lambda _body: {
            "status": "completed",
            "blockers": [],
            "canary_status": "passed",
        },
    )
    monkeypatch.setattr(
        camera_gpu,
        "terminate_canary_resources",
        lambda **_kwargs: {"provider_absence_confirmed": True},
    )
    monkeypatch.setattr(
        camera_gpu,
        "write_owner_teardown_cancel_request",
        lambda **_kwargs: {"status": "requested"},
    )
    monkeypatch.setattr(
        camera_gpu,
        "_wait_for_watchdog_terminal",
        lambda _root: {
            "control_plane_terminal": True,
            "campaign_budget_settlement": {"status": "settled"},
        },
    )

    result = camera_gpu._monitor_native_camera_output_and_teardown(
        root=tmp_path,
        output_secret_get_url="https://storage.example/output?signature=secret",
        provider=Provider(),
        armed={"status": "armed"},
        instance_id="camera-1",
        provider_name="vast",
        deadline_epoch=time.time() + 300,
        poll_interval_seconds=0.1,
    )

    assert result["status"] == "completed"
    assert result["advance_to_policy_wam"] is True
    assert result["provider_absence_confirmed"] is True
    assert result["control_plane_terminal"] is True
    assert result["continuing_spend"] is False


def test_native_camera_monitor_exits_and_tears_down_when_worker_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Provider:
        def inspect(self, _instance_id: str) -> dict:
            return {
                "status": "absent",
                "provider_absence_confirmed": True,
                "api_confirmed": True,
            }

    def missing_output(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://storage.example/output", 404, "missing", {}, None
        )

    monkeypatch.setattr(camera_gpu, "safe_http_request", missing_output)
    monkeypatch.setattr(
        camera_gpu,
        "terminate_canary_resources",
        lambda **_kwargs: {"provider_absence_confirmed": True},
    )
    monkeypatch.setattr(
        camera_gpu,
        "_camera_cleanup_handoff",
        lambda **_kwargs: {
            "provider_absence_confirmed": True,
            "control_plane_terminal": True,
            "continuing_spend": False,
        },
    )

    result = camera_gpu._monitor_native_camera_output_and_teardown(
        root=tmp_path,
        output_secret_get_url="https://storage.example/output?signature=secret",
        provider=Provider(),
        armed={"status": "armed"},
        instance_id="camera-1",
        provider_name="vast",
        deadline_epoch=time.time() + 300,
        poll_interval_seconds=0.1,
    )

    assert result["status"] == "failed"
    assert result["advance_to_policy_wam"] is False
    assert result["continuing_spend"] is False
    assert result["blockers"] == ["native_camera_worker_terminal_without_output"]


def test_native_camera_execute_arms_guards_before_vast_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, bundle, preflight, _spend = _inputs()
    paths: dict[str, Path] = {}
    for name, value in (("release", release), ("bundle", bundle), ("preflight", preflight)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    for name in ("input", "output-put", "output-get"):
        path = tmp_path / f"{name}.url"
        path.write_text(
            f"https://storage.example/{name}?signature={name}-secret",
            encoding="utf-8",
        )
        path.chmod(0o600)
        paths[name] = path
    monkeypatch.setenv("BLUEPRINT_PENDING_TEARDOWN_DIR", str(tmp_path / "pending"))
    monkeypatch.setenv(
        "BLUEPRINT_PAID_PROVIDER_LANE_LEASE_DIR", str(tmp_path / "leases")
    )
    events: list[str] = []
    launch_root = tmp_path / "launch"

    class Provider:
        def capacity_preflight(self, _request=None, **_kwargs) -> dict:
            events.append("capacity")
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
            assert spec.bootstrap_argv[1].startswith("exec /isaac-sim/python.sh")
            assert spec.env[INPUT_SHA256_ENV] == bundle["bundle_sha256"]
            return {"create_payload": {"env": spec.env}}

        def launch(self, _root, _request, **kwargs) -> dict:
            events.append("launch")
            assert "armed" in events
            receipt = json.loads(
                (launch_root / "provider_lane_handoff_receipt.json").read_text()
            )
            assert receipt["status"] == "accepted"
            assert receipt["paid_lane"] == camera_gpu.PAID_LANE
            assert kwargs["paid_resource_admission_grant"]
            return {"status": "launched", "instance_id": "vast-camera-1"}

    class Process:
        pid = 99_997

        def poll(self):
            return None

        def wait(self, timeout=None):
            del timeout
            return 0

    monkeypatch.setattr(camera_gpu, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        camera_gpu,
        "GLOBAL_PAID_GPU_LAUNCH_LOCK",
        tmp_path / "paid_gpu_global_launch.lock",
    )
    monkeypatch.setattr(
        camera_gpu,
        "collect_openpi_policy_ranking_vast_preflight",
        lambda **_kwargs: preflight,
    )
    monkeypatch.setattr(camera_gpu.subprocess, "Popen", lambda *_a, **_k: Process())
    monkeypatch.setattr(
        camera_gpu,
        "_wait_for_watchdog",
        lambda **kwargs: events.append("armed")
        or {
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
        camera_gpu,
        "_monitor_native_camera_output_and_teardown",
        lambda **_kwargs: {
            "status": "completed",
            "continuing_spend": False,
            "advance_to_policy_wam": True,
        },
    )

    result = camera_gpu.run_native_camera_gpu_lane(
        release_evidence=paths["release"],
        input_bundle_receipt=paths["bundle"],
        preflight_bundle=paths["preflight"],
        admission_out=launch_root / "admission.json",
        bound_request_out=launch_root / "bound.json",
        adapter_output=launch_root / "adapter.json",
        pod_name="blueprint-native-warehouse-camera-v2",
        expected_source_commit="a" * 40,
        launcher_source_commit="d" * 40,
        execute=True,
        hard_ttl_seconds=1200,
        max_spend_usd=1.0,
        input_secret_url_file=paths["input"],
        output_secret_put_url_file=paths["output-put"],
        output_secret_get_url_file=paths["output-get"],
        campaign_budget_ledger=tmp_path / "budget.json",
        campaign_initial_spent_usd=0.0,
        campaign_initial_used_gpu_seconds=0,
        campaign_wall_cap_seconds=3600,
        provider_name="vast",
    )

    assert result["status"] == "completed"
    assert events.index("armed") < events.index("launch")
    persisted = "\n".join(
        path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json")
    )
    assert "input-secret" not in persisted
    assert "output-put-secret" not in persisted
    assert "output-get-secret" not in persisted


def test_native_camera_gpu_admission_blocks_dirty_image_rank_access_and_inventory() -> None:
    release, bundle, preflight, spend = _inputs()
    release["source_dirty_patch_sha256"] = "d" * 64
    bundle["manifest"]["rankings_or_policy_outcomes_accessed"] = True
    bundle["manifest"]["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in bundle["manifest"].items() if key != "manifest_sha256"}
    )
    bundle["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in bundle.items() if key != "receipt_sha256"}
    )
    preflight["provider_inventory_verified_zero"] = False

    result = build_native_camera_gpu_admission(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit="a" * 40,
    )

    assert result["status"] == "blocked"
    assert "native_camera_gpu_release_dirty_overlay_forbidden" in result["blockers"]
    assert "native_camera_gpu_input_freeze_invalid" in result["blockers"]
    assert "native_camera_gpu_global_paid_gpu_ceiling_not_proven" in result["blockers"]


def test_native_camera_gpu_admission_fails_closed_on_malformed_capacity_values() -> None:
    release, bundle, preflight, spend = _inputs()
    preflight["gpu_memory_bytes"] = "24GiB"
    preflight["container_disk_bytes"] = None

    result = build_native_camera_gpu_admission(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit="a" * 40,
    )

    assert result["status"] == "blocked"
    assert "native_camera_gpu_memory_below_floor" in result["blockers"]
    assert "native_camera_gpu_container_disk_below_floor" in result["blockers"]
