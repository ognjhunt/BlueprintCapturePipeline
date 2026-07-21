from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from blueprint_pipeline import groot_oscar_runpod_storage_volume as storage
from blueprint_pipeline import groot_oscar_runpod_preflight as preflight
from blueprint_pipeline.groot_oscar_runpod_storage_volume import (
    _stop_replaced_watchdog,
    build_storage_volume_admission,
    launch_detached,
    retain_verified_model_cache,
    run_storage_model_volume,
)
from blueprint_pipeline.paid_resource_admission import (
    require_paid_resource_admission,
)


VERIFY_RETAINED_REMOTE = storage._verify_retained_model_cache_remote


def test_replaced_watchdog_exit_race_is_recorded_as_stopped() -> None:
    state_path = str(Path("/tmp/replaced-watchdog-state.json").resolve())
    result = _stop_replaced_watchdog(
        source_pid=123,
        source_state_path=state_path,
        process_argv_probe=lambda _pid: (
            "python",
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_model_volume",
            "watchdog",
            "--state",
            state_path,
        ),
        process_signaler=lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
        pid_probe=lambda _pid: True,
    )

    assert result == {
        "status": "stopped",
        "reason": "source_watchdog_exited_during_signal",
        "last_signal": "SIGTERM",
    }


def test_replaced_watchdog_signal_failure_is_evidence_not_exception() -> None:
    state_path = str(Path("/tmp/replaced-watchdog-state.json").resolve())
    result = _stop_replaced_watchdog(
        source_pid=123,
        source_state_path=state_path,
        process_argv_probe=lambda _pid: (
            "python",
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_model_volume",
            "watchdog",
            "--state",
            state_path,
        ),
        process_signaler=lambda *_args: (_ for _ in ()).throw(PermissionError()),
        pid_probe=lambda _pid: True,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "source_watchdog_signal_failed"
    assert result["error_type"] == "PermissionError"


def _patch_retention_remote_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        storage,
        "_verify_retained_model_cache_remote",
        lambda **_kwargs: {
            "schema_version": storage.RETENTION_REMOTE_VERIFICATION_SCHEMA_VERSION,
            "status": "passed",
            "blockers": [],
            "model_manifest_digest": "sha256:" + "a" * 64,
            "verified_file_count": 1,
            "verified_size_bytes": 7,
            "verification_method": "full_s3_streaming_sha256_manifest_verification",
            "provider_mutations_performed": 0,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        },
    )


def _admission(**overrides):
    values = {
        "data_center_id": "US-WA-1",
        "volume_size_gib": 50,
        "storage_ttl_seconds": 14_400,
        "storage_hourly_rate_usd": 0.005,
        "max_storage_spend_usd": 0.05,
        "builder_ttl_seconds": 7200,
        "inventory_verified_zero": True,
        "credentials_verified": True,
        "source_clean": True,
        "local_staging_bytes": 2 * 1024**3,
        "paid_mutation_authorized": True,
        "watchdog_armed_before_allocation": True,
    }
    values.update(overrides)
    return build_storage_volume_admission(**values)


def test_storage_volume_admission_accepts_bounded_no_gpu_tuple() -> None:
    admission = _admission()
    assert admission["status"] == "admitted"
    grant = require_paid_resource_admission(
        admission,
        resource_class="model_volume",
        expected_schema_version=storage.SCHEMA_VERSION,
    )
    assert grant.resource_class == "model_volume"
    assert admission["limits"]["runpod_gpu_pod_limit"] == 0


def test_storage_volume_admission_accepts_120_gib_carrier_volume_size() -> None:
    admission = _admission(volume_size_gib=120)
    assert admission["status"] == "admitted"
    assert admission["limits"]["maximum_volume_gib"] == 200


def test_storage_volume_admission_rejects_one_hour_and_near_canary_deadlines() -> None:
    one_hour = _admission(storage_ttl_seconds=3600)
    assert "storage_model_volume_ttl_outside_guardrail" in one_hour["blockers"]
    assert "storage_model_volume_ttl_does_not_cover_builder_and_canary" in one_hour["blockers"]
    near = _admission(storage_ttl_seconds=10_000)
    assert "storage_model_volume_ttl_does_not_cover_builder_and_canary" in near["blockers"]


@pytest.mark.parametrize("tamper", [False, True])
def test_retention_streams_every_remote_object_and_rejects_digest_tamper(
    tmp_path: Path, tamper: bool
) -> None:
    access = tmp_path / "access"
    secret = tmp_path / "secret"
    access.write_text("access", encoding="utf-8")
    secret.write_text("secret", encoding="utf-8")
    access.chmod(0o600)
    secret.chmod(0o600)
    content = b"verified model bytes"
    manifest = {
        "schema_version": "groot_oscar_external_model_cache.v2",
        "generated_at": "2026-07-15T00:00:00Z",
        "repositories": {},
        "required_files": ["model.bin"],
        "files": [
            {
                "path": "model.bin",
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        ],
        "file_count": 1,
        "total_size_bytes": len(content),
        "network_download_allowed_at_runtime": False,
        "models_embedded_in_release_image": False,
    }
    manifest["manifest_digest"] = storage._canonical_digest(manifest)
    manifest_bytes = json.dumps(manifest).encode()
    prefix = storage.DEFAULT_REMOTE_PREFIX

    class Client:
        @staticmethod
        def get_object(*, Bucket, Key):
            assert Bucket == "volume-1"
            if Key == f"{prefix}/{storage.RETENTION_MANIFEST_NAME}":
                value = manifest_bytes
            else:
                assert Key == f"{prefix}/model.bin"
                value = b"tampered model bytes" if tamper else content
            return {"Body": io.BytesIO(value)}

        @staticmethod
        def list_objects_v2(**kwargs):
            assert kwargs == {"Bucket": "volume-1", "Prefix": prefix + "/"}
            return {
                "IsTruncated": False,
                "Contents": [
                    {"Key": f"{prefix}/{storage.RETENTION_MANIFEST_NAME}"},
                    {"Key": f"{prefix}/model.bin"},
                ],
            }

    result = VERIFY_RETAINED_REMOTE(
        volume_id="volume-1",
        data_center_id="EUR-IS-1",
        expected_manifest_digest=manifest["manifest_digest"],
        access_key_file=access,
        secret_key_file=secret,
        client=Client(),
    )
    assert result["provider_mutations_performed"] == 0
    assert result["raw_secret_values_recorded"] is False
    if tamper:
        assert result["status"] == "blocked"
        assert result["verified_file_count"] == 0
        assert result["verified_size_bytes"] == 0
        assert any(
            blocker.startswith("bounded_cache_retention_remote_verification_failed")
            for blocker in result["blockers"]
        )
    else:
        assert result["status"] == "passed"
        assert result["model_manifest_digest"] == manifest["manifest_digest"]
        assert result["verified_file_count"] == 1
        assert result["verified_size_bytes"] == len(content)


def _retention_source(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "source"
    source.mkdir()
    manifest = "sha256:" + "a" * 64
    pending = tmp_path / "pending.json"
    pending.write_text(
        json.dumps(
            {
                "status": "open",
                "provider": "runpod",
                "lane": storage.PROVIDER_LANE,
                "resource_kind": "network_volume",
                "instance_id": "volume-1",
            }
        ),
        encoding="utf-8",
    )
    binding = {
        "provider": "runpod",
        "lane": storage.PROVIDER_LANE,
        "volume_id": "volume-1",
        "pending_teardown_record": str(pending),
        "watchdog_nonce": "old-nonce",
        "watchdog_deadline_epoch": 20_000.0,
    }
    lane_handoff = {
        "schema_version": "paid_provider_lane_lease_handoff.v1",
        "status": "pending_canary_acceptance",
        "lease_path": str(tmp_path / "provider.lease.json"),
        "source_owner_pid": 111,
        "binding": binding,
    }
    rows = {
        "model_volume_result.json": {
            "status": "completed",
            "volume_id": "volume-1",
            "model_manifest_digest": manifest,
        },
        "network_volume_evidence.json": {
            "status": "verified",
            "id": "volume-1",
            "name": "blueprint-groot-oscar-models-nonce1",
            "data_center_id": "EUR-IS-1",
            "size_bytes": 50 * 1024**3,
            "allocation_nonce": "nonce1",
        },
        "model_cache_verification.json": {
            "status": "passed",
            "provider_volume_id": "volume-1",
            "model_manifest_digest": manifest,
            "cache_root": storage.MODEL_CACHE_PATH,
            "runtime_path_mapping_verified": True,
        },
        "model_cache_transport_result.json": {
            "status": "completed",
            "provider_volume_id": "volume-1",
            "model_manifest_digest": manifest,
            "multipart_absence_verified": True,
            "multipart_cleanup_required": False,
            "cleanup_delete_attempt_count": 0,
            "cleanup_delete_success_count": 0,
        },
        "watchdog_handoff.json": {
            "schema_version": storage.WATCHDOG_HANDOFF_SCHEMA_VERSION,
            "status": "volume_ready_watchdog_retained",
            "volume_id": "volume-1",
            "volume_name": "blueprint-groot-oscar-models-nonce1",
            "pod_name_prefix": "blueprint-storage-only-no-pod-nonce1",
            "watchdog_pid": 111,
            "watchdog_state_path": str(source / "watchdog_state.json"),
            "watchdog_nonce": "old-nonce",
            "watchdog_deadline_epoch": 20_000.0,
            "provider_lane_handoff": lane_handoff,
        },
    }
    for name, row in rows.items():
        (source / name).write_text(json.dumps(row), encoding="utf-8")
    return source, lane_handoff


def test_retention_resolves_current_watchdog_after_terminal_canary_return(
    tmp_path: Path,
) -> None:
    source, original_handoff = _retention_source(tmp_path)
    current = tmp_path / "current-retention"
    current.mkdir()
    state_path = current / "watchdog_state.json"
    state_path.write_text(
        json.dumps(
            {
                "deadline_epoch": 30_000.0,
                "pod_name_prefix": "blueprint-storage-only-no-pod-nonce1",
                "volume_name": "blueprint-groot-oscar-models-nonce1",
                "watchdog_nonce": "current-nonce",
            }
        ),
        encoding="utf-8",
    )
    current_handoff = {
        **original_handoff,
        "status": "accepted",
        "source_owner_pid": 222,
        "binding": {
            **original_handoff["binding"],
            "watchdog_nonce": "current-nonce",
            "watchdog_deadline_epoch": 30_000.0,
        },
    }

    observed = storage._retention_source_watchdog_mapping(
        json.loads((source / "watchdog_handoff.json").read_text()),
        current_handoff,
        process_argv_probe=lambda _pid: (
            "python",
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_model_volume",
            "watchdog",
            "--state",
            str(state_path),
        ),
    )

    assert observed == {
        "watchdog_pid": 222,
        "watchdog_state_path": str(state_path.resolve()),
        "watchdog_deadline_epoch": 30_000.0,
        "pod_name_prefix": "blueprint-storage-only-no-pod-nonce1",
        "volume_name": "blueprint-groot-oscar-models-nonce1",
        "watchdog_nonce": "current-nonce",
    }


def test_verified_cache_enters_bounded_retention_and_remains_canary_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _lane_handoff = _retention_source(tmp_path)
    live = {111, 333}
    retention_seconds = 72 * 60 * 60

    class Provider:
        @staticmethod
        def _key() -> str:
            return "runpod-key"

    class Process:
        pid = 333

        @staticmethod
        def poll():
            return None

        @staticmethod
        def terminate():
            live.discard(333)

        @staticmethod
        def wait(timeout):
            del timeout
            return 0

    retention_root = tmp_path / "retained"

    def arm(**kwargs):
        del kwargs
        retention_root.mkdir(exist_ok=True)
        state_path = retention_root / "watchdog_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "deadline_epoch": 1000.0 + retention_seconds,
                    "pod_name_prefix": "blueprint-storage-only-no-pod-nonce1",
                    "volume_name": "blueprint-groot-oscar-models-nonce1",
                    "watchdog_nonce": "new-nonce",
                }
            ),
            encoding="utf-8",
        )
        return Process(), {
            "armed": True,
            "pid": 333,
            "state_path": str(state_path),
            "watchdog_deadline_epoch": 1000.0 + retention_seconds,
            "pod_name_prefix": "blueprint-storage-only-no-pod-nonce1",
            "volume_name": "blueprint-groot-oscar-models-nonce1",
            "watchdog_nonce": "new-nonce",
        }

    monkeypatch.setattr(storage, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(storage, "_matching_resources", lambda **_kwargs: ([], ["volume-1"], True))
    monkeypatch.setattr(storage, "preflight_runpod_s3", lambda **_kwargs: {"status": "ready"})
    _patch_retention_remote_verification(monkeypatch)
    monkeypatch.setattr(storage, "_arm_watchdog", arm)
    monkeypatch.setattr(storage, "_pid_is_alive", lambda pid: pid in live)

    def rotate(_handoff, **kwargs):
        prepared_state = json.loads(
            (retention_root / "watchdog_state.json").read_text(encoding="utf-8")
        )
        assert prepared_state["provider_lane_handoff"]["lease_path"] == str(
            tmp_path / "provider.lease.json"
        )
        assert (
            prepared_state["pending_teardown_record"]
            == kwargs["retention_binding"]["pending_teardown_record"]
        )
        return {
            "schema_version": "paid_provider_lane_lease_handoff.v1",
            "status": "pending_canary_acceptance",
            "source_owner_pid": 333,
            "binding": kwargs["retention_binding"],
        }

    monkeypatch.setattr(storage, "rotate_paid_provider_lane_lease_to_retention_watchdog", rotate)

    def argv(pid):
        state = (
            source / "watchdog_state.json" if pid == 111 else retention_root / "watchdog_state.json"
        )
        return (
            "python",
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_model_volume",
            "watchdog",
            "--state",
            str(state.resolve()),
        )

    result = retain_verified_model_cache(
        output_dir=retention_root,
        source_output_dir=source,
        retention_ttl_seconds=retention_seconds,
        # RunPod's published under-1-TB rate for a 120 GiB network volume.
        storage_hourly_rate_usd=120 * 0.07 / (30 * 24),
        max_retention_spend_usd=1.0,
        campaign_spent_to_date_usd=13.0,
        campaign_total_spend_cap_usd=20.0,
        runpod_s3_access_key_file=tmp_path / "access",
        runpod_s3_secret_key_file=tmp_path / "secret",
        allow_paid=True,
        clock=lambda: 1000.0,
        sleeper=lambda _seconds: None,
        process_argv_probe=argv,
        process_signaler=lambda pid, _sig: live.discard(pid),
    )

    assert result["status"] == "retained"
    assert result["paid_compute_retained"] is False
    assert result["whitelisted_storage_resource_count"] == 1
    assert result["later_canary_handoff_ready"] is True
    assert result["source_watchdog_stopped"] is True
    assert result["retention_policy"] == {
        "zero_paid_compute_required": True,
        "storage_resource_kind": "runpod_network_volume",
        "storage_resource_id": "volume-1",
        "content_digest": "sha256:" + "a" * 64,
        "content_mutation_policy": "no_writes_after_verification",
        "automatic_delete_at_deadline": True,
    }
    assert result["maximum_retention_spend_usd"] == pytest.approx(0.84)
    admission = json.loads(
        (retention_root / "bounded_model_cache_retention_admission.json").read_text()
    )
    assert admission["provider_inventory"] == {
        "api_confirmed": True,
        "live_pod_ids": [],
        "live_network_volume_ids": ["volume-1"],
        "whitelisted_network_volume_id": "volume-1",
    }
    handoff = json.loads((retention_root / "watchdog_handoff.json").read_text())
    assert handoff["status"] == "volume_ready_watchdog_retained"
    assert handoff["retention_class"] == "bounded_persistent_verified_model_cache"
    assert handoff["provider_lane_handoff"]["source_owner_pid"] == 333
    monkeypatch.setattr(preflight, "_process_alive", lambda _pid: True)
    later_canary = preflight.build_model_volume_watchdog_handoff_evidence(
        handoff=handoff,
        network_volume_id="volume-1",
        canary_watchdog_deadline_epoch=2000.0,
        clock=lambda: 1000.0,
        process_argv_probe=argv,
    )
    assert later_canary["status"] == "verified"
    assert later_canary["provider_lane_handoff"]["source_owner_pid"] == 333


def test_bounded_retention_rejects_unreconciled_spend_before_new_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _lane_handoff = _retention_source(tmp_path)

    class Provider:
        @staticmethod
        def _key() -> str:
            return "runpod-key"

    monkeypatch.setattr(storage, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(storage, "_matching_resources", lambda **_kwargs: ([], ["volume-1"], True))
    monkeypatch.setattr(storage, "preflight_runpod_s3", lambda **_kwargs: {"status": "ready"})
    _patch_retention_remote_verification(monkeypatch)
    monkeypatch.setattr(
        storage,
        "_arm_watchdog",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("watchdog started")),
    )
    result = retain_verified_model_cache(
        output_dir=tmp_path / "retained",
        source_output_dir=source,
        retention_ttl_seconds=7 * 24 * 3600,
        storage_hourly_rate_usd=0.004861111111,
        max_retention_spend_usd=1.0,
        campaign_spent_to_date_usd=1.0,
        campaign_total_spend_cap_usd=20.0,
        runpod_s3_access_key_file=tmp_path / "access",
        runpod_s3_secret_key_file=tmp_path / "secret",
        allow_paid=True,
        clock=lambda: 1000.0,
    )
    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0


@pytest.mark.parametrize(
    ("failure_mode", "expected_blocker"),
    [
        ("unarmed", "bounded_cache_retention_watchdog_not_armed"),
        ("state_write", "bounded_cache_retention_watchdog_state_write_failed"),
    ],
)
def test_retention_pre_rotation_failure_keeps_original_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    expected_blocker: str,
) -> None:
    source, _lane_handoff = _retention_source(tmp_path)
    retention_root = tmp_path / "retained"
    state_path = retention_root / "watchdog_state.json"
    process_stopped = False
    rotation_called = False

    class Provider:
        @staticmethod
        def _key() -> str:
            return "runpod-key"

    class Process:
        pid = 333

        @staticmethod
        def poll():
            return 0 if process_stopped else None

        @staticmethod
        def terminate():
            nonlocal process_stopped
            process_stopped = True

        @staticmethod
        def wait(timeout):
            del timeout
            return 0

    def arm(**_kwargs):
        state_path.write_text(
            json.dumps(
                {
                    "deadline_epoch": 1000.0 + 7 * 24 * 3600,
                    "pod_name_prefix": "blueprint-storage-only-no-pod-nonce1",
                    "volume_name": "blueprint-groot-oscar-models-nonce1",
                    "watchdog_nonce": "new-nonce",
                }
            ),
            encoding="utf-8",
        )
        return Process(), {
            "armed": failure_mode != "unarmed",
            "pid": 333,
            "state_path": str(state_path),
            "watchdog_deadline_epoch": 1000.0 + 7 * 24 * 3600,
            "pod_name_prefix": "blueprint-storage-only-no-pod-nonce1",
            "volume_name": "blueprint-groot-oscar-models-nonce1",
            "watchdog_nonce": "new-nonce",
        }

    real_write_json = storage.write_json

    def fail_prepared_state(path, payload):
        if (
            failure_mode == "state_write"
            and Path(path) == state_path
            and "provider_lane_handoff" in payload
        ):
            raise OSError("disk full")
        return real_write_json(path, payload)

    def rotate(*_args, **_kwargs):
        nonlocal rotation_called
        rotation_called = True
        return {}

    monkeypatch.setattr(storage, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(storage, "_matching_resources", lambda **_kwargs: ([], ["volume-1"], True))
    monkeypatch.setattr(storage, "preflight_runpod_s3", lambda **_kwargs: {"status": "ready"})
    _patch_retention_remote_verification(monkeypatch)
    monkeypatch.setattr(storage, "_arm_watchdog", arm)
    monkeypatch.setattr(storage, "write_json", fail_prepared_state)
    monkeypatch.setattr(storage, "rotate_paid_provider_lane_lease_to_retention_watchdog", rotate)
    result = retain_verified_model_cache(
        output_dir=retention_root,
        source_output_dir=source,
        retention_ttl_seconds=7 * 24 * 3600,
        storage_hourly_rate_usd=0.004861111111,
        max_retention_spend_usd=1.0,
        campaign_spent_to_date_usd=13.0,
        campaign_total_spend_cap_usd=20.0,
        runpod_s3_access_key_file=tmp_path / "access",
        runpod_s3_secret_key_file=tmp_path / "secret",
        allow_paid=True,
        clock=lambda: 1000.0,
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == [expected_blocker]
    assert process_stopped is True
    if failure_mode == "unarmed":
        assert result["retention_watchdog_cleanup_verified"] is True
    assert rotation_called is False


def test_storage_route_has_watchdog_lease_ledger_and_no_pod_create() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/blueprint_pipeline/groot_oscar_runpod_storage_volume.py"
    ).read_text(encoding="utf-8")
    run = source[
        source.index("def run_storage_model_volume(") : source.index("def launch_detached(")
    ]
    assert '"/pods"' not in run
    assert run.index("acquire_paid_provider_lane_lease(") < run.index(
        '"POST",\n            "/networkvolumes"'
    )
    assert run.index("open_pending_teardown(") < run.index('"POST",\n            "/networkvolumes"')
    assert run.index("require_paid_resource_admission(") < run.index(
        "accept_paid_provider_lane_lease_handoff("
    )
    assert run.index("open_pending_teardown(") < run.index(
        "accept_paid_provider_lane_lease_handoff("
    )
    assert run.index("accept_paid_provider_lane_lease_handoff(") < run.index(
        '"POST",\n            "/networkvolumes"'
    )
    assert '"handoff_coverage_deadline_epoch"' in run
    assert "replacement_handoff_coverage_deadline" in run
    assert run.index("bind_pending_teardown_instance(") > run.index(
        '"POST",\n            "/networkvolumes"'
    )
    assert run.index("_arm_watchdog(") < run.index("require_paid_resource_admission(")
    assert "build_runpod_network_volume_evidence(" in run
    assert "storage_model_volume_deadline_too_near_for_canary" in run


def _patch_preallocation(
    monkeypatch: pytest.MonkeyPatch, *, runpod_key: str = "runpod-key"
) -> None:
    class Provider:
        @staticmethod
        def _key() -> str:
            return runpod_key

    monkeypatch.setattr(storage, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        storage,
        "_source_identity",
        lambda _root: ("a" * 40, hashlib.sha256(b"").hexdigest(), False),
    )
    monkeypatch.setattr(storage, "_read_secret", lambda _path: "secret")
    monkeypatch.setattr(storage, "_read_private_secret", lambda _path: "secret")
    monkeypatch.setattr(
        storage,
        "_host_key_material",
        lambda _path: ("private", "public", "SHA256:" + "d" * 43),
    )
    monkeypatch.setattr(
        storage,
        "_live_profile",
        lambda **_kwargs: (
            {"status": "verified", "observed": {"price_hourly_usd": 0.16}},
            [],
        ),
    )
    monkeypatch.setattr(
        storage,
        "preflight_runpod_s3",
        lambda **_kwargs: {"status": "ready", "blockers": []},
    )
    monkeypatch.setattr(
        storage,
        "build_model_cache_wheelhouse",
        lambda **_kwargs: {
            "status": "ready",
            "wheelhouse_path": "wheels",
            "manifest_path": "manifest.json",
        },
    )
    monkeypatch.setattr(storage, "_matching_resources", lambda **_kwargs: ([], [], True))


def _inputs(tmp_path: Path) -> dict:
    builder = tmp_path / "builder.json"
    builder.write_text(
        json.dumps(
            {
                "provider": "digitalocean",
                "purpose": "model_cache_s3",
                "platform": "linux/amd64",
                "python_runtime_verified": True,
                "python_version": "3.12",
                "dependency_lock_verified": True,
                "dependency_wheelhouse_verified": True,
                "dns_resolution_verified": True,
                "outbound_https_verified": True,
                "s3_endpoint_host": "s3api-us-wa-1.runpod.io",
                "free_disk_bytes": 320 * 1024**3,
                "independent_teardown_watchdog": True,
                "ssh_host_key_sha256": "SHA256:" + "d" * 43,
                "ssh_host_key_independently_verified": True,
                "ssh_host_key_verification_method": "launch_bound_generated_host_key",
                "expected_source_commit": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    spend = tmp_path / "spend.json"
    spend.write_text(
        json.dumps(
            {
                "paid_mutation_authorized": True,
                "max_spend_usd": 0.35,
                "hard_ttl_seconds": 7200,
                "one_resource_limit": True,
                "independent_teardown_watchdog": True,
            }
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": tmp_path / "out",
        "repo_root": tmp_path,
        "data_center_id": "US-WA-1",
        "volume_size_gib": 50,
        "storage_ttl_seconds": 14_400,
        "storage_hourly_rate_usd": 0.005,
        "max_storage_spend_usd": 0.05,
        "builder_evidence_path": builder,
        "builder_spend_path": spend,
        "digitalocean_token_file": tmp_path / "do-token",
        "hf_token_file": tmp_path / "hf-token",
        "runpod_s3_access_key_file": tmp_path / "s3-access",
        "runpod_s3_secret_key_file": tmp_path / "s3-secret",
        "login_private_key": tmp_path / "login-key",
        "host_private_key": tmp_path / "host-key",
        "ssh_key_id": 7,
        "region": "sfo3",
        "allow_paid": True,
    }


def _runtime_source_evidence(
    tmp_path: Path,
    *,
    release_ref: str,
    models_externalized: bool = True,
) -> Path:
    path = tmp_path / "runtime-source-release.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "groot_oscar_thin_remote_build_result.v1",
                "status": "completed",
                "blockers": [],
                "release_image_ref": release_ref,
                "resolved_digest_ref": release_ref,
                "runnable_platform": "linux/amd64",
                "required_cuda_version": "12.8",
                "source_commit": "a" * 40,
                "thin_release_contract_status": "passed",
                "thin_release_contract": {
                    "schema_version": "groot_oscar_thin_release_image_contract.v1",
                    "status": "passed",
                    "blockers": [],
                    "release_image_ref": release_ref,
                    "models_externalized": models_externalized,
                    "release_delta_budget_passed": True,
                },
                "models_embedded": not models_externalized,
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_builder_live_preflight_failure_blocks_before_runpod_volume_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preallocation(monkeypatch)
    monkeypatch.setattr(
        storage,
        "_live_profile",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("profile unavailable")),
    )
    monkeypatch.setattr(
        storage,
        "_runpod_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider mutation reached")
        ),
    )
    result = run_storage_model_volume(**_inputs(tmp_path))
    assert result["status"] == "blocked_before_allocation"
    assert result["provider_mutation_attempted"] is False


@pytest.mark.parametrize("volume_size_gib", [50, 121, 200])
def test_runtime_bundle_requires_exactly_120_gib_before_runpod_volume_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, volume_size_gib: int
) -> None:
    _patch_preallocation(monkeypatch)
    monkeypatch.setattr(
        storage,
        "_runpod_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider mutation reached")
        ),
    )
    inputs = _inputs(tmp_path)
    inputs["volume_size_gib"] = volume_size_gib
    release_ref = "docker.io/blueprint/release@sha256:" + "1" * 64
    inputs.update(
        runtime_source_release_image_ref=release_ref,
        carrier_image_ref="pytorch/pytorch:runtime@sha256:" + "2" * 64,
        runtime_source_release_evidence_path=_runtime_source_evidence(
            tmp_path, release_ref=release_ref
        ),
    )
    result = run_storage_model_volume(**inputs)
    assert result["status"] == "blocked_before_allocation"
    assert result["provider_mutation_attempted"] is False


def test_sealed_runtime_source_is_rejected_before_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preallocation(monkeypatch)
    monkeypatch.setattr(
        storage,
        "_runpod_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider mutation reached")
        ),
    )
    intended_ref = "docker.io/blueprint/thin@sha256:" + "1" * 64
    sealed_ref = "docker.io/blueprint/sealed@sha256:" + "9" * 64
    inputs = _inputs(tmp_path)
    inputs.update(
        volume_size_gib=120,
        storage_ttl_seconds=28_800,
        max_storage_spend_usd=0.05,
        runtime_source_release_image_ref=intended_ref,
        carrier_image_ref="pytorch/pytorch:runtime@sha256:" + "2" * 64,
        runtime_source_release_evidence_path=_runtime_source_evidence(
            tmp_path,
            release_ref=sealed_ref,
            models_externalized=False,
        ),
    )
    result = run_storage_model_volume(**inputs)
    assert result["status"] == "blocked_before_allocation"
    assert result["provider_mutation_attempted"] is False
    verification = json.loads(
        (inputs["output_dir"] / "runtime_source_release_evidence_verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert verification["status"] == "blocked"
    assert "runtime_source_release_ref_mismatch" in verification["blockers"]


def test_duplicate_lane_blocks_before_watchdog_or_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preallocation(monkeypatch)
    monkeypatch.setattr(
        storage,
        "acquire_paid_provider_lane_lease",
        lambda **_kwargs: {"status": "blocked", "blockers": ["already-owned"]},
    )
    monkeypatch.setattr(
        storage,
        "_arm_watchdog",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("watchdog started")),
    )
    monkeypatch.setattr(
        storage,
        "_runpod_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider mutation reached")
        ),
    )
    result = run_storage_model_volume(**_inputs(tmp_path))
    assert result["status"] == "blocked_before_allocation"
    assert result["blockers"] == ["already-owned"]


def test_ambiguous_create_reports_authoritative_terminal_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preallocation(monkeypatch)

    class Process:
        pid = 1234

        @staticmethod
        def poll():
            return None

    state = tmp_path / "watchdog_state.json"
    state.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        storage,
        "_arm_watchdog",
        lambda **_kwargs: (
            Process(),
            {
                "armed": True,
                "pid": Process.pid,
                "state_path": str(state),
                "watchdog_nonce": "nonce",
                "pod_name_prefix": "blueprint-storage-only-no-pod-nonce",
                "volume_name": "blueprint-groot-oscar-models-nonce",
                "watchdog_deadline_epoch": 9_999_999_999.0,
            },
        ),
    )
    monkeypatch.setattr(
        storage,
        "acquire_paid_provider_lane_lease",
        lambda **_kwargs: {
            "status": "acquired",
            "path": str(tmp_path / "provider.lease.json"),
            "lease": {"owner_pid": 1},
        },
    )
    monkeypatch.setattr(
        storage,
        "release_paid_provider_lane_lease",
        lambda *_args, **_kwargs: {"status": "released", "released": True},
    )
    pending = tmp_path / "pending.json"
    monkeypatch.setattr(
        storage,
        "open_pending_teardown",
        lambda **_kwargs: {"path": str(pending), "status": "open"},
    )
    monkeypatch.setattr(storage, "mark_pending_teardown_ambiguous", lambda *_a, **_k: {})
    monkeypatch.setattr(storage, "cancel_pending_teardown", lambda *_a, **_k: {})
    monkeypatch.setattr(storage, "load_pending_teardowns", lambda: [])

    def runpod_call(method, path, *_args, **_kwargs):
        assert (method, path) == ("POST", "/networkvolumes")
        return 500, {"error": "provider failure"}

    monkeypatch.setattr(storage, "_runpod_call", runpod_call)

    inputs = _inputs(tmp_path)
    inputs["hf_token_file"].write_text("hf-token", encoding="utf-8")
    inputs["hf_token_file"].chmod(0o600)
    result = run_storage_model_volume(**inputs)

    assert result["status"] == "failed"
    assert result["volume_id"] is None
    assert result["failure_volume_teardown"] == {
        "provider_absence_confirmed": True,
        "verification_method": "authenticated_global_provider_inventory",
        "observed_volume_id": None,
    }
    handoff_path = inputs["output_dir"] / "watchdog_handoff.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert handoff_path.is_file()
    assert handoff["status"] == "failure_cleanup_provider_terminal"
    assert handoff["failure_volume_absence_confirmed"] is True


def test_storage_detached_launch_is_single_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        pid = 1234

    popen_kwargs: dict[str, object] = {}

    def popen(*_args: object, **kwargs: object) -> Process:
        popen_kwargs.update(kwargs)
        return Process()

    monkeypatch.setattr(storage.subprocess, "Popen", popen)
    launched = launch_detached(output_dir=tmp_path, run_arguments=["--allow-paid"])
    assert launched["status"] == "supervisor_started"
    assert launched["local_sigint_ignored"] is True
    assert popen_kwargs["start_new_session"] is True
    assert popen_kwargs["env"]["BLUEPRINT_DETACHED_MODEL_VOLUME_SUPERVISOR"] == "1"
    with pytest.raises(ValueError, match="already_has_supervisor"):
        launch_detached(output_dir=tmp_path, run_arguments=["--allow-paid"])
