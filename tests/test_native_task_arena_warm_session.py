from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.error
import zipfile

import pytest

from blueprint_pipeline.common import write_json
import blueprint_pipeline.native_task_arena_warm_authority as authority
import blueprint_pipeline.native_task_arena_warm_vast as warm_vast


COMMIT = "a" * 40
NOW = 2_000_000_000.0


def _prepared(tmp_path: Path) -> tuple[dict, Path]:
    bundle = tmp_path / "native_task_arena_provider_bundle.zip"
    bundle.write_bytes(b"bundle")
    prepared = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "controls",
        "bundle_path": str(bundle),
        "bundle_sha256": "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "input_digest": "sha256:" + "c" * 64,
        "implementation_commit": COMMIT,
        "container_image": "image@sha256:" + "d" * 64,
        "runtime_source_packet": {
            "packet_sha256": "sha256:" + "e" * 64,
            "packet_size_bytes": 4_400_000_000,
        },
        "expected_output_filename": "native_task_arena_control_result.v1.json",
    }
    receipt = tmp_path / "native_task_arena_provider_bundle_receipt.v1.json"
    write_json(receipt, prepared)
    return prepared, receipt


def _session(prepared: dict) -> dict:
    session = {
        "schema_version": authority.SESSION_SCHEMA_VERSION,
        "generated_at": "fixed",
        "status": "ready",
        "provider": "vast",
        "instance_id": 123,
        "container_image": prepared["container_image"],
        "runtime_dependency_packet_sha256": prepared["runtime_source_packet"][
            "packet_sha256"
        ],
        "runtime_dependency_packet_size_bytes": prepared["runtime_source_packet"][
            "packet_size_bytes"
        ],
        "runtime_dependency_cache_ready": True,
        "ssh_host": "ssh.example",
        "ssh_port": 12345,
        "watchdog_pid": 456,
        "watchdog_deadline_epoch": NOW + 3600,
        "max_hourly_rate_usd": 0.8,
        "hard_cap_usd": 2.0,
        "continuing_spend": True,
        "raw_secret_values_recorded": False,
    }
    session["session_digest"] = authority._session_digest(session)
    return session


def test_warm_authority_binds_session_bundle_and_zero_allocations(
    tmp_path: Path,
) -> None:
    prepared, receipt = _prepared(tmp_path)
    session = _session(prepared)
    session_path = tmp_path / "warm-session.json"
    write_json(session_path, session)

    issued = authority.materialize_native_task_arena_warm_attempt_authority(
        warm_session_path=session_path,
        bundle_receipt_path=receipt,
        prepared_bundle=prepared,
        authorization_reference="current production goal",
        authorized_by="user",
        authorized_on="2026-08-21",
        output_path=tmp_path / "warm-authority.json",
        observed_now_epoch=NOW,
    )

    assert issued["maximum_provider_allocations"] == 0
    assert issued["provider_instance_id"] == 123
    assert issued["warm_session_digest"] == session["session_digest"]
    assert authority.validate_native_task_arena_warm_attempt_authority(
        issued,
        warm_session=session,
        prepared_bundle=prepared,
        observed_now_epoch=NOW,
    )["authorization_digest"] == issued["authorization_digest"]


def test_warm_session_refuses_wrong_dependency_or_expiring_watchdog(
    tmp_path: Path,
) -> None:
    prepared, _receipt = _prepared(tmp_path)
    session = _session(prepared)
    prepared["runtime_source_packet"] = {
        **prepared["runtime_source_packet"],
        "packet_sha256": "sha256:" + "f" * 64,
    }
    with pytest.raises(ValueError, match="runtime_dependency_sha256_mismatch"):
        authority.validate_native_task_arena_warm_session(
            session, prepared_bundle=prepared, observed_now_epoch=NOW
        )

    prepared["runtime_source_packet"]["packet_sha256"] = "sha256:" + "e" * 64
    with pytest.raises(ValueError, match="watchdog_window_too_short"):
        authority.validate_native_task_arena_warm_session(
            session, prepared_bundle=prepared, observed_now_epoch=NOW + 3000
        )


def test_remote_dispatcher_requires_existing_cache_and_uploads_small_result() -> None:
    script = warm_vast._remote_attempt_script(
        bundle_url="https://objects.example/bundle",
        output_put_url="https://objects.example/output",
        bundle_sha256="sha256:" + "b" * 64,
        runtime_dependency_sha256="sha256:" + "e" * 64,
        runtime_dependency_size_bytes=4_400_000_000,
        attempt_key="b" * 16,
    )

    assert "/workspace/native_task_runtime_dependency_cache" in script
    assert "BLUEPRINT_ARENA_WARM_DEPENDENCY_CACHE_HIT" in script
    assert "dependency_cache_miss" in script
    assert "runtime_dependency_download" not in script
    assert "BLUEPRINT_ARENA_WARM_PROVIDER_OUTPUT_UPLOAD_OK" in script


def test_close_accepts_provider_404_as_observed_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_api_json(*, method, **_kwargs):
        nonlocal calls
        calls += 1
        if method == "DELETE":
            return 204, {}
        raise urllib.error.HTTPError(
            url="https://provider.example/instances/123/",
            code=404,
            msg="not found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(warm_vast, "_api_json", fake_api_json)

    result = warm_vast._close_warm_instance(
        instance_id=123, api_key="secret", timeout_seconds=1
    )

    assert calls == 2
    assert result["status"] == "completed"
    assert result["provider_instance_absent"] is True
    assert result["continuing_spend_from_this_run"] is False


def test_warm_execution_reuses_instance_without_allocating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, receipt = _prepared(tmp_path)
    session = _session(prepared)
    session_path = tmp_path / "warm-session.json"
    write_json(session_path, session)
    issued = authority.materialize_native_task_arena_warm_attempt_authority(
        warm_session_path=session_path,
        bundle_receipt_path=receipt,
        prepared_bundle=prepared,
        authorization_reference="current production goal",
        authorized_by="user",
        authorized_on="2026-08-21",
        output_path=tmp_path / "warm-authority.json",
        observed_now_epoch=NOW,
    )
    # Rebind the session deadline to real test time after authority validation.
    monkeypatch.setattr(warm_vast.time, "time", lambda: NOW)
    monkeypatch.setattr(warm_vast, "_read_api_key", lambda: "secret-api-key")
    monkeypatch.setattr(warm_vast, "_truthy", lambda _name: True)
    monkeypatch.setattr(
        warm_vast,
        "consume_native_task_arena_warm_authority_once",
        lambda _value: {
            "status": "consumed",
            "authorization_digest": issued["authorization_digest"],
        },
    )
    monkeypatch.setattr(
        warm_vast,
        "_api_json",
        lambda **_kwargs: (
            200,
            {
                "instances": {
                    "id": 123,
                    "actual_status": "running",
                    "ssh_host": "ssh.example",
                    "ssh_port": 12345,
                }
            },
        ),
    )

    def fake_stage(*, job_dir, **_kwargs):
        root = Path(job_dir)
        root.mkdir(parents=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (root / name).write_text("https://objects.example/value\n")
        return {"status": "completed"}

    dispatches: list[str] = []

    def fake_execute(*, command, output_log_path, **_kwargs):
        dispatches.append(command)
        path = Path(output_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "warm_runtime.log":
            path.write_text(
                "BLUEPRINT_ARENA_WARM_DEPENDENCY_CACHE_HIT:sha256:test\n"
                "BLUEPRINT_ARENA_WARM_PROVIDER_OUTPUT_UPLOAD_OK\n"
            )
        else:
            path.write_text("1234\n")
        return {"http_status_code": 200, "output_log_path": str(path)}

    def fake_download(*, destination, **_kwargs):
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr(
                "native_task_arena_control_result.v1.json",
                json.dumps(
                    {
                        "status": "completed",
                        "candidate_policy_queried": False,
                        "blockers": [],
                    }
                ),
            )
        return True

    monkeypatch.setattr(warm_vast, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(warm_vast, "_execute_and_fetch", fake_execute)
    monkeypatch.setattr(warm_vast, "_download_when_ready", fake_download)
    monkeypatch.setattr(
        warm_vast,
        "cleanup_staged_wam_provider_objects",
        lambda _path: {"all_objects_absent": True},
    )
    monkeypatch.setattr(
        warm_vast,
        "_close_warm_instance",
        lambda **_kwargs: {
            "status": "completed",
            "provider_instance_absent": True,
            "continuing_spend_from_this_run": False,
        },
    )

    result = warm_vast.run_native_task_arena_warm_controls_vast(
        job_dir=tmp_path / "job",
        prepared_bundle=prepared,
        warm_session=session,
        warm_attempt_authority=issued,
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        execute=True,
    )

    assert result["status"] == "completed"
    assert result["provider_allocations_performed"] == 0
    assert result["provider_instance_id"] == 123
    assert result["continuing_spend_from_this_run"] is False
    assert result["warm_session_closeout"]["provider_instance_absent"] is True
    assert Path(result["artifact_manifest_path"]).is_file()
    assert Path(result["teardown_manifest_path"]).is_file()
    teardown = json.loads(Path(result["teardown_manifest_path"]).read_text())
    assert teardown["continuing_spend_from_this_run"] is False
    assert teardown["vast_instance_ids"] == [123]
    assert len(dispatches) == 2
