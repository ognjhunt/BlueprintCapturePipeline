from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

import pytest

from blueprint_pipeline import wam_compute_providers as providers


def _bundle(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"provider-bundle")
    return path


def _spec(bundle_path: Path, **overrides: Any) -> providers.WamComputeLaunchSpec:
    values: dict[str, Any] = {
        "name": "test-wam",
        "bundle_path": bundle_path,
        "provider_bundle_kind": "wam",
        "image": "docker.io/example/wam:latest",
        "provider_bundle_url": "https://store.example/bundle.zip?secret",
        "provider_output_put_url": "https://store.example/output.zip?secret",
        "provider_output_get_url": "https://store.example/output.zip?secret",
        "expected_video_count": 1,
        "max_wait_seconds": 1,
    }
    values.update(overrides)
    return providers.WamComputeLaunchSpec(**values)


def test_wam_compute_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(providers.PROVIDER_ORDER_ENV, "vast,runpod")

    assert providers.get_wam_compute_provider("vast").name == "vast"
    assert providers.get_wam_compute_provider("runpod").name == "runpod"
    assert providers.get_wam_compute_provider("auto").name == "vast"
    listed = providers.list_wam_compute_providers()

    assert {row["provider"] for row in listed} == {"vast", "runpod"}
    with pytest.raises(ValueError, match="unknown_wam_compute_provider"):
        providers.get_wam_compute_provider("lambda")


def test_wam_compute_launch_spec_validates_provider_bundle_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported_provider_bundle_kind"):
        _spec(_bundle(tmp_path / "bundle.zip"), provider_bundle_kind="other")


def test_vast_no_paid_launch_blocks_without_calling_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_create(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("no-spend path must not call Vast create")

    monkeypatch.setattr(providers, "create_async_vast_wam_run", fail_create)
    spec = _spec(_bundle(tmp_path / "bundle.zip"))
    result = providers.VastWamComputeProvider().create(
        spec,
        tmp_path / "vast_provider_run",
        allow_paid_launch=False,
    )

    assert result.status == "blocked"
    assert result.continuing_spend_from_this_run is False
    assert result.output_availability == "not_available"
    assert "paid_wam_compute_launch_not_authorized:vast" in result.blockers
    persisted = tmp_path / "vast_provider_run" / "wam_compute_run_result.json"
    assert persisted.is_file()
    persisted_text = persisted.read_text(encoding="utf-8")
    assert "store.example" not in persisted_text
    assert "X-Amz-Signature" not in persisted_text


def test_vast_paid_launch_requires_existing_env_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_create(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("missing env gate must not call Vast create")

    monkeypatch.delenv(providers.VAST_WAM_PAID_LAUNCH_GATE_ENV, raising=False)
    monkeypatch.setattr(providers, "create_async_vast_wam_run", fail_create)
    spec = _spec(_bundle(tmp_path / "bundle.zip"))

    result = providers.VastWamComputeProvider().create(
        spec,
        tmp_path / "vast_provider_run",
        allow_paid_launch=True,
    )

    assert result.status == "blocked"
    assert f"missing_env_{providers.VAST_WAM_PAID_LAUNCH_GATE_ENV}" in result.blockers
    assert result.continuing_spend_from_this_run is False


def test_runpod_no_paid_launch_blocks_without_calling_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_create(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("no-spend path must not call RunPod create")

    monkeypatch.setattr(providers, "create_runpod_wam_async_run", fail_create)
    spec = _spec(_bundle(tmp_path / "bundle.zip"))
    result = providers.RunPodWamComputeProvider().create(
        spec,
        tmp_path / "runpod_provider_run",
        allow_paid_launch=False,
    )

    assert result.status == "blocked"
    assert result.continuing_spend_from_this_run is False
    assert result.output_availability == "not_available"
    assert "paid_wam_compute_launch_not_authorized:runpod" in result.blockers


def test_inspect_output_rejects_zero_byte_zip(tmp_path: Path) -> None:
    output_zip = tmp_path / "runpod_provider_runtime_output.zip"
    output_zip.write_bytes(b"")

    inspection = providers.RunPodWamComputeProvider().inspect_output(tmp_path, output_zip)

    assert inspection["status"] == "not_available"
    assert inspection["zip_present"] is False
    assert inspection["mp4_count"] == 0
    assert "provider_runtime_output_zip_missing_or_empty" in inspection[
        "runtime_result_blockers"
    ]


def test_run_wam_compute_job_normalizes_vast_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path / "bundle.zip")
    captured_create: dict[str, Any] = {}
    captured_poll: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> dict[str, Any]:
        captured_create.update(kwargs)
        return {
            "schema_version": "vast_wam_async_create_manifest.v1",
            "generated_at": "now",
            "status": "instance_created",
            "instance_id": 123,
            "output_path": str(kwargs["output_path"]),
            "blockers": [],
            "raw_secret_values_recorded": False,
        }

    def fake_poll(**kwargs: Any) -> dict[str, Any]:
        captured_poll.update(kwargs)
        output_zip = Path(kwargs["job_dir"]) / "vast_provider_runtime_output.zip"
        with zipfile.ZipFile(output_zip, "w") as archive:
            archive.writestr("oscar_generated_rollout.mp4", b"fake")
        return {
            "schema_version": "vast_wam_async_poll_manifest.v1",
            "generated_at": "now",
            "status": "completed",
            "instance_id": 123,
            "instance_status": "exited",
            "provider_command_status": "completed",
            "provider_command_blockers": [],
            "output_zip_present": True,
            "runtime_result_status": "completed",
            "runtime_result_blockers": [],
            "mp4_count": 1,
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
            "raw_secret_values_recorded": False,
        }

    monkeypatch.setattr(providers, "create_async_vast_wam_run", fake_create)
    monkeypatch.setattr(providers, "poll_async_vast_wam_run", fake_poll)
    monkeypatch.setenv(providers.VAST_WAM_PAID_LAUNCH_GATE_ENV, "1")
    monkeypatch.setattr(
        providers,
        "_inspect_provider_runtime_output_zip",
        lambda *_args, **_kwargs: {
            "zip_present": True,
            "runtime_result_status": "completed",
            "runtime_result_blockers": [],
            "mp4_count": 1,
            "mp4_validation": {"files": []},
        },
    )

    result = providers.run_wam_compute_job(
        spec=_spec(bundle, max_wait_seconds=7, min_gpu_ram_mb=48000),
        job_dir=tmp_path / "compute",
        provider_order=["vast"],
        allow_paid_launch=True,
    )

    assert result.status == "completed"
    assert result.provider == "vast"
    assert result.instance_id == "123"
    assert result.output_zip_present is True
    assert result.output_availability == "available"
    assert result.continuing_spend_from_this_run is False
    assert captured_create["allow_paid_vast_launch"] is True
    assert captured_create["min_gpu_ram_mb"] == 48000
    assert captured_poll["max_wait_seconds"] == 7
    assert captured_poll["teardown"] is True


def test_run_wam_compute_job_normalizes_runpod_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path / "bundle.zip")
    captured_create: dict[str, Any] = {}
    captured_poll: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> dict[str, Any]:
        captured_create.update(kwargs)
        return {
            "schema_version": "runpod_wam_async_create_manifest.v1",
            "generated_at": "now",
            "status": "pod_created",
            "pod_id": "pod-123",
            "output_path": str(kwargs["output_path"]),
            "blockers": [],
            "raw_secret_values_recorded": False,
        }

    def fake_poll(**kwargs: Any) -> dict[str, Any]:
        captured_poll.update(kwargs)
        output_zip = Path(kwargs["job_dir"]) / "runpod_provider_runtime_output.zip"
        with zipfile.ZipFile(output_zip, "w") as archive:
            archive.writestr("oscar_generated_rollout.mp4", b"fake")
        return {
            "schema_version": "runpod_wam_async_poll_manifest.v1",
            "generated_at": "now",
            "status": "completed",
            "pod_id": "pod-123",
            "pod_status": "EXITED",
            "provider_command_status": "completed",
            "provider_command_blockers": [],
            "output_zip_present": True,
            "provider_runtime_output_zip_path": str(
                Path(kwargs["job_dir"]) / "runpod_provider_runtime_output.zip"
            ),
            "runtime_result_status": "completed",
            "runtime_result_blockers": [],
            "mp4_count": 1,
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
            "raw_secret_values_recorded": False,
        }

    monkeypatch.setattr(providers, "create_runpod_wam_async_run", fake_create)
    monkeypatch.setattr(providers, "poll_runpod_wam_async_run", fake_poll)
    monkeypatch.setattr(
        providers,
        "_inspect_provider_runtime_output_zip",
        lambda *_args, **_kwargs: {
            "zip_present": True,
            "runtime_result_status": "completed",
            "runtime_result_blockers": [],
            "mp4_count": 1,
            "mp4_validation": {"files": []},
        },
    )

    result = providers.run_wam_compute_job(
        spec=_spec(bundle, image="docker.io/example/runpod-wam:latest"),
        job_dir=tmp_path / "compute",
        provider_order=["runpod"],
        allow_paid_launch=True,
    )

    assert result.status == "completed"
    assert result.provider == "runpod"
    assert result.instance_id == "pod-123"
    assert result.output_availability == "available"
    assert captured_create["allow_paid_runpod_launch"] is True
    assert captured_create["image_name"] == "docker.io/example/runpod-wam:latest"
    assert captured_poll["teardown"] is True


def test_run_wam_compute_job_records_runpod_stop_teardown_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path / "bundle.zip")

    def fake_create(**kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": "runpod_wam_async_create_manifest.v1",
            "generated_at": "now",
            "status": "pod_created",
            "pod_id": "pod-123",
            "output_path": str(kwargs["output_path"]),
            "blockers": [],
            "raw_secret_values_recorded": False,
        }

    def fake_poll(**kwargs: Any) -> dict[str, Any]:
        provider_job = Path(kwargs["job_dir"])
        output_zip = provider_job / "runpod_provider_runtime_output.zip"
        with zipfile.ZipFile(output_zip, "w") as archive:
            archive.writestr("oscar_generated_rollout.mp4", b"fake")
        (provider_job / "runpod_wam_async_stop_manifest.json").write_text(
            (
                '{"schema_version":"runpod_wam_async_stop_manifest.v1",'
                '"status":"completed","pod_id":"pod-123"}'
            ),
            encoding="utf-8",
        )
        return {
            "schema_version": "runpod_wam_async_poll_manifest.v1",
            "generated_at": "now",
            "status": "completed",
            "pod_id": "pod-123",
            "pod_status": "RUNNING",
            "provider_command_status": "completed",
            "provider_command_blockers": [],
            "output_zip_present": True,
            "provider_runtime_output_zip_path": str(output_zip),
            "runtime_result_status": "completed",
            "runtime_result_blockers": [],
            "mp4_count": 1,
            "teardown_action": "stop",
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
            "raw_secret_values_recorded": False,
        }

    monkeypatch.setattr(providers, "create_runpod_wam_async_run", fake_create)
    monkeypatch.setattr(providers, "poll_runpod_wam_async_run", fake_poll)
    monkeypatch.setattr(
        providers,
        "_inspect_provider_runtime_output_zip",
        lambda *_args, **_kwargs: {
            "zip_present": True,
            "runtime_result_status": "completed",
            "runtime_result_blockers": [],
            "mp4_count": 1,
            "mp4_validation": {"files": []},
        },
    )

    result = providers.run_wam_compute_job(
        spec=_spec(bundle),
        job_dir=tmp_path / "compute",
        provider_order=["runpod"],
        allow_paid_launch=True,
    )

    assert result.status == "completed"
    assert result.provider == "runpod"
    assert result.teardown_manifest_path is not None
    assert result.teardown_manifest_path.endswith("runpod_wam_async_stop_manifest.json")
    assert result.teardown_status == "completed"
    assert result.teardown_performed is True
    assert result.continuing_spend_from_this_run is False


def test_run_wam_compute_job_blocks_when_expected_mp4_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path / "bundle.zip")

    monkeypatch.setattr(
        providers,
        "create_runpod_wam_async_run",
        lambda **kwargs: {
            "status": "pod_created",
            "pod_id": "pod-123",
            "output_path": str(kwargs["output_path"]),
            "raw_secret_values_recorded": False,
        },
    )

    def _fake_runpod_poll_without_mp4(**kwargs: Any) -> dict[str, Any]:
        output_zip = Path(kwargs["job_dir"]) / "runpod_provider_runtime_output.zip"
        with zipfile.ZipFile(output_zip, "w") as archive:
            archive.writestr("wam_runtime_result.json", '{"status":"completed"}')
        return {
            "status": "completed",
            "pod_id": "pod-123",
            "provider_command_status": "completed",
            "output_zip_present": True,
            "provider_runtime_output_zip_path": str(output_zip),
            "runtime_result_status": "completed",
            "mp4_count": 0,
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
            "raw_secret_values_recorded": False,
        }

    monkeypatch.setattr(
        providers,
        "poll_runpod_wam_async_run",
        _fake_runpod_poll_without_mp4,
    )
    monkeypatch.setattr(
        providers,
        "_inspect_provider_runtime_output_zip",
        lambda *_args, **_kwargs: {
            "zip_present": True,
            "runtime_result_status": "completed",
            "runtime_result_blockers": [],
            "mp4_count": 0,
            "mp4_validation": {"files": []},
        },
    )

    result = providers.run_wam_compute_job(
        spec=_spec(bundle, expected_video_count=1),
        job_dir=tmp_path / "compute",
        provider_order=["runpod"],
        allow_paid_launch=True,
    )

    assert result.status == "blocked"
    assert result.output_availability == "zip_present_but_expected_generated_videos_missing"
    assert "zip_present_but_expected_generated_videos_missing" in result.blockers
