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

    assert providers.get_wam_compute_provider("deepinfra").name == "deepinfra"
    assert providers.get_wam_compute_provider("vast").name == "vast"
    assert providers.get_wam_compute_provider("runpod").name == "runpod"
    assert providers.get_wam_compute_provider("auto").name == "vast"
    listed = providers.list_wam_compute_providers()

    assert {row["provider"] for row in listed} == {"deepinfra", "vast", "runpod"}
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


def test_deepinfra_no_paid_launch_writes_redacted_request_and_cost_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_post(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("no-spend path must not call DeepInfra")

    monkeypatch.setenv(providers.DEEPINFRA_API_KEY_ENV, "deepinfra-secret-value")
    monkeypatch.setenv(providers.DEEPINFRA_API_GATE_ENV, "1")
    monkeypatch.setattr(providers, "_deepinfra_post_json", fail_post)

    spec = _spec(_bundle(tmp_path / "bundle.zip"))
    result = providers.DeepInfraCosmos3NanoProvider().create(
        spec,
        tmp_path / "deepinfra_provider_run",
        allow_paid_launch=False,
    )

    assert result.status == "blocked"
    assert "paid_wam_compute_launch_not_authorized:deepinfra" in result.blockers
    request = tmp_path / "deepinfra_provider_run" / "deepinfra_cosmos3_request_manifest.json"
    cost = tmp_path / "deepinfra_provider_run" / "deepinfra_cosmos3_cost_control_ledger.json"
    execution = tmp_path / "deepinfra_provider_run" / "deepinfra_cosmos3_execution_manifest.json"
    assert request.is_file()
    assert cost.is_file()
    assert execution.is_file()
    persisted_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (request, cost, execution)
    )
    assert "deepinfra-secret-value" not in persisted_text
    assert "raw_secret_values_recorded" in persisted_text


def test_deepinfra_provider_downloads_video_and_writes_zip_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "provider_runtime/wam_provider_runtime_manifest.json",
            (
                '{"prompt":"Open the refrigerator from the robot point of view",'
                '"num_frames":45,"fps":15.0,"width":640,"height":480,"seed":99}'
            ),
        )

    captured_post: dict[str, Any] = {}
    captured_download: dict[str, Any] = {}

    def fake_post(**kwargs: Any) -> dict[str, Any]:
        captured_post.update(kwargs)
        return {
            "request_id": "di-req-1",
            "video_url": "/generated/deepinfra-cosmos3.mp4",
            "inference_status": {"status": "succeeded", "cost": 0.0324},
        }

    def fake_download(**kwargs: Any) -> dict[str, Any]:
        captured_download.update(kwargs)
        target = Path(kwargs["target_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake-mp4-bytes")
        return {"path": str(target), "size_bytes": target.stat().st_size, "sha256": "abc"}

    monkeypatch.setenv(providers.DEEPINFRA_API_GATE_ENV, "1")
    monkeypatch.setenv(providers.DEEPINFRA_API_KEY_ENV, "deepinfra-secret-value")
    monkeypatch.setattr(providers, "_deepinfra_post_json", fake_post)
    monkeypatch.setattr(providers, "_deepinfra_download_file", fake_download)
    monkeypatch.setattr(
        providers,
        "validate_generated_mp4_for_review",
        lambda path: {
            "schema_version": "wam_generated_video_review_validation.v1",
            "status": "completed",
            "path": str(path),
            "exists": True,
            "size_bytes": Path(path).stat().st_size,
            "frame_count": 45,
            "fps": 15.0,
            "width": 640,
            "height": 480,
            "blockers": [],
        },
    )
    monkeypatch.setattr(
        providers,
        "visual_smoke_generated_rollouts_for_review",
        lambda **kwargs: {
            "schema_version": "wam_generated_rollout_visual_smoke.v1",
            "status": "passed_visual_quality_smoke",
            "blockers": [],
            "review_usefulness_status": "reviewable_for_task_success",
            "review_usefulness_blockers": [],
            "rollout_count": len(kwargs["rollouts"]),
            "rollouts": [],
            "claim_boundary": {
                "visual_rollout_useful_for_task_success_review": True,
                "generated_observation_review_support_only": True,
            },
        },
    )
    monkeypatch.setattr(
        providers,
        "_inspect_provider_runtime_output_zip",
        lambda *_args, **_kwargs: {
            "zip_present": True,
            "runtime_result_status": "completed",
            "runtime_result_blockers": [],
            "mp4_count": 1,
            "mp4_validation": {"files": []},
            "extracted_video_paths": [str(tmp_path / "video.mp4")],
        },
    )

    result = providers.run_wam_compute_job(
        spec=_spec(bundle, expected_video_count=1),
        job_dir=tmp_path / "compute",
        provider_order=["deepinfra"],
        allow_paid_launch=True,
    )

    provider_job = tmp_path / "compute" / "deepinfra_provider_run"
    assert result.status == "completed"
    assert result.provider == "deepinfra"
    assert result.output_availability == "available"
    assert result.budget_ledger_path
    assert result.budget_ledger_path.endswith("deepinfra_cosmos3_cost_control_ledger.json")
    assert captured_post["payload"]["prompt"] == (
        "Open the refrigerator from the robot point of view"
    )
    assert captured_post["payload"]["duration_seconds"] == 3.0
    assert captured_download["url"] == "https://api.deepinfra.com/generated/deepinfra-cosmos3.mp4"
    output_zip = provider_job / "deepinfra_provider_runtime_output.zip"
    assert output_zip.is_file()
    with zipfile.ZipFile(output_zip) as archive:
        names = set(archive.namelist())
    assert {
        "deepinfra_cosmos3_generated_rollout.mp4",
        "wam_provider_output.json",
        "wam_runtime_result.json",
        "wam_rollout_visual_quality_report.json",
        "deepinfra_cosmos3_visual_quality_report.json",
        "deepinfra_cosmos3_request_manifest.json",
        "deepinfra_cosmos3_execution_manifest.json",
        "deepinfra_cosmos3_cost_control_ledger.json",
        "deepinfra_cosmos3_artifact_checksums.json",
    }.issubset(names)
    cost = providers._read_json_file(provider_job / "deepinfra_cosmos3_cost_control_ledger.json")
    assert cost["actual_cost_usd"] == 0.0324
    execution = providers._read_json_file(
        provider_job / "deepinfra_cosmos3_execution_manifest.json"
    )
    assert execution["output_zip_present"] is True
    checksums = providers._read_json_file(provider_job / "deepinfra_cosmos3_artifact_checksums.json")
    assert checksums["artifact_count"] >= 4
    checksum_names = {row["name"] for row in checksums["artifacts"]}
    assert {
        "deepinfra_provider_runtime_output.zip",
        "deepinfra_cosmos3_generated_rollout.mp4",
        "deepinfra_cosmos3_request_manifest.json",
        "deepinfra_cosmos3_execution_manifest.json",
        "deepinfra_cosmos3_cost_control_ledger.json",
        "deepinfra_cosmos3_visual_quality_report.json",
    }.issubset(checksum_names)
    provider_payload = providers._read_json_file(provider_job / "wam_provider_output.json")
    assert provider_payload["claim_boundary"]["deepinfra_api_success_is_not_task_success"] is True
    assert (
        provider_payload["claim_boundary"]["generated_world_rank_fidelity_result_proven"]
        is False
    )
    persisted_text = "\n".join(path.read_text(encoding="utf-8") for path in provider_job.glob("*.json"))
    assert "deepinfra-secret-value" not in persisted_text


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
        spec=_spec(
            bundle,
            max_wait_seconds=7,
            min_gpu_ram_mb=48000,
            min_reliability=0.99,
            require_direct_port=True,
            preferred_gpu_keywords=("RTX A6000", "L40S"),
            preferred_geolocation_regex="california|oregon",
            prefer_isaac_rt=False,
        ),
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
    assert captured_create["min_reliability"] == 0.99
    assert captured_create["require_direct_port"] is True
    assert captured_create["preferred_gpu_keywords"] == ("RTX A6000", "L40S")
    assert captured_create["preferred_geolocation_regex"] == "california|oregon"
    assert captured_create["prefer_isaac_rt"] is False
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
