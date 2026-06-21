from __future__ import annotations

import json
import os
import sys
import types
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import robot_eval_provider_input_setup as setup


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bundle_helpers_select_files_and_archive_names(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    (capture_root / "pipeline" / "robot_eval_dataset").mkdir(parents=True)
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")
    (capture_root / "pipeline" / "robot_eval_dataset" / "site_card.json").write_text(
        "{}", encoding="utf-8"
    )
    (capture_root / "raw").mkdir()
    (capture_root / "raw" / "walkthrough.mp4").write_bytes(b"video")

    without_media = list(
        setup._iter_bundle_files(
            capture_root,
            include_raw_media=False,
            include_paths=(
                "capture_descriptor.json",
                "pipeline/robot_eval_dataset",
                "raw/walkthrough.mp4",
                "missing.json",
            ),
        )
    )
    with_media = list(
        setup._iter_bundle_files(
            capture_root,
            include_raw_media=True,
            include_paths=("raw/walkthrough.mp4",),
        )
    )

    assert capture_root / "raw" / "walkthrough.mp4" not in without_media
    assert capture_root / "raw" / "walkthrough.mp4" in with_media
    assert setup._bundle_arcname(
        Path("plain-capture"),
        Path("plain-capture") / "capture_descriptor.json",
    ) == Path("capture-root") / "capture_descriptor.json"
    assert setup._bundle_arcname(
        Path("scenes") / "scene-1" / "captures" / "capture-1",
        Path("scenes") / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json",
    ) == Path("capture-root") / "capture_descriptor.json"

    staged_root = (
        tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    )
    (staged_root / "raw").mkdir(parents=True)
    (staged_root / "raw" / "manifest.json").write_text("{}", encoding="utf-8")
    assert setup._bundle_arcname(staged_root, staged_root / "raw" / "manifest.json") == Path(
        "bucket/scenes/scene-1/captures/capture-1/raw/manifest.json"
    )

    bundle = setup.build_capture_root_bundle(
        capture_root=capture_root,
        output_path=tmp_path / "bundle" / "capture-root.zip",
        include_raw_media=True,
        include_paths=("capture_descriptor.json", "pipeline/robot_eval_dataset", "raw/walkthrough.mp4"),
    )
    with zipfile.ZipFile(bundle["path"]) as archive:
        assert sorted(archive.namelist()) == [
            "capture-root/capture_descriptor.json",
            "capture-root/pipeline/robot_eval_dataset/site_card.json",
            "capture-root/raw/walkthrough.mp4",
        ]


def test_upload_helpers_cover_supported_schemes_and_error_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")

    class RunResult:
        stdout = "abc123def456\n"

    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: RunResult())
    assert setup.default_image_ref(simulator="mujoco", repo_root=tmp_path).endswith(
        ":20260621-abc123def456"
    )

    def fail_run(*args: object, **kwargs: object) -> object:
        raise RuntimeError("git unavailable")

    monkeypatch.setattr(setup.subprocess, "run", fail_run)
    assert setup.default_image_ref(simulator="isaac", repo_root=tmp_path).endswith("-unknownsha")
    assert setup._string("text") == "text"
    assert setup._string(123) == ""

    uploaded: dict[str, object] = {}

    class FakeBlob:
        def upload_from_filename(self, filename: str) -> None:
            uploaded["gs_filename"] = filename

    class FakeBucket:
        def blob(self, key: str) -> FakeBlob:
            uploaded["gs_key"] = key
            return FakeBlob()

    class FakeStorageClient:
        def bucket(self, bucket: str) -> FakeBucket:
            uploaded["gs_bucket"] = bucket
            return FakeBucket()

    storage_module = types.SimpleNamespace(Client=lambda: FakeStorageClient())
    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    cloud_module.storage = storage_module
    google_module.cloud = cloud_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)
    assert setup._upload_file_to_gs(source, "gs://bucket/path/source.txt")["status"] == "uploaded"
    assert uploaded == {
        "gs_bucket": "bucket",
        "gs_key": "path/source.txt",
        "gs_filename": str(source),
    }
    monkeypatch.setattr(
        setup,
        "_upload_file_to_gs",
        lambda path, destination_uri: {
            "status": "uploaded",
            "source": str(path),
            "destination_uri": destination_uri,
            "storage_scheme": "gs",
        },
    )
    assert setup.upload_file(source, "gs://bucket/path/source.txt")["storage_scheme"] == "gs"

    class FakeBoto3:
        def client(self, service: str, **kwargs: object) -> object:
            uploaded["s3_service"] = service
            uploaded["s3_kwargs"] = kwargs

            class Client:
                def upload_file(self, filename: str, bucket: str, key: str) -> None:
                    uploaded["s3_upload"] = (filename, bucket, key)

            return Client()

    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3())
    monkeypatch.setenv("BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL", "https://r2.example")
    monkeypatch.setenv("AWS_REGION", "auto")
    r2_upload = setup._upload_file_to_s3_compatible(source, "r2://bucket/path/source.txt")
    assert r2_upload["storage_scheme"] == "r2"
    assert uploaded["s3_kwargs"] == {
        "endpoint_url": "https://r2.example",
        "region_name": "auto",
    }

    file_destination = tmp_path / "copied.txt"
    assert setup.upload_file(source, str(file_destination))["storage_scheme"] == "file"
    assert file_destination.read_text(encoding="utf-8") == "payload"
    assert setup.upload_file(source, "ftp://example/source.txt")["blockers"] == [
        "unsupported_upload_uri_scheme:ftp"
    ]

    def denied_upload(path: Path, destination_uri: str) -> dict[str, object]:
        raise RuntimeError("AccessDenied")

    monkeypatch.setattr(setup, "_upload_file_to_s3_compatible", denied_upload)
    blocked = setup.upload_file(source, "s3://bucket/source.txt")
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == ["upload_failed:s3_access_denied"]
    assert setup._classify_upload_error(
        scheme="gs",
        error=RuntimeError("Billing account absent"),
    ) == "upload_failed:gs_billing_account_disabled"
    assert setup._classify_upload_error(scheme="s3", error=RuntimeError("403")) == (
        "upload_failed:s3_forbidden"
    )
    assert setup._classify_upload_error(scheme="file", error=OSError("disk full")) == (
        "upload_failed:OSError"
    )


def test_manifest_scripts_rewrite_and_annotation_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESTORED_ENV", "before")
    with setup._patched_env({"NEW_ENV": "value", "RESTORED_ENV": ""}):
        assert os.environ["NEW_ENV"] == "value"
        assert "RESTORED_ENV" not in os.environ
    assert "NEW_ENV" not in os.environ
    assert os.environ["RESTORED_ENV"] == "before"

    env_file = setup._write_env_file(tmp_path / "env" / "provider.sh", {"A": "one two"})
    assert "export A='one two'" in Path(env_file["path"]).read_text(encoding="utf-8")
    assert setup._dedupe([" a ", "a", "", "b"]) == ["a", "b"]
    assert setup._storage_upload_commands(source="/tmp/a", destination_uri="gs://bucket/a") == [
        'gcloud storage cp "/tmp/a" "gs://bucket/a"'
    ]
    assert setup._storage_upload_commands(source="/tmp/a", destination_uri="s3://bucket/a") == [
        'aws s3 cp "/tmp/a" "s3://bucket/a"'
    ]
    assert len(setup._storage_upload_commands(source="/tmp/a", destination_uri="r2://bucket/a")) == 2
    assert setup._storage_upload_commands(source="/tmp/a", destination_uri="file:///tmp/b") == [
        'cp "/tmp/a" "file:///tmp/b"'
    ]
    assert setup._storage_upload_commands(source="/tmp/a", destination_uri="https://host/a") == [
        '# Upload "/tmp/a" to provider-readable URI "https://host/a".'
    ]
    assert setup._worker_entrypoint_command(
        allow_simulator_execution=True,
        allowed_simulators=("mujoco",),
        simulator_commands={"mujoco": "python run.py"},
    ) == (
        "blueprint-run-robot-eval-worker --allow-simulator-execution "
        "--allowed-simulator mujoco --simulator-command 'mujoco=python run.py'"
    )

    assert setup._rewrite_provider_launch_worker_command(
        path=tmp_path / "missing.json",
        allow_simulator_execution=False,
        allowed_simulators=(),
        simulator_commands=None,
    )["status"] == "not_available"
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("[]", encoding="utf-8")
    assert setup._rewrite_provider_launch_worker_command(
        path=invalid_path,
        allow_simulator_execution=False,
        allowed_simulators=(),
        simulator_commands=None,
    )["blockers"] == ["invalid_provider_launch_json"]
    request_path = tmp_path / "request.json"
    _write_json(request_path, {"provider_request_shape": {"env": {}}})
    rewritten = setup._rewrite_provider_launch_worker_command(
        path=request_path,
        allow_simulator_execution=True,
        allowed_simulators=("mujoco",),
        simulator_commands={"mujoco": "python sim.py"},
    )
    assert rewritten["status"] == "updated"
    assert "python sim.py" in json.loads(request_path.read_text(encoding="utf-8"))[
        "provider_request_shape"
    ]["command"]

    script = setup._write_publish_resolution_script(
        path=tmp_path / "publish.sh",
        image_ref="registry/worker:tag",
        simulator="isaac",
        bundle_path="/tmp/capture-root.zip",
        bundle_uri="r2://bucket/job/capture-root.zip",
        worker_manifest_path="/tmp/worker.json",
        worker_manifest_uri="s3://bucket/job/worker.json",
    )
    script_text = Path(script["path"]).read_text(encoding="utf-8")
    assert "deploy/docker/robot_eval_worker/isaac/Dockerfile" in script_text
    assert "--endpoint-url" in script_text
    assert setup._read_optional_mapping(tmp_path / "none.json") == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert setup._read_optional_mapping(list_json) == {}
    mapping_json = tmp_path / "mapping.json"
    _write_json(mapping_json, {"ok": True})
    assert setup._read_optional_mapping(mapping_json) == {"ok": True}

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    setup._refresh_remote_cloud_execution_closure_manifest(
        job_dir=job_dir,
        job_id="job-1",
        provisioner="runpod",
        simulator="mujoco",
    )
    assert not (job_dir / "remote_cloud_execution_closure_manifest.json").exists()
    _write_json(job_dir / "gpu_provider_launch_request.json", {"status": "ready"})
    _write_json(job_dir / "worker_manifest.json", {"worker": True})
    monkeypatch.setattr(
        setup,
        "_remote_cloud_execution_closure_manifest",
        lambda **kwargs: {"status": "closure", "job_id": kwargs["job_id"]},
    )
    setup._refresh_remote_cloud_execution_closure_manifest(
        job_dir=job_dir,
        job_id="job-1",
        provisioner="runpod",
        simulator="mujoco",
    )
    assert json.loads(
        (job_dir / "remote_cloud_execution_closure_manifest.json").read_text(encoding="utf-8")
    ) == {"status": "closure", "job_id": "job-1"}

    setup._annotate_provider_launch_request(
        job_dir=tmp_path / "no-request",
        job_id="job-1",
        provisioner="runpod",
        simulator="mujoco",
        setup_manifest={},
        setup_manifest_path=tmp_path / "manifest.json",
    )
    bad_request_dir = tmp_path / "bad-request"
    _write_json(bad_request_dir / "gpu_provider_launch_request.json", [])
    setup._annotate_provider_launch_request(
        job_dir=bad_request_dir,
        job_id="job-1",
        provisioner="runpod",
        simulator="mujoco",
        setup_manifest={},
        setup_manifest_path=tmp_path / "manifest.json",
    )
    assert json.loads(
        (bad_request_dir / "gpu_provider_launch_request.json").read_text(encoding="utf-8")
    ) == []

    annotated_dir = tmp_path / "annotated"
    _write_json(
        annotated_dir / "gpu_provider_launch_request.json",
        {"status": "ready", "blockers": ["existing"]},
    )
    refreshed: list[str] = []
    monkeypatch.setattr(
        setup,
        "_refresh_remote_cloud_execution_closure_manifest",
        lambda **kwargs: refreshed.append(kwargs["job_id"]),
    )
    setup._annotate_provider_launch_request(
        job_dir=annotated_dir,
        job_id="job-2",
        provisioner="runpod",
        simulator="mujoco",
        setup_manifest={
            "status": "prepared_with_external_blockers",
            "blockers": ["missing_upload", "missing_upload"],
            "proof_boundary": {
                "provider_inputs_uploaded": True,
                "image_ref_published_proven": True,
            },
            "capture_root_bundle_uri": "r2://bucket/capture-root.zip",
            "worker_manifest_uri": "r2://bucket/worker.json",
            "artifact_output_uri": "r2://bucket/artifacts",
            "publish_resolution": {"path": "publish.sh"},
        },
        setup_manifest_path=tmp_path / "manifest.json",
    )
    annotated = json.loads(
        (annotated_dir / "gpu_provider_launch_request.json").read_text(encoding="utf-8")
    )
    assert annotated["status"] == "blocked_provider_input_setup"
    assert annotated["blockers"] == [
        "existing",
        "provider_input_setup_blocked",
        "missing_upload",
    ]
    assert annotated["provider_input_setup"]["provider_inputs_uploaded"] is True
    assert refreshed == ["job-2"]


def test_prepare_provider_inputs_uses_job_builder_uploads_and_status_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    (capture_root / "raw").mkdir(parents=True)
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")
    job_request = tmp_path / "job-request.json"
    _write_json(job_request, {"job_id": "job"})

    def fake_build_robot_eval_job(**kwargs: object) -> dict[str, object]:
        job_id = str(kwargs["job_id"])
        job_dir = capture_root / "pipeline" / "robot_eval_jobs" / job_id
        job_dir.mkdir(parents=True)
        if "missing-worker" not in job_id:
            _write_json(job_dir / "worker_manifest.json", {"job_id": job_id})
        _write_json(
            job_dir / "gpu_provider_launch_request.json",
            {"status": "ready", "provider_request_shape": {}, "blockers": []},
        )
        image_env_var = setup.WORKER_IMAGE_REF_ENV_BY_SIMULATOR.get(
            str(kwargs["simulator"]),
            "BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF",
        )
        assert os.environ[image_env_var]
        assert os.environ[setup.WORKER_CAPTURE_ROOT_BUNDLE_URI_ENV].endswith("capture-root.zip")
        return {"status": "ready", "job_dir": str(job_dir)}

    uploads: list[str] = []

    def fake_upload_file(source: str | Path, destination_uri: str) -> dict[str, object]:
        uploads.append(destination_uri)
        return {
            "status": "uploaded",
            "source": str(source),
            "destination_uri": destination_uri,
            "storage_scheme": "r2",
        }

    monkeypatch.setattr(setup, "build_robot_eval_job", fake_build_robot_eval_job)
    monkeypatch.setattr(setup, "upload_file", fake_upload_file)
    ready = setup.prepare_robot_eval_provider_inputs(
        capture_root=capture_root,
        job_request=job_request,
        job_id="ready-job",
        artifact_root_uri="r2://bucket/jobs/ready-job",
        simulator="mujoco",
        provisioner="runpod",
        image_ref="registry/worker:ready",
        output_dir=tmp_path / "ready-inputs",
        upload=True,
        allow_gpu_provisioning=True,
        allow_simulator_execution=True,
        allowed_simulators=("mujoco",),
        simulator_commands={"mujoco": "python sim.py"},
    )

    assert ready["status"] == "ready_for_provider_launcher_inputs"
    assert ready["proof_boundary"]["provider_inputs_uploaded"] is True  # type: ignore[index]
    assert uploads == [
        "r2://bucket/jobs/ready-job/capture-root.zip",
        "r2://bucket/jobs/ready-job/worker_manifest.json",
    ]
    provider_request = json.loads(
        (
            capture_root
            / "pipeline"
            / "robot_eval_jobs"
            / "ready-job"
            / "gpu_provider_launch_request.json"
        ).read_text(encoding="utf-8")
    )
    assert provider_request["provider_input_setup"]["image_ref_published_proven"] is True

    uploads.clear()
    blocked = setup.prepare_robot_eval_provider_inputs(
        capture_root=capture_root,
        job_request=job_request,
        job_id="missing-worker-job",
        artifact_root_uri="r2://bucket/jobs/missing-worker-job",
        simulator="isaac",
        provisioner="runpod",
        image_ref="registry/worker:isaac",
        output_dir=tmp_path / "blocked-inputs",
        upload=True,
    )

    assert blocked["status"] == "prepared_with_external_blockers"
    assert blocked["blockers"] == [
        "worker_manifest_upload_missing",
        "provider_inputs_upload_not_proven",
    ]
    assert blocked["image_ref"]["candidate_build_command"] is None  # type: ignore[index]
    assert uploads == ["r2://bucket/jobs/missing-worker-job/capture-root.zip"]

    candidate = setup.prepare_robot_eval_provider_inputs(
        capture_root=capture_root,
        job_request=job_request,
        job_id="candidate-image-job",
        artifact_root_uri="file:///tmp/candidate-image-job",
        simulator="mujoco",
        provisioner="fixture_local",
        output_dir=tmp_path / "candidate-inputs",
        upload=False,
    )

    assert candidate["status"] == "prepared_with_external_blockers"
    assert candidate["blockers"] == ["worker_image_ref_is_candidate_until_built_and_pushed"]


def test_parser_main_and_simulator_command_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert setup._parse_simulator_commands(["mujoco=python sim.py"]) == {
        "mujoco": "python sim.py"
    }
    with pytest.raises(ValueError, match="--simulator-command"):
        setup._parse_simulator_commands(["missing-command"])

    calls: list[dict[str, object]] = []

    def fake_prepare_robot_eval_provider_inputs(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        status = "ready_for_provider_launcher_inputs" if len(calls) == 1 else "prepared"
        blockers = [] if len(calls) == 1 else ["blocked"]
        return {
            "status": status,
            "blockers": blockers,
            "env_file": {"path": str(tmp_path / f"env-{len(calls)}.sh")},
        }

    monkeypatch.setattr(setup, "prepare_robot_eval_provider_inputs", fake_prepare_robot_eval_provider_inputs)
    base_args = [
        "--capture-root",
        str(tmp_path / "capture"),
        "--job-request",
        str(tmp_path / "request.json"),
        "--job-id",
        "job-1",
        "--artifact-root-uri",
        "r2://bucket/job-1",
        "--image-ref",
        "registry/worker:tag",
        "--upload",
        "--include-raw-media",
        "--allow-simulator-execution",
        "--allowed-simulator",
        "mujoco",
        "--simulator-command",
        "mujoco=python sim.py",
        "--timeout-seconds",
        "30",
        "--budget-usd",
        "1.5",
    ]

    assert setup.main(base_args) == 0
    assert setup.main(base_args) == 2
    output = capsys.readouterr().out
    assert "provider_input_setup_manifest.json" in output
    assert "blockers=blocked" in output
    assert calls[0]["include_raw_media"] is True
    assert calls[0]["simulator_commands"] == {"mujoco": "python sim.py"}
