from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import robot_eval_worker as rew


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _install_fake_gcs(monkeypatch: pytest.MonkeyPatch, uploaded: list[tuple[str, str]]) -> None:
    storage_mod = types.ModuleType("google.cloud.storage")

    class FakeBlob:
        def __init__(self, key: str) -> None:
            self.key = key

        def upload_from_filename(self, filename: str) -> None:
            uploaded.append((self.key, filename))

    class FakeBucket:
        def __init__(self, name: str) -> None:
            self.name = name

        def blob(self, key: str) -> FakeBlob:
            return FakeBlob(key)

    class FakeClient:
        def bucket(self, name: str) -> FakeBucket:
            return FakeBucket(name)

    storage_mod.Client = FakeClient  # type: ignore[attr-defined]
    cloud_mod = types.ModuleType("google.cloud")
    cloud_mod.storage = storage_mod  # type: ignore[attr-defined]
    google_mod = types.ModuleType("google")
    google_mod.cloud = cloud_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_mod)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_mod)


def _install_fake_boto3(monkeypatch: pytest.MonkeyPatch, uploaded: list[tuple[str, str, str]]) -> None:
    boto3_mod = types.ModuleType("boto3")

    class FakeClient:
        def download_file(self, bucket: str, key: str, filename: str) -> None:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            Path(filename).write_text(f"{bucket}/{key}", encoding="utf-8")

        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            uploaded.append((filename, bucket, key))

    boto3_mod.client = lambda service, **kwargs: FakeClient()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)


def test_robot_eval_worker_scalar_redaction_and_preflight_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_SECRET_TOKEN", "abcd-secret")
    assert rew._string_list("") == []
    assert rew._string_list(["x", 3, ""]) == ["x"]
    assert rew._bool("true") is True
    assert rew._bool("off") is False
    assert rew._bool("maybe") is None
    assert not rew._artifact_output_uri_is_provider_writable("", live_provider=False)
    assert rew._artifact_output_uri_is_provider_writable(str(tmp_path), live_provider=False)
    assert rew._output_text(b"hello") == "hello"
    assert rew._output_text(None) == ""
    assert rew._redact_runtime_value(("https://x.test/a?x-goog-signature=secret",)) == [
        "https://x.test/a?x-goog-signature=<redacted:signed-url-signature>"
    ]
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert rew._optional_json_mapping(bad_json) == {}
    assert rew._runtime_preflight_command(
        {"runtime_preflight_commands": {"mujoco": "echo ok"}}, "mujoco"
    ) == "echo ok"

    kwargs = {
        "manifest_uri": "manifest.json",
        "job_id": "job-1",
        "capture_root": str(tmp_path / "capture"),
        "provisioner": "runpod",
        "simulator": "mujoco",
        "contract": {},
        "timeout_seconds": 1,
        "generated_at": "2026-06-20T00:00:00Z",
        "payload": {},
    }
    blocked = rew._write_worker_runtime_preflight(
        work_dir=tmp_path / "preflight-no-gate",
        allow_simulator_execution=False,
        **kwargs,
    )
    assert blocked["blockers"] == ["missing_simulator_execution_gate_for_runtime_preflight"]
    invalid = rew._write_worker_runtime_preflight(
        work_dir=tmp_path / "preflight-invalid",
        allow_simulator_execution=True,
        payload={"runtime_preflight_command": "'unterminated"},
        **{key: value for key, value in kwargs.items() if key != "payload"},
    )
    assert invalid["blockers"] == ["invalid_runtime_preflight_command"]

    def fake_missing_run(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("missing")

    monkeypatch.setattr(rew.subprocess, "run", fake_missing_run)
    missing = rew._write_worker_runtime_preflight(
        work_dir=tmp_path / "preflight-missing",
        allow_simulator_execution=True,
        payload={"runtime_preflight_command": "missing-tool"},
        **{key: value for key, value in kwargs.items() if key != "payload"},
    )
    assert missing["blockers"] == ["missing_runtime_preflight_command_dependency"]

    def fake_timeout_run(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["slow"], timeout=1, output=b"abcd-secret", stderr=b"err")

    monkeypatch.setattr(rew.subprocess, "run", fake_timeout_run)
    timed_out = rew._write_worker_runtime_preflight(
        work_dir=tmp_path / "preflight-timeout",
        allow_simulator_execution=True,
        payload={
            "runtime_preflight_command": "slow",
            "secret_env_var_names": ["BLUEPRINT_SECRET_TOKEN"],
        },
        **{key: value for key, value in kwargs.items() if key != "payload"},
    )
    assert timed_out["blockers"] == ["runtime_preflight_command_timeout"]
    assert "<redacted:BLUEPRINT_SECRET_TOKEN>" in (
        tmp_path / "preflight-timeout" / "worker_runtime_preflight.stdout.log"
    ).read_text(encoding="utf-8")

    def fake_run(*args: object, **kwargs: object) -> object:
        env = kwargs["env"]
        _write_json(Path(env[rew.RUNTIME_PREFLIGHT_DETAIL_OUTPUT_ENV]), {"status": "warn", "blockers": ["detail"]})
        return SimpleNamespace(returncode=1, stdout="out", stderr="err")

    monkeypatch.setattr(rew.subprocess, "run", fake_run)
    failed = rew._write_worker_runtime_preflight(
        work_dir=tmp_path / "preflight-failed",
        allow_simulator_execution=True,
        payload={"runtime_preflight_command": "tool"},
        **{key: value for key, value in kwargs.items() if key != "payload"},
    )
    assert failed["detail_status"] == "warn"
    assert failed["blockers"] == ["runtime_preflight_command_failed"]


def test_robot_eval_worker_uri_bundle_and_upload_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads: list[tuple[str, str]] = []
    s3_uploads: list[tuple[str, str, str]] = []
    _install_fake_gcs(monkeypatch, uploads)
    _install_fake_boto3(monkeypatch, s3_uploads)
    monkeypatch.delenv("R2_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("BLUEPRINT_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    with pytest.raises(RuntimeError, match="r2:// storage requires"):
        rew._s3_compatible_endpoint_url("r2")
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://r2.local")
    monkeypatch.setenv("AWS_REGION", "us-test-1")

    assert rew._parse_s3_compatible_uri("s3://bucket/key.json") == ("s3", "bucket", "key.json")
    with pytest.raises(ValueError):
        rew._parse_s3_compatible_uri("https://bucket/key.json")
    with pytest.raises(ValueError):
        rew._parse_s3_compatible_uri("s3://bucket")
    assert rew._s3_compatible_endpoint_url("s3") == "https://r2.local"
    assert rew._s3_compatible_client("s3://bucket/key.json") is not None

    local_manifest = tmp_path / "manifest.json"
    _write_json(local_manifest, {"ok": True})
    assert rew._uri_to_local_path(str(local_manifest), tmp_path) == local_manifest

    class FakeResponse:
        status = 201

        def __init__(self, body: bytes = b'{"ok": true}') -> None:
            self.body = body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return self.body

    monkeypatch.setattr(rew.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    def fake_secure_fetch(*args, output_path=None, **kwargs):  # type: ignore[no-untyped-def]
        assert output_path is not None
        output_path.write_bytes(b'{"ok": true}')
        return type(
            "BoundedResponse",
            (),
            {"body": b"", "status": 200, "content_type": "application/json"},
        )()

    monkeypatch.setattr(rew, "fetch_bounded_https", fake_secure_fetch)
    assert rew._uri_to_local_path("https://trusted.invalid/manifest.json", tmp_path).is_file()
    monkeypatch.setattr(
        rew,
        "ensure_local_uri_path",
        lambda uri, *, gcs_root, scratch_dir: tmp_path / "mounted.json",
    )
    assert rew._uri_to_local_path("gs://bucket/manifest.json", tmp_path) == tmp_path / "mounted.json"
    assert rew._uri_to_local_path("s3://bucket/manifest.json", tmp_path).is_file()
    with pytest.raises(ValueError):
        rew._uri_to_local_path("ftp://host/manifest.json", tmp_path)

    assert rew._uri_to_local_file("https://trusted.invalid/bundle.zip", tmp_path, filename="bundle.zip").is_file()
    assert rew._uri_to_local_file("gs://bucket/bundle.zip", tmp_path, filename="bundle.zip") == tmp_path / "mounted.json"
    assert rew._uri_to_local_file("r2://bucket/bundle.zip", tmp_path, filename="bundle.zip").is_file()
    with pytest.raises(ValueError):
        rew._uri_to_local_file("ftp://host/bundle.zip", tmp_path, filename="bundle.zip")

    extract_root = tmp_path / "extract-root"
    extract_root.mkdir()
    _write_json(extract_root / "capture_descriptor.json", {})
    assert rew._find_extracted_capture_root(extract_root) == extract_root
    child_root = tmp_path / "extract-child"
    _write_json(child_root / "child" / "capture_descriptor.json", {})
    assert rew._find_extracted_capture_root(child_root) == child_root / "child"
    deep_root = tmp_path / "extract-deep"
    _write_json(deep_root / "a" / "b" / "capture_descriptor.json", {})
    assert rew._find_extracted_capture_root(deep_root) == deep_root / "a" / "b"
    empty_root = tmp_path / "extract-empty"
    empty_root.mkdir()
    assert rew._find_extracted_capture_root(empty_root) is None

    valid_zip = tmp_path / "valid.zip"
    with zipfile.ZipFile(valid_zip, "w") as archive:
        archive.writestr("nested/capture_descriptor.json", "{}")
    manifest = rew._extract_capture_root_bundle(str(valid_zip), tmp_path / "worker-zip")
    assert manifest["status"] == "extracted"
    existing_extract_dir = tmp_path / "worker-zip-existing" / "capture_root_bundle"
    existing_extract_dir.mkdir(parents=True)
    (existing_extract_dir / "stale.txt").write_text("stale", encoding="utf-8")
    manifest = rew._extract_capture_root_bundle(str(valid_zip), tmp_path / "worker-zip-existing")
    assert manifest["status"] == "extracted"
    unsafe_zip = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe_zip, "w") as archive:
        archive.writestr("../capture_descriptor.json", "{}")
    with pytest.raises(ValueError, match="unsafe zip"):
        rew._extract_capture_root_bundle(str(unsafe_zip), tmp_path / "worker-unsafe")
    valid_tar = tmp_path / "valid.tar"
    with tarfile.open(valid_tar, "w") as archive:
        data = b"{}"
        info = tarfile.TarInfo("nested/capture_descriptor.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    assert rew._extract_capture_root_bundle(str(valid_tar), tmp_path / "worker-tar")["status"] == "extracted"
    unsafe_tar = tmp_path / "unsafe.tar"
    with tarfile.open(unsafe_tar, "w") as archive:
        data = b"{}"
        info = tarfile.TarInfo("../capture_descriptor.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="unsafe tar"):
        rew._extract_capture_root_bundle(str(unsafe_tar), tmp_path / "worker-unsafe-tar")
    invalid_archive = tmp_path / "invalid.txt"
    invalid_archive.write_text("not archive", encoding="utf-8")
    with pytest.raises(ValueError, match="must be"):
        rew._extract_capture_root_bundle(str(invalid_archive), tmp_path / "worker-invalid")
    no_descriptor_zip = tmp_path / "no-descriptor.zip"
    with zipfile.ZipFile(no_descriptor_zip, "w") as archive:
        archive.writestr("nested/other.json", "{}")
    with pytest.raises(ValueError, match="did not contain"):
        rew._extract_capture_root_bundle(str(no_descriptor_zip), tmp_path / "worker-no-descriptor")

    non_mapping_manifest = tmp_path / "non-mapping-manifest.json"
    non_mapping_manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected worker manifest object"):
        rew._load_manifest(str(non_mapping_manifest), tmp_path)

    source_dir = tmp_path / "copy-source"
    _write_json(source_dir / "nested" / "file.json", {"x": 1})
    destination = tmp_path / "copy-dest"
    (destination / "nested").mkdir(parents=True)
    (destination / "nested" / "old.txt").write_text("old", encoding="utf-8")
    rew._copy_directory_contents(source_dir, destination)
    assert (destination / "nested" / "file.json").is_file()
    assert rew._upload_directory_to_gs(source_dir, "gs://bucket/prefix") == 1
    assert uploads[-1][0] == "prefix/nested/file.json"

    runtime_manifest = tmp_path / "worker_runtime_manifest.json"
    _write_json(runtime_manifest, {"status": "completed"})
    assert rew._copy_runtime_manifest_to_artifact_output(
        runtime_manifest_path=runtime_manifest, artifact_output_uri=str(tmp_path / "local-output")
    )["status"] == "completed"
    assert rew._copy_runtime_manifest_to_artifact_output(
        runtime_manifest_path=runtime_manifest, artifact_output_uri="gs://bucket/out"
    )["object_key"] == "out/worker_runtime_manifest.json"
    assert rew._copy_runtime_manifest_to_artifact_output(
        runtime_manifest_path=runtime_manifest, artifact_output_uri="r2://bucket/out"
    )["storage_scheme"] == "r2"
    assert rew._copy_runtime_manifest_to_artifact_output(
        runtime_manifest_path=runtime_manifest, artifact_output_uri="ftp://host/out"
    )["status"] == "blocked"

    worker_dir = tmp_path / "worker-files"
    _write_json(worker_dir / "worker_runtime_manifest.json", {"status": "completed"})
    _write_json(worker_dir / "worker_runtime_preflight.json", {"status": "passed"})
    assert rew._copy_worker_runtime_files_to_artifact_output(
        work_dir=worker_dir,
        artifact_output_uri=str(tmp_path / "artifact-local"),
        relative_paths=["worker_runtime_manifest.json", "missing.json"],
    )["copied_file_count"] == 1
    assert rew._copy_worker_runtime_files_to_artifact_output(
        work_dir=worker_dir,
        artifact_output_uri="gs://bucket/artifacts",
        relative_paths=["worker_runtime_manifest.json"],
    )["uploaded_file_count"] == 1
    assert rew._copy_worker_runtime_files_to_artifact_output(
        work_dir=worker_dir,
        artifact_output_uri="s3://bucket/artifacts",
        relative_paths=["worker_runtime_manifest.json"],
    )["storage_scheme"] == "s3"
    assert rew._copy_worker_runtime_files_to_artifact_output(
        work_dir=worker_dir,
        artifact_output_uri="ftp://host/artifacts",
        relative_paths=["worker_runtime_manifest.json"],
    )["status"] == "blocked"

    assert rew._copy_artifacts(job_dir=source_dir, artifact_output_uri=str(tmp_path / "copy-output"))[
        "status"
    ] == "completed"
    assert rew._copy_artifacts(job_dir=source_dir, artifact_output_uri="gs://bucket/copy")[
        "uploaded_file_count"
    ] == 1
    assert rew._copy_artifacts(job_dir=source_dir, artifact_output_uri="r2://bucket/copy")[
        "storage_scheme"
    ] == "r2"
    assert rew._copy_artifacts(job_dir=source_dir, artifact_output_uri="ftp://host/copy")[
        "status"
    ] == "blocked"


def test_robot_eval_worker_signed_put_provider_and_refresh_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_manifest_path = tmp_path / "worker_runtime_manifest.json"
    _write_json(runtime_manifest_path, {"status": "completed"})
    assert rew._upload_runtime_manifest_to_signed_put_url(runtime_manifest_path)["status"] == "not_configured"

    class FakePutResponse:
        status = 503

        def __enter__(self) -> "FakePutResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setenv("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", "https://trusted.invalid/put")
    monkeypatch.setattr(rew.urllib.request, "urlopen", lambda *args, **kwargs: FakePutResponse())
    assert rew._upload_runtime_manifest_to_signed_put_url(runtime_manifest_path)["status"] == "blocked"

    calls = {"count": 0}

    def fake_signed_put(path: Path) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": "completed", "http_status_code": 200}
        raise RuntimeError("final failed")

    monkeypatch.setattr(rew, "_upload_runtime_manifest_to_signed_put_url", fake_signed_put)
    manifest = rew._write_runtime_manifest_with_signed_put(runtime_manifest_path, {"status": "completed"})
    assert manifest["signed_put_runtime_manifest_upload"]["final_manifest_reupload_status"] == "blocked"

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    assert rew._record_provider_runtime_gpu_time(
        job_dir=job_dir, started_monotonic=0.0, generated_at="now"
    )["status"] == "not_provider_runtime"
    monkeypatch.setenv("BLUEPRINT_ROBOT_EVAL_PROVIDER_RUNTIME", "1")
    assert rew._record_provider_runtime_gpu_time(
        job_dir=job_dir, started_monotonic=0.0, generated_at="now"
    )["status"] == "blocked"
    (job_dir / "gpu_cost_control_ledger.json").write_text("[]", encoding="utf-8")
    assert rew._record_provider_runtime_gpu_time(
        job_dir=job_dir, started_monotonic=0.0, generated_at="now"
    )["blockers"] == ["invalid_gpu_cost_control_ledger"]
    _write_json(job_dir / "gpu_cost_control_ledger.json", {"gpu_time": {}})
    assert rew._record_provider_runtime_gpu_time(
        job_dir=job_dir, started_monotonic=0.0, generated_at="now"
    )["status"] == "recorded"

    assert rew._provider_shutdown_evidence_from_runtime(job_dir)["status"] == "missing"
    _write_json(job_dir / "provider_shutdown_proof.json", {"provider_shutdown_proven": True})
    assert rew._provider_shutdown_evidence_from_runtime(job_dir)["status"] == "provided"
    configured_proof = tmp_path / "configured_shutdown.json"
    _write_json(configured_proof, {"clean_shutdown_proven": True})
    monkeypatch.setenv("BLUEPRINT_PROVIDER_SHUTDOWN_PROOF", str(configured_proof))
    assert rew._provider_shutdown_evidence_from_runtime(job_dir)["clean_shutdown_proven"] is True

    (job_dir / "gpu_cost_control_ledger.json").write_text("[]", encoding="utf-8")
    assert rew._record_provider_runtime_finalizer_proof(
        job_dir=job_dir,
        artifact_output_uri="gs://bucket/out",
        artifact_upload={"status": "blocked"},
        finalizer_refresh_upload={"status": "blocked"},
        runtime_manifest_upload={"status": "blocked"},
        generated_at="now",
    )["blockers"] == ["invalid_gpu_cost_control_ledger"]
    _write_json(job_dir / "gpu_cost_control_ledger.json", {"artifact_finalizer": {}})
    proof = rew._record_provider_runtime_finalizer_proof(
        job_dir=job_dir,
        artifact_output_uri="gs://bucket/out",
        artifact_upload={"status": "blocked", "blockers": ["upload"]},
        finalizer_refresh_upload={"status": "completed"},
        runtime_manifest_upload={"status": "completed"},
        generated_at="now",
    )
    assert "artifact_upload_not_completed_before_shutdown" in proof["blockers"]
    proof = rew._record_provider_runtime_finalizer_proof(
        job_dir=job_dir,
        artifact_output_uri="gs://bucket/out",
        artifact_upload={"status": "completed", "uploaded_file_count": 1},
        finalizer_refresh_upload={"status": "blocked", "blockers": ["refresh"]},
        runtime_manifest_upload={"status": "blocked", "blockers": ["runtime"]},
        generated_at="now",
    )
    assert "finalizer_refresh_upload_not_completed_before_shutdown" in proof["blockers"]
    assert proof["provider_shutdown_proven"] is True

    monkeypatch.setattr(
        rew,
        "_remote_cloud_execution_closure_manifest",
        lambda **kwargs: {"status": "closed", "remote_cloud_execution_proven": True, "clean_shutdown_proven": True},
    )
    assert rew._refresh_remote_cloud_closure_after_worker_runtime(
        job_dir=tmp_path / "empty-job", generated_at="now"
    )["status"] == "not_available"
    _write_json(job_dir / "job_run_manifest.json", {"job_id": "job-1", "artifacts": {}})
    closure = rew._refresh_remote_cloud_closure_after_worker_runtime(job_dir=job_dir, generated_at="now")
    assert closure["status"] == "closed"

    startup_mod = types.ModuleType("blueprint_pipeline.robot_eval_startup_architecture_audit")
    startup_mod.build_robot_eval_startup_architecture_audit = lambda **kwargs: {
        "status": "passed",
        "architecture_compliant": True,
        "blockers": [],
    }
    monkeypatch.setitem(
        sys.modules,
        "blueprint_pipeline.robot_eval_startup_architecture_audit",
        startup_mod,
    )
    assert rew._refresh_job_startup_audit_with_worker_runtime(tmp_path / "no-run-manifest")[
        "status"
    ] == "passed"
    invalid_run_manifest_dir = tmp_path / "invalid-run-manifest"
    invalid_run_manifest_dir.mkdir()
    (invalid_run_manifest_dir / "job_run_manifest.json").write_text("[]", encoding="utf-8")
    assert rew._refresh_job_startup_audit_with_worker_runtime(invalid_run_manifest_dir)[
        "status"
    ] == "passed"
    assert rew._refresh_job_startup_audit_with_worker_runtime(job_dir)["status"] == "passed"

    no_upload = rew._attach_worker_failure_artifact_upload(
        runtime_manifest={"status": "blocked", "blockers": ["b"]},
        work_dir=tmp_path / "failure-worker",
        artifact_output_uri=None,
    )
    assert no_upload["status"] == "blocked"

    def failing_copy(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("copy failed")

    monkeypatch.setattr(rew, "_copy_worker_runtime_files_to_artifact_output", failing_copy)
    failed_upload = rew._attach_worker_failure_artifact_upload(
        runtime_manifest={"status": "blocked", "blockers": []},
        work_dir=tmp_path / "failure-worker-2",
        artifact_output_uri="gs://bucket/out",
    )
    assert "artifact_upload_failed:RuntimeError" in failed_upload["blockers"]

    assert rew._parse_simulator_commands(["mujoco=python run.py"]) == {"mujoco": "python run.py"}
    with pytest.raises(ValueError):
        rew._parse_simulator_commands(["fixture=bad"])


def test_robot_eval_worker_main_and_selected_runtime_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_runtime = rew.run_robot_eval_worker(
        manifest_uri=str(tmp_path / "missing-worker-manifest.json"),
        work_dir=tmp_path / "worker-load-failed",
    )
    assert missing_runtime["blockers"] == ["worker_manifest_load_failed:FileNotFoundError"]

    manifest_path = tmp_path / "worker_manifest.json"
    _write_json(manifest_path, {"job_id": "job-1"})
    runtime = rew.run_robot_eval_worker(manifest_uri=str(manifest_path), work_dir=tmp_path / "worker")
    assert runtime["blockers"] == ["missing_capture_root"]

    _write_json(manifest_path, {"capture_root": str(tmp_path / "capture")})
    runtime = rew.run_robot_eval_worker(manifest_uri=str(manifest_path), work_dir=tmp_path / "worker-2")
    assert runtime["blockers"] == ["missing_job_id"]

    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    _write_json(
        manifest_path,
        {
            "schema_version": rew.WORKER_INPUT_MANIFEST_SCHEMA_VERSION,
            "capture_root": str(capture_root),
            "job_id": "job-1",
            "provisioner": "runpod",
            "simulator": "mujoco",
            "job_request": {"job_id": "job-1", "capture_root": str(capture_root)},
            "artifact_output_uri": "gs://bucket/out",
            "artifact_output_uri_provider_writable": True,
            "artifact_output_write_auth_contract_ready": True,
            "runtime_preflight_contract": {
                "required_before_scene_load": True,
                "worker_blocks_scene_load_on_failed_preflight": True,
                "run_before": "scene_load_and_policy_execution",
                "result_artifact": "worker_runtime_preflight.json",
                "required_checks": ["basic"],
                "runtime_preflight_is_not_simulator_proof": True,
            },
        },
    )
    monkeypatch.setattr(
        rew,
        "_copy_worker_runtime_files_to_artifact_output",
        lambda **kwargs: {"status": "completed", "uploaded_file_count": 1},
    )
    runtime = rew.run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker-3",
        capture_root=capture_root,
        allow_simulator_execution=True,
    )
    assert runtime["blockers"] == ["worker_runtime_preflight_blocked"]

    fake_result_job = tmp_path / "job-result"
    fake_result_job.mkdir()
    _write_json(fake_result_job / "job_run_manifest.json", {"status": "completed", "artifacts": {}})

    def fake_build_robot_eval_job(**kwargs: object) -> dict[str, object]:
        return {
            "status": "completed",
            "job_dir": str(fake_result_job),
            "manifest_path": str(fake_result_job / "job_run_manifest.json"),
        }

    monkeypatch.setattr(
        rew, "execute_legacy_robot_eval_request_as_evaluation_run", fake_build_robot_eval_job
    )
    startup_mod = types.ModuleType("blueprint_pipeline.robot_eval_startup_architecture_audit")
    startup_mod.build_robot_eval_startup_architecture_audit = lambda **kwargs: {
        "status": "passed",
        "architecture_compliant": True,
        "blockers": [],
    }
    monkeypatch.setitem(
        sys.modules,
        "blueprint_pipeline.robot_eval_startup_architecture_audit",
        startup_mod,
    )
    _write_json(
        manifest_path,
        {
            "schema_version": rew.WORKER_INPUT_MANIFEST_SCHEMA_VERSION,
            "capture_root": str(capture_root),
            "job_id": "job-1",
            "provisioner": "fixture_local",
            "simulator": "fixture",
            "job_request": {"job_id": "job-1", "capture_root": str(capture_root)},
        },
    )
    runtime = rew.run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker-4",
    )
    assert runtime["status"] == "completed"

    def fake_run_robot_eval_worker(**kwargs: object) -> dict[str, object]:
        assert kwargs["wam_provider_commands"] == {"cosmos3_wam": "run"}
        assert kwargs["simulator_commands"] == {"mujoco": "simulate"}
        return {"status": "completed"}

    monkeypatch.setattr(rew, "run_robot_eval_worker", fake_run_robot_eval_worker)
    assert rew.main(
        [
            "--manifest",
            str(manifest_path),
            "--wam-provider-command",
            "cosmos3_wam=run",
            "--simulator-command",
            "mujoco=simulate",
            "--allow-missing-artifact-output-uri",
        ]
    ) == 0
    with pytest.raises(SystemExit):
        rew.main([])


def test_robot_eval_worker_failure_fallback_and_finalizer_upload_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    manifest_path = tmp_path / "worker_manifest.json"

    _write_json(
        manifest_path,
        {
            "capture_root_bundle_uri": str(tmp_path / "not-an-archive.zip"),
            "job_id": "job-1",
        },
    )
    (tmp_path / "not-an-archive.zip").write_text("not an archive", encoding="utf-8")
    runtime = rew.run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker-bundle-failed",
    )
    assert runtime["blockers"] == ["capture_root_bundle_extract_failed:ValueError"]

    _write_json(
        manifest_path,
        {
            "capture_root": str(capture_root),
            "job_id": "job-1",
            "simulator": "fixture",
            "job_request": {"job_id": "job-1", "capture_root": str(capture_root)},
        },
    )

    def raising_build(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("orchestrator failed")

    monkeypatch.setattr(
        rew, "execute_legacy_robot_eval_request_as_evaluation_run", raising_build
    )
    runtime = rew.run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker-orchestrator-failed",
    )
    assert runtime["blockers"] == ["worker_orchestrator_failed:RuntimeError"]

    job_dir = tmp_path / "job-blocked"
    job_dir.mkdir()
    _write_json(job_dir / "job_run_manifest.json", {"status": "blocked", "blockers": ["job"]})

    def blocked_build(**kwargs: object) -> dict[str, object]:
        return {
            "status": "blocked",
            "blockers": ["job"],
            "job_dir": str(job_dir),
            "manifest_path": str(job_dir / "job_run_manifest.json"),
        }

    startup_mod = types.ModuleType("blueprint_pipeline.robot_eval_startup_architecture_audit")
    startup_mod.build_robot_eval_startup_architecture_audit = lambda **kwargs: {
        "status": "passed",
        "architecture_compliant": True,
        "blockers": [],
    }
    monkeypatch.setitem(
        sys.modules,
        "blueprint_pipeline.robot_eval_startup_architecture_audit",
        startup_mod,
    )
    monkeypatch.setattr(
        rew, "execute_legacy_robot_eval_request_as_evaluation_run", blocked_build
    )
    _write_json(
        manifest_path,
        {
            "capture_root": str(capture_root),
            "job_id": "job-1",
            "simulator": "mujoco",
            "simulator_commands": {"mujoco": "simulate"},
            "job_request": {"job_id": "job-1", "capture_root": str(capture_root)},
        },
    )
    runtime = rew.run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker-fallback-blocked",
    )
    assert runtime["provider_runtime_simulator_command_result"]["blockers"] == [
        "missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION",
        "missing_cli_allow_simulator_execution",
        "missing_cli_allow_simulator_mujoco",
        "missing_provider_runtime_scenario_eval_matrix",
    ]

    job_upload_dir = tmp_path / "job-upload"
    job_upload_dir.mkdir()
    _write_json(job_upload_dir / "job_run_manifest.json", {"status": "completed", "artifacts": {}})

    def completed_build(**kwargs: object) -> dict[str, object]:
        return {
            "status": "completed",
            "job_dir": str(job_upload_dir),
            "manifest_path": str(job_upload_dir / "job_run_manifest.json"),
        }

    monkeypatch.setattr(
        rew, "execute_legacy_robot_eval_request_as_evaluation_run", completed_build
    )
    monkeypatch.setattr(
        rew,
        "_copy_artifacts",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("upload failed")),
    )
    _write_json(
        manifest_path,
        {
            "capture_root": str(capture_root),
            "job_id": "job-1",
            "simulator": "fixture",
            "artifact_output_uri": str(tmp_path / "artifact-output"),
            "job_request": {"job_id": "job-1", "capture_root": str(capture_root)},
        },
    )
    runtime = rew.run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker-artifact-upload-failed",
    )
    assert runtime["status"] == "blocked"
    assert runtime["blockers"] == ["artifact_upload_failed:RuntimeError"]

    monkeypatch.setattr(rew, "_copy_artifacts", lambda **kwargs: {"status": "completed"})
    monkeypatch.setattr(
        rew,
        "_copy_worker_runtime_files_to_artifact_output",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("finalizer copy failed")),
    )
    runtime = rew.run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker-finalizer-upload-failed",
    )
    assert runtime["status"] == "blocked"
    assert runtime["blockers"] == ["worker_runtime_manifest_upload_failed:RuntimeError"]


def test_provider_runtime_allows_only_runpod_serverless_volume_local_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BLUEPRINT_ROBOT_EVAL_PROVIDER_RUNTIME", "true")
    monkeypatch.setenv(
        "BLUEPRINT_RUNPOD_SERVERLESS_NETWORK_VOLUME_RUNTIME", "true"
    )
    allowed = "/runpod-volume/jobs/smoke/manifest.json"

    assert rew._uri_to_local_path(allowed, tmp_path) == Path(allowed)
    with pytest.raises(ValueError, match="local worker manifest sources are disabled"):
        rew._uri_to_local_path("/tmp/manifest.json", tmp_path)
    with pytest.raises(ValueError, match="file:// worker manifest sources are disabled"):
        rew._uri_to_local_path("file:///runpod-volume/jobs/manifest.json", tmp_path)
