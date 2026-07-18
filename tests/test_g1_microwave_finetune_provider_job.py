from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.error
import zipfile

from blueprint_pipeline import g1_microwave_finetune_provider_bundle as bundle_module
from blueprint_pipeline import g1_microwave_finetune_provider_job as job


class _FakeProvider:
    name = "runpod"

    def __init__(self) -> None:
        self.request = None

    def build_request(self, spec, job_dir):
        assert spec.requires_rtx is False
        assert spec.min_gpu_ram_mb == 40_000
        assert spec.gpu_types == ("NVIDIA A40",)
        assert spec.env[bundle_module.BUNDLE_URL_ENV].startswith("https://")
        assert spec.env[bundle_module.OUTPUT_PUT_URL_ENV].startswith("https://")
        assert spec.env[bundle_module.CHECKPOINT_PUT_URL_ENV].startswith("https://")
        return {"max_hourly_rate_usd": spec.max_hourly_rate_usd}

    def billable_inventory(self, *, name_prefix):
        return {
            "status": "observed",
            "api_confirmed": True,
            "live_resource_count": 0,
            "resources": [],
        }

    def capacity_preflight(self, request):
        self.request = request
        return {
            "status": "available",
            "viable_gpu_types": [
                {"display_name": "A40", "on_demand_price_usd_per_hour": 0.44}
            ],
        }


def _dataset_archive(tmp_path: Path) -> Path:
    root = tmp_path / "microwave_owned_lerobot_v21_20260717"
    root.mkdir()
    (root / "groot_n17_finetune_preflight.json").write_text(
        json.dumps(
            {
                "status": "qualified_exact_groot_n1_7_training_data_preflight",
                "bounded_finetune_plan": {
                    "warm_starts_from_sealed_sonic_checkpoint": True,
                    "max_steps": 500,
                },
            }
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "dataset.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(root, arcname=root.name)
    return archive


def test_provider_job_dry_run_is_admitted_without_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    provider_bundle = tmp_path / "provider_bundle.zip"
    bundle_module.build_provider_bundle(
        dataset_archive=_dataset_archive(tmp_path), output_path=provider_bundle
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in (
        "provider_bundle_url.txt",
        "provider_output_put_url.txt",
        "provider_output_get_url.txt",
    ):
        path = stage / name
        path.write_text(f"https://objects.example/{name}?signed=test", encoding="utf-8")
        path.chmod(0o600)
    (stage / "wam_provider_object_store_staging_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "bundle_path": str(provider_bundle),
                "bundle_size_bytes": provider_bundle.stat().st_size,
                "signed_output_round_trip": {"status": "passed"},
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    checkpoint_stage = tmp_path / "checkpoint-stage"
    checkpoint_stage.mkdir()
    for name in (
        "provider_bundle_url.txt",
        "provider_output_put_url.txt",
        "provider_output_get_url.txt",
    ):
        path = checkpoint_stage / name
        path.write_text(
            f"https://checkpoint-objects.example/{name}?signed=test", encoding="utf-8"
        )
        path.chmod(0o600)
    (checkpoint_stage / "wam_provider_object_store_staging_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "bundle_path": str(provider_bundle),
                "bundle_size_bytes": provider_bundle.stat().st_size,
                "signed_output_round_trip": {"status": "passed"},
                "output_url_object_binding_sha256": "c" * 64,
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "status": "completed",
                "resolved_digest_ref": bundle_module.IMAGE_REF,
            }
        ),
        encoding="utf-8",
    )
    provider = _FakeProvider()
    monkeypatch.setattr(job, "get_render_provider", lambda name: provider)

    result = job.run_finetune_job(
        provider_name="runpod",
        provider_bundle=provider_bundle,
        object_store_stage_dir=stage,
        checkpoint_object_store_stage_dir=checkpoint_stage,
        release_evidence=release,
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "result.json",
        pod_name="",
        execute=False,
    )

    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0
    assert result["admission"]["status"] == "admitted"
    assert result["admission"]["maximum_estimated_spend_usd"] == 2.29
    assert result["bound_request"]["requires_rtx"] is False
    assert result["bound_request"]["minimum_gpu_ram_mb"] == 40_000
    assert result["bound_request"]["pod_name_prefix"].startswith(
        "blueprint-groot-oscar-canary-"
    )
    assert result["bound_request"]["signed_bundle_url_present"] is True
    assert result["bound_request"]["signed_checkpoint_output_urls_present"] is True
    assert result["bound_request"]["checkpoint_output_object_binding_sha256"] == (
        "c" * 64
    )
    assert result["bound_request"]["raw_secret_values_recorded"] is False
    assert len(result["bound_request"]["provider_bootstrap_sha256"]) == 64
    assert provider.request["prelaunch_spend_guard"] == {
        "schema_version": "g1_microwave_finetune_prelaunch_spend_guard.v1",
        "required_before_provider_launch": True,
        "can_launch": True,
        "blockers": [],
        "max_hourly_rate_usd": 1.1,
        "maximum_live_seconds": 7_500,
        "maximum_estimated_spend_usd": 2.29,
    }
    assert result["bound_request"]["prelaunch_spend_guard"] == (
        provider.request["prelaunch_spend_guard"]
    )


def test_output_collector_stops_when_provider_runtime_exits(
    tmp_path: Path, monkeypatch
) -> None:
    class ExitedProvider:
        def inspect(self, instance_id):
            assert instance_id == "pod-1"
            return {
                "status": "observed",
                "desiredStatus": "EXITED",
                "runtime_present": True,
                "error": None,
            }

    def missing_output(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://objects.example/output", 404, "missing", {}, None
        )

    monkeypatch.setattr(job.urllib.request, "urlopen", missing_output)
    monkeypatch.setattr(job, "POLL_SECONDS", 0)

    result = job._collect_output(
        get_url="https://objects.example/output",
        output_dir=tmp_path / "collected",
        max_seconds=60,
        provider=ExitedProvider(),
        instance_id="pod-1",
    )

    assert result["status"] == "blocked"
    assert result["runtime_seen"] is True
    assert result["blockers"] == [
        "g1_microwave_finetune_provider_runtime_terminated_before_output"
    ]


def test_checkpoint_collector_verifies_binding_and_extracts_one_model(
    tmp_path: Path, monkeypatch
) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("checkpoint-500/config.json", "{}")
        archive.writestr("checkpoint-500/model.safetensors", b"weights")
    raw = payload.getvalue()

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(job.urllib.request, "urlopen", lambda *args, **kwargs: Response(raw))
    result = job._collect_checkpoint(
        get_urls=["https://objects.example/checkpoint"],
        output_dir=tmp_path / "collected-checkpoint",
        worker_report={
            "checkpoint_archive": {
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "upload": {"status": "passed"},
            }
        },
    )

    assert result["status"] == "completed"
    assert Path(result["checkpoint_path"]).name == "checkpoint-500"
    assert (Path(result["checkpoint_path"]) / "model.safetensors").read_bytes() == b"weights"
    assert len(result["checkpoint_tree_sha256"]) == 64


def test_checkpoint_collector_selects_numbered_final_from_legacy_root_mirror(
    tmp_path: Path, monkeypatch
) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("config.json", "{}")
        archive.writestr("model.safetensors", b"mirror")
        archive.writestr("checkpoint-500/config.json", "{}")
        archive.writestr("checkpoint-500/model.safetensors", b"qualified")
    raw = payload.getvalue()

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(job.urllib.request, "urlopen", lambda *a, **k: Response(raw))
    result = job._collect_checkpoint(
        get_urls=["https://objects.example/checkpoint"],
        output_dir=tmp_path / "collected-checkpoint",
        worker_report={
            "checkpoint_archive": {
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "upload": {"status": "passed"},
            }
        },
    )

    assert result["status"] == "completed"
    assert Path(result["checkpoint_path"]).name == "checkpoint-500"
    assert (Path(result["checkpoint_path"]) / "model.safetensors").read_bytes() == b"qualified"


def test_checkpoint_collector_reassembles_ordered_hash_bound_parts(
    tmp_path: Path, monkeypatch
) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("config.json", "{}")
        archive.writestr("model.safetensors", b"qualified")
    raw = payload.getvalue()
    chunks = [raw[: len(raw) // 2], raw[len(raw) // 2 :]]

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def urlopen(url: str, **kwargs):
        index = int(url.rsplit("/", 1)[-1]) - 1
        return Response(chunks[index])

    monkeypatch.setattr(job.urllib.request, "urlopen", urlopen)
    result = job._collect_checkpoint(
        get_urls=["https://objects.example/1", "https://objects.example/2"],
        output_dir=tmp_path / "collected-checkpoint",
        worker_report={
            "checkpoint_archive": {
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "upload": {
                    "status": "passed",
                    "transport": "ordered_parts",
                    "parts": [
                        {
                            "part_number": index + 1,
                            "size_bytes": len(chunk),
                            "sha256": hashlib.sha256(chunk).hexdigest(),
                        }
                        for index, chunk in enumerate(chunks)
                    ],
                },
            }
        },
    )

    assert result["status"] == "completed"
    assert (Path(result["checkpoint_path"]) / "model.safetensors").read_bytes() == b"qualified"


def test_large_checkpoint_collector_preserves_verified_object_store_parts(
    tmp_path: Path, monkeypatch
) -> None:
    chunks = [b"abcd", b"ef"]

    class Response(io.BytesIO):
        status = 206

        def __init__(self, payload: bytes):
            super().__init__(payload[:1])
            self.headers = {
                "Content-Range": f"bytes 0-0/{len(payload)}",
                "Content-Length": "1",
            }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def urlopen(request, **_kwargs):
        index = int(request.full_url.rsplit("/", 1)[-1]) - 1
        assert request.headers["Range"] == "bytes=0-0"
        return Response(chunks[index])

    monkeypatch.setattr(job, "MAX_LOCAL_CHECKPOINT_COLLECTION_BYTES", 1)
    monkeypatch.setattr(job.urllib.request, "urlopen", urlopen)
    raw = b"".join(chunks)
    result = job._collect_checkpoint(
        get_urls=["https://objects.example/1", "https://objects.example/2"],
        output_dir=tmp_path / "collected-checkpoint",
        worker_report={
            "checkpoint_archive": {
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "upload": {
                    "status": "passed",
                    "transport": "ordered_parts",
                    "parts": [
                        {
                            "part_number": index + 1,
                            "size_bytes": len(chunk),
                            "sha256": hashlib.sha256(chunk).hexdigest(),
                        }
                        for index, chunk in enumerate(chunks)
                    ],
                },
            }
        },
    )

    assert result["status"] == "completed"
    assert result["collection_mode"] == "object_store_bound_ordered_parts"
    assert result["checkpoint_object_store_bound"] is True
    assert result["checkpoint_host_collected"] is False
    assert result["archive_sha256"] == hashlib.sha256(raw).hexdigest()
    assert not (tmp_path / "g1_microwave_finetune_checkpoint.zip").exists()


def test_resume_checkpoint_transfer_uses_existing_parts_without_runpod(
    tmp_path: Path, monkeypatch
) -> None:
    report = tmp_path / "worker-report.json"
    report.write_text(
        json.dumps(
            {
                "status": "completed",
                "blockers": [],
                "open_loop_qualification": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        job,
        "_bundle_evidence",
        lambda _path: {"bundle": {"sha256": "a" * 64}},
    )
    monkeypatch.setattr(
        job,
        "_staging_evidence",
        lambda _stage, _bundle: {"status": "completed"},
    )
    monkeypatch.setattr(
        job,
        "_read_secret_url",
        lambda path, **_kwargs: f"https://objects.example/{path.parent.name}",
    )
    monkeypatch.setattr(
        job,
        "_load_vast_checkpoint_target",
        lambda _manifest: {
            "instance_id": "vast-123",
            "resource_name": "qualification-receiver",
            "_admission_manifest": {
                "continuing_spend": True,
                "watchdog_deadline_epoch": job.time.time() + 3600,
                "resource_name": "qualification-receiver",
            },
            "_admission_inspection": {
                "status": "observed",
                "instance_id": "vast-123",
                "name": "qualification-receiver",
            },
        },
    )

    def fake_collect_checkpoint(**kwargs):
        captured.update(kwargs)
        return {"status": "completed", "checkpoint_path": "/workspace/checkpoint-500", "blockers": []}

    monkeypatch.setattr(job, "_collect_checkpoint", fake_collect_checkpoint)
    output = tmp_path / "resume.json"
    dry_run = job.resume_checkpoint_transfer_to_vast(
        provider_bundle=tmp_path / "bundle.zip",
        checkpoint_object_store_stage_dirs=[tmp_path / "part-001"],
        worker_report_path=report,
        checkpoint_vast_session_manifest=tmp_path / "session.json",
        admission_out=None,
        adapter_output=output,
        execute=False,
    )
    assert dry_run["status"] == "blocked"
    assert not captured

    result = job.resume_checkpoint_transfer_to_vast(
        provider_bundle=tmp_path / "bundle.zip",
        checkpoint_object_store_stage_dirs=[
            tmp_path / "part-001",
            tmp_path / "part-002",
            tmp_path / "part-003",
        ],
        worker_report_path=report,
        checkpoint_vast_session_manifest=tmp_path / "session.json",
        admission_out=tmp_path / "resume-admission.json",
        adapter_output=output,
        execute=True,
    )

    assert result["status"] == "completed"
    assert result["runpod_allocation_performed"] is False
    assert result["checkpoint_retraining_performed"] is False
    assert result["claim_boundary"]["checkpoint_installed_on_vast"] is True
    admission = json.loads((tmp_path / "resume-admission.json").read_text())
    assert admission["action"] == "install-checkpoint"
    assert admission["component"] == "groot_microwave_finetune"
    assert len(captured["get_urls"]) == 3
    assert captured["vast_target"]["instance_id"] == "vast-123"
    assert json.loads(output.read_text())["status"] == "completed"


def test_main_exposes_checkpoint_resume_without_runpod(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_resume(**kwargs):
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(job, "resume_checkpoint_transfer_to_vast", fake_resume)
    output = tmp_path / "resume.json"
    exit_code = job.main(
        [
            "--provider-bundle",
            str(tmp_path / "bundle.zip"),
            "--checkpoint-object-store-part-stage-dir",
            str(tmp_path / "part-001"),
            "--checkpoint-object-store-part-stage-dir",
            str(tmp_path / "part-002"),
            "--worker-report",
            str(tmp_path / "worker-report.json"),
            "--checkpoint-vast-session-manifest",
            str(tmp_path / "session.json"),
            "--adapter-output",
            str(output),
            "--admission-out",
            str(tmp_path / "resume-admission.json"),
            "--resume-checkpoint-to-vast",
            "--execute",
        ]
    )

    assert exit_code == 0
    assert captured["provider_bundle"] == str(tmp_path / "bundle.zip")
    assert captured["checkpoint_object_store_stage_dirs"] == [
        str(tmp_path / "part-001"),
        str(tmp_path / "part-002"),
    ]
    assert captured["worker_report_path"] == str(tmp_path / "worker-report.json")
    assert captured["checkpoint_vast_session_manifest"] == str(
        tmp_path / "session.json"
    )
    assert captured["adapter_output"] == str(output)
    assert captured["admission_out"] == str(tmp_path / "resume-admission.json")
    assert captured["execute"] is True


def test_remote_checkpoint_receiver_accepts_legacy_numbered_final() -> None:
    source = job._checkpoint_receiver_script()

    assert 'expected_numbered = snapshot / "checkpoint-500"' in source
    assert 'numbered == [expected_numbered]' in source


def test_checkpoint_collector_rejects_report_binding_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    raw = b"not-the-bound-checkpoint"

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(job.urllib.request, "urlopen", lambda *args, **kwargs: Response(raw))
    result = job._collect_checkpoint(
        get_urls=["https://objects.example/checkpoint"],
        output_dir=tmp_path / "collected-checkpoint",
        worker_report={
            "checkpoint_archive": {
                "size_bytes": len(raw),
                "sha256": "0" * 64,
                "upload": {"status": "passed"},
            }
        },
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == [
        "g1_microwave_finetune_checkpoint_collection_failed"
    ]


def test_checkpoint_streams_to_vast_without_local_archive(
    tmp_path: Path, monkeypatch
) -> None:
    raw = b"bound-checkpoint-stream"
    expected_sha = hashlib.sha256(raw).hexdigest()
    remote_receipt = {
        "status": "completed",
        "checkpoint_path": job.REMOTE_FINAL_CHECKPOINT,
        "checkpoint_files": [
            {
                "relative_path": "model.safetensors",
                "size_bytes": 7,
                "sha256": "a" * 64,
            }
        ],
        "checkpoint_tree_sha256": "b" * 64,
        "archive_sha256": expected_sha,
        "archive_size_bytes": len(raw),
    }

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO((json.dumps(remote_receipt) + "\n").encode())
            self.stderr = io.BytesIO()
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(job.urllib.request, "urlopen", lambda *a, **k: Response(raw))
    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    result = job._stream_checkpoint_to_vast(
        download_plan=[
            ("https://objects.example/checkpoint", len(raw), expected_sha)
        ],
        expected_sha=expected_sha,
        expected_size=len(raw),
        target={
            "identity_file": str(tmp_path / "id"),
            "known_hosts_file": str(tmp_path / "known_hosts"),
            "ssh_host": "vast.example",
            "ssh_port": 22,
            "instance_id": "vast-1",
            "resource_name": "retained-vast",
            "checkpoint_path": job.REMOTE_FINAL_CHECKPOINT,
        },
    )

    assert result["status"] == "completed"
    assert result["archive_streamed_direct_to_vast"] is True
    assert result["vast_instance_id"] == "vast-1"
    assert not list(tmp_path.glob("*.zip"))
