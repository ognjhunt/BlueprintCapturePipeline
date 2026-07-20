from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.error
import zipfile

import pytest

from blueprint_pipeline import g1_microwave_finetune_provider_bundle as bundle_module
from blueprint_pipeline import g1_microwave_finetune_provider_job as job


class _FakeProvider:
    name = "runpod"

    def __init__(self) -> None:
        self.request = None
        self.resources: list[dict[str, object]] = []
        self.launch_called = False

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
            "live_resource_count": len(self.resources),
            "resources": self.resources,
        }

    def capacity_preflight(self, request):
        self.request = request
        return {
            "status": "available",
            "viable_gpu_types": [
                {"display_name": "A40", "on_demand_price_usd_per_hour": 0.44}
            ],
        }

    def launch(self, *_args, **_kwargs):
        self.launch_called = True
        raise AssertionError("provider launch reached without current spend lock")


def test_ambiguous_launch_state_is_preserved_for_teardown_settlement(
    tmp_path: Path, monkeypatch
) -> None:
    recorded: dict[str, object] = {}

    def fake_mark(path, *, reason, evidence):
        recorded.update(path=path, reason=reason, evidence=dict(evidence))

    monkeypatch.setattr(job, "mark_pending_teardown_ambiguous", fake_mark)
    launch: dict[str, object] = {}
    pending_path = tmp_path / "pending_teardown.json"

    job._preserve_ambiguous_launch(
        pending_path,
        launch,
        reason="finetune_launch_or_collection_exception:TimeoutError",
    )

    assert launch["allocation_outcome_ambiguous"] is True
    assert recorded == {
        "path": pending_path,
        "reason": "finetune_launch_or_collection_exception:TimeoutError",
        "evidence": {"allocation_outcome_ambiguous": True},
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
    stage_urls = {
        "provider_bundle_url.txt": "https://objects.example/bundle.zip?signed=get",
        "provider_output_put_url.txt": "https://objects.example/proof-output.zip?signed=put",
        "provider_output_get_url.txt": "https://objects.example/proof-output.zip?signed=get",
    }
    for name, url in stage_urls.items():
        path = stage / name
        path.write_text(url, encoding="utf-8")
        path.chmod(0o600)
    stage_binding = job.signed_output_object_binding_sha256(
        stage_urls["provider_output_put_url.txt"],
        stage_urls["provider_output_get_url.txt"],
    )
    (stage / "wam_provider_object_store_staging_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "bundle_path": str(provider_bundle),
                "bundle_size_bytes": provider_bundle.stat().st_size,
                "signed_output_round_trip": {"status": "passed"},
                "output_url_object_binding_sha256": stage_binding,
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    checkpoint_stage = tmp_path / "checkpoint-stage"
    checkpoint_stage.mkdir()
    checkpoint_urls = {
        "provider_bundle_url.txt": "https://checkpoint-objects.example/bundle.zip?signed=get",
        "provider_output_put_url.txt": "https://checkpoint-objects.example/checkpoint.zip?signed=put",
        "provider_output_get_url.txt": "https://checkpoint-objects.example/checkpoint.zip?signed=get",
    }
    for name, url in checkpoint_urls.items():
        path = checkpoint_stage / name
        path.write_text(url, encoding="utf-8")
        path.chmod(0o600)
    checkpoint_binding = job.signed_output_object_binding_sha256(
        checkpoint_urls["provider_output_put_url.txt"],
        checkpoint_urls["provider_output_get_url.txt"],
    )
    (checkpoint_stage / "wam_provider_object_store_staging_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "bundle_path": str(provider_bundle),
                "bundle_size_bytes": provider_bundle.stat().st_size,
                "signed_output_round_trip": {"status": "passed"},
                "output_url_object_binding_sha256": checkpoint_binding,
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
        checkpoint_binding
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
        "inventory_scope": {
            "schema_version": "g1_microwave_finetune_inventory_scope.v1",
            "status": "passed",
            "api_confirmed": True,
            "source_live_resource_count": 0,
            "bound_retained_instance_id": None,
            "bound_retained_instance_present": False,
            "other_live_resource_count": 0,
            "other_live_resources": [],
            "blockers": [],
        },
    }
    assert result["bound_request"]["prelaunch_spend_guard"] == (
        provider.request["prelaunch_spend_guard"]
    )

    execute_without_lock = job.run_finetune_job(
        provider_name="runpod",
        provider_bundle=provider_bundle,
        object_store_stage_dir=stage,
        checkpoint_object_store_stage_dir=checkpoint_stage,
        release_evidence=release,
        admission_out=tmp_path / "execute-admission.json",
        bound_request_out=tmp_path / "execute-bound.json",
        adapter_output=tmp_path / "execute-result.json",
        pod_name="",
        execute=True,
    )
    assert execute_without_lock["status"] == "blocked"
    assert execute_without_lock["provider_mutations_performed"] == 0
    assert provider.launch_called is False
    assert any(
        blocker.startswith("qualification_pre_spend:spend_admission:")
        for blocker in execute_without_lock["blockers"]
    )

    blocked_paths = {
        "admission_out": tmp_path / "blocked-admission.json",
        "bound_request_out": tmp_path / "blocked-bound.json",
        "adapter_output": tmp_path / "blocked-result.json",
    }
    blocked = job.run_finetune_job(
        provider_name="runpod",
        provider_bundle=provider_bundle,
        object_store_stage_dir=stage,
        checkpoint_object_store_stage_dir=checkpoint_stage,
        release_evidence=tmp_path / "missing-release.json",
        pod_name="",
        execute=False,
        **blocked_paths,
    )
    assert blocked["status"] == "blocked"
    assert "finetune_release_evidence_missing_or_invalid" in blocked["blockers"]
    assert all(path.is_file() for path in blocked_paths.values())

    checkpoint_manifest = checkpoint_stage / "wam_provider_object_store_staging_manifest.json"
    colliding = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    for name in ("provider_output_put_url.txt", "provider_output_get_url.txt"):
        path = checkpoint_stage / name
        path.write_text(stage_urls[name], encoding="utf-8")
        path.chmod(0o600)
    colliding["output_url_object_binding_sha256"] = stage_binding
    checkpoint_manifest.write_text(json.dumps(colliding), encoding="utf-8")
    collision_result_path = tmp_path / "collision-result.json"
    collision = job.run_finetune_job(
        provider_name="runpod",
        provider_bundle=provider_bundle,
        object_store_stage_dir=stage,
        checkpoint_object_store_stage_dir=checkpoint_stage,
        release_evidence=release,
        admission_out=tmp_path / "collision-admission.json",
        bound_request_out=tmp_path / "collision-bound.json",
        adapter_output=collision_result_path,
        pod_name="",
        execute=False,
    )
    assert collision["status"] == "blocked"
    assert "finetune_proof_and_checkpoint_output_channels_must_be_distinct" in (
        collision["blockers"]
    )
    assert json.loads(collision_result_path.read_text(encoding="utf-8")) == collision

    replaced_url = stage / "provider_output_get_url.txt"
    replaced_url.write_text(
        "https://objects.example/unrelated-output.zip?signed=get", encoding="utf-8"
    )
    replaced_url.chmod(0o600)
    replaced = job.run_finetune_job(
        provider_name="runpod",
        provider_bundle=provider_bundle,
        object_store_stage_dir=stage,
        checkpoint_object_store_stage_dir=checkpoint_stage,
        release_evidence=release,
        admission_out=tmp_path / "replaced-admission.json",
        bound_request_out=tmp_path / "replaced-bound.json",
        adapter_output=tmp_path / "replaced-result.json",
        pod_name="",
        execute=False,
    )
    assert replaced["status"] == "blocked"
    assert "finetune_object_store_staging_not_qualified" in replaced["blockers"]


def test_vast_finetune_inventory_allows_only_exact_bound_retained_target(
    tmp_path: Path, monkeypatch
) -> None:
    provider = _FakeProvider()
    provider.name = "vast"
    provider.resources = [{"instance_id": "vast-target", "name": "retained"}]
    monkeypatch.setattr(job, "get_render_provider", lambda _name: provider)
    monkeypatch.setattr(
        job,
        "_bundle_evidence",
        lambda _path: {
            "bundle": {"sha256": "b" * 64},
            "dataset": {"sha256": "d" * 64},
            "worker": {"sha256": "w" * 64},
        },
    )
    monkeypatch.setattr(
        job,
        "_staging_evidence",
        lambda path, _bundle, **_kwargs: {
            "output_url_object_binding_sha256": (
                "c" * 64 if "checkpoint" in str(path) else "a" * 64
            )
        },
    )
    monkeypatch.setattr(
        job,
        "_read_secret_url",
        lambda _path, *, name: f"https://objects.example/{name}",
    )
    monkeypatch.setattr(
        job,
        "_load_mapping",
        lambda _path, *, name: {
            "status": "completed",
            "resolved_digest_ref": bundle_module.IMAGE_REF,
        },
    )
    monkeypatch.setattr(
        job,
        "_load_vast_checkpoint_target",
        lambda _path, **_kwargs: {"instance_id": "vast-target"},
    )

    def run(name: str) -> dict[str, object]:
        return job.run_finetune_job(
            provider_name="vast",
            provider_bundle=tmp_path / "provider-bundle.zip",
            object_store_stage_dir=tmp_path / "input-stage",
            checkpoint_object_store_stage_dir=tmp_path / "checkpoint-stage",
            release_evidence=tmp_path / "release.json",
            admission_out=tmp_path / f"{name}-admission.json",
            bound_request_out=tmp_path / f"{name}-bound.json",
            adapter_output=tmp_path / f"{name}-result.json",
            pod_name="",
            execute=False,
            checkpoint_vast_session_manifest=tmp_path / "session.json",
        )

    admitted = run("admitted")
    assert admitted["status"] == "dry_run_ready"
    scope = admitted["bound_request"]["prelaunch_spend_guard"]["inventory_scope"]
    assert scope["bound_retained_instance_present"] is True
    assert scope["other_live_resource_count"] == 0

    provider.resources = []
    missing = run("missing")
    assert missing["status"] == "blocked"
    missing_scope = missing["bound_request"]["prelaunch_spend_guard"][
        "inventory_scope"
    ]
    assert missing_scope["blockers"] == [
        "bound_retained_instance_inventory_binding_invalid"
    ]

    provider.resources = [
        {"instance_id": "vast-target", "name": "retained"},
        {"instance_id": "unrelated", "name": "other"},
    ]
    blocked = run("blocked")
    assert blocked["status"] == "blocked"
    assert "g1_microwave_finetune_prelaunch_inventory_not_zero" in blocked["blockers"]


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


def test_output_collector_treats_running_vast_instance_as_started(
    tmp_path: Path, monkeypatch
) -> None:
    class RunningVastProvider:
        calls = 0

        def inspect(self, instance_id):
            assert instance_id == "vast-1"
            self.calls += 1
            if self.calls == 1:
                return {"status": "observed", "actual_status": "running", "cur_state": "running"}
            return {"status": "observed", "desiredStatus": "EXITED"}

    def missing_output(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://objects.example/output", 404, "missing", {}, None
        )

    provider = RunningVastProvider()
    monkeypatch.setattr(job.urllib.request, "urlopen", missing_output)
    monkeypatch.setattr(job, "POLL_SECONDS", 0)
    monkeypatch.setattr(job, "STARTUP_TIMEOUT_SECONDS", 0)

    result = job._collect_output(
        get_url="https://objects.example/output",
        output_dir=tmp_path / "collected",
        max_seconds=60,
        provider=provider,
        instance_id="vast-1",
    )

    assert provider.calls == 2
    assert result["runtime_seen"] is True
    assert result["blockers"] == [
        "g1_microwave_finetune_provider_runtime_terminated_before_output"
    ]


def test_output_collector_rejects_oversized_download(tmp_path: Path, monkeypatch) -> None:
    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(job, "MAX_OUTPUT_ARCHIVE_BYTES", 3)
    monkeypatch.setattr(
        job.urllib.request, "urlopen", lambda *args, **kwargs: Response(b"oversized")
    )

    result = job._collect_output(
        get_url="https://objects.example/output",
        output_dir=tmp_path / "collected",
        max_seconds=60,
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == [
        "g1_microwave_finetune_output_not_collected_before_deadline"
    ]
    assert not (tmp_path / "g1_microwave_finetune_output.zip.tmp").exists()


def test_output_extractor_enforces_member_and_expansion_limits(
    tmp_path: Path, monkeypatch
) -> None:
    archive_path = tmp_path / "output.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("g1_microwave_finetune_worker_report.json", "{}")
        archive.writestr("payload.bin", b"data")

    monkeypatch.setattr(job, "MAX_OUTPUT_ARCHIVE_MEMBERS", 1)
    with pytest.raises(ValueError, match="finetune_output_archive_member_count_invalid"):
        job._safe_extract_output(archive_path, tmp_path / "member-limited")

    monkeypatch.setattr(job, "MAX_OUTPUT_ARCHIVE_MEMBERS", 10)
    monkeypatch.setattr(job, "MAX_OUTPUT_UNCOMPRESSED_BYTES", 3)
    with pytest.raises(
        ValueError, match="finetune_output_archive_uncompressed_size_invalid"
    ):
        job._safe_extract_output(archive_path, tmp_path / "size-limited")


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


def test_vast_checkpoint_target_honors_qualification_identity_file(
    tmp_path: Path, monkeypatch
) -> None:
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("vast.example ssh-ed25519 AAAA\n", encoding="utf-8")
    identity = tmp_path / "operator-key"
    identity.write_text("test-key\n", encoding="utf-8")
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "provider": "vast",
                "continuing_spend": True,
                "instance_id": "123",
                "resource_name": "qualification-receiver",
                "ssh_connection": {
                    "ssh_host": "vast.example",
                    "ssh_port": 22022,
                },
                "ssh_host_key": {"known_hosts_file": str(known_hosts)},
            }
        ),
        encoding="utf-8",
    )

    class Provider:
        def inspect(self, instance_id):
            assert instance_id == "123"
            return {
                "status": "observed",
                "actual_status": "running",
                "instance_id": "123",
                "name": "qualification-receiver",
                "ssh_host": "vast.example",
                "ssh_port": 22022,
            }

    monkeypatch.setattr(job, "get_render_provider", lambda _name: Provider())
    target = job._load_vast_checkpoint_target(
        session,
        identity_file=identity,
    )

    assert target["identity_file"] == str(identity.resolve())


def test_vast_checkpoint_collection_admits_before_streaming(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []

    def fake_admit(*_args, **_kwargs):
        events.append("admit")
        return object()

    def fake_collect(**_kwargs):
        assert events == ["admit"]
        events.append("collect")
        return {"status": "completed", "blockers": []}

    monkeypatch.setattr(job, "admit_qualification_control_mutation", fake_admit)
    monkeypatch.setattr(job, "_collect_checkpoint", fake_collect)
    admission = tmp_path / "checkpoint-install-admission.json"
    result = job._collect_checkpoint_with_vast_admission(
        get_urls=["https://objects.example/checkpoint"],
        output_dir=tmp_path / "checkpoint",
        worker_report={},
        vast_target={
            "instance_id": "vast-123",
            "_admission_manifest": {"continuing_spend": True},
            "_admission_inspection": {"status": "observed"},
        },
        admission_out=admission,
    )

    assert events == ["admit", "collect"]
    assert result["qualification_control_admission_passed"] is True
    assert result["qualification_control_admission_path"] == str(admission.resolve())


def test_vast_checkpoint_collection_refuses_streaming_when_admission_fails(
    tmp_path: Path, monkeypatch
) -> None:
    collected = False

    def fake_admit(*_args, **_kwargs):
        raise ValueError("qualification_control_session_ttl_expired")

    def fake_collect(**_kwargs):
        nonlocal collected
        collected = True
        return {"status": "completed", "blockers": []}

    monkeypatch.setattr(job, "admit_qualification_control_mutation", fake_admit)
    monkeypatch.setattr(job, "_collect_checkpoint", fake_collect)
    result = job._collect_checkpoint_with_vast_admission(
        get_urls=["https://objects.example/checkpoint"],
        output_dir=tmp_path / "checkpoint",
        worker_report={},
        vast_target={
            "instance_id": "vast-123",
            "_admission_manifest": {"continuing_spend": True},
            "_admission_inspection": {"status": "observed"},
        },
        admission_out=tmp_path / "checkpoint-install-admission.json",
    )

    assert collected is False
    assert result["status"] == "blocked"
    assert result["qualification_control_admission_passed"] is False
    assert result["blockers"] == ["qualification_control_session_ttl_expired"]


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
        lambda _stage, _bundle, **_kwargs: {"status": "completed"},
    )
    monkeypatch.setattr(
        job,
        "_read_secret_url",
        lambda path, **_kwargs: f"https://objects.example/{path.parent.name}",
    )
    monkeypatch.setattr(
        job,
        "_load_vast_checkpoint_target",
        lambda _manifest, **_kwargs: {
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
            "--qualification-identity-file",
            str(tmp_path / "operator-key"),
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
    assert captured["qualification_identity_file"] == str(tmp_path / "operator-key")
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
