from __future__ import annotations

import argparse
import json
import hashlib
import os
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.retained_gpu_session_lifecycle import record_retained_gpu_state
from blueprint_pipeline import task_evaluation_scene_configuration_provider_cleanup as provider_cleanup
from blueprint_pipeline import task_evaluation_scene_configuration_allocator as allocator
from blueprint_pipeline import task_evaluation_scene_configuration_warm_diagnostic as warm
from blueprint_pipeline import task_evaluation_scene_configuration_warm_overlay as warm_overlay
from blueprint_pipeline import task_evaluation_scene_configuration_warm_remote_protocol as warm_remote
from blueprint_pipeline import task_evaluation_scene_configuration_warm_transport as warm_transport
from blueprint_pipeline import task_evaluation_scene_configuration_warm_execution_contract as warm_contract
from blueprint_pipeline import vast_provider_adapter as vpa
from blueprint_pipeline import vast_scene_warm_secret_probe as secret_probe
from blueprint_pipeline.wam_provider_object_store import (
    signed_output_object_binding_sha256,
)
from scripts import task_evaluation_scene_configuration_diagnostic_provider_runner as provider_runner


def _release_tree(root: Path) -> Path:
    (root / "src/blueprint_pipeline").mkdir(parents=True)
    (root / "src/blueprint_pipeline/__init__.py").write_text("VALUE = 1\n")
    (root / "scripts").mkdir()
    (root / "scripts/task_evaluation_scene_configuration_diagnostic_provider_runner.py").write_text(
        "print('runner')\n"
    )
    shell = root / "scripts/run_task_evaluation_scene_configuration_provider.sh"
    shell.write_text("#!/bin/sh\nexit 0\n")
    os.chmod(shell, 0o755)
    return root


def _checkpoint(root: Path, *, carried: int = 3) -> dict:
    root.mkdir()
    return {
        "checkpoint_digest": "sha256:" + "1" * 64,
        "scientific_bindings": {"binding_digest": "sha256:" + "2" * 64},
        "completed_stage_prefix_count": carried,
        "completed_stage_results": [
            {"stage_id": f"stage-{index + 1}"} for index in range(carried)
        ],
    }


def _release_receipt(path: Path, release_root: Path, *, digest: str) -> dict:
    return {
        "release_path": str(release_root),
        "remote_ref": "refs/heads/codex/warm-test",
        "remote_ref_tip_commit": "a" * 40,
        "receipt_digest": digest,
    }


def test_scene_warm_retention_requires_runtime_and_direct_access() -> None:
    decision = vpa._retention_decision(
        requested=True,
        watchdog_handoff={
            "status": "armed",
            "independent_process": True,
            "watchdog_armed_before_allocation": True,
            "watchdog_pid": 123,
            "watchdog_deadline_epoch": 2_000.0,
        },
        instance_ids=[456],
        startup_probe={"status": "completed", "startup_probe_proven": True},
        gpu_sanity={"status": "completed", "gpu_sanity_proven": True},
        video_smoke={},
        retention_mode=vpa.SCENE_CONFIGURATION_WARM_RETENTION_MODE,
        warm_worker_evidence={
            "provider_bundle_kind": "task_evaluation_scene_configuration",
            "scene_configuration_bundle_downloaded": True,
            "scene_configuration_bundle_sha256_verified": True,
            "scene_configuration_entrypoint_started": True,
            "scene_configuration_runtime_root_ready": True,
            "scene_configuration_runtime_secrets_scrubbed": True,
            "fresh_ssh_runtime_secret_environment_absent": True,
            "instance_running": True,
            "workload_independent_access_recorded": True,
            "ssh_host": "ssh.example.test",
            "ssh_port": 12345,
        },
        observed_now_epoch=1_000.0,
    )

    assert decision["status"] == "retained_owned"
    assert decision["blockers"] == []

    decision["warm_worker_evidence"]["scene_configuration_runtime_root_ready"] = False
    refused = vpa._retention_decision(
        requested=True,
        watchdog_handoff={
            "status": "armed",
            "independent_process": True,
            "watchdog_armed_before_allocation": True,
            "watchdog_pid": 123,
            "watchdog_deadline_epoch": 2_000.0,
        },
        instance_ids=[456],
        startup_probe={"status": "completed", "startup_probe_proven": True},
        gpu_sanity={"status": "completed", "gpu_sanity_proven": True},
        video_smoke={},
        retention_mode=vpa.SCENE_CONFIGURATION_WARM_RETENTION_MODE,
        warm_worker_evidence=decision["warm_worker_evidence"],
        observed_now_epoch=1_000.0,
    )
    assert refused["status"] == "teardown_required"
    assert "retention_scene_configuration_runtime_root_not_ready" in refused["blockers"]

    decision["warm_worker_evidence"]["scene_configuration_runtime_root_ready"] = True
    decision["warm_worker_evidence"][
        "fresh_ssh_runtime_secret_environment_absent"
    ] = False
    refused_secret_env = vpa._retention_decision(
        requested=True,
        watchdog_handoff={
            "status": "armed",
            "independent_process": True,
            "watchdog_armed_before_allocation": True,
            "watchdog_pid": 123,
            "watchdog_deadline_epoch": 2_000.0,
        },
        instance_ids=[456],
        startup_probe={"status": "completed", "startup_probe_proven": True},
        gpu_sanity={"status": "completed", "gpu_sanity_proven": True},
        video_smoke={},
        retention_mode=vpa.SCENE_CONFIGURATION_WARM_RETENTION_MODE,
        warm_worker_evidence=decision["warm_worker_evidence"],
        observed_now_epoch=1_000.0,
    )
    assert refused_secret_env["status"] == "teardown_required"
    assert (
        "retention_scene_configuration_fresh_ssh_secret_environment_not_absent"
        in refused_secret_env["blockers"]
    )


def test_fresh_ssh_probe_and_child_entrypoint_never_inherit_secret_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        secret_probe,
        "enroll_vast_ssh_host_key",
        lambda *_args, **_kwargs: {
            "status": "enrolled",
            "known_hosts_file": str(tmp_path / "known-hosts"),
        },
    )

    def absent_ssh(**kwargs) -> dict:
        command = str(kwargs["remote_argv"][-1])
        assert "compgen -e" in command
        assert "BLUEPRINT_VAST_RUNTIME_SECRET_B64_*" in command
        return {
            "status": "completed",
            "stdout": "RUNTIME_SECRET_ENVIRONMENT:absent\n",
        }

    monkeypatch.setattr(
        "blueprint_pipeline.native_task_arena_warm_vast._run_pinned_ssh",
        absent_ssh,
    )
    proof = secret_probe.probe_fresh_ssh_secret_environment_absent(
        {"ssh_host": "ssh.example.test", "ssh_port": 1234},
        attempt_dir=tmp_path / "probe",
    )
    assert proof["status"] == "completed"
    assert proof["fresh_ssh_runtime_secret_environment_absent"] is True
    assert proof["name_only_probe"] is True
    assert proof["raw_secret_values_recorded"] is False

    command = warm_remote._warm_no_secret_shell_command(
        "python3 -c 'import os; print(any(k.startswith(\"BLUEPRINT_VAST_RUNTIME_SECRET_B64_\") or k in {\"OPENAI_API_KEY\", \"HF_TOKEN\"} for k in os.environ))'"
    )
    child_env = dict(os.environ)
    fake_secret = "fake-value-must-not-appear"
    child_env.update(
        {
            "BLUEPRINT_VAST_RUNTIME_SECRET_B64_OPENAI_API_KEY_FILE": fake_secret,
            "OPENAI_API_KEY": fake_secret,
            "HF_TOKEN": fake_secret,
        }
    )
    result = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=child_env,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "False"
    assert fake_secret not in result.stdout
    assert fake_secret not in result.stderr

    monkeypatch.setattr(
        secret_probe,
        "enroll_vast_ssh_host_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("spawn detail must stay redacted")
        ),
    )
    failed = secret_probe.probe_fresh_ssh_secret_environment_absent(
        {"ssh_host": "ssh.example.test", "ssh_port": 1234},
        attempt_dir=tmp_path / "failed-probe",
    )
    assert failed["status"] == "blocked"
    assert failed["fresh_ssh_runtime_secret_environment_absent"] is False
    assert "spawn detail must stay redacted" not in json.dumps(failed)


def _warm_staging_fixture(tmp_path: Path) -> tuple[Path, Path, dict, float]:
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    overlay = tmp_path / "overlay.zip"
    overlay.write_bytes(b"immutable-overlay")
    overlay_sha = hashlib.sha256(overlay.read_bytes()).hexdigest()
    bundle_key = f"blueprint/test/bundles/sha256/{overlay_sha}.zip"
    output_key = "blueprint/test/output/run-1.zip"
    deadline = datetime(2030, 1, 1, 0, 58, tzinfo=timezone.utc).timestamp()
    query = "X-Amz-Date=20300101T000003Z&X-Amz-Expires=3600&X-Amz-Signature=fake"
    urls = {
        "provider_bundle_url.txt": f"https://objects.example/{bundle_key}?{query}",
        "provider_output_put_url.txt": f"https://objects.example/{output_key}?{query}&method=put",
        "provider_output_get_url.txt": f"https://objects.example/{output_key}?{query}&method=get",
    }
    statuses = {}
    for name, value in urls.items():
        path = staging_dir / name
        path.write_text(value + "\n")
        path.chmod(0o600)
        stat_result = path.stat()
        statuses[name] = {
            "path": str(path),
            "present": True,
            "mode_is_0600": True,
            "size_bytes": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
        }
    binding_payload = json.dumps(
        {
            "bundle_key": bundle_key,
            "bundle_sha256": overlay_sha,
            "output_key": output_key,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    binding_digest = hashlib.sha256(binding_payload).hexdigest()
    binding = {
        "schema_version": "wam_provider_object_store_binding.v1",
        "bundle_key": bundle_key,
        "bundle_sha256": overlay_sha,
        "output_key": output_key,
        "staging_binding_sha256": binding_digest,
    }
    put_url = urls["provider_output_put_url.txt"]
    get_url = urls["provider_output_get_url.txt"]
    manifest = {
        "status": "completed",
        "bundle_sha256": overlay_sha,
        "bundle_size_bytes": overlay.stat().st_size,
        "bundle_key": bundle_key,
        "output_key": output_key,
        "staging_binding_sha256": binding_digest,
        "output_key_run_unique": True,
        "output_url_object_binding_sha256": signed_output_object_binding_sha256(
            put_url, get_url
        ),
        "presigned_url_expiry": {
            # Staging began at 00:00:00.5, while S3 signed three seconds later
            # using a whole-second X-Amz-Date. The manifest is conservative;
            # semantic bounds, not fabricated timestamp equality, bind them.
            "expires_at": "2030-01-01T01:00:00.500000Z"
        },
        "provider_bundle_url_file": statuses["provider_bundle_url.txt"],
        "provider_output_put_url_file": statuses["provider_output_put_url.txt"],
        "provider_output_get_url_file": statuses["provider_output_get_url.txt"],
    }
    (staging_dir / "wam_provider_object_store_staging_binding.json").write_text(
        json.dumps(binding, sort_keys=True) + "\n"
    )
    (staging_dir / "wam_provider_object_store_staging_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    return staging_dir, overlay, manifest, deadline


def test_warm_staging_reopens_url_object_and_real_expiry_binding(
    tmp_path: Path,
) -> None:
    staging_dir, overlay, manifest, deadline = _warm_staging_fixture(tmp_path)
    urls = warm_transport.validated_warm_staging_urls(
        staging_dir=staging_dir,
        staging=manifest,
        overlay_archive=overlay,
        watchdog_deadline_epoch=deadline,
    )
    assert urls["overlay_url"].startswith("https://objects.example/")
    assert urls["output_put_url"].split("?")[0] == urls["output_get_url"].split("?")[0]

    get_path = staging_dir / "provider_output_get_url.txt"
    get_path.write_text(get_path.read_text().replace("output/run-1", "output/run-2"))
    with pytest.raises(
        warm.SceneConfigurationWarmDiagnosticError,
        match="scene_configuration_warm_staging_url_record_invalid",
    ):
        warm_transport.validated_warm_staging_urls(
            staging_dir=staging_dir,
            staging=manifest,
            overlay_archive=overlay,
            watchdog_deadline_epoch=deadline,
        )


def test_bootstrap_execution_cross_binds_base_scientific_identity() -> None:
    receipt = {
        "source_commit": "a" * 40,
        "run_id": "run-1",
        "toolchain_digest": "sha256:" + "1" * 64,
        "portable_construction_envelope_digest": "sha256:" + "2" * 64,
    }
    authority = {"source_checkpoint_digest": "sha256:" + "3" * 64}
    execution = {
        "diagnostic_source_commit": receipt["source_commit"],
        "diagnostic_run_id": receipt["run_id"],
        "diagnostic_toolchain_digest": receipt["toolchain_digest"],
        "diagnostic_construction_envelope_digest": receipt[
            "portable_construction_envelope_digest"
        ],
        "source_checkpoint_digest": authority["source_checkpoint_digest"],
    }
    assert warm_contract.warm_bootstrap_execution_binding_blockers(
        execution=execution,
        bundle_receipt=receipt,
        session_authority=authority,
        advanced_checkpoint={"checkpoint_digest": "sha256:" + "4" * 64},
    ) == []

    execution["diagnostic_run_id"] = "stale-run"
    blockers = warm_contract.warm_bootstrap_execution_binding_blockers(
        execution=execution,
        bundle_receipt=receipt,
        session_authority=authority,
        advanced_checkpoint={"checkpoint_digest": "sha256:" + "4" * 64},
    )
    assert "scene_configuration_warm_bootstrap_execution_run_id_mismatch" in blockers


def test_warm_output_get_retries_only_transient_and_refuses_redirects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = "https://objects.example/output.zip?signature=fake"
    destination = tmp_path / "output.zip"

    class Response:
        headers = {"Content-Length": "7"}

        def __init__(self, *, final_url: str = url) -> None:
            self.final_url = final_url
            self.read_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def geturl(self) -> str:
            return self.final_url

        def read(self, _size: int) -> bytes:
            self.read_count += 1
            return b"payload" if self.read_count == 1 else b""

    calls = 0

    def transient_then_complete(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(url, 503, "retry", {}, None)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", transient_then_complete)
    monkeypatch.setattr(warm_transport.time, "sleep", lambda _seconds: None)
    assert warm_transport._download_bounded_when_ready(
        url=url,
        destination=destination,
        maximum_bytes=1024,
        deadline_monotonic=warm_transport.time.monotonic() + 5,
    )
    assert calls == 2
    assert destination.read_bytes() == b"payload"

    destination.unlink()
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.HTTPError(url, 403, "expired", {}, None)
        ),
    )
    with pytest.raises(
        warm.SceneConfigurationWarmDiagnosticError,
        match="scene_configuration_warm_output_signed_url_rejected",
    ):
        warm_transport._download_bounded_when_ready(
            url=url,
            destination=destination,
            maximum_bytes=1024,
            deadline_monotonic=warm_transport.time.monotonic() + 5,
        )
    assert not list(tmp_path.glob(".*.partial"))

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(
            final_url="https://redirect.example/output.zip?signature=fake"
        ),
    )
    with pytest.raises(
        warm.SceneConfigurationWarmDiagnosticError,
        match="scene_configuration_warm_output_redirect_refused",
    ):
        warm_transport._download_bounded_when_ready(
            url=url,
            destination=destination,
            maximum_bytes=1024,
            deadline_monotonic=warm_transport.time.monotonic() + 5,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.partial"))


def test_output_readiness_rejects_redirect_and_cold_cleanup_waits_for_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ReadyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def geturl(self) -> str:
            return "https://redirect.example/output.zip"

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *_args, **_kwargs: ReadyResponse()
    )
    assert warm_transport._output_object_ready(
        "https://objects.example/output.zip"
    ) is False

    cleanup_called = False

    def cleanup(_path: Path) -> dict:
        nonlocal cleanup_called
        cleanup_called = True
        return {"status": "completed", "all_objects_absent": True}

    deferred = provider_cleanup.cleanup_scene_staging(
        adapter={
            "continuing_spend_from_this_run": True,
            "retained_owned": False,
        },
        staging_dir=tmp_path,
        cleanup=cleanup,
    )
    assert deferred["status"] == "deferred_until_provider_absent"
    assert deferred["all_objects_absent"] is False
    assert cleanup_called is False

    completed = provider_cleanup.cleanup_scene_staging(
        adapter={
            "continuing_spend_from_this_run": False,
            "retained_owned": False,
        },
        staging_dir=tmp_path,
        cleanup=cleanup,
    )
    assert completed["all_objects_absent"] is True
    assert cleanup_called is True


def test_overlay_is_exact_pushed_sha_inventory_and_revalidates_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_root = _release_tree(tmp_path / "release")
    release_receipt = tmp_path / "release.json"
    release_receipt.write_text("{}\n")
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint = _checkpoint(checkpoint_root)
    calls: list[str] = []

    def validate_release(path: str | Path, *, source_commit: str) -> dict:
        assert Path(path) == release_receipt
        calls.append(source_commit)
        return _release_receipt(
            release_receipt, release_root, digest="sha256:" + "3" * 64
        )

    monkeypatch.setattr(warm_overlay, "_validated_release_receipt", validate_release)
    monkeypatch.setattr(
        warm_overlay,
        "validate_scene_configuration_diagnostic_checkpoint",
        lambda **_kwargs: checkpoint,
    )

    receipt = warm.build_scene_configuration_warm_source_overlay(
        diagnostic_release_receipt_path=release_receipt,
        source_commit="a" * 40,
        checkpoint_root=checkpoint_root,
        output_root=tmp_path / "overlay",
    )
    validated = warm.validate_scene_configuration_warm_source_overlay(
        receipt["receipt_path"],
        expected_source_commit="a" * 40,
        expected_checkpoint_digest=checkpoint["checkpoint_digest"],
    )

    assert validated["source_commit"] == "a" * 40
    assert validated["completed_stage_prefix_count"] == 3
    assert len(calls) == 3  # before build, immediately before seal, reopen
    manifest = json.loads(Path(validated["manifest"]["path"]).read_text())
    assert {
        "provider_runtime/task_evaluation_scene_configuration_provider_runner.py",
        "provider_runtime/run_task_evaluation_scene_configuration_provider.sh",
    }.issubset({row["provider_relative_path"] for row in manifest["inventory"]})


def test_overlay_rejects_release_change_during_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_root = _release_tree(tmp_path / "release")
    release_receipt = tmp_path / "release.json"
    release_receipt.write_text("{}\n")
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint = _checkpoint(checkpoint_root)
    calls = 0

    def validate_release(_path: str | Path, *, source_commit: str) -> dict:
        nonlocal calls
        calls += 1
        return _release_receipt(
            release_receipt,
            release_root,
            digest="sha256:" + ("3" if calls == 1 else "4") * 64,
        )

    monkeypatch.setattr(warm_overlay, "_validated_release_receipt", validate_release)
    monkeypatch.setattr(
        warm_overlay,
        "validate_scene_configuration_diagnostic_checkpoint",
        lambda **_kwargs: checkpoint,
    )

    with pytest.raises(
        warm.SceneConfigurationWarmDiagnosticError,
        match="scene_configuration_warm_overlay_release_changed_during_build",
    ):
        warm.build_scene_configuration_warm_source_overlay(
            diagnostic_release_receipt_path=release_receipt,
            source_commit="a" * 40,
            checkpoint_root=checkpoint_root,
            output_root=tmp_path / "overlay",
        )


def test_remote_iteration_is_fixed_immutable_reflink_overlay() -> None:
    authority = {
        "iteration_id": "i001-" + "a" * 12,
        "source_overlay_archive_sha256": "sha256:" + "1" * 64,
        "source_overlay_manifest_digest": "sha256:" + "2" * 64,
        "source_commit": "a" * 40,
        "source_checkpoint_digest": "sha256:" + "3" * 64,
        "scientific_binding_digest": "sha256:" + "4" * 64,
        "remote_checkpoint_root": (
            warm.BASE_RUNTIME_ROOT + "/input/diagnostic_checkpoint"
        ),
        "watchdog_deadline_epoch": 2_000.0,
        "maximum_output_archive_bytes": 1_000_000_000,
    }

    script = warm._remote_iteration_script(
        authority=authority,
        session={
            "session_digest": "sha256:" + "5" * 64,
            "provider_instance_id": 123,
            "bootstrap_allocation_binding_digest": "sha256:" + "6" * 64,
        },
        overlay_url="https://objects.example/overlay?signature=secret",
        output_put_url="https://objects.example/output?signature=secret",
    )
    subprocess.run(
        ["bash", "-n"], input=script, text=True, check=True, capture_output=True
    )

    assert "cp -a --reflink=auto" in script
    assert "copy_function=os.link" not in script
    assert script.index("mkdir -p /workspace/task_evaluation_scene_configuration_warm/iterations") < script.index("exec 9>")
    assert 'if [ -e "$ITERATION_ROOT" ]' in script
    assert 'eval "$' not in script
    assert 'bash -c "$' not in script
    assert "destination.unlink()" in script
    assert "blueprint_runtime_secret_exports.sh" not in script
    assert "--retry-all-errors" not in script
    assert "--connect-timeout 15" in script
    assert "--max-time \"$OVERLAY_TIMEOUT\"" in script
    assert "BLUEPRINT_SCENE_WARM_BLOCKED:overlay_download_failed" in script
    assert "BLUEPRINT_SCENE_WARM_BLOCKED:output_upload_failed" in script
    assert "blueprint_upload_put" in script
    assert "output_expansion_invalid" in script
    assert "iteration_gc_unproven" in script
    assert "iteration_disk_capacity_insufficient" in script
    assert 'shutil.rmtree(package_root)' in script
    assert "overlay_final_inventory_mismatch" in script
    assert "BLUEPRINT_SCENE_WARM_REMOTE_SETUP_STARTED_EPOCH_NS" in script
    for forbidden_env in (
        "OPENAI_API_KEY",
        "OPENAI_API_KEY_FILE",
        "OPENAI_ADMIN_API_KEY_FILE",
        "BLUEPRINT_OPENAI_ADMIN_KEY",
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE",
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
        "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
        "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE",
        "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
        "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
        "BLUEPRINT_OPENAI_COST_SCOPE_ATTESTATION_FILE",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        assert f"-u {forbidden_env}" in script


def test_scene_base_bundle_is_hashed_before_unzip_and_warm_readiness() -> None:
    expected = "sha256:" + "a" * 64
    script = vpa._probe_shell_script(
        "https://example.test/heartbeat",
        enable_blueprint_bundle=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
        expected_provider_bundle_sha256=expected,
    )

    digest_check = script.index("scene_configuration_bundle_digest_mismatch")
    verified = script.index(
        "BLUEPRINT_VAST_SCENE_CONFIGURATION_BUNDLE_SHA256_VERIFIED"
    )
    unzip = script.index("-m zipfile -e")
    assert expected in script
    assert digest_check < verified < unzip


def test_blocked_stage_retains_only_a_prefix_with_all_paid_stages(tmp_path: Path) -> None:
    source_root = tmp_path / "source-checkpoint"
    source_root.mkdir()
    manifest_name = (
        "task_evaluation_scene_configuration_diagnostic_checkpoint.v1.json"
    )
    (source_root / manifest_name).write_text("{}\n")
    prefix_three = {
        "checkpoint_digest": "sha256:" + "1" * 64,
        "completed_stage_prefix_count": 3,
        "inventory": [],
    }

    carried = provider_runner._retained_checkpoint_after_failure(
        output=tmp_path / "output-carried",
        checkpoint_root=source_root,
        checkpoint=prefix_three,
        advanced=prefix_three,
        advanced_root=None,
    )
    assert carried is not None
    assert carried["completed_stage_prefix_count"] == 3
    assert "carried-source-prefix-3" in carried["provider_output_relative_root"]

    prefix_two = {**prefix_three, "completed_stage_prefix_count": 2}
    assert provider_runner._retained_checkpoint_after_failure(
        output=tmp_path / "output-refused",
        checkpoint_root=source_root,
        checkpoint=prefix_two,
        advanced=prefix_two,
        advanced_root=None,
    ) is None

    advanced_root = tmp_path / "output-advanced/diagnostic_checkpoints/after-stage-3"
    advanced_root.mkdir(parents=True)
    (advanced_root / manifest_name).write_text("{}\n")
    advanced = provider_runner._retained_checkpoint_after_failure(
        output=tmp_path / "output-advanced",
        checkpoint_root=source_root,
        checkpoint=prefix_two,
        advanced=prefix_three,
        advanced_root=advanced_root,
    )
    assert advanced is not None
    assert advanced["completed_stage_prefix_count"] == 3
    assert advanced["provider_output_relative_root"].endswith("after-stage-3")


def test_quiescence_refuses_pid_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_command: list[str] = []

    def run_ssh(**kwargs) -> dict:
        observed_command.append(str(kwargs["remote_argv"][-1]))
        return {
            "status": "blocked",
            "stdout": "REMOTE_SESSION:identity_mismatch\n",
        }

    monkeypatch.setattr(warm, "_run_pinned_ssh", run_ssh)
    result = warm._quiesce_remote_dispatch(
        session={},
        dispatch={
            "remote_pid": 321,
            "remote_process_group_id": 321,
            "remote_session_id": 321,
            "host_key_enrollment": {"known_hosts_file": "/tmp/known-hosts"},
        },
        attempt_key="a" * 16,
    )

    assert result["status"] == "unproven"
    assert result["remote_session_absent"] is False
    assert "/proc/$root/cmdline" in observed_command[0]
    assert "/native_task_arena_warm_dispatches/aaaaaaaaaaaaaaaa/run.sh" in observed_command[0]
    assert "session_members" in observed_command[0]
    assert "BLUEPRINT_SCENE_WARM_DISPATCH_ATTEMPT=$attempt" in observed_command[0]
    assert "kill -TERM" in observed_command[0]

    invalid = warm._quiesce_remote_dispatch(
        session={},
        dispatch={
            "remote_pid": 321,
            "remote_process_group_id": 321,
            "remote_session_id": 321,
            "host_key_enrollment": {"known_hosts_file": "/tmp/known-hosts"},
        },
        attempt_key="not-hex",
    )
    assert invalid == {"status": "unproven", "remote_session_absent": False}


def test_quiescence_does_not_accept_dead_leader_while_session_child_lives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_command: list[str] = []

    def run_ssh(**kwargs) -> dict:
        command = str(kwargs["remote_argv"][-1])
        observed_command.append(command)
        if 'if ! kill -0 "$root"' in command:
            return {"status": "completed", "stdout": "REMOTE_SESSION:absent\n"}
        return {"status": "blocked", "stdout": "REMOTE_SESSION:running\n"}

    monkeypatch.setattr(warm, "_run_pinned_ssh", run_ssh)
    result = warm._quiesce_remote_dispatch(
        session={},
        dispatch={
            "remote_pid": 654,
            "remote_process_group_id": 654,
            "remote_session_id": 654,
            "host_key_enrollment": {"known_hosts_file": "/tmp/known-hosts"},
        },
        attempt_key="b" * 16,
    )

    assert result["status"] == "unproven"
    assert result["remote_session_absent"] is False
    command = observed_command[0]
    assert command.index("members=$(session_members") < command.index(
        'if kill -0 "$root"'
    )
    assert 'if ! kill -0 "$root"' not in command
    assert 'members=$(session_members || true)' in command
    assert "REMOTE_SESSION:absent" in command


def _session_root(tmp_path: Path) -> Path:
    root = tmp_path / "session"
    root.mkdir()
    watchdog = tmp_path / "watchdog"
    watchdog.mkdir()
    session = {
        "schema_version": warm.SESSION_SCHEMA_VERSION,
        "status": "ready",
        "provider": "vast",
        "provider_instance_id": 123,
        "ssh_host": "ssh.example.test",
        "ssh_port": 2222,
        "source_commit": "a" * 40,
        "bundle_sha256": "sha256:" + "b" * 64,
        "run_id": "run-warm-test",
        "toolchain_digest": "sha256:" + "c" * 64,
        "construction_envelope_digest": "sha256:" + "e" * 64,
        "source_checkpoint_digest": "sha256:" + "1" * 64,
        "scientific_binding_digest": "sha256:" + "2" * 64,
        "maximum_warm_iterations": 3,
        "maximum_warm_output_archive_bytes": 1_000_000_000,
        "aggregate_provider_compute_spend_cap_usd": 6.0,
        "bootstrap_allocation_binding_digest": "sha256:" + "d" * 64,
        "carried_completed_stage_prefix_count": 3,
        "carried_completed_stage_ids": ["stage-1", "stage-2", "stage-3"],
        "carried_paid_model_stages": [
            "artifixer_semantic_teacher",
            "artifixer_visual_review",
            "content_agents",
        ],
        "watchdog_deadline_epoch": 9_999_999_999.0,
        "watchdog_out_dir": str(watchdog),
        "watchdog_pod_name_prefix": "blueprint-task-evaluation-scene-config-test-",
        "continuing_spend": True,
        "diagnostic_only": True,
        "development_only": True,
        "qualification_eligible": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "arbitrary_command_permitted": False,
        "raw_secret_values_recorded": False,
        "session_digest": "",
    }
    session["session_digest"] = canonical_digest(session, digest_field="session_digest")
    warm._write_exclusive(root / f"{warm.SESSION_SCHEMA_VERSION}.json", session)
    state = {
        "schema_version": warm.SESSION_STATE_SCHEMA_VERSION,
        "status": "ready",
        "session_digest": session["session_digest"],
        "attempted_iteration_count": 0,
        "completed_iteration_count": 0,
        "current_checkpoint_digest": session["source_checkpoint_digest"],
        "current_remote_checkpoint_root": warm.BASE_RUNTIME_ROOT + "/input/diagnostic_checkpoint",
        "current_completed_stage_prefix_count": 3,
        "current_completed_stage_ids": ["stage-1", "stage-2", "stage-3"],
        "current_carried_paid_model_stages": list(
            warm.WARM_CARRIED_PAID_MODEL_STAGES
        ),
        "scientific_binding_digest": session["scientific_binding_digest"],
        "consumed_openai_cost_scope_attestation_digests": [],
        "continuing_spend": True,
        "state_digest": "",
    }
    state["state_digest"] = canonical_digest(state, digest_field="state_digest")
    warm._write_exclusive(root / warm.SESSION_STATE_NAME, state)
    for lifecycle in ("allocated", "container_starting", "healthy", "retained_owned"):
        record_retained_gpu_state(root, lifecycle)
    return root


def test_canonical_allocator_closeout_reaches_teardown_required_session(
    tmp_path: Path,
) -> None:
    root = _session_root(tmp_path)
    state_path = root / warm.SESSION_STATE_NAME
    state = json.loads(state_path.read_text())
    state.update({"status": "teardown_required", "state_digest": ""})
    state["state_digest"] = canonical_digest(state, digest_field="state_digest")
    warm._write_state(root, state)
    adapter_output = tmp_path / "adapter-output.json"

    result = allocator._run_scene_configuration_warm_action(
        argparse.Namespace(
            scene_configuration_warm_action="closeout",
            scene_configuration_warm_session_root=str(root),
            scene_configuration_warm_closeout_receipt=str(
                tmp_path / "closeout.json"
            ),
            scene_configuration_warm_iteration_authority=None,
            scene_configuration_job_dir=None,
            provider="vast",
            admission_out=str(tmp_path / "admission.json"),
            adapter_output=str(adapter_output),
            execute=False,
        )
    )

    assert result == 0
    assert json.loads(adapter_output.read_text())["status"] == "dry_run_ready"


def test_iteration_exception_never_cleans_objects_without_remote_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _session_root(tmp_path)
    session = json.loads(
        (root / f"{warm.SESSION_SCHEMA_VERSION}.json").read_text()
    )
    state = json.loads((root / warm.SESSION_STATE_NAME).read_text())
    authority = {
        "iteration_id": "i001-" + "a" * 12,
        "iteration_index": 1,
        "authority_digest": "sha256:" + "7" * 64,
        "source_commit": "b" * 40,
        "source_overlay_manifest_digest": "sha256:" + "8" * 64,
        "source_checkpoint_digest": session["source_checkpoint_digest"],
        "scientific_binding_digest": session["scientific_binding_digest"],
        "carried_completed_stage_prefix_count": 3,
        "carried_completed_stage_ids": ["stage-1", "stage-2", "stage-3"],
        "carried_paid_model_stages": list(warm.WARM_CARRIED_PAID_MODEL_STAGES),
    }
    monkeypatch.setattr(
        warm,
        "validate_scene_configuration_warm_iteration_authority",
        lambda **_kwargs: (session, state, authority),
    )
    monkeypatch.setattr(
        warm, "require_paid_resource_admission_grant", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        warm,
        "_consume_iteration_authority",
        lambda **_kwargs: {"status": "consumed"},
    )
    cleanup_called = False

    def cleanup(_path: Path) -> dict:
        nonlocal cleanup_called
        cleanup_called = True
        return {"status": "completed", "all_objects_absent": True}

    monkeypatch.setattr(warm, "cleanup_staged_wam_provider_objects", cleanup)
    marked: dict[str, object] = {}

    def mark(**kwargs) -> dict:
        marked.update(kwargs)
        return {}

    monkeypatch.setattr(warm, "_mark_iteration_state", mark)

    class Provider:
        def inspect(self, _instance_id: str) -> dict:
            raise OSError("ambiguous retained connection")

    job = tmp_path / "iteration"
    (job / "object_store_staging").mkdir(parents=True)
    result = warm.run_scene_configuration_warm_iteration(
        session_root=root,
        authority_path=tmp_path / "authority.json",
        job_dir=job,
        paid_resource_admission_grant=None,
        execute=True,
        provider=Provider(),
    )

    assert result["status"] == "blocked_diagnostic_only"
    assert cleanup_called is False
    assert result["object_store_cleanup"] == {}
    assert marked["status"] == "teardown_required"


def test_stage_failure_advances_safe_checkpoint_for_next_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _session_root(tmp_path)
    state_path = root / warm.SESSION_STATE_NAME
    state = json.loads(state_path.read_text())
    state.update(
        {
            "status": "iteration_running",
            "attempted_iteration_count": 1,
            "active_iteration_id": "i001-" + "a" * 12,
            "active_iteration_authority_digest": "sha256:" + "7" * 64,
            "state_digest": "",
        }
    )
    state["state_digest"] = canonical_digest(state, digest_field="state_digest")
    warm._write_state(root, state)
    advanced_digest = "sha256:" + "8" * 64
    remote_root = (
        f"{warm.REMOTE_ROOT}/iterations/i001-{'a' * 12}"
        "/output/diagnostic_checkpoints/after-stage-4"
    )
    advanced_state = warm._mark_iteration_state(
        root=root,
        state=state,
        authority={
            "iteration_id": "i001-" + "a" * 12,
            "authority_digest": "sha256:" + "7" * 64,
        },
        status="iteration_failed",
        result_digest="sha256:" + "9" * 64,
        advanced_checkpoint_digest=advanced_digest,
        advanced_checkpoint_prefix_count=4,
        advanced_checkpoint_stage_ids=[
            "stage-1",
            "stage-2",
            "stage-3",
            "stage-4",
        ],
        advanced_remote_checkpoint_root=remote_root,
    )
    assert advanced_state["status"] == "iteration_failed"

    overlay_path = tmp_path / "overlay-receipt.json"
    overlay_path.write_text("{}\n")
    overlay = {
        "source_commit": "b" * 40,
        "remote_ref": "refs/heads/codex/warm-next",
        "receipt_digest": "sha256:" + "a" * 64,
        "overlay_archive": {"sha256": "sha256:" + "b" * 64},
        "manifest_digest": "sha256:" + "c" * 64,
        "source_checkpoint_digest": advanced_digest,
        "scientific_binding_digest": "sha256:" + "2" * 64,
    }
    monkeypatch.setattr(
        warm,
        "validate_scene_configuration_warm_source_overlay",
        lambda *_args, **_kwargs: overlay,
    )
    authority = warm.materialize_scene_configuration_warm_iteration_authority(
        session_root=root,
        overlay_receipt_path=overlay_path,
        output_path=tmp_path / "iteration-2-authority.json",
        observed_now_epoch=1.0,
    )
    assert authority["iteration_index"] == 2
    assert authority["source_checkpoint_digest"] == advanced_digest
    assert authority["carried_completed_stage_prefix_count"] == 4
    assert authority["carried_completed_stage_ids"][-1] == "stage-4"
    assert authority["remote_checkpoint_root"] == remote_root


def test_closeout_is_resumable_after_terminate_exception_and_double_proves_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _session_root(tmp_path)
    monkeypatch.setattr(warm, "require_paid_resource_admission_grant", lambda *args, **kwargs: None)

    class Provider:
        absent = False
        fail = True

        def terminate(self, instance_id: str) -> dict:
            assert instance_id == "123"
            if self.fail:
                raise OSError("credential must stay redacted")
            self.absent = True
            return {"status": "stopped"}

        def inspect(self, instance_id: str) -> dict:
            if self.absent:
                return {
                    "status": "absent",
                    "provider_absence_confirmed": True,
                    "api_confirmed": True,
                }
            return {"status": "observed", "api_confirmed": True}

        def billable_inventory(self, *, name_prefix: str) -> dict:
            return {
                "api_confirmed": True,
                "live_resource_count": 0 if self.absent else 1,
                "resources": [] if self.absent else [{"instance_id": "123"}],
            }

    provider = Provider()
    ticks = iter(range(100))
    first = warm.close_scene_configuration_warm_session(
        session_root=root,
        output_path=tmp_path / "closeout-failed.json",
        paid_resource_admission_grant=None,
        execute=True,
        provider=provider,
        sleep=lambda _seconds: None,
        monotonic=lambda: float(next(ticks)),
        timeout_seconds=1,
    )
    assert first["status"] == "blocked"
    assert "credential must stay redacted" not in json.dumps(first)

    provider.fail = False
    ticks = iter(range(100))
    second = warm.close_scene_configuration_warm_session(
        session_root=root,
        output_path=tmp_path / "closeout-completed.json",
        paid_resource_admission_grant=None,
        execute=True,
        provider=provider,
        sleep=lambda _seconds: None,
        monotonic=lambda: float(next(ticks)),
        timeout_seconds=10,
    )
    assert second["status"] == "completed"
    assert second["provider_instance_absent"] is True
    assert second["global_provider_zero_proven"] is True
    assert len(second["global_billable_inventory_observations"]) == 2
