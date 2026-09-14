from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import vast_provider_output_recovery as recovery


def _install_identity(monkeypatch, tmp_path: Path) -> Path:
    identity = tmp_path / "id_ed25519"
    identity.write_text("private-test-key", encoding="utf-8")
    identity.chmod(0o600)
    monkeypatch.setenv(recovery.VAST_SSH_IDENTITY_FILE_ENV, str(identity))
    known_hosts = tmp_path / "attempt" / "vast_ssh_known_hosts"
    known_hosts.parent.mkdir()
    known_hosts.write_text("pinned", encoding="utf-8")
    monkeypatch.setattr(
        recovery,
        "enroll_vast_ssh_host_key",
        lambda *_args, **_kwargs: {
            "status": "enrolled",
            "known_hosts_file": str(known_hosts),
        },
    )
    monkeypatch.setattr(
        recovery,
        "_validated_vast_known_hosts_pin",
        lambda *_args, **_kwargs: (known_hosts, "a" * 64),
    )
    return identity


def test_recovery_streams_stdout_to_partial_file_and_verifies_digest(
    monkeypatch, tmp_path: Path
) -> None:
    _install_identity(monkeypatch, tmp_path)
    payload = b"sealed-provider-archive"
    digest = hashlib.sha256(payload).hexdigest()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if kwargs.get("text"):
            return SimpleNamespace(returncode=0, stdout=f"{len(payload)} {digest}\n")
        assert "stdout" in kwargs
        kwargs["stdout"].write(payload)
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(recovery.subprocess, "run", fake_run)
    output = tmp_path / "result.zip"
    result = recovery.recover_provider_output_before_teardown(
        connection={"ssh_host": "example.invalid", "ssh_port": 2222},
        provider_bundle_kind="native_task_arena_policy_canary_session",
        output_path=output,
        attempt_dir=tmp_path / "attempt",
        expected_size_bytes=len(payload),
    )

    assert result["status"] == "completed"
    assert result["streamed_to_disk"] is True
    assert output.read_bytes() == payload
    assert len(calls) == 2
    assert calls[1][1].get("stdout") is not None
    assert calls[1][1].get("capture_output") is None


def test_recovery_refuses_remote_size_mismatch(monkeypatch, tmp_path: Path) -> None:
    _install_identity(monkeypatch, tmp_path)
    monkeypatch.setattr(
        recovery.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=f"99 {'b' * 64}\n"
        ),
    )

    result = recovery.recover_provider_output_before_teardown(
        connection={"ssh_host": "example.invalid", "ssh_port": 2222},
        provider_bundle_kind="native_task_arena_policy_canary_session",
        output_path=tmp_path / "result.zip",
        attempt_dir=tmp_path / "attempt",
        expected_size_bytes=100,
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == [
        "provider_output_ssh_recovery_remote_size_mismatch"
    ]


def test_unknown_result_size_is_bounded_and_reserved_for_sam(monkeypatch, tmp_path):
    _install_identity(monkeypatch, tmp_path)
    result = recovery.recover_provider_output_before_teardown(
        connection={'ssh_host':'example.invalid','ssh_port':2222},
        provider_bundle_kind='native_task_arena', output_path=tmp_path/'result.zip',
        attempt_dir=tmp_path/'attempt', expected_size_bytes=None, maximum_size_bytes=64)
    assert result['status'] == 'blocked'
    assert result['blockers'] == ['provider_output_ssh_recovery_expected_size_invalid']


def test_metadata_and_transfer_share_one_recovery_deadline(monkeypatch, tmp_path):
    _install_identity(monkeypatch, tmp_path)
    moments = iter([0., 4., 8.])
    monkeypatch.setattr(recovery.time, 'monotonic', lambda: next(moments))
    payload = b'{}'
    timeouts = []
    def ssh(_command, **kwargs):
        timeouts.append(kwargs['timeout'])
        if kwargs.get('text'):
            return SimpleNamespace(returncode=0, stdout=f'2 {hashlib.sha256(payload).hexdigest()}\n')
        kwargs['stdout'].write(payload)
        return SimpleNamespace(returncode=0, stderr=b'')
    monkeypatch.setattr(recovery.subprocess, 'run', ssh)
    result = recovery.recover_provider_output_before_teardown(
        connection={'ssh_host':'example.invalid','ssh_port':2222},
        provider_bundle_kind='sam31_source_tracks', output_path=tmp_path/'result.json',
        attempt_dir=tmp_path/'attempt', expected_size_bytes=None, maximum_size_bytes=64,
        timeout_seconds=12)
    assert result['status'] == 'completed'
    assert timeouts == [8., 4.]



@pytest.mark.parametrize("stage_checkpoint", [False, True])
def test_scene_configuration_recovers_unmarked_partial_archive_with_bound(monkeypatch, tmp_path, stage_checkpoint):
    _install_identity(monkeypatch, tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("stages/stage-1/checkpoint.pt", b"completed-training")
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if kwargs.get("text"):
            return SimpleNamespace(returncode=0, stdout=f"{len(payload)} {digest}\n")
        kwargs["stdout"].write(payload)
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(recovery.subprocess, "run", run)
    output = tmp_path / "partial.zip"
    result = recovery.recover_provider_output_before_teardown(
        connection={"ssh_host": "example.invalid", "ssh_port": 2222},
        provider_bundle_kind="task_evaluation_scene_configuration", output_path=output,
        attempt_dir=tmp_path / "attempt", expected_size_bytes=None, maximum_size_bytes=1024, stage_checkpoint=stage_checkpoint,
    )
    assert result["status"] == "completed" and output.read_bytes() == payload
    filename = ("task_evaluation_scene_configuration_stage_checkpoint.zip" if stage_checkpoint
                else "task_evaluation_scene_configuration_provider_output.zip")
    assert all(filename in c[-1] for c in calls)


def test_scene_configuration_unknown_size_cannot_be_unbounded(tmp_path):
    result = recovery.recover_provider_output_before_teardown(
        connection={"ssh_host": "example.invalid", "ssh_port": 2222},
        provider_bundle_kind="task_evaluation_scene_configuration", output_path=tmp_path / "result.zip",
        attempt_dir=tmp_path / "attempt", expected_size_bytes=None,
    )
    assert result["blockers"] == ["provider_output_ssh_recovery_expected_size_invalid"]
