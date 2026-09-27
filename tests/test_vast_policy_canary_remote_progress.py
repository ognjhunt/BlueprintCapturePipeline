from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline import vast_policy_canary_remote_progress as progress


def test_remote_progress_keeps_only_fixed_milestones(tmp_path: Path, monkeypatch) -> None:
    identity = tmp_path / "identity"
    identity.write_text("test-only", encoding="ascii")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("test-only", encoding="ascii")
    monkeypatch.setattr(progress, "_identity_file", lambda: identity)
    monkeypatch.setattr(
        progress, "enroll_vast_ssh_host_key",
        lambda *_args, **_kwargs: {
            "status": "enrolled", "known_hosts_file": str(known_hosts),
        },
    )
    monkeypatch.setattr(
        progress, "_validated_vast_known_hosts_pin",
        lambda *_args, **_kwargs: (known_hosts, "a" * 64),
    )
    monkeypatch.setattr(progress, "_ssh_command", lambda **_kwargs: ["ssh"])
    monkeypatch.setattr(
        progress.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                b"REMOTE_STAGE:runtime_source_receipt\n"
                b"BLUEPRINT_POLICY_CANARY_PROGRESS:cell=0:stage=static_preflight_passed\n"
                b"raw secret value must never be retained\n"
                b"BLUEPRINT_POLICY_CANARY_PROGRESS:cell=99:stage=policy_loaded\n"
            ),
        ),
    )
    result = progress.probe_policy_canary_remote_progress(
        {"ssh_host": "ssh.vast.ai", "ssh_port": 1234}, attempt_dir=tmp_path,
    )
    assert result["status"] == "observed"
    assert result["milestones"] == [
        "REMOTE_STAGE:runtime_source_receipt",
        "BLUEPRINT_POLICY_CANARY_PROGRESS:cell=0:stage=static_preflight_passed",
    ]
    assert "secret" not in str(result)
    assert result["raw_remote_output_recorded"] is False


def test_remote_progress_requires_pinned_ssh(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(progress, "_identity_file", lambda: None)
    result = progress.probe_policy_canary_remote_progress(
        {"ssh_host": "ssh.vast.ai", "ssh_port": 1234}, attempt_dir=tmp_path,
    )
    assert result == {
        "status": "unavailable", "milestones": [],
        "reason": "endpoint_or_identity_unavailable",
    }
