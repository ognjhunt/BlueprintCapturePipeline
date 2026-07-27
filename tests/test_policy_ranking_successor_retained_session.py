from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import policy_ranking_successor_retained_session as retained
from blueprint_pipeline.retained_gpu_session_lifecycle import record_retained_gpu_state


def _prime_lifecycle(root: Path) -> None:
    for state in (
        "allocated",
        "container_starting",
        "image_pulling",
        "healthy",
        "retained_owned",
    ):
        record_retained_gpu_state(root, state)


def _session(root: Path) -> Path:
    path = root / retained.SESSION_NAME
    path.write_text(
        json.dumps(
            {
                "schema_version": retained.SCHEMA_VERSION,
                "status": "retained_owned",
                "continuing_spend": True,
                "provider_instance_id": "123",
                "source_commit": "a" * 40,
                "dirty_state_declaration": "clean_exact_commit",
                "current_runtime_bundle_sha256": "b" * 64,
                "authorization_receipt_sha256": "c" * 64,
                "image_digest": "sha256:" + "d" * 64,
                "checkpoint": "nvidia/Cosmos3-Nano",
                "checkpoint_revision": "e" * 40,
                "watchdog_deadline_epoch": 9_999_999_999.0,
                "ssh_connection": {},
                "known_hosts_file": str(root / "known_hosts"),
                "refresh_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return path


def test_create_retained_session_binds_provider_and_tofu_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Provider:
        @staticmethod
        def inspect(instance_id: str) -> dict[str, object]:
            assert instance_id == "123"
            return {
                "status": "observed",
                "direct_port_ready": True,
                "ssh_connection": {"ssh_host": "127.0.0.1", "ssh_port": 2222},
            }

    monkeypatch.setattr(retained, "get_render_provider", lambda name: Provider())
    monkeypatch.setattr(
        retained,
        "enroll_vast_ssh_host_key",
        lambda *args, **kwargs: {
            "status": "enrolled",
            "tofu_pinned": True,
            "known_hosts_file": str(tmp_path / "known_hosts"),
            "known_hosts_sha256": "f" * 64,
        },
    )

    result = retained.create_retained_session_manifest(
        job_dir=tmp_path,
        adapter_result={"retained_owned": True, "vast_instance_ids": [123]},
        watchdog_handoff={"watchdog_pid": 99, "watchdog_deadline_epoch": 2000.0},
        source_commit="a" * 40,
        dirty_state_declaration="clean_exact_commit",
        bundle_sha256="b" * 64,
        authorization_receipt_sha256="c" * 64,
        image_digest="sha256:" + "d" * 64,
        checkpoint="nvidia/Cosmos3-Nano",
        checkpoint_revision="e" * 40,
    )

    assert result["status"] == "retained_owned"
    assert result["provider_instance_id"] == "123"
    assert Path(result["session_manifest"]).stat().st_mode & 0o077 == 0


def test_failed_refresh_returns_to_retained_owned_without_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prime_lifecycle(tmp_path)
    session = _session(tmp_path)
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"bundle")
    token = tmp_path / "token"
    token.write_text("secret", encoding="utf-8")
    monkeypatch.setattr(
        retained,
        "_run_remote_refresh",
        lambda **kwargs: {
            "status": "blocked",
            "blockers": ["runtime_bug"],
            "remote_result": {"server_remained_loaded": True},
        },
    )

    result = retained.refresh_retained_session(
        session_manifest=session,
        bundle_path=bundle,
        public_base_url="https://example.test",
        token_file=token,
        source_commit="a" * 40,
        dirty_state_declaration="clean_exact_commit",
        authorization_receipt_sha256="c" * 64,
    )

    assert result["status"] == "retained_owned"
    assert result["continuing_spend"] is True
    persisted = json.loads(session.read_text())
    assert persisted["last_refresh"]["signed_urls_stored"] is False
    assert "secret" not in session.read_text()


def test_successful_refresh_tears_down_and_proves_provider_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prime_lifecycle(tmp_path)
    session = _session(tmp_path)
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"bundle")
    token = tmp_path / "token"
    token.write_text("secret", encoding="utf-8")
    monkeypatch.setattr(
        retained,
        "_run_remote_refresh",
        lambda **kwargs: {
            "status": "completed",
            "blockers": [],
            "remote_result": {
                "server_remained_loaded": True,
                "audit_sha256": "f" * 64,
            },
        },
    )

    class Provider:
        @staticmethod
        def terminate(instance_id: str) -> dict[str, object]:
            return {"status": "terminated", "instance_id": instance_id}

        @staticmethod
        def inspect(instance_id: str) -> dict[str, object]:
            return {
                "status": "absent",
                "instance_id": instance_id,
                "provider_absence_confirmed": True,
            }

    monkeypatch.setattr(retained, "get_render_provider", lambda name: Provider())

    result = retained.refresh_retained_session(
        session_manifest=session,
        bundle_path=bundle,
        public_base_url="https://example.test",
        token_file=token,
        source_commit="a" * 40,
        dirty_state_declaration="clean_exact_commit",
        authorization_receipt_sha256="c" * 64,
    )

    assert result["status"] == "provider_absent"
    assert result["continuing_spend"] is False
    assert result["provider_zero"]["provider_absence_confirmed"] is True
