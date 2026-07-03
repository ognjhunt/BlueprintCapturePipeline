from __future__ import annotations

import json
import subprocess
from pathlib import Path

from blueprint_pipeline import unitree_groot_sonic_provider_readiness as R


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_provider_readiness_blocks_when_sealed_image_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="not found")

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    monkeypatch.setattr(R, "_file_status", lambda path: {"path": str(path), "present": True})
    packet = _write_json(tmp_path / "packet.json", {"status": "ready", "tarball_path": "packet.tgz"})
    staging = _write_json(
        tmp_path / "staging.json",
        {
            "status": "completed",
            "bundle_key": "blueprint/key",
            "presigned_url_expiry": {"expires_at": "2026-07-04T00:00:00Z"},
        },
    )

    result = R.build_provider_readiness(
        output_path=tmp_path / "readiness.json",
        image_ref="registry.example/img:tag",
        remote_build_packet_manifest=packet,
        object_store_staging_manifest=staging,
        generated_at="2026-07-03T00:00:00+00:00",
    )

    assert result["status"] == "blocked_before_paid_provider_canaries"
    assert "sealed_image_not_registry_fetchable" in result["blockers"]
    assert result["providers"]["runpod"]["paid_launch_allowed_by_readiness"] is False
    assert result["providers"]["digitalocean"]["paid_launch_allowed_by_readiness"] is False
    assert result["next_required_action"] == "run_remote_build_packet_and_push_sealed_image"
    assert result["claim_boundary"]["readiness_audit_is_no_spend"] is True


def test_provider_readiness_ready_when_image_and_credentials_present(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = json.dumps({"mediaType": "application/vnd.oci.image.index.v1+json"})

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=manifest, stderr="")

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    monkeypatch.setattr(
        R,
        "_file_status",
        lambda path: {"path": str(path), "present": True, "raw_secret_values_recorded": False},
    )

    result = R.build_provider_readiness(
        output_path=tmp_path / "readiness.json",
        image_ref="registry.example/img:tag",
        generated_at="2026-07-03T00:00:00+00:00",
    )

    assert result["status"] == "ready_for_paid_provider_canaries"
    assert result["paid_runtime_comparison_allowed"] is True
    assert result["providers"]["runpod"]["status"] == "ready_for_paid_canary"
    assert result["providers"]["digitalocean"]["status"] == "ready_for_paid_canary"
    assert result["next_required_action"] == "run_paid_provider_startup_canaries_before_task_episode"
