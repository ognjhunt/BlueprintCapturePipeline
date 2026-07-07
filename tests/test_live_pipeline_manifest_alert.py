from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import live_pipeline_manifest_alert as alert


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_live_pipeline_manifest_alert_noops_when_manifest_ready(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "alert.json"
    _write_json(manifest, {"status": "processed_jobs", "blockers": []})

    audit = alert.build_live_pipeline_manifest_alert(
        manifest_path=manifest,
        output_path=output,
        require_webhook=True,
    )

    assert audit["alert_required"] is False
    assert audit["notification_status"] == "not_required"
    assert alert._exit_code(audit) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == (
        alert.LIVE_PIPELINE_MANIFEST_ALERT_SCHEMA_VERSION
    )


def test_live_pipeline_manifest_alert_fails_closed_when_blocked_without_required_webhook(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "status": "local_ready_live_external_blocked",
            "blockers": ["missing_delivery_command"],
        },
    )

    audit = alert.build_live_pipeline_manifest_alert(
        manifest_path=manifest,
        output_path=tmp_path / "alert.json",
        require_webhook=True,
    )

    assert audit["alert_required"] is True
    assert audit["notification_status"] == "blocked_missing_required_webhook"
    assert audit["webhook_required"] is True
    assert alert._exit_code(audit) == 2
    assert "missing_delivery_command" in audit["message_text"]


def test_live_pipeline_manifest_alert_sends_bounded_webhook(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "status": "blocked",
            "job_id": "job-1",
            "capture_root": "/captures/capture-1",
            "blockers": ["missing_capture_root"],
        },
    )
    sent: list[tuple[str, dict[str, object], float]] = []

    def fake_post(url: str, payload: dict[str, object], *, timeout_seconds: float) -> None:
        sent.append((url, payload, timeout_seconds))

    monkeypatch.setattr(alert, "_post_webhook", fake_post)

    audit = alert.build_live_pipeline_manifest_alert(
        manifest_path=manifest,
        output_path=tmp_path / "alert.json",
        webhook_url="https://hooks.example/blueprint",
        require_webhook=True,
        timeout_seconds=3,
    )

    assert audit["notification_status"] == "sent"
    assert alert._exit_code(audit) == 0
    assert sent == [
        (
            "https://hooks.example/blueprint",
            {
                "text": (
                    "Blueprint live pipeline control plane is blocked: status=blocked. "
                    f"manifest={manifest.resolve()} job_id=job-1 "
                    "capture_root=/captures/capture-1 blockers=missing_capture_root"
                )
            },
            3,
        )
    ]
    assert "hooks.example" not in json.dumps(audit)


def test_live_pipeline_manifest_alert_cli_exit_codes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"status": "blocked", "blockers": ["missing_inbox"]})

    assert (
        alert.main(
            [
                "--manifest-path",
                str(manifest),
                "--output-path",
                str(tmp_path / "alert.json"),
                "--require-webhook",
            ]
        )
        == 2
    )
    assert (
        alert.main(
            [
                "--manifest-path",
                str(manifest),
                "--output-path",
                str(tmp_path / "alert-dry-run.json"),
                "--webhook-url",
                "https://hooks.example/blueprint",
                "--dry-run",
            ]
        )
        == 0
    )
