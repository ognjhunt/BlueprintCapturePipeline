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


def test_live_pipeline_manifest_alert_rejects_unsafe_webhook_url(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"status": "blocked", "blockers": ["operator_action"]})

    audit = alert.build_live_pipeline_manifest_alert(
        manifest_path=manifest,
        output_path=tmp_path / "alert.json",
        webhook_url="file:///etc/passwd",
        require_webhook=True,
    )

    assert audit["notification_status"] == "failed"
    assert audit["notification_attempted"] is True
    assert "credential-free HTTPS origin" in audit["notification_error"]
    assert alert._exit_code(audit) == 1


def test_spend_admission_lock_uses_dedicated_critical_page_message(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "paid-spend-admission.json"
    _write_json(
        manifest,
        {
            "schema_version": "blueprint.paid_spend_admission_lock.v1",
            "status": "blocked",
            "effective_spend_usd": 5000.0,
            "hard_stop_usd": 5000.0,
            "blockers": ["cohort_hard_stop_reached"],
        },
    )

    audit = alert.build_live_pipeline_manifest_alert(
        manifest_path=manifest,
        output_path=tmp_path / "page-audit.json",
        webhook_url="https://hooks.example/blueprint",
        require_webhook=True,
        dry_run=True,
    )

    assert audit["alert_required"] is True
    assert audit["notification_status"] == "dry_run"
    assert "paid spend admission is locked" in audit["message_text"]
    assert "effective_spend_usd=5000.0" in audit["message_text"]
    assert "cohort_hard_stop_reached" in audit["message_text"]


def test_spend_override_still_requires_threshold_crossing_page(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "paid-spend-admission.json"
    _write_json(
        manifest,
        {
            "schema_version": "blueprint.paid_spend_admission_lock.v1",
            "status": "override_open",
            "effective_spend_usd": 5000.0,
            "hard_stop_usd": 5000.0,
            "blockers": [],
            "page_event": {
                "required": True,
                "delivery_status": "external_pending",
            },
        },
    )

    audit = alert.build_live_pipeline_manifest_alert(
        manifest_path=manifest,
        output_path=tmp_path / "page-audit.json",
        require_webhook=True,
    )

    assert audit["alert_required"] is True
    assert audit["notification_status"] == "blocked_missing_required_webhook"
    assert "paid spend override is active" in audit["message_text"]
    assert "threshold crossing requires operator notification" in audit["message_text"]


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
