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


def test_live_run_projects_setup_blockers_instead_of_generic_status(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"status": "blocked", "blockers": [],
                          "setup_blockers": ["real_arena_execution:missing_simulator_command"]})
    audit = alert.build_live_pipeline_manifest_alert(
        manifest_path=manifest, output_path=tmp_path / "alert.json", dry_run=True,
    )
    assert audit["alert_required"] is True
    assert audit["blockers"] == ["real_arena_execution:missing_simulator_command"]
    assert "real_arena_execution:missing_simulator_command" in audit["message_text"]
    assert "status contains blocked" not in audit["message_text"]


def test_blocked_alert_counts_the_blockers_it_does_not_list(tmp_path: Path) -> None:
    """2026-09-30: a pass with 13 setup blockers reached Slack as a bare status."""
    manifest = tmp_path / "manifest.json"
    setup = [f"setup_blocker_{index:02d}" for index in range(13)]
    _write_json(manifest, {"status": "blocked", "blockers": [], "setup_blockers": setup})

    audit = alert.build_live_pipeline_manifest_alert(
        manifest_path=manifest, output_path=tmp_path / "alert.json", dry_run=True,
    )

    assert audit["blocker_count"] == 13
    assert audit["blockers"] == setup[:12]
    assert audit["message_text"].endswith("blockers=" + ", ".join(setup[:5]) + " (+8 more)")


def _sending_run(tmp_path: Path, monkeypatch, sent: list[str], *, fail: bool = False):
    def fake_post(url: str, payload: dict[str, object], *, timeout_seconds: float) -> None:
        if fail:
            raise RuntimeError("webhook returned HTTP 500")
        sent.append(str(payload["text"]))

    monkeypatch.setattr(alert, "_post_webhook", fake_post)
    return lambda manifest, now: alert.build_live_pipeline_manifest_alert(
        manifest_path=manifest, output_path=tmp_path / "alert.json",
        webhook_url="https://hooks.example/blueprint", require_webhook=True, now=now,
    )


def test_an_unchanged_blocked_pass_repeats_hourly_instead_of_every_pass(
    tmp_path: Path, monkeypatch,
) -> None:
    """The control-plane timer runs every five minutes; the blocked state did not change."""
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"status": "blocked", "setup_blockers": ["missing_simulator_command"]})
    sent: list[str] = []
    run = _sending_run(tmp_path, monkeypatch, sent)

    assert run(manifest, 1_000.0)["notification_status"] == "sent"
    repeat = run(manifest, 1_300.0)
    assert repeat["notification_status"] == "suppressed_unchanged"
    assert repeat["last_sent_at_epoch"] == 1_000.0 and alert._exit_code(repeat) == 0
    assert run(manifest, 1_000.0 + 3_599)["notification_status"] == "suppressed_unchanged"
    assert run(manifest, 1_000.0 + 3_600)["notification_status"] == "sent"
    assert len(sent) == 2

    _write_json(manifest, {"status": "blocked",
                           "setup_blockers": ["missing_simulator_command", "missing_inbox"]})
    assert run(manifest, 4_700.0)["notification_status"] == "sent"
    _write_json(manifest, {"status": "blocked_on_inbox",
                           "setup_blockers": ["missing_simulator_command", "missing_inbox"]})
    assert run(manifest, 4_800.0)["notification_status"] == "sent"
    assert len(sent) == 4


def test_a_failed_delivery_is_retried_on_the_next_pass(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"status": "blocked", "setup_blockers": ["missing_inbox"]})
    sent: list[str] = []

    failed = _sending_run(tmp_path, monkeypatch, sent, fail=True)(manifest, 1_000.0)
    assert failed["notification_status"] == "failed" and failed["last_sent_at_epoch"] is None
    retried = _sending_run(tmp_path, monkeypatch, sent)(manifest, 1_300.0)
    assert retried["notification_status"] == "sent" and len(sent) == 1


def test_spend_lock_and_threshold_pages_are_never_suppressed(tmp_path: Path, monkeypatch) -> None:
    sent: list[str] = []
    run = _sending_run(tmp_path, monkeypatch, sent)
    for name, payload in (
        ("spend.json", {"schema_version": "blueprint.paid_spend_admission_lock.v1",
                        "status": "blocked", "blockers": ["cohort_hard_stop_reached"]}),
        ("page.json", {"status": "ready", "page_event": {"required": True}}),
    ):
        manifest = tmp_path / name
        _write_json(manifest, payload)
        assert run(manifest, 1_000.0)["notification_status"] == "sent"
        assert run(manifest, 1_300.0)["notification_status"] == "sent"
    assert len(sent) == 4


def test_setup_projection_keeps_the_existing_blocker_bound() -> None:
    assert alert._manifest_blockers({"blockers": [f"blocked-{i}" for i in range(12)],
                                     "setup_blockers": ["setup"],
                                     "setup": {"blockers": ["nested"]}}) == [
                                         f"blocked-{i}" for i in range(12)]


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
