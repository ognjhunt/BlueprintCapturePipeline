"""Capacity is measured, forecast, alerted and grown before intake has to refuse anything."""

from __future__ import annotations

import json
import os
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_capacity_controller as cap
from blueprint_pipeline import control_plane_disk_budget as disk_budget

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))
from operator_door.secrets_guard import scan_bytes  # noqa: E402
from operator_door.status import _small_json  # noqa: E402

Usage = namedtuple("Usage", "total used free")
GIB = 1024**3
MIB = 1024**2


def _usage(free_gib: float, total_gib: float = 154.0):
    total = int(total_gib * GIB)
    free = int(free_gib * GIB)
    return lambda _path: Usage(total, total - free, free)


def _reservation(root: Path, name: str, *, expected_bytes: int, expires_at: float) -> None:
    # A live ledger entry as the ledger writes it: on the tmp mount's device and
    # held by a live pid, so it counts under the ledger's own liveness rules.
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.json").write_text(
        json.dumps({"expected_bytes": expected_bytes, "expires_at_epoch": expires_at,
                    "device": root.stat().st_dev, "pid": os.getpid()}),
        encoding="utf-8",
    )


def test_measurement_projects_admission_exactly_as_intake_does(tmp_path: Path) -> None:
    """The host at 9.67 GiB free: an 8 GiB floor leaves 1.67 GiB, so every 2 GiB
    stage is refused.  That was the 503 nobody saw coming."""
    ledger = tmp_path / "reservations"
    _reservation(ledger, "live", expected_bytes=GIB, expires_at=2_000.0)
    _reservation(ledger, "expired", expected_bytes=5 * GIB, expires_at=500.0)

    row = cap.measure_mount(tmp_path, reservation_root=ledger, disk_usage=_usage(9.67), now=1_000.0)

    assert row["status"] == "measured" and row["level"] == "critical"
    assert row["floor_bytes"] == 8 * GIB
    assert row["reserved_bytes"] == GIB and row["live_reservations"] == 1
    assert row["refused_roles"] == sorted(cap.CHAIN_ROLES)
    assert row["free_needed_for_one_role_bytes"] == 10 * GIB
    healthy = cap.measure_mount(tmp_path, reservation_root=ledger, disk_usage=_usage(60.0), now=1_000.0)
    assert healthy["level"] == "ok" and healthy["refused_roles"] == []
    warning = cap.measure_mount(tmp_path, reservation_root=ledger, disk_usage=_usage(40.0), now=1_000.0)
    assert warning["level"] == "warning" and warning["refused_roles"] == []


def test_forecast_uses_the_oldest_observation_inside_the_window() -> None:
    now = 10 * 86400.0
    history = [
        {"mount": "/m", "status": "measured", "observed_at_epoch": now - 2 * 86400, "free_bytes": 40 * GIB, "floor_bytes": 8 * GIB},
        {"mount": "/m", "status": "measured", "observed_at_epoch": now - 30 * 86400, "free_bytes": 100 * GIB, "floor_bytes": 8 * GIB},
        {"mount": "/other", "status": "measured", "observed_at_epoch": now - 86400, "free_bytes": 1, "floor_bytes": 0},
    ]
    current = {"mount": "/m", "status": "measured", "free_bytes": 30 * GIB, "floor_bytes": 8 * GIB}

    result = cap.forecast(history, current, now=now)

    assert result["status"] == "growing"
    assert result["growth_bytes_per_day"] == 5 * GIB
    assert result["days_until_floor"] == pytest.approx(4.4, abs=0.01)
    assert cap.forecast([], current, now=now) == {"status": "insufficient_history"}
    assert cap.forecast(history[:1], {**current, "free_bytes": 50 * GIB}, now=now)["status"] == "not_growing"


def test_three_days_of_headroom_pages_and_unrouted_alerts_are_loud(tmp_path: Path) -> None:
    now = 10 * 86400.0
    history = [{"mount": str(tmp_path), "status": "measured", "observed_at_epoch": now - 86400,
                "free_bytes": 23 * GIB, "floor_bytes": 8 * GIB}]
    report = cap.build_capacity_report(mounts=[str(tmp_path)], reservation_root=tmp_path / "r",
                                       history=history, disk_usage=_usage(18.0), now=now)
    forecast = next(a for a in report["alerts"] if a["code"] == "floor_within_three_days")
    assert forecast["severity"] == "page"
    assert forecast["days_until_floor"] == pytest.approx(2.0)

    unrouted = cap.run_controller(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
                                  reservation_root=tmp_path / "r", webhook_url="", volume=None,
                                  ack="", token="", survey=None, disk_usage=_usage(80.0), now=now)
    assert unrouted["level"] in {"warning", "critical"}
    assert {a["code"]: a["severity"] for a in unrouted["alerts"]}["operator_alert_route_unconfigured"] == "page"
    assert "operator_alert_route_unconfigured" in {a["code"] for a in cap.capacity_summary(unrouted)["alerts"]}


def test_fast_growth_pages_even_while_current_utilization_is_ok(tmp_path: Path) -> None:
    now = 10 * 86400.0
    mount = str(tmp_path)
    report_root = tmp_path / "capacity"
    report_root.mkdir()
    (report_root / "history.jsonl").write_text(json.dumps({
        "mount": mount, "status": "measured", "observed_at_epoch": now - 86400,
        "free_bytes": 120 * GIB, "floor_bytes": 8 * GIB,
    }) + "\n")
    posted = []
    report = cap.run_controller(
        mounts=[mount], report_root=report_root, reservation_root=tmp_path / "reservations",
        webhook_url="https://alerts.example/hook", volume=None, ack="", token="",
        survey=None, disk_usage=_usage(80.0), now=now,
        poster=lambda _url, value: posted.append(value),
        release_retirement_summary_path=tmp_path / "absent-retirement",
        break_glass_notes_root=tmp_path / "absent-notes",
    )
    assert report["level"] == "ok"
    assert any(a["code"] == "floor_within_three_days" and a["severity"] == "page"
               for a in report["alerts"])
    assert report["alert_posted"] is True and len(posted) == 1


def test_blocked_volume_growth_pages(tmp_path: Path) -> None:
    posted = []
    report = cap.run_controller(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
                                reservation_root=tmp_path / "r", webhook_url="https://alerts.example/hook",
                                volume={"id": "vol-1", "mount": str(tmp_path), "current_size_gib": 100,
                                        "max_gib": 100}, ack="", token="", survey=None,
                                poster=lambda _url, value: posted.append(value),
                                disk_usage=_usage(9.0), now=1000.0)
    growth = next(a for a in report["alerts"] if a["code"] == "volume_growth_blocked")
    assert growth["severity"] == "page" and growth["reason"] == "volume_at_maximum"
    assert len(posted) == 1 and posted[0]["volume_resize"]["status"] == "blocked"


def test_blocked_growth_page_keeps_provider_status_out_of_public_summary(tmp_path: Path) -> None:
    def reject(_plan, **_kwargs):
        raise cap.ControlPlaneCapacityError("control_plane_capacity_resize_rejected:private_provider_status")

    report = cap.run_controller(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
                                reservation_root=tmp_path / "r", webhook_url="https://alerts.example/hook",
                                volume={"id": "vol-1", "mount": str(tmp_path), "current_size_gib": 100,
                                        "max_gib": 200}, ack=cap.RESIZE_ACK, token="dummy", resizer=reject,
                                survey=None, disk_usage=_usage(9.0), now=1000.0)
    assert report["volume_resize"]["reason"] == "control_plane_capacity_resize_rejected"
    summary = cap.capacity_summary(report)
    assert "private_provider_status" not in json.dumps(summary)
    assert next(a for a in summary["alerts"] if a["code"] == "volume_growth_blocked")["reason"] == (
        "control_plane_capacity_resize_rejected")


def test_new_page_alert_posts_even_at_the_same_level() -> None:
    first = {"level": "critical", "last_alert_epoch": 1000.0, "last_alert_fingerprint": "old",
             "alerts": [{"code": "admission_refused", "mount": "/first", "severity": "page"}]}
    second = {"level": "critical", "alerts": [
        {"code": "admission_refused", "mount": "/first", "severity": "page"},
        {"code": "volume_growth_blocked", "mount": "/first", "severity": "page"}],
    }
    assert cap.alert_due(first, second, now=1100.0)
    assert cap.alert_fingerprint(first) != cap.alert_fingerprint(second)


def test_page_webhook_has_route_fingerprint_runbook_and_short_summary(monkeypatch) -> None:
    payloads = []

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def urlopen(request, *, timeout):
        assert timeout == 10.0
        payloads.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(cap.urllib.request, "urlopen", urlopen)
    report = {"level": "critical", "mounts": [], "alerts": [
        {"code": "floor_within_three_days", "mount": "/var/lib/blueprint",
         "days_until_floor": 2.0, "severity": "page"}]}
    cap.post_alert("https://alerts.example/hook", report)
    payload = payloads[0]
    assert payload["page"] is True and payload["severity"] == "page"
    assert payload["fingerprint"] == cap.alert_fingerprint(report)
    assert payload["runbook"] == "docs/runbooks/control-plane-capacity.md"
    assert "2.0 days" in payload["summary"] and len(payload["summary"]) <= 200
    report["alerts"].append({
        "code": "reclaim_ineffective", "severity": "page",
        "top_retained_reasons": ["protected_process", "young", "pinned"],
    })
    cap.post_alert("https://alerts.example/hook", report)
    assert "protected_process, young, pinned" in payloads[1]["summary"]


def test_release_retirement_attention_appears_in_capacity_report(tmp_path: Path) -> None:
    summary = tmp_path / "release-retention" / "latest-deploy-retirement.json"
    summary.parent.mkdir()
    summary.write_text(json.dumps({"status": "blocked", "alerts": ["lease_unreadable"]}))
    report = cap.run_controller(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
                                reservation_root=tmp_path / "ledger", webhook_url="https://alerts.example/hook",
                                volume=None, ack="", token="", survey=None,
                                release_retirement_summary_path=summary,
                                break_glass_notes_root=tmp_path / "absent-notes",
                                disk_usage=_usage(80.0), now=1000.0)
    attention = next(a for a in report["alerts"] if a["code"] == "release_retirement_attention")
    assert attention == {"code": "release_retirement_attention", "status": "blocked",
                         "alert_count": 1, "severity": "warn"}
    assert any(a["code"] == "release_retirement_attention" for a in cap.capacity_summary(report)["alerts"])


@pytest.mark.parametrize("bad_status", [
    pytest.param("/var/lib/blueprint/secrets/private-token", id="path"),
    pytest.param("secret-text-" * 2000, id="long"),
])
def test_release_retirement_attention_redacts_malformed_status(tmp_path: Path, bad_status: str) -> None:
    summary = tmp_path / "release-retention" / "latest-deploy-retirement.json"
    summary.parent.mkdir()
    summary.write_text(json.dumps({"status": bad_status, "alerts": ["failed"]}))

    report = cap.run_controller(
        mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
        reservation_root=tmp_path / "ledger", webhook_url="https://alerts.example/hook",
        volume=None, ack="", token="", survey=None,
        release_retirement_summary_path=summary,
        break_glass_notes_root=tmp_path / "absent-notes",
        disk_usage=_usage(80.0), now=1000.0,
        poster=lambda *_args: None,
    )

    attention = next(a for a in report["alerts"] if a["code"] == "release_retirement_attention")
    assert attention["status"] == "unreadable"
    assert bad_status not in json.dumps(cap.capacity_summary(report))


def test_unreported_break_glass_note_warns_without_disclosing_note(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import control_plane_break_glass

    monkeypatch.setattr(control_plane_break_glass, "unreported_notes",
                        lambda root: [{"name": "private-note.json", "reason": "sensitive detail"}])
    report = cap.run_controller(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
                                reservation_root=tmp_path / "ledger", webhook_url="https://alerts.example/hook",
                                volume=None, ack="", token="", survey=None,
                                release_retirement_summary_path=tmp_path / "absent-summary",
                                break_glass_notes_root=tmp_path / "notes",
                                disk_usage=_usage(80.0), now=1000.0)
    attention = next(a for a in report["alerts"] if a["code"] == "break_glass_notes_unreported")
    assert attention == {"code": "break_glass_notes_unreported", "count": 1, "severity": "warn"}
    assert "sensitive detail" not in json.dumps(cap.capacity_summary(report))


def test_unsafe_break_glass_notes_root_warns_without_aborting_tick(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.symlink_to(tmp_path / "missing-notes", target_is_directory=True)
    report = cap.run_controller(
        mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
        reservation_root=tmp_path / "ledger", webhook_url="https://alerts.example/hook",
        volume=None, ack="", token="", survey=None,
        release_retirement_summary_path=tmp_path / "absent-summary",
        break_glass_notes_root=notes, disk_usage=_usage(80.0), now=1000.0,
    )
    assert report["level"] == "warning"
    assert any(a["code"] == "break_glass_notes_unreadable" for a in report["alerts"])


@pytest.mark.parametrize(("outlook", "expected"), [
    ({"volume_growth": "planned", "reclaimable_bytes": 0}, (1600.0, "volume_growth")),
    ({"volume_growth": "applied", "reclaimable_bytes": 0}, (1600.0, "volume_growth")),
    ({"volume_growth": "blocked", "reclaimable_bytes": 50, "next_reclaim_epoch": 3600.0},
     (3600.0, "reclaim_scheduled")),
    ({"volume_growth": "blocked", "reclaimable_bytes": 5, "next_reclaim_epoch": 3600.0},
     (None, "operator_action_required")),
])
def test_capacity_eta_bases(outlook, expected) -> None:
    assert cap.capacity_eta(10, summary={"reclaim_outlook": outlook}, now=1000.0) == {
        "eta_epoch": expected[0], "eta_basis": expected[1]}


def test_capacity_eta_is_unknown_without_readable_summary() -> None:
    assert cap.capacity_eta(10, summary=None, now=1000.0) == {
        "eta_epoch": None, "eta_basis": "unknown"}
    assert cap.capacity_eta(10, summary={"reclaim_outlook": {
        "volume_growth": "not_configured", "reclaimable_bytes": None,
        "next_reclaim_epoch": None,
    }}, now=1000.0) == {"eta_epoch": None, "eta_basis": "unknown"}


@pytest.mark.parametrize("outlook", [
    {"volume_growth": []},
    {"volume_growth": {}},
    {"volume_growth": "blocked", "reclaimable_bytes": float("inf"),
     "next_reclaim_epoch": float("inf")},
    {"volume_growth": "blocked", "reclaimable_bytes": float("nan"),
     "next_reclaim_epoch": 3600.0},
    {"volume_growth": "blocked", "reclaimable_bytes": 10**400,
     "next_reclaim_epoch": 3600.0},
])
def test_capacity_eta_is_unknown_for_malformed_outlook(outlook) -> None:
    parsed = json.loads(json.dumps({"reclaim_outlook": outlook}))
    assert cap.capacity_eta(10, summary=parsed, now=1000.0) == {
        "eta_epoch": None, "eta_basis": "unknown"}


def test_critical_capacity_pages_when_gc_reclaims_nothing(tmp_path: Path) -> None:
    from blueprint_pipeline.control_plane_storage_gc_reasons import build_storage_gc_summary

    gc_root = tmp_path / "storage-gc"
    gc_root.mkdir()
    gc_summary = gc_root / "summary.json"
    gc_summary.write_text(json.dumps(build_storage_gc_summary({
        "observed_at_epoch": 900.0,
        "status": "applied",
        "opt_in": {"evidence_offload": True, "scene_workspace_retirement": False},
        "derived_directories": {"status": "applied", "candidate_bytes": 0,
                                "removed_bytes": 0,
                                "retained_by_reason": {"pinned": {"count": 1, "bytes": 3 * GIB}}},
        "content_store": {"status": "applied", "candidate_bytes": 0,
                          "removed_bytes": 0,
                          "retained_by_reason": {"young": {"count": 1, "bytes": 4 * GIB}}},
        "evidence_offload": {"status": "applied", "candidate_bytes": 0,
                             "offloaded_bytes": 0,
                             "retained_by_reason": {"protected_process": {"count": 1, "bytes": 5 * GIB}}},
        "scene_workspaces": {"status": "dry_run", "candidate_bytes": 10 * GIB,
                             "retained_by_reason": {"not_enabled": {"count": 1, "bytes": 2 * GIB}}},
    })), encoding="utf-8")
    common = dict(
        mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
        reservation_root=tmp_path / "ledger", webhook_url="https://alerts.example/hook",
        volume=None, ack="", token="", survey=None,
        release_retirement_summary_path=tmp_path / "absent-retirement",
        break_glass_notes_root=tmp_path / "notes", storage_gc_summary_path=gc_summary,
        poster=lambda *_args: None, disk_usage=_usage(20.0),
    )
    report = cap.run_controller(**common, now=1000.0)
    alert = next(row for row in report["alerts"] if row["code"] == "reclaim_ineffective")
    assert alert["severity"] == "page"
    assert alert["top_retained_reasons"] == [
        "protected_process", "young", "pinned",
    ]
    assert report["reclaim_outlook"]["reclaimable_bytes"] == 0
    assert cap.capacity_summary(report)["reclaim_outlook"] == report["reclaim_outlook"]

    stale = cap.run_controller(**common, now=10_000.0)
    assert not any(row["code"] == "reclaim_ineffective" for row in stale["alerts"])
    assert stale["reclaim_outlook"]["reclaimable_bytes"] is None


@pytest.mark.parametrize(("candidate", "reclaimed", "expected_ineffective"), [
    (2 * GIB, 0, False), (0, GIB, False), (0, 0, True),
])
def test_reclaim_ineffective_requires_zero_candidates_and_zero_reclaimed(
    candidate: int, reclaimed: int, expected_ineffective: bool,
) -> None:
    summary = {
        "schema_version": "control_plane_storage_gc_summary.v1",
        "observed_at_epoch": 1000.0, "status": "applied",
        "top_retained_reasons": [],
        "skipped_roots": [], "phase_errors": [],
        "opt_in": {"evidence_offload": False, "scene_workspace_retirement": False},
        "phases": {
            "derived_directories": {"status": "applied", "candidate_bytes": candidate,
                                    "removed_or_offloaded_bytes": reclaimed},
            "content_store": {"status": "applied", "candidate_bytes": 0,
                              "removed_or_offloaded_bytes": 0},
        },
    }
    outlook, _reasons, ineffective = cap._reclaim_outlook(
        summary, now=1100.0, volume_growth="blocked",
    )
    assert outlook["reclaimable_bytes"] == candidate
    assert ineffective is expected_ineffective


def test_reclaim_outlook_fails_closed_without_complete_applied_phase() -> None:
    summary = {
        "schema_version": "control_plane_storage_gc_summary.v1",
        "observed_at_epoch": 1000.0, "status": "applied",
        "skipped_roots": [], "phase_errors": [],
        "opt_in": {"evidence_offload": True},
        "phases": {
            "derived_directories": {"status": "applied", "candidate_bytes": 0,
                                    "removed_or_offloaded_bytes": 0},
            "content_store": {"status": "applied", "candidate_bytes": 0,
                              "removed_or_offloaded_bytes": 0},
            "evidence_offload": {"status": "error", "candidate_bytes": 0,
                                 "removed_or_offloaded_bytes": 0},
        },
    }
    outlook, reasons, ineffective = cap._reclaim_outlook(
        summary, now=1100.0, volume_growth="not_configured",
    )
    assert outlook["reclaimable_bytes"] is None
    assert reasons == [] and ineffective is False
    # An older or incomplete applied receipt with no candidate count remains
    # unknown; the current producer records the count after PR #2393.
    summary["opt_in"]["evidence_offload"] = False
    summary["phases"]["content_store"]["candidate_bytes"] = None
    outlook, reasons, ineffective = cap._reclaim_outlook(
        summary, now=1100.0, volume_growth="not_configured",
    )
    assert outlook["reclaimable_bytes"] is None
    assert reasons == [] and ineffective is False


def test_reclaim_outlook_does_not_claim_exact_top_reasons_from_legacy_capped_rows() -> None:
    summary = {
        "schema_version": "control_plane_storage_gc_summary.v1",
        "observed_at_epoch": 1000.0, "status": "applied",
        "skipped_roots": [], "phase_errors": [],
        "opt_in": {"evidence_offload": False, "scene_workspace_retirement": False},
        "phases": {
            "derived_directories": {"status": "applied", "candidate_bytes": 0,
                                    "removed_or_offloaded_bytes": 0},
            "content_store": {"status": "applied", "candidate_bytes": 0,
                              "removed_or_offloaded_bytes": 0},
        },
        "top_retained": [
            {"phase": "derived_directories", "reason": "shared", "bytes": 4 * GIB},
            {"phase": "content_store", "reason": "shared", "bytes": 4 * GIB},
            {"phase": "derived_directories", "reason": "single", "bytes": 5 * GIB},
            {"phase": "content_store", "reason": "third", "bytes": 3 * GIB},
        ],
    }

    _outlook, reasons, ineffective = cap._reclaim_outlook(
        summary, now=1100.0, volume_growth="blocked",
    )
    assert ineffective is False
    assert reasons == []


def test_reclaim_outlook_prefers_uncapped_global_reason_totals() -> None:
    summary = {
        "schema_version": "control_plane_storage_gc_summary.v1",
        "observed_at_epoch": 1000.0, "status": "applied",
        "skipped_roots": [], "phase_errors": [],
        "opt_in": {"evidence_offload": False, "scene_workspace_retirement": False},
        "phases": {
            "derived_directories": {"status": "applied", "candidate_bytes": 0,
                                    "removed_or_offloaded_bytes": 0},
            "content_store": {"status": "applied", "candidate_bytes": 0,
                              "removed_or_offloaded_bytes": 0},
        },
        "top_retained": [
            {"phase": "derived_directories", "reason": "single", "bytes": 5 * GIB},
            {"phase": "content_store", "reason": "third", "bytes": 3 * GIB},
        ],
        "top_retained_reasons": [
            {"reason": "shared", "bytes": 8 * GIB},
            {"reason": "single", "bytes": 5 * GIB},
            {"reason": "third", "bytes": 3 * GIB},
        ],
    }

    _outlook, reasons, ineffective = cap._reclaim_outlook(
        summary, now=1100.0, volume_growth="blocked",
    )
    assert ineffective is True
    assert reasons == ["shared", "single", "third"]


@pytest.mark.parametrize("incomplete", [
    {"skipped_roots": ["/var/lib/blueprint/task-evaluation-inputs"]},
    {"phase_errors": ["replay_caches"]},
])
def test_reclaim_outlook_does_not_claim_zero_when_gc_skipped_work(incomplete) -> None:
    from blueprint_pipeline.control_plane_storage_gc_reasons import build_storage_gc_summary

    summary = build_storage_gc_summary({
        "status": "applied", "observed_at_epoch": 1000.0,
        "opt_in": {"evidence_offload": False, "scene_workspace_retirement": False},
        "derived_directories": {"status": "applied", "candidate_bytes": 0,
                                "removed_bytes": 0},
        "content_store": {"status": "applied", "candidate_bytes": 0,
                          "removed_bytes": 0},
        **incomplete,
    })

    outlook, reasons, ineffective = cap._reclaim_outlook(
        summary, now=1100.0, volume_growth="blocked",
    )
    assert outlook["reclaimable_bytes"] is None
    assert reasons == [] and ineffective is False


def test_reclaim_outlook_counts_other_applying_gc_phases() -> None:
    summary = {
        "schema_version": "control_plane_storage_gc_summary.v1",
        "observed_at_epoch": 1000.0, "status": "applied",
        "skipped_roots": [], "phase_errors": [],
        "opt_in": {"evidence_offload": False, "scene_workspace_retirement": False,
                   "replay_cache_retention": False},
        "phases": {
            "derived_directories": {"status": "applied", "candidate_bytes": 0,
                                    "removed_or_offloaded_bytes": 0},
            "content_store": {"status": "applied", "candidate_bytes": 0,
                              "removed_or_offloaded_bytes": 0},
            "scratch_directories": {"status": "applied", "candidate_bytes": 2 * GIB,
                                    "removed_or_offloaded_bytes": 0},
            "workspace_bundles": {"status": "applied", "candidate_bytes": GIB,
                                  "removed_or_offloaded_bytes": 0},
        },
    }
    outlook, _reasons, ineffective = cap._reclaim_outlook(
        summary, now=1100.0, volume_growth="not_configured",
    )
    assert outlook["reclaimable_bytes"] == 3 * GIB
    assert outlook["sources"]["scratch_directories"] == 2 * GIB
    assert outlook["sources"]["workspace_bundles"] == GIB
    assert ineffective is False


def test_gc_summary_reader_accepts_producer_size_bound(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"schema_version": "control_plane_storage_gc_summary.v1",
                                "padding": "x" * (129 * 1024)}), encoding="utf-8")
    assert cap._read_attention_summary(path, max_bytes=256 * 1024)["schema_version"] == (
        "control_plane_storage_gc_summary.v1")
    assert cap._read_attention_summary(path) == {"status": "unreadable"}


def _residue_summary(*, evidence_enabled=True, residue_enabled=True, phase_enabled=True):
    return {
        "schema_version": "control_plane_storage_gc_summary.v1",
        "observed_at_epoch": 1000.0, "status": "applied",
        "skipped_roots": [], "phase_errors": [], "top_retained_reasons": [],
        "opt_in": {"evidence_offload": evidence_enabled,
                   "result_residue_offload": residue_enabled},
        "phases": {
            "derived_directories": {"status": "applied", "candidate_bytes": 0,
                                    "removed_or_offloaded_bytes": 0},
            "result_residue_offload": {"status": "applied", "enabled": phase_enabled,
                                       "candidate_bytes": 2 * GIB,
                                       "removed_or_offloaded_bytes": 0},
        },
    }


def test_residue_offload_candidates_prevent_false_ineffective_page_and_supply_eta() -> None:
    outlook, _reasons, ineffective = cap._reclaim_outlook(
        _residue_summary(), now=1100.0, volume_growth="not_configured",
    )
    assert outlook["sources"]["result_residue_offload"] == 2 * GIB
    assert outlook["reclaimable_bytes"] == 2 * GIB
    assert ineffective is False
    assert cap.capacity_eta(GIB, summary={"reclaim_outlook": outlook}, now=1100.0) == {
        "eta_epoch": 4600.0, "eta_basis": "reclaim_scheduled",
    }


@pytest.mark.parametrize(("evidence", "residue", "enabled"), [
    (False, True, True), (True, False, False), (False, False, False),
])
def test_residue_bytes_do_not_supply_eta_without_both_owner_opt_ins(evidence, residue, enabled) -> None:
    summary = _residue_summary(evidence_enabled=evidence, residue_enabled=residue,
                               phase_enabled=enabled)
    summary["phases"]["result_residue_offload"]["status"] = "dry_run"
    outlook, _reasons, ineffective = cap._reclaim_outlook(
        summary, now=1100.0, volume_growth="not_configured",
    )
    assert outlook["sources"]["result_residue_offload"] is None
    assert outlook["reclaimable_bytes"] == 0
    assert ineffective is True
    assert cap.capacity_eta(GIB, summary={"reclaim_outlook": outlook}, now=1100.0) == {
        "eta_epoch": None, "eta_basis": "operator_action_required",
    }


@pytest.mark.parametrize("incomplete", ["missing", "disabled", "unknown_opt_in", "unsized", "dry_run"])
def test_applying_residue_with_incomplete_evidence_cannot_prove_zero(incomplete) -> None:
    summary = _residue_summary()
    phase = summary["phases"]["result_residue_offload"]
    if incomplete == "missing":
        del summary["phases"]["result_residue_offload"]
    elif incomplete == "disabled":
        phase["enabled"] = False
    elif incomplete == "unknown_opt_in":
        summary["opt_in"]["result_residue_offload"] = None
    elif incomplete == "unsized":
        phase["candidate_bytes"] = None
    else:
        phase["status"] = "dry_run"
    outlook, reasons, ineffective = cap._reclaim_outlook(
        summary, now=1100.0, volume_growth="not_configured",
    )
    assert outlook["reclaimable_bytes"] is None
    assert reasons == [] and ineffective is False


def test_plan_only_replay_estimates_and_released_pin_counts_are_not_reclaim_bytes() -> None:
    summary = _residue_summary(residue_enabled=False, phase_enabled=False)
    summary["opt_in"]["replay_cache_retention"] = False
    summary["phases"]["replay_caches"] = {
        "status": "dry_run", "estimated_candidate_bytes": 8 * GIB,
        "candidate_bytes": None, "removed_or_offloaded_bytes": 0,
    }
    summary["phases"]["terminal_cache_pins"] = {
        "status": "applied", "enabled": True, "candidate_count": 142, "released_count": 142,
    }
    outlook, _reasons, ineffective = cap._reclaim_outlook(
        summary, now=1100.0, volume_growth="not_configured",
    )
    assert outlook["reclaimable_bytes"] == 0
    assert outlook["sources"]["replay_caches"] is None
    assert ineffective is True


def test_controller_writes_evidence_alerts_on_escalation_and_repeats_hourly_while_critical(
    tmp_path: Path,
) -> None:
    posted: list[tuple[str, str]] = []

    def poster(url: str, report) -> None:
        posted.append((url, report["level"]))

    common = dict(
        mounts=["/var/lib/blueprint"],
        report_root=tmp_path / "capacity",
        reservation_root=tmp_path / "reservations",
        webhook_url="https://alerts.example/hook",
        volume=None,
        ack="",
        token="",
        poster=poster,
        survey=None,
    )
    ok = cap.run_controller(**common, disk_usage=_usage(80.0), now=1_000.0)
    assert ok["level"] == "ok" and posted == []
    assert (tmp_path / "capacity" / "latest.json").is_file()

    critical = cap.run_controller(**common, disk_usage=_usage(9.0), now=2_000.0)
    assert critical["level"] == "critical" and critical["alert_posted"] is True
    assert posted == [("https://alerts.example/hook", "critical")]
    assert {a["code"] for a in critical["alerts"]} == {"admission_refused", "utilization_critical"}

    again = cap.run_controller(**common, disk_usage=_usage(9.0), now=2_600.0)
    assert again["alert_posted"] is False and len(posted) == 1
    later = cap.run_controller(**common, disk_usage=_usage(9.0), now=2_000.0 + 3_601)
    assert later["alert_posted"] is True and len(posted) == 2

    history = (tmp_path / "capacity" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 4
    latest = json.loads((tmp_path / "capacity" / "latest.json").read_text(encoding="utf-8"))
    assert latest["report_digest"] == later["report_digest"]


def test_volume_grows_one_step_only_when_critical_acknowledged_and_under_the_maximum(
    tmp_path: Path,
) -> None:
    report_ok = cap.build_capacity_report(
        mounts=["/mnt/work"], reservation_root=tmp_path / "r", disk_usage=_usage(60.0), now=1.0
    )
    assert cap.plan_volume_resize(report_ok, volume_id="vol-1", volume_mount="/mnt/work", current_size_gib=100, max_gib=500) is None
    report_full = cap.build_capacity_report(
        mounts=["/mnt/work"], reservation_root=tmp_path / "r", disk_usage=_usage(9.0), now=1.0
    )
    plan = cap.plan_volume_resize(report_full, volume_id="vol-1", volume_mount="/mnt/work", current_size_gib=100, max_gib=500)
    assert plan == {
        "status": "planned",
        "volume_id": "vol-1",
        "mount": "/mnt/work",
        "current_size_gib": 100,
        "target_size_gib": 150,
    }
    capped = cap.plan_volume_resize(report_full, volume_id="vol-1", volume_mount="/mnt/work", current_size_gib=500, max_gib=500)
    assert capped["status"] == "blocked" and capped["reason"] == "volume_at_maximum"

    calls: list = []

    def api(url, *, token, method, payload=None):
        calls.append((url, token, method, payload))
        return {"action": {"status": "in-progress"}}

    class Done:
        returncode = 0

    def runner(command, **_kwargs):
        calls.append(tuple(command))
        return Done()

    with pytest.raises(cap.ControlPlaneCapacityError, match="not_acknowledged"):
        cap.resize_volume(plan, ack="", token="t", device="/dev/sda", api=api, runner=runner)
    receipt = cap.resize_volume(plan, ack=cap.RESIZE_ACK, token="t", device="/dev/sda", api=api, runner=runner, now=5.0)
    assert receipt["status"] == "applied" and receipt["to_size_gib"] == 150
    assert receipt["provider_mutation_performed"] is True
    assert calls[0] == (
        "https://api.digitalocean.com/v2/volumes/vol-1/actions",
        "t",
        "POST",
        {"type": "resize", "size_gigabytes": 150},
    )
    assert calls[1] == ("resize2fs", "/dev/sda")

    def rejecting(url, *, token, method, payload=None):
        return {"action": {"status": "errored"}}

    with pytest.raises(cap.ControlPlaneCapacityError, match="resize_rejected"):
        cap.resize_volume(plan, ack=cap.RESIZE_ACK, token="t", device="/dev/sda", api=rejecting, runner=runner)


def test_controller_blocks_resize_without_acknowledgement_and_records_the_plan(tmp_path: Path) -> None:
    volume = {
        "id": "vol-1",
        "mount": "/var/lib/blueprint",
        "device": "/dev/sda",
        "current_size_gib": 100,
        "max_gib": 300,
        "step_gib": 50,
    }
    common = dict(
        mounts=["/var/lib/blueprint"],
        report_root=tmp_path / "capacity",
        reservation_root=tmp_path / "r",
        webhook_url="",
        volume=volume,
        poster=lambda *_args: None,
        survey=None,
    )
    blocked = cap.run_controller(**common, ack="", token="", disk_usage=_usage(9.0), now=1.0)
    assert blocked["volume_resize"]["status"] == "blocked"
    assert blocked["volume_resize"]["reason"] == "resize_not_acknowledged"

    resized: list = []

    def resizer(plan, **_kwargs):
        resized.append(plan)
        return {"status": "applied", "to_size_gib": plan["target_size_gib"]}

    applied = cap.run_controller(
        **common, ack=cap.RESIZE_ACK, token="tok", resizer=resizer, disk_usage=_usage(9.0), now=2.0
    )
    assert applied["volume_resize"] == {"status": "applied", "to_size_gib": 150}
    assert len(resized) == 1
    healthy = cap.run_controller(**common, ack=cap.RESIZE_ACK, token="tok", resizer=resizer, disk_usage=_usage(80.0), now=3.0)
    assert healthy["volume_resize"] == {"status": "not_needed"} and len(resized) == 1


def test_whole_chain_admission_rejects_space_that_fits_only_one_stage(tmp_path):
    result = cap.whole_chain_admission(tmp_path, reservation_root=tmp_path/"ledger", now=1000,
                                       disk_usage=_usage(12.0))
    assert result["status"] == "waiting_for_capacity"
    assert result["measurement"]["refused_roles"] == []
    assert result["required_workspace_bytes"] == 10 * GIB
    assert result["reservation_granted"] is False
    assert cap.whole_chain_admission(tmp_path, reservation_root=tmp_path/"ledger", now=1000,
                                     disk_usage=_usage(20.0))["status"] == "admitted"
    _reservation(tmp_path/"ledger", "another-run", expected_bytes=3*GIB, expires_at=2000)
    assert cap.whole_chain_admission(tmp_path, reservation_root=tmp_path/"ledger", now=1000,
                                     disk_usage=_usage(20.0))["status"] == "waiting_for_capacity"


def test_whole_chain_admission_groups_roles_by_device(tmp_path) -> None:
    system = tmp_path / "system"
    scratch = tmp_path / "scratch"
    system.mkdir()
    scratch.mkdir()

    def usage(path):
        free = 6 * GIB if Path(path) == scratch else 30 * GIB
        return Usage(100 * GIB, 100 * GIB - free, free)

    def device_of(path):
        return 202 if Path(path) == scratch else 101

    result = cap.whole_chain_admission(
        system, reservation_root=tmp_path / "ledger", now=1000,
        disk_usage=usage, device_of=device_of,
        role_targets={"launch_dispatch": scratch, "policy_canary_dispatch": scratch},
    )

    assert result["status"] == "waiting_for_capacity"
    devices = {row["device"]: row for row in result["devices"]}
    assert devices[101]["required_bytes"] == 6 * GIB and devices[101]["passed"] is True
    assert set(devices[101]["roles"]) == set(cap.CHAIN_ROLES) - {"launch_dispatch", "policy_canary_dispatch"}
    assert devices[202]["required_bytes"] == 4 * GIB and devices[202]["passed"] is False
    assert devices[202]["available_bytes"] == 0


def test_full_scratch_volume_refuses_chain_but_admits_deploy_and_listener_state(tmp_path) -> None:
    system = tmp_path / "system"
    scratch = tmp_path / "scratch"
    system.mkdir()
    scratch.mkdir()
    ledger = system / "reservations"

    def usage(path):
        free = 6 * GIB if Path(path) == scratch else 30 * GIB
        return Usage(100 * GIB, 100 * GIB - free, free)

    def device_of(path):
        return 202 if Path(path) == scratch else 101

    gate = cap.whole_chain_admission(
        system, role_targets={"launch_dispatch": scratch}, device_of=device_of,
        reservation_root=ledger, disk_usage=usage, now=1000,
    )
    assert gate["status"] == "waiting_for_capacity"
    assert next(row for row in gate["devices"] if row["device"] == 202)["passed"] is False
    deploy = disk_budget.reserve_control_plane_disk(
        "control_plane_deploy", target_root=system, reservation_root=ledger,
        expected_bytes=450 * MIB, disk_usage=usage, device_of=device_of,
    )
    assert deploy.floor_bytes == GIB
    with pytest.raises(disk_budget.ControlPlaneDiskBudgetError, match="control_plane_disk_budget_exceeded"):
        disk_budget.reserve_control_plane_disk(
            "handoff_staging", target_root=scratch, reservation_root=ledger,
            expected_bytes=64 * MIB, disk_usage=usage, device_of=device_of,
        )
    # The scratch disk is below its bulk floor, not literally at zero free.
    # The protected bytes on that same volume still hold the listener's ledger.
    listener_ledger = scratch / "pipeline_job_ledger.json"
    listener_ledger.write_text('{"status":"retryable_blocked"}\n', encoding="utf-8")
    assert listener_ledger.is_file()
    deploy.release()


def test_whole_chain_admission_fails_closed_for_absent_targets(tmp_path) -> None:
    system = tmp_path / "system"
    system.mkdir()
    missing = tmp_path / "unmounted-scratch"

    def roomy(_path):
        return Usage(100 * GIB, 70 * GIB, 30 * GIB)

    for mount, role_targets in (
        (missing, {}),
        (system, {"launch_dispatch": missing}),
    ):
        result = cap.whole_chain_admission(
            mount, reservation_root=tmp_path / "ledger", now=1000,
            disk_usage=roomy, role_targets=role_targets,
            device_of=lambda _path: 101,
        )
        assert result["status"] == "waiting_for_capacity"


def test_absent_configured_mount_is_not_critical(tmp_path) -> None:
    missing = tmp_path / "not-mounted"

    def unavailable(_path):
        raise FileNotFoundError(2, "not mounted")

    report = cap.build_capacity_report(
        mounts=[missing], reservation_root=tmp_path / "ledger", disk_usage=unavailable, now=1000,
    )
    assert report["mounts"][0]["status"] == "absent"
    assert report["level"] != "critical"


def test_measure_mount_reports_critical_admission_band(tmp_path) -> None:
    row = cap.measure_mount(
        tmp_path, reservation_root=tmp_path / "ledger", now=1000,
        disk_usage=lambda _path: Usage(100 * GIB, 99 * GIB, GIB),
    )
    assert row["critical_floor_bytes"] == GIB
    assert row["critical_available_bytes"] == 0
    assert "control_plane_deploy" in row["critical_roles_refused"]
    report = cap.build_capacity_report(
        mounts=[tmp_path], reservation_root=tmp_path / "ledger", now=1000,
        disk_usage=lambda _path: Usage(100 * GIB, 99 * GIB, GIB),
    )
    assert "critical_admission_refused" in {alert["code"] for alert in report["alerts"]}


def test_measured_p95_admits_a_chain_the_constants_refuse(tmp_path):
    ledger = tmp_path / "ledger"
    for role in cap.CHAIN_ROLES:
        for _ in range(10):
            disk_budget.record_footprint_sample(reservation_root=ledger, role=role,
                observed_bytes=400 * MIB, reserved_bytes=2 * GIB, now=lambda: 1.0)
    usage = _usage(free_gib=8.0 + 5.0)            # 5 GiB above the 8 GiB floor
    admitted = cap.whole_chain_admission(tmp_path, reservation_root=ledger, now=1.0, disk_usage=usage)
    assert admitted["status"] == "admitted"
    assert admitted["required_workspace_bytes"] == 5 * 500 * MIB
    assert admitted["required_workspace_basis"] == "measured_p95"
    empty = tmp_path / "empty-ledger"
    refused = cap.whole_chain_admission(tmp_path, reservation_root=empty, now=1.0, disk_usage=usage)
    assert refused["status"] == "waiting_for_capacity"
    assert refused["required_workspace_bytes"] == 10 * GIB
    assert refused["required_workspace_basis"] == "declared_default"


def test_measure_mount_honors_the_floor_override_and_ignores_other_devices(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES", str(4 * GIB))
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "other.json").write_text(json.dumps({"device": -5, "pid": os.getpid(), "expected_bytes": 50 * GIB,
                                                   "expires_at_epoch": 1e12}))
    row = cap.measure_mount(tmp_path, reservation_root=ledger, disk_usage=_usage(free_gib=10.0), now=1.0)
    assert row["floor_bytes"] == max(4 * GIB, int(154 * GIB * 0.05))
    assert row["reserved_bytes"] == 0


def test_invalid_budget_configuration_waits_instead_of_crashing_the_gate(tmp_path, monkeypatch):
    # The ledger refuses every reservation under a malformed override; the
    # controller reports it and the chain gate waits rather than raising out of
    # scene progression.
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_LAUNCH_ACTIVATION_BYTES", "not-a-number")
    row = cap.measure_mount(tmp_path, reservation_root=tmp_path / "ledger", disk_usage=_usage(80.0), now=1.0)
    assert row["status"] == "configuration_invalid"
    assert row["blocker"].startswith("control_plane_disk_budget_configuration_invalid:")
    gate = cap.whole_chain_admission(tmp_path, reservation_root=tmp_path / "ledger", now=1.0,
                                     disk_usage=_usage(80.0))
    assert gate["status"] == "waiting_for_capacity"
    assert gate["required_workspace_basis"] == "declared_default"
    report = cap.build_capacity_report(mounts=[tmp_path], reservation_root=tmp_path / "ledger",
                                       disk_usage=_usage(80.0), now=1.0)
    assert report["level"] == "critical"
    assert report["alerts"] == [{"mount": str(tmp_path), "code": "mount_configuration_invalid", "severity": "warn"}]


def _survey_result(**overrides):
    result = {"schema_version": "control_plane_disk_usage_survey.v1", "status": "complete",
              "observed_at_epoch": 1_000.0, "mounts": [], "by_class": [], "top_roots": [],
              "top_owners": [], "unclassified_roots": []}
    result.update(overrides)
    return result


def _no_project_spend(monkeypatch, value=None):
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_scene_spend.refresh_configured_scene_project_spend",
        lambda: value,
    )


def test_summary_is_door_readable_and_secret_free(tmp_path, monkeypatch):
    # Project spend is in the root-only report and never in the door-readable summary.
    _no_project_spend(monkeypatch, {"spend_usd": 12.5})
    previous = os.umask(0o077)  # the capacity unit's UMask
    try:
        report = cap.run_controller(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
            reservation_root=tmp_path / "ledger", webhook_url="", volume=None, ack="", token="",
            disk_usage=_usage(free_gib=100.0), now=1_000.0,
            survey=lambda **_k: {"schema_version": "control_plane_disk_usage_survey.v1", "status": "complete",
                                 "observed_at_epoch": 1_000.0, "mounts": [], "by_class": [], "top_roots": [],
                                 "top_owners": [], "unclassified_roots": [{"root": "/var/lib/blueprint/x",
                                                                           "allocated_bytes": 2 * GIB}]})
    finally:
        os.umask(previous)
    summary_path = tmp_path / "capacity" / "summary.json"
    assert oct(summary_path.stat().st_mode & 0o777) == "0o644"
    assert oct((tmp_path / "capacity").stat().st_mode & 0o777) == "0o755"
    assert oct((tmp_path / "capacity" / "usage-latest.json").stat().st_mode & 0o777) == "0o644"
    summary = json.loads(summary_path.read_text())
    assert summary["schema_version"] == "control_plane_capacity_summary.v1"
    assert "project_spend" not in summary and "provider_funding" not in summary
    assert report["project_spend"] == {"spend_usd": 12.5}
    assert {"mount": None, "code": "usage_unclassified_root", "root": "/var/lib/blueprint/x"} in [
        {k: a.get(k) for k in ("mount", "code", "root")} for a in report["alerts"]]
    assert report["level"] == summary["level"] == "warning"
    assert summary["usage"]["unclassified_roots"] == [{"root": "/var/lib/blueprint/x", "allocated_bytes": 2 * GIB}]
    assert summary["usage"]["age_seconds"] == 0.0
    assert set(summary["mounts"][0]) <= {"mount", "status", "total_bytes", "free_bytes", "used_fraction",
                                         "floor_bytes", "reserved_bytes", "available_bytes", "refused_roles",
                                         "forecast", "level"}
    assert summary["report_digest"] == report["report_digest"]
    latest = json.loads((tmp_path / "capacity" / "latest.json").read_text())
    assert latest["usage"] == summary["usage"]


def test_survey_runs_at_most_hourly(tmp_path, monkeypatch):
    _no_project_spend(monkeypatch)
    calls = []
    def survey(**kwargs):
        calls.append(kwargs)
        return {"schema_version": "control_plane_disk_usage_survey.v1", "status": "complete",
                "observed_at_epoch": 1_000.0, "mounts": [], "by_class": [], "top_roots": [],
                "top_owners": [], "unclassified_roots": []}
    for now in (1_000.0, 1_600.0, 4_700.0):
        cap.run_controller(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
            reservation_root=tmp_path / "ledger", webhook_url="", volume=None, ack="", token="",
            disk_usage=_usage(free_gib=100.0), now=now, survey=survey)
    assert len(calls) == 2
    assert calls[0] == {"mounts": [str(tmp_path), "/"]}
    forced = cap.run_controller(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
        reservation_root=tmp_path / "ledger", webhook_url="", volume=None, ack="", token="",
        disk_usage=_usage(free_gib=100.0), now=4_800.0, survey=survey, force_survey=True)
    assert len(calls) == 3 and forced["usage"]["age_seconds"] == 3_800.0


def test_survey_mounts_include_the_attached_work_volume_only(tmp_path, monkeypatch):
    service = Path("deploy/systemd/blueprint-control-plane-capacity.service").read_text()
    assert "Environment=BLUEPRINT_CAPACITY_MOUNTS=/:/var/lib/blueprint:/mnt/blueprint-work" in service
    monkeypatch.setattr(cap.os.path, "ismount", lambda path: path == "/mnt/blueprint-work")
    assert cap.survey_mounts(["/var/lib/blueprint"]) == [
        "/var/lib/blueprint", "/", "/mnt/blueprint-work"]
    monkeypatch.setattr(cap.os.path, "ismount", lambda _path: False)
    assert cap.survey_mounts(["/var/lib/blueprint"]) == ["/var/lib/blueprint", "/"]


def test_direct_controller_call_surveys_by_default(tmp_path, monkeypatch):
    _no_project_spend(monkeypatch)
    calls = []

    def survey(**kwargs):
        calls.append(kwargs)
        return _survey_result()

    monkeypatch.setattr(cap, "survey_usage", survey)
    report = cap.run_controller(
        mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
        reservation_root=tmp_path / "ledger", webhook_url="", volume=None,
        ack="", token="", disk_usage=_usage(free_gib=100.0), now=1_000.0,
    )
    assert calls == [{"mounts": [str(tmp_path), "/"]}]
    assert report["usage"]["status"] == "complete"


@pytest.mark.parametrize("suspicious", [
    "sk-" + "A" * 30,
    "asset?key=" + "A" * 20,
    "asset?signature=" + "a" * 40,
    "asset?auth=" + "A" * 20,
    "https://discord.com/api/webhooks/123/" + "A" * 20,
])
def test_credential_shaped_survey_names_are_redacted_before_publication(tmp_path, monkeypatch, suspicious):
    _no_project_spend(monkeypatch)
    report = cap.run_controller(
        mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
        reservation_root=tmp_path / "ledger", webhook_url="", volume=None,
        ack="", token="", disk_usage=_usage(free_gib=100.0), now=1_000.0,
        survey=lambda **_kwargs: _survey_result(unclassified_roots=[
            {"root": f"/var/lib/blueprint/{suspicious}", "allocated_bytes": 2 * GIB}]),
    )
    summary = (tmp_path / "capacity" / "summary.json").read_text()
    usage = (tmp_path / "capacity" / "usage-latest.json").read_text()
    assert suspicious not in summary + usage
    assert scan_bytes(summary.encode()) is None
    assert scan_bytes(usage.encode()) is None
    assert _small_json(tmp_path / "capacity" / "summary.json")["usage"]["status"] == "complete"
    assert report["usage"]["unclassified_roots"][0]["allocated_bytes"] == 2 * GIB


def test_a_failed_survey_keeps_the_last_one_and_never_stops_the_tick(tmp_path, monkeypatch):
    _no_project_spend(monkeypatch)
    common = dict(mounts=[str(tmp_path)], report_root=tmp_path / "capacity", reservation_root=tmp_path / "ledger",
                  webhook_url="", volume=None, ack="", token="", disk_usage=_usage(free_gib=100.0))
    cap.run_controller(**common, now=1_000.0, survey=lambda **_k: _survey_result(
        top_owners=[{"owner": "scene:s", "root": "/r", "storage_class": "work", "allocated_bytes": 1}]))

    def broken(**_kwargs):
        raise RuntimeError("walk failed")

    report = cap.run_controller(**common, now=9_000.0, survey=broken)
    assert report["usage"]["error"] == "usage_survey_failed:RuntimeError"
    assert report["usage"]["top_owners"][0]["owner"] == "scene:s"
    assert report["usage"]["age_seconds"] == 8_000.0
    assert json.loads((tmp_path / "capacity" / "summary.json").read_text())["usage"]["error"] == (
        "usage_survey_failed:RuntimeError")
    unsurveyed = cap.run_controller(**{**common, "report_root": tmp_path / "fresh"}, now=1_000.0, survey=broken)
    assert unsurveyed["usage"] == {"status": "unavailable", "error": "usage_survey_failed:RuntimeError"}
    assert unsurveyed["level"] == "warning"
    assert any(a["code"] == "operator_alert_route_unconfigured" for a in unsurveyed["alerts"])


def test_failed_survey_attempt_is_throttled_across_ticks(tmp_path, monkeypatch):
    _no_project_spend(monkeypatch)
    calls = []

    def broken(**_kwargs):
        calls.append(1)
        raise RuntimeError("walk failed")

    common = dict(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
                  reservation_root=tmp_path / "ledger", webhook_url="", volume=None,
                  ack="", token="", disk_usage=_usage(free_gib=100.0), survey=broken)
    first = cap.run_controller(**common, now=1_000.0)
    assert first["usage"]["error"] == "usage_survey_failed:RuntimeError"
    marker = json.loads((tmp_path / "capacity" / cap.USAGE_ATTEMPT_FILENAME).read_text())
    assert marker["attempted_at_epoch"] == 1_000.0 and marker["status"] == "failed"

    second = cap.run_controller(**common, now=1_600.0)
    assert calls == [1]
    assert second["usage"]["error"] == first["usage"]["error"]
    cap.run_controller(**common, now=4_700.0)
    assert calls == [1, 1]


def test_interrupted_survey_attempt_is_throttled_after_restart(tmp_path, monkeypatch):
    _no_project_spend(monkeypatch)
    report_root = tmp_path / "capacity"
    report_root.mkdir()
    (report_root / cap.USAGE_ATTEMPT_FILENAME).write_text(json.dumps({
        "schema_version": cap.USAGE_ATTEMPT_SCHEMA_VERSION,
        "attempted_at_epoch": 1_000.0,
        "status": "running",
    }))
    calls = []

    def survey(**_kwargs):
        calls.append(1)
        return _survey_result(observed_at_epoch=4_700.0)

    common = dict(mounts=[str(tmp_path)], report_root=report_root,
                  reservation_root=tmp_path / "ledger", webhook_url="", volume=None,
                  ack="", token="", disk_usage=_usage(free_gib=100.0), survey=survey)
    cap.run_controller(**common, now=1_600.0)
    assert calls == []
    cap.run_controller(**common, now=4_700.0)
    assert calls == [1]


def test_new_capacity_warning_pages_while_usage_warning_persists(tmp_path, monkeypatch):
    _no_project_spend(monkeypatch)
    posted = []
    common = dict(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
                  reservation_root=tmp_path / "ledger", webhook_url="https://alerts.example/hook",
                  volume=None, ack="", token="", poster=lambda _url, report: posted.append(report),
                  survey=lambda **_kwargs: _survey_result(unclassified_roots=[
                      {"root": "/var/lib/blueprint/unknown", "allocated_bytes": 2 * GIB}]))
    usage_only = cap.run_controller(**common, now=1_000.0, disk_usage=_usage(free_gib=80.0))
    assert usage_only["level"] == "warning" and usage_only["alert_posted"] is True
    capacity_warning = cap.run_controller(**common, now=1_600.0, disk_usage=_usage(free_gib=40.0))
    assert capacity_warning["level"] == "warning"
    assert {row["code"] for row in capacity_warning["alerts"]} >= {
        "usage_unclassified_root", "utilization_warning"}
    assert capacity_warning["alert_posted"] is True and len(posted) == 2


def test_a_second_mount_warning_pages_with_the_same_alert_code():
    first = {"level": "warning", "alert_posted": True, "last_alert_epoch": 1_000.0,
             "alerts": [{"code": "utilization_warning", "mount": "/first"}]}
    second = {"level": "warning", "alerts": [
        {"code": "utilization_warning", "mount": "/first"},
        {"code": "utilization_warning", "mount": "/second"}]}
    assert cap.alert_due(first, second, now=1_600.0)


def test_failed_warning_webhook_retries_on_the_next_tick(tmp_path):
    attempts = []

    def poster(_url, _report):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("webhook temporarily unavailable")

    common = dict(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
                  reservation_root=tmp_path / "ledger", webhook_url="https://alerts.example/hook",
                  volume=None, ack="", token="", poster=poster, survey=None,
                  disk_usage=_usage(free_gib=40.0))
    first = cap.run_controller(**common, now=1_000.0)
    assert first["level"] == "warning" and first["alert_posted"] is False
    second = cap.run_controller(**common, now=1_600.0)
    assert second["alert_posted"] is True and len(attempts) == 2


def test_unchanged_warning_does_not_page_again_after_a_quiet_tick(tmp_path):
    posted = []
    common = dict(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
                  reservation_root=tmp_path / "ledger", webhook_url="https://alerts.example/hook",
                  volume=None, ack="", token="", poster=lambda _url, report: posted.append(report),
                  survey=None, disk_usage=_usage(free_gib=40.0))

    first = cap.run_controller(**common, now=1_000.0)
    second = cap.run_controller(**common, now=1_600.0)
    third = cap.run_controller(**common, now=2_200.0)

    assert first["alert_posted"] is True
    assert second["alert_posted"] is False
    assert third["alert_posted"] is False
    assert len(posted) == 1


def test_low_attribution_warns_but_never_masks_critical(tmp_path, monkeypatch):
    _no_project_spend(monkeypatch)
    survey = lambda **_k: _survey_result(mounts=[  # noqa: E731
        {"mount": "/", "used_bytes": 100, "surveyed_bytes": 50, "classified_bytes": 40, "attributed_fraction": 0.5}])
    common = dict(mounts=[str(tmp_path)], reservation_root=tmp_path / "ledger", webhook_url="", volume=None,
                  ack="", token="", survey=survey, now=1_000.0)
    warned = cap.run_controller(**common, report_root=tmp_path / "a", disk_usage=_usage(free_gib=100.0))
    assert {"mount": "/", "code": "usage_attribution_low", "attributed_fraction": 0.5,
            "severity": "warn"} in warned["alerts"]
    assert warned["level"] == "warning"
    critical = cap.run_controller(**common, report_root=tmp_path / "b", disk_usage=_usage(free_gib=9.0))
    assert critical["level"] == "critical"


def test_summary_stays_under_128_kib(tmp_path):
    report = {"schema_version": cap.SCHEMA_VERSION, "observed_at_epoch": 1.0, "level": "warning",
              "report_digest": "sha256:" + "0" * 64, "mounts": [],
              "alerts": [{"code": "usage_unclassified_root", "root": f"/var/lib/blueprint/{index:05d}" + "x" * 200,
                          "allocated_bytes": 2 * GIB} for index in range(2_000)],
              "usage": {"status": "complete", "top_owners": [], "top_roots": [], "unclassified_roots": []}}
    cap.write_report(tmp_path / "capacity", report)
    text = (tmp_path / "capacity" / "summary.json").read_text()
    summary = json.loads(text)
    assert len(text.encode()) <= 128 * 1024
    assert summary["truncated"] is True and 0 < len(summary["alerts"]) < 2_000
    assert summary["alerts"][0]["root"].startswith("/var/lib/blueprint/00000")


def test_cli_survey_flag_forces_a_survey(tmp_path, monkeypatch):
    seen = {}

    def fake_run_controller(**kwargs):
        seen.update(kwargs)
        return {"level": "ok", "alerts": [], "report_digest": "sha256:0"}

    monkeypatch.setattr(cap, "run_controller", fake_run_controller)
    monkeypatch.setenv("BLUEPRINT_CAPACITY_SURVEY_INTERVAL_SECONDS", "900")
    monkeypatch.delenv("BLUEPRINT_CAPACITY_VOLUME_ID", raising=False)
    assert cap.main(["--survey", "--mount", str(tmp_path), "--report-root", str(tmp_path / "capacity")]) == 0
    assert seen["force_survey"] is True and seen["survey_interval_seconds"] == 900
    assert seen["survey"] is cap.survey_usage
    assert cap.main(["--mount", str(tmp_path), "--report-root", str(tmp_path / "capacity")]) == 0
    assert seen["force_survey"] is False


def test_a_report_directory_owned_by_the_service_account_keeps_its_reports(tmp_path, monkeypatch):
    # Deploy provisions a missing sandbox directory as the service account, and the
    # unit holds no CAP_FOWNER, so root cannot chmod it; the door owns it and reads
    # the summary anyway. The tick must not fail over it.
    real_chmod = os.chmod

    def chmod(path, mode, *args, **kwargs):
        if Path(path) == tmp_path / "capacity":
            raise PermissionError("not the owner")
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(cap.os, "chmod", chmod)
    cap.write_report(tmp_path / "capacity", {"level": "ok", "mounts": [], "alerts": []})
    summary = tmp_path / "capacity" / "summary.json"
    assert json.loads(summary.read_text())["level"] == "ok"
    assert oct(summary.stat().st_mode & 0o777) == "0o644"

@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-0000 ledger anyway")
def test_an_unreadable_ledger_is_an_unreadable_mount_not_an_empty_one(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    ledger.chmod(0)
    try:
        row = cap.measure_mount(tmp_path, reservation_root=ledger, disk_usage=_usage(80.0), now=1.0)
        gate = cap.whole_chain_admission(tmp_path, reservation_root=ledger, now=1.0, disk_usage=_usage(80.0))
    finally:
        ledger.chmod(0o770)
    assert row["status"] == "unreadable"
    assert row["blocker"] == "control_plane_disk_budget_ledger_unreadable"
    assert gate["status"] == "waiting_for_capacity"


def test_capacity_history_rows_leave_the_footprints_to_the_latest_report(tmp_path):
    common = dict(mounts=[str(tmp_path)], report_root=tmp_path / "capacity", reservation_root=tmp_path / "r",
                  webhook_url="", volume=None, ack="", token="", poster=lambda *_args: None, survey=None)
    cap.run_controller(**common, disk_usage=_usage(80.0), now=1.0)
    latest = json.loads((tmp_path / "capacity" / "latest.json").read_text(encoding="utf-8"))
    [row] = [json.loads(line) for line in
             (tmp_path / "capacity" / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    # The history is never pruned; it keeps the measurement, not a per-role map per tick.
    assert "footprints" in latest["mounts"][0] and "footprints" not in row
    assert row["free_bytes"] == latest["mounts"][0]["free_bytes"]
