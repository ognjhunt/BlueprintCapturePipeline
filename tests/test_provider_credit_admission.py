"""ADP-009D: funding checks cannot allocate, leak account data, or admit unknown credit."""

import json

import pytest

from blueprint_pipeline import provider_credit_admission as credit
from blueprint_pipeline import control_plane_capacity_controller as capacity


def observation(amount=20.0):
    return credit.observe_vast_credit(api_key="test-secret", now=100,
                                     request=lambda **kw: (200, {"credit": amount, "email": "private"}))


@pytest.mark.parametrize("amount", [None, True, "20", float("nan"), float("inf")])
def test_unknown_or_malformed_credit_cannot_admit(amount):
    row = observation(amount)
    assert row["status"] == "unknown"
    assert credit.credit_admission(row, required_usd=2, now=100)["status"] == "blocked"
    assert "private" not in json.dumps(row)


def test_exact_funding_bound_and_freshness():
    row = observation(3)
    assert credit.credit_admission(row, required_usd=2, now=100)["status"] == "admitted"
    assert credit.credit_admission(row, required_usd=2.01, now=100)["blockers"] == [
        "provider_credit_insufficient"]
    for now in [99, 161]:
        assert "provider_credit_observation_stale" in credit.credit_admission(
            row, required_usd=2, now=now)["blockers"]
    assert credit.credit_admission({**row, "credit_usd": 100}, required_usd=2,
                                   now=100)["status"] == "blocked"


def test_transport_diagnostics_never_include_secret():
    def failed(**kwargs):
        raise RuntimeError("test-secret private account")
    row = credit.observe_vast_credit(api_key="test-secret", request=failed)
    assert row["blockers"] == ["provider_credit_transport_failed"]
    assert "test-secret" not in json.dumps(row)


def test_production_guard_is_fail_closed_on_bad_config(monkeypatch):
    monkeypatch.setenv(credit.ENABLED_ENV, "typo")
    assert credit.configured_vast_credit_admission(api_key="k", required_usd=2)["status"] == "blocked"
    monkeypatch.setenv(credit.ENABLED_ENV, "true")
    monkeypatch.setenv(credit.RESERVE_ENV, "NaN")
    row = observation()
    monkeypatch.setattr(credit, "observe_vast_credit", lambda **kw: row)
    result = credit.configured_vast_credit_admission(api_key="k", required_usd=2)
    assert "provider_credit_requirement_invalid" in result["blockers"]


def test_capacity_reports_low_credit_without_disk_pressure(tmp_path):
    from collections import namedtuple
    usage = namedtuple("Usage", "total used free")(100 * 1024**3, 10 * 1024**3, 90 * 1024**3)
    alerts = []
    result = capacity.run_controller(
        mounts=["/test"], report_root=tmp_path / "report", reservation_root=tmp_path / "ledger",
        webhook_url="https://test.invalid", volume=None, ack="", token="",
        disk_usage=lambda _: usage, poster=lambda url, report: alerts.append(report["alerts"]),
        credit_collector=lambda: observation(2), now=100, survey=None,
    )
    assert result["level"] == "critical"
    assert result["provider_funding"]["blockers"] == ["provider_credit_insufficient"]
    assert alerts and result["alert_posted"]



def test_run_controller_does_not_flag_a_just_taken_credit_observation_stale(tmp_path):
    """The credit GET happens after the pass-start clock (disk measurement + HTTP run first),
    so the observation's epoch is later than run_controller's ``now``.  Freshness must be judged
    against a clock no earlier than the observation, or every live reading is falsely stale."""
    from collections import namedtuple

    usage = namedtuple("Usage", "total used free")(100 * 1024**3, 10 * 1024**3, 90 * 1024**3)
    # Pass-start now=100; the observation is taken at 105 (as the real GET is, after the pass began).
    later = credit.observe_vast_credit(api_key="k", now=105, request=lambda **kw: (200, {"credit": 20.0}))
    result = capacity.run_controller(
        mounts=["/test"], report_root=tmp_path / "report", reservation_root=tmp_path / "ledger",
        webhook_url="", volume=None, ack="", token="", disk_usage=lambda _: usage,
        credit_collector=lambda: later, now=100, survey=None,
    )
    assert "provider_credit_observation_stale" not in result["provider_funding"]["blockers"]
    assert result["provider_funding"]["status"] == "admitted"

    # A genuinely old observation is still caught: epoch far before the pass clock.
    old = credit.observe_vast_credit(api_key="k", now=10, request=lambda **kw: (200, {"credit": 20.0}))
    stale = capacity.run_controller(
        mounts=["/test"], report_root=tmp_path / "report2", reservation_root=tmp_path / "ledger2",
        webhook_url="", volume=None, ack="", token="", disk_usage=lambda _: usage,
        credit_collector=lambda: old, now=1000, survey=None,
    )
    assert stale["provider_funding"]["blockers"] == ["provider_credit_observation_stale"]


def test_allocation_guard_is_wired_before_offer_search():
    # The adapter's existing blocked-inventory path releases its lock and emits
    # no-allocation teardown evidence; funding uses that same path.
    from pathlib import Path
    from blueprint_pipeline import vast_provider_adapter
    source = Path(vast_provider_adapter.__file__).read_text()
    assert "prelaunch_inventory_blockers.extend(credit_guard[\"blockers\"])" in source
    assert source.index("credit_guard = record_vast_credit_admission") < source.index(
        "if prelaunch_inventory_blockers:")


def test_low_credit_stops_real_render_launcher_before_create(tmp_path, monkeypatch):
    from blueprint_pipeline.gpu_render_providers import VastRenderProvider
    from blueprint_pipeline import vast_api_transport as adapter
    from blueprint_pipeline.paid_resource_admission import (
        build_paid_lane_admission, require_paid_resource_admission, PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    grant = require_paid_resource_admission(build_paid_lane_admission(resource_class="gpu_render"),
        resource_class="gpu_render", expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    monkeypatch.setenv(credit.ENABLED_ENV, "true")
    monkeypatch.setattr(VastRenderProvider, "_key", lambda self: "test-key")
    calls = []
    def api(**kwargs):
        calls.append((kwargs["method"], kwargs["path"]))
        assert kwargs["method"] == "GET" and kwargs["path"] == "/users/current/"
        return 200, {"credit": 0.5}
    monkeypatch.setattr(adapter, "_api_json", api)
    result = VastRenderProvider().launch(tmp_path, {"prelaunch_spend_guard": {
        "required_before_provider_launch": True, "can_launch": True, "max_spend_usd": 2,
    }}, paid_resource_admission_grant=grant)
    assert result["blockers"] == ["provider_credit_insufficient"]
    assert result["allocation_created"] is False
    assert calls == [("GET", "/users/current/")]


def _controller(tmp_path, amount, **kwargs):
    from collections import namedtuple
    usage = namedtuple("Usage", "total used free")(100 * 1024**3, 10 * 1024**3, 90 * 1024**3)
    return capacity.run_controller(
        mounts=["/test"], report_root=tmp_path / "report", reservation_root=tmp_path / "ledger",
        webhook_url="", volume=None, ack="", token="", disk_usage=lambda _: usage,
        credit_collector=lambda: observation(amount), now=100, survey=None, **kwargs,
    )


def test_low_vast_credit_pages_with_its_balance(tmp_path):
    # Vast stops every instance when the balance runs out (capture-447 lost an
    # attempt that way); a warn-level alert was posted once and never repeated.
    result = _controller(tmp_path, 8)
    alert = next(a for a in result["alerts"] if a["code"] == "provider_credit_insufficient")
    assert alert["severity"] == "page"
    assert (alert["credit_usd"], alert["required_usd"]) == (8, credit.PAGE_CREDIT_USD + 1)
    summary_alert = next(a for a in capacity.capacity_summary(result)["alerts"]
                         if a["code"] == "provider_credit_insufficient")
    assert (summary_alert["credit_usd"], summary_alert["severity"]) == (8, "page")


def test_default_page_line_is_two_launch_allowances_plus_reserve(tmp_path):
    assert credit.PAGE_CREDIT_USD == 10
    assert _controller(tmp_path, 10.99)["provider_funding"]["blockers"] == ["provider_credit_insufficient"]
    assert _controller(tmp_path, 11)["provider_funding"]["status"] == "admitted"
    assert _controller(tmp_path, 11, credit_warning_usd=20)["provider_funding"]["status"] == "blocked"


GIB = 1024**3


def _gc_found_nothing(tmp_path, now):
    """A fresh, applied storage-GC summary with nothing to free: with a critical level it
    is exactly what pages ``reclaim_ineffective``."""
    from blueprint_pipeline.control_plane_storage_gc_reasons import build_storage_gc_summary

    path = tmp_path / "storage-gc" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_storage_gc_summary({
        "observed_at_epoch": now - 60, "status": "applied",
        "opt_in": {"evidence_offload": True, "scene_workspace_retirement": False},
        "derived_directories": {"status": "applied", "candidate_bytes": 0, "removed_bytes": 0,
                                "retained_by_reason": {"pinned": {"count": 1, "bytes": 3 * GIB}}},
        "content_store": {"status": "applied", "candidate_bytes": 0, "removed_bytes": 0,
                          "retained_by_reason": {"young": {"count": 1, "bytes": 4 * GIB}}},
        "evidence_offload": {"status": "applied", "candidate_bytes": 0, "offloaded_bytes": 0,
                             "retained_by_reason": {"protected_process": {"count": 1, "bytes": 5 * GIB}}},
        "scene_workspaces": {"status": "dry_run", "candidate_bytes": 10 * GIB,
                             "retained_by_reason": {"not_enabled": {"count": 1, "bytes": 2 * GIB}}},
    })), encoding="utf-8")
    return path


def _gc_tick(tmp_path, *, free_gib, total_gib, credit_collector, now=100.0):
    from collections import namedtuple
    usage = namedtuple("Usage", "total used free")(total_gib * GIB, (total_gib - free_gib) * GIB, free_gib * GIB)
    return capacity.run_controller(
        mounts=[str(tmp_path)], report_root=tmp_path / "report", reservation_root=tmp_path / "ledger",
        webhook_url="https://test.invalid", volume=None, ack="", token="", survey=None, now=now,
        disk_usage=lambda _: usage, poster=lambda *_args: None, credit_collector=credit_collector,
        release_retirement_summary_path=tmp_path / "absent-retirement",
        break_glass_notes_root=tmp_path / "absent-notes", storage_gc_summary_path=_gc_found_nothing(tmp_path, now))


def _blip():
    def failed(**_kwargs):
        raise RuntimeError("Vast API timeout")
    return credit.observe_vast_credit(api_key="test-secret", now=100, request=failed)


def test_a_warn_only_credit_code_on_healthy_disks_is_a_warning_and_never_a_disk_page(tmp_path):
    # Since 3887f77f6 (2026-09-05) any funding blocker made the tick critical: a Vast
    # API blip repeated hourly and, with GC finding nothing to free, paged reclaim_ineffective.
    result = _gc_tick(tmp_path, free_gib=90, total_gib=100, credit_collector=_blip)
    assert result["provider_funding"]["blockers"] == ["provider_credit_unverifiable"]
    assert result["level"] == "warning"
    severities = {row["code"]: row["severity"] for row in result["alerts"]}
    assert severities["provider_credit_unverifiable"] == "warn"
    assert "reclaim_ineffective" not in severities and "page" not in severities.values()


def test_a_low_balance_page_makes_the_tick_critical_without_a_disk_page(tmp_path):
    result = _gc_tick(tmp_path, free_gib=90, total_gib=100, credit_collector=lambda: observation(8))
    assert result["level"] == "critical"
    severities = {row["code"]: row["severity"] for row in result["alerts"]}
    assert severities["provider_credit_insufficient"] == "page"
    # The disks are healthy, so the disk-only reclaim page stays quiet.
    assert "reclaim_ineffective" not in severities


def test_a_disk_critical_tick_with_a_warn_only_credit_code_stays_critical(tmp_path):
    result = _gc_tick(tmp_path, free_gib=20, total_gib=154, credit_collector=_blip)
    assert result["level"] == "critical"
    severities = {row["code"]: row["severity"] for row in result["alerts"]}
    assert severities["provider_credit_unverifiable"] == "warn"
    # A real disk-critical tick still pages reclaim_ineffective.
    assert severities["reclaim_ineffective"] == "page"
