# Covers (for impacted-test selection):
#   src/blueprint_pipeline/preparation_fault_alert.py
#   src/blueprint_pipeline/control_plane_stall_alarm.py
"""A preparation stopped on Blueprint's own records pages the owner once (2026-11-11).

Nothing read the preparation ledger before: such a job had six timed retries, then waited for its window,
and the site was told only that it needs our team's review.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_capacity_controller as capacity
from blueprint_pipeline import control_plane_stall_alarm as stall
from blueprint_pipeline import preparation_fault_alert as fault

NOW = 1_800_000_000.0


def _ledger(root: Path, capture: str, **fields) -> Path:
    path = root / f"bucket/scenes/site-{capture}/captures/walkthrough-{capture}/pipeline_job_ledger.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": "pipeline_job_ledger.v1", **fields}))
    return path


@pytest.mark.parametrize("code", fault.FAULT_CODES)
def test_a_preparation_stopped_on_our_own_records_pages_once_then_stays_quiet(tmp_path, code):
    _ledger(tmp_path, "a", status="failed_retryable",
            last_error=f"StageError: website_scene_preparation: {code}")
    _ledger(tmp_path, "b", status="failed_retryable", last_error="website_registration_poor_fit")  # the site's ask
    _ledger(tmp_path, "c", status="completed", last_error=None)
    memory: dict = {}
    section, alerts = fault.check_preparation_faults(tmp_path, now=NOW, memory_out=memory)
    assert section["stopped"] == 1 and [row["code"] for row in alerts] == [fault.CODE]
    assert alerts[0]["blockers"] == [code] and alerts[0]["label"] == "walkthrough-a"
    assert capacity._severity(fault.CODE) == "page"
    assert code in stall.describe(alerts[0]) and "ours to fix" in stall.describe(alerts[0])
    # Each later tick remembers when it was first seen: within the announce window it may repeat once
    # more if a delivery failed; after it, the same stop never pages again, however many retries follow.
    later: dict = {}
    assert fault.check_preparation_faults(tmp_path, now=NOW + fault.ANNOUNCE_SECONDS, remembered=memory,
                                          memory_out=later)[1] == []
    assert later == memory


def test_the_stall_tick_runs_it_on_the_capture_roots(tmp_path):
    _ledger(tmp_path, "a", status="failed_retryable",
            last_error="website_scene_preparation: website_registration_anchor_frame_missing")
    config = stall.StallConfig(None, None, None, None, atlas_storage_root=tmp_path)
    result = stall.observe_stalls(config, now=NOW, unit_reader=lambda _units: {})
    assert result["checks"]["preparation_faults"]["stopped"] == 1
    assert any(row["code"] == fault.CODE for row in result["alerts"])
    again = stall.observe_stalls(config, now=NOW + 2 * fault.ANNOUNCE_SECONDS, previous=result,
                                 unit_reader=lambda _units: {})
    assert not any(row["code"] == fault.CODE for row in again["alerts"])


def test_more_ledgers_than_one_tick_reads_is_unknown_not_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(fault, "MAX_LEDGERS", 1)
    _ledger(tmp_path, "a", status="completed")
    _ledger(tmp_path, "b", status="completed")
    section, alerts = fault.check_preparation_faults(tmp_path, now=NOW)
    assert section["truncated"] is True
    assert [row["code"] for row in alerts] == [stall.INCOMPLETE_CODE]


def test_a_provider_refusing_blueprints_key_pages_once_per_service_while_jobs_wait(tmp_path):
    """2026-11-22: a 401, 402 or 403 from an outside service is an outage the room builds wait on; one page."""
    for capture, reason in (("a", "world_labs_http_401:invalid key"), ("b", "world_labs_http_402"),
                            ("c", "world_labs_http_413")):
        path = tmp_path / f"bucket/scenes/site-{capture}/captures/walkthrough-{capture}/pipeline/atlas/preview_job.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"state": "processing", "transient_failures": 2, "last_transient_reason": reason}))
    memory: dict = {}
    section, alerts = fault.check_preparation_faults(tmp_path, now=NOW, memory_out=memory)
    assert [row["code"] for row in alerts] == [fault.KEY_CODE] and alerts[0]["count"] == 1
    # Since 2026-11-23 the page names how many jobs wait on the service (the 413 is no account refusal).
    assert alerts[0]["label"] == "world_labs (2 jobs waiting)" and section["provider_account_refusals_announced"] == 1
    assert capacity._severity(fault.KEY_CODE) == "page" and "key or account" in stall.describe(alerts[0])
    assert fault.check_preparation_faults(tmp_path, now=NOW + fault.ANNOUNCE_SECONDS, remembered=memory)[1] == []
