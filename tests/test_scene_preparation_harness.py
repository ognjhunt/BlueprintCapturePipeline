"""The old scene-preparation stage, website intent to `completed/scene_prepared`, pinned as one record.

This is the comparison basis for the stage rebuild (S7): every rebuild step must give the same record.
Regenerate the goldens after an intended change with:

    BLUEPRINT_HARNESS_REGENERATE=1 INVOCATION_ID=ci-runner PYTHONPATH=$PWD/src python -m pytest -p no:cacheprovider tests/test_scene_preparation_harness.py -k golden
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.scene_preparation_harness import fixed_root, run_website_intent

GOLDEN = Path(__file__).parent / "fixtures" / "scene_preparation_harness"
KINDS = ("pick_and_place", "articulated_open_close")


def _launched_then_prepared(record: dict) -> None:
    ticks = [tick["progression"] for tick in record["ticks"]]
    # Tick 1: progression -> owned worker -> progression; the activation is staged, nothing submitted yet.
    assert record["ticks"][0]["passes"][0] == [{"status": "running", "phase": "source_preparation", "blockers": []}]
    assert record["ticks"][0]["worker"] == [{"status": "queued_for_production_scene_configuration", "blockers": None}]
    assert ticks[:2] == [{"status": "awaiting_execution", "phase": "scene_configuration", "blockers": []}] * 2
    # Tick 2: launch submitted once, within the standing authorization's spend.
    [submission] = record["launch_submissions"]
    assert submission["spend"] == {"max_spend_usd": 11.0}
    assert submission["rights"]["scope"] == "recorded-owner-rights"
    # One $0 preparation attempt and one paid hold at the request's hard cap.
    by_kind = {row["attempt_id"].split("-")[0]: row for row in record["attempts"]}
    assert by_kind["source"]["maximum_spend_usd"] == 0 and by_kind["source"]["provider"] == "control_plane"
    assert by_kind["scene"] == {**by_kind["scene"], "provider": "vast", "counts_as_paid_attempt": True,
                                "maximum_spend_usd": record["preparation_request"]["hard_cap_usd"]}
    assert record["reconciler"]["status"] == "passed"
    assert [row["status"] for row in record["reconciler"]["rows"]] == ["provider_zero_confirmed",
                                                                       "webapp_sync_succeeded"]
    # Tick 3: the completion join closes the job and enqueues the scene-packet request.
    assert ticks[2] == {"status": "completed", "phase": "scene_prepared", "blockers": []}
    assert record["status"] == {"status": "completed", "phase": "scene_prepared", "blockers": []}
    assert record["team_scene_requests"] == 1 and record["outcome"] is None


def _golden(name: str, record: dict) -> None:
    path = GOLDEN / f"{name}.json"
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    if os.environ.get("BLUEPRINT_HARNESS_REGENERATE") == "1":
        path.write_text(text)
    assert json.loads(path.read_text()) == record, f"record changed; regenerate {path.name} if intended"


@pytest.mark.parametrize("kind", KINDS)
def test_website_job_reaches_launch_then_scene_prepared_and_matches_its_golden(kind: str) -> None:
    """(a) rigid pick-and-place, (b) articulated open/close; (d) each record is pinned."""
    with fixed_root(kind) as root:
        record = run_website_intent(root, task_kind=kind)
    assert record["intent"]["strategy"] == kind
    _launched_then_prepared(record)
    _golden(kind, record)


def test_two_runs_give_an_equal_record() -> None:
    """(c) the normalised record is deterministic (same root, same clock, same release)."""
    records = []
    for _ in range(2):
        with fixed_root("determinism") as root:
            records.append(run_website_intent(root, task_kind="pick_and_place"))
    assert records[0] == records[1]


def test_hard_stops_show_in_the_record(tmp_path: Path) -> None:
    """(e) revocation, a policy candidate and a missing registration, each before any paid step."""
    revoked = run_website_intent(tmp_path / "revoked", hard_stop="revoked")
    assert [tick["progression"] for tick in revoked["ticks"]] == [
        {"status": "blocked", "phase": "authority", "blockers": ["scene_intake_authority_revoked"]}]
    assert revoked["outcome"] == {"kind": "withdrawn", "terminal": True, "reason_code": "authority_revoked"}
    assert revoked["status"]["status"] == "revoked" and revoked["attempts"] == []

    policy = run_website_intent(tmp_path / "policy", hard_stop="policy_candidate")
    assert policy["intake"] == {"status": "refused",
                                "code": "scene_intake_scene_preparation_must_not_select_robot_or_policy"}
    assert policy["intents"] == [True] and policy["source_work"] == [] and policy["ticks"] == []

    waiting = run_website_intent(tmp_path / "unregistered", hard_stop="unregistered")
    assert [tick["progression"] for tick in waiting["ticks"]] == [
        {"status": "awaiting_source", "phase": "source", "blockers": ["website_prepared_source_pending"]}]
    assert waiting["status"]["status"] == "awaiting_source" and waiting["attempts"] == []
    assert waiting["outcome"] is None and "launch_submissions" in waiting and waiting["launch_submissions"] == []
