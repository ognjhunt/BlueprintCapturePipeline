"""A live job that stops moving pages on the capacity route; intended holds never do."""

from __future__ import annotations

import fcntl
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.deploy_control_plane_commit as deploy
from blueprint_pipeline import control_plane_capacity_controller as cap
from blueprint_pipeline import control_plane_stall_alarm as stall
from blueprint_pipeline import task_evaluation_launch_reconciler as reconciler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))
from operator_door.secrets_guard import scan_bytes  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NOW = 1_793_500_000.0
MINUTE = 60.0
INTENT = "scene-" + "ab" * 32


def _budget() -> stall._Budget:
    return stall._Budget(deadline=float("inf"), events=stall.MAX_EVENT_READS, clock=lambda: 0.0)


def _config(tmp_path: Path, **overrides) -> stall.StallConfig:
    values = {
        "scene_intent_root": tmp_path / "intents",
        "launch_queue_root": tmp_path / "launches",
        "launch_state_root": tmp_path / "runs",
        "deploy_tool": None,
    }
    values.update(overrides)
    return stall.StallConfig(**values)


def _intent(root: Path, events: list[tuple[float, str, str, list[str]]], *, name: str = INTENT,
            expires_at: float = NOW + 86_400, state_churn: bool = False) -> Path:
    """A retained intent: events are (observed_at, status, phase, blockers), oldest first."""

    directory = root / name
    (directory / "progression-events").mkdir(parents=True)
    (directory / "intent.json").write_text(json.dumps({
        "intent_id": name, "accepted_at_epoch": events[0][0] if events else NOW - 3600,
        "request": {"execution": {"expires_at_epoch": expires_at},
                    "submission_id": "walkthrough-capture-447ae4f4-c253"},
    }))
    for sequence, (observed, status, phase, blockers) in enumerate(events, start=1):
        event = {"schema_version": "task_evaluation_scene_progression_event.v1", "intent_id": name,
                 "sequence": sequence, "observed_at_epoch": observed, "status": status, "phase": phase,
                 "blockers": sorted(blockers), "state": {"tick": sequence} if state_churn else {}}
        (directory / "progression-events" / f"{sequence:06d}.json").write_text(json.dumps(event))
    if events:
        observed, status, phase, blockers = events[-1]
        (directory / "progression.json").write_text(json.dumps({
            "schema_version": "task_evaluation_scene_progression.v1", "intent_id": name,
            "event_sequence": len(events), "status": status, "phase": phase,
            "blockers": sorted(blockers), "updated_at_epoch": observed, "state": {},
        }))
    return directory


def _progressions(tmp_path: Path, **config):
    return stall.check_scene_progressions(_config(tmp_path, **config), now=NOW, budget=_budget())


# --------------------------------------------------------------------------- #
# Scene progression
# --------------------------------------------------------------------------- #


def test_one_blocker_held_for_hours_pages_with_the_oldest_offender(tmp_path: Path) -> None:
    # 2026-11-01: blocked with one code for about two hours while state kept moving.
    root = tmp_path / "intents"
    _intent(root, [
        (NOW - 3 * 3600, "awaiting_execution", "scene_configuration", []),
        (NOW - 2 * 3600, "blocked", "preparation", ["scene_progression_ownership_reconciliation_required"]),
        (NOW - 30 * MINUTE, "blocked", "preparation", ["scene_progression_ownership_reconciliation_required"]),
        (NOW - 5 * MINUTE, "blocked", "preparation", ["scene_progression_ownership_reconciliation_required"]),
    ], state_churn=True)
    section, alerts = _progressions(tmp_path)
    assert section["stalled"] == 1 and section["dispositions"] == {"watched": 1}
    [alert] = alerts
    assert alert["code"] == "scene_progression_stalled" and alert["count"] == 1
    assert alert["target"] == INTENT and alert["targets"] == [INTENT]
    assert alert["label"] == "walkthrough-capture-447ae4f4-c253"
    assert (alert["status"], alert["phase"]) == ("blocked", "preparation")
    assert alert["blockers"] == ["scene_progression_ownership_reconciliation_required"]
    # State-only events never reset the clock: it started at the first blocked event.
    assert alert["age_seconds"] == 2 * 3600 and alert["age_is_lower_bound"] is False
    assert alert["threshold_seconds"] == 90 * MINUTE
    line = stall.describe({**alert, "severity": "page"})
    assert line.startswith("scene progression unchanged 2.0h at blocked/preparation")
    assert "walkthrough-capture-447ae4f4-c253" in line


def test_a_row_that_changed_inside_the_threshold_does_not_page(tmp_path: Path) -> None:
    _intent(tmp_path / "intents", [
        (NOW - 5 * 3600, "blocked", "preparation", ["scene_progression_ownership_reconciliation_required"]),
        (NOW - 80 * MINUTE, "blocked", "preparation", ["scene_intake_record_unreadable"]),
    ])
    section, alerts = _progressions(tmp_path)
    assert section["stalled"] == 0 and alerts == []


def test_a_handed_off_row_waits_out_a_downstream_run_before_paging(tmp_path: Path) -> None:
    # A scene configuration launch may legitimately hold this row for its 7.5 h TTL.
    _intent(tmp_path / "intents", [(NOW - 3 * 3600, "awaiting_execution", "scene_configuration", [])])
    assert _progressions(tmp_path)[1] == []
    stalled = stall.check_scene_progressions(
        _config(tmp_path), now=NOW + 5 * 3600 + 1, budget=_budget())[1]
    assert stalled[0]["threshold_seconds"] == 8 * 3600 and stalled[0]["age_seconds"] == 8 * 3600 + 1


@pytest.mark.parametrize(("status", "phase", "blockers", "reason"), [
    ("completed", "scene_prepared", [], "terminal"),
    ("needs_input", "source", ["source_kind_not_supported_by_progression"], "needs_input"),
    ("awaiting_execution", "paused", ["scene_intent_paused"], "operator_pause"),
    ("awaiting_execution", "publication_ready", ["scene_submission_paused"], "operator_pause"),
    ("awaiting_execution", "construction_prepared", [], "operator_pause"),
    ("blocked", "authority", ["scene_intake_authority_revoked"], "rights"),
    ("blocked", "authority", ["scene_intake_authority_expired"], "execution_window"),
    ("blocked", "preparation", ["scene_intake_spend_cap_exhausted"], "money_approval"),
    ("blocked", "preparation", ["scene_intake_attempt_cap_exhausted"], "money_approval"),
    ("blocked", "source", ["capture_consent_missing"], "privacy"),
    ("blocked", "source", ["source_rights_reference_mismatch"], "rights"),
])
def test_intended_holds_never_page(tmp_path: Path, status, phase, blockers, reason) -> None:
    assert stall.hold_reason(status, phase, blockers) == reason
    _intent(tmp_path / "intents", [(NOW - 30 * 86400, status, phase, blockers)])
    section, alerts = _progressions(tmp_path)
    assert alerts == [] and section["dispositions"] == {reason: 1}


def test_pipeline_blockers_are_not_mistaken_for_holds() -> None:
    for code in ("scene_progression_ownership_reconciliation_required", "scene_progression_dependency_unavailable",
                 "policy_canary_episode_failure:DroidActionExecutionError", "preparation_failed",
                 "scene_whole_chain_capacity_insufficient", "scene_progression_recovery_provider_zero_required"):
        assert stall.hold_reason("blocked", "preparation", [code]) is None, code


def test_expired_or_revoked_windows_are_excluded_but_an_unreadable_intent_is_not(tmp_path: Path) -> None:
    root = tmp_path / "intents"
    blocked = [(NOW - 5 * 3600, "blocked", "policy_canary_blocked", ["provider_output_upload_marker_missing"])]
    _intent(root, blocked, name="scene-expired", expires_at=NOW - 1)
    (_intent(root, blocked, name="scene-revoked") / "revoked.json").write_text("{}")
    unreadable = _intent(root, blocked, name="scene-unknown-window")
    (unreadable / "intent.json").write_text("not json")
    section, alerts = _progressions(tmp_path)
    assert section["dispositions"] == {"execution_window": 1, "rights": 1, "watched": 1}
    assert alerts[0]["targets"] == ["scene-unknown-window"] and "label" not in alerts[0]


def test_an_intent_the_progression_never_picked_up_pages(tmp_path: Path) -> None:
    root = tmp_path / "intents"
    _intent(root, [], name="scene-never-started")
    json_path = root / "scene-never-started" / "intent.json"
    intent = json.loads(json_path.read_text())
    intent["accepted_at_epoch"] = NOW - 2 * 3600
    json_path.write_text(json.dumps(intent))
    [alert] = _progressions(tmp_path)[1]
    assert (alert["status"], alert["phase"], alert["age_seconds"]) == ("not_started", "intake", 7200)


def test_a_long_unchanged_chain_reports_a_lower_bound_within_the_event_limit(tmp_path: Path) -> None:
    events = [(NOW - 10 * 3600 + index * MINUTE, "blocked", "preparation", ["preparation_failed"])
              for index in range(stall.MAX_EVENTS_PER_INTENT + 20)]
    _intent(tmp_path / "intents", events, state_churn=True)
    [alert] = _progressions(tmp_path)[1]
    assert alert["age_is_lower_bound"] is True
    newest = len(events) - 1
    assert alert["age_seconds"] == round(NOW - events[newest - stall.MAX_EVENTS_PER_INTENT][0])
    assert stall.describe({**alert, "severity": "page"}).startswith("scene progression unchanged over ")


def test_oldest_offender_leads_and_count_covers_every_stalled_intent(tmp_path: Path) -> None:
    root = tmp_path / "intents"
    for name, hours in (("scene-a", 2), ("scene-b", 6), ("scene-c", 3)):
        _intent(root, [(NOW - hours * 3600, "blocked", "preparation", ["preparation_failed"])], name=name)
    [alert] = _progressions(tmp_path)[1]
    assert alert["target"] == "scene-b" and alert["count"] == 3
    assert alert["targets"] == ["scene-b", "scene-c", "scene-a"]


def test_malformed_rows_and_a_missing_root_do_not_raise(tmp_path: Path) -> None:
    assert _progressions(tmp_path)[0] == {"status": "absent"}
    bad = tmp_path / "intents" / "scene-bad"
    bad.mkdir(parents=True)
    (bad / "progression.json").write_text(json.dumps({"schema_version": "other", "intent_id": "scene-bad"}))
    section, alerts = _progressions(tmp_path)
    assert section["unreadable"] == 1
    assert alerts == [{"code": stall.INCOMPLETE_CODE, "reason": "scene_progression_unreadable", "count": 1}]


def test_progression_check_never_writes(tmp_path: Path) -> None:
    root = tmp_path / "intents"
    _intent(root, [(NOW - 3 * 3600, "blocked", "preparation", ["preparation_failed"])])
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in root.rglob("*") if path.is_file()}
    _progressions(tmp_path)
    after = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in root.rglob("*") if path.is_file()}
    assert before == after


# --------------------------------------------------------------------------- #
# Launch leases
# --------------------------------------------------------------------------- #


def _lease(tmp_path: Path, launch_id: str, *, enqueued: float, started: dict | None = None) -> Path:
    processing = tmp_path / "launches" / "processing"
    processing.mkdir(parents=True, exist_ok=True)
    (tmp_path / "launches" / ".dispatcher.lock").touch()
    path = processing / f"{launch_id}-digest.json"
    path.write_text(json.dumps({"launch_id": launch_id}))
    os.utime(path, (enqueued, enqueued))
    if started is not None:
        run = tmp_path / "runs" / launch_id
        run.mkdir(parents=True)
        (run / "launch_started.json").write_text(json.dumps(started))
    return path


def _leases(tmp_path: Path, now: float = NOW):
    return stall.check_launch_leases(_config(tmp_path), now=now, budget=_budget())


def test_an_unowned_pre_run_lease_pages_after_the_reconciler_bound_and_grace(tmp_path: Path) -> None:
    _lease(tmp_path, "website-aa-launch", enqueued=NOW - 44 * MINUTE)
    assert _leases(tmp_path)[1] == []  # 15 min bound + 30 min grace not yet passed
    section, [alert] = _leases(tmp_path, now=NOW + 2 * MINUTE)
    assert section == {"status": "observed", "processing": 1, "unowned": 1, "stuck": 1, "truncated": False}
    assert alert["code"] == "launch_lease_stuck" and alert["target"] == "website-aa-launch"
    assert alert["phase"] == "pre_run" and alert["bound_seconds"] == 900
    assert "with no dispatcher: website-aa-launch" in stall.describe(alert)


def test_a_started_run_gets_its_hard_ttl(tmp_path: Path) -> None:
    started_at = "2026-11-01T02:58:41+00:00"
    started_epoch = 1_793_501_921.0
    _lease(tmp_path, "website-bb-launch", enqueued=started_epoch - 60,
           started={"started_at": started_at, "hard_ttl_seconds": 27_000})
    assert _leases(tmp_path, now=started_epoch + 27_000 + 29 * MINUTE)[1] == []
    [alert] = _leases(tmp_path, now=started_epoch + 27_000 + 31 * MINUTE)[1]
    assert alert["phase"] == "run" and alert["bound_seconds"] == 27_000


def test_a_live_dispatcher_owns_every_lease(tmp_path: Path) -> None:
    _lease(tmp_path, "website-cc-launch", enqueued=NOW - 10 * 3600)
    descriptor = os.open(tmp_path / "launches" / ".dispatcher.lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        section, alerts = _leases(tmp_path)
    finally:
        os.close(descriptor)
    assert alerts == [] and section["unowned"] == 0 and section["processing"] == 1
    assert _leases(tmp_path)[0]["stuck"] == 1  # the dispatcher is gone


def test_without_a_dispatcher_lock_leases_count_as_owned(tmp_path: Path) -> None:
    _lease(tmp_path, "website-dd-launch", enqueued=NOW - 10 * 3600)
    (tmp_path / "launches" / ".dispatcher.lock").unlink()
    assert _leases(tmp_path) == ({"status": "ownership_unknown", "processing": 1}, [])


def test_lease_bounds_mirror_the_launch_reconciler() -> None:
    assert stall.LAUNCH_PRE_RUN_STALE_SECONDS == reconciler.PRE_RUN_STALE_SECONDS
    default = inspect.signature(reconciler.reconcile_launches).parameters["fallback_stale_seconds"].default
    assert stall.LAUNCH_FALLBACK_STALE_SECONDS == default


# --------------------------------------------------------------------------- #
# Always-armed triggers
# --------------------------------------------------------------------------- #


def _deploy_tool(tmp_path: Path, holds: Path) -> Path:
    path = tmp_path / "deploy_tool.py"
    path.write_text(
        f"DEFAULT_DOOR_HOLDS_DIR = {str(holds)!r}\n"
        "DEFAULT_ALWAYS_ARM_PATH_UNITS = ('a.path',)\n"
        "DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS = ('b.path',)\n"
        "DEFAULT_ALWAYS_ARM_TIMER_UNITS = ('c.timer', 'd.timer', 'e.timer', 'cc.timer', 'f.timer')\n"
        "DEPLOY_EXEMPT_TRIGGER_UNITS = ('g.timer',)\n"
        "CONFIGURED_CONTROLS_AUTOMATION_UNITS = ('cc.timer',)\n"
    )
    return path


def test_an_inactive_always_armed_trigger_without_a_hold_pages(tmp_path: Path) -> None:
    holds = tmp_path / "holds"
    holds.mkdir()
    (holds / "d.timer.json").write_text("{}")
    tool = _deploy_tool(tmp_path, holds)
    boot = 100_000.0

    def inactive(since_monotonic: float) -> dict[str, str]:
        return {"LoadState": "loaded", "ActiveState": "inactive",
                "InactiveEnterTimestampMonotonic": str(int(since_monotonic * 1_000_000))}

    states = {
        "a.path": {"LoadState": "loaded", "ActiveState": "active"},
        "b.path": inactive(boot - 2 * 3600),             # pages: two hours, no hold
        "c.timer": inactive(boot - 10 * MINUTE),         # a deploy's trigger pause
        "d.timer": inactive(boot - 30 * 86400),          # owner hold record
        "e.timer": {"LoadState": "not-found", "ActiveState": "inactive"},  # not deployed here
        "f.timer": {"LoadState": "loaded", "ActiveState": "failed", "InactiveEnterTimestampMonotonic": "0"},
        "g.timer": {"LoadState": "loaded", "ActiveState": "active"},
    }
    asked = []

    def reader(units):
        asked.append(tuple(units))
        return states

    section, [alert] = stall.check_triggers(
        _config(tmp_path, deploy_tool=tool), unit_reader=reader, monotonic=lambda: boot)
    # The configured-controls pair follows the deploy's preserve flag and is not watched.
    assert asked == [("a.path", "b.path", "c.timer", "d.timer", "e.timer", "f.timer", "g.timer")]
    assert section == {"status": "observed", "watched": 7, "loaded": 6, "held": ["d.timer"], "inactive_unheld": 2}
    assert alert["code"] == "trigger_inactive_unheld" and alert["targets"] == ["f.timer", "b.path"]
    assert alert["age_seconds"] == boot and alert["status"] == "failed"
    assert stall.describe(alert) == "f.timer failed 27.8h with no owner hold (+1 more)"


def test_the_trigger_contract_is_the_deploy_tools_own_lists() -> None:
    units, holds = stall.read_deploy_trigger_contract(REPO / "scripts" / "deploy_control_plane_commit.py")
    expected = {
        *deploy.DEFAULT_ALWAYS_ARM_PATH_UNITS, *deploy.DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS,
        *deploy.DEFAULT_ALWAYS_ARM_TIMER_UNITS, *deploy.DEPLOY_EXEMPT_TRIGGER_UNITS,
    } - set(deploy.CONFIGURED_CONTROLS_AUTOMATION_UNITS)
    assert set(units) == expected and holds == Path(deploy.DEFAULT_DOOR_HOLDS_DIR)
    assert "blueprint-task-evaluation-launch-reconciler.timer" in units
    assert "blueprint-task-evaluation-scene-progression.timer" in units
    assert stall.default_deploy_tool() == REPO / "scripts" / "deploy_control_plane_commit.py"


def test_an_unreadable_deploy_tool_or_systemctl_failure_is_incomplete_not_a_page(tmp_path: Path) -> None:
    missing = _config(tmp_path, deploy_tool=tmp_path / "absent.py")
    assert stall.check_triggers(missing)[1] == [{"code": stall.INCOMPLETE_CODE, "reason": "deploy_tool_unreadable"}]
    tool = _deploy_tool(tmp_path, tmp_path / "holds")

    def failing(_argv, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    section, alerts = stall.check_triggers(
        _config(tmp_path, deploy_tool=tool),
        unit_reader=lambda units: stall.read_unit_states(units, runner=failing))
    assert section["status"] == "unavailable"
    assert alerts == [{"code": stall.INCOMPLETE_CODE, "reason": "systemctl_show_failed"}]


def test_systemctl_show_is_one_bounded_read_only_call() -> None:
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=(
            "Id=a.path\nLoadState=loaded\nActiveState=active\nSubState=waiting\n"
            "InactiveEnterTimestampMonotonic=0\n\n"
            "Id=b.timer\nLoadState=loaded\nActiveState=inactive\nSubState=dead\n"
            "InactiveEnterTimestampMonotonic=123000000\n"))

    states = stall.read_unit_states(["a.path", "b.timer"], runner=runner)
    assert states["b.timer"]["InactiveEnterTimestampMonotonic"] == "123000000"
    assert states["a.path"]["ActiveState"] == "active"
    [(argv, kwargs)] = calls
    assert argv[:3] == ["systemctl", "show", "--no-pager"] and argv[-3:] == ["--", "a.path", "b.timer"]
    assert kwargs["timeout"] == stall.SYSTEMCTL_TIMEOUT_SECONDS


# --------------------------------------------------------------------------- #
# Configuration and the capacity route
# --------------------------------------------------------------------------- #


def test_configuration_comes_from_the_capacity_units_environment() -> None:
    assert stall.config_from_environment({stall.DISABLE_ENV: "0"}) is None
    config = stall.config_from_environment({
        stall.SCENE_INTENT_ROOT_ENV: "/srv/intents", stall.PROGRESSION_MINUTES_ENV: "45",
        stall.TRIGGER_MINUTES_ENV: "soon",
    })
    assert config.scene_intent_root == Path("/srv/intents") and config.launch_queue_root is None
    assert config.progression_seconds == 45 * MINUTE and config.handoff_seconds == 8 * 3600
    assert config.trigger_seconds == 60 * MINUTE
    assert config.config_errors == (f"threshold_invalid:{stall.TRIGGER_MINUTES_ENV}",)
    unit = (REPO / "deploy" / "systemd" / "blueprint-control-plane-capacity.service").read_text()
    reconciler_unit = (REPO / "deploy" / "systemd" / "blueprint-task-evaluation-launch-reconciler.service").read_text()
    for name in (stall.SCENE_INTENT_ROOT_ENV, stall.LAUNCH_QUEUE_ROOT_ENV, stall.LAUNCH_STATE_ROOT_ENV):
        line = next(row for row in unit.splitlines() if row.startswith(f"Environment={name}="))
        assert line in reconciler_unit.splitlines(), name


def test_unconfigured_roots_are_reported_as_incomplete(tmp_path: Path) -> None:
    config = stall.StallConfig(None, None, None, None, config_errors=("threshold_invalid:X",))
    result = stall.observe_stalls(config, now=NOW, unit_reader=lambda _units: {})
    assert set(result["checks"]) == {"scene_progression", "launch_leases", "triggers"}
    assert [row["reason"] for row in result["alerts"]] == [
        "threshold_invalid:X", "scene_intent_root_unconfigured", "launch_roots_unconfigured",
        "deploy_tool_unconfigured"]
    assert all(row["code"] == stall.INCOMPLETE_CODE for row in result["alerts"])


def _stall_page(target: str = INTENT) -> dict:
    return {"code": "scene_progression_stalled", "count": 1, "targets": [target], "target": target,
            "label": "walkthrough-capture-447ae4f4-c253", "status": "blocked", "phase": "preparation",
            "blockers": ["scene_progression_ownership_reconciliation_required"], "age_seconds": 7200,
            "age_is_lower_bound": False, "threshold_seconds": 5400}


def _tick(tmp_path: Path, now: float, alerts: list[dict], posted: list, **kwargs) -> dict:
    return cap.run_controller(
        mounts=[str(tmp_path)], report_root=tmp_path / "capacity", reservation_root=tmp_path / "r",
        webhook_url="https://alerts.example/hook", volume=None, ack="", token="", survey=None,
        disk_usage=lambda _path: SimpleNamespace(total=100 * 1024**3, used=10 * 1024**3, free=90 * 1024**3),
        now=now, poster=lambda _url, value: posted.append(value),
        release_retirement_summary_path=tmp_path / "absent-retirement",
        break_glass_notes_root=tmp_path / "absent-notes",
        storage_gc_summary_path=tmp_path / "absent-gc",
        stall_check=lambda observed, _prior: {"schema_version": stall.SCHEMA_VERSION, "observed_at_epoch": observed,
                                      "checks": {"scene_progression": {"status": "observed"}},
                                      "alerts": [dict(row) for row in alerts]},
        **kwargs,
    )


def test_a_stall_pages_on_the_capacity_webhook_without_changing_the_disk_level(tmp_path: Path) -> None:
    posted: list = []
    report = _tick(tmp_path, NOW, [_stall_page()], posted)
    page = next(a for a in report["alerts"] if a["code"] == "scene_progression_stalled")
    assert page["severity"] == "page" and report["level"] == "ok"
    assert report["alert_posted"] is True and len(posted) == 1
    assert report["stall"]["checks"] == {"scene_progression": {"status": "observed"}}
    # Persisting: no repost inside the hour, then the hourly repeat.
    assert _tick(tmp_path, NOW + 600, [_stall_page()], posted)["alert_posted"] is False
    assert _tick(tmp_path, NOW + 3601, [_stall_page()], posted)["alert_posted"] is True
    # A newly stalled job posts at once even though the code already paged.
    second = {**_stall_page(), "count": 2, "targets": [INTENT, "scene-new"]}
    assert _tick(tmp_path, NOW + 3700, [second], posted)["alert_posted"] is True
    assert len(posted) == 3


def test_the_webhook_payload_leads_with_the_stall(monkeypatch) -> None:
    payloads = []

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(cap.urllib.request, "urlopen",
                        lambda request, *, timeout: payloads.append(json.loads(request.data)) or Response())
    disk = {"code": "orphan_scratch_large", "severity": "page", "allocated_bytes": 26 * 1024**3, "count": 625}
    report = {"level": "critical", "mounts": [], "alerts": [disk, {**_stall_page(), "severity": "page"}]}
    cap.post_alert("https://alerts.example/hook", report)
    payload = payloads[0]
    assert payload["page"] is True and payload["severity"] == "page"
    assert payload["summary"].startswith(
        "scene progression unchanged 2.0h at blocked/preparation "
        "[scene_progression_ownership_reconciliation_required]: walkthrough-capture-447ae4f4-c253")
    assert len(payload["summary"]) <= 200
    assert payload["text"].startswith("control-plane stall: scene progression unchanged 2.0h")
    assert "None orphan_scratch_large" in payload["text"] and "None scene_progression_stalled" not in payload["text"]
    assert payload["stall_runbook"] == "docs/runbooks/control-plane-capacity.md#stalled-jobs"
    assert "## Stalled jobs" in (REPO / "docs" / "runbooks" / "control-plane-capacity.md").read_text()
    # Only a stall: still a page, and the summary is the stall alone.
    cap.post_alert("https://alerts.example/hook", {"level": "ok", "mounts": [],
                                                   "alerts": [{**_stall_page(), "severity": "page"}]})
    assert payloads[1]["page"] is True and payloads[1]["summary"].startswith("scene progression unchanged")


def test_a_failing_stall_check_never_stops_the_capacity_tick(tmp_path: Path) -> None:
    def broken(_observed, _prior):
        raise RuntimeError("boom")

    report = cap.run_controller(
        mounts=[str(tmp_path)], report_root=tmp_path / "capacity", reservation_root=tmp_path / "r",
        webhook_url="", volume=None, ack="", token="", survey=None, now=NOW, stall_check=broken,
        disk_usage=lambda _path: SimpleNamespace(total=100, used=10, free=90),
        release_retirement_summary_path=tmp_path / "absent-retirement",
        break_glass_notes_root=tmp_path / "absent-notes", storage_gc_summary_path=tmp_path / "absent-gc",
    )
    incomplete = next(a for a in report["alerts"] if a["code"] == stall.INCOMPLETE_CODE)
    assert incomplete == {"code": stall.INCOMPLETE_CODE, "reason": "stall_check_failed", "severity": "warn"}
    assert (tmp_path / "capacity" / "latest.json").is_file()


def test_the_door_readable_summary_carries_the_stall_and_passes_the_secret_scan(tmp_path: Path) -> None:
    posted: list = []
    report = _tick(tmp_path, NOW, [_stall_page()], posted)
    summary = cap.capacity_summary(report)
    row = next(a for a in summary["alerts"] if a["code"] == "scene_progression_stalled")
    assert row["target"] == INTENT and row["blockers"] and row["age_seconds"] == 7200
    assert summary["stall"]["checks"]["scene_progression"]["status"] == "observed"
    raw = (tmp_path / "capacity" / "summary.json").read_bytes()
    assert scan_bytes(raw) is None and b"scene_progression_stalled" in raw


def test_a_churning_stall_reaches_its_threshold_across_ticks(tmp_path: Path) -> None:
    """Codex P1: minute-spaced state-only events cap one walk at ~64 min of history."""

    start = NOW - 6 * 3600
    code = ["scene_whole_chain_capacity_insufficient"]

    def chain(until: float) -> list:
        return [(start + index * MINUTE, "blocked", "preparation", code)
                for index in range(int((until - start) // MINUTE))]

    # The intent's files as an early tick saw them, and as a later tick sees them.
    first_tick = start + 100 * MINUTE
    _intent(tmp_path / "early" / "intents", chain(first_tick), state_churn=True)
    _intent(tmp_path / "late" / "intents", chain(NOW), state_churn=True)
    memory: dict = {}
    _section, alerts = stall.check_scene_progressions(_config(tmp_path / "early"), now=first_tick,
                                                      budget=_budget(), memory_out=memory)
    assert not [a for a in alerts if a["code"] == "scene_progression_stalled"]
    assert memory[INTENT]["exact"] is False
    remembered_since = memory[INTENT]["since"]
    # Without the memory, the later walk alone sees about 64 minutes and never pages.
    _section, alone = stall.check_scene_progressions(_config(tmp_path / "late"), now=NOW, budget=_budget())
    assert not [a for a in alone if a["code"] == "scene_progression_stalled"]
    # With it, the earlier start carries over and the stall pages.
    later: dict = {}
    _section, alerts = stall.check_scene_progressions(_config(tmp_path / "late"), now=NOW, budget=_budget(),
                                                      remembered=memory, memory_out=later)
    [page] = [a for a in alerts if a["code"] == "scene_progression_stalled"]
    assert page["age_seconds"] == round(NOW - remembered_since) and page["age_seconds"] > 90 * 60
    assert page["age_is_lower_bound"] is True and later[INTENT]["since"] == remembered_since


def test_a_changed_signature_does_not_inherit_an_older_start(tmp_path: Path) -> None:
    events = [(NOW - 3 * 3600, "blocked", "preparation", ["preparation_failed"]),
              (NOW - 10 * MINUTE, "blocked", "preparation", ["other_blocker"])]
    _intent(tmp_path / "intents", events)
    stale = {INTENT: {"signature": ["blocked", "preparation", ["other_blocker"]], "since": NOW - 5 * 3600,
                      "exact": True}}
    memory: dict = {}
    _section, alerts = stall.check_scene_progressions(_config(tmp_path), now=NOW, budget=_budget(),
                                                      remembered=stale, memory_out=memory)
    # The walk found where this signature began, so the remembered start is ignored.
    assert not [a for a in alerts if a["code"] == "scene_progression_stalled"]
    assert memory[INTENT] == {"signature": ["blocked", "preparation", ["other_blocker"]],
                              "since": NOW - 10 * MINUTE, "exact": True}


def test_observe_stalls_carries_its_memory_between_ticks(tmp_path: Path) -> None:
    events = [(NOW - 2 * 3600, "blocked", "preparation", ["preparation_failed"])]
    _intent(tmp_path / "intents", events)
    first = stall.observe_stalls(_config(tmp_path), now=NOW, unit_reader=lambda _units: {})
    assert first["memory"]["scene_progression"][INTENT]["since"] == NOW - 2 * 3600
    second = stall.observe_stalls(_config(tmp_path), now=NOW + 600, previous=first, unit_reader=lambda _units: {})
    assert second["memory"]["scene_progression"][INTENT]["since"] == NOW - 2 * 3600


def test_a_blind_stall_check_is_delivered_on_a_healthy_host(tmp_path: Path) -> None:
    """Codex P2: an incomplete check on an ``ok`` host used to be dropped by ``alert_due``."""

    posted: list = []
    blind = [{"code": stall.INCOMPLETE_CODE, "reason": "scene_intent_root_unreadable"}]
    report = _tick(tmp_path, NOW, blind, posted)
    assert report["level"] == "ok" and report["alert_posted"] is True and len(posted) == 1
    # Persisting, it is not re-sent every tick.
    assert _tick(tmp_path, NOW + 600, blind, posted)["alert_posted"] is False
    # A healthy tick with nothing to say still posts nothing.
    assert _tick(tmp_path, NOW + 1200, [], posted)["alert_posted"] is False
