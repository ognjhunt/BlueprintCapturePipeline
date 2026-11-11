"""The off-host GPU reaper decides from the host's published deadlines and proves each destroy. All fakes, $0."""

from __future__ import annotations

import json

import pytest

from blueprint_pipeline import off_host_gpu_reaper as reaper
from blueprint_pipeline.off_host_gpu_reaper import Instance

NOW = 1_800_000_000.0
TERMINAL = frozenset({"exited", "stopped", "destroyed"})
LABEL = "blueprint-task-evaluation-scene-config-20261111t120000z-" + "a" * 8


class Store:
    def __init__(self, states=None):
        self.states = list(states or [])
        self.writes = {}

    def read(self, name):
        assert name == reaper.STATE_OBJECT
        if not self.states:
            return None
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        if isinstance(state, Exception):
            raise state
        return state if isinstance(state, bytes) else json.dumps(state).encode()

    def write(self, name, data):
        self.writes[name] = data


class Fleet:
    """A fake Vast account: ``destroy`` removes the row unless told the delete fails or does not take."""

    def __init__(self, rows, *, delete_status="stopped", delete_takes=True, fail_lists=()):
        self.rows = {row.instance_id: row for row in rows}
        self.delete_status, self.delete_takes = delete_status, delete_takes
        self.fail_lists = set(fail_lists)
        self.lists = 0
        self.destroyed = []

    def inventory(self):
        self.lists += 1
        if self.lists in self.fail_lists or "all" in self.fail_lists:
            raise ValueError("vast_inventory_unreadable:http_500")
        return list(self.rows.values())

    def destroy(self, instance_id):
        self.destroyed.append(instance_id)
        if self.delete_status == "stopped" and self.delete_takes:
            self.rows.pop(instance_id)
        return {"status": self.delete_status, "http": 200 if self.delete_status == "stopped" else 500}


def _state(*, heartbeat=NOW - 60, entries=(), longest=27_000.0, complete=None):
    state = {"schema_version": reaper.STATE_SCHEMA_VERSION, "heartbeat_epoch": heartbeat,
             "longest_deadline_seconds": longest, "entries": [dict(entry) for entry in entries]}
    return state if complete is None else {**state, "deadlines_complete": complete}


def _machine(instance_id=101, *, label=LABEL, status="running", age=3600.0):
    return Instance(instance_id=instance_id, label=label, status=status,
                    started_epoch=None if age is None else NOW - age)


def _run(store, fleet, *, now=NOW):
    return reaper.run_pass(store=store, inventory=fleet.inventory, destroy=fleet.destroy, clock=lambda: now,
                           sleep=lambda _seconds: None, terminal=TERMINAL)


def _outcome(report, instance_id=101):
    return next(row for row in report["instances"] if row["instance_id"] == instance_id)


def test_host_dead_and_a_machine_past_its_deadline_is_destroyed_and_its_absence_proven():
    store = Store([_state(heartbeat=NOW - 2 * 3600, entries=[{"label": LABEL, "deadline_epoch": NOW - 60}])])
    fleet = Fleet([_machine()])
    report = _run(store, fleet)
    assert fleet.destroyed == [101]
    row = _outcome(report)
    assert row["outcome"] == "destroyed" and row["proof"] == "absent_from_provider_list"
    assert row["reason"] == "past_deadline_host_heartbeat_stale"
    assert report["status"] == "ok" and report["destroyed"] == [101] and reaper.exit_code(report) == 0
    assert report["budget_admission"] == "not_required_spend_effect_stop"


@pytest.mark.parametrize("heartbeat_age", [60, 2 * 3600])
def test_a_machine_inside_its_deadline_is_left_host_alive_or_dead(heartbeat_age):
    store = Store([_state(heartbeat=NOW - heartbeat_age,
                          entries=[{"instance_id": 101, "label": LABEL, "deadline_epoch": NOW + 600}])])
    fleet = Fleet([_machine(age=10 * 86_400)])
    report = _run(store, fleet)
    assert fleet.destroyed == []
    assert _outcome(report)["outcome"] == "left" and _outcome(report)["reason"] == "within_deadline"
    assert report["status"] == "ok"


def test_host_alive_a_machine_past_its_deadline_waits_out_the_grace_then_is_destroyed():
    entries = [{"label_prefix": LABEL[:-8], "deadline_epoch": NOW - reaper.DEADLINE_GRACE_SECONDS + 30}]
    fleet = Fleet([_machine()])
    report = _run(Store([_state(entries=entries)]), fleet)
    assert fleet.destroyed == [] and _outcome(report)["outcome"] == "left"
    report = _run(Store([_state(entries=entries)]), fleet, now=NOW + 60)
    assert fleet.destroyed == [101] and _outcome(report)["reason"] == "past_deadline_and_grace"


@pytest.mark.parametrize("label", ["someone-elses-box", "", "Blueprint-Upper", "blueprint-"])
def test_a_machine_that_is_not_blueprints_is_never_touched(label):
    # Host dead and the machine far older than any deadline: were it Blueprint's, it would be destroyed.
    store = Store([_state(heartbeat=NOW - 86_400)])
    fleet = Fleet([_machine(label=label, age=30 * 86_400)])
    report = _run(store, fleet)
    assert fleet.destroyed == []
    row = _outcome(report)
    assert row["outcome"] == "left" and row["ownership"] == "unknown" and row["label"] is None
    assert report["status"] == "ok"


def test_a_failing_provider_list_is_unknown_never_nothing_running():
    fleet = Fleet([_machine()], fail_lists={"all"})
    report = _run(Store([_state(heartbeat=NOW - 86_400, entries=[{"label": LABEL, "deadline_epoch": NOW - 86_400}])]),
                  fleet)
    assert report["status"] == "unknown" and report["blockers"] == ["provider_inventory_unknown"]
    assert report["instances"] == [] and fleet.destroyed == [] and reaper.exit_code(report) == 1


@pytest.mark.parametrize("raw, blocker", [
    (None, "host_heartbeat_never_published"),
    (b"not json", "host_state_unreadable"),
    (OSError("bucket down"), "host_state_unreadable"),
    ({"schema_version": "something_else"}, "host_state_schema_invalid"),
    ({**_state(), "entries": [{"deadline_epoch": NOW}]}, "host_state_entry_invalid"),
    ({**_state(), "longest_deadline_seconds": None}, "host_state_longest_deadline_invalid"),
])
def test_no_heartbeat_or_an_unreadable_one_acts_on_nothing(raw, blocker):
    fleet = Fleet([_machine(age=30 * 86_400)])
    report = _run(Store([] if raw is None else [raw]), fleet)
    assert report["status"] == "unknown" and report["blockers"] == [blocker]
    assert fleet.destroyed == [] and fleet.lists == 0 and reaper.exit_code(report) == 1


@pytest.mark.parametrize("age, outcome, reason", [
    (27_000 + 60, "destroyed", "not_published_older_than_longest_deadline_host_heartbeat_stale"),
    (27_000 - 60, "left", "not_published_within_longest_deadline"),
    (None, "unknown", "not_published_host_heartbeat_stale_age_unknown"),
])
def test_host_dead_an_unpublished_machine_is_bounded_by_the_longest_deadline(age, outcome, reason):
    fleet = Fleet([_machine(age=age)])
    report = _run(Store([_state(heartbeat=NOW - 3600)]), fleet)
    row = _outcome(report)
    assert row["outcome"] == outcome and row["reason"] == reason
    assert fleet.destroyed == ([101] if outcome == "destroyed" else [])
    assert report["status"] == ("unknown" if outcome == "unknown" else "ok")


def test_host_alive_an_unpublished_machine_is_left_however_old():
    fleet = Fleet([_machine(age=30 * 86_400)])
    report = _run(Store([_state()]), fleet)
    assert fleet.destroyed == [] and _outcome(report)["reason"] == "not_published_host_alive"


def test_a_delete_that_fails_is_failed_never_destroyed():
    fleet = Fleet([_machine()], delete_status="stop_failed")
    report = _run(Store([_state(heartbeat=NOW - 3600, entries=[{"label": LABEL, "deadline_epoch": NOW - 7200}])]), fleet)
    assert _outcome(report)["outcome"] == "failed" and report["destroyed"] == []
    assert report["status"] == "failed" and reaper.exit_code(report) == 1


def test_a_delete_the_list_still_shows_running_is_failed():
    fleet = Fleet([_machine()], delete_takes=False)
    report = _run(Store([_state(heartbeat=NOW - 3600, entries=[{"label": LABEL, "deadline_epoch": NOW - 7200}])]), fleet)
    row = _outcome(report)
    assert row["outcome"] == "failed" and row["proof"] == "absence_not_verified"
    assert report["status"] == "failed"


def test_a_delete_whose_proof_list_fails_is_failed():
    fleet = Fleet([_machine()], fail_lists={3, 4, 5})
    report = _run(Store([_state(heartbeat=NOW - 3600, entries=[{"label": LABEL, "deadline_epoch": NOW - 7200}])]), fleet)
    assert fleet.destroyed == [101] and _outcome(report)["outcome"] == "failed"


def test_the_published_deadlines_are_read_again_before_acting():
    # The host came back between the two readings and published a superseding watchdog's later deadline.
    stale = _state(heartbeat=NOW - 3600, entries=[{"label": LABEL, "deadline_epoch": NOW - 60}])
    extended = _state(heartbeat=NOW, entries=[{"label": LABEL, "deadline_epoch": NOW - 60},
                                              {"label": LABEL, "deadline_epoch": NOW + 3600}])
    fleet = Fleet([_machine()])
    report = _run(Store([stale, extended]), fleet)
    assert fleet.destroyed == []
    assert _outcome(report)["outcome"] == "left" and _outcome(report)["deadline_epoch"] == NOW + 3600


def test_a_failed_reread_acts_on_nothing():
    stale = _state(heartbeat=NOW - 3600, entries=[{"label": LABEL, "deadline_epoch": NOW - 60}])
    fleet = Fleet([_machine()], fail_lists={2})
    report = _run(Store([stale]), fleet)
    assert fleet.destroyed == [] and report["status"] == "unknown"
    assert report["blockers"] == ["reread:provider_inventory_unknown"]


def test_terminal_instances_are_not_reported_or_destroyed():
    fleet = Fleet([_machine(status="exited", age=30 * 86_400)])
    report = _run(Store([_state(heartbeat=NOW - 3600)]), fleet)
    assert report["instances"] == [] and fleet.destroyed == []


def test_an_instance_id_entry_covers_its_machine_even_when_the_label_differs():
    entries = [{"instance_id": 101, "label_prefix": "blueprint-other-", "deadline_epoch": NOW + 3600}]
    fleet = Fleet([_machine(age=86_400)])
    report = _run(Store([_state(heartbeat=NOW - 3600, entries=entries)]), fleet)
    assert fleet.destroyed == [] and _outcome(report)["reason"] == "within_deadline"


def test_inventory_reader_refuses_a_partial_or_failed_list(monkeypatch):
    from blueprint_pipeline import vast_provider_adapter

    responses = {"payload": (200, {"instances": [{"id": 7, "actual_status": "running", "label": LABEL,
                                                  "start_date": NOW - 10}]})}
    monkeypatch.setattr(vast_provider_adapter, "_api_json", lambda **_kwargs: responses["payload"])
    assert reaper.read_vast_inventory("key") == [Instance(7, LABEL, "running", NOW - 10)]
    for payload in [(500, {}), (200, {"success": False}), (200, {"instances": [{"id": "x", "actual_status": "running"}]})]:
        responses["payload"] = payload
        with pytest.raises(ValueError):
            reaper.read_vast_inventory("key")


def test_main_without_a_bucket_reports_unknown(capsys, monkeypatch):
    monkeypatch.delenv(reaper.BUCKET_ENV, raising=False)
    assert reaper.main([]) == 1
    assert json.loads(capsys.readouterr().out)["blockers"] == ["reaper_bucket_not_configured"]


# ----------------------------- a watchdog that died before the host -----------------------------

LAST_HEARTBEAT = NOW - reaper.HEARTBEAT_STALE_SECONDS - 60  # stale


def test_a_machine_whose_watchdog_died_before_the_last_heartbeat_is_destroyed_once_stale():
    # Running for an hour when the host last read every watchdog, and none covered it.
    fleet = Fleet([_machine(age=NOW - (LAST_HEARTBEAT - 3600))])
    report = _run(Store([_state(heartbeat=LAST_HEARTBEAT, complete=True)]), fleet)
    assert fleet.destroyed == [101] and _outcome(report)["outcome"] == "destroyed"
    assert _outcome(report)["reason"] == "not_published_in_complete_list_host_heartbeat_stale"
    # The same machine while the heartbeat is still fresh is left: the host may still own it.
    fleet = Fleet([_machine(age=NOW - (NOW - 60 - 3600))])
    report = _run(Store([_state(heartbeat=NOW - 60, complete=True)]), fleet)
    assert fleet.destroyed == [] and _outcome(report)["reason"] == "not_published_host_alive"


@pytest.mark.parametrize("started_before_heartbeat", [reaper.ORPHAN_START_MARGIN_SECONDS - 30, -120])
def test_a_machine_launched_in_the_hosts_last_moments_keeps_the_longest_deadline(started_before_heartbeat):
    started = LAST_HEARTBEAT - started_before_heartbeat
    fleet = Fleet([_machine(age=NOW - started)])
    report = _run(Store([_state(heartbeat=LAST_HEARTBEAT, complete=True)]), fleet)
    assert fleet.destroyed == [] and _outcome(report)["reason"] == "not_published_within_longest_deadline"
    # Its fallback still ends it once it outlives the longest deadline the code can arm.
    later = started + 27_000 + 60
    report = _run(Store([_state(heartbeat=LAST_HEARTBEAT, complete=True)]), fleet, now=later)
    assert fleet.destroyed == [101]


@pytest.mark.parametrize("complete", [False, None])
def test_an_incomplete_or_unmarked_list_never_applies_the_orphan_rule(complete):
    fleet = Fleet([_machine(age=NOW - (LAST_HEARTBEAT - 3600))])
    report = _run(Store([_state(heartbeat=LAST_HEARTBEAT, complete=complete)]), fleet)
    assert fleet.destroyed == [] and _outcome(report)["reason"] == "not_published_within_longest_deadline"


def test_a_machine_with_no_usable_start_time_is_unknown_and_kept():
    fleet = Fleet([_machine(age=None)])
    report = _run(Store([_state(heartbeat=LAST_HEARTBEAT, complete=True)]), fleet)
    assert fleet.destroyed == [] and _outcome(report)["outcome"] == "unknown"
    assert report["status"] == "unknown"


def test_a_completeness_flag_that_is_not_a_boolean_is_unreadable():
    report = _run(Store([{**_state(), "deadlines_complete": "yes"}]), Fleet([_machine()]))
    assert report["status"] == "unknown" and report["blockers"] == ["host_state_completeness_invalid"]
