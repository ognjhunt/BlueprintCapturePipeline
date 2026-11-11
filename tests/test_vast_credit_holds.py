"""G18 / WP-09: Vast credit admitted across launch slots never passes the balance less the reserve."""

import json
import os
import threading
import time

import pytest

from blueprint_pipeline import provider_credit_admission as credit
from blueprint_pipeline import vast_credit_holds as holds


@pytest.fixture()
def balance(monkeypatch, tmp_path):
    """A $11 balance, observed fresh on every call; holds kept under a temporary root."""

    monkeypatch.setenv(holds.HOLD_ROOT_ENV, str(tmp_path / "locks"))
    monkeypatch.setenv(credit.ENABLED_ENV, "true")
    monkeypatch.setenv(credit.RESERVE_ENV, "1")
    amount = {"usd": 11.0}
    original = credit.observe_vast_credit
    monkeypatch.setattr(credit, "observe_vast_credit", lambda **kw: original(
        api_key="k", request=lambda **_: (200, {"credit": amount["usd"]})))
    return amount


def _teardown(job_dir, continuing):
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / holds.TEARDOWN_FILENAME).write_text(json.dumps({"continuing_spend_from_this_run": continuing}))


def test_three_launches_together_against_a_balance_that_fits_two(balance, tmp_path):
    barrier = threading.Barrier(3)
    results = {}

    def launch(index):
        job_dir = tmp_path / f"job{index}"
        job_dir.mkdir()
        barrier.wait()
        results[index] = credit.record_vast_credit_admission(job_dir, "k", 5.0)

    threads = [threading.Thread(target=launch, args=(index,)) for index in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    statuses = sorted(row["status"] for row in results.values())
    assert statuses == ["admitted", "admitted", "blocked"]
    refused = next(row for row in results.values() if row["status"] == "blocked")
    assert refused["blockers"] == ["provider_credit_insufficient"]
    assert refused["held_by_other_launches_usd"] == 10.0
    admitted = [row for row in results.values() if row["status"] == "admitted"]
    assert all(row["credit_reserved"] is True for row in admitted)
    assert sum(row["required_usd"] for row in admitted) <= 11.0 - 1.0


def test_teardown_written_after_the_hold_closes_it_and_an_older_one_does_not(balance, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _teardown(first, False)  # an earlier attempt's manifest in the same directory
    old = time.time() - 60
    os.utime(first / holds.TEARDOWN_FILENAME, (old, old))
    second.mkdir()
    assert credit.record_vast_credit_admission(first, "k", 5.0)["status"] == "admitted"
    assert credit.record_vast_credit_admission(second, "k", 5.0)["status"] == "admitted"
    third = tmp_path / "third"
    third.mkdir()
    assert credit.record_vast_credit_admission(third, "k", 5.0)["blockers"] == ["provider_credit_insufficient"]
    _teardown(first, True)  # retained under a watchdog: still billable, still held
    assert credit.record_vast_credit_admission(third, "k", 5.0)["status"] == "blocked"
    _teardown(first, False)  # destroyed: the hold closes by itself
    assert credit.record_vast_credit_admission(third, "k", 5.0)["status"] == "admitted"


def test_read_only_check_subtracts_open_holds_without_taking_one(balance, tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    assert credit.record_vast_credit_admission(job, "k", 5.0)["status"] == "admitted"
    for _ in range(2):
        result = credit.configured_vast_credit_admission(api_key="k", required_usd=5.0)
        assert result["status"] == "admitted" and result["credit_reserved"] is False
    assert credit.configured_vast_credit_admission(api_key="k", required_usd=5.01)["status"] == "blocked"


def test_same_job_relaunch_replaces_its_hold_and_release_closes_it(balance, tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    for _ in range(3):
        assert credit.record_vast_credit_admission(job, "k", 5.0)["status"] == "admitted"
    assert holds.open_holds_usd() == (5.0, [])
    assert holds.release_hold(job) is True
    assert holds.open_holds_usd() == (0.0, [])


def test_hold_without_teardown_expires_after_a_day_and_says_so(balance, tmp_path):
    job, other = tmp_path / "job", tmp_path / "other"
    job.mkdir()
    other.mkdir()
    assert credit.record_vast_credit_admission(job, "k", 10.0)["status"] == "admitted"
    later = time.time() + holds.MAX_HOLD_AGE_SECONDS + 1
    result = holds.admit_and_hold(lambda held: credit.credit_admission(
        credit.observe_vast_credit(), required_usd=10.0, held_usd=held, now=time.time()),
        job_dir=other, required_usd=10.0, now=later)
    assert result["status"] == "admitted"
    assert result["expired_holds"][0]["state"] == "expired_without_teardown"


def test_unreadable_holds_refuse_new_credit(balance, tmp_path):
    root = tmp_path / "locks"
    root.mkdir()
    (root / holds.HOLDS_FILENAME).write_text("not json")
    job = tmp_path / "job"
    job.mkdir()
    assert credit.record_vast_credit_admission(job, "k", 1.0)["blockers"] == [holds.HOLDS_UNREADABLE]
    assert credit.configured_vast_credit_admission(api_key="k", required_usd=1.0)["blockers"] == [
        holds.HOLDS_UNREADABLE]


@pytest.mark.parametrize("lock_file", [None, "locks/vast_paid_launch.lock"])
def test_holds_live_beside_the_launch_slot_locks(monkeypatch, tmp_path, lock_file):
    from blueprint_pipeline.vast_launch_slots import _vast_launch_lock_path

    monkeypatch.delenv(holds.HOLD_ROOT_ENV, raising=False)
    monkeypatch.setenv("VAST_API_KEY_FILE", str(tmp_path / "secrets" / "vast_api_key"))
    if lock_file is None:
        monkeypatch.delenv("VAST_LAUNCH_LOCK_FILE", raising=False)
    else:
        monkeypatch.setenv("VAST_LAUNCH_LOCK_FILE", str(tmp_path / lock_file))
    assert holds.hold_root() == _vast_launch_lock_path().parent


def test_an_unreadable_holds_file_older_than_any_hold_is_set_aside_and_admission_proceeds(balance, tmp_path):
    root = tmp_path / "locks"
    root.mkdir()
    path = root / holds.HOLDS_FILENAME
    path.write_text("not json")
    old = time.time() - holds.MAX_HOLD_AGE_SECONDS - 60
    os.utime(path, (old, old))
    job = tmp_path / "job"
    job.mkdir()
    result = credit.record_vast_credit_admission(job, "k", 5.0)
    assert result["status"] == "admitted" and result["credit_reserved"] is True
    aside = result["holds_file_set_aside"]
    assert aside.startswith(str(path) + ".unreadable-") and (root / aside.rsplit("/", 1)[1]).read_text() == "not json"
    assert holds.open_holds_usd() == (5.0, [])  # a fresh holds file, holding this launch


def test_a_young_unreadable_holds_file_keeps_refusing_in_both_checks(balance, tmp_path):
    root = tmp_path / "locks"
    root.mkdir()
    path = root / holds.HOLDS_FILENAME
    path.write_text("not json")
    young = time.time() - holds.MAX_HOLD_AGE_SECONDS + 60
    os.utime(path, (young, young))
    job = tmp_path / "job"
    job.mkdir()
    assert credit.record_vast_credit_admission(job, "k", 1.0)["blockers"] == [holds.HOLDS_UNREADABLE]
    assert holds.open_holds_usd() == (None, [holds.HOLDS_UNREADABLE])
    assert path.read_text() == "not json" and list(root.glob("*.unreadable-*")) == []


def test_the_read_only_check_also_sets_an_old_unreadable_file_aside(balance, tmp_path):
    root = tmp_path / "locks"
    root.mkdir()
    path = root / holds.HOLDS_FILENAME
    path.write_text("{}")
    old = time.time() - holds.MAX_HOLD_AGE_SECONDS - 60
    os.utime(path, (old, old))
    assert credit.configured_vast_credit_admission(api_key="k", required_usd=5.0)["status"] == "admitted"
    assert not path.exists() and len(list(root.glob("*.unreadable-*"))) == 1
