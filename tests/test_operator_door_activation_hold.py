"""A deploy waits while a scene-configuration activation is live on the running release."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/activation_hold.py

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door import activation_hold as hold
from operator_door import parked_settle, spool_runner

ACTIVE = "a" * 40
OTHER = "b" * 40
DEPLOY_ID = "20261031T200000Z-deploy-0000abcd"


@pytest.fixture()
def roots(tmp_path: Path) -> dict[str, Path]:
    queue = tmp_path / "task-evaluation-launch-activations"
    for name in ("pending", "processing", "results", "prepared"):
        (queue / name).mkdir(parents=True)
    parked = tmp_path / "parked"
    parked.mkdir()
    return {"queue": queue, "progression": tmp_path / "task-evaluation-configured-controls", "parked": parked}


def _envelope(path: Path, *, commit: str = ACTIVE, lane: str = hold.LANE, activation_id: str = "act-1") -> Path:
    path.write_text(json.dumps({"schema_version": "task_evaluation_launch_activation_envelope.v1",
                                "request": {"activation_id": activation_id, "lane": lane,
                                            "expected_production_commit": commit}}), encoding="utf-8")
    return path


def _decide(roots: dict[str, Path], *, held: set[int] | None = None, override: str = "",
            now: float | None = None, active: str = ACTIVE) -> tuple[int, str]:
    return hold.decide(queue_root=roots["queue"], progression_root=roots["progression"], active=active,
                       parked_dir=roots["parked"], override=override,
                       now=time.time() if now is None else now,
                       held=hold.locked_inodes() if held is None else held)


@pytest.mark.skipif(not Path("/proc/locks").exists(), reason="needs Linux /proc/locks")
def test_a_claim_whose_worker_holds_its_lock_parks_the_deploy(roots) -> None:
    claim = _envelope(roots["queue"] / "processing" / "act-1-abc.json")
    with claim.open("rb") as lease:
        # The worker's own lease: an exclusive flock on the claim inode.
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, reason = _decide(roots)
    assert code == hold.EXIT_HOLD
    assert reason == "scene_configuration_activation_live:processing:act-1"
    record = json.loads((roots["parked"] / hold.HOLD_FILE).read_text(encoding="utf-8"))
    assert record["active_release"] == ACTIVE
    assert record["cap_at_epoch"] - record["first_held_at_epoch"] == hold.CAP_SECONDS
    assert 0 < record["recheck_after_epoch"] - time.time() <= hold.RECHECK_SECONDS


def test_a_stuck_claim_without_its_lock_lets_the_deploy_go(roots) -> None:
    _envelope(roots["queue"] / "processing" / "act-1-abc.json")
    code, reason = _decide(roots, held=set())
    assert code == hold.EXIT_DEPLOY and reason == "no_live_scene_configuration_activation"
    assert not (roots["parked"] / hold.HOLD_FILE).exists()


def test_another_releases_claim_lets_the_deploy_go(roots) -> None:
    claim = _envelope(roots["queue"] / "processing" / "act-1-abc.json", commit=OTHER)
    _envelope(roots["queue"] / "pending" / "act-2-abc.json", commit=OTHER, activation_id="act-2")
    # Its worker refuses it anyway (launch_activation_worker_source_commit_mismatch).
    code, _ = _decide(roots, held={claim.stat().st_ino})
    assert code == hold.EXIT_DEPLOY


def test_another_lanes_claim_lets_the_deploy_go(roots) -> None:
    claim = _envelope(roots["queue"] / "processing" / "act-1-abc.json", lane="task_evaluation_native")
    assert _decide(roots, held={claim.stat().st_ino})[0] == hold.EXIT_DEPLOY


def test_the_override_lets_the_deploy_go(roots) -> None:
    claim = _envelope(roots["queue"] / "processing" / "act-1-abc.json")
    (roots["parked"] / hold.HOLD_FILE).write_text("{}", encoding="utf-8")
    code, reason = _decide(roots, held={claim.stat().st_ino}, override="privacy")
    assert code == hold.EXIT_DEPLOY and reason == "activation_hold_overridden:privacy"
    assert not (roots["parked"] / hold.HOLD_FILE).exists()


def test_the_cap_lets_the_deploy_go(roots) -> None:
    claim = _envelope(roots["queue"] / "processing" / "act-1-abc.json")
    held = {claim.stat().st_ino}
    start = time.time()
    assert _decide(roots, held=held, now=start)[0] == hold.EXIT_HOLD
    # Still held just before the cap, from the same first hold.
    assert _decide(roots, held=held, now=start + hold.CAP_SECONDS - 60)[0] == hold.EXIT_HOLD
    code, reason = _decide(roots, held=held, now=start + hold.CAP_SECONDS)
    assert code == hold.EXIT_DEPLOY and reason.startswith("activation_hold_cap_reached:")
    assert not (roots["parked"] / hold.HOLD_FILE).exists()


def test_the_cap_is_the_activation_time_limit_plus_a_submit_window() -> None:
    assert hold.CAP_SECONDS == 255 * 60 + 30 * 60
    # A parked deploy expires at 8 h; the hold must end well before that.
    assert hold.CAP_SECONDS + 3 * 3600 <= spool_runner._PARKED_MAX_AGE_SECONDS + 3600


def test_a_hold_from_another_release_does_not_shorten_this_one(roots) -> None:
    claim = _envelope(roots["queue"] / "processing" / "act-1-abc.json")
    now = time.time()
    (roots["parked"] / hold.HOLD_FILE).write_text(json.dumps({
        "schema": hold.HOLD_SCHEMA, "active_release": OTHER, "first_held_at_epoch": now - hold.CAP_SECONDS,
    }), encoding="utf-8")
    assert _decide(roots, held={claim.stat().st_ino}, now=now)[0] == hold.EXIT_HOLD


def test_a_pending_activation_parks_the_deploy(roots) -> None:
    _envelope(roots["queue"] / "pending" / "act-1-abc.json")
    code, reason = _decide(roots, held=set())
    assert code == hold.EXIT_HOLD and reason.endswith("pending:act-1")


def _materialized(roots: dict[str, Path], *, age: float = 0.0, commit: str = ACTIVE) -> Path:
    path = roots["queue"] / "results" / "act-1-abc.json"
    path.write_text(json.dumps({"status": hold.MATERIALIZED_STATUS, "lane": hold.LANE, "source_commit": commit,
                                "activation_id": "act-1", "preparation_id": "prep-1"}), encoding="utf-8")
    if age:
        moment = time.time() - age
        os.utime(path, (moment, moment))
    return path


def test_a_finished_activation_whose_launch_is_not_submitted_parks_the_deploy(roots) -> None:
    _materialized(roots)
    code, reason = _decide(roots, held=set())
    assert code == hold.EXIT_HOLD and reason.endswith("awaiting_launch_submission:act-1")


def test_a_finished_activation_whose_launch_is_queued_lets_the_deploy_go(roots) -> None:
    _materialized(roots)
    state = roots["progression"] / "scene-configuration-activations" / "prep-1"
    state.mkdir(parents=True)
    (state / "launch_progression.json").write_text(
        json.dumps({"status": hold.LAUNCH_QUEUED_STATUS}), encoding="utf-8")
    assert _decide(roots, held=set())[0] == hold.EXIT_DEPLOY


def test_a_finished_activation_past_the_submit_window_lets_the_deploy_go(roots) -> None:
    _materialized(roots, age=hold.SUBMIT_WINDOW_SECONDS + 60)
    assert _decide(roots, held=set())[0] == hold.EXIT_DEPLOY


def test_an_unknown_running_release_holds_nothing(roots) -> None:
    _envelope(roots["queue"] / "pending" / "act-1-abc.json")
    assert _decide(roots, held=set(), active="")[0] == hold.EXIT_DEPLOY


def test_unreadable_queue_files_do_not_hold_the_deploy(roots) -> None:
    (roots["queue"] / "pending" / "junk.json").write_text("not json", encoding="utf-8")
    (roots["queue"] / "pending" / "link.json").symlink_to(roots["queue"] / "pending" / "junk.json")
    assert _decide(roots, held=set())[0] == hold.EXIT_DEPLOY


def test_proc_locks_counts_holders_not_waiters(tmp_path: Path) -> None:
    locks = tmp_path / "locks"
    locks.write_text("1: FLOCK  ADVISORY  WRITE 100 fd:01:4242 0 EOF\n"
                     "1: -> FLOCK  ADVISORY  WRITE 101 fd:01:5151 0 EOF\n"
                     "2: POSIX  ADVISORY  WRITE 102 fd:01:6161 0 EOF\n", encoding="ascii")
    assert hold.locked_inodes(str(locks)) == {4242}
    assert hold.locked_inodes(str(tmp_path / "missing")) == set()


def test_a_self_parked_deploy_is_a_record_the_runner_and_settle_accept(roots) -> None:
    assert hold.park(parked_dir=roots["parked"], request_id=DEPLOY_ID, commit=ACTIVE, wait_for_idle=True,
                     active=OTHER, now=123.0) == hold.EXIT_DEPLOY
    path = roots["parked"] / f"{DEPLOY_ID}.json"
    assert parked_settle._read_record(path) == parked_settle.Parked(DEPLOY_ID, ACTIVE, OTHER)
    request_id, request, record = spool_runner._parked_request(path)
    assert request_id == DEPLOY_ID and request["commit"] == ACTIVE and request["wait_for_idle"] is True
    assert record["parked_at_epoch"] == 123.0
    assert hold.PARKED_SCHEMA == spool_runner._PARKED_SCHEMA == parked_settle.PARKED_SCHEMA


def test_park_refuses_past_the_parked_bound(roots, monkeypatch) -> None:
    monkeypatch.setattr(hold, "PARKED_MAX", 1)
    assert hold.park(parked_dir=roots["parked"], request_id=DEPLOY_ID, commit=ACTIVE, wait_for_idle=False,
                     active="", now=1.0) == hold.EXIT_DEPLOY
    other = "20261031T200100Z-deploy-0000abce"
    assert hold.park(parked_dir=roots["parked"], request_id=other, commit=ACTIVE, wait_for_idle=False,
                     active="", now=1.0) == hold.EXIT_CANNOT_PARK


def test_the_door_and_the_check_accept_the_same_override_reasons() -> None:
    from operator_door.requests import ACTIVATION_HOLD_OVERRIDE

    assert hold.OVERRIDE_REASON.pattern == ACTIVATION_HOLD_OVERRIDE.pattern
    for reason in ("fixes_live_activation", "money", "privacy", "rights", "pr-3206:hotfix.v2"):
        assert hold.OVERRIDE_REASON.fullmatch(reason)


def test_a_fix_for_the_live_activation_overrides_the_hold(roots) -> None:
    claim = _envelope(roots["queue"] / "processing" / "act-1-abc.json")
    code, reason = _decide(roots, held={claim.stat().st_ino}, override="fixes_live_activation")
    assert code == hold.EXIT_DEPLOY and reason == "activation_hold_overridden:fixes_live_activation"


def _hold_activation_path(holds_dir: Path, status: str = "active") -> None:
    holds_dir.mkdir(parents=True, exist_ok=True)
    (holds_dir / f"{hold.ACTIVATION_PATH_UNIT}.json").write_text(
        json.dumps({"unit": hold.ACTIVATION_PATH_UNIT, "status": status}), encoding="utf-8")


def test_pending_work_nothing_can_claim_does_not_hold_the_deploy(roots, tmp_path: Path) -> None:
    _envelope(roots["queue"] / "pending" / "act-1-abc.json")
    holds_dir = tmp_path / "holds"
    _hold_activation_path(holds_dir)
    assert hold.activation_path_held(holds_dir) is True
    code, reason = hold.decide(queue_root=roots["queue"], progression_root=roots["progression"], active=ACTIVE,
                               parked_dir=roots["parked"], override="", now=time.time(), held=set(),
                               pending_claimable=False)
    assert code == hold.EXIT_DEPLOY and reason == "no_live_scene_configuration_activation"


def test_a_held_path_unit_does_not_release_a_claim_already_running(roots) -> None:
    claim = _envelope(roots["queue"] / "processing" / "act-1-abc.json")
    code, _ = hold.decide(queue_root=roots["queue"], progression_root=roots["progression"], active=ACTIVE,
                          parked_dir=roots["parked"], override="", now=time.time(), held={claim.stat().st_ino},
                          pending_claimable=False)
    assert code == hold.EXIT_HOLD


@pytest.mark.parametrize("status", ["released", "releasing", "expired_released"])
def test_only_an_active_hold_stops_the_path_unit(tmp_path: Path, status: str) -> None:
    _hold_activation_path(tmp_path / "holds", status)
    assert hold.activation_path_held(tmp_path / "holds") is False
    assert hold.activation_path_held(tmp_path / "missing") is False
    assert hold.activation_path_held(None) is False


def test_a_hold_reports_when_to_retry_and_when_the_cap_lets_it_go(roots) -> None:
    _envelope(roots["queue"] / "pending" / "act-1-abc.json")
    start = 1_800_000_000.0
    times: dict[str, float] = {}
    assert hold.decide(queue_root=roots["queue"], progression_root=roots["progression"], active=ACTIVE,
                       parked_dir=roots["parked"], override="", now=start, held=set(),
                       times=times)[0] == hold.EXIT_HOLD
    assert times == {"retry_after_epoch": start + hold.RECHECK_SECONDS, "cap_at_epoch": start + hold.CAP_SECONDS}
    # A later retry keeps the first hold's cap, so retrying never pushes the cap back.
    later: dict[str, float] = {}
    hold.decide(queue_root=roots["queue"], progression_root=roots["progression"], active=ACTIVE,
                parked_dir=roots["parked"], override="", now=start + 600, held=set(), times=later)
    assert later["cap_at_epoch"] == start + hold.CAP_SECONDS
    assert hold.retry_line(times) == (
        "2027-01-15T08:05:00+00:00 1800000300 2027-01-15T12:45:00+00:00 1800017100")


def test_no_retry_times_when_the_deploy_goes(roots) -> None:
    times: dict[str, float] = {}
    assert _decide(roots, held=set())[0] == hold.EXIT_DEPLOY
    assert hold.decide(queue_root=roots["queue"], progression_root=roots["progression"], active=ACTIVE,
                       parked_dir=roots["parked"], override="", now=time.time(), held=set(),
                       times=times)[0] == hold.EXIT_DEPLOY
    assert times == {}
