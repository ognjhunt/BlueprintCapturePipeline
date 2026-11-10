"""A deploy waits while a paid stage of a website job is live on the running release."""

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


def test_the_cap_lets_the_deploy_go(roots) -> None:
    claim = _envelope(roots["queue"] / "processing" / "act-1-abc.json")
    held = {claim.stat().st_ino}
    start = time.time()
    assert _decide(roots, held=held, now=start)[0] == hold.EXIT_HOLD
    # Still held just before the cap, from the same first hold.
    assert _decide(roots, held=held, now=start + hold.CAP_SECONDS - 60)[0] == hold.EXIT_HOLD
    code, reason = _decide(roots, held=held, now=start + hold.CAP_SECONDS)
    assert code == hold.EXIT_DEPLOY and reason.startswith("activation_hold_cap_reached:")


def test_a_late_retry_after_the_cap_still_deploys(roots) -> None:
    """Codex #3206: a retry long after deploys_by must not buy a second 285 min hold."""
    claim = _envelope(roots["queue"] / "processing" / "act-1-abc.json")
    held = {claim.stat().st_ino}
    start = time.time() - hold.CAP_SECONDS - 3 * 3600
    assert _decide(roots, held=held, now=start)[0] == hold.EXIT_HOLD
    code, reason = _decide(roots, held=held, now=start + hold.CAP_SECONDS + 3 * 3600)
    assert code == hold.EXIT_DEPLOY and reason.startswith("activation_hold_cap_reached:")


def test_a_deploy_after_a_capped_one_failed_keeps_the_spent_cap(roots) -> None:
    """Codex #3206: the capped deploy failed before switching; the next one goes too."""
    claim = _envelope(roots["queue"] / "processing" / "act-1-abc.json")
    held = {claim.stat().st_ino}
    start = time.time() - hold.CAP_SECONDS - 600
    _decide(roots, held=held, now=start)
    assert _decide(roots, held=held, now=start + hold.CAP_SECONDS)[0] == hold.EXIT_DEPLOY
    # Still the same release (the deploy never switched), and the same activation.
    assert _decide(roots, held=held, now=start + hold.CAP_SECONDS + 300)[0] == hold.EXIT_DEPLOY
    record = json.loads((roots["parked"] / hold.HOLD_FILE).read_text(encoding="utf-8"))
    assert record["first_held_at_epoch"] == start and record["recheck_after_epoch"] <= time.time()


def test_pending_work_becoming_the_running_claim_is_the_same_hold(roots) -> None:
    pending = _envelope(roots["queue"] / "pending" / "act-1-abc.json")
    start = time.time() - 3600
    _decide(roots, held=set(), now=start)
    claim = pending.rename(roots["queue"] / "processing" / pending.name)
    _decide(roots, held={claim.stat().st_ino}, now=start + 1800)
    record = json.loads((roots["parked"] / hold.HOLD_FILE).read_text(encoding="utf-8"))
    assert record["first_held_at_epoch"] == start


def test_new_work_that_shares_nothing_with_the_old_hold_starts_its_own(roots) -> None:
    claim = _envelope(roots["queue"] / "processing" / "act-1-abc.json")
    start = time.time() - hold.CAP_SECONDS - 60
    _decide(roots, held={claim.stat().st_ino}, now=start)
    claim.unlink()
    _envelope(roots["queue"] / "pending" / "act-2-abc.json", activation_id="act-2")
    now = time.time()
    assert _decide(roots, held=set(), now=now)[0] == hold.EXIT_HOLD
    record = json.loads((roots["parked"] / hold.HOLD_FILE).read_text(encoding="utf-8"))
    assert record["first_held_at_epoch"] == now


def test_a_check_with_nothing_live_ends_the_hold(roots) -> None:
    claim = _envelope(roots["queue"] / "processing" / "act-1-abc.json")
    _decide(roots, held={claim.stat().st_ino})
    assert _decide(roots, held=set())[0] == hold.EXIT_DEPLOY
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


def test_old_results_cannot_crowd_out_a_recent_one(roots) -> None:
    """Codex #3206: more than MAX_SCAN historical results, all sorting before the new one."""
    old = time.time() - 30 * 86400
    for index in range(hold.MAX_SCAN + 88):
        path = roots["queue"] / "results" / f"a-{index:04d}.json"
        path.write_text("{}", encoding="utf-8")
        os.utime(path, (old, old))
    recent = _materialized(roots)
    recent.rename(roots["queue"] / "results" / "z-recent.json")
    code, reason = _decide(roots, held=set())
    assert code == hold.EXIT_HOLD and reason.endswith("awaiting_launch_submission:act-1")


def test_the_scan_bound_keeps_the_newest_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hold, "MAX_SCAN", 2)
    for index, age in enumerate((300, 100, 200)):
        path = tmp_path / f"r{index}.json"
        path.write_text("{}", encoding="utf-8")
        moment = time.time() - age
        os.utime(path, (moment, moment))
    (tmp_path / "skip.txt").write_text("", encoding="utf-8")
    (tmp_path / "link.json").symlink_to(tmp_path / "r0.json")
    assert [path.name for path in hold._json_files(tmp_path)] == ["r1.json", "r2.json"]
    assert [path.name for path in hold._json_files(tmp_path, modified_since=time.time() - 150)] == ["r1.json"]


# --- Every paid stage of a website job (WP-17): launches, paid-launch slots, screening ---


@pytest.fixture()
def state(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "state"
    queue = root / "task-evaluation-launch-activations"
    for name in ("pending", "processing", "results"):
        (queue / name).mkdir(parents=True)
    launches = root / hold.LAUNCH_QUEUE
    for name in ("pending", "processing", "completed", "blocked"):
        (launches / name).mkdir(parents=True)
    (launches / hold.DISPATCHER_LOCK).write_text("", encoding="utf-8")
    (root / hold.PROVIDER_LOCKS).mkdir()
    (root / hold.SCREENING_ROOT / "plans").mkdir(parents=True)
    parked = tmp_path / "parked"
    parked.mkdir()
    proc = tmp_path / "proc"
    proc.mkdir()
    return {"root": root, "queue": queue, "progression": root / "task-evaluation-configured-controls",
            "launches": launches, "locks": root / hold.PROVIDER_LOCKS,
            "screening": root / hold.SCREENING_ROOT, "parked": parked, "proc": proc}


def _all(state: dict[str, Path], *, held: set[int] | None = None, holders: dict[int, set[int]] | None = None,
         override: str = "", now: float | None = None, active: str = ACTIVE, request_id: str = "",
         launch_pending_claimable: bool = True, times: dict[str, float] | None = None) -> tuple[int, str]:
    holders = holders or {}
    return hold.decide(
        queue_root=state["queue"], progression_root=state["progression"], active=active,
        parked_dir=state["parked"], override=override, now=time.time() if now is None else now,
        held=set(holders) | (held or set()), times=times,
        launch_queue_root=state["launches"], launch_pending_claimable=launch_pending_claimable,
        lock_dir=state["locks"], holders=holders, proc_root=state["proc"],
        screening_root=state["screening"], request_id=request_id)


def _record(state: dict[str, Path]) -> dict:
    return json.loads((state["parked"] / hold.HOLD_FILE).read_text(encoding="utf-8"))


def _submitted(state: dict[str, Path], *, commit: str = ACTIVE, status: str = hold.LAUNCH_QUEUED_STATUS,
               launch_id: str = "act-1-launch", preparation: str = "prep-1") -> None:
    directory = state["progression"] / "scene-configuration-activations" / preparation
    directory.mkdir(parents=True, exist_ok=True)
    (directory / hold.LAUNCH_STATE).write_text(json.dumps({
        "status": status, "expected_production_commit": commit, "launch_id": launch_id,
        "activation_id": "act-1", "preparation_id": preparation}), encoding="utf-8")


def _launch_request(state: dict[str, Path], where: str, *, launch_id: str = "act-1-launch") -> Path:
    path = state["launches"] / where / f"{launch_id}-0123456789abcdef.json"
    path.write_text(json.dumps({"launch_id": launch_id, "run_id": launch_id}), encoding="utf-8")
    return path


def _dispatcher_alive(state: dict[str, Path]) -> set[int]:
    return {(state["launches"] / hold.DISPATCHER_LOCK).stat().st_ino}


def test_a_submitted_launch_pending_for_this_release_parks_the_deploy(state) -> None:
    _submitted(state)
    _launch_request(state, "pending")
    code, reason = _all(state)
    assert code == hold.EXIT_HOLD and reason == "paid_stage_live:launch_pending:act-1-launch"
    record = _record(state)
    assert record["stages"] == [hold.STAGE_LAUNCH]
    assert record["live"] == [{"stage": hold.STAGE_LAUNCH, "state": "launch_pending", "id": "act-1-launch",
                               "activation_id": "act-1"}]
    # The existing fields stay, with their meaning.
    assert record["activation_ids"] == ["act-1"] and record["cap_at_epoch"] - record["first_held_at_epoch"] == hold.CAP_SECONDS


def test_a_launch_claimed_by_a_live_dispatcher_parks_the_deploy(state) -> None:
    _submitted(state)
    _launch_request(state, "processing")
    code, reason = _all(state, held=_dispatcher_alive(state))
    assert code == hold.EXIT_HOLD and reason.endswith("launch_processing:act-1-launch")


def test_a_launch_claim_no_dispatcher_owns_lets_the_deploy_go(state) -> None:
    _submitted(state)
    _launch_request(state, "processing")
    assert _all(state, held=set())[0] == hold.EXIT_DEPLOY


def test_a_finished_launch_lets_the_deploy_go(state) -> None:
    _submitted(state)
    request = _launch_request(state, "processing")
    assert _all(state, held=_dispatcher_alive(state))[0] == hold.EXIT_HOLD
    request.rename(state["launches"] / "completed" / request.name)
    assert _all(state, held=_dispatcher_alive(state))[0] == hold.EXIT_DEPLOY
    assert not (state["parked"] / hold.HOLD_FILE).exists()


def test_a_launch_submitted_for_another_release_lets_the_deploy_go(state) -> None:
    _submitted(state, commit=OTHER)
    _launch_request(state, "pending")
    assert _all(state)[0] == hold.EXIT_DEPLOY


def test_a_launch_never_submitted_through_progression_lets_the_deploy_go(state) -> None:
    _submitted(state, status="scene_configuration_activation_queued")
    _launch_request(state, "pending")
    _launch_request(state, "pending", launch_id="unrelated-launch")
    assert _all(state)[0] == hold.EXIT_DEPLOY


def test_unreadable_launch_files_do_not_hold_the_deploy(state) -> None:
    directory = state["progression"] / "scene-configuration-activations" / "prep-1"
    directory.mkdir(parents=True)
    (directory / hold.LAUNCH_STATE).write_text("not json", encoding="utf-8")
    _submitted(state, preparation="prep-2")
    (state["launches"] / "pending" / "act-1-launch-0123456789abcdef.json").write_text("[1]", encoding="utf-8")
    (state["launches"] / "pending" / "act-1-launch-fedcba9876543210.json").symlink_to(
        state["launches"] / "pending" / "act-1-launch-0123456789abcdef.json")
    assert _all(state)[0] == hold.EXIT_DEPLOY


def test_a_pending_launch_nothing_can_claim_does_not_hold_the_deploy(state) -> None:
    _submitted(state)
    _launch_request(state, "pending")
    assert _all(state, launch_pending_claimable=False)[0] == hold.EXIT_DEPLOY


def _slot(state: dict[str, Path], name: str = "vast_paid_launch.lock") -> int:
    path = state["locks"] / name
    path.write_text(json.dumps({"pid": 4242, "job_dir": "/x"}), encoding="utf-8")
    return path.stat().st_ino


def _runs_from(state: dict[str, Path], pid: int, commit: str) -> None:
    (state["proc"] / str(pid)).mkdir()
    (state["proc"] / str(pid) / "cwd").symlink_to(
        f"/opt/blueprint/task-evaluation-control-plane-releases/{commit}")


def test_a_paid_launch_slot_held_by_this_release_parks_the_deploy(state) -> None:
    inode = _slot(state)
    _runs_from(state, 4242, ACTIVE)
    code, reason = _all(state, holders={inode: {4242}})
    assert code == hold.EXIT_HOLD and reason == "paid_stage_live:paid_launch_lock_held:vast_paid_launch.lock"
    assert _record(state)["stages"] == [hold.STAGE_PAID_LOCK]


def test_a_released_slot_lets_the_deploy_go(state) -> None:
    inode = _slot(state)
    _runs_from(state, 4242, ACTIVE)
    assert _all(state, holders={inode: {4242}})[0] == hold.EXIT_HOLD
    # The run ended: the kernel dropped its flock. The file and its old record stay.
    assert _all(state, holders={})[0] == hold.EXIT_DEPLOY


def test_a_slot_held_by_another_release_or_an_unknown_process_lets_the_deploy_go(state) -> None:
    inode = _slot(state)
    _runs_from(state, 4242, OTHER)
    assert _all(state, holders={inode: {4242}})[0] == hold.EXIT_DEPLOY
    # No /proc entry for the pid: its release is unknown.
    assert _all(state, holders={inode: {777}})[0] == hold.EXIT_DEPLOY


def test_the_deploy_gate_does_not_hold_the_deploy(state) -> None:
    inode = _slot(state, "vast_paid_launch.gate.lock")
    _runs_from(state, 4242, ACTIVE)
    assert _all(state, holders={inode: {4242}})[0] == hold.EXIT_DEPLOY


def test_a_slot_that_is_a_symlink_does_not_hold_the_deploy(state, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.lock"
    target.write_text("", encoding="utf-8")
    (state["locks"] / "vast_paid_launch.lock").symlink_to(target)
    _runs_from(state, 4242, ACTIVE)
    assert _all(state, holders={target.stat().st_ino: {4242}})[0] == hold.EXIT_DEPLOY


@pytest.mark.skipif(not Path("/proc/locks").exists(), reason="needs Linux /proc/locks")
def test_a_real_flock_from_this_release_parks_the_deploy(state, tmp_path: Path, monkeypatch) -> None:
    release = tmp_path / "releases" / ACTIVE
    release.mkdir(parents=True)
    monkeypatch.chdir(release)
    slot = state["locks"] / "vast_paid_launch.slot1.lock"
    slot.write_text("", encoding="utf-8")
    with slot.open("rb") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        holders = hold.lock_holders()
        assert os.getpid() in holders[slot.stat().st_ino]
        assert hold.process_release(os.getpid(), Path("/proc")) == ACTIVE
        code, _ = hold.decide(queue_root=state["queue"], progression_root=state["progression"], active=ACTIVE,
                              parked_dir=state["parked"], override="", now=time.time(), held=set(holders),
                              lock_dir=state["locks"], holders=holders, proc_root=Path("/proc"))
    assert code == hold.EXIT_HOLD


def _screening(state: dict[str, Path], *, status: str = "running", commit: str | None = ACTIVE,
               campaign: str = "camp-1", candidate: str = "cand-1") -> None:
    (state["screening"] / hold.SCREENING_LEDGER).write_text(json.dumps({
        "schema_version": "blueprint.beta_policy_screening_ledger.v1",
        "campaigns": {campaign: {"site_id": "site-1", "candidates": {candidate: {"status": status}}}}}),
        encoding="utf-8")
    spec = None if commit is None else {"implementation_commit": commit}
    (state["screening"] / "plans" / f"{campaign}.json").write_text(json.dumps({
        "campaign_id": campaign, "candidates": [{"candidate_id": candidate, "execution_spec": spec}]}),
        encoding="utf-8")


@pytest.mark.parametrize("status", ["launching", "running"])
def test_a_screening_candidate_launching_or_running_parks_the_deploy(state, status: str) -> None:
    _screening(state, status=status)
    code, reason = _all(state)
    assert code == hold.EXIT_HOLD and reason == f"paid_stage_live:screening_{status}:camp-1/cand-1"
    assert _record(state)["stages"] == [hold.STAGE_SCREENING]


@pytest.mark.parametrize("status", ["completed", "blocked", "pending", "reconciliation_required"])
def test_a_screening_candidate_not_launching_or_running_lets_the_deploy_go(state, status: str) -> None:
    _screening(state, status=status)
    assert _all(state)[0] == hold.EXIT_DEPLOY


def test_a_screening_candidate_ending_ends_the_hold(state) -> None:
    _screening(state)
    assert _all(state)[0] == hold.EXIT_HOLD
    _screening(state, status="completed")
    assert _all(state)[0] == hold.EXIT_DEPLOY
    assert not (state["parked"] / hold.HOLD_FILE).exists()


def test_a_screening_candidate_of_another_release_lets_the_deploy_go(state) -> None:
    _screening(state, commit=OTHER)
    assert _all(state)[0] == hold.EXIT_DEPLOY
    _screening(state, commit=None)
    assert _all(state)[0] == hold.EXIT_DEPLOY


def test_unreadable_screening_files_do_not_hold_the_deploy(state) -> None:
    _screening(state)
    (state["screening"] / "plans" / "camp-1.json").write_text("garbage", encoding="utf-8")
    assert _all(state)[0] == hold.EXIT_DEPLOY
    for ledger in ("garbage", "[]", json.dumps({"campaigns": []}), json.dumps({"campaigns": {"camp-1": 3}}),
                   json.dumps({"campaigns": {"../x": {"candidates": {"c": {"status": "running"}}}}})):
        (state["screening"] / hold.SCREENING_LEDGER).write_text(ledger, encoding="utf-8")
        assert _all(state)[0] == hold.EXIT_DEPLOY


def test_one_broken_source_hides_no_other(state, monkeypatch) -> None:
    def broken(**_: object) -> None:
        raise TypeError("bug")

    monkeypatch.setattr(hold, "live_screening", broken)
    monkeypatch.setattr(hold, "live_paid_locks", broken)
    _screening(state)
    assert _all(state)[0] == hold.EXIT_DEPLOY
    _envelope(state["queue"] / "pending" / "act-1-abc.json")
    code, reason = _all(state)
    assert code == hold.EXIT_HOLD and reason == "scene_configuration_activation_live:pending:act-1"


@pytest.mark.parametrize("stage", ["launch", "paid_lock", "screening"])
def test_the_cap_and_the_override_apply_to_every_new_stage(state, stage: str) -> None:
    holders: dict[int, set[int]] = {}
    held: set[int] = set()
    if stage == "launch":
        _submitted(state)
        _launch_request(state, "processing")
        held = _dispatcher_alive(state)
    elif stage == "paid_lock":
        holders = {_slot(state): {4242}}
        _runs_from(state, 4242, ACTIVE)
    else:
        _screening(state)
    start = time.time() - hold.CAP_SECONDS - 60
    assert _all(state, held=held, holders=holders, now=start)[0] == hold.EXIT_HOLD
    assert _all(state, held=held, holders=holders, now=start + hold.CAP_SECONDS - 60)[0] == hold.EXIT_HOLD
    code, reason = _all(state, held=held, holders=holders, now=start + hold.CAP_SECONDS)
    assert code == hold.EXIT_DEPLOY and reason.startswith("activation_hold_cap_reached:")
    code, reason = _all(state, held=held, holders=holders, override="money")
    assert code == hold.EXIT_DEPLOY and reason == "activation_hold_overridden:money"


def test_one_cap_however_the_stages_hand_over(state) -> None:
    """An activation, then its launch, then a paid slot, then screening: one clock."""
    start = time.time() - hold.CAP_SECONDS - 3600
    pending = _envelope(state["queue"] / "pending" / "act-1-abc.json")
    assert _all(state, now=start)[0] == hold.EXIT_HOLD
    pending.unlink()
    _submitted(state)
    request = _launch_request(state, "pending")
    assert _all(state, now=start + 3000)[0] == hold.EXIT_HOLD  # shares act-1, after a long gap
    request.unlink()
    holders = {_slot(state): {4242}}
    _runs_from(state, 4242, ACTIVE)
    moment = start + 3000
    while moment < start + hold.CAP_SECONDS - hold.RECHECK_SECONDS:
        moment += hold.RECHECK_SECONDS  # a parked deploy's rechecks
        assert _all(state, holders=holders, now=moment)[0] == hold.EXIT_HOLD
        if moment > start + 9000 and holders:
            holders = {}
            _screening(state)  # the slot ends as screening starts, between two rechecks
    assert _record(state)["first_held_at_epoch"] == start
    code, reason = _all(state, now=start + hold.CAP_SECONDS)
    assert code == hold.EXIT_DEPLOY and reason == "activation_hold_cap_reached:screening_running:camp-1/cand-1"


def test_a_held_request_keeps_its_first_hold_across_a_gap(state) -> None:
    start = time.time() - 4 * 3600
    _screening(state)
    assert _all(state, now=start, request_id=DEPLOY_ID)[0] == hold.EXIT_HOLD
    _screening(state, campaign="camp-2")  # unrelated work, an hour after the last check
    assert _all(state, now=start + 3600, request_id=DEPLOY_ID)[0] == hold.EXIT_HOLD
    record = _record(state)
    assert record["first_held_at_epoch"] == start and record["requests"] == {DEPLOY_ID: start}
    # A different request after the gap is a new hold, as before.
    other = "20261031T210000Z-deploy-0000abce"
    _screening(state, campaign="camp-3")
    assert _all(state, now=start + 7200, request_id=other)[0] == hold.EXIT_HOLD
    assert _record(state)["first_held_at_epoch"] == start + 7200


def test_a_record_from_an_older_door_still_continues_its_hold(state) -> None:
    """Records written before this change have only activation_ids; they still count."""
    start = time.time() - 3600
    (state["parked"] / hold.HOLD_FILE).write_text(json.dumps({
        "schema": hold.HOLD_SCHEMA, "active_release": ACTIVE, "first_held_at_epoch": start,
        "last_checked_at_epoch": start, "activation_ids": ["act-1"],
        "live": [{"activation_id": "act-1", "state": "processing"}]}), encoding="utf-8")
    _envelope(state["queue"] / "pending" / "act-1-abc.json")
    assert _all(state)[0] == hold.EXIT_HOLD
    assert _record(state)["first_held_at_epoch"] == start


def test_main_reads_every_stage_from_the_state_root(state, tmp_path: Path, monkeypatch, capsys) -> None:
    release = tmp_path / "releases" / ACTIVE
    release.mkdir(parents=True)
    link = tmp_path / "active"
    link.symlink_to(release)
    _screening(state)
    monkeypatch.setenv("DOOR_STATE_ROOT", str(state["root"]))
    monkeypatch.setenv("DOOR_ACTIVE_RELEASE_LINK", str(link))
    monkeypatch.setenv("DOOR_ACTIVATION_HOLD_PARKED_DIR", str(state["parked"]))
    monkeypatch.setenv("DOOR_REQUEST_ID", DEPLOY_ID)
    monkeypatch.delenv("DOOR_ACTIVATION_HOLD_OVERRIDE", raising=False)
    assert hold.main(["activation_hold.py"]) == hold.EXIT_HOLD
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "paid_stage_live:screening_running:camp-1/cand-1" and len(out) == 2
    assert _record(state)["requests"] == {DEPLOY_ID: _record(state)["first_held_at_epoch"]}
    monkeypatch.setenv("DOOR_ACTIVATION_HOLD_OVERRIDE", "fixes_live_activation")
    assert hold.main(["activation_hold.py"]) == hold.EXIT_DEPLOY
