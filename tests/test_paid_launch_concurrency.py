"""Three paid launches at once, and never a fourth.

The launch lock was single-flight, so every lane queued behind the slowest run
in flight -- a Content Agents run held the provider for 78 minutes on
2026-08-13 while three other lanes waited on it. Raised to 3 on explicit
authorization.

What must not change: each attempt still carries its own hard cap, TTL, and
watchdog, so the worst case is N times one attempt's ceiling rather than an
unbounded fleet. And the deploy has to be exclusive with *every* slot, not just
the one that kept the historical filename.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from blueprint_pipeline import vast_provider_adapter as vpa

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "deploy_control_plane_commit", REPO_ROOT / "scripts" / "deploy_control_plane_commit.py"
)
deploy = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(deploy)


def _acquire(tmp_path: Path, name: str, base: Path):
    job = tmp_path / name
    job.mkdir(parents=True, exist_ok=True)
    return vpa._try_acquire_vast_launch_lock(
        job_dir=job, generated_at="2026-08-13T00:00:00Z", lock_path=base
    )


def test_three_launches_hold_the_provider_and_a_fourth_is_refused(tmp_path: Path) -> None:
    base = tmp_path / "locks" / "vast_paid_launch.lock"
    held = []
    for index in range(3):
        handle, manifest = _acquire(tmp_path, f"job{index}", base)
        assert handle is not None, f"slot {index} should have been free"
        assert manifest["status"] == "acquired"
        held.append(handle)

    # The authorized ceiling, not a queue.
    handle, manifest = _acquire(tmp_path, "job3", base)
    assert handle is None
    assert manifest["blockers"] == ["vast_paid_launch_lock_busy"]

    # Releasing one frees exactly one.
    vpa._release_vast_launch_lock(held.pop())
    handle, manifest = _acquire(tmp_path, "job4", base)
    assert handle is not None
    assert manifest["status"] == "acquired"
    held.append(handle)

    for handle in held:
        vpa._release_vast_launch_lock(handle)


def test_slot_zero_keeps_the_historical_filename(tmp_path: Path) -> None:
    """A reaper or operator that knows only the old name still sees a lock."""

    base = tmp_path / "locks" / "vast_paid_launch.lock"
    slots = vpa.vast_launch_lock_paths(base)

    assert slots[0] == base
    assert [path.name for path in slots[1:]] == [
        "vast_paid_launch.slot1.lock",
        "vast_paid_launch.slot2.lock",
    ]


def test_the_env_override_cannot_widen_past_the_compiled_policy(monkeypatch) -> None:
    """A typo in a unit file must not authorize more concurrent spend."""

    base = Path("/tmp/locks/vast_paid_launch.lock")

    monkeypatch.setenv(vpa.MAX_CONCURRENT_PAID_LAUNCHES_ENV, "50")
    assert len(vpa.vast_launch_lock_paths(base)) == vpa.DEFAULT_MAX_CONCURRENT_PAID_LAUNCHES

    monkeypatch.setenv(vpa.MAX_CONCURRENT_PAID_LAUNCHES_ENV, "1")
    assert len(vpa.vast_launch_lock_paths(base)) == 1

    # And never deadlock every lane.
    for bad in ("0", "-4", "banana", ""):
        monkeypatch.setenv(vpa.MAX_CONCURRENT_PAID_LAUNCHES_ENV, bad)
        assert len(vpa.vast_launch_lock_paths(base)) >= 1


def test_the_deploy_is_exclusive_with_every_slot_not_just_the_first(
    tmp_path: Path,
) -> None:
    """Holding only slot 0 would look correct while two GPUs ran."""

    import fcntl

    base = tmp_path / "locks" / "vast_paid_launch.lock"
    slots = vpa.vast_launch_lock_paths(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    for slot in slots:
        slot.touch()

    # A launch holding the *last* slot must still stop the deploy.
    with slots[-1].open("r", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(deploy.ControlPlaneDeployError) as excinfo:
            with deploy._holding_paid_launch_locks([str(base)]):
                pass
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)

    assert str(excinfo.value).startswith("deploy_refused_paid_launch_in_flight:")


def test_a_deploy_blocks_every_slot_while_it_holds_them(tmp_path: Path) -> None:
    base = tmp_path / "locks" / "vast_paid_launch.lock"
    base.parent.mkdir(parents=True, exist_ok=True)
    for slot in vpa.vast_launch_lock_paths(base):
        slot.touch()

    with deploy._holding_paid_launch_locks([str(base)]):
        handle, manifest = _acquire(tmp_path, "job-during-deploy", base)

    assert handle is None, "a launch started while the deploy held every slot"
    assert manifest["blockers"] == ["vast_paid_launch_lock_busy"]


def _unusable(slot: Path) -> None:
    """Make a slot exist but be unopenable, for any uid including root.

    Production got here by a different route -- a tool run as root created
    `slot1`/`slot2` owned `root:root` at 0644, which the `blueprint` service
    account can neither open for append nor chmod. A mode-based fixture would
    silently pass under a root test runner, so the slot is made a directory
    instead: `open("a+")` raises `IsADirectoryError` regardless of privilege.
    """

    slot.parent.mkdir(parents=True, exist_ok=True)
    slot.mkdir()


def test_a_slot_the_launcher_cannot_open_is_skipped_not_raised(tmp_path: Path) -> None:
    """Falling through to an unusable slot must not crash at the money boundary.

    `open("a+")` and `chmod` sat outside the `try:` that caught only
    `BlockingIOError`, so a slot the launching account could not open raised an
    unhandled `PermissionError` out of the acquisition instead of refusing.
    Because slot 0 is tried first and is usually fine, this only fired when a
    concurrent lane already held slot 0 -- exactly when concurrency was meant
    to help.
    """

    base = tmp_path / "locks" / "vast_paid_launch.lock"
    slots = vpa.vast_launch_lock_paths(base)
    base.parent.mkdir(parents=True, exist_ok=True)

    blocker_handle, _ = _acquire(tmp_path, "holds-slot0", base)
    assert blocker_handle is not None
    _unusable(slots[1])

    try:
        handle, manifest = _acquire(tmp_path, "falls-through", base)
        assert handle is not None, "an unusable slot must not consume the attempt"
        assert manifest["status"] == "acquired"
        assert manifest["lock_path"] == str(slots[2])
        # A degraded semaphore has to be visible on the success path too,
        # otherwise the fleet quietly runs at lower concurrency forever.
        assert any(
            slots[1].name in entry for entry in manifest["unusable_lock_slots"]
        ), manifest["unusable_lock_slots"]
        vpa._release_vast_launch_lock(handle)
    finally:
        vpa._release_vast_launch_lock(blocker_handle)


def test_every_slot_unusable_refuses_with_its_own_blocker(tmp_path: Path) -> None:
    """`busy` and `unusable` are different faults and must not be conflated.

    "Busy" is the fleet at its authorized concurrency and is nobody's bug.
    "Unusable" is a provisioning fault that no amount of waiting will clear,
    so it gets a blocker an operator can act on.
    """

    base = tmp_path / "locks" / "vast_paid_launch.lock"
    base.parent.mkdir(parents=True, exist_ok=True)
    for slot in vpa.vast_launch_lock_paths(base):
        _unusable(slot)

    handle, manifest = _acquire(tmp_path, "no-usable-slot", base)

    assert handle is None
    assert manifest["blockers"] == ["vast_paid_launch_lock_unusable"]
    assert manifest["status"] == "blocked"
    assert len(manifest["unusable_lock_slots"]) == len(vpa.vast_launch_lock_paths(base))


def test_the_blocked_phase_reason_is_the_cause_the_run_actually_hit() -> None:
    """Retained phase artifacts must not name a cause the run did not hit.

    The blocked-phase writer took the literal string `vast_paid_launch_lock_busy`
    for all four phases, so a host whose slots the service account cannot open
    produced artifacts telling the next operator to wait for capacity -- which
    would never clear, because capacity was never the problem.
    """

    assert (
        vpa._lock_blocked_phase_reason({"blockers": ["vast_paid_launch_lock_unusable"]})
        == "vast_paid_launch_lock_unusable"
    )
    assert (
        vpa._lock_blocked_phase_reason({"blockers": ["vast_paid_launch_lock_busy"]})
        == "vast_paid_launch_lock_busy"
    )
    # A manifest that names no cause still has to refuse, and "busy" is the
    # conservative reading: it claims nothing about the host's provisioning.
    assert vpa._lock_blocked_phase_reason({}) == "vast_paid_launch_lock_busy"
    assert vpa._lock_blocked_phase_reason({"blockers": []}) == "vast_paid_launch_lock_busy"


def test_the_deploy_repairs_a_slot_the_service_account_cannot_use(tmp_path: Path) -> None:
    """The deploy is the only root-run repo code that touches these every time.

    The guard runs as `blueprint` and can detect a mis-owned slot but never
    repair one. Leaving the repair to an operator's `chown` is the remembered
    ritual a rebuilt host does not perform -- and with the guard blocking on an
    unusable slot, an unrepaired host would refuse to start intake at all.
    """

    base = tmp_path / "locks" / "vast_paid_launch.lock"
    base.parent.mkdir(parents=True, exist_ok=True)
    slots = vpa.vast_launch_lock_paths(base)
    for slot in slots:
        slot.touch()
        slot.chmod(0o644)  # the mode a root-run tool leaves under the default umask

    chowned: list[tuple[str, int, int]] = []

    def fake_chown(path, uid, gid):  # type: ignore[no-untyped-def]
        chowned.append((Path(path).name, uid, gid))

    receipt = deploy._repair_paid_launch_lock_slots(
        [str(base)], owner_uid=4242, owner_gid=4242, chown=fake_chown
    )

    assert [name for name, _, _ in chowned] == [slot.name for slot in slots]
    assert all((uid, gid) == (4242, 4242) for _, uid, gid in chowned)
    for slot in slots:
        assert slot.stat().st_mode & 0o777 == 0o600, oct(slot.stat().st_mode)
    assert receipt["repaired_slots"] == [str(slot) for slot in slots]


def test_the_deploy_repair_skips_slots_that_do_not_exist(tmp_path: Path) -> None:
    """An absent slot is created correctly by the guard as the service account."""

    base = tmp_path / "locks" / "vast_paid_launch.lock"
    base.parent.mkdir(parents=True, exist_ok=True)

    receipt = deploy._repair_paid_launch_lock_slots(
        [str(base)], owner_uid=4242, owner_gid=4242, chown=lambda *a: None
    )

    assert receipt["repaired_slots"] == []
    assert not base.exists()
