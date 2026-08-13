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
