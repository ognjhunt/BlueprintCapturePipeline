"""A deploy must not move the ground under a running paid attempt.

Activating the release symlink swaps the tree a running allocator was started
from, while that allocator is holding a rented GPU. That happened on
2026-08-13: a deploy repointed the link 20 minutes into another lane's paid
Content Agents run, which had passed admission under the previous commit and
was mid-heartbeat on a live instance.

It did no visible harm -- the process had already imported its modules -- but
"probably fine" is not a property worth relying on with an instance billing by
the second, and a lane that reads any file from that path afterwards reads bytes
from a commit it was never admitted under.

The lock already existed. `vast_provider_adapter` writes it before the launch
API call and records the holding pid; the deploy just never looked.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "deploy_control_plane_commit", REPO_ROOT / "scripts" / "deploy_control_plane_commit.py"
)
deploy = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(deploy)


def _lock(tmp_path: Path, **overrides) -> Path:
    record = {
        "acquired_at": "2026-08-13T12:49:36.475276+00:00",
        "job_dir": "/var/lib/blueprint/.../vast_provider_run",
        "pid": os.getpid(),
        "purpose": "vast_paid_instance_launch_single_flight_guard",
    }
    record.update(overrides)
    path = tmp_path / "vast_paid_launch.lock"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_a_held_lock_is_reported_with_the_run_that_holds_it(tmp_path: Path) -> None:
    held = deploy._paid_launch_in_flight([str(_lock(tmp_path))])

    assert len(held) == 1
    assert held[0]["pid"] == os.getpid()
    assert held[0]["job_dir"].endswith("vast_provider_run")


def test_no_lock_is_not_a_paid_launch(tmp_path: Path) -> None:
    assert deploy._paid_launch_in_flight([str(tmp_path / "absent.lock")]) == []


def test_a_lock_whose_holder_is_gone_does_not_block_forever(tmp_path: Path) -> None:
    """A stale lock is the paid-lane reaper's problem, not a deploy freeze."""

    # PID 2^22 is above the default pid_max on Linux and macOS alike, so no
    # live process can hold it.
    assert deploy._paid_launch_in_flight([str(_lock(tmp_path, pid=4_194_304))]) == []


@pytest.mark.parametrize(
    "body", ["not json at all", '"a string"', "[]"], ids=["garbage", "scalar", "list"]
)
def test_an_unreadable_lock_counts_as_held(tmp_path: Path, body: str) -> None:
    """Absence of proof is not proof of absence when a GPU may be running."""

    path = tmp_path / "vast_paid_launch.lock"
    path.write_text(body, encoding="utf-8")

    held = deploy._paid_launch_in_flight([str(path)])

    assert [row["holder"] for row in held] == ["unreadable"]


def test_the_deploy_refuses_while_a_paid_launch_holds_the_lock(
    tmp_path: Path, monkeypatch
) -> None:
    """And refuses *before* touching either surface."""

    moved: list[str] = []
    monkeypatch.setattr(
        deploy, "_move_source_checkout", lambda repo, commit: moved.append(commit)
    )

    with pytest.raises(deploy.ControlPlaneDeployError) as excinfo:
        deploy.deploy_control_plane_commit(
            source_repo=tmp_path,
            source_commit="a" * 40,
            release_root=tmp_path / "releases",
            state_root=tmp_path / "state",
            active_link=tmp_path / "active",
            paid_launch_locks=(str(_lock(tmp_path)),),
        )

    assert str(excinfo.value).startswith("deploy_refused_paid_launch_in_flight:")
    assert moved == [], "the source checkout moved before the refusal"


def test_the_canonical_lock_is_checked_by_default() -> None:
    """An operator who forgets the flag still gets the guard."""

    assert deploy.DEFAULT_PAID_LAUNCH_LOCKS == (
        "/var/lib/blueprint/pipeline-control-plane/provider-locks/vast_paid_launch.lock",
    )
