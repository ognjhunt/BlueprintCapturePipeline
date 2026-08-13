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
import io
import json
import os
import subprocess
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


def test_the_canonical_lock_is_checked_by_default() -> None:
    """An operator who forgets the flag still gets the guard."""

    assert deploy.DEFAULT_PAID_LAUNCH_LOCKS == (
        "/var/lib/blueprint/pipeline-control-plane/provider-locks/vast_paid_launch.lock",
    )


def test_the_real_intake_restart_cannot_be_omitted() -> None:
    assert deploy._required_restart_units(()) == (
        "blueprint-pipeline-intake.service",
    )
    assert deploy._required_restart_units(("another.service",)) == (
        "blueprint-pipeline-intake.service",
        "another.service",
    )


def test_restart_reloads_drop_ins_before_restarting_the_intake(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def completed(argv, **kwargs):
        calls.append(tuple(argv))
        stdout = "active\n" if argv[:2] == ["systemctl", "is-active"] else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", completed)

    restarted = deploy._restart_units(("blueprint-pipeline-intake.service",))

    assert calls == [
        ("systemctl", "daemon-reload"),
        ("systemctl", "restart", "blueprint-pipeline-intake.service"),
        ("systemctl", "is-active", "blueprint-pipeline-intake.service"),
    ]
    assert restarted == [
        {"unit": "blueprint-pipeline-intake.service", "state": "active"}
    ]


def test_the_deploy_holds_the_lock_for_its_whole_duration(tmp_path: Path) -> None:
    """Not a check-then-deploy: a launch can start between the two.

    That is not hypothetical. On 2026-08-13 the check passed and the parallel
    lane acquired the lock 20 seconds later, mid-deploy.
    """

    import fcntl

    lock = _lock(tmp_path)
    observed: list[bool] = []

    with deploy._holding_paid_launch_locks([str(lock)]):
        # A launch trying to start now must be refused, which is what the
        # adapter's own non-blocking flock does.
        with lock.open("r", encoding="utf-8") as probe:
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                observed.append(True)
                fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
            except BlockingIOError:
                observed.append(False)

    assert observed == [False], "a launch could start while the deploy held the lock"

    # And released afterwards, or the next launch could never start.
    with lock.open("r", encoding="utf-8") as probe:
        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)


def test_a_lock_held_by_a_launch_refuses_the_deploy_by_name(tmp_path: Path) -> None:
    import fcntl

    lock = _lock(tmp_path)
    with lock.open("r", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(deploy.ControlPlaneDeployError) as excinfo:
            with deploy._holding_paid_launch_locks([str(lock)]):
                pass
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)

    # Names the run, not the file.
    assert str(excinfo.value).startswith("deploy_refused_paid_launch_in_flight:")
    assert "vast_provider_run" in str(excinfo.value)


def test_an_absent_lock_is_not_created_by_the_deploy(tmp_path: Path) -> None:
    """The adapter creates it as the service account at 0600.

    A deploy running as root that created it first would leave a file the
    service can never open again, taking every paid lane down.
    """

    absent = tmp_path / "never-launched" / "vast_paid_launch.lock"

    with deploy._holding_paid_launch_locks([str(absent)]):
        pass

    assert not absent.exists()
    assert not absent.parent.exists()


def test_the_deploy_does_not_move_a_surface_while_refusing(tmp_path: Path, monkeypatch) -> None:
    import fcntl

    moved: list[str] = []
    monkeypatch.setattr(
        deploy, "_move_source_checkout", lambda repo, commit: moved.append(commit)
    )
    lock = _lock(tmp_path)

    with lock.open("r", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(deploy.ControlPlaneDeployError):
            deploy.deploy_control_plane_commit(
                source_repo=tmp_path,
                source_commit="a" * 40,
                release_root=tmp_path / "releases",
                state_root=tmp_path / "state",
                active_link=tmp_path / "active",
                paid_launch_locks=(str(lock),),
            )
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)

    assert moved == []


def test_runtime_identity_drop_in_is_atomic_and_contains_no_credentials(
    tmp_path: Path,
) -> None:
    drop_in = tmp_path / "intake.service.d" / "90-deploy-identity.conf"

    receipt = deploy._install_intake_runtime_identity_drop_in(
        drop_in,
        source_repo=tmp_path / "repo",
        source_commit="b" * 40,
    )

    content = drop_in.read_text(encoding="utf-8")
    assert "[Service]" in content
    assert f"Environment=BLUEPRINT_SOURCE_COMMIT={'b' * 40}" in content
    assert f"Environment=BLUEPRINT_PIPELINE_REPO={tmp_path / 'repo'}" in content
    assert "TOKEN" not in content
    assert "SECRET" not in content
    assert drop_in.stat().st_mode & 0o777 == 0o644
    assert receipt["credential_environment_file_opened"] is False
    assert receipt["credential_values_recorded"] is False


def test_intake_version_probe_rejects_a_stale_running_process(monkeypatch) -> None:
    monkeypatch.setattr(
        deploy.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(
            json.dumps(
                {"commit_proven": True, "source_commit": "stale-runtime"}
            ).encode("utf-8")
        ),
    )

    with pytest.raises(
        deploy.ControlPlaneDeployError,
        match="deploy_intake_runtime_commit_mismatch:stale-runtime",
    ):
        deploy._verify_intake_runtime(
            "http://127.0.0.1:8765/api/live-pipeline/version",
            expected_commit="c" * 40,
        )


def test_intake_version_probe_is_loopback_only() -> None:
    with pytest.raises(
        deploy.ControlPlaneDeployError,
        match="deploy_intake_version_url_not_loopback_http",
    ):
        deploy._verify_intake_runtime(
            "https://production.example/api/live-pipeline/version",
            expected_commit="c" * 40,
        )


def test_intake_version_probe_retries_while_the_restarted_server_binds(
    monkeypatch,
) -> None:
    calls = 0

    def delayed_server(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("not listening yet")
        return io.BytesIO(
            json.dumps(
                {"commit_proven": True, "source_commit": "c" * 40}
            ).encode("utf-8")
        )

    monkeypatch.setattr(deploy.urllib.request, "urlopen", delayed_server)

    result = deploy._verify_intake_runtime(
        "http://127.0.0.1:8765/api/live-pipeline/version",
        expected_commit="c" * 40,
        attempts=2,
        retry_delay_seconds=0,
    )

    assert calls == 2
    assert result["source_commit"] == "c" * 40


def test_deploy_holds_paid_slot_through_restart_and_runtime_probe(
    tmp_path: Path, monkeypatch
) -> None:
    """A second launch cannot enter during the newly added runtime checks."""

    import fcntl

    commit = "d" * 40
    source = tmp_path / "source"
    source.mkdir()
    active = tmp_path / "active"
    release = tmp_path / "release"
    release.mkdir()
    active.symlink_to(release, target_is_directory=True)
    lock = _lock(tmp_path)
    observed: list[str] = []

    monkeypatch.setattr(deploy, "_move_source_checkout", lambda *args: None)
    monkeypatch.setattr(
        deploy,
        "stage_task_evaluation_control_plane_release",
        lambda **kwargs: {
            "source_commit": commit,
            "release_path": str(release),
            "created_release_checkout": True,
        },
    )
    monkeypatch.setattr(deploy, "_surface_commit", lambda *args, **kwargs: commit)
    monkeypatch.setattr(
        deploy,
        "_install_intake_runtime_identity_drop_in",
        lambda *args, **kwargs: {"source_commit": commit},
    )

    def assert_lock_held(stage: str):
        with lock.open("r", encoding="utf-8") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed.append(stage)

    monkeypatch.setattr(
        deploy,
        "_restart_units",
        lambda units: (assert_lock_held("restart") or [{"unit": units[0]}]),
    )
    monkeypatch.setattr(
        deploy,
        "_verify_intake_runtime",
        lambda *args, **kwargs: (
            assert_lock_held("runtime_probe")
            or {"commit_proven": True, "source_commit": commit}
        ),
    )

    receipt = deploy.deploy_control_plane_commit(
        source_repo=source,
        source_commit=commit,
        release_root=tmp_path / "releases",
        state_root=tmp_path / "state",
        active_link=active,
        paid_launch_locks=(str(lock),),
        intake_runtime_drop_in=tmp_path / "drop-in",
    )

    assert observed == ["restart", "runtime_probe"]
    assert receipt["intake_runtime"]["source_commit"] == commit
    assert receipt["restarted_units"][0]["unit"] == deploy.DEFAULT_RESTART_UNITS[0]


def test_the_receipt_records_every_slot_it_was_exclusive_with(tmp_path: Path) -> None:
    """A receipt that under-reports its own guarantee misleads its reader.

    The lock is an N-slot semaphore; recording the single base path the caller
    named would say "1 lock checked" for a deploy that actually held three.
    """

    from blueprint_pipeline.vast_provider_adapter import vast_launch_lock_paths

    base = tmp_path / "locks" / "vast_paid_launch.lock"
    held = deploy._expanded_slots([str(base)])

    assert held == vast_launch_lock_paths(base)
    assert len(held) > 1, "the semaphore should expand to more than the base path"
    assert held[0] == base
