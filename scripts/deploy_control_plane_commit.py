#!/usr/bin/env python3
"""Move every control-plane surface to one commit, and prove they all moved.

A deploy here is not one thing. `/opt/blueprint/BlueprintCapturePipeline` is a
mutable clone that runs intake and is the only surface with Git history; the
release symlink is the detached checkout the allocator runs from. Moving one
and not the other is silent -- the website still answers, launches still queue,
and the allocator refuses at the paid boundary on a commit mismatch, several
minutes and one consumed attempt authority later.

The failure that forced this was worse than a stale surface. A release tree
created with `git archive | tar -x` has the right bytes and no `.git`, so
`git rev-parse HEAD` fails inside it and the allocator's orchestrator identity
probe comes back empty. Two unrelated-looking admission blockers
(`gpu_canary_orchestrator_identity_probe_failed` and
`adp_content_agents_config_preflight_binding_invalid`, the latter because the
expected commit compared against was the empty string) had that one cause.

So the check that matters is not "did the command succeed" but "does every
surface now answer `git rev-parse HEAD` with the same commit". A surface that
cannot answer at all is the exact defect above and fails here rather than at
the paid boundary.

Runs Git and systemd on this host. Contacts no provider, reads no credential,
and rents nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import subprocess  # nosec B404 - fixed git/systemctl argv over validated paths
from pathlib import Path
from typing import Any, Mapping, Sequence

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_task_evaluation_control_plane_release import (  # noqa: E402
    ControlPlaneReleaseError,
    stage_task_evaluation_control_plane_release,
)

SCHEMA_VERSION = "control_plane_commit_deploy_receipt.v1"

#: The single-flight guard a lane holds for the whole life of a paid instance.
#: `vast_provider_adapter` writes it before the launch API call and clears it on
#: teardown, and it records the holding pid and job directory.
DEFAULT_PAID_LAUNCH_LOCKS = (
    "/var/lib/blueprint/pipeline-control-plane/provider-locks/vast_paid_launch.lock",
)


class ControlPlaneDeployError(ValueError):
    """A surface did not reach the requested commit, or cannot say that it did."""


@contextlib.contextmanager
def _holding_paid_launch_locks(lock_paths: Sequence[str]):
    """Hold the provider's own launch lock for the whole deploy.

    Checking whether a lock is held and then deploying is two steps, and a
    launch can start between them -- which is exactly what happened on
    2026-08-13: the check passed and the parallel lane acquired the lock 20
    seconds later, mid-deploy.

    `vast_provider_adapter` guards a paid launch with `fcntl.flock` on this
    file, so taking the same lock makes deploy and launch genuinely exclusive
    rather than politely sequenced. While the deploy holds it a launch refuses
    with `vast_paid_launch_lock_busy`, which is the correct outcome: a run must
    not start against a release that is being swapped underneath it.

    Opened read-only and never created. The adapter creates this file as the
    service account at 0600; a deploy running as root that created it first
    would leave a file the service can never open again, taking every paid lane
    down. A lock that does not exist yet means no adapter has ever launched
    here, so there is nothing to be exclusive with.
    """

    handles: list[Any] = []
    try:
        for raw in lock_paths:
            path = Path(raw).expanduser()
            try:
                handle = path.open("r", encoding="utf-8")
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ControlPlaneDeployError(
                    f"deploy_paid_launch_lock_unreadable:{path.name}"
                ) from exc
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.seek(0)
                holder = handle.read(1000)
                handle.close()
                for other in handles:
                    other.close()
                raise ControlPlaneDeployError(
                    "deploy_refused_paid_launch_in_flight:" + _holder_summary(holder)
                ) from None
            handles.append(handle)
        yield
    finally:
        for handle in handles:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()


def _holder_summary(holder: str) -> str:
    """Name the run that holds the lock, not the file that records it."""

    try:
        record = json.loads(holder)
    except (ValueError, json.JSONDecodeError):
        return "unparseable_holder"
    if not isinstance(record, Mapping):
        return "unparseable_holder"
    return str(record.get("job_dir") or record.get("pid") or "unknown_holder")


def _git(repo: Path, *arguments: str) -> tuple[int, str]:
    result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout.strip()


def _surface_commit(path: Path, *, name: str) -> str:
    """What commit does this surface *say* it is? Refuse if it cannot say."""

    code, head = _git(path, "rev-parse", "HEAD")
    if code != 0 or not head:
        # A tree extracted from an archive lands here: correct bytes, no
        # identity, and every downstream identity probe silently empty.
        raise ControlPlaneDeployError(f"deploy_surface_has_no_git_identity:{name}")
    code, dirty = _git(path, "status", "--porcelain")
    if code != 0:
        raise ControlPlaneDeployError(f"deploy_surface_status_unavailable:{name}")
    if dirty:
        raise ControlPlaneDeployError(f"deploy_surface_checkout_dirty:{name}")
    return head


def _move_source_checkout(repo: Path, commit: str) -> None:
    if _git(repo, "status", "--porcelain")[1]:
        # Never carry a local edit across a deploy: it would be running code
        # that is on no commit at all.
        raise ControlPlaneDeployError("deploy_source_checkout_dirty")
    if _git(repo, "fetch", "--quiet", "origin", "main")[0] != 0:
        raise ControlPlaneDeployError("deploy_source_fetch_failed")
    if _git(repo, "checkout", "--quiet", commit)[0] != 0:
        raise ControlPlaneDeployError(f"deploy_source_checkout_failed:{commit}")


def _restart_units(units: Sequence[str]) -> list[dict[str, Any]]:
    restarted: list[dict[str, Any]] = []
    for unit in units:
        result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["systemctl", "restart", unit], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise ControlPlaneDeployError(f"deploy_unit_restart_failed:{unit}")
        active = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["systemctl", "is-active", unit], capture_output=True, text=True, check=False
        )
        state = active.stdout.strip()
        if state != "active":
            raise ControlPlaneDeployError(f"deploy_unit_not_active:{unit}:{state}")
        restarted.append({"unit": unit, "state": state})
    return restarted


def deploy_control_plane_commit(
    *,
    source_repo: str | Path,
    source_commit: str,
    release_root: str | Path,
    state_root: str | Path,
    active_link: str | Path,
    restart_units: Sequence[str] = (),
    paid_launch_locks: Sequence[str] = DEFAULT_PAID_LAUNCH_LOCKS,
) -> dict[str, Any]:
    """Move the mutable clone and the release link, then verify both."""

    source = Path(source_repo).expanduser().resolve()
    active = Path(active_link).expanduser()

    # Held for the whole deploy, not sampled before it: a launch that starts
    # mid-deploy would read a release being swapped underneath it.
    with _holding_paid_launch_locks(paid_launch_locks):
        _move_source_checkout(source, source_commit)
        release = stage_task_evaluation_control_plane_release(
            source_repo=source,
            source_commit=source_commit,
            release_root=release_root,
            state_root=state_root,
            active_link=active,
            activate=True,
        )
    commit = str(release["source_commit"])

    surfaces = {
        "source_checkout": source,
        "active_release": active.resolve(),
    }
    observed: dict[str, str] = {}
    for name, path in surfaces.items():
        observed[name] = _surface_commit(path, name=name)
    disagreeing = sorted(
        name for name, head in observed.items() if head != commit
    )
    if disagreeing:
        # The whole point. One surface moving is not a deploy.
        raise ControlPlaneDeployError(
            "deploy_surfaces_disagree:" + ",".join(disagreeing)
        )

    restarted = _restart_units(restart_units)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "deployed",
        "source_commit": commit,
        "surfaces": [
            {"name": name, "path": str(path), "head": observed[name]}
            for name, path in sorted(surfaces.items())
        ],
        "release_path": release["release_path"],
        "created_release_checkout": release["created_release_checkout"],
        "restarted_units": restarted,
        "paid_launch_locks_checked": list(paid_launch_locks),
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "This receipt proves every named surface reports this commit and is "
            "clean. It says nothing about whether any launch profile, bundle, or "
            "preflight built at an earlier commit is still valid -- those bind "
            "the deployed commit and are rebuilt after a deploy, not before."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--active-link", required=True)
    parser.add_argument(
        "--restart-unit",
        action="append",
        default=[],
        help="A systemd unit to restart and confirm active. Repeatable.",
    )
    parser.add_argument(
        "--paid-launch-lock",
        action="append",
        default=None,
        help=(
            "A provider single-flight lock to check before activating. "
            "Repeatable. Defaults to the canonical Vast paid-launch lock."
        ),
    )
    parser.add_argument("--receipt-out")
    args = parser.parse_args(argv)

    try:
        receipt = deploy_control_plane_commit(
            source_repo=args.source_repo,
            source_commit=args.source_commit,
            release_root=args.release_root,
            state_root=args.state_root,
            active_link=args.active_link,
            restart_units=tuple(args.restart_unit),
            paid_launch_locks=tuple(args.paid_launch_lock or DEFAULT_PAID_LAUNCH_LOCKS),
        )
    except (OSError, ControlPlaneDeployError, ControlPlaneReleaseError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                indent=1,
                sort_keys=True,
            )
        )
        return 2

    if args.receipt_out:
        out = Path(args.receipt_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
