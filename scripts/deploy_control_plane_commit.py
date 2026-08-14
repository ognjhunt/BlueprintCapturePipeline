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
surface now answer `git rev-parse HEAD` with the same commit, and does the
restarted intake process report that commit from its version endpoint". A
surface that cannot answer at all, or a service still bound to an archived
checkout by its environment file, fails here rather than at the paid boundary.

Runs Git and systemd on this host. Contacts no provider, reads no credential,
and rents nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import stat
import subprocess  # nosec B404 - fixed git/systemctl argv over validated paths
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
DEFAULT_RESTART_UNITS = ("blueprint-pipeline-intake.service",)
DEFAULT_INTAKE_RUNTIME_DROP_IN = (
    "/etc/systemd/system/blueprint-pipeline-intake.service.d/"
    "90-blueprint-deploy-identity.conf"
)
DEFAULT_INTAKE_VERSION_URL = "http://127.0.0.1:8765/api/live-pipeline/version"


class ControlPlaneDeployError(ValueError):
    """A surface did not reach the requested commit, or cannot say that it did."""


def _expanded_slots(lock_paths: Sequence[str]) -> list[Path]:
    """Every concurrency slot, not just slot 0.

    The launch lock became an N-slot semaphore so lanes stop queueing behind
    the slowest run. A deploy that held only the historical filename would be
    exclusive with slot 0 and blind to the rest -- which is worse than the
    check it replaced, because it would look correct while swapping the release
    under two live GPUs.
    """

    from blueprint_pipeline.vast_provider_adapter import vast_launch_lock_paths

    expanded: list[Path] = []
    for raw in lock_paths:
        for slot in vast_launch_lock_paths(Path(raw).expanduser()):
            if slot not in expanded:
                expanded.append(slot)
    return expanded


DEFAULT_SERVICE_ACCOUNT = "blueprint"


def _repair_paid_launch_lock_slots(
    lock_paths: Sequence[str],
    *,
    owner_uid: int,
    owner_gid: int,
    chown: Any = os.chown,
) -> dict[str, Any]:
    """Give every existing lock slot back to the account that launches.

    The launch lock is an N-slot semaphore, and a slot only counts if the
    service account can open it. A paid-lane tool run once as root left
    `slot1`/`slot2` owned `root:root` at 0644 on the live control plane, so the
    authorized N=3 was really N=1 -- invisibly, because the lane holding slot 0
    kept succeeding.

    The runtime guard detects this on every service start but runs as the
    service account and can never repair it. This is the only root-run repo
    code that touches these files on every deploy, so the repair belongs here
    rather than in an operator's remembered `chown`, which a rebuilt host never
    performs. With the guard blocking on an unusable slot, an unrepaired host
    refuses to start intake at all.

    Absent slots are left alone: the guard creates those as the service account
    at 0600, and `open("a+")` never changes an existing file's owner, so a slot
    created correctly once survives every later root-run tool.
    """

    repaired: list[str] = []
    for path in _expanded_slots(lock_paths):
        if not path.is_file():
            continue
        chown(path, owner_uid, owner_gid)
        path.chmod(0o600)
        repaired.append(str(path))
    return {"repaired_slots": repaired, "owner_uid": owner_uid, "owner_gid": owner_gid}


def _service_account_ids(account: str) -> tuple[int, int] | None:
    """Resolve the service account, or report that this host has none."""

    try:
        import pwd

        entry = pwd.getpwnam(account)
    except (ImportError, KeyError):
        return None
    return entry.pw_uid, entry.pw_gid


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
        for path in _expanded_slots(lock_paths):
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
                # Slots already taken are released by the `finally` below.
                # Closing them here too made the unlock operate on a closed
                # file, which raises ValueError -- not the OSError the cleanup
                # suppresses -- so a refusal crashed instead of refusing.
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
    reload_result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
        ["systemctl", "daemon-reload"], capture_output=True, text=True, check=False
    )
    if reload_result.returncode != 0:
        raise ControlPlaneDeployError("deploy_systemd_daemon_reload_failed")
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


def _required_restart_units(units: Sequence[str]) -> tuple[str, ...]:
    """The intake restart is mandatory; callers may only add units."""

    required = list(DEFAULT_RESTART_UNITS)
    for unit in units:
        if unit not in required:
            required.append(unit)
    return tuple(required)


def _install_intake_runtime_identity_drop_in(
    drop_in: Path, *, source_repo: Path, source_commit: str
) -> dict[str, Any]:
    """Install a final non-secret env file without opening the credential file.

    The base unit loads ``/etc/blueprint/pipeline-control-plane.env``.  systemd
    gives values loaded from ``EnvironmentFile=`` precedence over values from
    ``Environment=``, even when the latter appears in a later drop-in.  The
    first production version of this deploy guard therefore restarted the
    service while the archived checkout named in the credential env file still
    won.  Load a second, identity-only env file last so it overrides those two
    non-secret identity and import-path keys while leaving every credential in
    the original file alone.
    """

    if drop_in.is_symlink():
        raise ControlPlaneDeployError("deploy_intake_runtime_drop_in_symlink")
    if drop_in.exists() and not stat.S_ISREG(drop_in.stat().st_mode):
        raise ControlPlaneDeployError("deploy_intake_runtime_drop_in_not_regular")
    identity_env = drop_in.with_suffix(".env")
    if identity_env.is_symlink():
        raise ControlPlaneDeployError("deploy_intake_runtime_identity_env_symlink")
    if identity_env.exists() and not stat.S_ISREG(identity_env.stat().st_mode):
        raise ControlPlaneDeployError("deploy_intake_runtime_identity_env_not_regular")
    if any(character.isspace() for character in str(source_repo)):
        raise ControlPlaneDeployError("deploy_intake_source_repo_contains_whitespace")
    if len(source_commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ControlPlaneDeployError("deploy_intake_source_commit_invalid")
    env_content = (
        "# Managed by scripts/deploy_control_plane_commit.py.\n"
        "# Contains deployment identity only; no credentials.\n"
        f"BLUEPRINT_PIPELINE_REPO={source_repo}\n"
        f"BLUEPRINT_SOURCE_COMMIT={source_commit}\n"
        f"PYTHONPATH={source_repo / 'src'}\n"
    )
    drop_in_content = (
        "# Managed by scripts/deploy_control_plane_commit.py.\n"
        "# Loaded after the base unit credential EnvironmentFile.\n"
        "[Service]\n"
        f"EnvironmentFile={identity_env}\n"
    )

    def atomic_write(path: Path, content: str) -> None:
        temp_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, 0o644)
            os.replace(temp_path, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            temp_path = None
        finally:
            if temp_path is not None:
                with contextlib.suppress(OSError):
                    temp_path.unlink()

    try:
        # Publish the referenced file first.  A concurrent daemon-reload can
        # therefore see the old complete pair or the new complete env file,
        # never a drop-in pointing at an absent file.
        atomic_write(identity_env, env_content)
        atomic_write(drop_in, drop_in_content)
    except OSError as exc:
        raise ControlPlaneDeployError("deploy_intake_runtime_drop_in_update_failed") from exc
    return {
        "path": str(drop_in),
        "identity_environment_file": str(identity_env),
        "source_repo": str(source_repo),
        "source_commit": source_commit,
        "pythonpath": str(source_repo / "src"),
        "credential_environment_file_opened": False,
        "credential_values_recorded": False,
    }


def _verify_intake_runtime(
    url: str,
    *,
    expected_commit: str,
    attempts: int = 30,
    retry_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    """Require the restarted process—not just its files—to report the SHA."""

    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ControlPlaneDeployError("deploy_intake_version_url_not_loopback_http")
    if attempts < 1:
        raise ControlPlaneDeployError("deploy_intake_version_probe_attempts_invalid")
    payload: Any = None
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=15) as response:  # nosec B310
                payload = json.load(response)
            break
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(retry_delay_seconds)
    else:
        raise ControlPlaneDeployError("deploy_intake_version_probe_failed") from last_error
    if not isinstance(payload, Mapping):
        raise ControlPlaneDeployError("deploy_intake_version_payload_invalid")
    observed = str(payload.get("source_commit") or "")
    if payload.get("commit_proven") is not True or observed != expected_commit:
        raise ControlPlaneDeployError(
            f"deploy_intake_runtime_commit_mismatch:{observed or 'missing'}"
        )
    return {
        "url": url,
        "commit_proven": True,
        "source_commit": observed,
        "service_schema_version": payload.get("service_schema_version"),
    }


def deploy_control_plane_commit(
    *,
    source_repo: str | Path,
    source_commit: str,
    release_root: str | Path,
    state_root: str | Path,
    active_link: str | Path,
    restart_units: Sequence[str] = DEFAULT_RESTART_UNITS,
    paid_launch_locks: Sequence[str] = DEFAULT_PAID_LAUNCH_LOCKS,
    intake_runtime_drop_in: str | Path = DEFAULT_INTAKE_RUNTIME_DROP_IN,
    intake_version_url: str = DEFAULT_INTAKE_VERSION_URL,
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

        runtime_binding = _install_intake_runtime_identity_drop_in(
            Path(intake_runtime_drop_in).expanduser(),
            source_repo=source,
            source_commit=commit,
        )
        # Before the restart, not after: the runtime guard blocks a unit from
        # starting on a lock slot the service account cannot use, so a host
        # carrying a root-created slot would fail its own restart here.
        service_ids = _service_account_ids(DEFAULT_SERVICE_ACCOUNT)
        if service_ids is None:
            lock_repair: dict[str, Any] = {
                "status": "not_applicable_no_service_account",
                "account": DEFAULT_SERVICE_ACCOUNT,
                "repaired_slots": [],
            }
        else:
            lock_repair = _repair_paid_launch_lock_slots(
                paid_launch_locks, owner_uid=service_ids[0], owner_gid=service_ids[1]
            )
            lock_repair["status"] = "repaired"
            lock_repair["account"] = DEFAULT_SERVICE_ACCOUNT
        restarted = _restart_units(_required_restart_units(restart_units))
        runtime = _verify_intake_runtime(
            intake_version_url, expected_commit=commit
        )

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
        "intake_runtime_binding": runtime_binding,
        "intake_runtime": runtime,
        # Every slot actually held, not the one base path the caller named.
        # The lock is an N-slot semaphore, so recording the input would
        # under-report what this deploy was exclusive with -- and a receipt
        # that under-reports its own guarantee is the thing a later reader
        # trusts.
        "paid_launch_locks_held": [
            str(path) for path in _expanded_slots(paid_launch_locks)
        ],
        "paid_launch_lock_repair": lock_repair,
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "This receipt proves every named filesystem surface reports this "
            "commit and is clean, and that the restarted intake process reports "
            "the same commit. It says nothing about whether any launch profile, "
            "bundle, or preflight built at an earlier commit is still valid -- "
            "those bind the deployed commit and are rebuilt after a deploy, not "
            "before."
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
        default=None,
        help=(
            "An additional systemd unit to restart and confirm active. "
            "Repeatable; the canonical intake unit is always restarted."
        ),
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
    parser.add_argument(
        "--intake-runtime-drop-in", default=DEFAULT_INTAKE_RUNTIME_DROP_IN
    )
    parser.add_argument("--intake-version-url", default=DEFAULT_INTAKE_VERSION_URL)
    args = parser.parse_args(argv)

    try:
        receipt = deploy_control_plane_commit(
            source_repo=args.source_repo,
            source_commit=args.source_commit,
            release_root=args.release_root,
            state_root=args.state_root,
            active_link=args.active_link,
            restart_units=tuple(args.restart_unit or ()),
            paid_launch_locks=tuple(args.paid_launch_lock or DEFAULT_PAID_LAUNCH_LOCKS),
            intake_runtime_drop_in=args.intake_runtime_drop_in,
            intake_version_url=args.intake_version_url,
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
