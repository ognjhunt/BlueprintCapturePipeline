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
import hashlib
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
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))

from stage_task_evaluation_control_plane_release import (  # noqa: E402
    ControlPlaneReleaseError,
    stage_task_evaluation_control_plane_release,
)
from bootstrap_task_evaluation_splat_render_prerequisites import (  # noqa: E402
    validate_splat_render_prerequisites,
)
from provision_task_evaluation_scene_configuration_release import (  # noqa: E402
    provision_scene_configuration_release,
    service_account_readback,
)

SCHEMA_VERSION = "control_plane_commit_deploy_receipt.v1"

#: The single-flight guard a lane holds for the whole life of a paid instance.
#: `vast_provider_adapter` writes it before the launch API call and clears it on
#: teardown, and it records the holding pid and job directory.
DEFAULT_PAID_LAUNCH_LOCKS = (
    "/var/lib/blueprint/pipeline-control-plane/provider-locks/vast_paid_launch.lock",
)
DEFAULT_RESTART_UNITS = ("blueprint-pipeline-intake.service",)
DEFAULT_DEPLOYED_SYSTEMD_UNITS = (
    "blueprint-task-evaluation-launch-dispatcher.service",
    "blueprint-task-evaluation-launch-dispatcher.path",
    "blueprint-task-evaluation-launch-preparation.service",
    "blueprint-task-evaluation-launch-preparation.path",
    "blueprint-task-evaluation-episode-compilation.service",
    "blueprint-task-evaluation-episode-compilation.path",
    "blueprint-task-evaluation-launch-activation.service",
    "blueprint-task-evaluation-launch-activation.path",
    "blueprint-scene-object-discovery.service",
    "blueprint-scene-object-discovery.path",
)
#: Watchers whose execution surface is provably no-spend and may be armed on a
#: fresh host without widening provider authority.  The paid dispatcher is
#: deliberately absent: its operator freeze must survive every deploy unless
#: ``--arm-path-units`` is explicitly supplied.
DEFAULT_ALWAYS_ARM_PATH_UNITS = (
    "blueprint-task-evaluation-launch-preparation.path",
    "blueprint-task-evaluation-episode-compilation.path",
    "blueprint-task-evaluation-launch-activation.path",
    "blueprint-scene-object-discovery.path",
)
#: The only unit kinds a release may install.  The oneshot ``.service`` and its
#: queue-watching ``.path`` are a pair: installing one without the other left
#: the durable queue watched by stale bytes (or by nothing on a rebuilt host).
#: Timers, sockets, and anything else stay operator-managed and are refused.
DEPLOYED_SYSTEMD_UNIT_SUFFIXES = (".service", ".path")
DEFAULT_SYSTEMD_DIR = "/etc/systemd/system"
DEFAULT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT = (
    "/var/lib/blueprint/pipeline-control-plane/scene-object-discoveries"
)
DEFAULT_SCENE_OBJECT_DISCOVERY_RUNTIME_DIRECTORIES = (
    DEFAULT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT,
    *(f"{DEFAULT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT}/{name}" for name in (
        "pending",
        "processing",
        "blocked",
        "results",
        "identities",
        "selections",
    )),
    "/var/lib/blueprint/task-evaluation-inputs/scene-object-discoveries",
    "/var/lib/blueprint/task-evaluation-inputs/scene-object-discovery-outputs",
)
DEFAULT_INTAKE_RUNTIME_DROP_IN = (
    "/etc/systemd/system/blueprint-pipeline-intake.service.d/"
    "90-blueprint-deploy-identity.conf"
)
DEFAULT_INTAKE_VERSION_URL = "http://127.0.0.1:8765/api/live-pipeline/version"
DEFAULT_SCENE_CONFIGURATION_ENVIRONMENT_FILE = (
    "/etc/blueprint/task-evaluation-scene-configuration-release.env"
)
DEFAULT_SCENE_CONFIGURATION_RUNTIME_ROOT = (
    "/var/lib/blueprint/task-evaluation-inputs/system-runtimes"
)
DEFAULT_SPLAT_RENDER_PREREQUISITE_ROOT = (
    "/var/lib/blueprint/task-evaluation-inputs/system-runtime-prerequisites/"
    "splat-render-v1"
)
DEFAULT_ARTIFIXER_SOURCE_ROOT = (
    "/var/lib/blueprint/task-evaluation-inputs/sources/artifixer-a392c4df"
)
DEFAULT_CONTENT_AGENTS_SOURCE_ROOT = (
    "/var/lib/blueprint/task-evaluation-inputs/sources/"
    "usd-content-agents-v0.5.2-36dbf3f2"
)
INTAKE_START_TIMEOUT_SECONDS = 180
DEPLOY_RELEASE_PROVENANCE_NAME = "deploy-release-provenance.json"
SUPERSEDED_ITERATION_PROVENANCE_NAME = (
    "deploy-release-provenance.iteration-superseded.json"
)


class ControlPlaneDeployError(ValueError):
    """A surface did not reach the requested commit, or cannot say that it did."""


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _validated_release_provenance(
    path: str | Path, *, source_commit: str
) -> tuple[Path, bytes, dict[str, Any]]:
    """Open the exact live-verified production-promotion receipt."""

    raw_source = Path(path).expanduser()
    try:
        if raw_source.is_symlink():
            raise ControlPlaneDeployError("deploy_release_provenance_invalid")
        source = raw_source.resolve()
        if not source.is_file():
            raise ControlPlaneDeployError("deploy_release_provenance_invalid")
        payload = source.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlPlaneDeployError("deploy_release_provenance_invalid") from exc
    if not isinstance(value, Mapping):
        raise ControlPlaneDeployError("deploy_release_provenance_invalid")
    collection = value.get("collection")
    claim_boundary = value.get("claim_boundary")
    if not (
        value.get("schema_version") == "blueprint.deploy_release_provenance.v1"
        and value.get("status") == "verified"
        and value.get("git_sha") == source_commit
        and value.get("workflow_name") == "Full Test Lane"
        and value.get("workflow_path") == ".github/workflows/full-test-lane.yml"
        and value.get("job_name") == "Full pytest lane on CPU runner"
        and type(value.get("run_id")) is int
        and value.get("run_id", 0) > 0
        and isinstance(collection, Mapping)
        and type(collection.get("test_count")) is int
        and collection.get("test_count", 0) > 0
        and isinstance(claim_boundary, Mapping)
        and claim_boundary.get("canonical_full_lane_verified") is True
    ):
        raise ControlPlaneDeployError("deploy_release_provenance_mismatch")
    return source, payload, dict(value)


def _install_release_provenance(
    *, payload: bytes, state_root: Path, source_commit: str, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Install promotion proof, permitting only the documented one-way upgrade.

    An iteration deploy writes a development-only receipt before the full lane
    finishes.  The same exact commit must later be promotable without deleting
    that earlier evidence.  Preserve the iteration receipt beside the canonical
    path, then atomically replace the canonical receipt with the verified one.
    Every other content change remains a conflict.
    """

    destination = state_root / source_commit / DEPLOY_RELEASE_PROVENANCE_NAME
    superseded_iteration = (
        destination.parent / SUPERSEDED_ITERATION_PROVENANCE_NAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ControlPlaneDeployError("deploy_release_provenance_destination_symlink")
    superseded_receipt: dict[str, Any] | None = None
    try:
        if destination.exists():
            if not destination.is_file():
                raise ControlPlaneDeployError("deploy_release_provenance_conflict")
            existing_payload = destination.read_bytes()
            if existing_payload != payload:
                try:
                    existing_receipt = json.loads(existing_payload)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise ControlPlaneDeployError(
                        "deploy_release_provenance_conflict"
                    ) from exc
                existing_claim = (
                    existing_receipt.get("claim_boundary")
                    if isinstance(existing_receipt, Mapping)
                    else None
                )
                incoming_claim = receipt.get("claim_boundary")
                is_same_commit_iteration = (
                    isinstance(existing_receipt, Mapping)
                    and existing_receipt.get("schema_version")
                    == "blueprint.deploy_release_provenance.v1"
                    and existing_receipt.get("status") == "iteration"
                    and existing_receipt.get("git_sha") == source_commit
                    and existing_receipt.get("promotion_eligible") is False
                    and isinstance(existing_claim, Mapping)
                    and existing_claim.get("canonical_full_lane_verified") is False
                    and existing_claim.get("promotion_eligible") is False
                )
                is_verified_upgrade = (
                    receipt.get("schema_version")
                    == "blueprint.deploy_release_provenance.v1"
                    and receipt.get("status") == "verified"
                    and receipt.get("git_sha") == source_commit
                    and receipt.get("promotion_eligible") is True
                    and isinstance(incoming_claim, Mapping)
                    and incoming_claim.get("canonical_full_lane_verified") is True
                )
                if not (is_same_commit_iteration and is_verified_upgrade):
                    raise ControlPlaneDeployError(
                        "deploy_release_provenance_conflict"
                    )
                if superseded_iteration.is_symlink():
                    raise ControlPlaneDeployError(
                        "deploy_release_provenance_supersession_conflict"
                    )
                if superseded_iteration.exists():
                    if (
                        not superseded_iteration.is_file()
                        or superseded_iteration.read_bytes() != existing_payload
                    ):
                        raise ControlPlaneDeployError(
                            "deploy_release_provenance_supersession_conflict"
                        )
                else:
                    with superseded_iteration.open("xb") as handle:
                        handle.write(existing_payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.chmod(superseded_iteration, 0o440)

                temporary_fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{DEPLOY_RELEASE_PROVENANCE_NAME}.",
                    suffix=".tmp",
                    dir=destination.parent,
                )
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(temporary_fd, "wb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.chmod(temporary, 0o440)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                superseded_receipt = {
                    "path": str(superseded_iteration),
                    "sha256": _sha256_bytes(existing_payload),
                    "size_bytes": len(existing_payload),
                    "git_sha": source_commit,
                    "status": "iteration",
                    "mode": "0440",
                }
        else:
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.chmod(destination, 0o440)
        reopened = destination.read_bytes()
    except OSError as exc:
        raise ControlPlaneDeployError("deploy_release_provenance_install_failed") from exc
    if reopened != payload:
        raise ControlPlaneDeployError("deploy_release_provenance_readback_mismatch")
    installed = {
        "path": str(destination),
        "sha256": _sha256_bytes(reopened),
        "size_bytes": len(reopened),
        "git_sha": source_commit,
        "run_id": receipt.get("run_id"),
        "run_url": receipt.get("run_url"),
        # Report what the installed receipt actually claims. Hardcoding
        # True told every reader of a deploy receipt that an iteration
        # release had passed the canonical Full Test Lane, while the
        # provenance file it summarised correctly said it had not.
        "canonical_full_lane_verified": bool(
            (receipt.get("claim_boundary") or {}).get(
                "canonical_full_lane_verified"
            )
        ),
        "promotion_eligible": bool(receipt.get("promotion_eligible")),
        "provenance_status": receipt.get("status"),
        "mode": "0440",
    }
    if superseded_receipt is not None:
        installed["superseded_iteration_provenance"] = superseded_receipt
    return installed


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


def _install_scene_object_discovery_runtime_directories(
    *,
    directories: Sequence[str] = DEFAULT_SCENE_OBJECT_DISCOVERY_RUNTIME_DIRECTORIES,
    account: str = DEFAULT_SERVICE_ACCOUNT,
) -> list[dict[str, Any]]:
    """Install the no-spend discovery queue and materialization roots.

    Exact-release deployment must be sufficient on an already-provisioned host;
    requiring an operator to remember to rerun the broad bootstrap installer
    leaves the newly installed path unit watching an absent directory.
    """

    account_ids = _service_account_ids(account)
    if account_ids is None:
        raise ControlPlaneDeployError(
            f"deploy_scene_object_discovery_account_missing:{account}"
        )
    owner_uid, owner_gid = account_ids
    receipts: list[dict[str, Any]] = []
    for raw_path in directories:
        path = Path(raw_path)
        if not path.is_absolute():
            raise ControlPlaneDeployError(
                "deploy_scene_object_discovery_directory_not_absolute"
            )
        if path.is_symlink():
            raise ControlPlaneDeployError(
                f"deploy_scene_object_discovery_directory_symlink:{path}"
            )
        try:
            path.mkdir(parents=True, exist_ok=True, mode=0o750)
            if path.is_symlink() or not path.is_dir():
                raise ControlPlaneDeployError(
                    f"deploy_scene_object_discovery_directory_invalid:{path}"
                )
            os.chown(path, owner_uid, owner_gid)
            path.chmod(0o750)
            stat_result = path.stat()
        except OSError as exc:
            raise ControlPlaneDeployError(
                f"deploy_scene_object_discovery_directory_install_failed:{path}"
            ) from exc
        if (
            stat_result.st_uid != owner_uid
            or stat_result.st_gid != owner_gid
            or stat.S_IMODE(stat_result.st_mode) != 0o750
        ):
            raise ControlPlaneDeployError(
                f"deploy_scene_object_discovery_directory_readback_mismatch:{path}"
            )
        receipts.append(
            {
                "path": str(path),
                "account": account,
                "owner_uid": owner_uid,
                "owner_gid": owner_gid,
                "mode": "0750",
            }
        )
    return receipts


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


def _install_release_systemd_units(
    *,
    release_path: str | Path,
    systemd_dir: str | Path,
    units: Sequence[str] = DEFAULT_DEPLOYED_SYSTEMD_UNITS,
) -> list[dict[str, Any]]:
    """Install exact release-owned unit bytes before daemon reload.

    Promoting a detached release without refreshing its installed unit left the
    dispatcher on older concurrency and watchdog-survival semantics.  The
    allocator then ran exact new Python under stale systemd controls and failed
    before provider allocation.  Install only the release-owned Task Evaluation
    queue pairs here; the ordinary restart seam immediately daemon-reloads them,
    and the next queue activation therefore uses the same release that authored
    the request or profile.

    The pair includes the queue-watching ``.path`` unit: a release that changed
    how the queue wakes the dispatcher (PR #1057 added ``PathChanged=``) was
    otherwise deployed with only its ``.service`` refreshed, leaving the
    watcher on whatever bytes an operator had once copied by hand.
    """

    release = Path(release_path).expanduser().resolve()
    destination_root = Path(systemd_dir).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    for unit in units:
        if Path(unit).name != unit or not unit.endswith(
            DEPLOYED_SYSTEMD_UNIT_SUFFIXES
        ):
            raise ControlPlaneDeployError("deploy_systemd_unit_name_invalid")
        source = release / "deploy" / "systemd" / unit
        destination = destination_root / unit
        if source.is_symlink() or not source.is_file():
            raise ControlPlaneDeployError(f"deploy_systemd_unit_source_invalid:{unit}")
        if destination.is_symlink():
            raise ControlPlaneDeployError(
                f"deploy_systemd_unit_destination_symlink:{unit}"
            )
        try:
            payload = source.read_bytes()
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{unit}.", suffix=".tmp", dir=destination_root
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o644)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            reopened = destination.read_bytes()
        except OSError as exc:
            raise ControlPlaneDeployError(
                f"deploy_systemd_unit_install_failed:{unit}"
            ) from exc
        if reopened != payload or destination.stat().st_mode & 0o777 != 0o644:
            raise ControlPlaneDeployError(
                f"deploy_systemd_unit_readback_mismatch:{unit}"
            )
        receipts.append(
            {
                "unit": unit,
                "source_path": str(source),
                "installed_path": str(destination),
                "sha256": _sha256_bytes(reopened),
                "size_bytes": len(reopened),
                "mode": "0644",
            }
        )
    return receipts


def _systemd_unit_state(unit: str) -> dict[str, str]:
    """Read enabled/active state without changing the unit."""

    states: dict[str, str] = {}
    for probe in ("is-enabled", "is-active"):
        try:
            result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
                ["systemctl", probe, unit],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise ControlPlaneDeployError(
                f"deploy_systemd_state_probe_failed:{unit}:{probe}"
            ) from exc
        state = result.stdout.strip() or (
            "disabled" if probe == "is-enabled" else "inactive"
        )
        if state in {"not-found", "unknown"}:
            state = "disabled" if probe == "is-enabled" else "inactive"
        states[probe] = state
    return {"enabled": states["is-enabled"], "state": states["is-active"]}


def _installed_path_unit_states(
    installed_units: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """Snapshot watcher intent before unit bytes or daemon state move."""

    return {
        unit: _systemd_unit_state(unit)
        for entry in installed_units
        if (unit := str(entry.get("unit") or "")).endswith(".path")
    }


def _quiesce_active_path_units(
    before: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    """Stop only watchers that were active, before release surfaces move.

    Paid-launch locks stop provider allocation, but an armed watcher could
    still claim a newly published request while source and release identities
    are changing.  Quiesce it first, retain the prior state, and restore that
    exact intent only after intake proves the new commit.
    """

    stopped: list[dict[str, str]] = []
    for unit, state in before.items():
        if state.get("state") != "active":
            continue
        result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["systemctl", "stop", unit],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ControlPlaneDeployError(
                f"deploy_path_unit_quiesce_failed:{unit}"
            )
        observed = _systemd_unit_state(unit)
        if observed["state"] != "inactive":
            raise ControlPlaneDeployError(
                f"deploy_path_unit_quiesce_state_mismatch:{unit}:"
                f"{observed['state']}"
            )
        stopped.append({"unit": unit, "state": observed["state"]})
    return stopped


def _restore_installed_path_units(
    installed_units: Sequence[Mapping[str, Any]],
    *,
    before: Mapping[str, Mapping[str, str]],
    arm_path_units: bool,
    always_arm_units: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Install new watcher bytes without widening operator launch authority.

    A stopped watcher is an operational freeze.  Deploy used to unconditionally
    ``enable`` and ``restart`` every installed path unit, silently reopening the
    paid queue.  Preserve both the boot and active state by default.  A fresh
    host therefore stays disabled/inactive.  The only widening operation is the
    explicit ``arm_path_units`` flag, whose before/requested/after state is
    retained in the deployment receipt.  A release may separately name a
    hard-coded no-spend watcher in ``always_arm_units``; this cannot be supplied
    by an HTTP request or launch profile and is retained as ``arm_no_spend``.
    """

    receipts: list[dict[str, Any]] = []
    for entry in installed_units:
        unit = str(entry.get("unit") or "")
        if not unit.endswith(".path"):
            continue
        prior = dict(before.get(unit) or {"enabled": "disabled", "state": "inactive"})
        arm_no_spend = unit in always_arm_units
        should_enable = (
            arm_path_units or arm_no_spend or prior.get("enabled") == "enabled"
        )
        should_start = (
            arm_path_units or arm_no_spend or prior.get("state") == "active"
        )
        commands = ["enable" if should_enable else "disable"]
        commands.append("restart" if should_start else "stop")
        for verb in commands:
            result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
                ["systemctl", verb, unit],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise ControlPlaneDeployError(
                    f"deploy_path_unit_state_restore_failed:{unit}:{verb}"
                )
        after = _systemd_unit_state(unit)
        expected_enabled = "enabled" if should_enable else "disabled"
        expected_state = "active" if should_start else "inactive"
        if after["enabled"] != expected_enabled:
            raise ControlPlaneDeployError(
                f"deploy_path_unit_enabled_state_mismatch:{unit}:"
                f"{after['enabled']}:{expected_enabled}"
            )
        if after["state"] != expected_state:
            raise ControlPlaneDeployError(
                f"deploy_path_unit_active_state_mismatch:{unit}:"
                f"{after['state']}:{expected_state}"
            )
        receipts.append(
            {
                "unit": unit,
                "before": prior,
                "requested_intent": (
                    "arm"
                    if arm_path_units
                    else "arm_no_spend"
                    if arm_no_spend
                    else "preserve"
                ),
                "after": after,
                "operator_freeze_preserved": (
                    not arm_path_units and not arm_no_spend and not should_start
                ),
            }
        )
    return receipts


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
        "BLUEPRINT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT="
        f"{DEFAULT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT}\n"
    )
    drop_in_content = (
        "# Managed by scripts/deploy_control_plane_commit.py.\n"
        "# Loaded after the base unit credential EnvironmentFile.\n"
        "[Service]\n"
        f"EnvironmentFile={identity_env}\n"
        f"TimeoutStartSec={INTAKE_START_TIMEOUT_SECONDS}s\n"
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
        "timeout_start_seconds": INTAKE_START_TIMEOUT_SECONDS,
        "credential_environment_file_opened": False,
        "credential_values_recorded": False,
    }


def _install_scene_configuration_environment(
    path: Path, *, environment: Mapping[str, str]
) -> dict[str, Any]:
    """Atomically install exact-release, non-secret scene runtime bindings."""

    expected_names = {
        "BLUEPRINT_TASK_EVALUATION_SPLAT_RENDER_RUNTIME_ROOT",
        "BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT",
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_RELEASE_WINDOW_PREFIX",
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_DESTINATION_PREFIX",
    }
    if (
        set(environment) != expected_names
        or path.is_symlink()
        or (path.exists() and not stat.S_ISREG(path.stat().st_mode))
        or any(
            not value
            or "\n" in value
            or "\r" in value
            or any(character.isspace() for character in value)
            for value in environment.values()
        )
    ):
        raise ControlPlaneDeployError("deploy_scene_configuration_environment_invalid")
    content = (
        "# Managed by scripts/deploy_control_plane_commit.py.\n"
        "# Exact-release paths and public object-store prefixes only; no credentials.\n"
        + "".join(f"{name}={environment[name]}\n" for name in sorted(environment))
    )
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ControlPlaneDeployError(
            "deploy_scene_configuration_environment_update_failed"
        ) from exc
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()
    payload = path.read_bytes()
    if payload != content.encode("utf-8"):
        raise ControlPlaneDeployError(
            "deploy_scene_configuration_environment_readback_mismatch"
        )
    return {
        "path": str(path),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "mode": "0644",
        "credential_values_recorded": False,
        "environment_names": sorted(environment),
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
    release_provenance: str | Path | None = None,
    iteration: bool = False,
    restart_units: Sequence[str] = DEFAULT_RESTART_UNITS,
    paid_launch_locks: Sequence[str] = DEFAULT_PAID_LAUNCH_LOCKS,
    intake_runtime_drop_in: str | Path = DEFAULT_INTAKE_RUNTIME_DROP_IN,
    intake_version_url: str = DEFAULT_INTAKE_VERSION_URL,
    systemd_dir: str | Path = DEFAULT_SYSTEMD_DIR,
    scene_configuration_environment_file: str | Path = (
        DEFAULT_SCENE_CONFIGURATION_ENVIRONMENT_FILE
    ),
    scene_configuration_runtime_root: str | Path = (
        DEFAULT_SCENE_CONFIGURATION_RUNTIME_ROOT
    ),
    splat_render_prerequisite_root: str | Path = (
        DEFAULT_SPLAT_RENDER_PREREQUISITE_ROOT
    ),
    artifixer_source_root: str | Path = DEFAULT_ARTIFIXER_SOURCE_ROOT,
    content_agents_source_root: str | Path = DEFAULT_CONTENT_AGENTS_SOURCE_ROOT,
    arm_path_units: bool = False,
) -> dict[str, Any]:
    """Move the mutable clone and the release link, then verify both."""

    source = Path(source_repo).expanduser().resolve()
    active = Path(active_link).expanduser()
    releases = Path(release_root).expanduser().resolve()
    raw_state = Path(state_root).expanduser()
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ControlPlaneDeployError("deploy_source_commit_invalid")
    if not raw_state.is_absolute():
        raise ControlPlaneDeployError("deploy_state_root_must_be_absolute")
    state = raw_state.resolve()
    if (
        state == source
        or state in source.parents
        or source in state.parents
        or state == releases
        or state in releases.parents
    ):
        raise ControlPlaneDeployError("deploy_state_root_overlaps_checkout")
    if iteration:
        # An iteration deploy trades promotion evidence for cycle time. The
        # full lane takes ~15 minutes; a fix-and-fire loop that waits for it
        # costs ~18 minutes per attempt, which dominates a campaign of dozens
        # of GPU runs. What must NOT be traded away is knowing exactly which
        # bytes ran: the release is still built from a real pushed commit, so
        # the running code and main can never silently diverge.
        #
        # The receipt says plainly that no lane verified it.
        # `_commit_has_verified_production_promotion` requires
        # status == "verified", so this can never be mistaken for a promoted
        # release, and paid admission still refuses it as an ancestor.
        if release_provenance is not None:
            raise ControlPlaneDeployError("deploy_iteration_provenance_conflict")
        # The guard belongs here, not in a wrapper script. Within an hour of
        # the wrapper being written its `git fetch` hit a permission error and
        # the obvious workaround was to call this tool directly -- which then
        # had no ancestry check at all. That is how a guard dies: the wrapper
        # is inconvenient once and the bypass becomes the habit.
        #
        # Iteration exists to skip the LANE, never to skip main. A commit that
        # is not an ancestor of origin/main is exactly the local-only drift
        # this mode replaces, so refuse it however this tool was invoked.
        # Fail closed: a stale or missing origin/main refuses rather than
        # admits, and the operator fetches and retries.
        code, _ = _git(source, "merge-base", "--is-ancestor", source_commit, "origin/main")
        if code != 0:
            raise ControlPlaneDeployError("deploy_iteration_commit_not_on_origin_main")
        provenance_receipt = {
            "schema_version": "blueprint.deploy_release_provenance.v1",
            "status": "iteration",
            "git_sha": source_commit,
            "promotion_eligible": False,
            "claim_boundary": {
                "canonical_full_lane_verified": False,
                "promotion_eligible": False,
                "evidence_grade": "development_only",
            },
        }
        provenance_payload = (
            json.dumps(provenance_receipt, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    else:
        if release_provenance is None:
            raise ControlPlaneDeployError("deploy_release_provenance_missing")
        _provenance_source, provenance_payload, provenance_receipt = (
            _validated_release_provenance(
                release_provenance, source_commit=source_commit
            )
        )
        provenance_receipt = dict(provenance_receipt)
        provenance_receipt.setdefault("promotion_eligible", True)

    # Held for the whole deploy, not sampled before it: a launch that starts
    # mid-deploy would read a release being swapped underneath it.
    with _holding_paid_launch_locks(paid_launch_locks):
        path_unit_names = [
            {"unit": unit}
            for unit in DEFAULT_DEPLOYED_SYSTEMD_UNITS
            if unit.endswith(".path")
        ]
        path_unit_states_before = _installed_path_unit_states(path_unit_names)
        quiesced_path_units = _quiesce_active_path_units(
            path_unit_states_before
        )
        installed_provenance = _install_release_provenance(
            payload=provenance_payload,
            state_root=state,
            source_commit=source_commit,
            receipt=provenance_receipt,
        )
        staged_release = stage_task_evaluation_control_plane_release(
            source_repo=source,
            source_commit=source_commit,
            release_root=release_root,
            state_root=state_root,
            active_link=active,
            activate=False,
        )
        try:
            prerequisite = validate_splat_render_prerequisites(
                root=splat_render_prerequisite_root,
                repository_root=staged_release["release_path"],
            )
            prerequisite_entrypoints = prerequisite["entrypoints"]
            scene_configuration_runtime = provision_scene_configuration_release(
                repository_root=staged_release["release_path"],
                source_commit=source_commit,
                runtime_root=scene_configuration_runtime_root,
                node_executable=prerequisite_entrypoints["node"],
                browser_root=prerequisite_entrypoints["browser_root"],
                browser_executable=prerequisite_entrypoints["browser"],
                node_modules_root=prerequisite_entrypoints["node_modules"],
                artifixer_root=artifixer_source_root,
                content_agents_root=content_agents_source_root,
                readback=service_account_readback(DEFAULT_SERVICE_ACCOUNT),
                readback_actor=f"service-account:{DEFAULT_SERVICE_ACCOUNT}",
            )
        except ValueError as exc:
            raise ControlPlaneDeployError(
                f"deploy_scene_configuration_runtime_invalid:{exc}"
            ) from exc
        scene_configuration_environment = _install_scene_configuration_environment(
            Path(scene_configuration_environment_file).expanduser(),
            environment=scene_configuration_runtime["environment"],
        )
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

        installed_systemd_units = _install_release_systemd_units(
            release_path=release["release_path"],
            systemd_dir=systemd_dir,
        )
        scene_object_discovery_runtime_directories = (
            _install_scene_object_discovery_runtime_directories()
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
        # Last inside the held locks: the queue watcher only starts watching
        # once the restarted intake has proven the new commit, and no launch
        # can slip in between the watcher restart and the lock release.
        path_unit_state_receipts = _restore_installed_path_units(
            installed_systemd_units,
            before=path_unit_states_before,
            arm_path_units=arm_path_units,
            always_arm_units=DEFAULT_ALWAYS_ARM_PATH_UNITS,
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
        "release_provenance": installed_provenance,
        "created_release_checkout": release["created_release_checkout"],
        "restarted_units": restarted,
        "installed_systemd_units": installed_systemd_units,
        "scene_object_discovery_runtime_directories": (
            scene_object_discovery_runtime_directories
        ),
        "path_unit_states": path_unit_state_receipts,
        "quiesced_path_units": quiesced_path_units,
        # Compatibility projection for readers that predate the state-preserving
        # receipt. It contains only watchers that are active after this deploy.
        "activated_path_units": [
            {
                "unit": row["unit"],
                "enabled": row["after"]["enabled"],
                "state": row["after"]["state"],
            }
            for row in path_unit_state_receipts
            if row["after"]["state"] == "active"
        ],
        "intake_runtime_binding": runtime_binding,
        "intake_runtime": runtime,
        "scene_configuration_runtime": scene_configuration_runtime,
        "scene_configuration_environment": scene_configuration_environment,
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
        "--release-provenance",
        default=None,
        help=(
            "Exact verified blueprint.deploy_release_provenance.v1 receipt "
            "for --source-commit. Required unless --iteration is given."
        ),
    )
    parser.add_argument(
        "--iteration",
        action="store_true",
        help=(
            "Deploy a pushed commit without waiting for the Full Test Lane. "
            "The release is stamped promotion_eligible=false and evidence "
            "grade development_only. Use for fix-and-fire iteration; promote "
            "with a lane-verified deploy before sealing evidence."
        ),
    )
    parser.add_argument(
        "--systemd-dir",
        default=DEFAULT_SYSTEMD_DIR,
        help="systemd unit directory receiving exact release-owned unit bytes",
    )
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
    parser.add_argument(
        "--scene-configuration-environment-file",
        default=DEFAULT_SCENE_CONFIGURATION_ENVIRONMENT_FILE,
    )
    parser.add_argument(
        "--scene-configuration-runtime-root",
        default=DEFAULT_SCENE_CONFIGURATION_RUNTIME_ROOT,
    )
    parser.add_argument(
        "--splat-render-prerequisite-root",
        default=DEFAULT_SPLAT_RENDER_PREREQUISITE_ROOT,
    )
    parser.add_argument("--artifixer-source-root", default=DEFAULT_ARTIFIXER_SOURCE_ROOT)
    parser.add_argument(
        "--content-agents-source-root", default=DEFAULT_CONTENT_AGENTS_SOURCE_ROOT
    )
    parser.add_argument(
        "--arm-path-units",
        action="store_true",
        help=(
            "Explicitly enable and start release-owned path watchers. By default "
            "deploy preserves the prior enabled/active state and leaves fresh "
            "installations disarmed."
        ),
    )
    args = parser.parse_args(argv)

    try:
        receipt = deploy_control_plane_commit(
            source_repo=args.source_repo,
            source_commit=args.source_commit,
            release_root=args.release_root,
            state_root=args.state_root,
            active_link=args.active_link,
            release_provenance=args.release_provenance,
            iteration=args.iteration,
            restart_units=tuple(args.restart_unit or ()),
            paid_launch_locks=tuple(args.paid_launch_lock or DEFAULT_PAID_LAUNCH_LOCKS),
            intake_runtime_drop_in=args.intake_runtime_drop_in,
            intake_version_url=args.intake_version_url,
            systemd_dir=args.systemd_dir,
            scene_configuration_environment_file=(
                args.scene_configuration_environment_file
            ),
            scene_configuration_runtime_root=args.scene_configuration_runtime_root,
            splat_render_prerequisite_root=args.splat_render_prerequisite_root,
            artifixer_source_root=args.artifixer_source_root,
            content_agents_source_root=args.content_agents_source_root,
            arm_path_units=args.arm_path_units,
        )
    except (OSError, ValueError, ControlPlaneReleaseError) as exc:
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
