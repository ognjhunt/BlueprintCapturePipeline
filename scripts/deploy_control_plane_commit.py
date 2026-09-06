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
from blueprint_pipeline.task_evaluation_configured_controls_autostart import (  # noqa: E402
    configured_controls_autostart_registry_name,
    validate_configured_controls_autostart_intent,
)
from blueprint_pipeline.control_plane_disk_budget import (  # noqa: E402
    ControlPlaneDiskBudgetError,
    reserve_control_plane_disk,
)
from blueprint_pipeline.control_plane_storage_pins import (  # noqa: E402
    DEFAULT_PINS_ROOT,
)
from blueprint_pipeline.control_plane_release_retirement import (  # noqa: E402
    ControlPlaneReleaseRetirementError,
    EXECUTE_ACK as RELEASE_RETIREMENT_ACK,
    apply_release_retirement_plan,
    build_release_retirement_plan,
)
from blueprint_pipeline.production_cad_skill_sources import (  # noqa: E402
    DEFAULT_ROOT as DEFAULT_CAD_SKILL_SOURCE_ROOT,
    ProductionCadSkillSourcesError,
    provision_production_cad_skill_sources,
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
    "blueprint-task-evaluation-sam31-preparation-execution.service",
    "blueprint-task-evaluation-sam31-preparation-execution.path",
    "blueprint-task-evaluation-sam31-preparation-execution.timer",
    "blueprint-task-evaluation-episode-compilation.service",
    "blueprint-task-evaluation-episode-compilation.path",
    "blueprint-task-evaluation-launch-activation.service",
    "blueprint-task-evaluation-launch-activation.path",
    "blueprint-task-evaluation-policy-canary-dispatcher.service",
    "blueprint-task-evaluation-policy-canary-dispatcher.path",
    "blueprint-scene-object-discovery.service",
    "blueprint-scene-object-discovery.path",
    "blueprint-task-evaluation-configured-controls-progression.service",
    "blueprint-task-evaluation-configured-controls-progression.timer",
    "blueprint-task-evaluation-configured-controls-progression.path",
    "blueprint-task-evaluation-scene-progression.service",
    "blueprint-task-evaluation-scene-progression.timer",
    "blueprint-task-evaluation-launch-supervisor.service",
    "blueprint-task-evaluation-launch-supervisor.timer",
    "blueprint-task-evaluation-launch-reconciler.service",
    "blueprint-task-evaluation-launch-reconciler.timer",
    "blueprint-task-evaluation-terminal-resource-release.service",
    "blueprint-task-evaluation-terminal-resource-release.path",
    "blueprint-gpu-spend-guard.service",
    "blueprint-gpu-spend-guard.timer",
    "blueprint-control-plane-storage-gc.service",
    "blueprint-control-plane-storage-gc.timer",
    "blueprint-control-plane-capacity.service",
    "blueprint-control-plane-capacity.timer",
    "blueprint-control-plane-preflight.service",
    "blueprint-control-plane-preflight.timer",
    "blueprint-pipeline-control-plane.service",
    "blueprint-pipeline-intake.service",
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
#: Paid execution is still impossible without a consumed, digest-bound
#: activation authority, a clear global spend guard, and the provider-zero
#: preflight inside the dispatcher.  Keeping this watcher active therefore
#: restores Website-to-GPU liveness without granting spend authority by
#: itself.  It is separated from the no-spend watchers so deploy receipts do
#: not blur that distinction.
DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS = (
    "blueprint-task-evaluation-sam31-preparation-execution.path",
    "blueprint-task-evaluation-policy-canary-dispatcher.path",
)
#: This fixed timer advances only a sealed, qualifying configured-scene plan
#: through the canonical Website APIs.  It cannot be supplied by a request or
#: launch profile and never invokes an allocator directly, but unlike the
#: no-spend queue watchers above it may eventually reach already-authorized
#: downstream spend.  Keep that authority distinct in the deployment receipt.
#: The compilation-result path watcher wakes the same oneshot service the
#: moment a no-spend canary compiles, so it carries the same progression
#: authority rather than the no-spend watcher category.
DEFAULT_ALWAYS_ARM_TIMER_UNITS = (
    "blueprint-task-evaluation-scene-progression.timer",
    "blueprint-task-evaluation-sam31-preparation-execution.timer",
    "blueprint-task-evaluation-configured-controls-progression.timer",
    "blueprint-task-evaluation-configured-controls-progression.path",
    # The storage reaper is no-spend housekeeping: it only ever removes
    # unpinned cache bytes and offloads sealed evidence behind pointers.
    "blueprint-control-plane-storage-gc.timer",
    "blueprint-control-plane-capacity.timer",
    "blueprint-control-plane-preflight.timer",
)
#: JSON under these roots names the commits that a launch may still need; a
#: commit named anywhere here is never retired by the deploy that supersedes it.
DEFAULT_RELEASE_RETIREMENT_REFERENCE_ROOTS = (
    "/etc/blueprint/task-evaluation-launch-profiles",
    # Terminal evidence can still require an older renderer after its queues empty.
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-release-retention-bindings",
    "/var/lib/blueprint/pipeline-control-plane/standing-authorizations",
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launches/pending",
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launches/processing",
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-preparations/pending",
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-preparations/processing",
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-preparations/awaiting_source_preparation",
    "/var/lib/blueprint/pipeline-control-plane/sam31-preparation-executions/pending",
    "/var/lib/blueprint/pipeline-control-plane/sam31-preparation-executions/processing",
    "/var/lib/blueprint/pipeline-control-plane/sam31-preparation-executions/waiting_external",
    "/var/lib/blueprint/pipeline-control-plane/sam31-preparation-executions/wake-pending",
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-episode-compilations/pending",
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-episode-compilations/processing",
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-activations/pending",
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-activations/processing",
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-policy-canary-dispatches/pending",
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-policy-canary-dispatches/processing",
)
DEFAULT_RELEASE_RETIREMENT_KEEP_LAST = 3
#: The only unit kinds a release may install.  Services and their queue-watching
#: paths stay paired, while the one fixed progression timer (and its
#: compilation-result path watcher) stays paired with its oneshot service.
#: Sockets, mounts, and anything else remain refused.
DEPLOYED_SYSTEMD_UNIT_SUFFIXES = (".service", ".path", ".timer")
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
DEFAULT_EPISODE_COMPILATION_QUEUE_ROOT = (
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-episode-compilations"
)
DEFAULT_EPISODE_COMPILATION_RUNTIME_DIRECTORIES = (
    DEFAULT_EPISODE_COMPILATION_QUEUE_ROOT,
    *(
        f"{DEFAULT_EPISODE_COMPILATION_QUEUE_ROOT}/{name}"
        for name in ("pending", "processing", "completed", "blocked")
    ),
)
DEFAULT_CONFIGURED_CONTROLS_PLAN_ROOT = (
    "/etc/blueprint/task-evaluation-configured-controls-plans"
)
DEFAULT_CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT = (
    "/etc/blueprint/task-evaluation-configured-controls-intents"
)
DEFAULT_CONFIGURED_CONTROLS_WEBAPP_SECRET = (
    "/etc/blueprint/provider-secrets/blueprint_task_evaluation_launch_submit_secret"
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
INTAKE_START_TIMEOUT_SECONDS = 300
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
    # Inside the installer, not at the call site: promotion proof the reader
    # cannot open is indistinguishable from proof that was never installed, so
    # no caller gets to skip this.
    provenance_access = _install_release_provenance_access(
        destination,
        superseded_iteration if superseded_receipt is not None else None,
    )
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
        "service_account_access": provenance_access,
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
        metadata = path.stat()
        changed = False
        if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
            chown(path, owner_uid, owner_gid)
            changed = True
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            path.chmod(0o600)
            changed = True
        if changed:
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



def _service_account_read_blocker(
    path: Path, *, owner_uid: int, owner_gid: int
) -> str | None:
    """Report why `owner_uid:owner_gid` cannot read `path`, or None if it can.

    Every ancestor directory is checked for traverse permission, not just the
    file's own read bit. A directory the service account cannot enter hides a
    perfectly readable file beneath it, and that is the shape the live control
    plane actually took: 304 root-owned directories, no `o+x`, holding
    promotion proof the blueprint-run services could not reach.
    """

    def _granted(metadata: os.stat_result, want: int) -> bool:
        mode = stat.S_IMODE(metadata.st_mode)
        if metadata.st_uid == owner_uid:
            bits = (mode >> 6) & 0o7
        elif metadata.st_gid == owner_gid:
            bits = (mode >> 3) & 0o7
        else:
            bits = mode & 0o7
        return bool(bits & want)

    for ancestor in reversed(path.parents):
        try:
            metadata = ancestor.stat()
        except OSError:
            return f"unstatable_directory:{ancestor}"
        if not _granted(metadata, 0o1):
            return f"untraversable_directory:{ancestor}"
    try:
        metadata = path.stat()
    except OSError:
        return f"unstatable_file:{path}"
    if not _granted(metadata, 0o4):
        return f"unreadable_file:{path}"
    return None


def _grant_service_account_read(
    paths: Sequence[Path], *, owner_gid: int, chown: Any = os.chown
) -> list[str]:
    """Give the service account group-read on what deploy just wrote.

    Only the group is moved; the owning uid is left alone deliberately. The
    provenance receipt is installed 0440 precisely so that nothing can rewrite
    it, and a root-owned file the service account merely reads keeps that
    property. Chowning it to the service account would hand the reader the
    power to chmod its own promotion proof.
    """

    adjusted: list[str] = []
    for path in paths:
        try:
            metadata = path.stat()
        except OSError:
            continue
        changed = False
        if metadata.st_gid != owner_gid:
            chown(path, -1, owner_gid)
            changed = True
            metadata = path.stat()
        wanted = 0o050 if path.is_dir() else 0o040
        mode = stat.S_IMODE(metadata.st_mode)
        if mode & wanted != wanted:
            path.chmod(mode | wanted)
            changed = True
        if changed:
            adjusted.append(str(path))
    return adjusted


def _install_release_provenance_access(
    destination: Path,
    superseded: Path | None,
    *,
    account: str = DEFAULT_SERVICE_ACCOUNT,
    chown: Any = os.chown,
) -> dict[str, Any]:
    """Make the promotion proof readable by the account that consumes it, then prove it.

    Deploy runs as root; every service that reads this file runs as
    `blueprint`. `os.chmod(..., 0o440)` alone therefore installed `root:root`
    promotion proof that the reader could not open, so a host could pass its
    whole deploy and still have every launch fall back to `development_only`
    -- with no error anywhere, because nothing asserted the reader's side.

    The grant is not trusted on its own. The gate below re-derives readability
    from the installed inode and fails the deploy if the service account still
    cannot reach it, so this can never silently regress to a no-op.
    """

    account_ids = _service_account_ids(account)
    if account_ids is None:
        return {
            "status": "not_applicable_no_service_account",
            "account": account,
            "adjusted_paths": [],
        }
    owner_uid, owner_gid = account_ids
    targets: list[Path] = [destination.parent, destination]
    if superseded is not None:
        targets.append(superseded)
    adjusted = _grant_service_account_read(
        targets, owner_gid=owner_gid, chown=chown
    )
    for path in targets:
        blocker = _service_account_read_blocker(
            path, owner_uid=owner_uid, owner_gid=owner_gid
        )
        if blocker is not None:
            raise ControlPlaneDeployError(
                "deploy_release_provenance_unreadable_by_service_account:"
                f"{account}:{blocker}"
            )
    return {
        "status": "readable",
        "account": account,
        "owner_uid": owner_uid,
        "owner_gid": owner_gid,
        "adjusted_paths": adjusted,
        "verified_paths": [str(path) for path in targets],
    }


def _install_disk_reservation_runtime_prerequisites(
    reservation_root: str | Path,
    *,
    account: str = DEFAULT_SERVICE_ACCOUNT,
    root_uid: int = 0,
    chown: Any = os.chown,
    stat_reader: Any = lambda path: path.stat(),
) -> dict[str, Any]:
    """Install the shared disk ledger so root and runtime workers can use it.

    Deploy reserves disk as root, while every queue worker reserves disk as the
    ``blueprint`` service account. Merely creating the ledger with mode 2770 is
    insufficient: a root-created directory and lock retain ``root:root``
    ownership, leaving the runtime unable to enter the directory or open the
    lock after an otherwise successful deploy.

    Reconcile both inodes before any service restart and verify their installed
    ownership and modes instead of trusting the privileged mutations.
    """

    account_ids = _service_account_ids(account)
    if account_ids is None:
        raise ControlPlaneDeployError(
            f"deploy_disk_reservation_account_missing:{account}"
        )
    _owner_uid, owner_gid = account_ids
    root = Path(reservation_root).expanduser()
    lock = root / ".lock"
    if not root.is_absolute():
        raise ControlPlaneDeployError(
            "deploy_disk_reservation_directory_not_absolute"
        )
    if root.is_symlink() or lock.is_symlink():
        raise ControlPlaneDeployError(
            f"deploy_disk_reservation_runtime_symlink:{root}"
        )

    repaired: list[str] = []
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o2770)
        if root.is_symlink() or not root.is_dir():
            raise ControlPlaneDeployError(
                f"deploy_disk_reservation_directory_invalid:{root}"
            )
        lock.touch(mode=0o660, exist_ok=True)
        if lock.is_symlink() or not lock.is_file():
            raise ControlPlaneDeployError(
                f"deploy_disk_reservation_lock_invalid:{lock}"
            )
        for path, wanted_mode in ((root, 0o2770), (lock, 0o660)):
            metadata = stat_reader(path)
            changed = False
            if metadata.st_uid != root_uid or metadata.st_gid != owner_gid:
                chown(path, root_uid, owner_gid)
                changed = True
                metadata = stat_reader(path)
            if stat.S_IMODE(metadata.st_mode) != wanted_mode:
                path.chmod(wanted_mode)
                changed = True
            if changed:
                repaired.append(str(path))
    except ControlPlaneDeployError:
        raise
    except OSError as exc:
        raise ControlPlaneDeployError(
            f"deploy_disk_reservation_runtime_install_failed:{root}"
        ) from exc

    installed: list[dict[str, Any]] = []
    for path, wanted_mode, kind in (
        (root, 0o2770, "directory"),
        (lock, 0o660, "lock"),
    ):
        metadata = stat_reader(path)
        if (
            metadata.st_uid != root_uid
            or metadata.st_gid != owner_gid
            or stat.S_IMODE(metadata.st_mode) != wanted_mode
        ):
            raise ControlPlaneDeployError(
                f"deploy_disk_reservation_runtime_readback_mismatch:{path}"
            )
        installed.append(
            {
                "kind": kind,
                "path": str(path),
                "owner": "root",
                "group": account,
                "owner_uid": root_uid,
                "owner_gid": owner_gid,
                "mode": f"{wanted_mode:04o}",
            }
        )
    return {
        "status": "ready",
        "account": account,
        "repaired_paths": repaired,
        "installed": installed,
    }


def _retire_superseded_release_trees(
    *,
    release_root: str | Path,
    runtime_root: str | Path,
    active_link: str | Path,
    current_commit: str,
    reference_roots: Sequence[str],
    keep_last: int,
) -> dict[str, Any]:
    """Retire release and runtime trees this deploy has superseded.

    Deploy is the only event that creates per-commit trees, so it is where
    they are retired.  Anything the plan cannot prove safe is left in place and
    reported; a retirement failure never fails a deploy whose surfaces already
    moved.
    """

    try:
        plan = build_release_retirement_plan(
            release_root=release_root,
            runtime_root=runtime_root,
            active_link=active_link,
            current_commit=current_commit,
            protected_reference_roots=list(reference_roots),
            keep_last=keep_last,
        )
        if plan["status"] != "dry_run":
            return {
                "status": "skipped",
                "blockers": list(plan["blockers"]),
                "plan_digest": plan["plan_digest"],
            }
        receipt = apply_release_retirement_plan(
            plan,
            ack=RELEASE_RETIREMENT_ACK,
            active_link=active_link,
            release_root=release_root,
        )
    except (ControlPlaneReleaseRetirementError, OSError, ValueError) as exc:
        return {
            "status": "blocked",
            "blockers": [f"deploy_release_retirement_failed:{type(exc).__name__}"],
        }
    return {
        "status": "applied",
        "plan_digest": plan["plan_digest"],
        "receipt_digest": receipt["result_digest"],
        "retired_commits": sorted({row["commit"] for row in receipt["removed"]}),
        "retired_bytes": plan["candidate_bytes"],
        "protected_commit_count": len(plan["protected_commits"]),
        "unmanaged_children": list(plan["unmanaged_children"]),
        "skipped": list(receipt["skipped"]),
    }


_SANDBOX_DIRECTIVES = ("ReadWritePaths=", "ReadOnlyPaths=")
_UNIT_PROVISIONABLE_PREFIX = "/var/lib/blueprint/"
_UNIT_FILE_SUFFIXES = (".json", ".jsonl", ".lock", ".env", ".sqlite", ".log", ".txt")


def _unit_sandbox_entries(unit_text: str) -> list[tuple[str, bool, str]]:
    """Every ``(path, optional, directive)`` a unit's filesystem sandbox names."""

    entries: list[tuple[str, bool, str]] = []
    for raw_line in unit_text.splitlines():
        line = raw_line.strip()
        for directive in _SANDBOX_DIRECTIVES:
            if not line.startswith(directive):
                continue
            for token in line[len(directive):].split():
                optional = token.startswith("-")
                path_text = token[1:] if optional else token
                if not path_text.startswith("/"):
                    continue
                entries.append((path_text.rstrip("/") or "/", optional, directive[:-1]))
    return entries


def _install_retention_plan_reader_access(root: Path, *, owner_gid: int) -> dict[str, Any]:
    """Repair only directory traversal for retained deployment plans, not file authority."""
    if not root.exists():
        return {"status": "absent", "path": str(root)}
    if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)) or not root.is_dir():
        raise ControlPlaneDeployError("deploy_retention_reader_root_unsafe")
    metadata = root.stat()
    if metadata.st_gid != owner_gid:
        os.chown(root, -1, owner_gid)
    root.chmod(0o750)
    observed = root.stat()
    if observed.st_uid != metadata.st_uid or observed.st_gid != owner_gid or stat.S_IMODE(observed.st_mode) != 0o750:
        raise ControlPlaneDeployError("deploy_retention_reader_readback_failed")
    return {"status": "readable_directory", "path": str(root), "owner_uid": observed.st_uid,
            "reader_gid": observed.st_gid, "mode": "0750", "file_permissions_changed": False}


def _install_unit_sandbox_paths(
    *,
    release_path: str | Path,
    units: Sequence[str] = DEFAULT_DEPLOYED_SYSTEMD_UNITS,
    root_prefix: str | Path | None = None,
    account: str = DEFAULT_SERVICE_ACCOUNT,
    owner_ids: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Make every path a deployed unit's sandbox names exist before the release moves.

    ``ProtectSystem=strict`` units fail to start when a ``ReadWritePaths`` or
    ``ReadOnlyPaths`` entry does not exist.  Twice (the disk-reservation ledger,
    then the storage-pin ledger) a unit gained a path that no deploy step
    created, and the dead worker was discovered only when a Website run
    stalled; each time the fix was one more hand-written installer.  This step
    reads the staged release's own unit files, creates any missing
    service-owned directory under ``/var/lib/blueprint`` (never repairing one
    that exists), and refuses the deploy when a path it may not create -- host
    configuration under ``/etc``, a file, another tree -- is absent and not
    marked optional with a leading ``-``.
    """

    release = Path(release_path).expanduser()
    retention_path = Path("/var/lib/blueprint/pipeline-control-plane/release-retention")
    if root_prefix is not None:
        retention_path = Path(root_prefix) / retention_path.relative_to("/")
    if retention_path.exists():
        reader_ids = owner_ids or _service_account_ids(account)
        if reader_ids is None:
            raise ControlPlaneDeployError("deploy_retention_reader_account_missing")
        _install_retention_plan_reader_access(retention_path, owner_gid=reader_ids[1])
    created: list[dict[str, Any]] = []
    verified: list[dict[str, str]] = []
    blockers: list[str] = []
    pending: list[tuple[str, str, Path]] = []
    seen: set[str] = set()
    for unit in units:
        source = release / "deploy" / "systemd" / unit
        if source.is_symlink() or not source.is_file():
            # The unit installer refuses an absent release unit on its own.
            continue
        for path_text, optional, directive in _unit_sandbox_entries(
            source.read_text(encoding="utf-8")
        ):
            if path_text in seen:
                continue
            seen.add(path_text)
            host_path = (
                Path(path_text)
                if root_prefix is None
                else Path(root_prefix).expanduser() / path_text.lstrip("/")
            )
            if host_path.exists():
                verified.append({"unit": unit, "path": path_text, "directive": directive})
                continue
            if optional:
                continue
            file_like = Path(path_text).suffix in _UNIT_FILE_SUFFIXES
            if path_text.startswith(_UNIT_PROVISIONABLE_PREFIX) and not file_like:
                pending.append((unit, path_text, host_path))
                continue
            blockers.append(f"deploy_unit_sandbox_path_missing:{unit}:{path_text}")
    if blockers:
        raise ControlPlaneDeployError(",".join(sorted(blockers)))
    if pending:
        ids = owner_ids or _service_account_ids(account)
        if ids is None:
            raise ControlPlaneDeployError(f"deploy_unit_sandbox_account_missing:{account}")
        owner_uid, owner_gid = ids
        for unit, path_text, host_path in pending:
            try:
                host_path.mkdir(parents=True, exist_ok=True, mode=0o750)
                if host_path.is_symlink() or not host_path.is_dir():
                    raise ControlPlaneDeployError(
                        f"deploy_unit_sandbox_path_invalid:{unit}:{path_text}"
                    )
                os.chown(host_path, owner_uid, owner_gid)
                host_path.chmod(0o750)
            except OSError as exc:
                raise ControlPlaneDeployError(
                    f"deploy_unit_sandbox_path_install_failed:{unit}:{path_text}"
                ) from exc
            created.append(
                {
                    "unit": unit,
                    "path": path_text,
                    "mode": "0750",
                    "owner_uid": owner_uid,
                    "owner_gid": owner_gid,
                }
            )
    return {
        "status": "ready",
        "unit_count": len(units),
        "verified_count": len(verified),
        "created_count": len(created),
        "created": created,
    }


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


def _install_episode_compilation_runtime_directories(
    *,
    directories: Sequence[str] = DEFAULT_EPISODE_COMPILATION_RUNTIME_DIRECTORIES,
    account: str = DEFAULT_SERVICE_ACCOUNT,
) -> list[dict[str, Any]]:
    """Install every state watched or consumed by the episode queue units."""

    account_ids = _service_account_ids(account)
    if account_ids is None:
        raise ControlPlaneDeployError(
            f"deploy_episode_compilation_account_missing:{account}"
        )
    owner_uid, owner_gid = account_ids
    receipts: list[dict[str, Any]] = []
    for raw_path in directories:
        path = Path(raw_path)
        if not path.is_absolute():
            raise ControlPlaneDeployError(
                "deploy_episode_compilation_directory_not_absolute"
            )
        if path.is_symlink():
            raise ControlPlaneDeployError(
                f"deploy_episode_compilation_directory_symlink:{path}"
            )
        try:
            path.mkdir(parents=True, exist_ok=True, mode=0o750)
            if path.is_symlink() or not path.is_dir():
                raise ControlPlaneDeployError(
                    f"deploy_episode_compilation_directory_invalid:{path}"
                )
            metadata = path.stat()
            if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
                os.chown(path, owner_uid, owner_gid)
            if stat.S_IMODE(metadata.st_mode) != 0o750:
                path.chmod(0o750)
            readback = path.stat()
        except OSError as exc:
            raise ControlPlaneDeployError(
                f"deploy_episode_compilation_directory_install_failed:{path}"
            ) from exc
        if (
            readback.st_uid != owner_uid
            or readback.st_gid != owner_gid
            or stat.S_IMODE(readback.st_mode) != 0o750
        ):
            raise ControlPlaneDeployError(
                f"deploy_episode_compilation_directory_readback_mismatch:{path}"
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


def _install_storage_pins_runtime_root(
    *,
    pins_root: str | Path = DEFAULT_PINS_ROOT,
    account: str = DEFAULT_SERVICE_ACCOUNT,
    chown: Any = os.chown,
    stat_reader: Any = lambda path: path.stat(),
) -> dict[str, Any]:
    """Install the service-owned root required by every storage-pin writer."""

    account_ids = _service_account_ids(account)
    if account_ids is None:
        raise ControlPlaneDeployError(
            f"deploy_storage_pins_account_missing:{account}"
        )
    owner_uid, owner_gid = account_ids
    root = Path(pins_root).expanduser()
    if not root.is_absolute():
        raise ControlPlaneDeployError("deploy_storage_pins_root_not_absolute")
    if root.is_symlink():
        raise ControlPlaneDeployError(
            f"deploy_storage_pins_root_symlink:{root}"
        )
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        if root.is_symlink() or not root.is_dir():
            raise ControlPlaneDeployError(
                f"deploy_storage_pins_root_invalid:{root}"
            )
        metadata = stat_reader(root)
        repaired = False
        if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
            chown(root, owner_uid, owner_gid)
            repaired = True
            metadata = stat_reader(root)
        if stat.S_IMODE(metadata.st_mode) != 0o750:
            root.chmod(0o750)
            repaired = True
        readback = stat_reader(root)
    except ControlPlaneDeployError:
        raise
    except OSError as exc:
        raise ControlPlaneDeployError(
            f"deploy_storage_pins_root_install_failed:{root}"
        ) from exc
    if (
        readback.st_uid != owner_uid
        or readback.st_gid != owner_gid
        or stat.S_IMODE(readback.st_mode) != 0o750
    ):
        raise ControlPlaneDeployError(
            f"deploy_storage_pins_root_readback_mismatch:{root}"
        )
    return {
        "status": "ready",
        "path": str(root),
        "account": account,
        "owner_uid": owner_uid,
        "owner_gid": owner_gid,
        "mode": "0750",
        "repaired": repaired,
    }


def _install_configured_controls_runtime_prerequisites(
    *,
    plan_root: str = DEFAULT_CONFIGURED_CONTROLS_PLAN_ROOT,
    webapp_secret: str = DEFAULT_CONFIGURED_CONTROLS_WEBAPP_SECRET,
    account: str = DEFAULT_SERVICE_ACCOUNT,
    root_uid: int = 0,
    chown: Any = os.chown,
    stat_reader: Any = lambda path: path.stat(),
) -> dict[str, Any]:
    """Provision the timer plan root and secret permission without reading it."""

    account_ids = _service_account_ids(account)
    if account_ids is None:
        raise ControlPlaneDeployError(
            f"deploy_configured_controls_account_missing:{account}"
        )
    owner_uid, owner_gid = account_ids
    root = Path(plan_root)
    secret = Path(webapp_secret)
    if (
        not root.is_absolute()
        or not secret.is_absolute()
        or root.is_symlink()
        or secret.is_symlink()
    ):
        raise ControlPlaneDeployError(
            "deploy_configured_controls_runtime_path_invalid"
        )
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        if not root.is_dir() or not secret.is_file():
            raise ControlPlaneDeployError(
                "deploy_configured_controls_runtime_prerequisite_missing"
            )
        root_metadata = stat_reader(root)
        if root_metadata.st_uid != owner_uid or root_metadata.st_gid != owner_gid:
            chown(root, owner_uid, owner_gid)
        if stat.S_IMODE(root_metadata.st_mode) != 0o750:
            root.chmod(0o750)
        secret_metadata = stat_reader(secret)
        if secret_metadata.st_uid != root_uid or secret_metadata.st_gid != owner_gid:
            chown(secret, root_uid, owner_gid)
        if stat.S_IMODE(secret_metadata.st_mode) != 0o440:
            secret.chmod(0o440)
        root_readback = stat_reader(root)
        secret_readback = stat_reader(secret)
    except OSError as exc:
        raise ControlPlaneDeployError(
            "deploy_configured_controls_runtime_prerequisite_install_failed"
        ) from exc
    if (
        root_readback.st_uid != owner_uid
        or root_readback.st_gid != owner_gid
        or stat.S_IMODE(root_readback.st_mode) != 0o750
        or secret_readback.st_uid != root_uid
        or secret_readback.st_gid != owner_gid
        or stat.S_IMODE(secret_readback.st_mode) != 0o440
    ):
        raise ControlPlaneDeployError(
            "deploy_configured_controls_runtime_prerequisite_readback_mismatch"
        )
    return {
        "plan_root": str(root),
        "plan_root_owner_uid": owner_uid,
        "plan_root_owner_gid": owner_gid,
        "plan_root_mode": "0750",
        "webapp_secret": str(secret),
        "webapp_secret_owner_uid": root_uid,
        "webapp_secret_owner_gid": owner_gid,
        "webapp_secret_mode": "0440",
        "secret_bytes_read": False,
    }


def _install_configured_controls_autostart_registry(
    *,
    intent_root: str = DEFAULT_CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT,
    intent_sources: Sequence[str] = (),
    source_commit: str,
    account: str = DEFAULT_SERVICE_ACCOUNT,
    root_uid: int = 0,
) -> dict[str, Any]:
    """Install immutable per-scene continuation intent with readback proof.

    This registry is a pre-admission boundary, not universal task inference: a
    scene-configuration activation is admitted only when its exact team/scene/
    task intent was provisioned at the same production commit.
    """

    account_ids = _service_account_ids(account)
    if account_ids is None:
        raise ControlPlaneDeployError(
            f"deploy_configured_controls_account_missing:{account}"
        )
    _owner_uid, owner_gid = account_ids
    root = Path(intent_root).expanduser()
    if not root.is_absolute() or root.is_symlink():
        raise ControlPlaneDeployError(
            "deploy_configured_controls_autostart_intent_root_invalid"
        )
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        if root.is_symlink() or not root.is_dir():
            raise ControlPlaneDeployError(
                "deploy_configured_controls_autostart_intent_root_invalid"
            )
        metadata = root.stat()
        if metadata.st_uid != root_uid or metadata.st_gid != owner_gid:
            os.chown(root, root_uid, owner_gid)
        if stat.S_IMODE(metadata.st_mode) != 0o750:
            root.chmod(0o750)
        readback = root.stat()
    except OSError as exc:
        raise ControlPlaneDeployError(
            "deploy_configured_controls_autostart_intent_root_install_failed"
        ) from exc
    if (
        readback.st_uid != root_uid
        or readback.st_gid != owner_gid
        or stat.S_IMODE(readback.st_mode) != 0o750
    ):
        raise ControlPlaneDeployError(
            "deploy_configured_controls_autostart_intent_root_readback_mismatch"
        )

    entries: list[dict[str, Any]] = []
    for raw_source in intent_sources:
        source = Path(raw_source).expanduser()
        if not source.is_absolute() or source.is_symlink() or not source.is_file():
            raise ControlPlaneDeployError(
                "deploy_configured_controls_autostart_intent_source_invalid"
            )
        try:
            payload = source.read_bytes()
            value = validate_configured_controls_autostart_intent(
                json.loads(payload)
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ControlPlaneDeployError(
                "deploy_configured_controls_autostart_intent_source_invalid"
            ) from exc
        if value["expected_production_commit"] != source_commit:
            raise ControlPlaneDeployError(
                "deploy_configured_controls_autostart_intent_commit_mismatch"
            )
        adoption = value["configuration_adoption"]
        if adoption["mode"] == "explicit_terminal_adoption":
            from blueprint_pipeline.task_evaluation_configured_controls_autostart import (
                configured_controls_autostart_adoption_registry_name,
            )

            destination = root / configured_controls_autostart_adoption_registry_name(
                team_namespace=value["team_namespace"],
                scene_id=value["scene_id"],
                task_id=value["task_id"],
                source_launch_id=adoption["source_launch_id"],
            )
        else:
            destination = root / configured_controls_autostart_registry_name(
                team_namespace=value["team_namespace"],
                scene_id=value["scene_id"],
                task_id=value["task_id"],
            )
        previous_sha256: str | None = None
        try:
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise ControlPlaneDeployError(
                        "deploy_configured_controls_autostart_intent_conflict"
                    )
                previous_payload = destination.read_bytes()
                try:
                    previous = validate_configured_controls_autostart_intent(
                        json.loads(previous_payload)
                    )
                except (
                    UnicodeError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    raise ControlPlaneDeployError(
                        "deploy_configured_controls_autostart_existing_intent_invalid"
                    ) from exc
                if (
                    previous["team_namespace"] != value["team_namespace"]
                    or previous["scene_id"] != value["scene_id"]
                    or previous["task_id"] != value["task_id"]
                ):
                    raise ControlPlaneDeployError(
                        "deploy_configured_controls_autostart_intent_conflict"
                    )
                previous_sha256 = _sha256_bytes(previous_payload)
            if not destination.exists() or destination.read_bytes() != payload:
                temporary: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        dir=root,
                        prefix=f".{destination.name}.",
                        delete=False,
                    ) as stream:
                        temporary = Path(stream.name)
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.chown(temporary, root_uid, owner_gid)
                    temporary.chmod(0o440)
                    os.replace(temporary, destination)
                    temporary = None
                    directory_fd = os.open(root, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                finally:
                    if temporary is not None:
                        with contextlib.suppress(OSError):
                            temporary.unlink()
            else:
                os.chown(destination, root_uid, owner_gid)
                destination.chmod(0o440)
            destination_payload = destination.read_bytes()
            destination_metadata = destination.stat()
        except OSError as exc:
            raise ControlPlaneDeployError(
                "deploy_configured_controls_autostart_intent_install_failed"
            ) from exc
        if (
            destination.is_symlink()
            or destination_payload != payload
            or destination_metadata.st_uid != root_uid
            or destination_metadata.st_gid != owner_gid
            or stat.S_IMODE(destination_metadata.st_mode) != 0o440
        ):
            raise ControlPlaneDeployError(
                "deploy_configured_controls_autostart_intent_readback_mismatch"
            )
        entries.append(
            {
                "path": str(destination),
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
                "mode": "0440",
                "expected_production_commit": source_commit,
                "team_namespace": value["team_namespace"],
                "scene_id": value["scene_id"],
                "task_id": value["task_id"],
                "intent_digest": value["intent_digest"],
                "configuration_adoption_mode": adoption["mode"],
                "replaced_previous_sha256": (
                    previous_sha256
                    if previous_sha256 != _sha256_bytes(payload)
                    else None
                ),
            }
        )
    return {
        "root": str(root),
        "root_owner_uid": root_uid,
        "root_owner_gid": owner_gid,
        "root_mode": "0750",
        "status": "provisioned" if entries else "empty_pre_admission_required",
        "pre_admission_required": True,
        "entry_count": len(entries),
        "entries": entries,
    }


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
    """Snapshot path/timer intent before unit bytes or daemon state move."""

    return {
        unit: _systemd_unit_state(unit)
        for entry in installed_units
        if (unit := str(entry.get("unit") or "")).endswith((".path", ".timer"))
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
    always_arm_authority_gated_units: Sequence[str] = (),
    always_arm_timer_units: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Restore path/timer intent without widening arbitrary launch authority.

    A stopped paid watcher is an operational freeze, so paths preserve both boot
    and active state unless explicitly armed or fixed as no-spend.  The one
    configured-controls timer is a separate fixed category: it only consumes
    sealed qualifying plans through canonical APIs, and its receipt names that
    progression authority instead of misreporting it as no-spend.
    """

    receipts: list[dict[str, Any]] = []
    for entry in installed_units:
        unit = str(entry.get("unit") or "")
        if not unit.endswith((".path", ".timer")):
            continue
        prior = dict(before.get(unit) or {"enabled": "disabled", "state": "inactive"})
        arm_no_spend = unit in always_arm_units
        arm_authority_gated = unit in always_arm_authority_gated_units
        arm_progression = unit in always_arm_timer_units
        if sum((arm_no_spend, arm_authority_gated, arm_progression)) > 1:
            raise ControlPlaneDeployError(
                f"deploy_automation_unit_authority_ambiguous:{unit}"
            )
        explicit_path_arm = arm_path_units and unit.endswith(".path")
        should_enable = (
            explicit_path_arm
            or arm_no_spend
            or arm_authority_gated
            or arm_progression
            or prior.get("enabled") == "enabled"
        )
        should_start = (
            explicit_path_arm
            or arm_no_spend
            or arm_authority_gated
            or arm_progression
            or prior.get("state") == "active"
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
                observed = _systemd_unit_state(unit)
                already_restored = (
                    verb == "disable" and observed["enabled"] == "disabled"
                ) or (verb == "stop" and observed["state"] == "inactive")
                if already_restored:
                    continue
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
                    if explicit_path_arm
                    else "arm_no_spend"
                    if arm_no_spend
                    else "arm_authority_gated_paid_dispatch"
                    if arm_authority_gated
                    else "arm_configured_controls_progression"
                    if arm_progression
                    else "preserve"
                ),
                "after": after,
                "operator_freeze_preserved": (
                    not explicit_path_arm
                    and not arm_no_spend
                    and not arm_authority_gated
                    and not arm_progression
                    and not should_start
                ),
            }
        )
    return receipts


@contextlib.contextmanager
def _restore_path_unit_states_on_deploy_failure(
    installed_units: Sequence[Mapping[str, Any]],
):
    """Quiesce watchers and restore their exact prior state on any failure.

    A deploy intentionally stops queue watchers before validating and
    publishing a release.  The success path restores them after the intake
    proves the new commit, but a failed prerequisite or restart previously
    exited with every formerly active watcher still stopped.  Recovery runs
    while the paid-launch locks remain held and never applies the no-spend or
    explicit-arm widening rules: it restores only the state observed before
    this attempt.
    """

    before = _installed_path_unit_states(installed_units)
    try:
        quiesced = _quiesce_active_path_units(before)
        yield before, quiesced
    except BaseException as deployment_error:
        try:
            _restore_installed_path_units(
                installed_units,
                before=before,
                arm_path_units=False,
                always_arm_units=(),
            )
        except Exception as restore_error:
            raise ControlPlaneDeployError(
                "deploy_failed_path_unit_restore_failed:"
                f"{type(deployment_error).__name__}:{restore_error}"
            ) from deployment_error
        raise


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
        # Preserve the virtualenv entrypoint. Resolving this symlink selects
        # the system interpreter and silently drops production dependencies.
        f"BLUEPRINT_PIPELINE_PYTHON={Path(sys.executable).absolute()}\n"
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
    canary: bool = False,
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
    cad_skill_source_root: str | Path = DEFAULT_CAD_SKILL_SOURCE_ROOT,
    configured_controls_autostart_intent_root: str | Path = (
        DEFAULT_CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT
    ),
    configured_controls_autostart_intent_sources: Sequence[str] = (),
    arm_path_units: bool = False,
    disk_reservation_root: str | Path | None = None,
    release_retirement_reference_roots: Sequence[str] = (
        DEFAULT_RELEASE_RETIREMENT_REFERENCE_ROOTS
    ),
    release_retirement_keep_last: int = DEFAULT_RELEASE_RETIREMENT_KEEP_LAST,
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
    if canary and not iteration:
        raise ControlPlaneDeployError("deploy_canary_requires_iteration")
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
        if canary:
            # A canary deploy exists for the fix-and-fire loop on a lane that
            # is already development_only: waiting for a merge (review, CI,
            # rebase churn against a fast-moving main) costs 5-15 minutes per
            # attempt and buys nothing the canary evidence can use. What is
            # still NOT traded away is immutability: the commit must be
            # reachable from some pushed origin ref, so the running bytes are
            # publicly recorded and can never silently diverge from the
            # repository. Local-only or dirty-tree commits still refuse.
            code, reachable = _git(
                source,
                "branch",
                "--remotes",
                "--contains",
                source_commit,
            )
            if code != 0 or not reachable.strip():
                raise ControlPlaneDeployError(
                    "deploy_canary_commit_not_pushed_to_origin"
                )
        else:
            code, _ = _git(source, "merge-base", "--is-ancestor", source_commit, "origin/main")
            if code != 0:
                raise ControlPlaneDeployError("deploy_iteration_commit_not_on_origin_main")
        provenance_receipt = {
            "schema_version": "blueprint.deploy_release_provenance.v1",
            "status": "canary" if canary else "iteration",
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

    disk_reservation = None
    disk_reservation_runtime = None
    if disk_reservation_root is not None:
        disk_reservation_runtime = _install_disk_reservation_runtime_prerequisites(
            disk_reservation_root
        )
        try:
            disk_reservation = reserve_control_plane_disk(
                "control_plane_deploy",
                target_root=releases,
                reservation_root=disk_reservation_root,
            )
        except ControlPlaneDiskBudgetError as exc:
            raise ControlPlaneDeployError(
                f"deploy_disk_budget_exceeded:{exc}"
            ) from exc
    disk_reservation_receipt = (
        disk_reservation.receipt() if disk_reservation is not None else None
    )
    automation_unit_names = [
        {"unit": unit}
        for unit in DEFAULT_DEPLOYED_SYSTEMD_UNITS
        if unit.endswith((".path", ".timer"))
    ]
    # Where a deploy's minutes go is otherwise invisible; the receipt records
    # each stage so a slow deploy is a measurement, not a feeling.
    stage_timings: dict[str, float] = {}
    stage_clock = [time.monotonic()]

    def _mark_stage(name: str) -> None:
        now = time.monotonic()
        stage_timings[name] = round(now - stage_clock[0], 3)
        stage_clock[0] = now

    # Held for the whole deploy, not sampled before it: a launch that starts
    # mid-deploy would read a release being swapped underneath it.  The nested
    # guard restores exact watcher intent before the paid locks are released if
    # any later deploy step fails.
    with (
        disk_reservation or contextlib.nullcontext(),
        _holding_paid_launch_locks(paid_launch_locks),
        _restore_path_unit_states_on_deploy_failure(automation_unit_names) as (
            automation_unit_states_before,
            quiesced_automation_units,
        ),
    ):
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
            allow_unmerged_remote_commit=canary,
        )
        _mark_stage("release_staged")
        # Every path the new units' sandboxes name must exist before the
        # release link moves, or the first worker to start after the switch
        # dies on mount setup and the deploy still reports success.
        unit_sandbox_paths = _install_unit_sandbox_paths(
            release_path=staged_release["release_path"]
        )
        _mark_stage("unit_sandbox_paths")
        try:
            cad_skill_sources = provision_production_cad_skill_sources(
                cad_skill_source_root
            )
            cad_sources_by_id = {
                str(row["id"]): str(row["path"])
                for row in cad_skill_sources["sources"]
            }
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
                text_to_cad_root=cad_sources_by_id["text-to-cad"],
                multi_agent_cad_root=cad_sources_by_id["multi-agent-cad"],
                readback=service_account_readback(DEFAULT_SERVICE_ACCOUNT),
                readback_actor=f"service-account:{DEFAULT_SERVICE_ACCOUNT}",
            )
        except (ValueError, ProductionCadSkillSourcesError) as exc:
            raise ControlPlaneDeployError(
                f"deploy_scene_configuration_runtime_invalid:{exc}"
            ) from exc
        scene_configuration_environment = _install_scene_configuration_environment(
            Path(scene_configuration_environment_file).expanduser(),
            environment=scene_configuration_runtime["environment"],
        )
        _mark_stage("runtime_trees_provisioned")
        _move_source_checkout(source, source_commit)
        release = stage_task_evaluation_control_plane_release(
            source_repo=source,
            source_commit=source_commit,
            release_root=release_root,
            state_root=state_root,
            active_link=active,
            activate=True,
            allow_unmerged_remote_commit=canary,
        )
        commit = str(release["source_commit"])
        _mark_stage("release_activated")

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
        episode_compilation_runtime_directories = (
            _install_episode_compilation_runtime_directories()
        )
        storage_pins_runtime = _install_storage_pins_runtime_root()
        configured_controls_runtime = (
            _install_configured_controls_runtime_prerequisites()
        )
        configured_controls_autostart_registry = (
            _install_configured_controls_autostart_registry(
                intent_root=str(configured_controls_autostart_intent_root),
                intent_sources=configured_controls_autostart_intent_sources,
                source_commit=commit,
            )
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
        _mark_stage("units_and_directories_installed")
        restarted = _restart_units(_required_restart_units(restart_units))
        runtime = _verify_intake_runtime(
            intake_version_url, expected_commit=commit
        )
        _mark_stage("intake_restarted_and_proven")
        # Last inside the held locks: the queue watcher only starts watching
        # once the restarted intake has proven the new commit, and no launch
        # can slip in between the watcher restart and the lock release.
        automation_unit_state_receipts = _restore_installed_path_units(
            installed_systemd_units,
            before=automation_unit_states_before,
            arm_path_units=arm_path_units,
            always_arm_units=DEFAULT_ALWAYS_ARM_PATH_UNITS,
            always_arm_authority_gated_units=(
                DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS
            ),
            always_arm_timer_units=DEFAULT_ALWAYS_ARM_TIMER_UNITS,
        )
        # Last, with the new release proven live: retire the trees this deploy
        # superseded, so per-commit growth is bounded by keep_last instead of
        # by the number of deploys ever made.
        release_retirement = _retire_superseded_release_trees(
            release_root=releases,
            runtime_root=scene_configuration_runtime_root,
            active_link=active,
            current_commit=commit,
            reference_roots=release_retirement_reference_roots,
            keep_last=release_retirement_keep_last,
        )
        _mark_stage("release_retirement")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "deployed",
        "release_retirement": release_retirement,
        "unit_sandbox_paths": unit_sandbox_paths,
        "stage_timings_seconds": stage_timings,
        "source_commit": commit,
        "disk_reservation": disk_reservation_receipt,
        "disk_reservation_runtime": disk_reservation_runtime,
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
        "episode_compilation_runtime_directories": (
            episode_compilation_runtime_directories
        ),
        "storage_pins_runtime": storage_pins_runtime,
        "configured_controls_runtime": configured_controls_runtime,
        "configured_controls_autostart_registry": (
            configured_controls_autostart_registry
        ),
        "path_unit_states": [
            row for row in automation_unit_state_receipts if row["unit"].endswith(".path")
        ],
        "timer_unit_states": [
            row for row in automation_unit_state_receipts if row["unit"].endswith(".timer")
        ],
        "quiesced_path_units": [
            row for row in quiesced_automation_units if row["unit"].endswith(".path")
        ],
        "quiesced_timer_units": [
            row for row in quiesced_automation_units if row["unit"].endswith(".timer")
        ],
        # Compatibility projection for readers that predate the state-preserving
        # receipt. It contains only watchers that are active after this deploy.
        "activated_path_units": [
            {
                "unit": row["unit"],
                "enabled": row["after"]["enabled"],
                "state": row["after"]["state"],
            }
            for row in automation_unit_state_receipts
            if row["unit"].endswith(".path") and row["after"]["state"] == "active"
        ],
        "activated_timer_units": [
            {
                "unit": row["unit"],
                "enabled": row["after"]["enabled"],
                "state": row["after"]["state"],
            }
            for row in automation_unit_state_receipts
            if row["unit"].endswith(".timer") and row["after"]["state"] == "active"
        ],
        "intake_runtime_binding": runtime_binding,
        "intake_runtime": runtime,
        "scene_configuration_runtime": scene_configuration_runtime,
        "cad_skill_sources": cad_skill_sources,
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
        "--canary",
        action="store_true",
        help=(
            "With --iteration: accept any commit pushed to origin (not only "
            "origin/main ancestors). For fix-and-fire debugging on a "
            "development_only lane; the release is stamped status=canary and "
            "promotion_eligible=false. Unpushed or dirty-tree commits still "
            "refuse."
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
        "--cad-skill-source-root", default=DEFAULT_CAD_SKILL_SOURCE_ROOT
    )
    parser.add_argument(
        "--configured-controls-autostart-intent-root",
        default=DEFAULT_CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT,
    )
    parser.add_argument(
        "--configured-controls-autostart-intent",
        action="append",
        default=None,
        help=(
            "Exact digest-bound per-scene continuation intent to provision. "
            "Repeatable; omitting it leaves scene configuration fail-closed "
            "until pre-admission is provisioned."
        ),
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
            canary=args.canary,
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
            cad_skill_source_root=args.cad_skill_source_root,
            configured_controls_autostart_intent_root=(
                args.configured_controls_autostart_intent_root
            ),
            configured_controls_autostart_intent_sources=tuple(
                args.configured_controls_autostart_intent or ()
            ),
            arm_path_units=args.arm_path_units,
            disk_reservation_root=(
                Path(args.state_root).expanduser() / "disk-reservations"
            ),
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
