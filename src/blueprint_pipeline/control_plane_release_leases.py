"""Typed, expiring protection for per-commit release trees.

Deploy-time retirement used to protect every 40-hex token it could find in any
JSON file under about twenty roots.  That caught git tree ids, commits embedded
in profile ids, other repositories' commits and the consumption records of
authorizations that expired long ago, so nothing ever lapsed: on 2026-09-26 the
host ran out of disk holding 95 release trees behind 513 "protected commits",
and deploy retired none of them.

Protection now comes only from typed sources.  Every row names the commit, who
needs it (``owner``), why (``reason``), until when (``expires_at_epoch``) and the
run it serves (``run_ref``):

``live_queue``
    A live envelope in a queue that executes a release, read through its typed
    commit fields and managed-tree paths only.  It lapses when it leaves the
    live states, and at the latest at its maximum lifetime.
``standing_authorization``
    A standing launch authorization that can still admit a launch of the
    profile that runs the commit.  It lapses when it expires or runs out of
    launches or spend; its consumption records never protect anything.
``retention_binding``
    A required-evidence binding.  Binding bytes never change (their writer
    compares whole documents on republish), so a legacy binding's lease lives
    in a sidecar under the lease root, written once by migration and renewed
    only while its run is live.  It lapses when its run ends, when it expires
    with its run unknown, and at the latest at its maximum lifetime.
``configured_runtime``
    A runtime path named by current host configuration.  It has no expiry: the
    configuration is re-read on every deploy.

No free-text search happens anywhere: a commit comes only from a named field or
from a path string that names a managed release or runtime tree.  Nothing here
deletes anything; the retirement plan decides from these rows.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


LEASE_SCHEMA = "control_plane_release_lease.v1"
PROTECTIONS_SCHEMA = "control_plane_release_protections.v1"
BINDING_SCHEMA = "task_evaluation_release_retention_binding.v1"
RETENTION_PLAN_SCHEMA = "task_evaluation_release_retention_plan.v1"
DEFAULT_CONTROL_PLANE_ROOT = Path("/var/lib/blueprint/pipeline-control-plane")
DEFAULT_LEASE_ROOT = DEFAULT_CONTROL_PLANE_ROOT / "release-leases"
DEFAULT_TTL_SECONDS = 14 * 24 * 3600
DEFAULT_MAX_LIFETIME_SECONDS = 30 * 24 * 3600
#: More trees than this held only by leases is itself worth a page.
LEASE_ALERT_THRESHOLD = 20
#: Protection that is a lease: it has an owner, a run and an expiry.
LEASE_KINDS = ("live_queue", "standing_authorization", "retention_binding")
#: Protection that is current configuration, re-read on every deploy.
CONFIG_KIND = "configured_runtime"
#: The queues that execute a release, and the states in which an envelope is
#: still going to run.  Terminal states and wake-pending markers (which hold
#: only a job digest) are deliberately absent.
LIVE_QUEUE_STATES: Mapping[str, tuple[str, ...]] = {
    "task-evaluation-launches": ("pending", "processing"),
    "task-evaluation-launch-preparations": (
        "pending",
        "processing",
        "awaiting_source_preparation",
        "awaiting_capacity",
    ),
    "sam31-preparation-executions": ("pending", "processing", "waiting_external"),
    "task-evaluation-episode-compilations": ("pending", "processing"),
    "task-evaluation-launch-activations": ("pending", "processing"),
    "task-evaluation-policy-canary-dispatches": ("pending", "processing"),
    "task-evaluation-scene-constructions": ("pending", "processing"),
    "task-evaluation-terminal-resource-releases": ("pending", "processing"),
}
DEFAULT_PROTECTION_CONFIG_FILES = (
    # Preparation can still use an older renderer after launch queues empty.
    Path("/etc/blueprint/task-evaluation-public-scene-machinery.json"),
    Path("/etc/blueprint/task-evaluation-scene-preparation-bootstrap.json"),
)

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
# A path naming a managed per-commit tree.  The commit must end the path
# component: followed by a separator, the end of the string, or any character
# that cannot continue a directory name.
_MANAGED_TREE = re.compile(
    r"/(?:task-evaluation-control-plane-releases|system-runtimes/(?:splat-render|scene-configuration))"
    r"/([0-9a-f]{40})(?![0-9A-Za-z_-])"
)
_COMMIT_FIELDS = ("expected_production_commit", "source_commit", "expected_source_commit")
_MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_SIDECAR_SUFFIX = ".lease.v1.json"
_OFFLOAD_POINTER_SUFFIX = ".offloaded.v1.json"
_INLINE_LEASE_FIELDS = ("owner", "expires_at_epoch", "max_expires_at_epoch", "run_ref")
_QUEUE_SCAN_ATTEMPTS = 3
# The blockers after which a standing authorization can never admit again.
_EXPECTED_TERMINAL_AUTHORIZATION_BLOCKERS = frozenset(
    {
        "standing_authorization_expired",
        "standing_authorization_launches_exhausted",
        "standing_authorization_spend_ceiling_reached",
    }
)


@dataclass(frozen=True)
class ProtectionSources:
    """Where typed release protection is read from on one host."""

    control_plane_root: Path  # queues live directly under it
    profile_dir: Path
    standing_authorization_dir: Path
    binding_root: Path
    lease_root: Path
    config_files: tuple[Path, ...] = ()
    intent_root: Path | None = None
    launch_run_root: Path | None = None


DEFAULT_PROTECTION_SOURCES = ProtectionSources(
    control_plane_root=DEFAULT_CONTROL_PLANE_ROOT,
    profile_dir=Path("/etc/blueprint/task-evaluation-launch-profiles"),
    standing_authorization_dir=DEFAULT_CONTROL_PLANE_ROOT / "standing-authorizations",
    binding_root=DEFAULT_CONTROL_PLANE_ROOT / "task-evaluation-release-retention-bindings",
    lease_root=DEFAULT_LEASE_ROOT,
    config_files=DEFAULT_PROTECTION_CONFIG_FILES,
    intent_root=DEFAULT_CONTROL_PLANE_ROOT / "task-evaluation-scene-intents",
    launch_run_root=DEFAULT_CONTROL_PLANE_ROOT / "task-evaluation-launch-runs",
)


def _valid_commit(value: Any) -> str | None:
    return value if isinstance(value, str) and _COMMIT.fullmatch(value) else None


def _absent(path: Path) -> bool:
    """True only when nothing exists at ``path``.

    Any other failure to look (a permission error, a file where a directory
    belongs) propagates, so the caller blocks instead of reading "unreadable"
    as "empty".
    """

    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    return False


def _require_regular(path: Path) -> None:
    """Refuse anything but a regular file before another module reads it.

    Those readers open with a blocking ``read_text``; a FIFO in their place
    would hang the deploy, so it is refused here first.
    """

    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError("release_protection_document_unsafe")


def _require_regular_children(directory: Path) -> None:
    """Every ``*.json`` another module will read from ``directory`` is a regular file."""

    if _absent(directory):
        return
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("release_protection_document_unsafe")
    for name in os.listdir(directory):
        if name.endswith(".json"):
            _require_regular(directory / name)


def _read_document(path: Path) -> tuple[bytes, Any, os.stat_result]:
    """Read one regular JSON file without following a symlink.

    Raises ``OSError`` or ``ValueError`` for anything that is not a regular
    file of at most 16 MiB holding valid JSON.  The open never blocks: a FIFO
    where a document belongs is refused instead of hanging the deploy.
    """

    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_DOCUMENT_BYTES:
            raise ValueError("release_protection_document_unsafe")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            payload = stream.read(_MAX_DOCUMENT_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(payload) > _MAX_DOCUMENT_BYTES:
        raise ValueError("release_protection_document_unsafe")
    return payload, json.loads(payload), info


def _json_names(directory: Path) -> list[str]:
    return sorted(name for name in os.listdir(directory) if name.endswith(".json"))


def _managed_tree_commits(value: Any) -> set[str]:
    """Commits named by a managed release or runtime tree path in any string value."""

    commits: set[str] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            commits.update(_MANAGED_TREE.findall(item))
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return commits


def _profile_id_commit(profile_id: str) -> str | None:
    """The commit a profile id names: ``<prefix>-<commit>[-<revision>][-binding-<digest>]``.

    The commit is the first 40-hex segment after the prefix; a later 40-hex
    segment is a revision or digest suffix, never the release.
    """

    for segment in profile_id.split("-"):
        if _COMMIT.fullmatch(segment):
            return segment
    return None


def _code_id(value: str) -> str:
    """An identity safe to embed in a typed refusal code.

    Names come from file names and envelope fields; anything that is not a
    plain identifier (a path, whitespace, a control character) is replaced by
    a short digest so a code never carries raw host input.
    """

    if _IDENTIFIER.fullmatch(value):
        return value
    digest = hashlib.sha256(value.encode("utf-8", "surrogateescape")).hexdigest()
    return f"invalid-{digest[:12]}"


class _Collection:
    def __init__(self) -> None:
        self.leases: list[dict[str, Any]] = []
        self.lapsed: list[dict[str, Any]] = []
        self.migrated: set[str] = set()
        self.renewed: set[str] = set()
        self.warnings: set[str] = set()
        self.blockers: set[str] = set()

    def protect(self, commits: set[str] | list[str], row: Mapping[str, Any]) -> None:
        for commit in sorted(set(commits)):
            self.leases.append({"commit": commit, **row})

    def lapse(self, commits: set[str] | list[str], row: Mapping[str, Any], why: str) -> None:
        for commit in sorted(set(commits)):
            self.lapsed.append({"commit": commit, **row, "why": why})

    def merge(self, other: _Collection) -> None:
        self.leases.extend(other.leases)
        self.lapsed.extend(other.lapsed)
        self.migrated |= other.migrated
        self.renewed |= other.renewed
        self.warnings |= other.warnings
        self.blockers |= other.blockers

    def result(self, now: float) -> dict[str, Any]:
        def order(row: Mapping[str, Any]) -> tuple[str, str, str]:
            return (str(row["kind"]), str(row["source"]), str(row["commit"] or ""))

        return {
            "schema_version": PROTECTIONS_SCHEMA,
            "collected_at_epoch": now,
            "leases": sorted(self.leases, key=order),
            "lapsed": sorted(self.lapsed, key=order),
            "migrated": sorted(self.migrated),
            "renewed": sorted(self.renewed),
            "warnings": sorted(self.warnings),
            "blockers": sorted(self.blockers),
        }


@dataclass(frozen=True)
class _Profiles:
    commits: Mapping[str, str | None]
    documents: Mapping[str, Mapping[str, Any]]


def _profile_commit(profile: Mapping[str, Any], profile_id: str) -> str | None:
    commit = _valid_commit(profile.get("source_commit"))
    if commit is not None:
        return commit
    allocator = profile.get("allocator")
    argv = allocator.get("argv") if isinstance(allocator, Mapping) else None
    if isinstance(argv, list):
        for index, item in enumerate(argv):
            if item == "--expected-source-commit" and index + 1 < len(argv):
                commit = _valid_commit(argv[index + 1])
            elif isinstance(item, str) and item.startswith("--expected-source-commit="):
                commit = _valid_commit(item.split("=", 1)[1])
            if commit is not None:
                return commit
    return _profile_id_commit(profile_id)


def _read_profiles(profile_dir: Path, collection: _Collection) -> _Profiles:
    """Map each launch profile to the commit it runs, reading structure only.

    A profile never protects by itself, so an unreadable one is a warning; a
    live reference to it is what blocks.
    """

    commits: dict[str, str | None] = {}
    documents: dict[str, Mapping[str, Any]] = {}
    try:
        if profile_dir.is_symlink() or not profile_dir.is_dir():
            raise ValueError("profile_dir_unsafe")
        names = _json_names(profile_dir)
    except (OSError, ValueError):
        collection.warnings.add("release_protection_profile_dir_unreadable")
        return _Profiles(commits, documents)
    for name in names:
        try:
            _payload, profile, _info = _read_document(profile_dir / name)
        except (OSError, ValueError):
            profile = None
        if not isinstance(profile, Mapping):
            collection.warnings.add(f"release_protection_profile_unreadable:{_code_id(name)}")
            continue
        declared = profile.get("profile_id")
        profile_id = declared if isinstance(declared, str) and declared else name[: -len(".json")]
        commits[profile_id] = _profile_commit(profile, profile_id)
        documents[profile_id] = profile
    return _Profiles(commits, documents)


def _envelope_commits(envelope: Mapping[str, Any]) -> set[str]:
    commits: set[str] = set()
    request = envelope.get("request")
    for container in (envelope, request if isinstance(request, Mapping) else {}):
        for field in _COMMIT_FIELDS:
            commit = _valid_commit(container.get(field))
            if commit is not None:
                commits.add(commit)
        release = container.get("release")
        if isinstance(release, Mapping):
            commit = _valid_commit(release.get("commit"))
            if commit is not None:
                commits.add(commit)
    return commits | _managed_tree_commits(envelope)


def _envelope_profile_ids(envelope: Mapping[str, Any]) -> set[str]:
    profile_ids: set[str] = set()
    request = envelope.get("request")
    for container in (envelope, request if isinstance(request, Mapping) else {}):
        value = container.get("launch_profile_id")
        if isinstance(value, str) and value.strip():
            profile_ids.add(value.strip())
    return profile_ids


def _scan_queues(
    root: Path,
    profiles: _Profiles,
    *,
    now: float,
    max_lifetime_seconds: int,
    collection: _Collection,
) -> tuple[set[tuple[str, str]], bool]:
    """One pass over every live queue state.

    Returns the ``(queue, name)`` pairs it accounted for (read, or blocked as
    unreadable) and whether an envelope vanished between listing and reading.
    """

    accounted: set[tuple[str, str]] = set()
    moved = False
    for queue, states in LIVE_QUEUE_STATES.items():
        try:
            if _absent(root / queue):
                # A whole queue missing is unusual enough to report; its states
                # being missing (never used on this host) is not.
                collection.warnings.add(f"release_protection_queue_root_missing:{queue}")
                continue
        except OSError:
            collection.blockers.add(f"release_protection_queue_unreadable:{queue}")
            continue
        for state in states:
            directory = root / queue / state
            try:
                if _absent(directory):
                    continue  # a queue state never used on this host holds nothing
                if directory.is_symlink() or not directory.is_dir():
                    raise ValueError("queue_state_unsafe")
                names = _json_names(directory)
            except (OSError, ValueError):
                collection.blockers.add(f"release_protection_queue_unreadable:{queue}/{state}")
                continue
            for name in names:
                source = f"{queue}/{state}/{name}"
                code_source = f"{queue}/{state}/{_code_id(name)}"
                try:
                    _payload, envelope, info = _read_document(directory / name)
                except FileNotFoundError:
                    moved = True  # a worker moved it on; the snapshot is retried
                    continue
                except (OSError, ValueError):
                    envelope = None
                accounted.add((queue, name))
                if not isinstance(envelope, Mapping):
                    collection.blockers.add(f"release_protection_queue_unreadable:{code_source}")
                    continue
                commits = _envelope_commits(envelope)
                for profile_id in sorted(_envelope_profile_ids(envelope)):
                    if profile_id not in profiles.commits:
                        collection.blockers.add(
                            f"release_protection_profile_missing:{_code_id(profile_id)}"
                        )
                        continue
                    commit = profiles.commits[profile_id]
                    if commit is None:
                        # The profile pins no release: it runs from the active one.
                        collection.warnings.add(
                            f"release_protection_profile_commit_unpinned:{_code_id(profile_id)}"
                        )
                    else:
                        commits.add(commit)
                expires_at = info.st_mtime + max_lifetime_seconds
                row = {
                    "kind": "live_queue",
                    "owner": queue,
                    "reason": f"live_queue:{queue}/{state}",
                    "run_ref": {
                        "kind": "queue_envelope",
                        "queue": queue,
                        "state": state,
                        "name": name,
                    },
                    "expires_at_epoch": expires_at,
                    "source": source,
                }
                if now >= expires_at:
                    collection.lapse(commits, row, "max_lifetime")
                    collection.warnings.add(
                        f"release_protection_queue_envelope_past_max_lifetime:{code_source}"
                    )
                else:
                    collection.protect(commits, row)
    return accounted, moved


def _live_queue_listing(root: Path) -> set[tuple[str, str]]:
    """``(queue, name)`` of every envelope now in a live state.

    A state that cannot be listed is skipped: the pass already blocked on it.
    """

    listed: set[tuple[str, str]] = set()
    for queue, states in LIVE_QUEUE_STATES.items():
        for state in states:
            try:
                listed.update((queue, name) for name in _json_names(root / queue / state))
            except OSError:
                continue
    return listed


def _collect_queues(
    root: Path,
    profiles: _Profiles,
    *,
    now: float,
    max_lifetime_seconds: int,
    collection: _Collection,
) -> None:
    """Protection from the live queues, read as one consistent snapshot.

    Workers move envelopes between states while the scan runs, and an
    envelope that moves from a state not yet listed into one already read
    would be missed.  After each pass every live state is listed again; when
    an envelope vanished mid-read or one appears that the pass never read, the
    pass is repeated, and a queue still moving after three passes blocks.
    """

    for _attempt in range(_QUEUE_SCAN_ATTEMPTS):
        scan = _Collection()
        accounted, moved = _scan_queues(
            root,
            profiles,
            now=now,
            max_lifetime_seconds=max_lifetime_seconds,
            collection=scan,
        )
        if not moved and _live_queue_listing(root) <= accounted:
            break
    else:
        scan.blockers.add("release_protection_queue_unstable")
    collection.merge(scan)


def _parse_epoch(value: Any) -> float | None:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _text_or(value: Any, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _standing_authorization_state(
    directory: Path,
    profile_id: str,
    documents: Mapping[str, Mapping[str, Any]],
    *,
    now: float,
) -> tuple[str, Mapping[str, Any] | None]:
    """``("live" | "terminal" | "invalid", authorization)`` for one profile id.

    Live means the authorization can still admit a launch.  Terminal means it
    never can again: expired, out of launches, or out of spend.  Anything else
    is invalid, which blocks rather than guesses.
    """

    from .task_evaluation_standing_launch_authorization import (
        consumption_totals,
        load_standing_authorization,
        validate_standing_authorization,
    )

    path = directory / f"{profile_id}.json"
    try:
        _require_regular(path)
        authorization = load_standing_authorization(profile_id=profile_id, directory=directory)
        if authorization is None or authorization.get("profile_id") != profile_id:
            raise ValueError("standing_authorization_identity_invalid")
        _require_regular_children(directory / "consumed" / profile_id)
        launches, spend = consumption_totals(directory=directory, profile_id=profile_id)
        profile = documents.get(profile_id) or {
            # The typed two-step tool's stand-in when the profile is unreadable:
            # it checks bounds and expiry without inventing a per-launch spend.
            "profile_id": profile_id,
            "profile_digest": authorization.get("profile_digest"),
            "allocator": {"max_spend_usd": 0.0},
        }
        blockers = set(
            validate_standing_authorization(
                authorization,
                profile=profile,
                launches_consumed=launches,
                spend_consumed_usd=spend,
                now=datetime.fromtimestamp(now, tz=timezone.utc),
            )
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return "invalid", None
    if not blockers:
        return "live", authorization
    if blockers <= _EXPECTED_TERMINAL_AUTHORIZATION_BLOCKERS:
        return "terminal", authorization
    return "invalid", authorization


def _collect_standing_authorizations(
    directory: Path, profiles: _Profiles, *, now: float, required: bool, collection: _Collection
) -> None:
    """A release stays while an authorization can still launch the profile that runs it.

    When ``required`` (the control-plane root exists), a missing directory is a
    source that cannot be read, not a host without authorizations.
    """

    try:
        if _absent(directory):
            if required:
                collection.blockers.add(
                    f"release_protection_source_missing:{_code_id(directory.name)}"
                )
            return
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("standing_authorization_root_unsafe")
        names = _json_names(directory)
    except (OSError, ValueError):
        collection.blockers.add("release_protection_standing_authorization_root_unreadable")
        return
    for name in names:
        # consumed/ is a directory and step logs are not JSON; only the
        # authorization documents themselves are read.
        profile_id = name[: -len(".json")]
        state, authorization = _standing_authorization_state(
            directory, profile_id, profiles.documents, now=now
        )
        if state == "invalid" or authorization is None:
            collection.blockers.add(
                f"release_protection_standing_authorization_invalid:{_code_id(profile_id)}"
            )
            continue
        commit = (
            profiles.commits[profile_id]
            if profile_id in profiles.commits
            else _profile_id_commit(profile_id)
        )
        row = {
            "kind": "standing_authorization",
            "owner": _text_or(authorization.get("authorized_by"), "unknown"),
            "reason": _text_or(
                authorization.get("authorization_reference"),
                f"unconsumed_standing_authorization:{profile_id}",
            ),
            "run_ref": {"kind": "standing_authorization", "profile_id": profile_id},
            "expires_at_epoch": _parse_epoch(authorization.get("expires_at")),
            "source": f"{directory.name}/{name}",
        }
        if commit is not None:
            if state == "live":
                collection.protect({commit}, row)
            else:
                collection.lapse({commit}, row, "run_terminal")
        elif state != "live":
            continue  # it can never launch again, so it needs no release
        elif profile_id in profiles.documents:
            # The readable profile pins no release, so it runs from the active one.
            collection.warnings.add(
                f"release_protection_profile_commit_unpinned:{_code_id(profile_id)}"
            )
        else:
            # It can still launch, but nothing says which release that needs.
            collection.blockers.add(
                f"release_protection_standing_authorization_commit_unknown:{_code_id(profile_id)}"
            )


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


class RunStateResolver:
    """Whether the run a lease serves can still use its release.

    ``state`` answers ``"terminal"``, ``"live"`` or ``"unknown"``.  A run that
    cannot be read is unknown, never terminal: an unknown run keeps its lease
    until the lease expires.
    """

    def __init__(
        self,
        intent_root: str | Path | None,
        launch_run_root: str | Path | None,
        control_plane_root: str | Path | None,
        now: float,
        *,
        standing_authorization_dir: str | Path | None = None,
        profile_documents: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.intent_root = Path(intent_root) if intent_root is not None else None
        self.launch_run_root = Path(launch_run_root) if launch_run_root is not None else None
        self.control_plane_root = (
            Path(control_plane_root) if control_plane_root is not None else None
        )
        self.now = float(now)
        self.standing_authorization_dir = (
            Path(standing_authorization_dir) if standing_authorization_dir is not None else None
        )
        self.profile_documents = dict(profile_documents or {})

    def state(self, run_ref: Any) -> str:
        if not isinstance(run_ref, Mapping):
            return "unknown"
        resolver = {
            "scene_intent": self._scene_intent,
            "queue_envelope": self._queue_envelope,
            "launch": self._launch,
            "standing_authorization": self._standing_authorization,
        }.get(run_ref.get("kind"))
        if resolver is None:
            return "unknown"
        try:
            return resolver(run_ref)
        except Exception:  # a run we cannot read is unknown, never terminal
            return "unknown"

    def _scene_intent(self, run_ref: Mapping[str, Any]) -> str:
        from . import task_evaluation_scene_intake as intake

        intent_id = run_ref.get("intent_id")
        if self.intent_root is None or not _identifier(intent_id):
            return "unknown"
        directory = self.intent_root / intent_id
        if directory.is_symlink() or not directory.is_dir():
            return "unknown"
        if not _absent(directory / "revoked.json"):
            return "terminal"
        _require_regular(directory / "intent.json")
        intent = intake._read(directory / "intent.json", "intent_digest")
        progression = directory / "progression.json"
        if not _absent(progression):
            _require_regular(progression)
            projection = intake._read(progression, "progression_digest")
            if projection.get("intent_digest") != intent.get("intent_digest"):
                return "unknown"
            if projection.get("status") == "completed":
                return "terminal"
        # The same effective window the progression worker enforces, including
        # owner-approved extensions.
        _require_regular_children(directory / "execution-window-extensions")
        if self.now >= intake.effective_execution_expiry(directory, intent):
            return "terminal"
        return "live"

    def _queue_envelope(self, run_ref: Mapping[str, Any]) -> str:
        queue, name = run_ref.get("queue"), run_ref.get("name")
        states = LIVE_QUEUE_STATES.get(queue) if isinstance(queue, str) else None
        if (
            self.control_plane_root is None
            or states is None
            or not isinstance(name, str)
            or not name.endswith(".json")
            or Path(name).name != name
        ):
            return "unknown"
        if any(not _absent(self.control_plane_root / queue / state / name) for state in states):
            return "live"
        return "terminal"

    def _launch(self, run_ref: Mapping[str, Any]) -> str:
        launch_id = run_ref.get("launch_id")
        if self.launch_run_root is None or not _identifier(launch_id):
            return "unknown"
        directory = self.launch_run_root / launch_id
        if not _absent(directory / "launch_receipt.json") or not _absent(
            self.launch_run_root / f"{launch_id}{_OFFLOAD_POINTER_SUFFIX}"
        ):
            return "terminal"
        if directory.is_dir() and not directory.is_symlink():
            return "live"
        return "unknown"

    def _standing_authorization(self, run_ref: Mapping[str, Any]) -> str:
        profile_id = run_ref.get("profile_id")
        if self.standing_authorization_dir is None or not _identifier(profile_id):
            return "unknown"
        state, _authorization = _standing_authorization_state(
            self.standing_authorization_dir, profile_id, self.profile_documents, now=self.now
        )
        return {"live": "live", "terminal": "terminal"}.get(state, "unknown")


def binding_commits(binding: Mapping[str, Any]) -> list[str] | None:
    """The commits a valid required-evidence binding names, or ``None``.

    ``retained_release.tree`` is a git tree id and is never read as a commit.
    """

    commit = _valid_commit(binding.get("source_commit"))
    reason = binding.get("reason")
    if (
        binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("status") != "required"
        or commit is None
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        return None
    commits = {commit}
    retained = binding.get("retained_release")
    if isinstance(retained, Mapping):
        retained_commit = _valid_commit(retained.get("source_commit"))
        if retained_commit is not None:
            commits.add(retained_commit)
    return sorted(commits)


def _run_ref_valid(value: Any) -> bool:
    return value is None or (isinstance(value, Mapping) and _identifier(value.get("kind")))


def _inline_lease(binding: Mapping[str, Any]) -> dict[str, Any] | None:
    """Lease fields a future binding writer carries inline; ``None`` for a legacy binding."""

    if not any(field in binding for field in _INLINE_LEASE_FIELDS):
        return None
    owner, expires_at = binding.get("owner"), binding.get("expires_at_epoch")
    # Inline bytes cannot be renewed, so without an explicit bound the
    # writer's own expiry is also the maximum lifetime.
    max_expires_at = binding.get("max_expires_at_epoch", expires_at)
    run_ref = binding.get("run_ref")
    if (
        not isinstance(owner, str)
        or not owner.strip()
        or not _number(expires_at)
        or not _number(max_expires_at)
        or not _run_ref_valid(run_ref)
    ):
        raise ValueError("release_protection_inline_lease_invalid")
    return {
        "owner": owner.strip(),
        "reason": binding["reason"],
        "run_ref": run_ref,
        "expires_at_epoch": float(expires_at),
        "max_expires_at_epoch": float(max_expires_at),
    }


def _scene_intent_run_ref(binding: Mapping[str, Any], intent_root: Path | None) -> dict | None:
    """The scene intent whose factory output holds the binding's evidence, if any.

    Factory output is laid out ``<factory_output_root>/<intent_id>/<attempt_id>/...``,
    so the first component of ``evidence.path`` that is an intent directory
    names the run.
    """

    if intent_root is None:
        return None
    evidence = binding.get("evidence")
    path_text = evidence.get("path") if isinstance(evidence, Mapping) else None
    if not isinstance(path_text, str) or not path_text:
        return None
    for part in Path(path_text).parts:
        if not _identifier(part):
            continue
        candidate = intent_root / part
        if candidate.is_dir() and not candidate.is_symlink():
            return {"kind": "scene_intent", "intent_id": part}
    return None


def _sealed_lease(lease: Mapping[str, Any]) -> dict[str, Any]:
    sealed = {**lease, "lease_digest": ""}
    sealed["lease_digest"] = canonical_digest(sealed, digest_field="lease_digest")
    return sealed


def _sidecar_problem(
    lease: Any, *, name: str, commits: list[str], binding_sha256: str
) -> str | None:
    if (
        not isinstance(lease, Mapping)
        or lease.get("schema_version") != LEASE_SCHEMA
        or lease.get("binding") != name
        or lease.get("lease_digest") != canonical_digest(lease, digest_field="lease_digest")
    ):
        return f"release_protection_lease_invalid:{_code_id(name)}"
    if lease.get("binding_sha256") != binding_sha256:
        return f"release_protection_binding_changed:{_code_id(name)}"
    recorded = lease.get("commits")
    if (
        not isinstance(recorded, list)
        or not all(isinstance(commit, str) for commit in recorded)
        or sorted(recorded) != commits
        or not isinstance(lease.get("owner"), str)
        or not lease["owner"].strip()
        or not isinstance(lease.get("reason"), str)
        or not lease["reason"].strip()
        or not _run_ref_valid(lease.get("run_ref"))
        or not _number(lease.get("expires_at_epoch"))
        or not _number(lease.get("max_expires_at_epoch"))
    ):
        return f"release_protection_lease_invalid:{_code_id(name)}"
    return None


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        view = view[os.write(descriptor, view):]


def _write_sidecar(path: Path, lease: Mapping[str, Any], *, replace: bool) -> None:
    """Publish a complete sidecar (mode 0640) or nothing.

    The bytes go to a temporary file in the same directory and are fsynced
    first; creation then links that file into place (refusing an existing
    sidecar) and renewal renames it over the old one.  A short write (a full
    disk) leaves no partial sidecar that would block every later deploy.
    """

    directory = path.parent
    for level in (directory.parent, directory):
        if _absent(level):
            level.mkdir(mode=0o750)
        elif level.is_symlink() or not level.is_dir():
            raise ValueError("release_protection_lease_root_unsafe")
    payload = (
        json.dumps(lease, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    try:
        try:
            os.fchmod(descriptor, 0o640)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    _fsync_directory(directory)


def evaluate_binding_lease(
    *,
    name: str,
    binding_sha256: str,
    binding: Mapping[str, Any],
    lease_root: str | Path | None,
    resolver: RunStateResolver,
    now: float,
    migrate: bool,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    max_lifetime_seconds: int = DEFAULT_MAX_LIFETIME_SECONDS,
) -> dict[str, Any]:
    """Decide whether one required-evidence binding still protects its commits.

    ``binding_sha256`` is ``"sha256:<hex>"`` of the binding's exact bytes,
    which are never written.  The lease comes from inline fields, else the sidecar
    ``<lease_root>/bindings/<name>.lease.v1.json``, else a legacy migration:
    written exclusively when ``migrate`` is true, otherwise only evaluated as
    the lease that migration would write.  Only ``migrate`` writes, and it also
    renews the sidecar of a live run that is within half a TTL of expiring.

    Returns ``{"status": "protected" | "lapsed" | "blocked", "commits", "lease",
    "run_state", "why", "blocker", "warnings", "migrated", "renewed",
    "sidecar"}``.
    """

    outcome: dict[str, Any] = {
        "status": "blocked",
        "commits": [],
        "lease": None,
        "run_state": None,
        "why": None,
        "blocker": None,
        "warnings": [],
        "migrated": False,
        "renewed": False,
        "sidecar": None,
    }

    def blocked(code: str) -> dict[str, Any]:
        outcome["blocker"] = code
        return outcome

    commits = binding_commits(binding)
    if commits is None:
        return blocked(f"release_protection_binding_invalid:{_code_id(name)}")
    outcome["commits"] = commits
    try:
        lease = _inline_lease(binding)
    except ValueError:
        return blocked(f"release_protection_binding_invalid:{_code_id(name)}")
    lease_source = "inline"
    sidecar = (
        Path(lease_root) / "bindings" / f"{name}{_SIDECAR_SUFFIX}"
        if lease_root is not None
        else None
    )
    try:
        sidecar_exists = sidecar is not None and not _absent(sidecar)
    except OSError:
        return blocked(f"release_protection_lease_invalid:{_code_id(name)}")
    if lease is None and sidecar_exists:
        try:
            sidecar_payload, recorded, _info = _read_document(sidecar)  # type: ignore[arg-type]
        except (OSError, ValueError):
            return blocked(f"release_protection_lease_invalid:{_code_id(name)}")
        problem = _sidecar_problem(
            recorded, name=name, commits=commits, binding_sha256=binding_sha256
        )
        if problem is not None:
            return blocked(problem)
        lease, lease_source = dict(recorded), "sidecar"
        outcome["sidecar"] = {
            "path": str(sidecar),
            "sha256": "sha256:" + hashlib.sha256(sidecar_payload).hexdigest(),
            "size_bytes": len(sidecar_payload),
        }
    elif lease is None:
        lease = _sealed_lease(
            {
                "schema_version": LEASE_SCHEMA,
                "binding": name,
                "binding_sha256": binding_sha256,
                "commits": commits,
                "owner": "legacy-migration",
                "reason": binding["reason"],
                "run_ref": _scene_intent_run_ref(binding, resolver.intent_root),
                "created_at_epoch": now,
                "expires_at_epoch": now + ttl_seconds,
                "max_expires_at_epoch": now + max_lifetime_seconds,
                "migrated": True,
            }
        )
        lease_source = "would_be_migrated"
        if migrate and sidecar is not None:
            try:
                _write_sidecar(sidecar, lease, replace=False)
            except (OSError, ValueError):
                return blocked(f"release_protection_lease_write_failed:{_code_id(name)}")
            lease_source, outcome["migrated"] = "migrated", True
    expires_at = float(lease["expires_at_epoch"])
    max_expires_at = float(lease["max_expires_at_epoch"])
    run_state = resolver.state(lease.get("run_ref"))
    outcome["run_state"] = run_state
    past_max_lifetime = f"release_protection_lease_past_max_lifetime:{_code_id(name)}"
    if run_state == "terminal":
        outcome["status"], outcome["why"] = "lapsed", "run_terminal"
    elif run_state == "live" and now >= max_expires_at:
        outcome["status"], outcome["why"] = "lapsed", "max_lifetime"
        outcome["warnings"].append(past_max_lifetime)
    elif run_state == "live":
        outcome["status"] = "protected"
        renewed_expiry = min(now + ttl_seconds, max_expires_at)
        if (
            migrate
            and lease_source == "sidecar"
            and expires_at - now < ttl_seconds / 2
            and renewed_expiry > expires_at
        ):
            renewed = _sealed_lease(
                {**lease, "expires_at_epoch": renewed_expiry, "renewed_at_epoch": now}
            )
            try:
                _write_sidecar(sidecar, renewed, replace=True)  # type: ignore[arg-type]
            except (OSError, ValueError):
                # The run is live, so it stays protected this time; the next
                # deploy retries the renewal.
                outcome["warnings"].append(f"release_protection_lease_renewal_failed:{_code_id(name)}")
            else:
                lease, expires_at, outcome["renewed"] = renewed, renewed_expiry, True
    elif now >= expires_at:
        # An unknown or unnamed run keeps its lease only until it expires.
        outcome["status"], outcome["why"] = "lapsed", "expired"
    elif now >= max_expires_at:
        # Only a hand-edited lease can outlive its own maximum lifetime.
        outcome["status"], outcome["why"] = "lapsed", "max_lifetime"
        outcome["warnings"].append(past_max_lifetime)
    else:
        outcome["status"] = "protected"
    outcome["lease"] = {
        "owner": lease["owner"],
        "reason": lease["reason"],
        "run_ref": lease.get("run_ref"),
        "expires_at_epoch": expires_at,
        "max_expires_at_epoch": max_expires_at,
        "lease_source": lease_source,
    }
    return outcome


def _collect_bindings(
    sources: ProtectionSources,
    resolver: RunStateResolver,
    *,
    now: float,
    migrate: bool,
    ttl_seconds: int,
    max_lifetime_seconds: int,
    required: bool,
    collection: _Collection,
) -> None:
    root = Path(sources.binding_root)
    try:
        if _absent(root):
            if required:
                collection.blockers.add(f"release_protection_source_missing:{_code_id(root.name)}")
            return
        if root.is_symlink() or not root.is_dir():
            raise ValueError("binding_root_unsafe")
        names = _json_names(root)
    except (OSError, ValueError):
        collection.blockers.add("release_protection_binding_root_unreadable")
        return
    for name in names:
        try:
            payload, binding, _info = _read_document(root / name)
        except (OSError, ValueError):
            binding = None
        if isinstance(binding, Mapping) and binding.get("schema_version") == RETENTION_PLAN_SCHEMA:
            # A dry-run plan written into the wrong directory lists every
            # commit; it is not evidence and protects nothing.
            collection.warnings.add(f"misplaced_retention_plan:{_code_id(name)}")
            continue
        if not isinstance(binding, Mapping):
            collection.blockers.add(f"release_protection_binding_invalid:{_code_id(name)}")
            continue
        outcome = evaluate_binding_lease(
            name=name,
            binding_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
            binding=binding,
            lease_root=sources.lease_root,
            resolver=resolver,
            now=now,
            migrate=migrate,
            ttl_seconds=ttl_seconds,
            max_lifetime_seconds=max_lifetime_seconds,
        )
        collection.warnings.update(outcome["warnings"])
        if outcome["migrated"]:
            collection.migrated.add(name)
        if outcome["renewed"]:
            collection.renewed.add(name)
        if outcome["status"] == "blocked":
            collection.blockers.add(outcome["blocker"])
            continue
        lease = outcome["lease"]
        row = {
            "kind": "retention_binding",
            "owner": lease["owner"],
            "reason": lease["reason"],
            "run_ref": lease["run_ref"],
            "expires_at_epoch": lease["expires_at_epoch"],
            "source": f"{root.name}/{name}",
        }
        if outcome["status"] == "protected":
            collection.protect(outcome["commits"], row)
        else:
            collection.lapse(outcome["commits"], row, outcome["why"])


def _collect_configuration(config_files: tuple[Path, ...], collection: _Collection) -> None:
    configured = [Path(path) for path in config_files]
    pending = [(path, False) for path in configured]
    seen: set[str] = set()
    while pending:
        path, followed = pending.pop(0)
        if str(path) in seen:
            continue
        seen.add(str(path))
        name = path.name
        try:
            if _absent(path) and not followed:
                continue  # an optional configuration file this host does not use
            _payload, value, _info = _read_document(path)
        except (OSError, ValueError):
            collection.blockers.add(f"release_protection_config_unreadable:{_code_id(name)}")
            continue
        collection.protect(
            _managed_tree_commits(value),
            {
                "kind": CONFIG_KIND,
                "owner": name,
                "reason": f"configured_runtime:{name}",
                "run_ref": None,
                "expires_at_epoch": None,
                "source": name,
            },
        )
        # The scene-preparation bootstrap may name a non-default machinery file.
        machinery = value.get("public_scene_machinery_path") if isinstance(value, Mapping) else None
        if isinstance(machinery, str) and machinery:
            target = Path(machinery)
            if target.is_absolute():
                pending.append((target, True))
            else:
                collection.blockers.add(f"release_protection_config_unreadable:{_code_id(name)}")


def collect_release_protections(
    sources: ProtectionSources,
    *,
    now: float,
    migrate: bool,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    max_lifetime_seconds: int = DEFAULT_MAX_LIFETIME_SECONDS,
) -> dict[str, Any]:
    """Typed protection: which commits something live still needs, and why.

    Returns ``{"schema_version", "collected_at_epoch", "leases", "lapsed",
    "migrated", "renewed", "warnings", "blockers"}``.  Every lease row is
    ``{"commit", "kind", "owner", "reason", "run_ref", "expires_at_epoch",
    "source"}``; a lapsed row adds ``why``.  Any blocker means the caller
    cannot know what is live and must retire nothing.

    ``migrate`` is the only write: it creates the sidecar lease of a legacy
    binding and renews the sidecar of a live run.  Dry runs pass ``False``.
    """

    if (
        isinstance(ttl_seconds, bool)
        or isinstance(max_lifetime_seconds, bool)
        or not isinstance(ttl_seconds, int)
        or not isinstance(max_lifetime_seconds, int)
        or not 0 < ttl_seconds <= max_lifetime_seconds
    ):
        raise ValueError("release_protection_input_invalid")
    collection = _Collection()
    profiles = _read_profiles(Path(sources.profile_dir), collection)
    control_plane = Path(sources.control_plane_root)
    present = not control_plane.is_symlink() and control_plane.is_dir()
    if not present:
        collection.blockers.add("release_protection_control_plane_root_missing")
    else:
        _collect_queues(
            control_plane,
            profiles,
            now=now,
            max_lifetime_seconds=max_lifetime_seconds,
            collection=collection,
        )
    _collect_standing_authorizations(
        Path(sources.standing_authorization_dir),
        profiles,
        now=now,
        required=present,
        collection=collection,
    )
    resolver = RunStateResolver(
        sources.intent_root,
        sources.launch_run_root,
        sources.control_plane_root,
        now,
        standing_authorization_dir=sources.standing_authorization_dir,
        profile_documents=profiles.documents,
    )
    _collect_bindings(
        sources,
        resolver,
        now=now,
        migrate=migrate,
        ttl_seconds=ttl_seconds,
        max_lifetime_seconds=max_lifetime_seconds,
        required=present,
        collection=collection,
    )
    _collect_configuration(tuple(sources.config_files), collection)
    return collection.result(now)


__all__ = [
    "BINDING_SCHEMA",
    "CONFIG_KIND",
    "DEFAULT_CONTROL_PLANE_ROOT",
    "DEFAULT_LEASE_ROOT",
    "DEFAULT_MAX_LIFETIME_SECONDS",
    "DEFAULT_PROTECTION_CONFIG_FILES",
    "DEFAULT_PROTECTION_SOURCES",
    "DEFAULT_TTL_SECONDS",
    "LEASE_ALERT_THRESHOLD",
    "LEASE_KINDS",
    "LEASE_SCHEMA",
    "LIVE_QUEUE_STATES",
    "PROTECTIONS_SCHEMA",
    "ProtectionSources",
    "RETENTION_PLAN_SCHEMA",
    "RunStateResolver",
    "binding_commits",
    "collect_release_protections",
    "evaluate_binding_lease",
]
