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
``configured_runtime``
    A runtime path named by current host configuration.  It has no expiry: the
    configuration is re-read on every deploy.

No free-text search happens anywhere: a commit comes only from a named field or
from a path string that names a managed release or runtime tree.  Nothing here
deletes anything; the retirement plan decides from these rows.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEASE_SCHEMA = "control_plane_release_lease.v1"
PROTECTIONS_SCHEMA = "control_plane_release_protections.v1"
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


def _read_document(path: Path) -> tuple[bytes, Any, os.stat_result]:
    """Read one regular JSON file without following a symlink.

    Raises ``OSError`` or ``ValueError`` for anything that is not a regular
    file of at most 16 MiB holding valid JSON.
    """

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
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


def _trailing_commit(profile_id: str) -> str | None:
    """The last dash-separated 40-hex segment of ``<prefix>-<commit>[-...]``."""

    for segment in reversed(profile_id.split("-")):
        if _COMMIT.fullmatch(segment):
            return segment
    return None


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
    return _trailing_commit(profile_id)


def _read_profiles(profile_dir: Path, collection: _Collection) -> _Profiles:
    """Map each launch profile to the commit it runs, reading structure only.

    A profile never protects by itself, so an unreadable one is a warning; a
    live reference to it is what blocks.
    """

    commits: dict[str, str | None] = {}
    documents: dict[str, Mapping[str, Any]] = {}
    if not profile_dir.is_dir() or profile_dir.is_symlink():
        collection.warnings.add("release_protection_profile_dir_unreadable")
        return _Profiles(commits, documents)
    for name in _json_names(profile_dir):
        try:
            _payload, profile, _info = _read_document(profile_dir / name)
        except (OSError, ValueError):
            profile = None
        if not isinstance(profile, Mapping):
            collection.warnings.add(f"release_protection_profile_unreadable:{name}")
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


def _collect_queues(
    root: Path,
    profiles: _Profiles,
    *,
    now: float,
    max_lifetime_seconds: int,
    collection: _Collection,
) -> None:
    for queue, states in LIVE_QUEUE_STATES.items():
        for state in states:
            directory = root / queue / state
            if not os.path.lexists(directory):
                continue  # a queue state never used on this host holds nothing
            try:
                if directory.is_symlink() or not directory.is_dir():
                    raise ValueError("queue_state_unsafe")
                names = _json_names(directory)
            except (OSError, ValueError):
                collection.blockers.add(f"release_protection_queue_unreadable:{queue}/{state}")
                continue
            for name in names:
                source = f"{queue}/{state}/{name}"
                try:
                    _payload, envelope, info = _read_document(directory / name)
                except (OSError, ValueError):
                    envelope = None
                if not isinstance(envelope, Mapping):
                    collection.blockers.add(f"release_protection_queue_unreadable:{source}")
                    continue
                commits = _envelope_commits(envelope)
                for profile_id in sorted(_envelope_profile_ids(envelope)):
                    if profile_id not in profiles.commits:
                        collection.blockers.add(f"release_protection_profile_missing:{profile_id}")
                        continue
                    commit = profiles.commits[profile_id]
                    if commit is None:
                        # The profile pins no release: it runs from the active one.
                        collection.warnings.add(
                            f"release_protection_profile_commit_unknown:{profile_id}"
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
                        f"release_protection_queue_envelope_past_max_lifetime:{source}"
                    )
                else:
                    collection.protect(commits, row)


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
    directory: Path, profile_id: str, profiles: _Profiles, *, now: float
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
        if path.is_symlink() or not path.is_file():
            raise ValueError("standing_authorization_source_invalid")
        authorization = load_standing_authorization(profile_id=profile_id, directory=directory)
        if authorization is None or authorization.get("profile_id") != profile_id:
            raise ValueError("standing_authorization_identity_invalid")
        launches, spend = consumption_totals(directory=directory, profile_id=profile_id)
        profile = profiles.documents.get(profile_id) or {
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
    directory: Path, profiles: _Profiles, *, now: float, collection: _Collection
) -> None:
    """A release stays while an authorization can still launch the profile that runs it."""

    if not os.path.lexists(directory):
        return
    try:
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
            directory, profile_id, profiles, now=now
        )
        if state == "invalid" or authorization is None:
            collection.blockers.add(
                f"release_protection_standing_authorization_invalid:{profile_id}"
            )
            continue
        commit = (
            profiles.commits[profile_id]
            if profile_id in profiles.commits
            else _trailing_commit(profile_id)
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
        if commit is None:
            # The profile pins no release, so it runs from the active one.
            collection.warnings.add(
                f"release_protection_standing_authorization_commit_unknown:{profile_id}"
            )
        elif state == "live":
            collection.protect({commit}, row)
        else:
            collection.lapse({commit}, row, "run_terminal")


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
        if not os.path.lexists(path) and not followed:
            continue  # an optional configuration file this host does not use
        try:
            _payload, value, _info = _read_document(path)
        except (OSError, ValueError):
            collection.blockers.add(f"release_protection_config_unreadable:{name}")
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
                collection.blockers.add(f"release_protection_config_unreadable:{name}")


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
    if control_plane.is_symlink() or not control_plane.is_dir():
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
        Path(sources.standing_authorization_dir), profiles, now=now, collection=collection
    )
    _collect_configuration(tuple(sources.config_files), collection)
    return collection.result(now)


__all__ = [
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
    "collect_release_protections",
]
