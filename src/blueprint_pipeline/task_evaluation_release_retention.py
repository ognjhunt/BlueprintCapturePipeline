"""Fail-closed retention for SHA-pinned Task Evaluation release artifacts.

Every control-plane deploy publishes three immutable trees keyed by the exact
protected-main commit: a detached Git worktree and the splat-render and scene-
configuration runtimes.  Those trees are intentionally never mutated, but the
deploy path historically never retired them either.  A sequence of production
iterations therefore retained a full Chrome runtime per SHA until the host
filled its disk.

This module makes retirement a two-step operation.  The first invocation writes
a digest-bound dry-run plan.  Applying that plan requires its exact bytes, an
explicit acknowledgement, and a fresh scan that must reproduce the same plan.
Anything ambiguous blocks the whole apply: malformed live state, symlinks at a
managed boundary, unknown managed children, a changed inode, or a changed
protected binding.  Only direct children named by a lowercase 40-hex commit are
ever candidates.

The tool contacts no provider and reads no secret.  It is intentionally exposed
as ``python -m blueprint_pipeline.task_evaluation_release_retention`` rather
than adding another project CLI entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess  # nosec B404 - fixed Git argv over validated worktrees
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    validate_launch_profile,
    validate_public_launch_profile_descriptor,
)
from blueprint_pipeline.task_evaluation_standing_launch_authorization import (
    StandingAuthorizationError,
    consumption_totals,
    load_standing_authorization,
    validate_standing_authorization,
)
from blueprint_pipeline.task_evaluation_release_reference_lock import (
    release_reference_lock,
)


SCHEMA_VERSION = "task_evaluation_release_retention_plan.v1"
APPLY_SCHEMA_VERSION = "task_evaluation_release_retention_apply.v1"
EVIDENCE_BINDING_SCHEMA_VERSION = "task_evaluation_release_retention_binding.v1"
APPLY_ACKNOWLEDGEMENT = "reap-task-evaluation-release-artifacts"

DEFAULT_RELEASE_ROOT = Path("/opt/blueprint/task-evaluation-control-plane-releases")
DEFAULT_RUNTIME_ROOT = Path(
    "/var/lib/blueprint/task-evaluation-inputs/system-runtimes"
)
DEFAULT_ACTIVE_LINK = Path("/opt/blueprint/task-evaluation-control-plane")
DEFAULT_PROFILE_DIR = Path("/etc/blueprint/task-evaluation-launch-profiles")
DEFAULT_PUBLIC_CATALOG = Path(
    "/var/lib/blueprint/pipeline-control-plane/"
    "task-evaluation-launch-profile-catalog.json"
)
DEFAULT_STANDING_AUTHORIZATION_DIR = Path(
    "/var/lib/blueprint/pipeline-control-plane/standing-authorizations"
)
DEFAULT_EVIDENCE_BINDING_ROOT = Path(
    "/var/lib/blueprint/pipeline-control-plane/"
    "task-evaluation-release-retention-bindings"
)
_QUEUE_BASE = Path("/var/lib/blueprint/pipeline-control-plane")
_LIVE_QUEUE_NAMES = (
    "task-evaluation-launch-preparations",
    "task-evaluation-scene-constructions",
    "task-evaluation-episode-compilations",
    "task-evaluation-launch-activations",
    "task-evaluation-launches",
    "task-evaluation-terminal-resource-releases",
)
DEFAULT_LIVE_REFERENCE_ROOTS = tuple(
    _QUEUE_BASE / queue / state
    for queue in _LIVE_QUEUE_NAMES
    for state in ("pending", "processing")
)
MANAGED_RUNTIME_COMPONENTS = ("splat-render", "scene-configuration")
DEFAULT_MINIMUM_AGE_SECONDS = 24 * 60 * 60

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_COMMIT_SEARCH_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")
_PUBLICATION_RECEIPT_RE = re.compile(r"([0-9a-f]{40})\.publication\.v1\.json")
_PROFILE_ID_KEYS = frozenset({"profile_id", "launch_profile_id"})
_EXPECTED_TERMINAL_AUTHORIZATION_BLOCKERS = frozenset(
    {
        "standing_authorization_expired",
        "standing_authorization_launches_exhausted",
        "standing_authorization_spend_ceiling_reached",
    }
)


class ReleaseRetentionError(ValueError):
    """The retirement boundary could not be proven safe."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_digest(value: Mapping[str, Any], *, digest_field: str) -> str:
    body = dict(value)
    body.pop(digest_field, None)
    return "sha256:" + hashlib.sha256(_canonical_json(body)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _valid_commit(value: Any) -> str | None:
    text = str(value or "")
    return text if _COMMIT_RE.fullmatch(text) else None


def _absolute_path(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ReleaseRetentionError(f"release_retention_{field}_must_be_absolute")
    return path


def _assert_directory(path: Path, *, blocker: str, allow_missing: bool = False) -> None:
    if not os.path.lexists(path):
        if allow_missing:
            return
        raise ReleaseRetentionError(blocker + "_missing")
    if path.is_symlink() or not path.is_dir():
        raise ReleaseRetentionError(blocker + "_invalid")


def _tree_size(path: Path) -> int:
    """Count allocated artifact bytes without following any symlink."""

    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            total += int(info.st_size)
            continue
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(current) as entries:
                stack.extend(Path(entry.path) for entry in entries)
            continue
        total += int(info.st_size)
    return total


def _artifact_snapshot(path: Path, *, kind: str, commit: str) -> dict[str, Any]:
    if path.name != commit or path.parent == path or path.is_symlink() or not path.is_dir():
        raise ReleaseRetentionError(
            f"release_retention_managed_target_invalid:{kind}:{commit}"
        )
    info = path.lstat()
    return {
        "kind": kind,
        "path": str(path),
        "source_commit": commit,
        "size_bytes": _tree_size(path),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mtime_ns": int(info.st_mtime_ns),
    }


_RUNTIME_PUBLICATION_RECEIPTS = {
    "runtime_splat_render": (
        "task_evaluation_splat_render_runtime_publication.v1",
        "runtime_root",
    ),
    "runtime_scene_configuration": (
        "task_evaluation_scene_configuration_toolchain_publication.v1",
        "toolchain_root",
    ),
}


def _publication_receipt_snapshot(
    path: Path, *, kind: str, commit: str, runtime_path: Path
) -> dict[str, Any]:
    expected = _RUNTIME_PUBLICATION_RECEIPTS.get(kind)
    if expected is None:
        raise ReleaseRetentionError(
            f"release_retention_publication_receipt_kind_invalid:{kind}"
        )
    schema_version, root_field = expected
    value, _evidence = _read_json_file(
        path, blocker="release_retention_publication_receipt_unreadable"
    )
    if not isinstance(value, Mapping):
        raise ReleaseRetentionError(
            f"release_retention_publication_receipt_invalid:{kind}:{commit}"
        )
    receipt = dict(value)
    if (
        receipt.get("schema_version") != schema_version
        or receipt.get("status") != "published_and_read_back"
        or receipt.get("source_commit") != commit
        or receipt.get(root_field) != str(runtime_path)
        or receipt.get("full_byte_service_account_readback_passed") is not True
        or receipt.get("provider_mutation_performed") is not False
        or receipt.get("paid_resource_allocated") is not False
        or receipt.get("receipt_digest")
        != _canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise ReleaseRetentionError(
            f"release_retention_publication_receipt_invalid:{kind}:{commit}"
        )
    info = path.lstat()
    return {
        "kind": f"{kind}_publication_receipt",
        "artifact_type": "file",
        "path": str(path),
        "source_commit": commit,
        "size_bytes": int(info.st_size),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mtime_ns": int(info.st_mtime_ns),
        "sha256": _sha256_bytes(path.read_bytes()),
    }


def _managed_children(root: Path, *, kind: str) -> dict[str, list[dict[str, Any]]]:
    _assert_directory(root, blocker=f"release_retention_{kind}_root")
    values: dict[str, list[dict[str, Any]]] = {}
    receipt_paths: dict[str, Path] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        commit = _valid_commit(path.name)
        if commit is not None and not path.is_symlink() and path.is_dir():
            values[commit] = [_artifact_snapshot(path, kind=kind, commit=commit)]
            continue
        receipt_match = _PUBLICATION_RECEIPT_RE.fullmatch(path.name)
        if (
            kind in _RUNTIME_PUBLICATION_RECEIPTS
            and receipt_match is not None
            and not path.is_symlink()
            and path.is_file()
        ):
            receipt_paths[receipt_match.group(1)] = path
            continue
        else:
            raise ReleaseRetentionError(
                f"release_retention_unknown_managed_child:{kind}:{path.name}"
            )
    if kind in _RUNTIME_PUBLICATION_RECEIPTS:
        if set(receipt_paths) != set(values):
            mismatched = sorted(set(receipt_paths) ^ set(values))[0]
            raise ReleaseRetentionError(
                f"release_retention_publication_receipt_pair_invalid:{kind}:{mismatched}"
            )
        for commit, receipt_path in receipt_paths.items():
            values[commit].append(
                _publication_receipt_snapshot(
                    receipt_path,
                    kind=kind,
                    commit=commit,
                    runtime_path=root / commit,
                )
            )
    return values


def _read_json_file(path: Path, *, blocker: str) -> tuple[Any, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseRetentionError(f"{blocker}:{path.name}")
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseRetentionError(f"{blocker}:{path.name}") from exc
    return value, {
        "path": str(path),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _json_documents(
    root: Path,
    *,
    blocker: str,
    allow_missing: bool = False,
) -> list[tuple[Path, Any, dict[str, Any]]]:
    _assert_directory(root, blocker=blocker, allow_missing=allow_missing)
    if not os.path.lexists(root):
        return []
    documents: list[tuple[Path, Any, dict[str, Any]]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.suffix != ".json" or path.is_symlink() or not path.is_file():
            raise ReleaseRetentionError(
                f"release_retention_unknown_reference_child:{path}"
            )
        value, evidence = _read_json_file(path, blocker=blocker + "_unreadable")
        documents.append((path, value, evidence))
    return documents


def _extract_bindings(value: Any) -> tuple[set[str], set[str]]:
    commits: set[str] = set()
    profile_ids: set[str] = set()

    def visit(item: Any, *, key: str | None = None) -> None:
        if isinstance(item, Mapping):
            for child_key, child in item.items():
                visit(child, key=str(child_key))
        elif isinstance(item, list):
            for child in item:
                visit(child, key=key)
        elif isinstance(item, str):
            commits.update(_COMMIT_SEARCH_RE.findall(item))
            if key in _PROFILE_ID_KEYS and item.strip():
                profile_ids.add(item.strip())

    visit(value)
    return commits, profile_ids


def _active_commit(*, active_link: Path, release_root: Path) -> str:
    if not active_link.is_symlink():
        raise ReleaseRetentionError("release_retention_active_link_invalid")
    try:
        target = active_link.resolve(strict=True)
    except OSError as exc:
        raise ReleaseRetentionError("release_retention_active_link_invalid") from exc
    try:
        relative = target.relative_to(release_root)
    except ValueError as exc:
        raise ReleaseRetentionError("release_retention_active_target_outside_root") from exc
    if len(relative.parts) != 1 or _valid_commit(relative.name) is None:
        raise ReleaseRetentionError("release_retention_active_target_invalid")
    if target.is_symlink() or not target.is_dir():
        raise ReleaseRetentionError("release_retention_active_target_invalid")
    return relative.name


def _profiles_and_catalog(
    *, profile_dir: Path, public_catalog: Path
) -> tuple[dict[str, dict[str, Any]], set[str], list[dict[str, Any]]]:
    profiles: dict[str, dict[str, Any]] = {}
    documents: list[dict[str, Any]] = []
    for path, value, evidence in _json_documents(
        profile_dir, blocker="release_retention_profile_directory"
    ):
        if not isinstance(value, Mapping):
            raise ReleaseRetentionError(f"release_retention_profile_invalid:{path.name}")
        profile = dict(value)
        blockers = validate_launch_profile(profile)
        if blockers:
            raise ReleaseRetentionError(
                f"release_retention_profile_invalid:{path.name}:{','.join(blockers)}"
            )
        profile_id = str(profile.get("profile_id") or "")
        if path.stem != profile_id or profile_id in profiles:
            raise ReleaseRetentionError(
                f"release_retention_profile_identity_invalid:{path.name}"
            )
        commit = profile.get("source_commit")
        if commit is not None and _valid_commit(commit) is None:
            raise ReleaseRetentionError(
                f"release_retention_profile_source_commit_invalid:{path.name}"
            )
        profiles[profile_id] = profile
        documents.append(evidence)

    catalog_value, catalog_evidence = _read_json_file(
        public_catalog, blocker="release_retention_public_catalog_unreadable"
    )
    documents.append(catalog_evidence)
    if not isinstance(catalog_value, list):
        raise ReleaseRetentionError("release_retention_public_catalog_invalid")
    raw_profiles = catalog_value
    catalog_ids: set[str] = set()
    for descriptor_value in raw_profiles:
        if not isinstance(descriptor_value, Mapping):
            raise ReleaseRetentionError("release_retention_public_catalog_invalid")
        descriptor = dict(descriptor_value)
        blockers = validate_public_launch_profile_descriptor(descriptor)
        if blockers:
            raise ReleaseRetentionError(
                "release_retention_public_catalog_invalid:" + ",".join(blockers)
            )
        profile_id = str(descriptor.get("profile_id") or "")
        profile = profiles.get(profile_id)
        if profile is None or descriptor.get("profile_digest") != profile.get(
            "profile_digest"
        ):
            raise ReleaseRetentionError(
                f"release_retention_public_catalog_profile_mismatch:{profile_id}"
            )
        if profile_id in catalog_ids:
            raise ReleaseRetentionError(
                f"release_retention_public_catalog_profile_duplicate:{profile_id}"
            )
        catalog_ids.add(profile_id)
    if catalog_ids != set(profiles):
        raise ReleaseRetentionError("release_retention_public_catalog_incomplete")
    return profiles, catalog_ids, documents


def _standing_authorization_protections(
    *,
    profiles: Mapping[str, Mapping[str, Any]],
    directory: Path,
    now: datetime,
) -> tuple[
    dict[str, set[str]], list[dict[str, Any]], list[dict[str, Any]]
]:
    _assert_directory(
        directory, blocker="release_retention_standing_authorization_root"
    )
    protected: dict[str, set[str]] = {}
    documents: list[dict[str, Any]] = []
    terminal_orphans: list[dict[str, Any]] = []
    known_authorization_files = {f"{profile_id}.json" for profile_id in profiles}
    for child in sorted(directory.iterdir(), key=lambda item: item.name):
        if child.name == "consumed" and child.is_dir() and not child.is_symlink():
            continue
        if child.is_symlink() or not child.is_file():
            raise ReleaseRetentionError(
                f"release_retention_unknown_standing_authorization_child:{child.name}"
            )
        if child.name in known_authorization_files:
            continue
        profile_id = child.stem
        try:
            authorization = load_standing_authorization(
                profile_id=profile_id, directory=directory
            )
            launches, spend = consumption_totals(
                directory=directory, profile_id=profile_id
            )
        except StandingAuthorizationError as exc:
            raise ReleaseRetentionError(str(exc)) from exc
        if authorization is None or authorization.get("profile_id") != profile_id:
            raise ReleaseRetentionError(
                f"release_retention_unknown_standing_authorization_child:{child.name}"
            )
        synthetic_profile = {
            "profile_id": profile_id,
            "profile_digest": authorization.get("profile_digest"),
            "allocator": {"max_spend_usd": 0.0},
        }
        blockers = set(
            validate_standing_authorization(
                authorization,
                profile=synthetic_profile,
                launches_consumed=launches,
                spend_consumed_usd=spend,
                now=now,
            )
        )
        if not blockers or blockers - _EXPECTED_TERMINAL_AUTHORIZATION_BLOCKERS:
            raise ReleaseRetentionError(
                f"release_retention_unknown_standing_authorization_child:{child.name}"
            )
        _value, evidence = _read_json_file(
            child,
            blocker="release_retention_standing_authorization_unreadable",
        )
        documents.append(evidence)
        terminal_orphans.append(
            {
                "path": str(child),
                "profile_id": profile_id,
                "terminal_blockers": sorted(blockers),
                "launches_consumed": launches,
                "spend_consumed_usd": spend,
                "source_release_protected": False,
            }
        )

    for profile_id, profile in sorted(profiles.items()):
        authorization_path = directory / f"{profile_id}.json"
        if not authorization_path.exists():
            continue
        _value, evidence = _read_json_file(
            authorization_path,
            blocker="release_retention_standing_authorization_unreadable",
        )
        documents.append(evidence)
        try:
            authorization = load_standing_authorization(
                profile_id=profile_id, directory=directory
            )
            launches, spend = consumption_totals(
                directory=directory, profile_id=profile_id
            )
        except StandingAuthorizationError as exc:
            raise ReleaseRetentionError(str(exc)) from exc
        if authorization is None:
            raise ReleaseRetentionError(
                f"release_retention_standing_authorization_disappeared:{profile_id}"
            )
        blockers = set(
            validate_standing_authorization(
                authorization,
                profile=profile,
                launches_consumed=launches,
                spend_consumed_usd=spend,
                now=now,
            )
        )
        unexpected = blockers - _EXPECTED_TERMINAL_AUTHORIZATION_BLOCKERS
        if unexpected:
            raise ReleaseRetentionError(
                f"release_retention_standing_authorization_invalid:{profile_id}:"
                + ",".join(sorted(unexpected))
            )
        if not blockers:
            commit = _valid_commit(profile.get("source_commit"))
            if commit is not None:
                protected.setdefault(commit, set()).add(
                    f"unconsumed_standing_authorization:{profile_id}"
                )
    return protected, documents, terminal_orphans


def _evidence_binding_protections(
    root: Path,
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    protected: dict[str, set[str]] = {}
    documents: list[dict[str, Any]] = []
    for path, value, evidence in _json_documents(
        root,
        blocker="release_retention_evidence_binding_root",
        allow_missing=True,
    ):
        if not isinstance(value, Mapping):
            raise ReleaseRetentionError(
                f"release_retention_evidence_binding_invalid:{path.name}"
            )
        commit = _valid_commit(value.get("source_commit"))
        reason = str(value.get("reason") or "").strip()
        if (
            value.get("schema_version") != EVIDENCE_BINDING_SCHEMA_VERSION
            or value.get("status") != "required"
            or commit is None
            or not reason
        ):
            raise ReleaseRetentionError(
                f"release_retention_evidence_binding_invalid:{path.name}"
            )
        protected.setdefault(commit, set()).add(f"required_evidence:{path.name}")
        documents.append(evidence)
    return protected, documents


def _merge_reasons(
    target: dict[str, set[str]], source: Mapping[str, set[str]]
) -> None:
    for commit, reasons in source.items():
        target.setdefault(commit, set()).update(reasons)


def build_release_retention_plan(
    *,
    release_root: str | Path,
    runtime_root: str | Path,
    active_link: str | Path,
    current_deploy_commit: str,
    profile_dir: str | Path,
    public_catalog: str | Path,
    standing_authorization_dir: str | Path,
    live_reference_roots: Sequence[str | Path],
    evidence_binding_root: str | Path,
    keep_commits: Sequence[str] = (),
    minimum_age_seconds: int = DEFAULT_MINIMUM_AGE_SECONDS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return the canonical, non-mutating retention decision."""

    if (
        not isinstance(minimum_age_seconds, int)
        or isinstance(minimum_age_seconds, bool)
        or minimum_age_seconds < 0
    ):
        raise ReleaseRetentionError("release_retention_minimum_age_invalid")
    deploy_commit = _valid_commit(current_deploy_commit)
    if deploy_commit is None:
        raise ReleaseRetentionError("release_retention_deploy_candidate_invalid")
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)

    releases = _absolute_path(release_root, field="release_root").resolve()
    runtimes = _absolute_path(runtime_root, field="runtime_root").resolve()
    active = _absolute_path(active_link, field="active_link")
    profiles_root = _absolute_path(profile_dir, field="profile_dir").resolve()
    catalog_path = _absolute_path(public_catalog, field="public_catalog")
    standing_root = _absolute_path(
        standing_authorization_dir, field="standing_authorization_dir"
    ).resolve()
    evidence_root = _absolute_path(
        evidence_binding_root, field="evidence_binding_root"
    ).resolve()
    live_roots = tuple(
        _absolute_path(path, field="live_reference_root").resolve()
        for path in live_reference_roots
    )
    if not live_roots:
        raise ReleaseRetentionError("release_retention_live_reference_roots_missing")
    if len(set(live_roots)) != len(live_roots):
        raise ReleaseRetentionError("release_retention_live_reference_root_duplicate")

    release_artifacts = _managed_children(releases, kind="control_plane_release")
    runtime_artifacts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for component in MANAGED_RUNTIME_COMPONENTS:
        component_root = runtimes / component
        runtime_artifacts[component] = _managed_children(
            component_root, kind=f"runtime_{component.replace('-', '_')}"
        )

    active_commit = _active_commit(active_link=active, release_root=releases)
    protected: dict[str, set[str]] = {
        active_commit: {"active_release"},
        deploy_commit: {"current_deploy_candidate"},
    }
    for raw_commit in keep_commits:
        commit = _valid_commit(raw_commit)
        if commit is None:
            raise ReleaseRetentionError("release_retention_keep_commit_invalid")
        protected.setdefault(commit, set()).add("operator_pin")

    profiles, _catalog_ids, documents = _profiles_and_catalog(
        profile_dir=profiles_root, public_catalog=catalog_path
    )
    live_profile_ids: set[str] = set()
    live_required_commits: set[str] = set()
    for root in live_roots:
        for path, value, evidence in _json_documents(
            root, blocker="release_retention_live_reference_root"
        ):
            commits, profile_ids = _extract_bindings(value)
            for commit in commits:
                protected.setdefault(commit, set()).add(
                    f"live_reference:{path.parent.name}/{path.name}"
                )
                live_required_commits.add(commit)
            live_profile_ids.update(profile_ids)
            documents.append(evidence)
    for profile_id in sorted(live_profile_ids):
        profile = profiles.get(profile_id)
        if profile is None:
            raise ReleaseRetentionError(
                f"release_retention_live_profile_missing:{profile_id}"
            )
        commit = _valid_commit(profile.get("source_commit"))
        if commit is not None:
            protected.setdefault(commit, set()).add(
                f"live_profile_reference:{profile_id}"
            )
            live_required_commits.add(commit)

    standing_protected, standing_documents, terminal_orphans = (
        _standing_authorization_protections(
        profiles=profiles, directory=standing_root, now=moment
        )
    )
    _merge_reasons(protected, standing_protected)
    live_required_commits.update(standing_protected)
    documents.extend(standing_documents)
    evidence_protected, evidence_documents = _evidence_binding_protections(
        evidence_root
    )
    _merge_reasons(protected, evidence_protected)
    live_required_commits.update(evidence_protected)
    documents.extend(evidence_documents)

    for commit in sorted(live_required_commits):
        if commit not in release_artifacts:
            raise ReleaseRetentionError(
                f"release_retention_live_reference_release_missing:{commit}"
            )

    by_commit: dict[str, list[dict[str, Any]]] = {}
    for commit, artifacts in release_artifacts.items():
        by_commit.setdefault(commit, []).extend(artifacts)
    for component in MANAGED_RUNTIME_COMPONENTS:
        for commit, artifacts in runtime_artifacts[component].items():
            by_commit.setdefault(commit, []).extend(artifacts)

    eligible: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    now_ns = int(moment.timestamp() * 1_000_000_000)
    minimum_age_ns = minimum_age_seconds * 1_000_000_000
    for commit, artifacts in sorted(by_commit.items()):
        artifact_age_ns = min(max(0, now_ns - item["mtime_ns"]) for item in artifacts)
        row = {
            "source_commit": commit,
            "artifacts": sorted(artifacts, key=lambda item: item["kind"]),
            "size_bytes": sum(item["size_bytes"] for item in artifacts),
            "minimum_artifact_age_seconds": artifact_age_ns // 1_000_000_000,
        }
        reasons = sorted(protected.get(commit, set()))
        if reasons:
            retained.append({**row, "retention_reasons": reasons})
        elif artifact_age_ns < minimum_age_ns:
            retained.append({**row, "retention_reasons": ["minimum_age_not_met"]})
        else:
            eligible.append(row)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run",
        "planned_at": moment.isoformat().replace("+00:00", "Z"),
        "managed_roots": {
            "control_plane_releases": str(releases),
            "splat_render_runtimes": str(runtimes / "splat-render"),
            "scene_configuration_runtimes": str(runtimes / "scene-configuration"),
        },
        "active_link": str(active),
        "active_source_commit": active_commit,
        "current_deploy_commit": deploy_commit,
        "profile_dir": str(profiles_root),
        "public_catalog": str(catalog_path),
        "standing_authorization_dir": str(standing_root),
        "live_reference_roots": sorted(str(path) for path in live_roots),
        "evidence_binding_root": str(evidence_root),
        "operator_keep_commits": sorted(set(keep_commits)),
        "minimum_age_seconds": minimum_age_seconds,
        "protected_commits": {
            commit: sorted(reasons) for commit, reasons in sorted(protected.items())
        },
        "terminal_orphan_standing_authorizations": terminal_orphans,
        "reference_documents": sorted(documents, key=lambda item: item["path"]),
        "eligible_commits": eligible,
        "retained_commits": retained,
        "predicted_removed_bytes": sum(row["size_bytes"] for row in eligible),
        "provider_mutation_performed": False,
        "production_artifact_mutation_performed": False,
        "symlinks_followed": False,
    }
    result["plan_digest"] = _canonical_digest(result, digest_field="plan_digest")
    return result


def _read_plan(path: Path) -> dict[str, Any]:
    value, _evidence = _read_json_file(
        path, blocker="release_retention_dry_run_plan_unreadable"
    )
    if not isinstance(value, Mapping):
        raise ReleaseRetentionError("release_retention_dry_run_plan_invalid")
    plan = dict(value)
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("status") != "dry_run"
        or plan.get("plan_digest")
        != _canonical_digest(plan, digest_field="plan_digest")
    ):
        raise ReleaseRetentionError("release_retention_dry_run_plan_invalid")
    return plan


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ReleaseRetentionError(
            f"release_retention_receipt_conflict:{path.name}"
        ) from exc


def _assert_receipt_outside_managed_roots(
    path: Path, managed_roots: Mapping[str, Any]
) -> None:
    resolved = path.resolve()
    for value in managed_roots.values():
        root = Path(str(value or "")).resolve()
        if resolved == root or root in resolved.parents:
            raise ReleaseRetentionError(
                "release_retention_receipt_inside_managed_root"
            )


def _assert_snapshot_current(artifact: Mapping[str, Any]) -> Path:
    path = Path(str(artifact.get("path") or ""))
    commit = _valid_commit(artifact.get("source_commit"))
    kind = str(artifact.get("kind") or "invalid")
    artifact_type = str(artifact.get("artifact_type") or "directory")
    expected_name = (
        commit
        if artifact_type == "directory"
        else f"{commit}.publication.v1.json"
    )
    if (
        commit is None
        or path.name != expected_name
        or path.is_symlink()
        or (artifact_type == "directory" and not path.is_dir())
        or (artifact_type == "file" and not path.is_file())
        or artifact_type not in {"directory", "file"}
    ):
        raise ReleaseRetentionError(
            f"release_retention_target_changed:{kind}:{commit or 'invalid'}"
        )
    info = path.lstat()
    observed = {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mtime_ns": int(info.st_mtime_ns),
        "size_bytes": _tree_size(path) if artifact_type == "directory" else int(info.st_size),
    }
    if artifact_type == "file":
        observed["sha256"] = _sha256_bytes(path.read_bytes())
    for field, value in observed.items():
        if artifact.get(field) != value:
            raise ReleaseRetentionError(
                f"release_retention_target_changed:{kind}:{commit}:{field}"
            )
    return path


def _remove_git_worktree(path: Path, *, commit: str) -> None:
    git_marker = path / ".git"
    if git_marker.is_symlink() or not git_marker.is_file():
        raise ReleaseRetentionError(
            f"release_retention_release_not_git_worktree:{commit}"
        )
    commands = (
        ("rev-parse", "--verify", "HEAD^{commit}"),
        ("status", "--porcelain", "--untracked-files=all"),
    )
    results: list[str] = []
    for args in commands:
        try:
            completed = subprocess.run(  # nosec B603
                ["git", "-C", str(path), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except OSError as exc:
            raise ReleaseRetentionError(
                f"release_retention_git_unavailable:{commit}"
            ) from exc
        if completed.returncode != 0:
            raise ReleaseRetentionError(
                f"release_retention_release_git_probe_failed:{commit}"
            )
        results.append(completed.stdout.strip())
    if results[0].lower() != commit or results[1]:
        raise ReleaseRetentionError(
            f"release_retention_release_worktree_not_clean:{commit}"
        )
    completed = subprocess.run(  # nosec B603
        ["git", "-C", str(path), "worktree", "remove", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0 or os.path.lexists(path):
        raise ReleaseRetentionError(
            f"release_retention_release_remove_failed:{commit}"
        )


def _remove_runtime_tree(path: Path, *, commit: str, kind: str) -> None:
    if path.is_symlink() or not path.is_dir() or path.name != commit:
        raise ReleaseRetentionError(
            f"release_retention_runtime_target_invalid:{kind}:{commit}"
        )

    def make_writable_and_retry(function: Any, target: str, _error: Any) -> None:
        os.chmod(target, 0o700 if os.path.isdir(target) else 0o600)
        function(target)

    try:
        shutil.rmtree(path, onerror=make_writable_and_retry)
    except OSError as exc:
        raise ReleaseRetentionError(
            f"release_retention_runtime_remove_failed:{kind}:{commit}"
        ) from exc
    if os.path.lexists(path):
        raise ReleaseRetentionError(
            f"release_retention_runtime_remove_failed:{kind}:{commit}"
        )


def _remove_publication_receipt(path: Path, *, commit: str, kind: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.name != f"{commit}.publication.v1.json"
    ):
        raise ReleaseRetentionError(
            f"release_retention_publication_receipt_target_invalid:{kind}:{commit}"
        )
    try:
        path.unlink()
    except OSError as exc:
        raise ReleaseRetentionError(
            f"release_retention_publication_receipt_remove_failed:{kind}:{commit}"
        ) from exc
    if os.path.lexists(path):
        raise ReleaseRetentionError(
            f"release_retention_publication_receipt_remove_failed:{kind}:{commit}"
        )


def _apply_release_retention_plan_locked(
    *,
    dry_run_plan_path: str | Path,
    acknowledgement: str,
    receipt_out: str | Path,
) -> dict[str, Any]:
    """Revalidate and apply exactly one previously written dry-run plan."""

    if acknowledgement != APPLY_ACKNOWLEDGEMENT:
        raise ReleaseRetentionError("release_retention_apply_acknowledgement_missing")
    plan_path = _absolute_path(dry_run_plan_path, field="dry_run_plan")
    output_path = _absolute_path(receipt_out, field="receipt_out")
    if plan_path.resolve() == output_path.resolve():
        raise ReleaseRetentionError("release_retention_apply_receipt_overlaps_plan")
    plan = _read_plan(plan_path)
    try:
        planned_at = datetime.fromisoformat(
            str(plan["planned_at"]).replace("Z", "+00:00")
        )
    except (KeyError, ValueError) as exc:
        raise ReleaseRetentionError("release_retention_dry_run_plan_invalid") from exc
    managed = dict(plan.get("managed_roots") or {})
    _assert_receipt_outside_managed_roots(plan_path, managed)
    _assert_receipt_outside_managed_roots(output_path, managed)
    runtime_root = Path(str(managed.get("splat_render_runtimes") or "")).parent
    current = build_release_retention_plan(
        release_root=str(managed.get("control_plane_releases") or ""),
        runtime_root=runtime_root,
        active_link=str(plan.get("active_link") or ""),
        current_deploy_commit=str(plan.get("current_deploy_commit") or ""),
        profile_dir=str(plan.get("profile_dir") or ""),
        public_catalog=str(plan.get("public_catalog") or ""),
        standing_authorization_dir=str(
            plan.get("standing_authorization_dir") or ""
        ),
        live_reference_roots=tuple(plan.get("live_reference_roots") or ()),
        evidence_binding_root=str(plan.get("evidence_binding_root") or ""),
        keep_commits=tuple(plan.get("operator_keep_commits") or ()),
        minimum_age_seconds=int(plan.get("minimum_age_seconds", -1)),
        now=planned_at,
    )
    if current.get("plan_digest") != plan.get("plan_digest"):
        raise ReleaseRetentionError("release_retention_plan_changed_since_dry_run")

    eligible = list(plan.get("eligible_commits") or [])
    # Re-stat every target before the first deletion. A partial apply can leave
    # harmless old bytes behind; it must never continue after discovering that
    # the reviewed plan no longer names the same inode and byte count.
    targets: list[tuple[dict[str, Any], Path]] = []
    for row in eligible:
        if not isinstance(row, Mapping):
            raise ReleaseRetentionError("release_retention_dry_run_plan_invalid")
        for artifact in row.get("artifacts") or []:
            if not isinstance(artifact, Mapping):
                raise ReleaseRetentionError("release_retention_dry_run_plan_invalid")
            targets.append((dict(artifact), _assert_snapshot_current(artifact)))

    removed: list[dict[str, Any]] = []
    for row in eligible:
        commit = _valid_commit(row.get("source_commit"))
        if commit is None:
            raise ReleaseRetentionError("release_retention_dry_run_plan_invalid")
        artifacts = sorted(
            (dict(item) for item in row.get("artifacts") or []),
            key=lambda item: (
                item.get("kind") != "control_plane_release",
                str(item.get("kind") or "").endswith("_publication_receipt"),
                item.get("kind"),
            ),
        )
        for artifact in artifacts:
            path = _assert_snapshot_current(artifact)
            kind = str(artifact["kind"])
            if kind == "control_plane_release":
                _remove_git_worktree(path, commit=commit)
            elif kind in {
                "runtime_splat_render",
                "runtime_scene_configuration",
            }:
                _remove_runtime_tree(path, commit=commit, kind=kind)
            elif kind in {
                "runtime_splat_render_publication_receipt",
                "runtime_scene_configuration_publication_receipt",
            }:
                _remove_publication_receipt(path, commit=commit, kind=kind)
            else:
                raise ReleaseRetentionError(
                    f"release_retention_artifact_kind_invalid:{kind}"
                )
            removed.append(
                {
                    "source_commit": commit,
                    "kind": kind,
                    "path": str(path),
                    "removed_bytes": artifact["size_bytes"],
                }
            )

    result: dict[str, Any] = {
        "schema_version": APPLY_SCHEMA_VERSION,
        "status": "applied",
        "dry_run_plan_path": str(plan_path.resolve()),
        "dry_run_plan_digest": plan["plan_digest"],
        "removed": removed,
        "removed_bytes": sum(item["removed_bytes"] for item in removed),
        "predicted_removed_bytes": plan["predicted_removed_bytes"],
        "active_source_commit": plan["active_source_commit"],
        "current_deploy_commit": plan["current_deploy_commit"],
        "provider_mutation_performed": False,
        "production_artifact_mutation_performed": bool(removed),
        "symlinks_followed": False,
    }
    result["receipt_digest"] = _canonical_digest(
        result, digest_field="receipt_digest"
    )
    _write_exclusive(output_path, result)
    return result


def apply_release_retention_plan(
    *,
    dry_run_plan_path: str | Path,
    acknowledgement: str,
    receipt_out: str | Path,
) -> dict[str, Any]:
    """Hold the global reference lock across final revalidation and deletion."""

    plan_path = _absolute_path(dry_run_plan_path, field="dry_run_plan")
    plan = _read_plan(plan_path)
    public_catalog = _absolute_path(
        str(plan.get("public_catalog") or ""), field="public_catalog"
    )
    with release_reference_lock(public_catalog.parent, exclusive=True):
        return _apply_release_retention_plan_locked(
            dry_run_plan_path=dry_run_plan_path,
            acknowledgement=acknowledgement,
            receipt_out=receipt_out,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", default=str(DEFAULT_RELEASE_ROOT))
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--active-link", default=str(DEFAULT_ACTIVE_LINK))
    parser.add_argument("--current-deploy-commit", required=False)
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR))
    parser.add_argument("--public-catalog", default=str(DEFAULT_PUBLIC_CATALOG))
    parser.add_argument(
        "--standing-authorization-dir",
        default=str(DEFAULT_STANDING_AUTHORIZATION_DIR),
    )
    parser.add_argument("--live-reference-root", action="append")
    parser.add_argument(
        "--evidence-binding-root", default=str(DEFAULT_EVIDENCE_BINDING_ROOT)
    )
    parser.add_argument("--keep-commit", action="append", default=[])
    parser.add_argument(
        "--minimum-age-seconds", type=int, default=DEFAULT_MINIMUM_AGE_SECONDS
    )
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run-plan")
    parser.add_argument("--ack")
    args = parser.parse_args(argv)
    try:
        if args.apply:
            if args.current_deploy_commit:
                raise ReleaseRetentionError(
                    "release_retention_apply_parameters_must_come_from_plan"
                )
            if not args.dry_run_plan:
                raise ReleaseRetentionError(
                    "release_retention_apply_dry_run_plan_missing"
                )
            result = apply_release_retention_plan(
                dry_run_plan_path=args.dry_run_plan,
                acknowledgement=str(args.ack or ""),
                receipt_out=args.receipt_out,
            )
        else:
            if args.dry_run_plan or args.ack:
                raise ReleaseRetentionError(
                    "release_retention_dry_run_apply_parameter_conflict"
                )
            if not args.current_deploy_commit:
                raise ReleaseRetentionError(
                    "release_retention_deploy_candidate_missing"
                )
            result = build_release_retention_plan(
                release_root=args.release_root,
                runtime_root=args.runtime_root,
                active_link=args.active_link,
                current_deploy_commit=args.current_deploy_commit,
                profile_dir=args.profile_dir,
                public_catalog=args.public_catalog,
                standing_authorization_dir=args.standing_authorization_dir,
                live_reference_roots=tuple(
                    args.live_reference_root or DEFAULT_LIVE_REFERENCE_ROOTS
                ),
                evidence_binding_root=args.evidence_binding_root,
                keep_commits=tuple(args.keep_commit),
                minimum_age_seconds=args.minimum_age_seconds,
            )
            output_path = _absolute_path(args.receipt_out, field="receipt_out")
            _assert_receipt_outside_managed_roots(
                output_path, dict(result["managed_roots"])
            )
            _write_exclusive(output_path, result)
    except (OSError, ReleaseRetentionError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                    "production_artifact_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "APPLY_ACKNOWLEDGEMENT",
    "APPLY_SCHEMA_VERSION",
    "DEFAULT_LIVE_REFERENCE_ROOTS",
    "EVIDENCE_BINDING_SCHEMA_VERSION",
    "ReleaseRetentionError",
    "SCHEMA_VERSION",
    "apply_release_retention_plan",
    "build_release_retention_plan",
    "main",
]
