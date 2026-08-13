"""Digest-bound artifact manifests for production Task Evaluation attempts.

The provider worker may describe artifacts in its own terminal receipt, but the
canonical allocator is responsible for independently inventorying the bytes it
actually retained.  This module deliberately accepts only paths beneath one
attempt root so a profile cannot turn manifest generation into an arbitrary
filesystem reader.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_artifact_manifest.v1"


class TaskEvaluationArtifactManifestError(ValueError):
    """Raised when allocator-retained evidence cannot be safely inventoried."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _under_attempt_root(path: Path, attempt_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != attempt_root and attempt_root not in resolved.parents:
        raise TaskEvaluationArtifactManifestError(
            "task_evaluation_artifact_path_outside_attempt_root"
        )
    return resolved


def _role_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return []


def build_task_evaluation_artifact_manifest(
    *,
    attempt_root: str | Path,
    artifact_roots: Mapping[str, str | Path],
    required_roles: Sequence[str],
    binding: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inventory allocator-retained evidence and write one immutable manifest."""

    root = Path(attempt_root).expanduser().resolve()
    if not root.is_dir():
        raise TaskEvaluationArtifactManifestError(
            "task_evaluation_artifact_attempt_root_missing"
        )
    destination = _under_attempt_root(
        Path(output_path) if output_path is not None else root / "artifact_manifest.json",
        root,
    )
    normalized_roles = {
        str(role): _under_attempt_root(Path(path), root)
        for role, path in artifact_roots.items()
        if str(role).strip()
    }
    required = sorted(set(str(role) for role in required_roles if str(role).strip()))
    missing_roles = sorted(
        role for role in required if not _role_files(normalized_roles.get(role, root / ".missing"))
    )

    roles_by_path: dict[Path, set[str]] = defaultdict(set)
    for role, path in sorted(normalized_roles.items()):
        for artifact in _role_files(path):
            resolved = _under_attempt_root(artifact, root)
            if resolved != destination:
                roles_by_path[resolved].add(role)

    files = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "roles": sorted(roles),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path, roles in sorted(roles_by_path.items(), key=lambda item: item[0].as_posix())
    ]
    blockers = [f"task_evaluation_artifact_role_missing:{role}" for role in missing_roles]
    if not files:
        blockers.append("task_evaluation_artifact_manifest_empty")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not blockers else "blocked",
        "binding": dict(binding),
        "required_roles": required,
        "observed_roles": sorted(
            role for role, path in normalized_roles.items() if _role_files(path)
        ),
        "file_count": len(files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "files": files,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    payload["manifest_digest"] = canonical_digest(
        payload, digest_field="manifest_digest"
    )
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskEvaluationArtifactManifestError(
                "task_evaluation_artifact_manifest_existing_invalid"
            ) from exc
        if existing != payload:
            raise TaskEvaluationArtifactManifestError(
                "task_evaluation_artifact_manifest_immutable_conflict"
            )
        return payload
    write_json(destination, payload)
    return payload


#: Every paid lane lays its attempt out the same way. Naming the convention
#: once is what lets a lane added tomorrow inherit the terminal contract
#: instead of rediscovering it.
PROVIDER_RUN_DIRNAME = "vast_provider_run"
PROVIDER_EVIDENCE_DIRNAME = "immutable_execution"
TEARDOWN_MANIFEST_NAME = "vast_teardown_manifest.json"
ADAPTER_RESULT_NAME = "vast_provider_adapter_result.json"


def _without_absent_teardown(
    result: Mapping[str, Any], teardown: Path
) -> dict[str, Any]:
    """Null a teardown path that names a file nobody wrote."""

    sealed = dict(result)
    named = str(sealed.get("teardown_manifest_path") or "").strip()
    if named and not Path(named).is_file():
        sealed["teardown_manifest_path"] = None
    elif "teardown_manifest_path" not in sealed and teardown.is_file():
        # Only fill a field the lane left out when there is real evidence to
        # name. Introducing a null into a result that never mentioned teardown
        # would change a dry run's shape for nothing.
        sealed["teardown_manifest_path"] = str(teardown)
    elif not sealed.get("teardown_manifest_path") and teardown.is_file():
        sealed["teardown_manifest_path"] = str(teardown)
    return sealed


#: The teardown status a lane writes when it never obtained a provider
#: resource. The provider adapter already uses this prefix for its own early
#: exits (`not_required_blueprint_bundle_preflight_blocked` and siblings), so
#: this stays one vocabulary rather than two.
UNALLOCATED_TEARDOWN_STATUS = "not_required_provider_adapter_never_invoked"


def seal_unallocated_provider_teardown(
    provider_run: str | Path,
    *,
    reason: str,
    generated_at: str | None = None,
) -> Path | None:
    """Record that nothing was allocated, when the lane failed before allocating.

    A lane that raises before it enters its provider adapter -- resolving a
    credential, reading a staged URL -- writes no teardown manifest at all,
    because from its own point of view no teardown happened. The launch then
    sits at `provider_zero_pending` forever waiting on a manifest that will
    never be written for a resource that was never obtained, and the reconciler
    fails on every sweep after that. A provider-zero signal that is permanently
    red is not a strict one; it is one nobody can read, which is worse than
    useless on the day something really does leak.

    This refuses to be the thing that lies. It writes only when the run's own
    evidence shows no resource was obtained:

    * an existing teardown manifest is left exactly as it is -- if the adapter
      wrote one, that one is the truth;
    * an adapter result naming any instance id means an instance existed, and a
      real teardown is the only acceptable closure. The launch stays pending,
      which is the correct outcome.

    Returns the path written, or None when it declined -- never raises, because
    a lane is already failing when it calls this.
    """

    run = Path(provider_run).expanduser()
    teardown = run / TEARDOWN_MANIFEST_NAME
    try:
        if teardown.exists():
            return None
        adapter = run / ADAPTER_RESULT_NAME
        if adapter.is_file():
            recorded = json.loads(adapter.read_text(encoding="utf-8"))
            if not isinstance(recorded, Mapping):
                return None
            for field in ("vast_instance_ids", "instance_ids"):
                if recorded.get(field):
                    return None
        run.mkdir(parents=True, exist_ok=True)
        write_json(
            teardown,
            {
                "schema_version": "vast_teardown_manifest.v1",
                "generated_at": generated_at or utc_now_iso(),
                "status": UNALLOCATED_TEARDOWN_STATUS,
                "vast_instance_ids": [],
                "teardown_actions_performed": [],
                "continuing_spend_from_this_run": False,
                "zero_continuing_spend_scope": (
                    "The lane failed before entering the provider adapter, so no "
                    "provider resource was ever requested: " + str(reason)
                ),
            },
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return teardown


def seal_lane_terminal_artifacts(
    result: Mapping[str, Any],
    *,
    attempt_root: str | Path,
    lane: str,
    extra_artifact_roots: Mapping[str, str | Path] | None = None,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Give any paid lane the two terminal artifacts its launch profile requires.

    Every production launch profile's terminal contract asks the allocator
    result for ``artifact_manifest_path`` and ``teardown_manifest_path``. Eight
    paid lanes emitted neither or only the second, so each of them ended
    ``allocator_terminal_artifact_missing:`` no matter what happened on the
    provider -- "all controls passed" was unreachable by construction, and the
    evidence they retained existed on disk with nothing pointing at it.

    Fixing that lane by lane leaves the next lane to rediscover it. This seals
    the contract from the shared convention every lane already follows, so a
    lane added tomorrow inherits it by calling one function, and a contract test
    fails the ones that do not.

    A blocked manifest is carried into the result's blockers rather than
    hidden: an attempt that retained no evidence must not report a terminal
    status that implies it did.
    """

    root = Path(attempt_root).expanduser().resolve()
    provider_run = root / PROVIDER_RUN_DIRNAME
    evidence = root / PROVIDER_EVIDENCE_DIRNAME
    teardown = provider_run / TEARDOWN_MANIFEST_NAME
    if not provider_run.is_dir() and not evidence.is_dir():
        # No attempt reached a provider, so there is nothing to inventory. A
        # dry run has no terminal artifacts to seal and must not be made to
        # look like a live one that lost them.
        #
        # A teardown path is still corrected. Lanes name theirs unconditionally
        # as `<provider_run>/vast_teardown_manifest.json`, so a run that failed
        # before the provider run directory existed reports a path to a file
        # that was never written -- which reads as teardown evidence that has
        # gone missing rather than teardown that never happened.
        return _without_absent_teardown(result, teardown)
    roots: dict[str, str | Path] = {}
    if evidence.is_dir():
        roots["provider_runtime_evidence"] = evidence
    adapter_result = provider_run / ADAPTER_RESULT_NAME
    if adapter_result.is_file():
        roots["allocator_adapter_result"] = adapter_result
    if teardown.is_file():
        roots["teardown_manifest"] = teardown
    if provider_run.is_dir():
        # Retained, never required: the logs and guards under here are what a
        # blocked attempt is diagnosed from, and are exactly what is absent
        # when it failed early.
        roots["provider_run_diagnostics"] = provider_run
    for name, path in (extra_artifact_roots or {}).items():
        roots[str(name)] = path

    sealed = dict(result)
    # Only roles this attempt could actually have produced are required, so a
    # run blocked before allocation is not asked to inventory a provider it
    # never reached -- its absence is already its own blocker.
    required = sorted(
        role
        for role in ("provider_runtime_evidence", "allocator_adapter_result", "teardown_manifest")
        if role in roots
    )
    destination = root / "artifact_manifest.json"
    try:
        manifest = build_task_evaluation_artifact_manifest(
            attempt_root=root,
            artifact_roots=roots,
            required_roles=required,
            binding={
                "allocator_lane": lane,
                "retry_cap": 0,
                **dict(binding or {}),
            },
            output_path=destination,
        )
    except TaskEvaluationArtifactManifestError as exc:
        blockers = [*(sealed.get("blockers") or []), f"terminal_artifact_manifest_invalid:{exc}"]
        sealed["blockers"] = sorted(set(str(item) for item in blockers))
        sealed["status"] = "blocked"
        return sealed

    if manifest.get("status") != "completed":
        # Only touched when the manifest itself is blocked, so a lane's own
        # blocker list and its ordering are left exactly as it reported them.
        blockers = [*(sealed.get("blockers") or []), *(manifest.get("blockers") or [])]
        sealed["blockers"] = sorted(set(str(item) for item in blockers))
        sealed["status"] = "blocked"
    sealed["artifact_manifest_path"] = str(destination) if destination.is_file() else None
    return _without_absent_teardown(sealed, teardown)


__all__ = [
    "ADAPTER_RESULT_NAME",
    "PROVIDER_EVIDENCE_DIRNAME",
    "PROVIDER_RUN_DIRNAME",
    "SCHEMA_VERSION",
    "TEARDOWN_MANIFEST_NAME",
    "TaskEvaluationArtifactManifestError",
    "build_task_evaluation_artifact_manifest",
    "seal_lane_terminal_artifacts",
]
