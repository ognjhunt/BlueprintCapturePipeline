"""Retain a scene's robot-free readiness receipt beside its source launch, automatically.

Beta journey step: L5 team policy evaluation. Every team evaluation is refused
unless ``robot_free_readiness.v1.json`` beside the source launch shows the
scene's robot-free probe passed at the launch's configured revision
(``robot_free_probe_gate``). Until now only an operator CLI wrote that file, so
a scene whose probe had passed still refused every team until someone copied
the receipt by hand.

Scene preparation now retains it as part of completing (and anything that
later adds probe evidence, such as a rigid probe's recorded trace, calls the
same function again). The published revision is fetched by its reference,
verified byte for byte and against the launch's revision digest, scored by the
articulated or rigid readiness scorer, and written beside the launch:

- a sealed decided receipt (``passed`` or ``failed``) is never replaced;
- a ``probe_unavailable`` receipt is replaced once evidence makes it decided;
- a deterministic refusal (the revision's task cannot be adapted or scored)
  is reported and retains nothing, so the gate keeps refusing by name;
- a transient failure (fetch, storage) raises, so the caller retries.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .robot_free_probe_gate import RECEIPT_FILENAME

TRACE_FILENAME = "rigid_robot_free_probe_trace.v1.json"
WORK_DIRECTORY = "robot-free-readiness"
DECIDED_STATUSES = frozenset({"passed", "failed"})
RIGID_STRATEGIES = frozenset({"pick_and_place", "planar_push"})
Fetcher = Callable[[Mapping[str, Any]], bytes]


class RobotFreeReadinessRetentionError(RuntimeError):
    """The receipt could not be retained now; retrying may succeed."""


def _safe(path: Path) -> Path:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise RobotFreeReadinessRetentionError("robot_free_readiness_path_unsafe")
    return path


def _read_json(path: Path) -> dict[str, Any] | None:
    if not _safe(path).is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RobotFreeReadinessRetentionError("robot_free_readiness_retained_file_unreadable") from exc
    return value if isinstance(value, dict) else None


def _write(path: Path, receipt: Mapping[str, Any]) -> str:
    existing = _read_json(path)
    if existing == receipt:
        return "unchanged"
    if existing is not None and existing.get("status") in DECIDED_STATUSES:
        # Sealed evidence is never replaced, even by a newer scorer.
        return "kept_existing_decided"
    text = json.dumps(receipt, sort_keys=True, indent=1) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".robot-free-", delete=False,
                                     encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(stream.name, 0o440)
    os.replace(stream.name, path)
    return "replaced_unavailable" if existing is not None else "written"


def _revision(reference: Mapping[str, Any], *, revision_digest: str, fetcher: Fetcher | None) -> dict[str, Any]:
    from . import task_evaluation_configured_controls_deferred_inputs as inputs
    from .task_evaluation_configured_scene_revision import validate_configured_scene_revision

    payload = inputs._fetched(reference, fetcher=fetcher or inputs.default_reference_fetcher)
    revision = validate_configured_scene_revision(json.loads(payload))
    if revision.get("revision_digest") != revision_digest:
        raise ValueError("robot_free_readiness_revision_digest_mismatch")
    return revision


def retain_robot_free_readiness(*, run_root: str | Path, revision_reference: Mapping[str, Any],
                                revision_digest: str, fetcher: Fetcher | None = None) -> dict[str, Any]:
    """Score the launch's published revision and retain the receipt where the gate reads it."""

    from . import task_evaluation_scene_rigid_robot_free_readiness as rigid
    from .task_evaluation_rigid_relocation_native_adapter import DEFINITION_CONTRACT_PATH
    from .task_evaluation_scene_robot_free_readiness import robot_free_readiness_for_revision

    root = _safe(Path(run_root))
    work = root / WORK_DIRECTORY
    try:
        revision = _revision(revision_reference, revision_digest=revision_digest, fetcher=fetcher)
        documents = rigid.fetch_revision_documents(revision=revision, output_root=work, fetcher=fetcher)
        strategy = json.loads(documents[DEFINITION_CONTRACT_PATH].read_text(encoding="utf-8")).get("strategy")
        if strategy == "articulated_open_close":
            receipt = robot_free_readiness_for_revision(revision=revision, output_root=work, fetcher=fetcher)
        elif strategy in RIGID_STRATEGIES:
            receipt = rigid.rigid_robot_free_readiness_for_revision(
                revision=revision, output_root=work, fetcher=fetcher,
                physics_probe=_read_json(root / TRACE_FILENAME))
        else:
            return {"status": "unavailable_for_task_kind", "reason": str(strategy), "retained": False}
    except ValueError as exc:
        # The revision itself cannot be adapted or scored: retrying cannot help,
        # and the gate keeps refusing by name because no receipt exists.
        return {"status": "refused", "reason": str(exc)[:200], "retained": False}
    except (OSError, RuntimeError) as exc:
        raise RobotFreeReadinessRetentionError(
            "robot_free_readiness_retention_failed:" + f"{type(exc).__name__}:{str(exc)[:160]}") from exc
    outcome = _write(_safe(root / RECEIPT_FILENAME), receipt)
    retained = _read_json(root / RECEIPT_FILENAME) or {}
    return {"status": retained.get("status"), "receipt_digest": retained.get("receipt_digest"),
            "retained": True, "write": outcome}


def retain_for_source_launch(run_root: str | Path, *, fetcher: Fetcher | None = None) -> dict[str, Any]:
    """The same retention for a completed source launch, read from its validated terminal evidence."""

    from .task_evaluation_configured_controls_progression_worker import _validate_source

    result, _receipt, _zero = _validate_source(Path(run_root))
    return retain_robot_free_readiness(
        run_root=run_root, revision_reference=result["configured_scene_revision_reference"],
        revision_digest=result["configured_scene_revision_digest"], fetcher=fetcher)


__all__ = [
    "DECIDED_STATUSES",
    "RobotFreeReadinessRetentionError",
    "TRACE_FILENAME",
    "retain_for_source_launch",
    "retain_robot_free_readiness",
]
