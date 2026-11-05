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
- a receipt or trace bound to another probe request (another scene revision or
  task specification: a request since superseded) is never what is read for
  the current one; whoever retains the current request's evidence first moves
  it, byte for byte, into ``robot_free_probe_history/`` beside the launch,
  under its content digest and beside a marker naming what superseded it
  (``set_aside``). Every replaced receipt is kept there the same way, so
  nothing retained here is overwritten or deleted;
- a deterministic refusal (the revision's task cannot be adapted or scored)
  is reported and retains nothing, so the gate keeps refusing by name;
- a transient failure (fetch, storage) raises, so the caller retries.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .common import utc_now_iso
from .decision_evidence_contracts import canonical_digest
from .robot_free_probe_gate import RECEIPT_FILENAME

TRACE_FILENAME = "rigid_robot_free_probe_trace.v1.json"
WORK_DIRECTORY = "robot-free-readiness"
HISTORY_DIRNAME = "robot_free_probe_history"
SUPERSESSION_SCHEMA = "blueprint.robot_free_probe_evidence_supersession.v1"
DECIDED_STATUSES = frozenset({"passed", "failed"})
RIGID_STRATEGIES = frozenset({"pick_and_place", "planar_push"})
# What a receipt binds of the probe request it scored (``team_robot_free_probe.probe_request``'s binding).
RECEIPT_BINDING_FIELDS = ("configured_scene_revision_digest", "task_spec_digest", "replacement_asset_digest")
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


def trace_bound_to_request(trace: Any, request: Any) -> bool:
    """The trace recorded this probe request: its binding and plan are the request's own.

    Both are sealed into the request's digest (with the task specification the binding's
    ``task_spec_digest`` names), so a trace of any other request differs in one of them.
    """

    return (isinstance(trace, Mapping) and isinstance(request, Mapping) and isinstance(request.get("binding"), Mapping)
            and trace.get("binding") == request["binding"] and trace.get("plan") == request.get("plan"))


def receipt_bound_to_request(receipt: Any, request: Any) -> bool:
    """The receipt scored this probe request's scene revision and task specification."""

    binding = request.get("binding") if isinstance(request, Mapping) else None
    return (isinstance(receipt, Mapping) and isinstance(binding, Mapping)
            and all(receipt.get(field) == binding.get(field) for field in RECEIPT_BINDING_FIELDS))


def _same_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(field) == right.get(field) for field in RECEIPT_BINDING_FIELDS)


def set_aside(root: str | Path, name: str, *, superseded_by: Mapping[str, Any]) -> dict[str, Any] | None:
    """Move ``root/name`` into the launch's append-only history; return the marker, or None when absent.

    The file keeps its bytes under ``robot_free_probe_history/<sha256>.<name>``, beside a marker
    naming what it held and what superseded it. The marker is written first, so a pass interrupted
    before the move leaves a marker that only describes; the file is still where it was. History is
    content-addressed, so a second set-aside of the same bytes keeps one copy and loses nothing.
    """

    path = _safe(Path(root) / name)
    if not path.is_file():
        return None
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    history = _safe(Path(root) / HISTORY_DIRNAME)
    history.mkdir(exist_ok=True, mode=0o750)
    kept = _safe(history / f"{digest}.{name}")
    marker_path = _safe(history / f"{digest}.{name}.supersession.json")
    try:
        value = json.loads(data)
    except ValueError:
        value = None
    value = value if isinstance(value, dict) else {}
    marker = {"schema_version": SUPERSESSION_SCHEMA, "file": name, "history_file": kept.name,
              "sha256": "sha256:" + digest, "superseded_status": value.get("status"),
              "superseded_digest": value.get("trace_digest") or value.get("receipt_digest"),
              "superseded_binding": (value.get("binding") if isinstance(value.get("binding"), Mapping)
                                     else {field: value.get(field) for field in RECEIPT_BINDING_FIELDS}),
              "superseded_by": dict(superseded_by), "superseded_at_iso": utc_now_iso(), "marker_digest": ""}
    marker["marker_digest"] = canonical_digest(marker, digest_field="marker_digest")
    if not marker_path.exists():
        with tempfile.NamedTemporaryFile("w", dir=history, prefix=".marker-", delete=False,
                                         encoding="utf-8") as stream:
            stream.write(json.dumps(marker, sort_keys=True, indent=1) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.chmod(stream.name, 0o440)
            os.link(stream.name, marker_path)
        except FileExistsError:
            pass
        finally:
            os.unlink(stream.name)
    if kept.exists() and kept.read_bytes() != data:
        raise RobotFreeReadinessRetentionError("robot_free_probe_history_conflict")
    os.replace(path, kept)  # same bytes as any copy already there: nothing is lost
    return _read_json(marker_path)


def _write(path: Path, receipt: Mapping[str, Any], *, supersede_other_request: bool = False) -> str:
    existing = _read_json(path)
    if existing == receipt:
        return "unchanged"
    other = existing is not None and not _same_binding(existing, receipt)
    if existing is not None and existing.get("status") in DECIDED_STATUSES and not (other and supersede_other_request):
        # Sealed evidence is never replaced, even by a newer scorer. Only a writer that holds the
        # current request's own receipt sets aside a decided one bound to another request.
        return "kept_existing_decided"
    if path.is_file():
        # Whatever is replaced is kept, byte for byte, in the launch's history.
        set_aside(path.parent, path.name, superseded_by={"receipt_digest": receipt.get("receipt_digest"),
                                                         "status": receipt.get("status"),
                                                         **{field: receipt.get(field)
                                                            for field in RECEIPT_BINDING_FIELDS}})
    text = json.dumps(receipt, sort_keys=True, indent=1) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".robot-free-", delete=False,
                                     encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(stream.name, 0o440)
    os.replace(stream.name, path)
    if existing is None:
        return "written"
    return "superseded_other_request" if other else "replaced_unavailable"


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


def retain_receipt(run_root: str | Path, receipt: Mapping[str, Any], *, supersede_other_request: bool = False) -> str:
    """Retain a receipt scored elsewhere (a team evaluation's recorded probe) where the gate reads it.

    ``supersede_other_request`` is for a caller that scored the capture's current probe request: a
    decided receipt bound to another request (superseded) is then set aside into the history instead
    of shadowing it. A decided receipt bound to the same request is never replaced.
    """

    return _write(_safe(Path(run_root) / RECEIPT_FILENAME), receipt, supersede_other_request=supersede_other_request)


def retain_for_source_launch(run_root: str | Path, *, fetcher: Fetcher | None = None) -> dict[str, Any]:
    """The same retention for a completed source launch, read from its validated terminal evidence."""

    from .task_evaluation_configured_controls_progression_worker import _validate_source

    result, _receipt, _zero = _validate_source(Path(run_root))
    return retain_robot_free_readiness(
        run_root=run_root, revision_reference=result["configured_scene_revision_reference"],
        revision_digest=result["configured_scene_revision_digest"], fetcher=fetcher)


__all__ = [
    "DECIDED_STATUSES",
    "HISTORY_DIRNAME",
    "RECEIPT_BINDING_FIELDS",
    "RobotFreeReadinessRetentionError",
    "SUPERSESSION_SCHEMA",
    "TRACE_FILENAME",
    "receipt_bound_to_request",
    "retain_for_source_launch",
    "retain_receipt",
    "retain_robot_free_readiness",
    "set_aside",
    "trace_bound_to_request",
]
