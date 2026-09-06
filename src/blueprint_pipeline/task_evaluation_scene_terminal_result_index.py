"""Owner terminal-result index: files REAL producer outputs for the owner reconciler.

The Spec E terminal reconciler
(``task_evaluation_scene_terminal_reconciler.reconcile_terminal_owner_result``)
joins the sealed receipts under ``terminal_result_root/<intent-id>/`` into the
persistent owner status. Those receipts are produced by DIFFERENT real producers
at different locations: the launch dispatcher retains the launch request/profile
bridge in the launch run root; the policy-canary dispatcher retains the result
projection, the authenticated Website sync, its Vast post-teardown provider-zero
receipt and the sealed ``dispatch_receipt.json`` that binds all three by file
digest in the canary run root; the GC retention step archives the whole canary
root to the artifact store and leaves a sealed evidence-offload pointer. Nothing
gathered them (audit findings A6, R8-R10), so the reconciler was inert.

This module is that index -- three idempotent stages, each validating the whole
cross-bound set in memory BEFORE staging, fsyncing and atomically publishing
byte-exact copies (an existing file must be byte-identical):

* :func:`index_launch_bridge` -- stage A, per launch run (owner-bound through
  ``internal_policy_canary_execution_plan.scene_policy_binding``; resolves the
  owner intent directory by ``intent_digest``);
* :func:`index_policy_canary_terminal` -- stage B, per canary run root carrying a
  sealed ``dispatch_receipt.json`` (joined to the bridge by ``run_id``);
* :func:`index_result_publication` -- stage C, derives the durable result
  publication from the offload pointer of THIS run's canary root, bound through
  the archive members' digests of the persisted projection and the sealed receipt
  -- never from a caller-supplied URI.

Stages A and B are driven by the launch reconciler tick (retention); stage C by
the terminal reconciler when a completed-unqualified result has no publication
yet. The index never launches, retries, tears down, allocates a provider,
rewrites a producer receipt or manufactures a missing one.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import task_evaluation_scene_intake as intake
from . import task_evaluation_scene_policy_binding as scene_policy
from .control_plane_evidence_offload import POINTER_SCHEMA_VERSION, POINTER_SUFFIX
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_policy_canary_result import (
    TaskEvaluationPolicyCanaryResultError,
    validate_policy_canary_result,
)
from .task_evaluation_scene_terminal_reconciler import (
    DISPATCH_RECEIPT_FILENAME,
    DISPATCH_RECEIPT_SCHEMA,
    POLICY_CANARY_RUN_KIND,
    PUBLICATION_SCHEMA,
    WEBAPP_SYNC_SCHEMA,
    confirmed_provider_zero,
)

INDEX_SCHEMA = "task_evaluation_scene_terminal_result_index.v1"
STATE_SCHEMA = "task_evaluation_scene_terminal_index_state.v1"
STATE_FILENAME = "terminal_index_state.json"
PUBLICATION_FILENAME = "terminal_result_publication.json"
LAUNCH_REQUEST_SCHEMA = "task_evaluation_launch_request.v1"
LAUNCH_PROFILE_SCHEMA = "task_evaluation_launch_profile.v1"
#: Where the dispatcher persists the two in-memory terminal records (R8).
PERSISTED_PROJECTION_RELATIVE_PATH = "artifacts/result_delivery/policy_canary_result_projection.json"
PERSISTED_WEBAPP_SYNC_RELATIVE_PATH = "artifacts/result_delivery/policy_canary_webapp_sync.json"
CANARY_PROVIDER_ZERO_FILENAME = "post_teardown_global_provider_zero.json"
CANARY_DISPATCH_RECEIPT_FILENAME = "dispatch_receipt.json"
_RESULT_URI_PREFIX = "s3://"


class TerminalResultIndexError(ValueError):
    """A producer receipt is absent, off-contract, unbound, or conflicts with an indexed file."""


def _fail(code: str) -> None:
    raise TerminalResultIndexError("terminal_result_index_" + code)


def _safe(path: str | Path) -> Path:
    item = Path(path)
    if not item.is_absolute() or any(p.is_symlink() for p in (item, *item.parents)):
        _fail("path_unsafe")
    return item


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and intake._DIGEST.fullmatch(value) is not None


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def _sealed(value: dict[str, Any], field: str) -> bool:
    return _is_digest(value.get(field)) and value[field] == canonical_digest(value, digest_field=field)


def _read_bytes(path: Path, *, reason: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        _fail(reason)
    try:
        return path.read_bytes()
    except OSError:
        _fail(reason)


def _parse(raw: bytes, *, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        _fail(reason)
    if not isinstance(value, dict):
        _fail(reason)
    return value


def _record(record: Any) -> bool:
    return (isinstance(record, dict) and _is_digest(record.get("sha256"))
            and type(record.get("size_bytes")) is int and record["size_bytes"] > 0)


def _bound(record: dict[str, Any], raw: bytes) -> bool:
    return record["sha256"] == _sha256(raw) and record["size_bytes"] == len(raw)


def file_record(path: str | Path) -> dict[str, Any]:
    """``{path, sha256, size_bytes}`` -- the dispatcher's receipt record shape."""
    raw = Path(path).read_bytes()
    return {"path": str(path), "sha256": _sha256(raw), "size_bytes": len(raw)}


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _publish(directory: Path, files: dict[str, bytes]) -> dict[str, str]:
    """Publish byte-exact files after the whole set validated (R10).

    Every existing target is checked for byte identity BEFORE anything is
    written; each new file is staged beside its target, fsynced and atomically
    renamed, so a reader never sees a partial file and a re-run is byte-identical.
    """
    for name, raw in files.items():
        target = directory / name
        if target.is_symlink() or (target.exists() and target.read_bytes() != raw):
            _fail("immutable_conflict")
    directory.mkdir(parents=True, exist_ok=True, mode=0o750)
    written: dict[str, str] = {}
    for name, raw in files.items():
        target = directory / name
        if not target.exists():
            temporary = directory / f".{name}.{os.getpid()}.tmp"
            with temporary.open("wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o440)
            os.replace(temporary, target)
        written[name] = str(target)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return written


def _base(stage: str, **fields: Any) -> dict[str, Any]:
    return {"schema_version": INDEX_SCHEMA, "stage": stage, "provider_mutation_performed": False, **fields}


# ------------------------------------------------------------------ stage A: launch bridge


def _owner_binding_of(profile: dict[str, Any]) -> dict[str, Any] | None:
    plan = profile.get("internal_policy_canary_execution_plan")
    value = plan.get("scene_policy_binding") if isinstance(plan, dict) else None
    if value is None:
        return None
    try:
        return scene_policy.validate_binding(value)
    except scene_policy.ScenePolicyBindingError:
        _fail("launch_profile_binding_invalid")


def _owner_intent_directory(scene_intent_root: str | Path, digest: str) -> Path | None:
    """The owner intent directory whose sealed ``intent.json`` carries ``digest``.

    ``intent_id`` is the owner+submission identity, not a function of the intent
    digest the binding carries, so the store is scanned (one sealed record per
    owner submission). ``None`` when no intent in this store claims the run.
    """
    root = _safe(scene_intent_root)
    if not root.is_dir():
        return None
    for path in sorted(root.glob("scene-*/intent.json")):
        if path.is_symlink() or path.parent.is_symlink():
            _fail("owner_store_unsafe")
        try:
            intent = intake._read(path, "intent_digest")
        except intake.SceneIntakeError:
            _fail("owner_intent_invalid")
        if intent.get("intent_digest") != digest:
            continue
        if intent.get("intent_id") != path.parent.name:
            _fail("owner_intent_invalid")
        return path.parent
    return None


def index_launch_bridge(*, launch_run_root: str | Path, scene_intent_root: str | Path,
                        terminal_result_root: str | Path) -> dict[str, Any]:
    """Stage A: file the launch request/profile bridge of an owner-bound canary launch.

    Reads the launch dispatcher's retained ``launch_request.json`` and sealed
    ``launch_profile.json``; a profile without an owner ``scene_policy_binding``
    is ``not_owner_bound`` (nothing written); a binding whose intent is not in the
    store is ``owner_intent_unresolved`` (nothing written). Off-contract files
    fail closed.
    """
    run_root = _safe(launch_run_root)
    request_raw = _read_bytes(run_root / "launch_request.json", reason="launch_request_absent")
    profile_raw = _read_bytes(run_root / "launch_profile.json", reason="launch_profile_absent")
    request = _parse(request_raw, reason="launch_request_invalid")
    profile = _parse(profile_raw, reason="launch_profile_invalid")
    if (profile.get("schema_version") != LAUNCH_PROFILE_SCHEMA or not _sealed(profile, "profile_digest")
            or not _is_commit(profile.get("source_commit"))):
        _fail("launch_profile_invalid")
    run_id = request.get("run_id")
    if (request.get("schema_version") != LAUNCH_REQUEST_SCHEMA
            or request.get("launch_profile_digest") != profile["profile_digest"]
            or request.get("source_commit") != profile["source_commit"]
            or not isinstance(run_id, str) or not run_id):
        _fail("launch_request_invalid")
    base = _base("launch_bridge", launch_run_root=str(run_root), run_id=run_id)
    binding = _owner_binding_of(profile)
    if binding is None:
        return {**base, "status": "not_owner_bound"}
    intent_directory = _owner_intent_directory(scene_intent_root, binding["scene_intent_digest"])
    if intent_directory is None:
        return {**base, "status": "owner_intent_unresolved", "scene_intent_digest": binding["scene_intent_digest"]}
    directory = _safe(terminal_result_root) / intent_directory.name
    written = _publish(directory, {"launch_request.json": request_raw, "launch_profile.json": profile_raw})
    return {**base, "status": "launch_bridge_indexed", "intent_id": intent_directory.name,
            "directory": str(directory), "files": written}


# ------------------------------------------------------------- stage B: canary terminal set


def _bridge_directory_for_run(terminal_root: Path, run_id: str) -> Path | None:
    matches: list[Path] = []
    for path in sorted(terminal_root.glob("scene-*/launch_request.json")):
        request = _parse(_read_bytes(path, reason="launch_request_invalid"), reason="launch_request_invalid")
        if request.get("run_id") == run_id:
            matches.append(path.parent)
    if len(matches) > 1:
        _fail("launch_bridge_ambiguous")
    return matches[0] if matches else None


def index_policy_canary_terminal(*, canary_run_root: str | Path,
                                 terminal_result_root: str | Path) -> dict[str, Any]:
    """Stage B: file the canary's sealed terminal set into its owner's directory.

    Requires the dispatcher's sealed ``dispatch_receipt.json`` (else
    ``dispatch_receipt_pending``) and a stage-A bridge whose ``run_id`` is this
    run's (else ``launch_bridge_pending`` -- also the permanent, truthful state of
    a canary no owner intent claims). Then the receipt's own records must bind
    the persisted projection, the persisted Website sync and the Vast provider-zero
    receipt by digest; the projection must validate and match the receipt; the
    sync must have succeeded against this projection; provider zero must be
    confirmed. Any failure is typed and nothing is written.
    """
    run_root = _safe(canary_run_root)
    terminal_root = _safe(terminal_result_root)
    base = _base("policy_canary_terminal", canary_run_root=str(run_root))
    receipt_path = run_root / CANARY_DISPATCH_RECEIPT_FILENAME
    if receipt_path.is_symlink():
        _fail("dispatch_receipt_invalid")
    if not receipt_path.is_file():
        return {**base, "status": "dispatch_receipt_pending"}
    receipt_raw = _read_bytes(receipt_path, reason="dispatch_receipt_absent")
    receipt = _parse(receipt_raw, reason="dispatch_receipt_invalid")
    run_id = receipt.get("run_id")
    if (receipt.get("schema_version") != DISPATCH_RECEIPT_SCHEMA or not _sealed(receipt, "receipt_digest")
            or receipt.get("run_kind") != POLICY_CANARY_RUN_KIND or not isinstance(run_id, str) or not run_id):
        _fail("dispatch_receipt_invalid")
    directory = _bridge_directory_for_run(terminal_root, run_id) if terminal_root.is_dir() else None
    if directory is None:
        return {**base, "status": "launch_bridge_pending", "run_id": run_id}
    records = {
        "projection": receipt.get("policy_canary_result_projection"),
        "sync": receipt.get("policy_canary_webapp_sync"),
        "provider_zero": receipt.get("provider_zero"),
    }
    if not all(_record(record) for record in records.values()):
        _fail("dispatch_receipt_records_missing")
    projection_raw = _read_bytes(run_root / PERSISTED_PROJECTION_RELATIVE_PATH, reason="projection_absent")
    sync_raw = _read_bytes(run_root / PERSISTED_WEBAPP_SYNC_RELATIVE_PATH, reason="webapp_sync_absent")
    zero_raw = _read_bytes(run_root / CANARY_PROVIDER_ZERO_FILENAME, reason="provider_zero_absent")
    if not (_bound(records["projection"], projection_raw) and _bound(records["sync"], sync_raw)
            and _bound(records["provider_zero"], zero_raw)):
        _fail("dispatch_receipt_binding_invalid")
    try:
        projection = validate_policy_canary_result(_parse(projection_raw, reason="projection_invalid"))
    except TaskEvaluationPolicyCanaryResultError:
        _fail("projection_invalid")
    if (projection["run_id"] != run_id
            or projection["projection_digest"] != receipt.get("policy_canary_projection_digest")
            or projection["result_delivery_digest"] != receipt.get("result_delivery_digest")
            or projection["result_status"] != receipt.get("status")):
        _fail("dispatch_receipt_binding_invalid")
    sync = _parse(sync_raw, reason="webapp_sync_invalid")
    if sync.get("status") != "succeeded":
        _fail("webapp_sync_not_succeeded")
    if (sync.get("schema_version") != WEBAPP_SYNC_SCHEMA or sync.get("run_id") != run_id
            or sync.get("request_digest") != projection["request_digest"]
            or sync.get("configuration_digest") != projection["configuration_digest"]
            or sync.get("result_status") != projection["result_status"]
            or sync.get("policy_canary_projection_digest") != projection["projection_digest"]
            or sync.get("notification_delivery") != receipt.get("notification_delivery")):
        _fail("webapp_sync_not_bound")
    zero = _parse(zero_raw, reason="provider_zero_invalid")
    if records["provider_zero"].get("provider_zero_verified") is not True or not confirmed_provider_zero(zero):
        _fail("provider_zero_not_confirmed")
    state = {"schema_version": STATE_SCHEMA, "run_id": run_id, "canary_run_root": str(run_root),
             "dispatch_receipt_digest": receipt["receipt_digest"],
             "projection_digest": projection["projection_digest"], "state_digest": ""}
    state["state_digest"] = canonical_digest(state, digest_field="state_digest")
    written = _publish(directory, {
        "policy_canary_result_projection.json": projection_raw,
        "policy_canary_webapp_sync.json": sync_raw,
        "provider_zero_closure.json": zero_raw,
        DISPATCH_RECEIPT_FILENAME: receipt_raw,
        STATE_FILENAME: _canonical(state),
    })
    return {**base, "status": "policy_canary_terminal_indexed", "run_id": run_id, "intent_id": directory.name,
            "directory": str(directory), "files": written}


# --------------------------------------------------------- stage C: durable publication


def index_result_publication(*, terminal_directory: str | Path) -> dict[str, Any]:
    """Stage C: derive the durable result publication from the offload pointer.

    The production GC retention step archives the canary root to the artifact
    store and leaves ``<root>.offloaded.v1.json`` (sealed; ``s3://`` URI, archive
    digest and size, per-member digests). The publication is written only when
    that pointer's members carry exactly the indexed projection bytes and the
    indexed sealed receipt bytes -- so the archive provably holds THIS run's
    result. Absent pointer: ``publication_pending``; unreadable: typed;
    another run's archive: ``publication_pointer_unbound``. Nothing is invented.
    """
    directory = _safe(terminal_directory)
    base = _base("result_publication", directory=str(directory))
    state_path = directory / STATE_FILENAME
    if state_path.is_symlink() or not state_path.is_file():
        return {**base, "status": "terminal_index_state_pending"}
    state = _parse(_read_bytes(state_path, reason="state_invalid"), reason="state_invalid")
    if state.get("schema_version") != STATE_SCHEMA or not _sealed(state, "state_digest"):
        _fail("state_invalid")
    run_root = Path(str(state.get("canary_run_root") or ""))
    if not run_root.is_absolute():
        _fail("state_invalid")
    pointer_path = run_root.parent / (run_root.name + POINTER_SUFFIX)
    located = {**base, "run_id": state.get("run_id"), "pointer_path": str(pointer_path)}
    if pointer_path.is_symlink():
        _fail("pointer_unsafe")
    if not pointer_path.exists():
        return {**located, "status": "publication_pending"}
    try:
        pointer_raw = pointer_path.read_bytes()
    except OSError as exc:
        return {**located, "status": "publication_pointer_unreadable", "error_type": type(exc).__name__}
    try:
        pointer = json.loads(pointer_raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {**located, "status": "publication_pointer_invalid"}
    uri = pointer.get("uri") if isinstance(pointer, dict) else None
    members = pointer.get("members") if isinstance(pointer, dict) else None
    if (not isinstance(pointer, dict) or pointer.get("schema_version") != POINTER_SCHEMA_VERSION
            or pointer.get("status") != "offloaded" or pointer.get("directory") != run_root.name
            or pointer.get("terminal_receipt") != CANARY_DISPATCH_RECEIPT_FILENAME
            or not _sealed(pointer, "pointer_digest")
            or not isinstance(uri, str) or not uri.startswith(_RESULT_URI_PREFIX) or "?" in uri
            or not _is_digest(pointer.get("digest"))
            or type(pointer.get("size_bytes")) is not int or pointer["size_bytes"] <= 0
            or not isinstance(members, list)):
        return {**located, "status": "publication_pointer_invalid"}
    projection_raw = _read_bytes(directory / "policy_canary_result_projection.json", reason="projection_absent")
    receipt_raw = _read_bytes(directory / DISPATCH_RECEIPT_FILENAME, reason="dispatch_receipt_absent")
    projection = _parse(projection_raw, reason="projection_invalid")
    receipt = _parse(receipt_raw, reason="dispatch_receipt_invalid")
    if (state.get("projection_digest") != projection.get("projection_digest")
            or state.get("run_id") != receipt.get("run_id")
            or state.get("dispatch_receipt_digest") != receipt.get("receipt_digest")):
        _fail("state_invalid")
    expected = {PERSISTED_PROJECTION_RELATIVE_PATH: _sha256(projection_raw),
                CANARY_DISPATCH_RECEIPT_FILENAME: _sha256(receipt_raw)}
    archived = {member.get("relative_path"): member.get("sha256") for member in members if isinstance(member, dict)}
    if any(archived.get(relative) != digest for relative, digest in expected.items()):
        return {**located, "status": "publication_pointer_unbound"}
    publication = {
        "schema_version": PUBLICATION_SCHEMA, "run_id": state["run_id"], "uri": uri,
        "digest": state["projection_digest"], "archive_digest": pointer["digest"],
        "size_bytes": pointer["size_bytes"], "archive_member_count": len(members),
        "pointer_digest": pointer["pointer_digest"], "provider_allocated": False, "publication_digest": "",
    }
    publication["publication_digest"] = canonical_digest(publication, digest_field="publication_digest")
    written = _publish(directory, {PUBLICATION_FILENAME: _canonical(publication)})
    return {**located, "status": "result_publication_indexed", "files": written}


__all__ = [
    "INDEX_SCHEMA",
    "PERSISTED_PROJECTION_RELATIVE_PATH",
    "PERSISTED_WEBAPP_SYNC_RELATIVE_PATH",
    "STATE_FILENAME",
    "TerminalResultIndexError",
    "file_record",
    "index_launch_bridge",
    "index_policy_canary_terminal",
    "index_result_publication",
]
