"""Remote CPU job records the host writes and reseals: teardowns, output pointers, record files.

Split from ``remote_cpu_job_contract``, which defines descriptors, receipts, heartbeats and the
shared record guard.  A teardown derives compute-zero and provider-zero from evidence, never from
a caller's assertion; an output pointer is resealed, never trimmed, when its state changes; and
every record file is written through the guard that refuses the transport schema and any URL- or
credential-shaped content.
"""

from __future__ import annotations

import errno
import json
import os
import re
import secrets
import stat
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .remote_cpu_job_contract import (
    _COMMIT,
    _IDENTITY,
    _IMAGE,
    _NAME,
    _OUTCOME,
    _ROW_SPEC,
    EXECUTION_PROVIDER,
    PERMITTED_PATH_ROOTS,
    STAGES,
    RemoteCpuContractError,
    _cas_uri_reasons,
    _check,
    _clone,
    _flag,
    _is_amount,
    _is_count,
    _is_digest,
    _matches,
    _nullable,
    _one_of,
    _optional_digest,
    _path_reasons,
    _positive,
    _raise_if,
    _row_id,
    _text,
    execution_name_of,
    forbidden_record_content,
    job_id_for,
    record_bytes,
    worker_identity_for,
)

TEARDOWN_SCHEMA_VERSION = "remote_cpu_job_teardown.v1"
POINTER_SCHEMA_VERSION = "remote_cpu_output_pointer.v1"
POINTER_STATES = ("landed", "restored_full")
# The result's references to bytes that stayed remote (the packet), each by path, digest and size, so that
# a reader such as the owner census can let the pointer stand for them.
MAX_POINTER_RAW_REFERENCES = 64
_MAX_RECORD_BYTES = 1024 * 1024
_TRANSPORT_OBJECT = re.compile(r"gs://[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]/transport/[a-z0-9-]+/[a-z0-9-]+\.json")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_remote_cpu_record(path: str | Path, value: Mapping[str, Any]) -> bool:
    """Create one immutable record (exclusive, fsynced, 0640); ``False`` if it already exists identically."""
    payload = record_bytes(value)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(8)}.tmp")
    try:
        with open(temporary, "xb") as stream:
            os.fchmod(stream.fileno(), 0o640)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        created = True
    except FileExistsError:
        created = False
    finally:
        temporary.unlink(missing_ok=True)
    if created:
        fsync_directory(destination.parent)
        return True
    with suppress(OSError):
        if not destination.is_symlink() and destination.read_bytes() == payload:
            return False
    raise RemoteCpuContractError(f"remote_cpu_record_conflict:{destination.name}")


# What opening a record path answers when a symlink (O_NOFOLLOW), a directory or a special file is there.
_NOT_A_RECORD = frozenset({errno.ELOOP, errno.EMLINK, errno.EISDIR, errno.ENXIO, errno.ENODEV})


def _existing_record_bytes(path: Path) -> bytes | None:
    """The bytes of a regular record file, or ``None`` when absent.  A symlink or a special file at the path
    conflicts; any other failure to open or read it (EMFILE, EIO, EACCES) is raised as it is, for the caller
    to retry, never taken for a conflict."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in _NOT_A_RECORD:
            raise RemoteCpuContractError(f"remote_cpu_record_conflict:{path.name}") from exc
        raise
    try:
        regular = stat.S_ISREG(os.fstat(descriptor).st_mode)
    except OSError:
        os.close(descriptor)
        raise
    if not regular:
        os.close(descriptor)
        raise RemoteCpuContractError(f"remote_cpu_record_conflict:{path.name}")
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read(_MAX_RECORD_BYTES + 1)


def replace_remote_cpu_record(path: str | Path, value: Mapping[str, Any], *, previous_digest: str | None,
                              digest_field: str, mode: int = 0o440) -> bool:
    """Write a sealed record, replacing only the record the caller read: a digest compare-and-swap.

    ``previous_digest`` is the ``digest_field`` of the record the caller read, or ``None`` to create
    the file exclusively.  An identical file means the write already happened (``False``); any other
    file is a conflict.  The caller serializes writers (the collector holds the attempt's lease).
    The default mode is 0440, as plan 14 §16 requires of the output pointer.
    """
    payload = record_bytes(value)
    if value.get(digest_field) != canonical_digest(value, digest_field=digest_field):
        raise RemoteCpuContractError("remote_cpu_record_unsealed")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    current = _existing_record_bytes(destination)
    if current == payload:
        return False
    if previous_digest is None and current is not None:
        raise RemoteCpuContractError(f"remote_cpu_record_conflict:{destination.name}")
    if previous_digest is not None:
        try:
            existing = json.loads(current) if current is not None else None
        except (ValueError, RecursionError):
            existing = None
        if (not isinstance(existing, dict) or existing.get(digest_field) != previous_digest
                or previous_digest != canonical_digest(existing, digest_field=digest_field)):
            raise RemoteCpuContractError(f"remote_cpu_record_cas_conflict:{destination.name}")
    temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(8)}.tmp")
    try:
        with open(temporary, "xb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if previous_digest is None:
            os.link(temporary, destination)
        else:
            os.replace(temporary, destination)
    except FileExistsError as exc:
        raise RemoteCpuContractError(f"remote_cpu_record_conflict:{destination.name}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    fsync_directory(destination.parent)
    return True


_COMPUTE_SPEC = {
    "execution_completed": _flag, "running_count": _is_count, "listing_complete": _flag, "listing_pages": _is_count,
    "executions_for_attempt": _is_count, "unfinished_executions_for_attempt": _is_count,
    # The evidence names the transport it proved deleted, so a lease can bind it to its own.
    "transport_object": _matches(_TRANSPORT_OBJECT), "transport_generation": _positive,
    "transport_deleted": _flag, "transport_absent_at_generation": _flag,
}
_PROVIDER_SPEC = {
    "staging_versions_deleted": _is_count, "staging_versions_remaining": _is_count,
    "staging_listing_complete": _flag, "write_urls_expire_at_epoch": _is_amount, "read_urls_expire_at_epoch": _is_amount,
}
_TEARDOWN_SPEC = {
    "schema_version": _one_of(TEARDOWN_SCHEMA_VERSION), "job_id": _text, "attempt": _positive,
    "attempt_id": _matches(_NAME), "stage": _one_of(*STAGES), "descriptor_digest": _is_digest,
    "worker_identity": _nullable(_matches(_IDENTITY)), "outcome": _matches(_OUTCOME),
    "compute_zero": _COMPUTE_SPEC, "compute_zero_proven": _flag, "provider_zero": _PROVIDER_SPEC,
    "provider_zero_proven": _flag, "observed_at_epoch": _is_amount, "private_url_recorded": _one_of(False),
    "teardown_digest": _is_digest,
}
_POINTER_SPEC = {
    "schema_version": _one_of(POINTER_SCHEMA_VERSION), "stage": _one_of(*STAGES), "compilation_id": _text,
    "queue_row": _ROW_SPEC, "attempt_id": _matches(_NAME), "descriptor_digest": _is_digest,
    "receipt_digest": _is_digest,
    "execution": {"provider": _one_of(EXECUTION_PROVIDER), "job": _text, "worker_identity": _matches(_IDENTITY),
                  "allocation_binding_digest": _is_digest, "spend_consumption": _is_digest},
    "code": {"source_commit": _matches(_COMMIT), "source_archive_digest": _is_digest, "image": _matches(_IMAGE),
             "environment_digest": _is_digest},
    "archive": {"uri": _text, "digest": _is_digest, "size_bytes": _positive},
    "index": {"uri": _text, "digest": _is_digest, "size_bytes": _positive},
    "output_root": _text, "paths_total": _is_count, "bytes_total": _is_count,
    "host_known": {"count": _is_count, "bytes": _is_count},
    "landed": {"subset": _matches(_OUTCOME), "paths": _is_count, "bytes": _is_count},
    "raw_references": [{"path": _text, "digest": _is_digest, "size_bytes": _positive}],
    "state": _one_of(*POINTER_STATES), "teardown_receipt_digest": _optional_digest,
    "provider_zero_proven": _flag, "pointer_digest": _is_digest,
}


def compute_zero_proven(compute: Mapping[str, Any], *, worker_identity: str | None) -> bool:
    """Terminal execution, a complete listing with nothing unfinished, transport gone at its generation."""
    reasons: list[str] = []
    _check(compute, _COMPUTE_SPEC, "compute", reasons)
    _raise_if(reasons)
    # Without a known execution (a lost dispatch response) no execution may carry the attempt.
    ran = (compute["execution_completed"] and compute["running_count"] == 0) if worker_identity else (
        compute["executions_for_attempt"] == 0)
    return bool(ran and compute["listing_complete"] and compute["listing_pages"] >= 1
                and compute["unfinished_executions_for_attempt"] == 0
                and compute["transport_deleted"] and compute["transport_absent_at_generation"])


def _provider_zero(compute_zero: bool, provider: Mapping[str, Any], observed_at_epoch: float) -> bool:
    # B2 only hides a deleted object, so the proof is an empty version listing.  Presigned GETs name
    # durable inputs and cannot write, so only the attempt's PUTs count, and deleting staging does not
    # revoke a PUT: only its expiry ends it.  The read expiry is recorded for audit alone.
    return bool(compute_zero and provider["staging_listing_complete"] and provider["staging_versions_remaining"] == 0
                and observed_at_epoch >= provider["write_urls_expire_at_epoch"])


def teardown_record(
    *, descriptor: Mapping[str, Any], worker_identity: str | None, outcome: str, compute: Mapping[str, Any],
    provider: Mapping[str, Any], observed_at_epoch: float,
) -> dict[str, Any]:
    """Seal ``remote_cpu_job_teardown.v1``; both zero proofs are derived from evidence, never asserted."""
    reasons = forbidden_record_content({"outcome": outcome, "compute": compute, "provider": provider})
    _check(compute, _COMPUTE_SPEC, "compute", reasons)
    _check(provider, _PROVIDER_SPEC, "provider", reasons)
    descriptor = _clone(descriptor, "descriptor")
    if descriptor.get("descriptor_digest") != canonical_digest(descriptor, digest_field="descriptor_digest"):
        reasons.append("remote_cpu_teardown_descriptor_unsealed")
    if reasons or not _is_amount(observed_at_epoch):
        raise RemoteCpuContractError(reasons or ["remote_cpu_field_invalid:observed_at_epoch"])
    if worker_identity is not None and worker_identity != worker_identity_for(
            descriptor.get("execution") or {}, execution_name_of(worker_identity)):
        raise RemoteCpuContractError("remote_cpu_teardown_worker_identity_unbound")
    compute_zero = compute_zero_proven(compute, worker_identity=worker_identity)
    record: dict[str, Any] = {
        "schema_version": TEARDOWN_SCHEMA_VERSION,
        **{name: descriptor.get(name) for name in ("job_id", "attempt", "attempt_id", "stage", "descriptor_digest")},
        "worker_identity": worker_identity, "outcome": outcome,
        "compute_zero": dict(compute), "compute_zero_proven": compute_zero, "provider_zero": dict(provider),
        "provider_zero_proven": _provider_zero(compute_zero, provider, float(observed_at_epoch)),
        "observed_at_epoch": float(observed_at_epoch), "private_url_recorded": False, "teardown_digest": "",
    }
    record["teardown_digest"] = canonical_digest(record, digest_field="teardown_digest")
    return validate_teardown(record)


def validate_teardown(value: Mapping[str, Any]) -> dict[str, Any]:
    """Check a sealed teardown and re-derive both proofs from its own evidence."""
    record = _clone(value, "teardown")
    reasons = forbidden_record_content(record)
    _check(record, _TEARDOWN_SPEC, "", reasons)
    if record.get("teardown_digest") != canonical_digest(record, digest_field="teardown_digest"):
        reasons.append("remote_cpu_teardown_digest_mismatch")
    _raise_if(reasons)
    transport = rf"gs://[^/]+/transport/{re.escape(record['job_id'])}/{re.escape(record['attempt_id'])}-[0-9a-f]{{32}}\.json"
    if not re.fullmatch(transport, record["compute_zero"]["transport_object"]):
        raise RemoteCpuContractError("remote_cpu_teardown_transport_unbound")
    compute_zero = compute_zero_proven(record["compute_zero"], worker_identity=record["worker_identity"])
    if record["compute_zero_proven"] is not compute_zero or record["provider_zero_proven"] is not _provider_zero(
            compute_zero, record["provider_zero"], record["observed_at_epoch"]):
        raise RemoteCpuContractError("remote_cpu_teardown_proof_not_derived")
    return record


def _pointer_reasons(pointer: Mapping[str, Any], *, output_roots: tuple[str, ...] = PERMITTED_PATH_ROOTS) -> list[str]:
    reasons = forbidden_record_content(pointer)
    _check(pointer, _POINTER_SPEC, "", reasons)
    if reasons:
        return reasons
    row, stage, execution = pointer["queue_row"], pointer["stage"], pointer["execution"]
    paths, total, known, landed = pointer["paths_total"], pointer["bytes_total"], pointer["host_known"], pointer["landed"]
    reasons.extend(_path_reasons(pointer["output_root"], "output_root", output_roots))
    references = pointer["raw_references"]
    for index, reference in enumerate(references):
        reasons.extend(_path_reasons(reference["path"], f"raw_references[{index}].path", (pointer["output_root"] + "/",)))
    if len(references) > MAX_POINTER_RAW_REFERENCES or len({row["path"] for row in references}) != len(references):
        reasons.append("remote_cpu_pointer_raw_references_invalid")
    for name in ("archive", "index"):
        reasons.extend(_cas_uri_reasons(pointer[name]["uri"], pointer[name]["digest"], f"{name}.uri"))
    reasons.extend(f"remote_cpu_pointer_{name}" for name, failed in {
        "queue_row_unbound": row["queue"] != STAGES[stage]["queue"] or pointer["compilation_id"] != _row_id(row),
        "attempt_unbound": not re.fullmatch(rf"{re.escape(job_id_for(stage, row['name']))}-a[1-9][0-9]*-[0-9a-f]{{32}}",
                                            pointer["attempt_id"]),
        "output_root_unbound": pointer["output_root"].rsplit("/", 1)[-1] != pointer["compilation_id"],
        "worker_identity_unbound": _IDENTITY.fullmatch(execution["worker_identity"]).group(3) != execution["job"],
        "totals_invalid": known["count"] > paths or known["bytes"] > total or landed["paths"] > paths
        or landed["bytes"] > total,
        "provider_zero_unbound": pointer["provider_zero_proven"] is not (pointer["teardown_receipt_digest"] is not None),
    }.items() if failed)
    return reasons


def validate_pointer(value: Mapping[str, Any], *, output_roots: tuple[str, ...] = PERMITTED_PATH_ROOTS) -> dict[str, Any]:
    """A sealed ``remote_cpu_output_pointer.v1`` exactly as the collector writes it, or a typed refusal.

    ``output_roots`` names where its output may lie: a reader of retained records, such as the owner census,
    checks the pointer against its own compiled-episodes root rather than this host's.
    """
    pointer = _clone(value, "pointer")
    reasons = _pointer_reasons(pointer, output_roots=tuple(output_roots))
    if pointer.get("pointer_digest") != canonical_digest(pointer, digest_field="pointer_digest"):
        reasons.append("remote_cpu_pointer_digest_mismatch")
    _raise_if(reasons)
    return pointer


def pointer_record(fields: Mapping[str, Any], *, previous: Mapping[str, Any] | None = None,
                   teardown: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Seal ``remote_cpu_output_pointer.v1``, or reseal ``previous``: only ``state`` advances
    (``landed`` then ``restored_full``) and a provider-zero teardown attaches; nothing is dropped."""
    fields = _clone(fields, "pointer_fields")
    reasons = forbidden_record_content(fields)
    _raise_if(reasons)
    if previous is None:
        reasons.extend(f"remote_cpu_pointer_field_derived:{name}" for name in (
            "schema_version", "teardown_receipt_digest", "provider_zero_proven", "pointer_digest") if name in fields)
        if fields.get("state") != "landed":
            reasons.append("remote_cpu_pointer_initial_state_invalid")
        pointer = {**fields, "schema_version": POINTER_SCHEMA_VERSION, "teardown_receipt_digest": None,
                   "provider_zero_proven": False, "pointer_digest": ""}
    else:
        pointer = _clone(previous, "pointer")
        prior = _pointer_reasons(pointer)
        if pointer.get("pointer_digest") != canonical_digest(pointer, digest_field="pointer_digest"):
            prior.append("remote_cpu_pointer_digest_mismatch")
        _raise_if(prior)
        for name, item in fields.items():
            if name != "state":
                if pointer.get(name, object()) != item:
                    reasons.append(f"remote_cpu_pointer_field_immutable:{name}")
            elif item not in POINTER_STATES:
                reasons.append("remote_cpu_pointer_state_invalid")
            elif POINTER_STATES.index(item) < POINTER_STATES.index(pointer["state"]):
                reasons.append("remote_cpu_pointer_state_regression")
            else:
                pointer["state"] = item
    if teardown is not None:
        record = validate_teardown(teardown)
        execution = pointer.get("execution") if isinstance(pointer.get("execution"), Mapping) else {}
        if (record["attempt_id"], record["descriptor_digest"]) != (pointer.get("attempt_id"),
                                                                   pointer.get("descriptor_digest")):
            reasons.append("remote_cpu_pointer_teardown_attempt_mismatch")
        if record["worker_identity"] != execution.get("worker_identity"):
            reasons.append("remote_cpu_pointer_teardown_worker_identity_mismatch")
        if not record["provider_zero_proven"]:
            reasons.append("remote_cpu_pointer_teardown_not_provider_zero")
        if pointer.get("teardown_receipt_digest") not in {None, record["teardown_digest"]}:
            reasons.append("remote_cpu_pointer_field_immutable:teardown_receipt_digest")
        pointer["teardown_receipt_digest"], pointer["provider_zero_proven"] = record["teardown_digest"], True
    _raise_if(reasons)
    pointer["pointer_digest"] = canonical_digest(pointer, digest_field="pointer_digest")
    reasons = _pointer_reasons(pointer)
    _raise_if(reasons)
    return pointer
