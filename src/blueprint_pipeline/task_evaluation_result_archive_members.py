"""Registered result artifacts that stay in a streamed attempt's promoted archive.

A streamed policy canary keeps under its evidence root only the members its
readers read as bytes (``provider_output_member_view``); every other registered
artifact is a member of the promoted archive in B2, answered by the view's
index. In download mode no view exists and nothing here is reached: every
caller keeps today's local path.

References file. After the registry seals, the delivery writes
``artifacts/result_delivery/archive_member_references.v1.json`` (schema
``task_evaluation_result_archive_member_references.v1``) beside
``remote_artifacts/``. It binds the run id and registry digest, the member
index digest, the archive {sha256, size_bytes, durable_reference}, the member
index file and the view descriptor (run-relative path and sha256), and holds,
per run-relative path of every registered artifact that is not on disk, its
``archive_path``, ``sha256``, ``size_bytes``, ``crc32``, ``method``,
``data_offset`` and ``compressed_size``. It is self-contained: a download reads
one member from B2 with it alone, and the storage GC counts those paths as
already remote. Because it names the index and the descriptor, residue offload
keeps them. It is written once; a replay must produce the same bytes. Artifact
ids (role, path, digest) are the download mode's.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

REFERENCES_SCHEMA = "task_evaluation_result_archive_member_references.v1"
REFERENCES_RELATIVE_PATH = "artifacts/result_delivery/archive_member_references.v1.json"
_ENTRY_KEYS = ("archive_path", "sha256", "size_bytes", "crc32", "method", "data_offset", "compressed_size")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
# View refusals meaning the durable copy's bytes are not the indexed member's:
# download mode's digest refusal, never a transient transport failure.
CONTENT_MISMATCH_CODES = frozenset({
    "provider_output_member_digest_mismatch",
    "provider_output_archive_deflate_invalid",
    "provider_output_archive_deflate_end_invalid",
    "provider_output_archive_member_size_mismatch",
})


class ArchiveMemberReferenceError(ValueError):
    """A typed, secret-free refusal; the message is the stable code."""


def open_evidence_view(evidence_root: str | Path, *, error_factory: Callable[[str], Exception] = ValueError):
    """The member view covering ``evidence_root``, or None in download mode.

    A descriptor that exists but does not bind its index, archive, durable copy
    or ingestion receipt is refused, never read as download mode.
    """
    from .provider_output_member_view import ProviderOutputMemberViewError, open_member_view

    try:
        return open_member_view(evidence_root)
    except ProviderOutputMemberViewError as exc:
        raise error_factory(f"result_delivery_member_view_invalid:{exc}") from None


def archive_member(view, path: str | Path) -> dict | None:
    """The index row of the absent file at ``path``, or None (no view, or not a member)."""
    return None if view is None else view.member_at(path)


def _file_record(path: Path, run_root: Path) -> dict[str, Any]:
    import hashlib

    data = path.read_bytes()
    return {"path": path.relative_to(run_root).as_posix(),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def build_archive_member_references(*, run_root: str | Path, registry: Mapping[str, Any], view) -> dict | None:
    """The references file for a sealed registry, or None when every artifact is local."""
    root = Path(run_root).resolve()
    members: dict[str, dict[str, Any]] = {}
    for record in registry.get("artifacts") or []:
        path = Path(str(record.get("evidence_root") or "")) / str(record.get("relative_path") or "")
        if path.is_file():
            continue
        row = archive_member(view, path)
        if row is None or (row["sha256"], row["size"]) != (record.get("sha256"), record.get("size_bytes")):
            raise ArchiveMemberReferenceError("result_archive_member_reference_unbound")
        entry = {"archive_path": row["path"], "sha256": row["sha256"], "size_bytes": row["size"],
                 "crc32": row["crc32"], "method": row["method"], "data_offset": row["data_offset"],
                 "compressed_size": row["compressed_size"]}
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError:
            raise ArchiveMemberReferenceError("result_archive_member_outside_run") from None
        if members.setdefault(relative, entry) != entry:
            raise ArchiveMemberReferenceError("result_archive_member_reference_alias_conflict")
    if not members:
        return None
    attempt = view.evidence_root.parent
    descriptor = attempt / (view.evidence_root.name + ".member_view.v1.json")
    archive = view.index["archive"]
    try:
        index_record = _file_record(attempt / view.descriptor["member_index"]["path"], root)
        descriptor_record = _file_record(descriptor, root)
    except ValueError:
        raise ArchiveMemberReferenceError("result_archive_member_view_outside_run") from None
    value = {
        "schema_version": REFERENCES_SCHEMA,
        "run_id": registry["run_id"],
        "registry_digest": registry["registry_digest"],
        "member_index_digest": view.index["index_digest"],
        "archive": {"sha256": archive["sha256"], "size_bytes": archive["size"],
                    "durable_reference": archive["durable_reference"]},
        "member_index": index_record,
        "member_view": {"path": descriptor_record["path"], "sha256": descriptor_record["sha256"],
                        "view_digest": view.descriptor["view_digest"]},
        "members": dict(sorted(members.items())),
        "private_url_recorded": False,
    }
    value["references_digest"] = canonical_digest(value, digest_field="references_digest")
    return value


def write_archive_member_references(run_root: str | Path, value: Mapping[str, Any]) -> None:
    """Write the references file once; the same bytes again are a no-op."""
    path = Path(run_root).resolve() / REFERENCES_RELATIVE_PATH
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ArchiveMemberReferenceError("result_archive_member_references_conflict")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ArchiveMemberReferenceError("result_archive_member_references_conflict") from None
    finally:
        temporary.unlink(missing_ok=True)


def load_archive_member_references(run_root: str | Path, registry: Mapping[str, Any]) -> dict | None:
    """The run's references file bound to ``registry``, or None when there is none."""
    path = Path(run_root) / REFERENCES_RELATIVE_PATH
    if not path.exists() and not path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if not path.is_symlink() else None
    except (OSError, UnicodeError, ValueError):
        value = None
    members = value.get("members") if isinstance(value, dict) else None
    archive = value.get("archive") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or value.get("schema_version") != REFERENCES_SCHEMA
            or value.get("private_url_recorded") is not False
            or value.get("references_digest") != _digest_or_none(value)
            or value.get("run_id") != registry.get("run_id")
            or value.get("registry_digest") != registry.get("registry_digest")
            or not isinstance(archive, dict) or not _DIGEST.fullmatch(str(archive.get("sha256")))
            or not _durable(archive)
            or not isinstance(members, dict)
            or not all(isinstance(entry, dict) and tuple(sorted(entry)) == tuple(sorted(_ENTRY_KEYS))
                       for entry in members.values())):
        raise ArchiveMemberReferenceError("result_archive_member_references_invalid")
    return value


def archive_member_entry(references: Mapping[str, Any] | None, relative: str,
                         record: Mapping[str, Any]) -> dict | None:
    """The reference entry for run-relative ``relative`` when it names ``record``'s bytes."""
    entry = ((references or {}).get("members") or {}).get(relative)
    if entry is None:
        return None
    if (entry.get("sha256"), entry.get("size_bytes")) != (record.get("sha256"), record.get("size_bytes")):
        raise ArchiveMemberReferenceError("result_archive_member_reference_mismatch")
    return dict(entry)


def _durable(archive: Mapping[str, Any]) -> bool:
    from .provider_output_member_index import ProviderOutputMemberIndexError, durable_reference_facts

    reference = archive.get("durable_reference")
    try:
        facts = durable_reference_facts(reference)
    except ProviderOutputMemberIndexError:
        return False
    return (facts == reference and facts["digest"] == archive.get("sha256")
            and facts["size_bytes"] == archive.get("size_bytes"))


def _digest_or_none(value: Mapping[str, Any]) -> str | None:
    try:
        return canonical_digest(value, digest_field="references_digest")
    except (TypeError, ValueError, RecursionError):
        return None


__all__ = [
    "CONTENT_MISMATCH_CODES",
    "REFERENCES_RELATIVE_PATH",
    "REFERENCES_SCHEMA",
    "ArchiveMemberReferenceError",
    "archive_member",
    "archive_member_entry",
    "build_archive_member_references",
    "load_archive_member_references",
    "open_evidence_view",
    "write_archive_member_references",
]
