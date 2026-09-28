"""Answer provider-output member digests and bytes from the durable archive.

A streamed attempt keeps under ``immutable_execution/`` only the members its
consumers read as bytes; every other member stays in the promoted archive in
B2. A reader that finds no local file asks the view instead.

Descriptor. ``<attempt>/immutable_execution.member_view.v1.json`` (schema
``provider_output_member_view.v1``) sits beside the evidence root it binds and
holds the root's name, ``archive_prefix: ""``, the member index's file record
{path, sha256, size_bytes} and ``member_index_digest``, the ``archive_sha256``,
the ``durable_reference``, the ingestion receipt's path and sha256, and
``view_digest``. Paths are relative to the attempt, so the pair survives a
move. It is written once, and only for a materialized ingestion of an index
sealed with its durable reference.

Discovery. ``open_member_view(path)`` looks for a sibling descriptor of the
path itself and of at most eight ancestors (the deepest Quick-10 member sits
eight levels below its root). It returns ``None`` when there is none -- a
download-mode attempt, whose readers then run exactly today's code -- and
refuses a descriptor that does not bind its index, archive, durable copy and
ingestion receipt (``provider_output_member_view_*``) rather than fall back.

API. ``member``, ``digest`` and ``verify`` answer from the index and read no
bytes. ``read_member(rel, maximum_bytes=...)`` is one range request to the
durable copy (after the reader's one-byte probe that pins the ETag), inflated
in memory and checked against the index's CRC-32 and SHA-256.
``fetch_to(rel, destination)`` streams the same into a ``0440`` file through a
partial, outside the evidence root. A view never writes into its evidence
root, so a resumed ingestion never finds a file it did not write there.

B2 must be explicitly configured for the default presign, exactly as for
promotion: the view never borrows the staging store's credentials.

CLI (read-only): ``python -m blueprint_pipeline.provider_output_member_view
plan --archive <zip> --contract policy_canary_output_member_contract.v1``
prints members, bytes by class and bytes by disposition under the contract,
and any entry-rule refusal the index would raise, from the central directory
alone. The contract here is the measurement rule the lane's contract module
will own: every ``.json`` file outside a ``policy-requests`` directory is
materialized; everything else stays remote.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
import zipfile
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .provider_output_member_index import (
    BULK_EXTENSIONS,
    MAX_EXPANDED_BYTES,
    MemberInflater,
    ProviderOutputMemberIndexError,
    check_archive_entries,
    inflate_step_bytes,
    read_indexed_member,
    validate_member_index,
)
from .provider_output_native_inventory import ProviderOutputInventoryError, safe_member_name
from .provider_output_range_ingestion import CasArchiveSource, ProviderOutputIngestionError
from .provider_output_range_transport import ProviderOutputTransportError
from .task_evaluation_configured_scene_object_store import (
    _ARTIFACT_STORE_FILE_ENV,
    presign_configured_scene_artifact,
)

SCHEMA = "provider_output_member_view.v1"
PLAN_SCHEMA = "provider_output_member_plan.v1"
DESCRIPTOR_SUFFIX = ".member_view.v1.json"
MAXIMUM_ANCESTORS = 8
PRESIGN_EXPIRATION_SECONDS = 3600
POLICY_CANARY_CONTRACT = "policy_canary_output_member_contract.v1"


def _policy_canary_member_needed(path: str) -> bool:
    member = PurePosixPath(path)
    return member.suffix.lower() == ".json" and "policy-requests" not in member.parts


CONTRACTS: dict[str, Callable[[str], bool]] = {POLICY_CANARY_CONTRACT: _policy_canary_member_needed}


class ProviderOutputMemberViewError(ValueError):
    """A typed, secret-free refusal; the message is the stable code."""


def _refuse(code: str) -> ProviderOutputMemberViewError:
    return ProviderOutputMemberViewError(code)


def _file_record(path: Path, parent: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise _refuse("provider_output_member_view_file_missing")
    try:
        relative = path.absolute().relative_to(parent.absolute()).as_posix()
    except ValueError:
        raise _refuse("provider_output_member_view_file_outside_attempt") from None
    data = path.read_bytes()
    return {"path": relative, "sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None


def _digest_or_none(value: Any, field: str) -> str | None:
    try:
        return canonical_digest(value, digest_field=field)
    except (TypeError, ValueError, RecursionError):
        return None


def build_member_view_descriptor(*, evidence_root: str | Path, index_path: str | Path,
                                 ingestion_receipt_path: str | Path) -> dict:
    """The view descriptor binding a materialized ingestion to its sealed index."""
    root = Path(evidence_root).absolute()
    parent = root.parent
    index_file, receipt_file = Path(index_path), Path(ingestion_receipt_path)
    index_record = _file_record(index_file, parent)
    receipt_record = _file_record(receipt_file, parent)
    index = _json(index_file)
    try:
        validate_member_index(index)
    except ProviderOutputMemberIndexError:
        raise _refuse("provider_output_member_view_index_invalid") from None
    if index["archive"]["durable_reference"] is None:
        raise _refuse("provider_output_member_view_index_not_durable")
    receipt = _json(receipt_file)
    if not isinstance(receipt, Mapping) or receipt.get("status") != "materialized":
        raise _refuse("provider_output_member_view_ingestion_not_materialized")
    if (receipt.get("member_index_digest") != index["index_digest"]
            or receipt.get("archive_sha256") != index["archive"]["sha256"]
            or receipt.get("receipt_digest") != _digest_or_none(receipt, "receipt_digest")):
        raise _refuse("provider_output_member_view_receipt_mismatch")
    descriptor = {
        "schema_version": SCHEMA,
        "evidence_root": root.name,
        "archive_prefix": "",
        "member_index": index_record,
        "member_index_digest": index["index_digest"],
        "archive_sha256": index["archive"]["sha256"],
        "durable_reference": index["archive"]["durable_reference"],
        "ingestion_receipt_path": receipt_record["path"],
        "ingestion_receipt_sha256": receipt_record["sha256"],
        "private_url_recorded": False,
    }
    descriptor["view_digest"] = canonical_digest(descriptor, digest_field="view_digest")
    return descriptor


def write_member_view_descriptor(*, evidence_root: str | Path, index_path: str | Path,
                                 ingestion_receipt_path: str | Path) -> dict:
    """Write the descriptor once, beside the evidence root; the same one again is a no-op."""
    descriptor = build_member_view_descriptor(evidence_root=evidence_root, index_path=index_path,
                                              ingestion_receipt_path=ingestion_receipt_path)
    root = Path(evidence_root).absolute()
    path = root.parent / (root.name + DESCRIPTOR_SUFFIX)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or _json(path) != descriptor:
            raise _refuse("provider_output_member_view_descriptor_conflict")
        return descriptor
    temporary = root.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(descriptor, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise _refuse("provider_output_member_view_descriptor_conflict") from None
    finally:
        temporary.unlink(missing_ok=True)
    return descriptor


def _default_presign(reference: Mapping[str, Any]) -> Callable[[], str]:
    if not all(str(os.environ.get(name) or "").strip() for name in _ARTIFACT_STORE_FILE_ENV.values()):
        raise _refuse("provider_output_member_view_artifact_store_not_configured")
    return lambda: presign_configured_scene_artifact(reference=reference,
                                                     expiration_seconds=PRESIGN_EXPIRATION_SECONDS)


class ProviderOutputMemberView:
    """Digests from a sealed member index; bytes by range from its durable copy."""

    def __init__(self, *, evidence_root: Path, descriptor: dict, index: dict,
                 presign: Callable[[], str] | None, opener: Callable | None):
        self.evidence_root, self.descriptor, self.index = evidence_root, descriptor, index
        self._presign, self._opener = presign, opener
        self._rows = {row["path"]: row for row in index["members"] if row["kind"] == "file"}

    def _path(self, relative: str) -> str:
        try:
            return self.descriptor["archive_prefix"] + safe_member_name(str(relative))
        except ProviderOutputInventoryError:
            raise _refuse("provider_output_member_view_path_invalid") from None

    def member(self, relative: str) -> dict | None:
        row = self._rows.get(self._path(relative))
        return dict(row) if row is not None else None

    def relative(self, path: str | Path) -> str | None:
        """The member path of ``path`` when it lies under the evidence root, else None."""
        for candidate in (Path(path).absolute(), Path(path).resolve()):
            try:
                return candidate.relative_to(self.evidence_root).as_posix()
            except ValueError:
                continue
        return None

    def member_at(self, path: str | Path) -> dict | None:
        """The index row of the file at ``path``, or None when it is no archive member."""
        relative = self.relative(path)
        if relative in (None, "."):
            return None
        try:
            return self.member(relative)
        except ProviderOutputMemberViewError:
            return None

    def digest(self, relative: str) -> str | None:
        row = self.member(relative)
        return row["sha256"] if row is not None else None

    def verify(self, relative: str, *, sha256: str, size_bytes: int) -> bool:
        row = self.member(relative)
        return row is not None and (row["sha256"], row["size"]) == (sha256, size_bytes)

    def _row(self, relative: str) -> dict:
        row = self.member(relative)
        if row is None:
            raise _refuse("provider_output_member_view_member_absent")
        return row

    def _reader(self):
        reference = self.index["archive"]["durable_reference"]
        presign = self._presign or _default_presign(reference)
        try:
            reader = CasArchiveSource(reference, presign=presign, opener=self._opener).open(
                self.index["archive"]["size"])
        except (ProviderOutputIngestionError, ProviderOutputTransportError) as exc:
            raise _refuse(str(exc)) from None
        if reader.identity["size_bytes"] != self.index["archive"]["size"]:
            raise _refuse("provider_output_remote_size_mismatch")
        return reader

    def read_member(self, relative: str, *, maximum_bytes: int) -> bytes:
        """One range request for the member's data, inflated in memory and checked."""
        row = self._row(relative)
        if type(maximum_bytes) is not int or row["size"] > maximum_bytes:
            raise _refuse("provider_output_member_read_cap_exceeded")
        try:
            return read_indexed_member(self._reader(), row, maximum_bytes=maximum_bytes)
        except ProviderOutputMemberIndexError as exc:
            raise _refuse(str(exc)) from None

    def fetch_to(self, relative: str, destination: str | Path) -> dict:
        """Stream one member into ``destination`` (``0440``, never replaced), outside the root."""
        target = Path(destination).absolute()
        root = self.evidence_root.resolve()
        resolved = target.resolve()
        if resolved == root or root in resolved.parents:
            raise _refuse("provider_output_member_view_write_inside_evidence_root")
        if target.exists() or target.is_symlink():
            raise _refuse("provider_output_member_view_destination_exists")
        row = self._row(relative)
        reader = self._reader()
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
        try:
            with partial.open("xb") as sink:
                self._stream(row, reader, sink.write)
                sink.flush()
                os.fsync(sink.fileno())
            partial.chmod(0o440)
            os.link(partial, target)
        except FileExistsError:
            raise _refuse("provider_output_member_view_destination_exists") from None
        finally:
            partial.unlink(missing_ok=True)
        return {"path": str(target), "size_bytes": row["size"], "sha256": row["sha256"]}

    def stream_member(self, relative: str, sink: Callable[[bytes], Any]) -> dict:
        """Pass one member's inflated bytes to ``sink`` in order, with one range request.

        The bytes are checked against the index's CRC-32 and SHA-256 as they
        pass; a mismatch raises ``provider_output_member_digest_mismatch``
        after the last chunk, so a caller that keeps what ``sink`` received
        must discard it on any refusal. Returns the member's index row.
        """
        row = self._row(relative)
        return self._stream(row, self._reader(), sink)

    @staticmethod
    def _stream(row: dict, reader, sink: Callable[[bytes], Any]) -> dict:
        digest, crc = hashlib.sha256(), [0]

        def emit(data):
            sink(data)
            digest.update(data)
            crc[0] = zlib.crc32(data, crc[0])

        try:
            inflater = MemberInflater(row["method"], row["size"], emit,
                                      step_bytes=inflate_step_bytes(reader.block_bytes))
            if row["compressed_size"]:
                reader.stream_to(inflater.feed, start=row["data_offset"],
                                 end=row["data_offset"] + row["compressed_size"])
            inflater.finish()
        except (ProviderOutputMemberIndexError, ProviderOutputTransportError) as exc:
            raise _refuse(str(exc)) from None
        if (crc[0] & 0xFFFFFFFF, "sha256:" + digest.hexdigest()) != (row["crc32"], row["sha256"]):
            raise _refuse("provider_output_member_digest_mismatch")
        return row


def _open_descriptor(path: Path, root: Path, presign, opener) -> ProviderOutputMemberView:
    descriptor = None if path.is_symlink() else _json(path)
    if (not isinstance(descriptor, dict) or descriptor.get("schema_version") != SCHEMA
            or descriptor.get("view_digest") != _digest_or_none(descriptor, "view_digest")
            or descriptor.get("private_url_recorded") is not False
            or descriptor.get("evidence_root") != root.name or descriptor.get("archive_prefix") != ""
            or not isinstance(descriptor.get("member_index"), dict)):
        raise _refuse("provider_output_member_view_descriptor_invalid")
    parent = path.parent
    try:
        index_path = parent / safe_member_name(str(descriptor["member_index"].get("path")))
        receipt_path = parent / safe_member_name(str(descriptor.get("ingestion_receipt_path")))
    except ProviderOutputInventoryError:
        raise _refuse("provider_output_member_view_descriptor_invalid") from None
    try:
        index_record = _file_record(index_path, parent)
    except ProviderOutputMemberViewError:
        raise _refuse("provider_output_member_view_index_mismatch") from None
    index = _json(index_path)
    try:
        validate_member_index(index)
    except ProviderOutputMemberIndexError:
        raise _refuse("provider_output_member_view_index_mismatch") from None
    if (index_record != descriptor["member_index"] or index["index_digest"] != descriptor.get("member_index_digest")
            or index["archive"]["sha256"] != descriptor.get("archive_sha256")
            or index["archive"]["durable_reference"] is None
            or index["archive"]["durable_reference"] != descriptor.get("durable_reference")):
        raise _refuse("provider_output_member_view_index_mismatch")
    try:
        receipt_record = _file_record(receipt_path, parent)
    except ProviderOutputMemberViewError:
        raise _refuse("provider_output_member_view_receipt_mismatch") from None
    if receipt_record["sha256"] != descriptor.get("ingestion_receipt_sha256"):
        raise _refuse("provider_output_member_view_receipt_mismatch")
    return ProviderOutputMemberView(evidence_root=root.resolve(), descriptor=descriptor, index=index,
                                    presign=presign, opener=opener)


def open_member_view(path: str | Path, *, presign: Callable[[], str] | None = None,
                     opener: Callable | None = None) -> ProviderOutputMemberView | None:
    """The view covering ``path``, or None when no descriptor names one of its ancestors."""
    candidate = Path(path).absolute()
    for level, ancestor in enumerate((candidate, *candidate.parents)):
        if level > MAXIMUM_ANCESTORS or not ancestor.name:
            break
        descriptor = ancestor.parent / (ancestor.name + DESCRIPTOR_SUFFIX)
        if descriptor.exists() or descriptor.is_symlink():
            return _open_descriptor(descriptor, ancestor, presign, opener)
    return None


def plan_member_dispositions(archive_path: str | Path, contract: str) -> dict:
    """Bytes by class and by disposition under ``contract``, from the central directory alone."""
    needed = CONTRACTS[contract]
    path = Path(archive_path)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
    refusal = None
    try:
        check_archive_entries(infos, MAX_EXPANDED_BYTES)
    except (ProviderOutputInventoryError, ProviderOutputMemberIndexError) as exc:
        refusal = str(exc)
    files = [info for info in infos if not info.is_dir()]
    groups: dict[str, list] = {"materialized": [], "remote": []}
    for info in files:
        groups["materialized" if needed(info.filename) else "remote"].append(info)
    total = sum(info.file_size for info in files)
    bulk = sum(info.file_size for info in files if PurePosixPath(info.filename).suffix.lower() in BULK_EXTENSIONS)
    largest = sorted(groups["materialized"], key=lambda info: (-info.file_size, info.filename))[:10]
    return {
        "schema_version": PLAN_SCHEMA,
        "archive": {"name": path.name, "size_bytes": path.stat().st_size},
        "contract": contract,
        "members": len(infos),
        "files": len(files),
        "directories": len(infos) - len(files),
        "bytes": total,
        "bytes_by_class": {"bulk": bulk, "small": total - bulk},
        "dispositions": {name: {"members": len(rows), "bytes": sum(info.file_size for info in rows),
                                "compressed_bytes": sum(info.compress_size for info in rows)}
                         for name, rows in groups.items()},
        "largest_materialized": [{"path": info.filename, "size": info.file_size} for info in largest],
        "entry_rule_refusal": refusal,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan a provider archive's member dispositions (read-only).")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--archive", required=True, type=Path)
    plan.add_argument("--contract", required=True, choices=sorted(CONTRACTS))
    args = parser.parse_args(argv)
    try:
        summary = plan_member_dispositions(args.archive, args.contract)
    except (OSError, zipfile.BadZipFile):
        print("provider_output_member_view refused: provider_output_member_plan_archive_invalid",
              file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI
    sys.exit(main())
