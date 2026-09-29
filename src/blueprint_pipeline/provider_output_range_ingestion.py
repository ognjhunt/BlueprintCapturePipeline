"""Collect one admitted cloud provider ZIP without keeping a second ZIP copy.

The archive is hashed as a stream and then read by conditional HTTP ranges.
Received evidence remains diagnostic input pending the existing finalizer;
collection never qualifies episodes, images, provider closeout, or publication.

``ingest_selected_members`` is the selective path: given a member index and a
selection bound to it, it fetches only the selected members of a durable CAS
archive, one range request each, and leaves every other member remote. Both
paths share the entry rules and the partial-file, journal and lock helpers.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import uuid
import zipfile
import zlib

from .decision_evidence_contracts import canonical_digest
from .provider_output_disk_capacity import observe_provider_output_disk_capacity
from .provider_output_member_index import (
    MemberInflater,
    ProviderOutputMemberIndexError,
    check_archive_entries,
    inflate_step_bytes,
    validate_member_index,
    validate_member_selection,
)
from .provider_output_native_inventory import (
    ProviderOutputInventoryError,
    index_member_records,
    safe_member_name,
    verify_native_inventory,
)
from .provider_output_range_transport import ProviderOutputRangeReader, ProviderOutputTransportError
from .provider_signed_object_binding import signed_output_object_binding_sha256

SCHEMA = 'provider_output_ingestion_binding.v1'
FLOOR_BYTES = 8 * 1024**3
CHUNK_BYTES = 1024**2
MAX_JSON_BYTES = 128 * 1024**2
MAX_JOURNAL_BYTES = 256 * 1024**2


class ProviderOutputIngestionError(ValueError):
    pass


def _hash_file(path):
    if path.is_symlink() or not path.is_file():
        raise ProviderOutputIngestionError('provider_output_local_file_unsafe')
    digest, crc, size = hashlib.sha256(), 0, 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
    return {'size_bytes': size, 'sha256': 'sha256:' + digest.hexdigest(), 'crc32': crc & 0xffffffff}


def _json(path, cap=MAX_JSON_BYTES):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > cap:
        raise ProviderOutputIngestionError('provider_output_json_missing_or_oversize')
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        raise ProviderOutputIngestionError('provider_output_json_invalid') from None
    if not isinstance(value, dict):
        raise ProviderOutputIngestionError('provider_output_json_invalid')
    return value


def _atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'))
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_ingestion_binding(binding, signed_get_url_file):
    """Bind the private GET to the already admitted canonical staging object."""
    if (not isinstance(binding, dict) or binding.get('schema_version') != SCHEMA
            or binding.get('binding_digest') != canonical_digest(binding, digest_field='binding_digest')
            or not re.fullmatch(r'[A-Za-z0-9._-]{1,192}', str(binding.get('run_id', '')))
            or not re.fullmatch(r'[1-9][0-9]{0,19}', str(binding.get('instance_id', '')))
            or not re.fullmatch(r'sha256:[0-9a-f]{64}', str(binding.get('runtime_inputs_digest', '')))):
        raise ProviderOutputIngestionError('provider_output_ingestion_binding_invalid')
    for field in ('maximum_archive_bytes', 'maximum_extracted_bytes', 'minimum_free_bytes'):
        if type(binding.get(field)) is not int or not 0 < binding[field] <= 1024**4:
            raise ProviderOutputIngestionError('provider_output_ingestion_capacity_bounds_invalid')
    if binding['minimum_free_bytes'] < FLOOR_BYTES:
        raise ProviderOutputIngestionError('provider_output_ingestion_floor_reduced')
    expected = binding.get('expected_archive_sha256')
    if expected is not None and not re.fullmatch(r'sha256:[0-9a-f]{64}', str(expected)):
        raise ProviderOutputIngestionError('provider_output_expected_archive_digest_invalid')
    for field in ('identity_document', 'result_document'):
        safe_member_name(binding.get(field))
    record = binding.get('staging_manifest') or {}
    staging_path = Path(str(record.get('path') or ''))
    observed = _hash_file(staging_path)
    if any(record.get(key) != observed[key] for key in ('sha256', 'size_bytes')):
        raise ProviderOutputIngestionError('provider_output_staging_manifest_digest_mismatch')
    staging = _json(staging_path, 4 * 1024**2)
    url_path = Path(signed_get_url_file)
    if (url_path.is_symlink() or not url_path.is_file() or url_path.stat().st_mode & 0o077
            or not 0 < url_path.stat().st_size <= 16384):
        raise ProviderOutputIngestionError('provider_output_signed_url_file_unsafe')
    url = url_path.read_text().strip()
    if 'sha256:' + hashlib.sha256(url.encode()).hexdigest() != binding.get('signed_get_url_sha256'):
        raise ProviderOutputIngestionError('provider_output_signed_url_digest_mismatch')
    try:
        object_identity = signed_output_object_binding_sha256(url, url)
    except ValueError:
        raise ProviderOutputIngestionError('provider_output_signed_url_invalid') from None
    if (staging.get('schema_version') != 'wam_provider_object_store_staging.v1'
            or staging.get('status') != 'completed' or staging.get('blockers') not in (None, [])
            or staging.get('output_key_run_unique') is not True
            or staging.get('output_url_object_binding_sha256') != object_identity):
        raise ProviderOutputIngestionError('provider_output_staging_authority_invalid')
    return url


def _archive_members(archive, maximum_extracted_bytes):
    # The rules are shared with the member index; the codes, and the path
    # refusal's inventory error type, are unchanged for this collector.
    try:
        return check_archive_entries(archive.infolist(), maximum_extracted_bytes)
    except ProviderOutputMemberIndexError as exc:
        raise ProviderOutputIngestionError(str(exc)) from None


def _safe_destination(root, relative):
    path = root / safe_member_name(relative)
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ProviderOutputIngestionError('provider_output_extraction_symlink_forbidden')
    if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
        raise ProviderOutputIngestionError('provider_output_extraction_destination_unsafe')
    return path


def _capacity(root, needed, provider):
    result = observe_provider_output_disk_capacity(destination_directory=root, required_free_bytes=needed,
        phase='streamed_provider_output_ingestion', schema_version='provider_output_ingestion_disk_capacity.v1',
        blocker_prefix='provider_output_ingestion', disk_usage_provider=provider)
    if result['status'] != 'ready':
        raise ProviderOutputIngestionError(result['blockers'][0])
    return result


def _journal_records(meta):
    records, total = {}, 0
    paths = sorted(meta.glob('members-*.jsonl'))
    if len(paths) > 256:
        raise ProviderOutputIngestionError('provider_output_resume_journal_count_exceeded')
    for path in paths:
        if path.is_symlink():
            raise ProviderOutputIngestionError('provider_output_resume_journal_unsafe')
        total += path.stat().st_size
        if total > MAX_JOURNAL_BYTES:
            raise ProviderOutputIngestionError('provider_output_resume_journal_size_exceeded')
        with path.open() as stream:
            for line in stream:
                # An interrupted final append is preserved and never trusted.
                if not line.endswith('\n'):
                    break
                try:
                    row = json.loads(line)
                except ValueError:
                    raise ProviderOutputIngestionError('provider_output_resume_journal_invalid') from None
                if row.get('record_digest') != canonical_digest(row, digest_field='record_digest'):
                    raise ProviderOutputIngestionError('provider_output_resume_journal_invalid')
                name = safe_member_name(row.get('relative_path'))
                if name in records and records[name] != row:
                    raise ProviderOutputIngestionError('provider_output_resume_journal_conflict')
                records[name] = row
    return records


def _append_journal(stream, name, record):
    value = {'relative_path': name, **record}
    value['record_digest'] = canonical_digest(value, digest_field='record_digest')
    stream.write(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
    stream.flush()
    os.fsync(stream.fileno())
    return value


def _partial_path(meta, name):
    return meta / (hashlib.sha256(name.encode()).hexdigest() + '.partial')


def _prepare_partial(target, partial, expected_size):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial_size = partial.stat().st_size if partial.exists() else 0
    if partial.is_symlink() or (partial.exists() and (not partial.is_file() or partial.stat().st_nlink != 1)):
        raise ProviderOutputIngestionError('provider_output_partial_file_unsafe')
    if partial_size > expected_size:
        raise ProviderOutputIngestionError('provider_output_partial_file_size_invalid')
    return partial_size


class _PartialWriter:
    """Write one member through its partial file, re-checking bytes a crash left there."""

    def __init__(self, sink, partial_size, expected_size, before_write):
        self._sink, self._partial_size, self._expected_size = sink, partial_size, expected_size
        self._before_write = before_write
        self._digest, self._crc, self._copied = hashlib.sha256(), 0, 0

    def write(self, chunk):
        if self._copied + len(chunk) > self._expected_size:
            raise ProviderOutputIngestionError('provider_output_entry_size_exceeded')
        retained = min(len(chunk), max(0, self._partial_size - self._copied))
        if retained and self._sink.read(retained) != chunk[:retained]:
            raise ProviderOutputIngestionError('provider_output_partial_bytes_mismatch')
        if retained < len(chunk):
            self._before_write(len(chunk) - retained)
            self._sink.write(chunk[retained:])
        self._digest.update(chunk)
        self._crc = zlib.crc32(chunk, self._crc)
        self._copied += len(chunk)

    def record(self):
        self._sink.flush()
        os.fsync(self._sink.fileno())
        return {'size_bytes': self._copied, 'sha256': 'sha256:' + self._digest.hexdigest(),
                'crc32': self._crc & 0xffffffff}


def _publish(partial, target):
    if target.exists():
        raise ProviderOutputIngestionError('provider_output_extraction_target_appeared')
    os.rename(partial, target)


def _extract_member(archive, info, target, partial, *, root, reserve, disk_usage_provider):
    partial_size = _prepare_partial(target, partial, info.file_size)
    with archive.open(info) as incoming, partial.open('r+b' if partial.exists() else 'x+b') as sink:
        writer = _PartialWriter(sink, partial_size, info.file_size,
                                lambda needed: _capacity(root, reserve + needed, disk_usage_provider))
        for chunk in iter(lambda: incoming.read(CHUNK_BYTES), b''):
            writer.write(chunk)
        record = writer.record()
    if record['size_bytes'] != info.file_size or record['crc32'] != info.CRC:
        raise ProviderOutputIngestionError('provider_output_entry_size_or_crc_mismatch')
    _publish(partial, target)
    return record


@contextmanager
def _locked_roots(members, meta, binding_record):
    """Hold the ingestion lock and bind ``meta`` to ``binding_record`` on first use."""
    if meta.is_symlink() or members.is_symlink():
        raise ProviderOutputIngestionError('provider_output_evidence_root_unsafe')
    meta.mkdir(exist_ok=True, mode=0o700)
    if any(path.is_symlink() for path in meta.iterdir()):
        raise ProviderOutputIngestionError('provider_output_resume_metadata_unsafe')
    descriptor = os.open(meta / 'lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ProviderOutputIngestionError('provider_output_ingestion_already_running') from None
        binding_path = meta / 'binding.json'
        if binding_path.exists():
            if _json(binding_path) != binding_record:
                raise ProviderOutputIngestionError('provider_output_resume_binding_mismatch')
        else:
            if members.exists() and any(members.iterdir()):
                raise ProviderOutputIngestionError('provider_output_evidence_root_not_owned')
            _atomic_json(binding_path, binding_record)
        members.mkdir(exist_ok=True, mode=0o700)
        yield meta
    finally:
        os.close(descriptor)


@contextmanager
def _output_lock(root, binding):
    if root.is_symlink():
        raise ProviderOutputIngestionError('provider_output_evidence_root_unsafe')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(path.name not in ('.ingestion', 'native') for path in root.iterdir()):
        raise ProviderOutputIngestionError('provider_output_evidence_root_not_owned')
    with _locked_roots(root / 'native', root / '.ingestion', {'binding_digest': binding['binding_digest']}) as meta:
        yield meta


def _record_failure(meta, result, exc):
    code = (str(exc) if isinstance(exc, (ProviderOutputIngestionError, ProviderOutputTransportError,
                                          ProviderOutputInventoryError, ProviderOutputMemberIndexError))
            else 'provider_output_archive_crc_or_structure_invalid' if isinstance(exc, zipfile.BadZipFile)
            else 'provider_output_archive_or_io_failed')
    if not re.fullmatch(r'[a-z0-9_]+', code):
        code = 'provider_output_ingestion_failed'
    result.update(status='not_ready' if code == 'provider_output_not_ready' else 'blocked',
                  blockers=[code], partial_evidence_retained=True, failure_type=type(exc).__name__)
    failure_path = meta / 'failures.jsonl'
    if not failure_path.exists() or failure_path.stat().st_size < 4 * 1024**2:
        with failure_path.open('a') as stream:
            stream.write(json.dumps({'time_ns': time.time_ns(), 'code': code, 'failure_type': type(exc).__name__}) + '\n')


def _write_receipt(meta, result):
    result['receipt_digest'] = canonical_digest(result, digest_field='receipt_digest')
    _atomic_json(meta / 'receipt.json', result)


def ingest_provider_output(*, binding: dict, signed_get_url_file: str | Path, output_root: str | Path,
                           opener=None, disk_usage_provider=None, deadline_seconds=3600) -> dict:
    """Ingest a single sealed binding; safe to retry against the same evidence root."""
    url = validate_ingestion_binding(binding, signed_get_url_file)
    root = Path(output_root).absolute()
    result = {'schema_version': 'provider_output_ingestion_receipt.v1', 'run_id': binding['run_id'],
        'instance_id': str(binding['instance_id']), 'runtime_inputs_digest': binding['runtime_inputs_digest'],
        'binding_digest': binding['binding_digest'], 'evidence_root': str(root / 'native'),
        'local_archive_copy_created': False, 'private_url_recorded': False,
        'scientific_qualification_performed': False, 'publication_performed': False}
    with _output_lock(root, binding) as meta:
        try:
            with ProviderOutputRangeReader(url, maximum_archive_bytes=binding['maximum_archive_bytes'],
                    deadline_seconds=deadline_seconds, opener=opener) as remote, zipfile.ZipFile(remote) as archive:
                entries, total = _archive_members(archive, binding['maximum_extracted_bytes'])
                source_path = meta / 'source.json'
                previous = _json(source_path) if source_path.exists() else None
                if previous and (previous.get('source_digest') != canonical_digest(previous, digest_field='source_digest')
                                 or previous.get('remote_identity') != remote.identity
                                 or previous.get('binding_digest') != binding['binding_digest']
                                 or not re.fullmatch(r'sha256:[0-9a-f]{64}', str(previous.get('archive_sha256', '')))):
                    raise ProviderOutputIngestionError('provider_output_resume_remote_identity_mismatch')
                records = _journal_records(meta)
                if set(records) - set(entries):
                    raise ProviderOutputIngestionError('provider_output_resume_inventory_changed')
                for path in (root / 'native').rglob('*'):
                    if path.is_symlink() or (path.is_file() and path.relative_to(root / 'native').as_posix() not in entries):
                        raise ProviderOutputIngestionError('provider_output_resume_inventory_changed')
                verified, remaining = {}, total
                for name, info in entries.items():
                    target = _safe_destination(root / 'native', name)
                    if target.exists():
                        actual = _hash_file(target)
                        if actual['size_bytes'] != info.file_size or actual['crc32'] != info.CRC:
                            raise ProviderOutputIngestionError('provider_output_resume_file_changed')
                        prior = records.get(name)
                        if prior and any(prior.get(key) != actual[key] for key in actual):
                            raise ProviderOutputIngestionError('provider_output_resume_file_changed')
                        if not prior:
                            # A crash after rename but before journal append: compare
                            # to the pinned remote member before adopting this file.
                            digest = hashlib.sha256()
                            with archive.open(info) as stream:
                                for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
                                    digest.update(chunk)
                            if actual['sha256'] != 'sha256:' + digest.hexdigest():
                                raise ProviderOutputIngestionError('provider_output_resume_file_changed')
                        verified[name] = actual
                        remaining -= info.file_size
                    else:
                        partial = _partial_path(meta, name)
                        if partial.exists() and not partial.is_symlink():
                            remaining -= min(partial.stat().st_size, info.file_size)
                reserve = binding['minimum_free_bytes'] + max(16 * 1024**2, len(entries) * 1024)
                result['disk_capacity'] = _capacity(root, reserve + remaining, disk_usage_provider)
                archive_digest = previous.get('archive_sha256') if previous else None
                if archive_digest is None:
                    archive_digest = remote.archive_sha256()
                if binding.get('expected_archive_sha256') not in (None, archive_digest):
                    raise ProviderOutputIngestionError('provider_output_archive_digest_mismatch')
                source = {'remote_identity': remote.identity, 'archive_sha256': archive_digest,
                          'binding_digest': binding['binding_digest']}
                source['source_digest'] = canonical_digest(source, digest_field='source_digest')
                _atomic_json(source_path, source)
                resumed_count = len(verified)
                with (meta / ('members-' + uuid.uuid4().hex + '.jsonl')).open('x') as journal:
                    for name, info in entries.items():
                        if name in verified:
                            if name not in records:
                                _append_journal(journal, name, verified[name])
                            continue
                        target = _safe_destination(root / 'native', name)
                        record = _extract_member(archive, info, target, _partial_path(meta, name), root=root,
                                                 reserve=reserve, disk_usage_provider=disk_usage_provider)
                        _append_journal(journal, name, record)
                        verified[name] = record
                native = verify_native_inventory(root / 'native', binding, verified)
                inventory = {'schema_version': 'provider_output_received_members.v1', 'members': verified,
                             'archive_sha256': archive_digest, 'binding_digest': binding['binding_digest']}
                inventory['inventory_digest'] = canonical_digest(inventory, digest_field='inventory_digest')
                _atomic_json(meta / 'member_inventory.json', inventory)
                result.update(status='collected_pending_finalization', archive_sha256=archive_digest,
                    archive_size_bytes=remote.identity['size_bytes'], remote_identity=remote.identity,
                    extracted_size_bytes=total, verified_member_count=len(verified), resumed_member_count=resumed_count,
                    transferred_bytes=remote.transferred_bytes, http_request_count=remote.request_count,
                    member_inventory_digest=inventory['inventory_digest'], native_inventory=native, blockers=[])
        except Exception as exc:
            _record_failure(meta, result, exc)
        _write_receipt(meta, result)
    return result


SELECTED_BINDING_SCHEMA = 'provider_output_selected_member_binding.v1'
SELECTED_RECEIPT_SCHEMA = 'provider_output_selected_member_receipt.v1'
CAS_REFERENCE_SCHEMA = 'task_evaluation_scene_artifact_reference.v1'


@dataclass(frozen=True)
class CasArchiveSource:
    """A durable B2 archive reference plus a callable issuing a short-lived GET URL.

    The URL is transport authority only: it is requested when a reader opens,
    held in memory by that reader alone, and never written to a file, journal,
    receipt or log.
    """

    reference: Mapping[str, Any]
    presign: Callable[[], str] = dataclass_field(repr=False)
    opener: Callable | None = dataclass_field(default=None, repr=False)
    block_bytes: int = 8 * 1024**2
    deadline_seconds: float = 3600

    def durable_reference(self) -> dict:
        """Secret-free facts naming the durable object, after checking the reference."""
        reference = self.reference if isinstance(self.reference, Mapping) else {}
        uri, digest, size = str(reference.get('uri') or ''), str(reference.get('digest') or ''), reference.get('size_bytes')
        if (reference.get('schema_version') != CAS_REFERENCE_SCHEMA or reference.get('status') != 'remote_verified'
                or reference.get('content_addressed_key') is not True
                or reference.get('remote_identity_verified') is not True
                or reference.get('full_byte_service_account_readback_passed') is not True
                or not re.fullmatch(r'sha256:[0-9a-f]{64}', digest)
                or not re.fullmatch(r's3://[A-Za-z0-9][A-Za-z0-9.-]{0,254}/[A-Za-z0-9._/-]{1,1024}', uri)
                or f"/sha256/{digest.removeprefix('sha256:')}/" not in uri
                or type(size) is not int or size <= 0):
            raise ProviderOutputIngestionError('provider_output_cas_reference_invalid')
        return {'uri': uri, 'digest': digest, 'size_bytes': size}

    def open(self, maximum_archive_bytes: int) -> ProviderOutputRangeReader:
        try:
            url = self.presign()
            if not isinstance(url, str) or not url:
                raise ValueError('presign returned no URL')
            return ProviderOutputRangeReader(url, maximum_archive_bytes=maximum_archive_bytes,
                                             block_bytes=self.block_bytes, deadline_seconds=self.deadline_seconds,
                                             opener=self.opener)
        except ProviderOutputTransportError:
            raise
        except Exception:
            raise ProviderOutputIngestionError('provider_output_cas_presign_invalid') from None


class _RunJournal:
    """One run's journal file, created by its first row.

    A run that records nothing (a refusal before any write, or a resume with
    nothing left to do) leaves no file behind, so such runs never count
    towards the resume's journal-file cap.
    """

    def __init__(self, meta):
        self._meta, self._stream = meta, None

    def append(self, name, record):
        if self._stream is None:
            self._stream = (self._meta / ('members-' + uuid.uuid4().hex + '.jsonl')).open('x')
        _append_journal(self._stream, name, record)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._stream is not None:
            self._stream.close()
        return False


def _reserve(reserve, needed):
    try:
        reserve(needed)
    except ProviderOutputIngestionError:
        raise
    except Exception as exc:
        code = str(exc)
        raise ProviderOutputIngestionError(code if re.fullmatch(r'[a-z0-9_]{1,128}', code)
                                           else 'provider_output_disk_reservation_refused') from None


def _materialize_member(remote, row, target, partial, *, step_bytes):
    """Fetch one member's record data with one range request; place it only once verified."""
    partial_size = _prepare_partial(target, partial, row['size'])
    if partial.exists():
        # A crash between the read-only chmod and the rename leaves a verified
        # 0440 partial; it is re-checked byte for byte below like any other.
        os.chmod(partial, 0o600)
    try:
        with partial.open('r+b' if partial.exists() else 'x+b') as sink:
            writer = _PartialWriter(sink, partial_size, row['size'], lambda needed: None)
            inflater = MemberInflater(row['method'], row['size'], writer.write, step_bytes=step_bytes)
            if row['compressed_size']:
                remote.stream_to(inflater.feed, start=row['data_offset'],
                                 end=row['data_offset'] + row['compressed_size'])
            inflater.finish()
            record = writer.record()
        if record != {'size_bytes': row['size'], 'sha256': row['sha256'], 'crc32': row['crc32']}:
            raise ProviderOutputIngestionError('provider_output_member_digest_mismatch')
    except (ProviderOutputIngestionError, ProviderOutputMemberIndexError):
        # Bytes that decode wrongly or disagree with the index are never left
        # for a resume to trust; a transport failure keeps its partial prefix.
        partial.unlink(missing_ok=True)
        raise
    os.chmod(partial, 0o440)
    try:
        _publish(partial, target)
    except OSError as exc:
        # Distinct mounts of one filesystem share st_dev but still refuse a rename.
        if exc.errno == errno.EXDEV:
            raise ProviderOutputIngestionError('provider_output_roots_cross_device') from None
        raise
    return record


def _device(path):
    return os.stat(path).st_dev


def _native_inventory(root: Path, binding: Mapping, index: Mapping) -> dict:
    """The native inventory checked against the index (design 4): an outcome, never a blocker."""
    try:
        return {'status': 'verified', **verify_native_inventory(root, binding, index_member_records(index))}
    except ProviderOutputInventoryError as exc:
        return {'status': 'failed', 'code': str(exc)}
    except Exception as exc:  # noqa: BLE001 - recorded, never allowed to block the ingestion
        return {'status': 'failed', 'code': f'provider_output_native_inventory_failed:{type(exc).__name__}'}


def ingest_selected_members(*, source: CasArchiveSource, index: Mapping, selection: Mapping,
                            members_root: str | Path, metadata_root: str | Path,
                            reserve: Callable[[int], object], disk_usage_provider=None,
                            native_inventory_binding: Mapping | None = None) -> dict:
    """Materialize only a selection's members from a pinned CAS archive; safe to resume.

    Each selected member is one range request for its record data under the
    reader's pinned ETag, inflated in bounded steps and checked against the
    index's CRC-32 and SHA-256 before it is renamed into place read-only (0440).
    Unselected members are never requested or written. ``disk_usage_provider``,
    when given, is sampled before and after each member write.

    ``reserve(bytes)`` is called before each member write with the total still
    outstanding for the run: that member and every later one, less what their
    partial files already hold. Each call supersedes the previous one, so a
    caller keeps a single reservation and resizes it. A refusal it raises stops
    writing and leaves the journal resumable.

    With ``native_inventory_binding`` ({identity_document, result_document,
    run_id, runtime_inputs_digest}) a materialized run also checks the native
    inventory against the index, digests without bytes
    (``verify_native_inventory``), and records the outcome in the receipt as
    ``native_inventory`` -- ``verified`` with its counts, or ``failed`` with
    its code -- never as a blocker (design 4).

    On resume a verified member is never fetched again. A member interrupted
    mid-transfer is fetched again from its data offset with one range request;
    the bytes its partial file holds are compared, not trusted or rewritten.
    Only runs that record a member create a journal file, and a resume refuses
    more than 256 of them.

    Refusals found before the run starts, including taking the lock and
    checking the binding, raise ``ProviderOutputIngestionError`` and write no
    receipt: an invalid index or selection or one bound to another index, a
    CAS reference that is invalid or names another archive, overlapping roots
    or roots on different devices, symlinked roots or a root holding files this
    binding did not write, a run already in progress, and a metadata root bound
    to another index or selection (``provider_output_resume_binding_mismatch``).
    Every later failure returns a receipt, also written to ``receipt.json``,
    with status ``blocked`` (``not_ready`` for a missing object): transport and
    presign failures, ``reserve`` refusals, digest mismatches, a changed remote,
    journal or member tree, and I/O errors.
    """
    try:
        validate_member_index(index)
        selected = validate_member_selection(selection, index)
    except ProviderOutputMemberIndexError as exc:
        raise ProviderOutputIngestionError(str(exc)) from None
    if not isinstance(source, CasArchiveSource) or not callable(reserve):
        raise ProviderOutputIngestionError('provider_output_selected_ingestion_arguments_invalid')
    durable = source.durable_reference()
    recorded = index['archive']['durable_reference']
    if (durable['digest'] != index['archive']['sha256'] or durable['size_bytes'] != index['archive']['size']
            or (recorded is not None and (recorded.get('uri'), recorded.get('digest')) != (durable['uri'], durable['digest']))):
        raise ProviderOutputIngestionError('provider_output_cas_reference_mismatch')
    members, meta = Path(members_root).absolute(), Path(metadata_root).absolute()
    if members == meta or members in meta.parents or meta in members.parents:
        raise ProviderOutputIngestionError('provider_output_selected_roots_invalid')
    if meta.is_symlink() or members.is_symlink():
        # Before anything reads or creates them: a dangling link would make mkdir raise untyped.
        raise ProviderOutputIngestionError('provider_output_evidence_root_unsafe')
    if meta.is_dir() and not (meta / 'binding.json').exists() and any(path.name != 'lock' for path in meta.iterdir()):
        raise ProviderOutputIngestionError('provider_output_evidence_root_not_owned')
    binding = {'schema_version': SELECTED_BINDING_SCHEMA, 'member_index_digest': index['index_digest'],
               'archive_sha256': index['archive']['sha256'], 'durable_reference': durable,
               'selection_schema_version': selection['schema_version'],
               'selection_version': selection['selection_version'], 'selection_digest': selection['selection_digest']}
    binding['binding_digest'] = canonical_digest(binding, digest_field='binding_digest')
    chosen = {row['path'] for row in selected}
    files = [row for row in index['members'] if row['kind'] == 'file']
    result = {'schema_version': SELECTED_RECEIPT_SCHEMA, 'binding_digest': binding['binding_digest'],
              'member_index_digest': index['index_digest'], 'archive_sha256': index['archive']['sha256'],
              'selection_version': selection['selection_version'], 'selection_digest': selection['selection_digest'],
              'durable_reference': durable, 'members_root': str(members),
              'local_archive_copy_created': False, 'private_url_recorded': False}
    samples = []

    def sample():
        if disk_usage_provider is not None:
            free = getattr(disk_usage_provider(members), 'free', None)
            if type(free) is int:
                samples.append(free)

    for directory in (meta, members):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if _device(members) != _device(meta):
        # Partials live under the metadata root and are renamed into the
        # members root; a rename cannot cross filesystems.
        raise ProviderOutputIngestionError('provider_output_roots_cross_device')
    with _locked_roots(members, meta, binding):
        try:
            with source.open(index['archive']['size']) as remote:
                if remote.identity['size_bytes'] != index['archive']['size']:
                    raise ProviderOutputIngestionError('provider_output_remote_size_mismatch')
                source_path = meta / 'source.json'
                previous = _json(source_path) if source_path.exists() else None
                if previous and (previous.get('source_digest') != canonical_digest(previous, digest_field='source_digest')
                                 or previous.get('remote_identity') != remote.identity
                                 or previous.get('binding_digest') != binding['binding_digest']):
                    raise ProviderOutputIngestionError('provider_output_resume_remote_identity_mismatch')
                records = _journal_records(meta)
                if set(records) - chosen:
                    raise ProviderOutputIngestionError('provider_output_resume_inventory_changed')
                for path in members.rglob('*'):
                    if path.is_symlink() or (path.is_file() and path.relative_to(members).as_posix() not in chosen):
                        raise ProviderOutputIngestionError('provider_output_resume_inventory_changed')
                verified, pending = {}, []
                for row in selected:
                    target = _safe_destination(members, row['path'])
                    if not target.exists():
                        pending.append(row)
                        continue
                    # Already verified (or renamed just before a crash): the index
                    # digest decides, so the member is never fetched again.
                    actual, prior = _hash_file(target), records.get(row['path'])
                    if (actual != {'size_bytes': row['size'], 'sha256': row['sha256'], 'crc32': row['crc32']}
                            or (prior and any(prior.get(key) != actual[key] for key in actual))):
                        raise ProviderOutputIngestionError('provider_output_resume_file_changed')
                    verified[row['path']] = actual
                if previous is None:
                    record = {'remote_identity': remote.identity, 'archive_sha256': index['archive']['sha256'],
                              'binding_digest': binding['binding_digest']}
                    record['source_digest'] = canonical_digest(record, digest_field='source_digest')
                    _atomic_json(source_path, record)
                resumed = len(verified)
                step_bytes = inflate_step_bytes(source.block_bytes)
                with _RunJournal(meta) as journal:
                    for path, record in verified.items():
                        if path not in records:
                            journal.append(path, record)
                    unwritten = {}
                    for row in pending:
                        partial = _partial_path(meta, row['path'])
                        kept = partial.stat().st_size if partial.exists() and not partial.is_symlink() else 0
                        unwritten[row['path']] = row['size'] - min(kept, row['size'])
                    needed = sum(unwritten.values())
                    for row in pending:
                        _reserve(reserve, needed)
                        sample()
                        record = _materialize_member(remote, row, _safe_destination(members, row['path']),
                                                     _partial_path(meta, row['path']), step_bytes=step_bytes)
                        journal.append(row['path'], record)
                        verified[row['path']] = record
                        needed -= unwritten[row['path']]
                        sample()
                result.update(
                    status='materialized', remote_identity=remote.identity,
                    members=[{'path': row['path'],
                              'disposition': 'materialized' if row['path'] in chosen else 'remote',
                              'size': row['size'], 'sha256': row['sha256'], 'crc32': row['crc32']}
                             for row in files],
                    materialized_member_count=len(selected), remote_member_count=len(files) - len(selected),
                    materialized_bytes=sum(row['size'] for row in selected),
                    remote_bytes=sum(row['size'] for row in files if row['path'] not in chosen),
                    resumed_member_count=resumed, transferred_bytes=remote.transferred_bytes,
                    http_request_count=remote.request_count, blockers=[])
                if native_inventory_binding is not None:
                    result['native_inventory'] = _native_inventory(members, native_inventory_binding, index)
        except Exception as exc:
            _record_failure(meta, result, exc)
        result.update(disk_usage_sample_count=len(samples),
                      minimum_observed_free_bytes=min(samples) if samples else None)
        _write_receipt(meta, result)
    return result
