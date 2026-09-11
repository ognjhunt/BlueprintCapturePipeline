"""Collect one admitted cloud provider ZIP without keeping a second ZIP copy.

The archive is hashed as a stream and then read by conditional HTTP ranges.
Received evidence remains diagnostic input pending the existing finalizer;
collection never qualifies episodes, images, provider closeout, or publication.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
import unicodedata
import uuid
import zipfile
import zlib

from .decision_evidence_contracts import canonical_digest
from .provider_output_disk_capacity import observe_provider_output_disk_capacity
from .provider_output_native_inventory import ProviderOutputInventoryError, safe_member_name, verify_native_inventory
from .provider_output_range_transport import ProviderOutputRangeReader, ProviderOutputTransportError
from .provider_signed_object_binding import signed_output_object_binding_sha256

SCHEMA = 'provider_output_ingestion_binding.v1'
FLOOR_BYTES = 8 * 1024**3
CHUNK_BYTES = 1024**2
MAX_ENTRIES = 200_000
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
    infos = archive.infolist()
    if not infos or len(infos) > MAX_ENTRIES:
        raise ProviderOutputIngestionError('provider_output_archive_entry_count_invalid')
    normalized, files, total = {}, {}, 0
    for info in infos:
        name = info.filename[:-1] if info.is_dir() else info.filename
        safe_member_name(name)
        if info.orig_filename != info.filename:
            raise ProviderOutputIngestionError('provider_output_archive_path_invalid')
        folded = unicodedata.normalize('NFC', name).casefold()
        if folded in normalized:
            raise ProviderOutputIngestionError('provider_output_archive_duplicate_path')
        normalized[folded] = info.is_dir()
        kind = stat.S_IFMT(info.external_attr >> 16)
        if (kind not in (0, stat.S_IFDIR, stat.S_IFREG)
                or info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                or info.file_size < 0 or info.compress_size < 0
                or (info.is_dir() and info.file_size != 0)
                or (not info.is_dir() and kind == stat.S_IFDIR)):
            raise ProviderOutputIngestionError('provider_output_archive_entry_type_invalid')
        if not info.is_dir():
            total += info.file_size
            if total > maximum_extracted_bytes:
                raise ProviderOutputIngestionError('provider_output_archive_expansion_cap_exceeded')
            files[name] = info
    for name in normalized:
        if any(parent.as_posix() in normalized and not normalized[parent.as_posix()]
               for parent in PurePosixPath(name).parents if parent.as_posix() != '.'):
            raise ProviderOutputIngestionError('provider_output_archive_file_directory_collision')
    return files, total


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


def _extract_member(archive, info, target, partial, *, root, reserve, disk_usage_provider):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial_size = partial.stat().st_size if partial.exists() else 0
    if partial.is_symlink() or (partial.exists() and (not partial.is_file() or partial.stat().st_nlink != 1)):
        raise ProviderOutputIngestionError('provider_output_partial_file_unsafe')
    if partial_size > info.file_size:
        raise ProviderOutputIngestionError('provider_output_partial_file_size_invalid')
    digest, crc, copied = hashlib.sha256(), 0, 0
    with archive.open(info) as incoming, partial.open('r+b' if partial.exists() else 'x+b') as sink:
        while True:
            chunk = incoming.read(CHUNK_BYTES)
            if not chunk:
                break
            if copied + len(chunk) > info.file_size:
                raise ProviderOutputIngestionError('provider_output_entry_size_exceeded')
            retained = min(len(chunk), max(0, partial_size - copied))
            if retained and sink.read(retained) != chunk[:retained]:
                raise ProviderOutputIngestionError('provider_output_partial_bytes_mismatch')
            if retained < len(chunk):
                _capacity(root, reserve + len(chunk) - retained, disk_usage_provider)
                sink.write(chunk[retained:])
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
            copied += len(chunk)
        sink.flush()
        os.fsync(sink.fileno())
    if copied != info.file_size or crc & 0xffffffff != info.CRC:
        raise ProviderOutputIngestionError('provider_output_entry_size_or_crc_mismatch')
    if target.exists():
        raise ProviderOutputIngestionError('provider_output_extraction_target_appeared')
    os.rename(partial, target)
    return {'size_bytes': copied, 'sha256': 'sha256:' + digest.hexdigest(), 'crc32': crc & 0xffffffff}


@contextmanager
def _output_lock(root, binding):
    if root.is_symlink():
        raise ProviderOutputIngestionError('provider_output_evidence_root_unsafe')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(path.name not in ('.ingestion', 'native') for path in root.iterdir()):
        raise ProviderOutputIngestionError('provider_output_evidence_root_not_owned')
    meta = root / '.ingestion'
    if meta.is_symlink() or (root / 'native').is_symlink():
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
            if _json(binding_path) != {'binding_digest': binding['binding_digest']}:
                raise ProviderOutputIngestionError('provider_output_resume_binding_mismatch')
        else:
            if (root / 'native').exists() and any((root / 'native').iterdir()):
                raise ProviderOutputIngestionError('provider_output_evidence_root_not_owned')
            _atomic_json(binding_path, {'binding_digest': binding['binding_digest']})
        (root / 'native').mkdir(exist_ok=True, mode=0o700)
        yield meta
    finally:
        os.close(descriptor)


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
                        partial = meta / (hashlib.sha256(name.encode()).hexdigest() + '.partial')
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
                        partial = meta / (hashlib.sha256(name.encode()).hexdigest() + '.partial')
                        record = _extract_member(archive, info, target, partial, root=root, reserve=reserve,
                                                 disk_usage_provider=disk_usage_provider)
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
            code = (str(exc) if isinstance(exc, (ProviderOutputIngestionError, ProviderOutputTransportError, ProviderOutputInventoryError))
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
        result['receipt_digest'] = canonical_digest(result, digest_field='receipt_digest')
        _atomic_json(meta / 'receipt.json', result)
    return result
