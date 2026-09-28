"""Index a provider-output ZIP by member digest and byte range in one pass.

Provider archives used to be downloaded whole to the control-plane host and
fully extracted. This module is the library half of "stream, don't download":
it reads an archive held in object storage and records, for every member, the
exact byte range of its record and the digests of its bytes, so a later step
can fetch only the members a consumer needs, with one range request each.

The pass (``build_member_index``) reads the end records and the central
directory by range under the pinned ETag or generation, then makes exactly one
pinned GET of the whole object. During that single stream it hashes the
archive, re-parses every local header and checks it against the directory,
inflates each member in bounded ``max_length`` steps while computing its
SHA-256 and CRC-32, and requires the records to tile ``[0, directory)``
exactly. It writes nothing to disk. It holds one transport block plus bounded
inflate steps (under two blocks for blocks of 16 KiB and up), however large
the archive. A 412 or a changed ETag between the directory read and the
stream is a refusal, never a retry against the new object.

``provider_output_member_index.v1`` holds archive facts only: the archive's
sha256, size, pinned etag/generation and ``durable_reference`` (null until a
later change promotes the archive to B2); the limits applied; per member its
path, kind, size, compressed_size, method, crc32, sha256, mode and
local-header, data and record-end offsets; totals with bytes by class
(``bulk`` uses the terminal-payload retention extension list);
``private_url_recorded: false``; and ``index_digest`` over the canonical JSON
without that field. Which members a consumer materializes is not an archive
fact. It belongs in a separate, versioned selection bound to
``index_digest``, never in the index.

Refusals raise ``ProviderOutputMemberIndexError`` whose message is a stable
code: ``provider_output_archive_*`` for archive bytes and structure (the
vocabulary the ingester already uses), ``provider_output_member_index_*`` for
the document and its arguments, and the transport's own ``provider_output_*``
codes unchanged.

Entry rules. ``check_archive_entries`` is the ingester's entry-safety rule set,
moved here so both paths apply one definition: a relative POSIX name with no
``\\``, ``:``, control character, empty, ``.`` or ``..`` component, no
component over 255 bytes and no name over 4096 bytes; the name exactly as
stored (no NUL truncation, no differing Unicode-path extra field); no NFC
case-folded duplicate; only unencrypted, stored or deflated regular files and
directories; no file that is also a directory's parent; and a cap on total
expanded bytes. The Vast scene-configuration lane
(``task_evaluation_scene_configuration_vast._extract_provider_output``) still
runs ``zipfile.extractall`` behind looser checks, unchanged here. Its checks
accept names containing ``:`` or ``\\``, control characters, empty or ``.``
components, ``a/../b`` (which resolves inside the root), components over 255
bytes (which then fail at the filesystem), NUL-truncated or
Unicode-path-overridden names, and case-folding or NFC duplicates: only exact
duplicates are refused, and on a case-insensitive filesystem the later member
overwrites the earlier. It does not refuse encryption, bzip2/lzma members or
special file types before extracting, bounds expansion by a ratio of the
declared archive ceiling, and restores archive permission bits. An index built
here can therefore refuse an archive that lane would extract.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import unicodedata
import zipfile
import zlib

from .decision_evidence_contracts import canonical_digest
from .provider_output_native_inventory import ProviderOutputInventoryError, safe_member_name
from .provider_output_range_transport import ProviderOutputTransportError

SCHEMA = 'provider_output_member_index.v1'
SUMMARY_SCHEMA = 'provider_output_member_index_summary.v1'
DEFAULT_MAXIMUM_MEMBERS = 10_000
MAX_ARCHIVE_ENTRIES = 200_000
MAX_EXPANDED_BYTES = 1024**4
INFLATE_STEP_BYTES = 256 * 1024
# The same list as the terminal scene-payload retention's binary payloads
# (``_BINARY_PAYLOAD_SUFFIXES``); a test keeps the two identical.
BULK_EXTENSIONS = frozenset({
    '.bin', '.jpeg', '.jpg', '.mp4', '.npy', '.npz', '.ply', '.png', '.pt', '.pth',
    '.splat', '.usd', '.usda', '.usdc', '.usdz', '.zip',
})

_LOCAL = struct.Struct('<4sHHHHHIIIHH')
_CENTRAL = struct.Struct('<4sBBBBHHHHIIIHHHHHII')
_END = struct.Struct('<4sHHHHIIH')
_END64 = struct.Struct('<4sQHHIIQQQQ')
_LOCATOR64 = struct.Struct('<4sIQI')
_TAIL_BYTES = _END.size + 0xFFFF + _LOCATOR64.size + _END64.size
_MAX32 = 0xFFFFFFFF
_ENCRYPTION_FLAGS = 0x0001 | 0x0040 | 0x2000  # traditional, strong, masked directory
_DESCRIPTOR_FLAG = 0x0008
_METHODS = {zipfile.ZIP_STORED: 'stored', zipfile.ZIP_DEFLATED: 'deflate'}
_SHA256 = re.compile(r'sha256:[0-9a-f]{64}')
_MEMBER_KEYS = frozenset({
    'path', 'kind', 'size', 'compressed_size', 'method', 'crc32', 'sha256', 'mode',
    'local_header_offset', 'data_offset', 'record_end_offset',
})


class ProviderOutputMemberIndexError(ValueError):
    """A typed, secret-free refusal; the message is the stable code."""


def check_archive_entries(infos: Sequence, maximum_extracted_bytes: int):
    """Apply the shared entry-safety rules to ``zipfile.ZipInfo``-like records.

    Returns ``({name: info}, total_bytes)`` for regular files. Path refusals
    raise ``ProviderOutputInventoryError`` from ``safe_member_name``; every
    other refusal raises ``ProviderOutputMemberIndexError``.
    """
    if not infos or len(infos) > MAX_ARCHIVE_ENTRIES:
        raise ProviderOutputMemberIndexError('provider_output_archive_entry_count_invalid')
    normalized, files, total = {}, {}, 0
    for info in infos:
        name = info.filename[:-1] if info.is_dir() else info.filename
        safe_member_name(name)
        if info.orig_filename != info.filename:
            raise ProviderOutputMemberIndexError('provider_output_archive_path_invalid')
        folded = unicodedata.normalize('NFC', name).casefold()
        if folded in normalized:
            raise ProviderOutputMemberIndexError('provider_output_archive_duplicate_path')
        normalized[folded] = info.is_dir()
        kind = stat.S_IFMT(info.external_attr >> 16)
        if (kind not in (0, stat.S_IFDIR, stat.S_IFREG)
                or info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                or info.file_size < 0 or info.compress_size < 0
                or (info.is_dir() and info.file_size != 0)
                or (not info.is_dir() and kind == stat.S_IFDIR)):
            raise ProviderOutputMemberIndexError('provider_output_archive_entry_type_invalid')
        if not info.is_dir():
            total += info.file_size
            if total > maximum_extracted_bytes:
                raise ProviderOutputMemberIndexError('provider_output_archive_expansion_cap_exceeded')
            files[name] = info
    for name in normalized:
        if any(parent.as_posix() in normalized and not normalized[parent.as_posix()]
               for parent in PurePosixPath(name).parents if parent.as_posix() != '.'):
            raise ProviderOutputMemberIndexError('provider_output_archive_file_directory_collision')
    return files, total


class MemberInflater:
    """Decode one member's record data in bounded steps.

    ``emit`` receives the decoded bytes in order and must not keep them; a
    deflate piece is never larger than ``step_bytes``. A member may not decode
    past its declared size, and a deflate stream must end exactly where its
    record data ends.
    """

    def __init__(self, method: str, size: int, emit: Callable, *, step_bytes: int = INFLATE_STEP_BYTES):
        if (method not in ('stored', 'deflate') or type(size) is not int or size < 0
                or type(step_bytes) is not int or step_bytes < 1):
            raise ProviderOutputMemberIndexError('provider_output_member_index_inflate_invalid')
        self._decoder = zlib.decompressobj(-zlib.MAX_WBITS) if method == 'deflate' else None
        self._size, self._emit, self._step = size, emit, step_bytes
        self.produced = 0

    def feed(self, data) -> None:
        if self._decoder is None:
            self._output(data)
            return
        view = memoryview(data)
        try:
            for start in range(0, len(view), self._step):
                if self._decoder.eof:
                    raise ProviderOutputMemberIndexError('provider_output_archive_deflate_end_invalid')
                self._output(self._decoder.decompress(view[start:start + self._step], self._step))
                while self._decoder.unconsumed_tail:
                    self._output(self._decoder.decompress(self._decoder.unconsumed_tail, self._step))
                if self._decoder.unused_data:
                    raise ProviderOutputMemberIndexError('provider_output_archive_deflate_end_invalid')
        except zlib.error:
            raise ProviderOutputMemberIndexError('provider_output_archive_deflate_invalid') from None

    def finish(self) -> None:
        if self._decoder is not None:
            try:
                while not self._decoder.eof:
                    output = self._decoder.decompress(b'', self._step)
                    if not output:
                        break
                    self._output(output)
            except zlib.error:
                raise ProviderOutputMemberIndexError('provider_output_archive_deflate_invalid') from None
            if not self._decoder.eof or self._decoder.unused_data:
                raise ProviderOutputMemberIndexError('provider_output_archive_deflate_end_invalid')
        if self.produced != self._size:
            raise ProviderOutputMemberIndexError('provider_output_archive_member_size_mismatch')

    def _output(self, data) -> None:
        if not len(data):
            return
        self.produced += len(data)
        if self.produced > self._size:
            raise ProviderOutputMemberIndexError('provider_output_archive_member_size_mismatch')
        self._emit(data)


class _Entry:
    """One central-directory record, shaped like ``zipfile.ZipInfo``."""

    __slots__ = ('raw_name', 'orig_filename', 'filename', 'flag_bits', 'compress_type', 'CRC',
                 'compress_size', 'file_size', 'header_offset', 'external_attr')

    def is_dir(self):
        return self.filename.endswith('/')


@dataclass(frozen=True)
class _Layout:
    directory_offset: int
    directory_size: int
    entries: int
    zip64: bool


def _refusal(code):
    return ProviderOutputMemberIndexError(code)


def _extra_fields(extra, code):
    """Split an extra field as ``zipfile`` does; a record past the end is corrupt."""
    fields, position = {}, 0
    while len(extra) - position >= 4:
        kind, length = struct.unpack_from('<HH', extra, position)
        if position + 4 + length > len(extra) or (kind in (0x0001, 0x7075) and kind in fields):
            raise _refusal(code)
        fields.setdefault(kind, bytes(extra[position + 4:position + 4 + length]))
        position += 4 + length
    return fields


def _zip64_values(fields, values):
    """Resolve 32-bit fields set to 0xFFFFFFFF from the ZIP64 extra, in order."""
    data = fields.get(0x0001)
    if data is None:
        return list(values), False
    resolved, position = [], 0
    for value in values:
        if value == _MAX32:
            if position + 8 > len(data):
                raise _refusal('provider_output_archive_zip64_invalid')
            value = struct.unpack_from('<Q', data, position)[0]
            position += 8
        resolved.append(value)
    return resolved, True


def _stored_names(raw_name, flags, fields):
    """The name as ``zipfile`` would report it: (as stored, as it would be used)."""
    try:
        original = raw_name.decode('utf-8' if flags & 0x0800 else 'cp437')
    except UnicodeDecodeError:
        raise _refusal('provider_output_archive_path_invalid') from None
    name = original.split('\0', 1)[0]
    unicode_path = fields.get(0x7075)
    if unicode_path is not None:
        if len(unicode_path) < 5:
            raise _refusal('provider_output_archive_directory_invalid')
        version, name_crc = struct.unpack_from('<BL', unicode_path)
        if version == 1 and name_crc == zlib.crc32(raw_name):
            try:
                override = unicode_path[5:].decode('utf-8')
            except UnicodeDecodeError:
                raise _refusal('provider_output_archive_path_invalid') from None
            if override:
                name = override.split('\0', 1)[0]
    return original, name


def _central_entry(record):
    (_, _made, _system, _needed, _reserved, flags, method, _time, _date, crc, csize, usize,
     name_length, extra_length, _comment_length, disk, _internal, external,
     offset) = _CENTRAL.unpack_from(record)
    raw_name = bytes(record[_CENTRAL.size:_CENTRAL.size + name_length])
    extra = record[_CENTRAL.size + name_length:_CENTRAL.size + name_length + extra_length]
    if disk != 0:
        raise _refusal('provider_output_archive_directory_invalid')
    fields = _extra_fields(extra, 'provider_output_archive_directory_invalid')
    (usize, csize, offset), _ = _zip64_values(fields, (usize, csize, offset))
    entry = _Entry()
    entry.raw_name = raw_name
    entry.orig_filename, entry.filename = _stored_names(raw_name, flags, fields)
    entry.flag_bits, entry.compress_type, entry.CRC = flags, method, crc
    entry.compress_size, entry.file_size, entry.header_offset = csize, usize, offset
    entry.external_attr = external
    return entry


class _DirectoryReader:
    """Parse central-directory records from one range stream, a record at a time."""

    def __init__(self, directory_size, entries):
        self._size, self._count = directory_size, entries
        self._consumed, self._record, self._need = 0, bytearray(), _CENTRAL.size
        self.entries = []
        self.digest = hashlib.sha256()

    def feed(self, chunk):
        # The span runs to the end of the object: the end records are hashed
        # with the directory so the whole-object stream can be compared.
        self.digest.update(chunk)
        view, offset = memoryview(chunk), 0
        while offset < len(view) and self._consumed < self._size:
            take = min(self._need - len(self._record), len(view) - offset, self._size - self._consumed)
            self._record += view[offset:offset + take]
            offset += take
            self._consumed += take
            if len(self._record) == _CENTRAL.size and self._need == _CENTRAL.size:
                fixed = _CENTRAL.unpack(self._record)
                if fixed[0] != b'PK\x01\x02' or len(self.entries) >= self._count:
                    raise _refusal('provider_output_archive_directory_invalid')
                self._need += fixed[12] + fixed[13] + fixed[14]
            if len(self._record) == self._need:
                self.entries.append(_central_entry(self._record))
                self._record, self._need = bytearray(), _CENTRAL.size

    def finish(self):
        if self._record or self._consumed != self._size or len(self.entries) != self._count:
            raise _refusal('provider_output_archive_directory_invalid')
        return self.entries, 'sha256:' + self.digest.hexdigest()


def _read_span(source, start, end):
    buffer = bytearray()
    source.stream_to(buffer.extend, start=start, end=end)
    return buffer


def _end_records(source, size, maximum_members):
    """Locate the directory from the end records, read with one range request."""
    tail_start = max(0, size - _TAIL_BYTES)
    tail = _read_span(source, tail_start, size)
    end, search_to = -1, len(tail)
    while True:
        end = tail.rfind(b'PK\x05\x06', 0, search_to)
        if end < 0:
            raise _refusal('provider_output_archive_end_record_invalid')
        if (len(tail) - end >= _END.size
                and end + _END.size + struct.unpack_from('<H', tail, end + 20)[0] == len(tail)):
            break
        search_to = end + 3
    (_, disk, directory_disk, disk_entries, entries, directory_size, directory_offset,
     _comment) = _END.unpack_from(tail, end)
    if disk != 0 or directory_disk != 0 or disk_entries != entries:
        raise _refusal('provider_output_archive_end_record_invalid')
    trailer_start, zip64 = tail_start + end, False
    if end >= _LOCATOR64.size and tail[end - _LOCATOR64.size:end - _LOCATOR64.size + 4] == b'PK\x06\x07':
        _, locator_disk, record_offset, disks = _LOCATOR64.unpack_from(tail, end - _LOCATOR64.size)
        record_at = end - _LOCATOR64.size - _END64.size
        if locator_disk != 0 or disks != 1 or record_at < 0:
            raise _refusal('provider_output_archive_zip64_invalid')
        (signature, record_size, _made, _needed, disk64, directory_disk64, disk_entries64, entries64,
         directory_size64, directory_offset64) = _END64.unpack_from(tail, record_at)
        if (signature != b'PK\x06\x06' or record_size != _END64.size - 12 or disk64 or directory_disk64
                or disk_entries64 != entries64):
            raise _refusal('provider_output_archive_zip64_invalid')
        trailer_start = tail_start + record_at
        if record_offset != trailer_start:
            raise _refusal('provider_output_archive_prepended_data' if record_offset < trailer_start
                           else 'provider_output_archive_zip64_invalid')
        for narrow, wide, limit in ((entries, entries64, 0xFFFF), (directory_size, directory_size64, _MAX32),
                                    (directory_offset, directory_offset64, _MAX32)):
            if narrow not in (limit, wide):
                raise _refusal('provider_output_archive_zip64_invalid')
        entries, directory_size, directory_offset, zip64 = entries64, directory_size64, directory_offset64, True
    if directory_offset + directory_size != trailer_start:
        # Offsets recorded relative to a later start are a self-extractor's
        # prepended stub; anything else is an inconsistent directory.
        raise _refusal('provider_output_archive_prepended_data' if directory_offset + directory_size < trailer_start
                       else 'provider_output_archive_directory_invalid')
    if entries > maximum_members:
        raise _refusal('provider_output_archive_member_cap_exceeded')
    return _Layout(directory_offset, directory_size, entries, zip64)


def _check_directory(entries, directory_offset, limits):
    for entry in entries:
        if entry.flag_bits & _ENCRYPTION_FLAGS or entry.compress_type == 99:
            raise _refusal('provider_output_archive_encrypted')
        if entry.compress_type not in _METHODS or entry.flag_bits & 0x0020:
            raise _refusal('provider_output_archive_method_unsupported')
    try:
        check_archive_entries(entries, limits['maximum_expanded_bytes'])
    except ProviderOutputInventoryError as exc:
        raise _refusal(str(exc)) from None
    for entry in entries:
        if entry.file_size > limits['maximum_member_inflated_bytes']:
            raise _refusal('provider_output_archive_member_inflate_bound_exceeded')
        if entry.compress_type == zipfile.ZIP_STORED and entry.compress_size != entry.file_size:
            raise _refusal('provider_output_archive_member_size_mismatch')
    ordered = sorted(entries, key=lambda entry: entry.header_offset)
    if ordered[0].header_offset != 0:
        raise _refusal('provider_output_archive_prepended_data')
    if (any(left.header_offset == right.header_offset for left, right in zip(ordered, ordered[1:]))
            or ordered[-1].header_offset >= directory_offset):
        raise _refusal('provider_output_archive_records_overlap')
    return ordered


class _Digest:
    __slots__ = ('sha256', 'crc32')

    def __init__(self):
        self.sha256, self.crc32 = hashlib.sha256(), 0

    def update(self, data):
        self.sha256.update(data)
        self.crc32 = zlib.crc32(data, self.crc32)


class _ArchiveStream:
    """Sink for the one whole-object GET: re-derives every record from its bytes."""

    def __init__(self, entries, directory_offset, directory_digest, step_bytes):
        self._entries, self._directory_offset = entries, directory_offset
        self._directory_digest, self._step = directory_digest, step_bytes
        self.digest, self._directory = hashlib.sha256(), hashlib.sha256()
        self.position, self._current, self.members = 0, 0, []
        self._state, self._need, self._carry = 'header', _LOCAL.size, bytearray()
        self._local = self._member = self._inflater = None
        self._remaining = self._gap = self._data_offset = 0
        self._zip64 = False

    def feed(self, chunk):
        self.digest.update(chunk)
        view, offset = memoryview(chunk), 0
        while offset < len(view):
            available = len(view) - offset
            if self._state == 'directory':
                self._directory.update(view[offset:])
                self.position += available
                return
            if self._state == 'data':
                take = min(self._remaining, available)
                self._inflater.feed(view[offset:offset + take])
                offset, self.position, self._remaining = offset + take, self.position + take, self._remaining - take
                if not self._remaining:
                    self._end_data()
                continue
            take = min(self._need - len(self._carry), available)
            self._carry += view[offset:offset + take]
            offset, self.position = offset + take, self.position + take
            if len(self._carry) == self._need:
                structure, self._carry = bytes(self._carry), bytearray()
                if self._state == 'header':
                    self._local_header(structure)
                elif self._state == 'names':
                    self._local_names(structure)
                else:
                    self._descriptor(structure)

    def finish(self):
        if self._state != 'directory' or self.position <= self._directory_offset:
            raise _refusal('provider_output_archive_directory_invalid')
        if 'sha256:' + self._directory.hexdigest() != self._directory_digest:
            raise _refusal('provider_output_archive_directory_changed')

    def _local_header(self, fixed):
        entry = self._entries[self._current]
        (signature, _version, flags, method, _time, _date, crc, csize, usize, name_length,
         extra_length) = _LOCAL.unpack(fixed)
        if signature != b'PK\x03\x04':
            raise _refusal('provider_output_archive_local_header_invalid')
        if name_length != len(entry.raw_name) or flags != entry.flag_bits or method != entry.compress_type:
            raise _refusal('provider_output_archive_local_header_mismatch')
        self._local = (crc, csize, usize)
        self._state, self._need = 'names', name_length + extra_length

    def _local_names(self, variable):
        entry = self._entries[self._current]
        if variable[:len(entry.raw_name)] != entry.raw_name:
            raise _refusal('provider_output_archive_local_header_mismatch')
        fields = _extra_fields(variable[len(entry.raw_name):], 'provider_output_archive_local_header_invalid')
        crc, csize, usize = self._local
        (usize, csize), zip64 = _zip64_values(fields, (usize, csize))
        expected = (entry.CRC, entry.compress_size, entry.file_size)
        observed = (crc, csize, usize)
        descriptor = bool(entry.flag_bits & _DESCRIPTOR_FLAG)
        if (any(value not in (0, wanted) for value, wanted in zip(observed, expected)) if descriptor
                else observed != expected):
            raise _refusal('provider_output_archive_local_header_mismatch')
        following = self._current + 1
        boundary = (self._entries[following].header_offset if following < len(self._entries)
                    else self._directory_offset)
        data_end = self.position + entry.compress_size
        if data_end > boundary:
            raise _refusal('provider_output_archive_records_overlap')
        gap = boundary - data_end
        if gap not in (((20, 24) if zip64 else (12, 16)) if descriptor else (0,)):
            raise _refusal('provider_output_archive_gap_invalid')
        self._gap, self._zip64, self._data_offset = gap, zip64, self.position
        self._member = _Digest()
        self._inflater = MemberInflater(_METHODS[entry.compress_type], entry.file_size,
                                        self._member.update, step_bytes=self._step)
        self._state, self._remaining = 'data', entry.compress_size
        if not self._remaining:
            self._end_data()

    def _end_data(self):
        self._inflater.finish()
        self._inflater = None
        if self._member.crc32 != self._entries[self._current].CRC:
            raise _refusal('provider_output_archive_crc_mismatch')
        if self._gap:
            self._state, self._need = 'descriptor', self._gap
        else:
            self._end_record()

    def _descriptor(self, descriptor):
        entry = self._entries[self._current]
        body = descriptor
        if len(descriptor) in (16, 24):
            if descriptor[:4] != b'PK\x07\x08':
                raise _refusal('provider_output_archive_descriptor_mismatch')
            body = descriptor[4:]
        if struct.unpack('<IQQ' if self._zip64 else '<III', body) != (
                entry.CRC, entry.compress_size, entry.file_size):
            raise _refusal('provider_output_archive_descriptor_mismatch')
        self._end_record()

    def _end_record(self):
        entry = self._entries[self._current]
        self.members.append({
            'path': entry.filename[:-1] if entry.is_dir() else entry.filename,
            'kind': 'directory' if entry.is_dir() else 'file',
            'size': entry.file_size,
            'compressed_size': entry.compress_size,
            'method': _METHODS[entry.compress_type],
            'crc32': entry.CRC,
            'sha256': 'sha256:' + self._member.sha256.hexdigest(),
            'mode': entry.external_attr >> 16,
            'local_header_offset': entry.header_offset,
            'data_offset': self._data_offset,
            'record_end_offset': self.position,
        })
        self._member, self._current = None, self._current + 1
        if self._current < len(self._entries):
            self._state, self._need = 'header', _LOCAL.size
        else:
            self._state = 'directory'


def _limits(maximum_members, maximum_expanded_bytes, maximum_member_inflated_bytes):
    if maximum_member_inflated_bytes is None:
        maximum_member_inflated_bytes = maximum_expanded_bytes
    if (type(maximum_members) is not int or not 0 < maximum_members <= MAX_ARCHIVE_ENTRIES
            or type(maximum_expanded_bytes) is not int or not 0 < maximum_expanded_bytes <= MAX_EXPANDED_BYTES
            or type(maximum_member_inflated_bytes) is not int
            or not 0 < maximum_member_inflated_bytes <= maximum_expanded_bytes):
        raise _refusal('provider_output_member_index_limits_invalid')
    return {'maximum_members': maximum_members, 'maximum_expanded_bytes': maximum_expanded_bytes,
            'maximum_member_inflated_bytes': maximum_member_inflated_bytes}


def _source_identity(source):
    identity = getattr(source, 'identity', None)
    if (not isinstance(identity, Mapping) or not callable(getattr(source, 'stream_to', None))
            or type(identity.get('size_bytes')) is not int or identity['size_bytes'] <= 0
            or any(not isinstance(identity.get(key), (str, type(None))) for key in ('etag', 'generation'))):
        raise _refusal('provider_output_member_index_source_invalid')
    return {key: identity.get(key) for key in ('size_bytes', 'etag', 'generation')}


def inflate_step_bytes(block_bytes: int) -> int:
    """Inflate step for a transport block: a quarter block, 4 KiB to 256 KiB."""
    return min(INFLATE_STEP_BYTES, max(4096, block_bytes // 4))


def build_member_index(source, *, maximum_expanded_bytes: int,
                       maximum_members: int = DEFAULT_MAXIMUM_MEMBERS,
                       maximum_member_inflated_bytes: int | None = None) -> dict:
    """Index ``source`` in one pinned pass and return ``provider_output_member_index.v1``.

    ``source`` is a pinned range source: ``identity`` (``size_bytes``, ``etag``,
    ``generation``) and ``stream_to(sink, *, start=0, end=None)``, such as
    ``ProviderOutputRangeReader`` or ``LocalArchiveRangeSource``.
    """
    limits = _limits(maximum_members, maximum_expanded_bytes, maximum_member_inflated_bytes)
    identity = _source_identity(source)
    try:
        layout = _end_records(source, identity['size_bytes'], limits['maximum_members'])
        directory = _DirectoryReader(layout.directory_size, layout.entries)
        source.stream_to(directory.feed, start=layout.directory_offset, end=identity['size_bytes'])
        entries, directory_digest = directory.finish()
        entries = _check_directory(entries, layout.directory_offset, limits)
        stream = _ArchiveStream(entries, layout.directory_offset, directory_digest,
                                inflate_step_bytes(getattr(source, 'block_bytes', 8 * 1024**2)))
        source.stream_to(stream.feed)
        stream.finish()
    except (ProviderOutputTransportError, ProviderOutputInventoryError) as exc:
        raise _refusal(str(exc)) from None
    members = stream.members
    files = [member for member in members if member['kind'] == 'file']
    total = sum(member['size'] for member in files)
    bulk = sum(member['size'] for member in files
               if PurePosixPath(member['path']).suffix.lower() in BULK_EXTENSIONS)
    index = {
        'schema_version': SCHEMA,
        'archive': {'sha256': 'sha256:' + stream.digest.hexdigest(), 'size': identity['size_bytes'],
                    'etag': identity['etag'], 'generation': identity['generation'],
                    'durable_reference': None},
        'limits': limits,
        'directory': {'offset': layout.directory_offset, 'size': layout.directory_size,
                      'zip64': layout.zip64},
        'members': members,
        'totals': {'members': len(members), 'files': len(files), 'directories': len(members) - len(files),
                   'bytes': total, 'compressed_bytes': sum(member['compressed_size'] for member in members),
                   'bytes_by_class': {'bulk': bulk, 'small': total - bulk}},
        'private_url_recorded': False,
    }
    index['index_digest'] = canonical_digest(index, digest_field='index_digest')
    return index


def _count(value, minimum=0):
    return type(value) is int and value >= minimum


def validate_member_index(index) -> Mapping:
    """Check an index's digest and invariants before any consumer trusts its offsets."""
    if (not isinstance(index, Mapping) or index.get('schema_version') != SCHEMA
            or index.get('private_url_recorded') is not False):
        raise _refusal('provider_output_member_index_invalid')
    if index.get('index_digest') != canonical_digest(index, digest_field='index_digest'):
        raise _refusal('provider_output_member_index_digest_mismatch')
    archive, directory, members = index.get('archive'), index.get('directory'), index.get('members')
    if (not isinstance(archive, Mapping) or not _SHA256.fullmatch(str(archive.get('sha256')))
            or not _count(archive.get('size'), 1) or not isinstance(directory, Mapping)
            or not _count(directory.get('offset')) or not _count(directory.get('size'))
            or directory['offset'] + directory['size'] >= archive['size']
            or not isinstance(members, list) or not members):
        raise _refusal('provider_output_member_index_invalid')
    position, seen = 0, set()
    for member in members:
        if (not isinstance(member, Mapping) or set(member) != _MEMBER_KEYS
                or member['kind'] not in ('file', 'directory') or member['method'] not in ('stored', 'deflate')
                or not all(_count(member[key]) for key in ('size', 'compressed_size', 'mode', 'data_offset',
                                                          'local_header_offset', 'record_end_offset'))
                or type(member['crc32']) is not int or not 0 <= member['crc32'] <= _MAX32
                or not _SHA256.fullmatch(str(member['sha256']))
                or member['local_header_offset'] != position
                or member['data_offset'] <= member['local_header_offset'] + _LOCAL.size
                or member['data_offset'] + member['compressed_size'] > member['record_end_offset']
                or (member['method'] == 'stored' and member['size'] != member['compressed_size'])
                or (member['kind'] == 'directory' and member['size'] != 0)):
            raise _refusal('provider_output_member_index_invalid')
        try:
            safe_member_name(member['path'])
        except ProviderOutputInventoryError:
            raise _refusal('provider_output_member_index_invalid') from None
        folded = unicodedata.normalize('NFC', member['path']).casefold()
        if folded in seen:
            raise _refusal('provider_output_member_index_invalid')
        seen.add(folded)
        position = member['record_end_offset']
    if position != directory['offset']:
        raise _refusal('provider_output_member_index_invalid')
    return index


class LocalArchiveRangeSource:
    """Read-only range source over one local ZIP, for the diagnostic CLI.

    A local file has no ETag. Its identity is pinned by device, inode, size and
    modification time, and a change between reads is refused exactly as a
    remote version change is.
    """

    def __init__(self, path, *, block_bytes: int = 8 * 1024**2):
        if type(block_bytes) is not int or block_bytes < 1:
            raise _refusal('provider_output_member_index_source_invalid')
        self._descriptor = os.open(path, os.O_RDONLY)
        status = os.fstat(self._descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
            os.close(self._descriptor)
            raise _refusal('provider_output_member_index_local_archive_invalid')
        self._pin = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
        self.identity = {'size_bytes': status.st_size, 'etag': None, 'generation': None}
        self.block_bytes, self.request_count, self.transferred_bytes = block_bytes, 0, 0

    def _check_pin(self):
        status = os.fstat(self._descriptor)
        if (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns) != self._pin:
            raise ProviderOutputTransportError('provider_output_remote_version_changed')

    def stream_to(self, sink, *, start=0, end=None):
        size = self.identity['size_bytes']
        end = size if end is None else end
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= size:
            raise ProviderOutputTransportError('provider_output_range_invalid')
        self._check_pin()
        self.request_count += 1
        position = start
        while position < end:
            chunk = os.pread(self._descriptor, min(self.block_bytes, end - position), position)
            if not chunk:
                raise ProviderOutputTransportError('provider_output_archive_truncated')
            position += len(chunk)
            self.transferred_bytes += len(chunk)
            sink(chunk)
            del chunk
        self._check_pin()
        return end - start

    def close(self):
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def summarize_member_index(index: Mapping) -> dict:
    return {'schema_version': SUMMARY_SCHEMA, 'index_digest': index['index_digest'],
            'archive': {key: index['archive'][key] for key in ('sha256', 'size', 'etag', 'generation')},
            'directory': index['directory'], 'limits': index['limits'], 'totals': index['totals']}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Index one local provider-output ZIP (read-only).')
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--maximum-expanded-bytes', type=int, default=MAX_EXPANDED_BYTES)
    parser.add_argument('--maximum-members', type=int, default=DEFAULT_MAXIMUM_MEMBERS)
    args = parser.parse_args(argv)
    try:
        with LocalArchiveRangeSource(args.archive) as source:
            index = build_member_index(source, maximum_expanded_bytes=args.maximum_expanded_bytes,
                                       maximum_members=args.maximum_members)
    except OSError:
        print('provider_output_member_index refused: provider_output_member_index_local_archive_invalid',
              file=sys.stderr)
        return 1
    except ProviderOutputMemberIndexError as exc:
        print(f'provider_output_member_index refused: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(summarize_member_index(index), indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
