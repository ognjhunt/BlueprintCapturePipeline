"""In-memory object-store and ZIP builders for provider-output archive tests.

Nothing here touches disk or the network. ``VirtualObject`` materializes only
the byte range a caller reads, so a multi-GB archive whose big members are
zero runs costs neither disk nor RAM beyond one transport read. ``RangeStore``
is an object-store double for one pinned object: byte ranges, ``ETag`` and
``If-Match`` (412), truncation, and a version swap between requests.
"""

from __future__ import annotations

import bisect
import builtins
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import io
import os
import re
import struct
import urllib.error
import zipfile
import zlib

from blueprint_pipeline.provider_output_range_transport import ProviderOutputRangeReader

URL = "https://storage.example.invalid/private/run.zip?X-Amz-Signature=SECRET_DO_NOT_RECORD"
SECRET = "SECRET_DO_NOT_RECORD"
STORED, DEFLATED = zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED
_ZERO_BLOCK = bytes(16 * 1024**2)


@dataclass(frozen=True)
class Zeros:
    """A lazily generated run of zero bytes."""

    size: int

    @property
    def crc32(self) -> int:
        return _zero_crc32(self.size)


@functools.lru_cache(maxsize=None)
def _zero_crc32(size: int) -> int:
    crc, remaining, view = 0, size, memoryview(_ZERO_BLOCK)
    while remaining:
        step = min(remaining, len(_ZERO_BLOCK))
        crc = zlib.crc32(view[:step], crc)
        remaining -= step
    return crc


def _length(segment) -> int:
    return segment.size if isinstance(segment, Zeros) else len(segment)


class VirtualObject:
    """An immutable byte string assembled from literal and zero-run segments."""

    def __init__(self, segments):
        self._starts, self._segments, position = [], [], 0
        for segment in segments:
            if _length(segment):
                self._starts.append(position)
                self._segments.append(segment)
                position += _length(segment)
        self.size = position

    def __len__(self) -> int:
        return self.size

    def read(self, start: int, end: int) -> bytearray:
        """Return bytes ``[start, end)`` in one freshly allocated buffer."""
        end = min(end, self.size)
        out = bytearray(max(0, end - start))
        index = max(0, bisect.bisect_right(self._starts, start) - 1)
        position = start
        while position < end and index < len(self._segments):
            begin, segment = self._starts[index], self._segments[index]
            stop = min(end, begin + _length(segment))
            if not isinstance(segment, Zeros):
                out[position - start:stop - start] = memoryview(segment)[position - begin:stop - begin]
            position, index = stop, index + 1
        return out

    def to_bytes(self) -> bytes:
        return bytes(self.read(0, self.size))

    def patched(self, offset: int, payload: bytes) -> VirtualObject:
        """A same-size copy with ``payload`` written over the bytes at ``offset``."""
        end, segments, inserted = offset + len(payload), [], False
        for start, segment in zip(self._starts, self._segments):
            stop = start + _length(segment)
            if stop <= offset or start >= end:
                segments.append(segment)
                continue
            if start < offset:
                segments.append(_slice(segment, 0, offset - start))
            if not inserted:
                segments.append(payload)
                inserted = True
            if stop > end:
                segments.append(_slice(segment, end - start, stop - start))
        return VirtualObject(segments)


def _slice(segment, start, stop):
    return Zeros(stop - start) if isinstance(segment, Zeros) else segment[start:stop]


class VirtualFile(io.RawIOBase):
    """Seekable read-only file view, so ``zipfile`` can cross-check a virtual archive."""

    def __init__(self, obj: VirtualObject):
        super().__init__()
        self._obj, self._position = obj, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._obj.size}[whence]
        self._position = max(0, base + offset)
        return self._position

    def readinto(self, buffer):
        data = self._obj.read(self._position, self._position + len(buffer))
        buffer[:len(data)] = data
        self._position += len(data)
        return len(data)


class _Response:
    def __init__(self, url, status, headers, obj, start, stop):
        self.status, self.headers, self._url = status, headers, url
        self._obj, self._position, self._stop = obj, start, stop

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._stop - self._position
        end = min(self._stop, self._position + size)
        chunk = self._obj.read(self._position, end) if end > self._position else b""
        self._position = end
        return chunk

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RangeStore:
    """Object-store double serving one pinned object through the real transport.

    ``requests`` logs every GET as ``{"range": (first, last) | None, "if_match": ...}``.
    Faults: ``truncate`` drops every response's final byte,
    ``truncate_whole_object`` only a whole-object GET's, and ``truncate_when``
    those whose logged request it accepts; ``ignore_if_match``
    serves whatever version is current; ``next_version`` swaps in a new
    ``(object, etag)`` just before the next whole-object GET.
    """

    def __init__(self, data, *, etag='"version1"', generation=None, url=URL):
        self.object = data if isinstance(data, VirtualObject) else VirtualObject([bytes(data)])
        self.etag, self.generation, self.url = etag, generation, url
        self.requests: list[dict] = []
        self.truncate = False
        self.truncate_whole_object = False
        self.truncate_when = None
        self.ignore_if_match = False
        self.next_version = None

    def opener(self, request, timeout, policy):
        headers = {key.lower(): value for key, value in request.header_items()}
        requested = headers.get("range")
        if requested is None and self.next_version is not None:
            self.object, self.etag = self.next_version
            self.next_version = None
        entry = {"range": None, "if_match": headers.get("if-match")}
        if requested is not None:
            first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", requested).groups())
            entry["range"] = (first, last)
        self.requests.append(entry)
        if not self.ignore_if_match and entry["if_match"] not in (None, self.etag):
            # The URL is deliberately not the signed one: errors must stay secret-free.
            raise urllib.error.HTTPError("redacted", 412, "Precondition Failed", {}, None)
        size = self.object.size
        if requested is not None:
            start, stop = first, min(last, size - 1) + 1
            status, response_headers = 206, {"Content-Range": f"bytes {start}-{stop - 1}/{size}"}
        else:
            start, stop, status, response_headers = 0, size, 200, {}
        response_headers.update({"ETag": self.etag, "Content-Length": str(stop - start)})
        if self.generation:
            response_headers["x-goog-generation"] = self.generation
        short = (self.truncate or (self.truncate_whole_object and requested is None)
                 or (self.truncate_when is not None and self.truncate_when(entry)))
        return _Response(request.full_url, status, response_headers, self.object, start,
                         stop - 1 if short else stop)

    def reader(self, **options) -> ProviderOutputRangeReader:
        options.setdefault("maximum_archive_bytes", max(1, self.object.size))
        return ProviderOutputRangeReader(self.url, opener=self.opener, **options)

    def ranges(self):
        return [row["range"] for row in self.requests if row["range"] is not None]

    def whole_object_gets(self) -> int:
        return sum(row["range"] is None for row in self.requests)


@dataclass
class Entry:
    """One record for ``build_zip``; the optional fields inject faults."""

    name: str | bytes
    data: bytes | Zeros = b""
    method: int = STORED
    descriptor: str | None = None  # None, "signed" or "unsigned"
    zip64: bool = False  # force ZIP64 extras in both headers
    mode: int | None = None
    flags: int = 0
    local_name: bytes | None = None
    crc: int | None = None
    payload: bytes | None = None  # compressed bytes as stored
    size: int | None = None  # declared uncompressed size
    gap_after: bytes = b""
    descriptor_crc: int | None = None
    central_extra: bytes = b""


def deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    return compressor.compress(data) + compressor.flush()


def build_zip(entries, *, zip64_end=False, prepend=b"", shift_offsets=True, comment=b"",
              central_offsets=None) -> VirtualObject:
    """Write a ZIP whose records are exactly as described, lazily for zero runs.

    ``shift_offsets=False`` with ``prepend`` reproduces a self-extractor: the
    recorded offsets ignore the prepended bytes.
    """
    segments = [prepend] if prepend else []
    physical, base, central = len(prepend), 0 if shift_offsets else len(prepend), []
    for entry in entries:
        raw_name = entry.name if isinstance(entry.name, bytes) else entry.name.encode("utf-8")
        flags = entry.flags | (0x08 if entry.descriptor else 0)
        if isinstance(entry.name, str) and not entry.name.isascii():
            flags |= 0x800
        if isinstance(entry.data, Zeros):
            size, crc = entry.data.size, entry.data.crc32
            payload = entry.data if entry.payload is None else entry.payload
        else:
            size, crc = len(entry.data), zlib.crc32(entry.data)
            payload = entry.payload
            if payload is None:
                payload = deflate(entry.data) if entry.method == DEFLATED else entry.data
        size = size if entry.size is None else entry.size
        crc = crc if entry.crc is None else entry.crc
        csize = _length(payload)
        zip64 = entry.zip64 or size >= 0xFFFFFFFF or csize >= 0xFFFFFFFF
        local = (0, 0, 0) if entry.descriptor else (crc, csize, size)
        local_extra = struct.pack("<HHQQ", 1, 16, local[2], local[1]) if zip64 else b""
        if zip64:
            local = (local[0], 0xFFFFFFFF, 0xFFFFFFFF)
        local_name = raw_name if entry.local_name is None else entry.local_name
        header = struct.pack("<4sHHHHHIIIHH", b"PK\x03\x04", 45 if zip64 else 20, flags, entry.method,
                             0, 0x21, *local, len(local_name), len(local_extra)) + local_name + local_extra
        record = [header, payload]
        if entry.descriptor:
            fields = struct.pack("<IQQ" if zip64 else "<III",
                                 crc if entry.descriptor_crc is None else entry.descriptor_crc, csize, size)
            record.append((b"PK\x07\x08" if entry.descriptor == "signed" else b"") + fields)
        record.append(entry.gap_after)
        mode = entry.mode if entry.mode is not None else (
            0o040755 if raw_name.endswith(b"/") else 0o100644)
        central.append((raw_name, flags, entry.method, crc, csize, size, physical - base, mode,
                        entry.zip64, entry.central_extra))
        segments += record
        physical += sum(map(_length, record))
    directory_offset, parts = physical - base, []
    for position, (raw_name, flags, method, crc, csize, size, offset, mode, forced,
                   extra_tail) in enumerate(central):
        offset = (central_offsets or {}).get(position, offset)
        values, extended = {}, []
        for key, value in (("size", size), ("csize", csize), ("offset", offset)):
            values[key] = 0xFFFFFFFF if forced or value >= 0xFFFFFFFF else value
            if values[key] == 0xFFFFFFFF:
                extended.append(value)
        extra = (struct.pack("<HH", 1, 8 * len(extended)) + b"".join(
            struct.pack("<Q", value) for value in extended)) if extended else b""
        extra += extra_tail
        external = (mode << 16) | (0x10 if raw_name.endswith(b"/") else 0)
        parts.append(struct.pack("<4sBBBBHHHHIIIHHHHHII", b"PK\x01\x02", 45, 3, 45 if extended else 20, 0,
                                 flags, method, 0, 0x21, crc, values["csize"], values["size"],
                                 len(raw_name), len(extra), 0, 0, 0, external, values["offset"])
                     + raw_name + extra)
    directory = b"".join(parts)
    segments.append(directory)
    physical += len(directory)
    count = len(central)
    if zip64_end or count >= 0xFFFF or len(directory) >= 0xFFFFFFFF or directory_offset >= 0xFFFFFFFF:
        segments.append(struct.pack("<4sQHHIIQQQQ", b"PK\x06\x06", 44, 45, 45, 0, 0, count, count,
                                    len(directory), directory_offset))
        segments.append(struct.pack("<4sIQI", b"PK\x06\x07", 0, physical - base, 1))

    def capped(value, limit):
        return limit if zip64_end or value >= limit else value

    segments.append(struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0, 0, capped(count, 0xFFFF),
                                capped(count, 0xFFFF), capped(len(directory), 0xFFFFFFFF),
                                capped(directory_offset, 0xFFFFFFFF), len(comment)) + comment)
    return VirtualObject(segments)


class _Unseekable(io.RawIOBase):
    def __init__(self):
        super().__init__()
        self.buffer = bytearray()

    def writable(self):
        return True

    def write(self, data):
        self.buffer += data
        return len(data)

    def tell(self):
        raise OSError("unseekable")


def python_zip(files, *, compression=DEFLATED, streamed=False, force_zip64=()) -> bytes:
    """An archive written by Python's own ``zipfile`` (an independent writer).

    ``streamed=True`` writes to an unseekable sink, which makes ``zipfile`` emit
    signed data descriptors; names in ``force_zip64`` get ZIP64 local headers.
    """
    sink = _Unseekable() if streamed else io.BytesIO()
    with zipfile.ZipFile(sink, "w", compression=compression) as archive:
        for name, payload in files.items():
            if name.endswith("/"):
                archive.writestr(name, b"")
            elif name in force_zip64:
                with archive.open(name, "w", force_zip64=True) as stream:
                    stream.write(payload)
            else:
                archive.writestr(name, payload)
    return bytes(sink.buffer) if streamed else sink.getvalue()


@contextmanager
def no_disk_writes(monkeypatch):
    """Record, and refuse, any attempt to create, write, move or delete a file."""
    attempts = []
    real_open, real_os_open = builtins.open, os.open
    writing = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in "wax+"):
            attempts.append(("open", str(file), mode))
            raise AssertionError("disk write attempted")
        return real_open(file, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & writing:
            attempts.append(("os.open", str(path), flags))
            raise AssertionError("disk write attempted")
        return real_os_open(path, flags, *args, **kwargs)

    def refuse(name):
        def refused(*args, **kwargs):
            attempts.append((name, args))
            raise AssertionError("disk mutation attempted")
        return refused

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", guarded_open)
        patch.setattr(io, "open", guarded_open)
        patch.setattr(os, "open", guarded_os_open)
        for name in ("mkdir", "makedirs", "rename", "replace", "remove", "unlink", "rmdir",
                     "symlink", "link", "truncate", "chmod"):
            patch.setattr(os, name, refuse(name))
        yield attempts
