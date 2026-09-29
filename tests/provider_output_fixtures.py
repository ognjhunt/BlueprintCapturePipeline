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
import hashlib
import io
import json
import os
from pathlib import PurePosixPath
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
        # Assign through a memoryview: bytearray slice assignment would first
        # copy the source, doubling this simulated network read in memory.
        with memoryview(out) as target:
            while position < end and index < len(self._segments):
                begin, segment = self._starts[index], self._segments[index]
                stop = min(end, begin + _length(segment))
                if not isinstance(segment, Zeros):
                    target[position - start:stop - start] = memoryview(segment)[position - begin:stop - begin]
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
    def __init__(self, url, status, headers, obj, start, stop, max_read=None):
        self.status, self.headers, self._url = status, headers, url
        self._obj, self._position, self._stop, self._max_read = obj, start, stop, max_read

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._stop - self._position
        if self._max_read:  # a short read, as a real socket may return
            size = min(size, self._max_read)
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
    ``(object, etag)`` just before the next whole-object GET. ``max_read``
    makes every response return at most that many bytes per read, and
    ``range_fault`` ("content_length", "content_range" or "overlong") breaks
    every ranged response in that way. ``absent`` answers every request 404,
    as a store does for an object that was never written or was deleted.
    """

    def __init__(self, data, *, etag='"version1"', generation=None, url=URL):
        self.object = data if isinstance(data, VirtualObject) else VirtualObject([bytes(data)])
        self.etag, self.generation, self.url = etag, generation, url
        self.requests: list[dict] = []
        self.truncate = False
        self.truncate_whole_object = False
        self.truncate_when = None
        self.max_read = None
        self.range_fault = None
        self.ignore_if_match = False
        self.next_version = None
        self.absent = False

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
        if self.absent:
            raise urllib.error.HTTPError("redacted", 404, "Not Found", {}, None)
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
        body_stop = stop - 1 if short else stop
        if requested is not None and self.range_fault == "content_length":
            response_headers["Content-Length"] = str(stop - start + 1)
        elif requested is not None and self.range_fault == "content_range":
            response_headers["Content-Range"] = f"bytes {start + 1}-{stop - 1}/{size}"
        elif requested is not None and self.range_fault == "overlong":
            body_stop += 1
        return _Response(request.full_url, status, response_headers, self.object, start,
                         body_stop, self.max_read)

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
    central_comment: bytes = b""


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
                        entry.zip64, entry.central_extra, entry.central_comment))
        segments += record
        physical += sum(map(_length, record))
    directory_offset, parts = physical - base, []
    for position, (raw_name, flags, method, crc, csize, size, offset, mode, forced,
                   extra_tail, file_comment) in enumerate(central):
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
                                 len(raw_name), len(extra), len(file_comment), 0, 0, external,
                                 values["offset"])
                     + raw_name + extra + file_comment)
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


QUICK10_RESULT = "native_task_arena_policy_canary_session_result.v1.json"
QUICK10_CAMERAS = ("external", "wrist", "overview")
QUICK10_CANDIDATES = ("pi05_droid", "groot_n17_droid")


@dataclass(frozen=True)
class Quick10Archive:
    """A Quick-10-shaped output: the archive plus which members are which."""

    archive: VirtualObject
    result: dict
    json_members: tuple[str, ...]
    bulk_members: tuple[str, ...]
    payloads: dict


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True).encode()


def quick10_shaped_archive(*, cells: int = 10, frames_per_camera: int = 4, png_bytes: int = 48 * 1024,
                           mp4_bytes: int = 256 * 1024, policy_requests_per_episode: int = 2,
                           policy_request_bytes: int = 64 * 1024,
                           result_status: str = "runtime_completed_unqualified_pending_closeout",
                           extra_members: dict | None = None) -> Quick10Archive:
    """A Quick-10-shaped provider output whose bulk members are zero runs.

    Members follow the provider packer's order (``sorted(rglob)``): every cell's
    tree, then the top-level JSON. The JSON members (the aggregate result, the
    ten child results, per-episode receipts and frame manifests, the telemetry
    index) are real and deflated. PNG frames, the three-camera MP4 reviews and
    the policy requests are stored ``Zeros``. ``mp4_bytes`` scales the archive:
    at 72 MiB the top-level result starts past 4 GiB, so its offsets are ZIP64.
    """
    payloads: dict = {}
    episodes = []
    for cell in range(cells):
        root = f"cell_runs/{cell:02d}"
        payloads[f"{root}/{QUICK10_RESULT}"] = _json_bytes({
            "schema_version": QUICK10_RESULT.removesuffix(".json"), "cell_index": cell,
            "status": "runtime_selected_cell_completed_pending_aggregation", "blockers": []})
        for candidate in QUICK10_CANDIDATES:
            episode = f"cell{cell:02d}-{candidate}"
            episodes.append({"episode_id": episode, "candidate_id": candidate, "cell_index": cell})
            payloads[f"{root}/episodes/{episode}.score_receipt.json"] = _json_bytes(
                {"episode_id": episode, "task_success": cell % 2 == 0})
            payloads[f"{root}/episodes/{episode}.state_trace.json"] = _json_bytes(
                {"episode_id": episode, "steps": list(range(40))})
            media = f"{root}/episodes/media/{episode}"
            payloads[f"{media}/frame_manifest.json"] = _json_bytes(
                {"episode_id": episode, "frames": frames_per_camera * len(QUICK10_CAMERAS)})
            for camera in QUICK10_CAMERAS:
                payloads[f"{media}/{camera}.mp4"] = Zeros(mp4_bytes)
                for frame in range(frames_per_camera):
                    payloads[f"{media}/frames/{camera}/{frame:06d}.png"] = Zeros(png_bytes)
            for request in range(policy_requests_per_episode):
                payloads[f"{media}/policy-requests/{request:04d}.json"] = Zeros(policy_request_bytes)
    result = {"schema_version": QUICK10_RESULT.removesuffix(".json"), "status": result_status,
              "blockers": [], "run_kind": "internal_policy_canary", "episodes": episodes}
    payloads[QUICK10_RESULT] = _json_bytes(result)
    payloads["policy_canary_telemetry_index.json"] = _json_bytes({"episodes": len(episodes)})
    payloads.update(extra_members or {})
    ordered = dict(sorted(payloads.items(), key=lambda item: PurePosixPath(item[0]).parts))
    archive = build_zip([Entry(name, data, method=STORED if isinstance(data, Zeros) else DEFLATED)
                         for name, data in ordered.items()])
    bulk = tuple(name for name, data in ordered.items() if isinstance(data, Zeros))
    small = tuple(name for name, data in ordered.items() if not isinstance(data, Zeros))
    return Quick10Archive(archive=archive, result=result, json_members=small, bulk_members=bulk,
                          payloads=ordered)


class _CasNotFound(KeyError):
    """A missing key, shaped as an S3-compatible client reports it."""

    response = {"ResponseMetadata": {"HTTPStatusCode": 404}, "Error": {"Code": "NoSuchKey"}}


class _CasPreconditionFailed(RuntimeError):
    response = {"ResponseMetadata": {"HTTPStatusCode": 412}, "Error": {"Code": "PreconditionFailed"}}


class _CasBody:
    def __init__(self, obj: VirtualObject, start: int, stop: int):
        self._obj, self._position, self._stop = obj, start, stop

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._stop - self._position
        end = min(self._stop, self._position + size)
        chunk = bytes(self._obj.read(self._position, end)) if end > self._position else b""
        self._position = end
        return chunk

    def close(self):
        pass


def virtual_sha256(obj: VirtualObject, step: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    for start in range(0, obj.size, step):
        digest.update(obj.read(start, start + step))
    return "sha256:" + digest.hexdigest()


class VirtualCasClient:
    """A B2 double: multipart and file uploads, HEAD, and whole or ranged GETs.

    Streamed parts are hashed in order and never kept. On completion the key
    serves the ``VirtualObject`` registered (``register``) for the streamed
    digest, so a multi-GB archive costs no memory; completing an unregistered
    digest is a test-setup error. ``upload_file`` keeps the small file's
    bytes. ``uploads``, ``aborted``, ``readback_bytes`` and ``calls`` record
    what happened; reads are thread-safe (ranged readback runs in parallel).
    """

    def __init__(self, *, bucket: str = "blueprint-artifacts"):
        import threading

        self.bucket = bucket
        self.sources: dict[str, VirtualObject] = {}
        self.objects: dict[str, tuple[VirtualObject, dict, str]] = {}
        self.pending: dict[str, dict] = {}
        self.calls: list[tuple] = []
        self.uploads = self.aborted = self.readback_bytes = 0
        self._lock = threading.Lock()

    def register(self, obj: VirtualObject) -> str:
        digest = virtual_sha256(obj)
        self.sources[digest] = obj
        return digest

    def _log(self, *call):
        with self._lock:
            self.calls.append(call)

    def head_object(self, *, Bucket, Key):
        self._log("head", Key)
        if Key not in self.objects:
            raise _CasNotFound(Key)
        obj, metadata, etag = self.objects[Key]
        return {"ContentLength": obj.size, "Metadata": dict(metadata), "ETag": etag}

    def create_multipart_upload(self, *, Bucket, Key, Metadata, ContentType):
        upload = f"upload-{len(self.pending) + len(self.calls)}"
        self._log("create_multipart_upload", Key)
        self.pending[upload] = {"metadata": dict(Metadata), "digest": hashlib.sha256(), "size": 0}
        return {"UploadId": upload}

    def upload_part(self, *, Bucket, Key, UploadId, PartNumber, Body):
        row = self.pending[UploadId]
        row["digest"].update(Body)
        row["size"] += len(Body)
        return {"ETag": f'"part-{PartNumber}"'}

    def complete_multipart_upload(self, *, Bucket, Key, UploadId, MultipartUpload):
        row = self.pending.pop(UploadId)
        digest = "sha256:" + row["digest"].hexdigest()
        self._store(Key, self.sources[digest], row["metadata"], digest)

    def abort_multipart_upload(self, *, Bucket, Key, UploadId):
        self.pending.pop(UploadId, None)
        self.aborted += 1

    def upload_file(self, source, bucket, key, ExtraArgs=None):
        with open(source, "rb") as stream:
            data = stream.read()
        self._log("upload_file", key)
        self._store(key, VirtualObject([data]), (ExtraArgs or {}).get("Metadata") or {},
                    "sha256:" + hashlib.sha256(data).hexdigest())

    def _store(self, key, obj, metadata, digest):
        self.objects[key] = (obj, dict(metadata), f'"b2-{digest[7:23]}"')
        self.uploads += 1

    def get_object(self, *, Bucket, Key, Range=None, IfMatch=None):
        self._log("get_object", Key, Range)
        obj, _, etag = self.objects[Key]
        if IfMatch is not None and IfMatch != etag:
            raise _CasPreconditionFailed(Key)
        if Range is None:
            with self._lock:
                self.readback_bytes += obj.size
            return {"Body": _CasBody(obj, 0, obj.size), "ETag": etag, "ContentLength": obj.size,
                    "ResponseMetadata": {"HTTPStatusCode": 200}}
        first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", Range).groups())
        stop = min(last, obj.size - 1) + 1
        with self._lock:
            self.readback_bytes += stop - first
        return {"Body": _CasBody(obj, first, stop), "ETag": etag, "ContentLength": stop - first,
                "ContentRange": f"bytes {first}-{stop - 1}/{obj.size}",
                "ResponseMetadata": {"HTTPStatusCode": 206}}


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


# -- Streamed attempts ------------------------------------------------------
# A download-mode evidence tree turned into the streamed layout the readers see:
# the same files zipped, indexed and sealed with a durable reference, only the
# needed members materialized under ``immutable_execution`` (0440), the view
# descriptor beside it, and view reads served by a ``RangeStore``.

STREAM_SELECTION_VERSION = "policy_canary_output_member_contract.v1"


def cas_reference(index) -> dict:
    """A remote-verified CAS reference naming the indexed archive's B2 copy."""
    digest = index["archive"]["sha256"]
    return {"schema_version": "task_evaluation_scene_artifact_reference.v1", "status": "remote_verified",
            "artifact_kind": "policy-canary-provider-output",
            "uri": ("s3://blueprint-artifacts/blueprint/arm-decision-proof-v1/configured-scenes/artifacts/"
                    f"policy-canary-provider-output/sha256/{digest.removeprefix('sha256:')}/"
                    "vast_provider_runtime_output.zip"),
            "digest": digest, "size_bytes": index["archive"]["size"], "content_addressed_key": True,
            "remote_identity_verified": True, "full_byte_service_account_readback_passed": True}


def zip_tree(root) -> VirtualObject:
    """Every regular file under ``root``, deflated, in relative-path order (the provider packer's)."""
    from pathlib import Path

    root = Path(root)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=(2026, 9, 28, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
    return VirtualObject([buffer.getvalue()])


@dataclass
class StreamedAttempt:
    """What ``stream_evidence_tree`` built: paths, the sealed index and the B2 double."""

    attempt: object
    evidence: object
    index: dict
    index_path: object
    receipt_path: object
    descriptor: dict
    store: RangeStore
    archive: VirtualObject

    @property
    def rows(self) -> dict:
        return {row["path"]: row for row in self.index["members"] if row["kind"] == "file"}

    def remote(self) -> list[str]:
        return sorted(path for path in self.rows if not (self.evidence / path).is_file())

    def data_ranges(self) -> list[tuple[int, int]]:
        """The store's ranged requests, less the one-byte probes that pin the ETag."""
        return [span for span in self.store.ranges() if span != (0, 0)]


def _contract_v1_needed(path: str) -> bool:
    """The lane's contract rule (JSON outside policy-requests, less the ten child results)."""
    from blueprint_pipeline.policy_canary_output_members import POLICY_CANARY_OUTPUT_CONTRACT

    return POLICY_CANARY_OUTPUT_CONTRACT.needed(path)


def stream_evidence_tree(source, attempt, *, needed=None, block_bytes=128 * 1024) -> StreamedAttempt:
    """Stream ``source``'s files into ``attempt``: index, seal, ingest ``needed``, write the view.

    ``needed`` decides which archive paths are materialized (the lane's contract
    v1 by default: JSON outside ``policy-requests``, less the ten per-cell child
    results). Nothing is written under ``source``.
    """
    from pathlib import Path
    from types import SimpleNamespace

    from blueprint_pipeline.provider_output_member_index import (
        build_member_index, build_member_selection, seal_durable_reference,
    )
    from blueprint_pipeline.provider_output_member_view import write_member_view_descriptor
    from blueprint_pipeline.provider_output_range_ingestion import CasArchiveSource, ingest_selected_members

    needed = needed or _contract_v1_needed
    attempt = Path(attempt)
    attempt.mkdir(parents=True, exist_ok=True)
    archive = zip_tree(source)
    store = RangeStore(archive)
    index = build_member_index(store.reader(block_bytes=block_bytes),
                               maximum_expanded_bytes=max(64 * 1024**2, 8 * archive.size))
    index = seal_durable_reference(index, cas_reference(index))
    index_path = attempt / "provider_output_member_index.v1.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths = [row["path"] for row in index["members"] if row["kind"] == "file" and needed(row["path"])]
    selection = build_member_selection(index, paths, selection_version=STREAM_SELECTION_VERSION)
    source_reference = CasArchiveSource(index["archive"]["durable_reference"], presign=lambda: URL,
                                        opener=store.opener, block_bytes=block_bytes)
    receipt = ingest_selected_members(
        source=source_reference, index=index, selection=selection, members_root=attempt / "immutable_execution",
        metadata_root=attempt / ".provider_output_ingestion", reserve=lambda outstanding: None,
        disk_usage_provider=lambda path: SimpleNamespace(free=10**12))
    assert receipt["status"] == "materialized", receipt.get("blockers")
    receipt_path = attempt / ".provider_output_ingestion" / "receipt.json"
    descriptor = write_member_view_descriptor(evidence_root=attempt / "immutable_execution",
                                              index_path=index_path, ingestion_receipt_path=receipt_path)
    store.requests.clear()
    return StreamedAttempt(attempt=attempt, evidence=attempt / "immutable_execution", index=index,
                           index_path=index_path, receipt_path=receipt_path, descriptor=descriptor,
                           store=store, archive=archive)


def serve_member_views(monkeypatch, store: RangeStore) -> None:
    """Route production view reads to ``store``: B2 configured, presign in memory, the store's opener."""
    from blueprint_pipeline import provider_output_member_view as views
    from blueprint_pipeline import provider_output_range_transport as transport
    from blueprint_pipeline.task_evaluation_configured_scene_object_store import _ARTIFACT_STORE_FILE_ENV

    for name in _ARTIFACT_STORE_FILE_ENV.values():
        monkeypatch.setenv(name, "/nonexistent/test-only-artifact-store-setting")
    monkeypatch.setattr(views, "presign_configured_scene_artifact", lambda **kwargs: URL)
    monkeypatch.setattr(transport, "_open_with_policy", store.opener)


def write_staged_absence_proof(attempt, *, promotion_status: str = "promoted", gated: bool = True):
    """A staging dir under ``attempt`` whose objects a completed cleanup proved absent, plus its sealed proof.

    Returns the proof path. ``gated=False`` writes a manifest without
    ``output_promotion_required`` (download mode's).
    """
    from pathlib import Path

    from blueprint_pipeline import provider_output_promotion_records as records
    from blueprint_pipeline.wam_provider_object_store import SCHEMA_VERSION

    staging = Path(attempt) / "object_store_staging"
    staging.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": SCHEMA_VERSION, "status": "completed",
                "object_store": {"key_prefix": "blueprint/task"},
                "bundle_key": "blueprint/task/job/bundles/sha256/" + "a" * 64 + ".zip",
                "output_key": "blueprint/task/job/runpod_provider_runtime_output_" + "0" * 32 + ".zip",
                **({"output_promotion_required": True} if gated else {})}
    (staging / records.STAGING_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    keys = records.staged_object_keys(manifest)
    cleanup = {"schema_version": records.CLEANUP_SCHEMA,
               "staging_manifest_sha256": records.staging_manifest_sha256(staging), "status": "completed",
               "blockers": [], "all_objects_absent": True, "all_ephemeral_objects_absent": True,
               "exact_object_count": len(keys),
               "objects": [{"key_sha256": records.key_sha256(key), "absence": {"absence_confirmed": True}}
                           for _, key in keys]}
    promotion = {"receipt_digest": "sha256:" + "1" * 64, "status": promotion_status} if gated else None
    records.write_staged_object_absence_proof(staging_dir=staging, cleanup=cleanup, promotion=promotion)
    return staging / records.ABSENCE_PROOF_FILENAME


# -- A production-scaled Quick-10 shape, for measuring the member contract ----------------
# Bytes of each JSON member of one cell of the lifecycle rehearsal's real Quick-10 worker
# output (tests/test_native_task_arena_policy_canary_lifecycle_rehearsal.py: the real
# orchestration, episode runner and policy clients over a fake Isaac), measured 2026-09-29.
# Per-episode rows are (pi05_droid, groot_n17_droid). The rehearsal's aggregate result is
# 4,407,038 bytes; production's is 190,573,875 (provider_output_native_inventory.py), so
# every JSON row but the policy requests is scaled by that ratio: the receipts the aggregate
# embeds are the ones the per-episode files and the child results repeat (review I7).
QUICK10_REHEARSAL_AGGREGATE_BYTES = 4_407_038
QUICK10_PRODUCTION_AGGREGATE_BYTES = 190_573_875
_REHEARSAL_CELL_JSON_BYTES = {
    QUICK10_RESULT: 442_171,  # the per-cell child result
    "policy_canary_static_startup_preflight.v1.json": 7_165,
    "policy_canary_telemetry_index.json": 953,
    "policy_canary_telemetry_schema.json": 239,
    "prepolicy_dependency_matrix.v1.json": 67,
    "prepolicy_observation_gate/post_gate_rtx_streaming_guard.v1.json": 547,
}
_REHEARSAL_EPISODE_JSON_BYTES = {
    "action_delivery_readback": (662, 662), "action_sequence": (9_938, 9_938),
    "contact_force_trace": (2_639, 2_639), "embodiment_parity_diagnostic": (687, 687),
    "episode_receipt": (122_115, 182_824), "policy_query_receipt": (20_578, 81_553),
    "reset_state": (5_277, 5_282), "score_receipt": (5_428, 5_428), "state_trace": (14_872, 14_872),
    "task_object_trajectory": (2_327, 2_327),
}
_REHEARSAL_FRAME_MANIFEST_BYTES = {"--prestart-readiness": (13_936, 13_976), "": (20_402, 20_457)}
_REHEARSAL_TOP_JSON_BYTES = {"policy_canary_telemetry_index.json": 1_136, "policy_canary_telemetry_schema.json": 239,
                             "adp009d_groot_worker_identity.groot_n17_droid.json": 578}
_REHEARSAL_POLICY_REQUEST_BYTES = (403_106, 462_852)


def quick10_production_shape(*, frames_per_camera: int = 45, png_bytes: int = 400_000, mp4_bytes: int = 12_000_000,
                             policy_requests_per_episode: int = 45) -> dict[str, int]:
    """Member path -> size of a production-scaled Quick-10 output (see the table above).

    The layout is the worker's: ten ``cell_runs/NN`` trees, each with two candidates' episode
    JSON and, per candidate, an episode and a prestart-readiness media directory holding three
    camera MP4s (120 in all), PNG frames per camera, a frame manifest, and policy requests.
    Bulk sizes only shape the archive; the contract never selects them.
    """
    from fractions import Fraction

    scale = Fraction(QUICK10_PRODUCTION_AGGREGATE_BYTES, QUICK10_REHEARSAL_AGGREGATE_BYTES)

    def scaled(size: int) -> int:
        return int(size * scale)

    files: dict[str, int] = {}
    for cell in range(10):
        root = f"cell_runs/{cell:02d}"
        files.update({f"{root}/{name}": scaled(size) for name, size in _REHEARSAL_CELL_JSON_BYTES.items()})
        files[f"{root}/policy_canary_telemetry.jsonl"] = scaled(1_273)
        files[f"{root}/cell_progress.log"] = 524
        for position, candidate in enumerate(QUICK10_CANDIDATES):
            episode = f"scene-839873-quick10--cell-{cell}--{candidate}"
            files.update({f"{root}/episodes/{episode}.{role}.json": scaled(sizes[position])
                          for role, sizes in _REHEARSAL_EPISODE_JSON_BYTES.items()})
            for suffix, sizes in _REHEARSAL_FRAME_MANIFEST_BYTES.items():
                media = f"{root}/episodes/media/{episode}{suffix}"
                files[f"{media}/multicamera_frame_manifest.json"] = scaled(sizes[position])
                for camera in QUICK10_CAMERAS:
                    files[f"{media}/{camera}.mp4"] = mp4_bytes
                    for frame in range(frames_per_camera):
                        files[f"{media}/frames/{camera}/{frame:06d}.png"] = png_bytes
                if not suffix:
                    for request in range(policy_requests_per_episode):
                        files[f"{media}/policy-requests/{request:06d}.json"] = _REHEARSAL_POLICY_REQUEST_BYTES[position]
    files[QUICK10_RESULT] = QUICK10_PRODUCTION_AGGREGATE_BYTES
    files.update({name: scaled(size) for name, size in _REHEARSAL_TOP_JSON_BYTES.items()})
    files["policy_canary_telemetry.jsonl"] = scaled(12_730)
    return dict(sorted(files.items(), key=lambda item: PurePosixPath(item[0]).parts))


def quick10_production_shaped_archive(**options) -> VirtualObject:
    """``quick10_production_shape`` as a stored ZIP of zero runs: only its directory is real."""
    return build_zip([Entry(name, Zeros(size)) for name, size in quick10_production_shape(**options).items()])
