"""Observe a provider output in object storage by range instead of downloading it.

``RemoteProviderOutputCollector`` is the Vast adapter's
``provider_output_collector``: the adapter calls it in place of the whole-object
download, with the same ``url``, ``output_path`` and ``minimum_free_bytes``
keywords. It opens a ``ProviderOutputRangeReader`` on the staged GET URL, which
learns the object's size and ETag from a one-byte probe and pins every later
request with ``If-Match``, and runs the adapter's two inspections over
``zipfile.ZipFile(reader)``. zipfile reads the end records, the central
directory and the few JSON members the inspections open (the shallowest
runtime result, the first entrypoint diagnostic, ``policy_structured_canary.json``).
The reader fetches aligned blocks (8 MiB by default), so what crosses the
network is a few blocks around those records, never the bulk members.

Nothing is written: no archive copy and no ``.mp4`` copy (the inspection runs
with ``video_extract_dir=None``). ``output_path`` is only the name the
inspection reports as ``zip_path``; no file exists there.

A completed collection returns the download transfer's keys, so the adapter
needs no new branch, plus ``delivery: "remote_only"``, ``remote_object``
{``size_bytes``, ``etag``}, ``inspection``, ``structured_policy_canary``,
``transferred_bytes`` and ``http_request_count``. The same observation stays on
``collector.observation`` for the promotion that follows.

Download-manifest zeros. Because nothing is downloaded, the adapter's download
manifest records ``output_zip_present_after_download: false`` and
``output_zip_size_bytes: 0`` for a completed remote observation. With MP4
members present, ``mp4_validation.blockers`` carries
``mp4_ffprobe_validation_not_requested``, because no MP4 reaches ffprobe; a
bundle kind that expects videos would therefore not prove video smoke, and no
stream-mode caller expects any.

Refusals return ``status: "blocked"`` with one stable code, and the adapter
then continues exactly as after a failed download (SSH recovery when a size
marker was logged): a missing object (``provider_output_not_ready``, HTTP 404),
any transport failure (a 412 or changed ETag, a truncated or malformed range,
the deadline), an object with no ETag to pin, and an inspected JSON member over
the read cap (``provider_output_inspected_member_over_read_cap``). An inspection
that saw any transport failure is discarded, never reported: the path
inspection falls back from an unreadable top-level result to a nested cell
result, which read by range would misreport a blocked run as one completed cell.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from .provider_output_range_transport import (
    ProviderOutputRangeReader,
    ProviderOutputTransportError,
)
from .vast_structured_policy_canary_inspection import (
    STRUCTURED_POLICY_CANARY_MEMBER,
    inspect_structured_policy_canary_archive,
    structured_policy_canary_summary,
)
from .wam_provider_output import (
    ENTRYPOINT_DIAGNOSTIC_FILENAME,
    RUNTIME_RESULT_FILENAMES,
    _invalid_archive,
    inspect_provider_runtime_output_archive,
)

DELIVERY = "remote_only"
READ_CAP_BYTES = 256 * 1024**2
BLOCK_BYTES = 8 * 1024**2
# The Quick-10 packer's own cap on its required terminal result.
MAXIMUM_INSPECTED_MEMBER_BYTES = 512 * 1024**2
DEADLINE_SECONDS = 1800
OVER_READ_CAP = "provider_output_inspected_member_over_read_cap"


class _RemoteCollectionRefusal(ValueError):
    """A typed refusal found before any member is read."""


class _RecordingSource(io.RawIOBase):
    """Pass reads to the pinned reader and keep the first transport failure.

    The inspections catch a failed member read and move on; this keeps the
    failure visible so the whole collection is refused instead.
    """

    def __init__(self, reader: ProviderOutputRangeReader):
        super().__init__()
        self._reader, self.failure = reader, None

    def _recorded(self, call, *args):
        try:
            return call(*args)
        except ProviderOutputTransportError as exc:
            if self.failure is None:
                self.failure = str(exc)
            raise

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._reader.tell()

    def seek(self, offset, whence=io.SEEK_SET):
        return self._recorded(self._reader.seek, offset, whence)

    def read(self, size=-1):
        return self._recorded(self._reader.read, size)


def _inspected(name: str) -> bool:
    """Whether either inspection may read this member's bytes."""
    return (PurePosixPath(name).name in RUNTIME_RESULT_FILENAMES
            or name.endswith(ENTRYPOINT_DIAGNOSTIC_FILENAME)
            or name == STRUCTURED_POLICY_CANARY_MEMBER)


def _capacity_not_required(output_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "vast_provider_output_disk_capacity.v1",
        "status": "not_required",
        "phase": "before_provider_output_get",
        "measurement_path": str(output_path.parent),
        "required_free_bytes": 0,
        "observed_free_bytes": None,
        "blockers": [],
    }


def _count(value, minimum=1) -> bool:
    return type(value) is int and value >= minimum


class RemoteProviderOutputCollector:
    """Inspect one staged provider output by range; see the module docstring."""

    def __init__(self, *, maximum_archive_bytes: int, expected_video_count: int | None,
                 opener: Callable | None = None, maximum_read_bytes: int = READ_CAP_BYTES,
                 block_bytes: int = BLOCK_BYTES,
                 maximum_inspected_member_bytes: int = MAXIMUM_INSPECTED_MEMBER_BYTES,
                 deadline_seconds: float = DEADLINE_SECONDS):
        if (not _count(maximum_archive_bytes) or not _count(maximum_read_bytes)
                or not _count(block_bytes) or block_bytes > maximum_read_bytes
                or not _count(maximum_inspected_member_bytes)
                or not 0 < deadline_seconds <= 86400):
            raise ValueError("provider_output_remote_collection_limits_invalid")
        self._maximum_archive_bytes = maximum_archive_bytes
        self._expected_video_count = expected_video_count
        self._opener = opener
        self._maximum_read_bytes = maximum_read_bytes
        self._block_bytes = block_bytes
        self._maximum_inspected_member_bytes = maximum_inspected_member_bytes
        self._deadline_seconds = deadline_seconds
        self.observation: dict[str, Any] | None = None

    def __call__(self, *, url: str, output_path: str | Path,
                 minimum_free_bytes: int = 0) -> dict[str, Any]:
        # Nothing is written, so no free-space floor applies.
        del minimum_free_bytes
        output_path = Path(output_path)
        self.observation = None
        reader: ProviderOutputRangeReader | None = None
        try:
            reader = ProviderOutputRangeReader(
                url, maximum_archive_bytes=self._maximum_archive_bytes,
                deadline_seconds=self._deadline_seconds, block_bytes=self._block_bytes,
                maximum_read_bytes=self._maximum_read_bytes, opener=self._opener)
            if not reader.identity.get("etag"):
                raise ProviderOutputTransportError("provider_output_remote_etag_missing")
            inspection, structured = self._inspect(reader, output_path)
        except (ProviderOutputTransportError, _RemoteCollectionRefusal) as exc:
            return self._blocked(str(exc), type(exc).__name__, reader, output_path)
        self.observation = {"size_bytes": reader.identity["size_bytes"],
                            "etag": reader.identity["etag"]}
        return {
            "status": "completed",
            "delivery": DELIVERY,
            "download_attempted": False,
            "downloaded_size_bytes": 0,
            "remote_object": dict(self.observation),
            "inspection": inspection,
            "structured_policy_canary": structured,
            "transferred_bytes": reader.transferred_bytes,
            "http_request_count": reader.request_count,
            "disk_capacity": _capacity_not_required(output_path),
            "blockers": [],
            "raw_secret_values_recorded": False,
        }

    def _inspect(self, reader, output_path):
        source = _RecordingSource(reader)
        size = reader.identity["size_bytes"]
        try:
            try:
                archive = zipfile.ZipFile(source)
            except Exception as exc:
                if source.failure is not None:
                    raise
                # Not a readable ZIP: the path inspections' own verdicts.
                return (_invalid_archive(str(output_path), size, exc),
                        structured_policy_canary_summary({}, ["structured_policy_canary_member_invalid"]))
            with archive:
                if any(not info.is_dir() and _inspected(info.filename)
                       and (info.file_size > self._maximum_inspected_member_bytes
                            or info.compress_size > self._maximum_read_bytes)
                       for info in archive.infolist()):
                    raise _RemoteCollectionRefusal(OVER_READ_CAP)
                inspection = inspect_provider_runtime_output_archive(
                    archive, zip_path=str(output_path), zip_size_bytes=size,
                    video_extract_dir=None, expected_video_count=self._expected_video_count)
                structured = inspect_structured_policy_canary_archive(archive)
        except ProviderOutputTransportError:
            if source.failure is None:
                raise
        if source.failure is not None:
            raise ProviderOutputTransportError(source.failure)
        return inspection, structured

    @staticmethod
    def _blocked(code, error_type, reader, output_path) -> dict[str, Any]:
        return {
            "status": "blocked",
            "delivery": DELIVERY,
            "download_attempted": False,
            "downloaded_size_bytes": 0,
            "http_status_code": 404 if code == "provider_output_not_ready" else None,
            "error_type": error_type,
            "transferred_bytes": reader.transferred_bytes if reader is not None else 0,
            "http_request_count": reader.request_count if reader is not None else 0,
            "disk_capacity": _capacity_not_required(output_path),
            "blockers": [code],
            "raw_secret_values_recorded": False,
        }


__all__ = ["DELIVERY", "RemoteProviderOutputCollector"]
