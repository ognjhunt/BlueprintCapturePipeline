"""Download exact native-runtime bytes concurrently through the existing range transport."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time

from .provider_output_range_transport import ProviderOutputRangeReader


class RuntimeDependencyDownloadError(ValueError):
    """Stable failure codes without signed URLs."""


def download_runtime_dependency(*, url, output_path, expected_size, expected_sha256,
                                workers=8, maximum_seconds=600, opener=None):
    """Publish only a whole, digest-verified object; preserve an existing partial on failure."""
    if (type(expected_size) is not int or expected_size <= 0
            or type(workers) is not int or not 1 <= workers <= 32
            or not 0 < maximum_seconds <= 3600
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(expected_sha256))):
        raise RuntimeDependencyDownloadError("runtime_dependency_download_limits_invalid")
    output = Path(output_path)
    if output.is_symlink() or not output.parent.is_dir():
        raise RuntimeDependencyDownloadError("runtime_dependency_download_path_invalid")
    deadline = time.monotonic() + maximum_seconds

    def reader():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeDependencyDownloadError("runtime_dependency_download_deadline_exceeded")
        return ProviderOutputRangeReader(url, maximum_archive_bytes=expected_size,
            deadline_seconds=remaining, block_bytes=1024**2, opener=opener)

    with reader() as initial:
        identity = dict(initial.identity)
    if identity["size_bytes"] != expected_size:
        raise RuntimeDependencyDownloadError("runtime_dependency_download_size_mismatch")
    count = min(workers, expected_size)
    stopped = threading.Event()
    fd, temporary = tempfile.mkstemp(prefix=".native-runtime-transfer-", dir=output.parent)
    try:
        os.ftruncate(fd, expected_size)

        def fetch(index):
            start, end = expected_size * index // count, expected_size * (index + 1) // count
            offset = start

            def write(chunk):
                nonlocal offset
                if stopped.is_set() or time.monotonic() >= deadline:
                    raise RuntimeDependencyDownloadError("runtime_dependency_download_cancelled")
                view = memoryview(chunk)
                while view:
                    written = os.pwrite(fd, view, offset)
                    if written <= 0:
                        raise RuntimeDependencyDownloadError("runtime_dependency_download_short_write")
                    offset += written
                    view = view[written:]

            try:
                with reader() as current:
                    if current.identity != identity:
                        raise RuntimeDependencyDownloadError("runtime_dependency_download_version_changed")
                    current.stream_to(write, start=start, end=end)
                if offset != end:
                    raise RuntimeDependencyDownloadError("runtime_dependency_download_short_range")
            except Exception:
                stopped.set()
                raise

        with ThreadPoolExecutor(max_workers=count) as pool:
            list(pool.map(fetch, range(count)))
        digest = hashlib.sha256()
        offset = 0
        while offset < expected_size:
            if time.monotonic() >= deadline:
                raise RuntimeDependencyDownloadError("runtime_dependency_download_deadline_exceeded")
            chunk = os.pread(fd, min(1024**2, expected_size - offset), offset)
            if not chunk:
                raise RuntimeDependencyDownloadError("runtime_dependency_download_short_readback")
            digest.update(chunk)
            offset += len(chunk)
        if "sha256:" + digest.hexdigest() != expected_sha256:
            raise RuntimeDependencyDownloadError("runtime_dependency_download_digest_mismatch")
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, output)
        return {"status": "verified", "size_bytes": expected_size, "sha256": expected_sha256,
                "parallel_ranges": count, "immutable_version_pinned": True}
    finally:
        if fd is not None:
            os.close(fd)
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
        result = download_runtime_dependency(url=os.environ["BLUEPRINT_RUNTIME_DEPENDENCY_URI"],
            output_path=args.output, expected_size=receipt["packet_size_bytes"],
            expected_sha256=receipt["packet_sha256"])
    except Exception as exc:
        # Provider URLs can contain credentials; never format an exception's text.
        print("BLUEPRINT_VAST_RUNTIME_DEPENDENCY_PARALLEL_BLOCKED:" + type(exc).__name__)
        return 1
    print("BLUEPRINT_VAST_RUNTIME_DEPENDENCY_PARALLEL_VERIFIED:" + result["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

