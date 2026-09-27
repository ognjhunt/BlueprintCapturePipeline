"""Fail-closed queue inventory for website scene workspace retirement."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Sequence

MAX_QUEUE_FILE_BYTES = 16 * 1024 * 1024
MAX_QUEUE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_QUEUE_ENTRIES = 10_000


class QueueInventoryUnavailable(RuntimeError):
    """A queue may contain a live scene reference that cannot be inspected."""


def queue_reference_text(roots: Sequence[Path], *, max_file_bytes: int = MAX_QUEUE_FILE_BYTES) -> str:
    chunks: list[str] = []
    total = 0
    entries = 0
    for root in roots:
        for state in ("pending", "processing"):
            directory = Path(root) / state
            try:
                info = os.lstat(directory)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise QueueInventoryUnavailable("queue_inventory_unreadable") from exc
            if not stat.S_ISDIR(info.st_mode):
                raise QueueInventoryUnavailable("queue_inventory_unreadable")
            try:
                directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    with os.scandir(directory_fd) as iterator:
                        for entry in iterator:
                            if not entry.name.endswith(".json"):
                                continue
                            entries += 1
                            if entries > MAX_QUEUE_ENTRIES:
                                raise QueueInventoryUnavailable("queue_inventory_unreadable")
                            file_fd = os.open(entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                              dir_fd=directory_fd)
                            try:
                                info = os.fstat(file_fd)
                                if not stat.S_ISREG(info.st_mode) or info.st_size > max_file_bytes:
                                    raise QueueInventoryUnavailable("queue_inventory_unreadable")
                                chunks_read: list[bytes] = []
                                read_bytes = 0
                                while read_bytes < info.st_size:
                                    chunk = os.read(file_fd, min(1 << 20, info.st_size - read_bytes))
                                    if not chunk:
                                        raise QueueInventoryUnavailable("queue_inventory_unreadable")
                                    chunks_read.append(chunk)
                                    read_bytes += len(chunk)
                                after = os.fstat(file_fd)
                                if (os.read(file_fd, 1) or after.st_size != info.st_size
                                        or after.st_mtime_ns != info.st_mtime_ns
                                        or after.st_ctime_ns != info.st_ctime_ns):
                                    raise QueueInventoryUnavailable("queue_inventory_unreadable")
                                data = b"".join(chunks_read)
                            finally:
                                os.close(file_fd)
                            total += len(data)
                            if len(data) > max_file_bytes or total > MAX_QUEUE_TOTAL_BYTES:
                                raise QueueInventoryUnavailable("queue_inventory_unreadable")
                            chunks.append(data.decode("utf-8"))
                finally:
                    os.close(directory_fd)
            except (OSError, UnicodeDecodeError) as exc:
                raise QueueInventoryUnavailable("queue_inventory_unreadable") from exc
    return "\n".join(chunks)
