"""The bounded JSON record reader website scene workspace retirement uses for every record."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

#: The most ``_load_json`` reads; retirement's receipt readers (the listener's receipt lookup,
#: restore) read through it, so a receipt may never outgrow it.
_MAX_RECORD_BYTES = 16 * 1024 * 1024


def _load_json(path: Path) -> tuple[str, dict[str, Any] | None, os.stat_result | None]:
    """``("ok", object, stat)``, ``("absent", None, None)`` or ``("unreadable", None, stat)``.

    Never follows a symlink and reads only regular files of bounded size.
    """

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (FileNotFoundError, NotADirectoryError):
        return "absent", None, None
    except OSError:
        return "unreadable", None, None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_RECORD_BYTES:
            return "unreadable", None, info
        data = bytearray()
        while chunk := os.read(descriptor, 1 << 20):
            data += chunk
            if len(data) > _MAX_RECORD_BYTES:
                return "unreadable", None, info
    except OSError:
        return "unreadable", None, None
    finally:
        os.close(descriptor)
    try:
        value = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return "unreadable", None, info
    return ("ok", value, info) if isinstance(value, dict) else ("unreadable", None, info)
