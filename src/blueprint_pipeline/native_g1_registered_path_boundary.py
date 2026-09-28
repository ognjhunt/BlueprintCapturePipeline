"""Provider-shipped refusal of host-only registered namespaces.

This leaf never supplies registration/lifetime authority. Provider default None
can execute ordinary paths without importing the host's private administrator.
"""
from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path


def refuse_unowned_registered_path(path):
    if not isinstance(path, Path):
        raise ValueError('experiment_consumer_path_unsafe')
    started = time.monotonic()
    text = str(path)
    if len(text) > 4096 or len(os.fsencode(text)) > 4096 or len(path.parts) > 64 or '..' in path.parts:
        raise ValueError('experiment_consumer_path_unsafe')
    parts = path.parts
    if any(parts[index] in ('g1', 'arena') and parts[index + 1].startswith('registered-')
           for index in range(len(parts) - 1)):
        raise ValueError('experiment_consumer_authority_required')
    absolute = path if path.is_absolute() else Path.cwd() / path
    if len(absolute.parts) > 64:
        raise ValueError('experiment_consumer_path_unsafe')
    parent = Path(absolute.anchor)
    last = started
    for component in absolute.parts[1:]:
        current = time.monotonic()
        if current < last or current - started >= 5:
            raise ValueError('experiment_consumer_path_unsafe')
        last = current
        parent = parent / component
        try:
            info = os.stat(parent, follow_symlinks=False)
        except FileNotFoundError:
            break
        except OSError:
            raise ValueError('experiment_consumer_path_unsafe') from None
        if stat.S_ISLNK(info.st_mode):
            system_var = (sys.platform == 'darwin' and parent == Path('/var')
                          and info.st_uid == 0 and os.readlink(parent) == 'private/var')
            if not system_var:
                raise ValueError('experiment_consumer_path_unsafe')
    current = time.monotonic()
    if current < last or current - started >= 5:
        raise ValueError('experiment_consumer_path_unsafe')
