"""Remove the scratch trees a stopped provision left behind.

A deploy rebuilds a new commit's scene-configuration components under the
temporary directory and publishes runtime trees beside the earlier ones. A run
stopped before its cleanup (``systemctl stop``, SIGKILL) left both behind: on
2026-10-23 three ~11 GB component copies filled the control plane's root disk,
and three half-published ``.<commit>.<random>`` folders (7.5 GB) blocked the
release retention plan. Deploys run one at a time and a provision is bounded at
two hours, so staging this old belongs to a dead run.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

STALE_STAGING_SECONDS = 6 * 3600
STAGING_PREFIXES = ("scene-configuration-components-", "blueprint-artisan-component-")
RUNTIME_COMPONENTS = ("scene-configuration", "splat-render")
#: ``tempfile.mkdtemp(prefix=f".{commit}.")`` beside a runtime tree.
PUBLICATION_STAGING = re.compile(r"\.[0-9a-f]{40}\.[a-z0-9_]{8}")


def remove_tree(root: Path) -> None:
    """Delete an owned tree, read-only directories included."""
    if not root.exists() or root.is_symlink():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o700)
    root.chmod(0o700)
    shutil.rmtree(root)


def _named(parent: Path, staged: Callable[[str], bool]) -> list[str]:
    try:
        return sorted(name for name in os.listdir(parent) if staged(name))
    except (FileNotFoundError, NotADirectoryError):
        return []


def sweep_stale_staging(temp_root: Path | None = None, *, runtime_root: Path | None = None,
                        now: float | None = None) -> list[str]:
    """Remove staging a killed provision left behind; return the names removed.

    Names are matched before anything is inspected, and an entry that vanishes
    meanwhile (another tool's temporary file, a concurrent sweep) is skipped.
    Runtime-tree staging is named ``<component>/<name>``.
    """
    root = Path(temp_root or tempfile.gettempdir())
    now = time.time() if now is None else now
    candidates = [(root / name, name) for name in _named(root, lambda name: name.startswith(STAGING_PREFIXES))]
    for component in RUNTIME_COMPONENTS if runtime_root is not None else ():
        parent = Path(runtime_root) / component
        candidates += [(parent / name, f"{component}/{name}")
                       for name in _named(parent, lambda name: PUBLICATION_STAGING.fullmatch(name) is not None)]
    removed = []
    for path, name in candidates:
        try:
            info = path.lstat()
            if (stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
                    and now - info.st_mtime >= STALE_STAGING_SECONDS):
                remove_tree(path)
                removed.append(name)
        except FileNotFoundError:
            continue
    return removed
