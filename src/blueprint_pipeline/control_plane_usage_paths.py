"""Read-only directory anchors for bounded usage observation.

Names are always opened relative to retained directory descriptors, without
following links. A path identity is checked at use, never inferred from a
cursor digest or an earlier lstat alone.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import PurePosixPath


class UsagePathError(OSError):
    """A directory could not be bound to the selected path identity."""


def same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, stat.S_IFMT(left.st_mode)) == (
        right.st_dev, right.st_ino, stat.S_IFMT(right.st_mode))


@dataclass(frozen=True)
class UsageDirectory:
    fd: int
    links: tuple[tuple[int, str, int], ...]
    parent: UsageDirectory | None = None

    def check(self) -> None:
        if self.parent is not None:
            self.parent.check()
        try:
            for parent_fd, name, child_fd in self.links:
                linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if not same_identity(linked, os.fstat(child_fd)):
                    raise UsagePathError("directory_identity_changed")
        except UsagePathError:
            raise
        except OSError as exc:
            raise UsagePathError("directory_identity_unverified") from exc


@contextmanager
def _open_components(
    start_fd: int, components: Sequence[str], *, expected: os.stat_result,
    parent: UsageDirectory | None = None,
) -> Iterator[UsageDirectory]:
    descriptors = []
    links = []
    current = start_fd
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        try:
            if parent is not None:
                parent.check()
            for name in components:
                child_fd = os.open(name, flags, dir_fd=current)
                descriptors.append(child_fd)
                links.append((current, name, child_fd))
                current = child_fd
            if not same_identity(os.fstat(current), expected):
                raise UsagePathError("directory_identity_changed")
            anchor = UsageDirectory(current, tuple(links), parent)
            anchor.check()
        except UsagePathError:
            raise
        except OSError as exc:
            raise UsagePathError("directory_identity_unverified") from exc
        yield anchor
        anchor.check()
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


@contextmanager
def anchored_directory(path: str, expected: os.stat_result) -> Iterator[UsageDirectory]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        root_fd = os.open("/", flags)
    except OSError as exc:
        raise UsagePathError("directory_identity_unverified") from exc
    try:
        with _open_components(root_fd, PurePosixPath(path).parts[1:], expected=expected) as anchor:
            yield anchor
    finally:
        os.close(root_fd)


@contextmanager
def descendant_directory(
    root: UsageDirectory, components: Sequence[str], expected: os.stat_result,
) -> Iterator[UsageDirectory]:
    if any(name in ("", ".", "..") or "/" in name for name in components):
        raise UsagePathError("directory_component_invalid")
    with _open_components(root.fd, components, expected=expected, parent=root) as anchor:
        yield anchor
