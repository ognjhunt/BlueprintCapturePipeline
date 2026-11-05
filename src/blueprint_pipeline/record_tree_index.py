"""Find small named records anywhere under large trees in one walk.

Ownership and spend checks must observe every lease, pending teardown and
provider-owner id file under their configured roots, wherever a run wrote it.
They used to do that with one ``Path.glob("**/...")`` per pattern, so a single
check walked the same multi-hundred-gigabyte trees several times, and a pass
that ran the check once per intent walked them dozens of times.

``find_records`` keeps the observation complete but pays for it once:

* one walk per root answers every rule;
* a directory's listing is reused only while its ``(device, inode, mtime,
  ctime)`` signature is unchanged. Creating, deleting or renaming an entry
  updates the directory's mtime, so an unchanged signature means an unchanged
  listing. Later scans in the same process cost one ``stat`` per directory
  instead of reading every entry again.
* a listing taken while the directory's mtime was still within
  ``RACY_WINDOW_NS`` of the clock is never reused (coarse timestamp
  granularity could otherwise hide a change made in the same tick).

Matches are returned exactly as the equivalent ``Path.glob`` patterns would:
``**`` never descends through a symlinked directory, but a named parent
component (``**/pending_teardowns/*.json``) is followed when it is a symlink,
and names match files and directories alike. File contents are never cached;
callers read every matched record fresh.
"""
from __future__ import annotations

import os
import stat
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

RACY_WINDOW_NS = 2_000_000_000
MAX_CACHED_DIRECTORIES = 1_000_000


@dataclass(frozen=True)
class Rule:
    """Match ``name`` (exact) or ``suffix``, optionally only under a parent named ``parent``.

    ``Rule(suffix=".lease.json")`` is ``**/*.lease.json``;
    ``Rule(parent="pending_teardowns", suffix=".json")`` is ``**/pending_teardowns/*.json``;
    ``Rule(name="started_pod_id.txt")`` is ``**/started_pod_id.txt``.
    """

    name: str | None = None
    suffix: str | None = None
    parent: str | None = None

    def __post_init__(self) -> None:
        if (self.name is None) == (self.suffix is None):
            raise ValueError("rule_needs_exactly_one_of_name_or_suffix")
        for value in (self.name, self.suffix, self.parent):
            if value is not None and (not value or "/" in value):
                raise ValueError("rule_component_invalid")

    def matches(self, parent: str | None, name: str) -> bool:
        if self.parent is not None and parent != self.parent:
            return False
        return name == self.name if self.name is not None else name.endswith(self.suffix or "")


# (rules, root, directory) -> (signature, subdirectories, matched names, followed symlink parents)
_CACHE: dict[tuple[tuple[Rule, ...], str, str], tuple[tuple[int, int, int, int], tuple[str, ...],
                                                       tuple[str, ...], tuple[str, ...]]] = {}


def clear_cache() -> None:
    _CACHE.clear()


def _signature(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns)


def _matches(rules: tuple[Rule, ...], parent: str | None, name: str) -> bool:
    return any(rule.matches(parent, name) for rule in rules)


def _list(directory: str, rules: tuple[Rule, ...], parent: str | None, parents: frozenset[str]):
    subdirectories, matched, followed = [], [], []
    with os.scandir(directory) as entries:
        for entry in entries:
            name = entry.name
            if _matches(rules, parent, name):
                matched.append(name)
            try:
                if entry.is_dir(follow_symlinks=False):
                    subdirectories.append(name)
                elif name in parents and entry.is_symlink() and entry.is_dir():
                    followed.append(name)
            except OSError:
                continue
    return tuple(sorted(subdirectories)), tuple(sorted(matched)), tuple(sorted(followed))


def _follow(directory: str, name: str, rules: tuple[Rule, ...]) -> list[str]:
    """List a symlinked named parent fresh; its target may live anywhere."""
    path = os.path.join(directory, name)
    # Only the rule naming this parent applies: ``**`` itself never recurses
    # through a symlink, so parentless rules do not see these children.
    named = tuple(rule for rule in rules if rule.parent == name)
    try:
        with os.scandir(path) as entries:
            return sorted(os.path.join(path, entry.name) for entry in entries if _matches(named, name, entry.name))
    except OSError:
        return []


def find_records(roots: Iterable[str | os.PathLike[str]], rules: Iterable[Rule]) -> list[Path]:
    """Every path under ``roots`` matching any rule, sorted and de-duplicated per root."""
    rules = tuple(rules)
    if not rules:
        return []
    parents = frozenset(rule.parent for rule in rules if rule.parent is not None)
    found: list[Path] = []
    if len(_CACHE) > MAX_CACHED_DIRECTORIES:
        _CACHE.clear()
    for root in roots:
        root_text = os.fspath(root)
        results: list[str] = []
        stack = [root_text]
        while stack:
            directory = stack.pop()
            key = (rules, root_text, directory)
            try:
                before = os.stat(directory)
            except OSError:
                _CACHE.pop(key, None)
                continue
            if not stat.S_ISDIR(before.st_mode):
                _CACHE.pop(key, None)
                continue
            signature = _signature(before)
            cached = _CACHE.get(key)
            if cached is not None and cached[0] == signature:
                subdirectories, matched, followed = cached[1:]
            else:
                parent = None if directory == root_text else os.path.basename(directory)
                try:
                    subdirectories, matched, followed = _list(directory, rules, parent, parents)
                except OSError:
                    _CACHE.pop(key, None)
                    continue
                if time.time_ns() - before.st_mtime_ns > RACY_WINDOW_NS:
                    _CACHE[key] = (signature, subdirectories, matched, followed)
                else:
                    _CACHE.pop(key, None)
            results.extend(os.path.join(directory, name) for name in matched)
            for name in followed:
                results.extend(_follow(directory, name, rules))
            stack.extend(os.path.join(directory, name) for name in reversed(subdirectories))
        found.extend(sorted({Path(path) for path in results}))
    return found
