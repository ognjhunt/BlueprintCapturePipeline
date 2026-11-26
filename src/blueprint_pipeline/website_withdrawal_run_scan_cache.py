"""What each run's records name, kept between withdrawal plans so a plan reads only the runs that changed.

Beta journey step: site withdrawal. ``website_capture_withdrawal_deletion`` binds the runs a
withdrawal deletes by reading every JSON and JSONL record of every run under the run roots, for the
site's identifiers, the runs they name, other sites and providers. On the host that was every run
of every job ever kept there (at least 9 GiB of records on 2026-11-26, mostly the retired early
trial's policy canaries), read again for each plan and again for its apply, at half a core of the
lowest priority. The first withdrawal ever planned there was killed at its unit's 45-minute start
timeout on each of its three tries on 2026-11-25, so it was never planned and never deleted.

This module keeps, per run directory, what the plan's reading found there, and the plan reads a
run's records again only when they changed. A plan that runs out of its time reading stops at a
run's edge (``ScanBudgetSpent``), after keeping at least one new run; everything it kept stays,
and the next hourly pass goes on from there. A run it cannot keep is read again by every plan, as
before, and never counts as that progress. Nothing here deletes a run or decides anything: the plan
still binds every run from these facts exactly as it would from reading.

Per run directory (one file, named by the run's root and name):

- a fingerprint of exactly the records the plan reads: each JSON and JSONL file's relative path,
  device, inode, size, mode, modification and change times. A record written, replaced, touched
  or re-permissioned changes it (the change time cannot be set back), and the run is read again.
  A fingerprint taken before reading that differs from one taken after keeps nothing: the run is
  read again next time.
- every token of those records that could be a word of any plan's vocabulary (``cacheable``): at
  least ``_MIN_CLOSURE_NAME`` (12) characters, a hyphen, and a letter that is not a hex digit.
  Every site's scene and capture id, every scene intent id and every run name on the host has
  that shape (2026-11-26); numbers, digests, UUIDs and file names without a hyphen do not, which
  keeps the list short. A plan whose vocabulary holds a word without that shape reads every run
  itself, as before, and uses no cache (``plan_withdrawal_deletion``).
- every site id the records name in the shapes the plan reads (``_OTHER_SITE``) and every
  provider (``_PROVIDER``), unfiltered: the plan drops its own site.

So the cache answers exactly what reading answers: the words a run names are its cached tokens
that are in the plan's vocabulary, and the reading here uses the plan's own windows (``_CHUNK``,
``_OVERLAP``) and patterns. Only the token pass differs in how it runs: one pattern for tokens of
``_MIN_CLOSURE_NAME`` or more characters, with the window's trailing partial token cut off, which
finds the same tokens as ``_TOKEN`` with the end check (``tests/test_website_withdrawal_run_scan_cache.py``
holds the two to the same answers). Reading a run costs what it cost before (about 30 MiB/s per
core on synthetic policy-canary records, 2026-11-26): the gain is in not reading it again. A run
whose cacheable tokens exceed ``MAX_TOKENS`` is answered from this reading and not kept.

The cache sits under the withdrawal unit's own report root (``run-scan-cache/``), private to the
service account. An entry whose run is gone is removed by the next plan that lists its root whole,
and an apply removes the entry of each run it deletes, so no entry outlives its run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

SCHEMA = "website_withdrawal_run_scan_cache.v1"
DIRNAME = "run-scan-cache"
MAX_TOKENS = 1_000_000
_TOKEN_CHARS = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
_NON_HEX_LETTER = re.compile(rb"[G-Zg-z]")


class ScanBudgetSpent(Exception):
    """The plan's reading ran out of time at a run's edge; what it read is kept for the next pass.

    Not a ``ValueError`` or ``OSError``: the plan records those as unresolved, and this must end
    the plan instead.
    """

    def __init__(self, progress: Mapping[str, int]):
        super().__init__("website_withdrawal_run_scan_budget_spent")
        self.progress = dict(progress)


def _patterns() -> dict[str, Any]:
    from . import website_capture_withdrawal_deletion as deletion

    return {"token": deletion._TOKEN, "other_site": deletion._OTHER_SITE, "provider": deletion._PROVIDER,
            "chunk": deletion._CHUNK, "overlap": deletion._OVERLAP, "min_token": deletion._MIN_CLOSURE_NAME,
            "suffixes": deletion._SCANNED_SUFFIXES}


def min_token() -> int:
    return int(_patterns()["min_token"])


def cacheable(word: bytes) -> bool:
    """Whether a cached run can answer for ``word``: the shape every kept token has."""
    return len(word) >= min_token() and b"-" in word and _NON_HEX_LETTER.search(word) is not None


_LONG: re.Pattern[bytes] | None = None


def _long() -> re.Pattern[bytes]:
    global _LONG
    if _LONG is None:
        _LONG = re.compile(rb"(?<![A-Za-z0-9._-])[A-Za-z0-9._-]{%d,}(?![A-Za-z0-9._-])" % min_token())
    return _LONG


def scanner_identity() -> str:
    """Changes whenever anything that decides a run's facts changes, so no older entry is read."""
    p = _patterns()
    parts = [SCHEMA, p["token"].pattern.decode(), p["other_site"].pattern.decode(), p["provider"].pattern.decode(),
             _long().pattern.decode(), _NON_HEX_LETTER.pattern.decode(), str(p["chunk"]), str(p["overlap"]),
             str(p["min_token"]), ",".join(p["suffixes"])]
    return "sha256:" + hashlib.sha256("\0".join(parts).encode()).hexdigest()


class Facts:
    """What one run's records name, unfiltered: tokens of the cacheable shape, site ids, providers."""

    def __init__(self, tokens: Iterable[bytes] = (), sites: Iterable[bytes] = (), providers: Iterable[bytes] = ()):
        self.tokens, self.sites, self.providers = set(tokens), set(sites), set(providers)

    def feed(self, stream: BinaryIO) -> None:
        """The plan's own windows (``_Scan.feed``): a match touching a window's end waits for the overlap."""
        p = _patterns()
        tail = b""
        while True:
            chunk = stream.read(p["chunk"])
            window, final = tail + chunk, not chunk
            if window:
                self._window(window, final, p)
            if final:
                return
            tail = window[-p["overlap"]:]

    def _window(self, window: bytes, final: bool, p: Mapping[str, Any]) -> None:
        # Cutting the trailing run of token characters leaves exactly the tokens that end before the
        # window's end: each ends at a character that is not a token character, all still inside.
        body = window if final else window.rstrip(_TOKEN_CHARS)
        for token in _long().findall(body):
            if b"-" in token and _NON_HEX_LETTER.search(token) is not None:
                self.tokens.add(token)
        # Every shape ``_OTHER_SITE`` reads holds one of these two words.
        if b"site-" in window or b"walkthrough-" in window:
            for match in p["other_site"].finditer(window):
                if final or match.end() < len(window):
                    self.sites.add(next(group for group in match.groups() if group is not None))
        for match in p["provider"].finditer(window):
            self.providers.add(match.group(1))


def _fingerprint(directory: Path, records: Sequence[str]) -> str:
    rows = []
    for relative in records:
        info = os.lstat(directory / relative)
        rows.append([relative, info.st_dev, info.st_ino, info.st_size, info.st_mode, info.st_mtime_ns,
                     info.st_ctime_ns])
    return "sha256:" + hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _seal(entry: Mapping[str, Any]) -> str:
    body = {key: value for key, value in entry.items() if key != "digest"}
    return "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class RunScanCache:
    """One plan's view of the cache: answers per run, a reading deadline, and what it read and reused."""

    def __init__(self, root: Path, *, deadline: float | None = None, clock=time.monotonic):
        self.root, self.deadline, self.clock = Path(root), deadline, clock
        self.identity = scanner_identity()
        self.progress = {"runs_read": 0, "runs_kept": 0, "runs_reused": 0, "runs_not_kept": 0}

    def _root_dir(self, run_root: str) -> Path:
        return self.root / _key(str(run_root))[:32]

    def _path(self, run_root: str, name: str) -> Path:
        return self._root_dir(run_root) / f"{_key(name)}.json"

    def _read(self, path: Path, run_root: str, name: str, fingerprint: str) -> Facts | None:
        try:
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode):
                return None
            entry = json.loads(path.read_bytes())
        except (OSError, ValueError):
            return None
        if not (isinstance(entry, dict) and entry.get("schema_version") == SCHEMA
                and entry.get("scanner") == self.identity and entry.get("root") == str(run_root)
                and entry.get("name") == name and entry.get("fingerprint") == fingerprint
                and entry.get("digest") == _seal(entry)):
            return None
        try:
            return Facts(*(map(str.encode, entry[key]) for key in ("tokens", "sites", "providers")))
        except (KeyError, TypeError, AttributeError):
            return None

    def _write(self, path: Path, run_root: str, name: str, fingerprint: str, facts: Facts) -> None:
        entry: dict[str, Any] = {"schema_version": SCHEMA, "scanner": self.identity, "root": str(run_root),
                                 "name": name, "fingerprint": fingerprint,
                                 "tokens": sorted(token.decode() for token in facts.tokens),
                                 "sites": sorted(site.decode() for site in facts.sites),
                                 "providers": sorted(provider.decode() for provider in facts.providers)}
        entry["digest"] = _seal(entry)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(entry, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def facts(self, run_root: str, name: str, directory: Path, records: Sequence[str]) -> Facts:
        """The run's facts: kept ones when its records are unchanged, else read now (and kept).

        Raises ``OSError`` as reading would (a record gone, unreadable or no longer a regular file),
        and ``ScanBudgetSpent`` when the run would have to be read after the deadline and this plan
        has already kept one.
        """
        before = _fingerprint(directory, records)
        path = self._path(run_root, name)
        kept = self._read(path, run_root, name, before)
        if kept is not None:
            # Reading would open each record; one this account cannot open is unresolved, as before.
            for relative in records:
                os.close(os.open(directory / relative, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC))
            self.progress["runs_reused"] += 1
            return kept
        # A plan keeps at least one new run before it may stop, so every pass moves on however late it
        # started. A run read but not kept (too many tokens, or changed while read) does not count: a
        # pass that only re-read it would stop where the last one did, and the plan would never finish.
        if self.deadline is not None and self.progress["runs_kept"] and self.clock() >= self.deadline:
            raise ScanBudgetSpent(self.progress)
        from .website_capture_withdrawal_deletion import _require

        facts = Facts()
        for relative in records:
            record = directory / relative
            _require(stat.S_ISREG(os.lstat(record).st_mode), "unsafe_entry")
            with record.open("rb") as stream:
                facts.feed(stream)
        self.progress["runs_read"] += 1
        if len(facts.tokens) <= MAX_TOKENS and _fingerprint(directory, records) == before:
            self._write(path, run_root, name, before, facts)
            self.progress["runs_kept"] += 1
        else:
            self.progress["runs_not_kept"] += 1
        return facts

    def prune(self, run_root: str, names: Iterable[str]) -> int:
        """Remove the entries of runs no longer under ``run_root``; only for a root listed whole."""
        directory = self._root_dir(run_root)
        keep = {f"{_key(name)}.json" for name in names}
        removed = 0
        try:
            children = list(os.scandir(directory))
        except FileNotFoundError:
            return 0
        for child in children:
            if child.name in keep:
                continue
            if child.is_dir(follow_symlinks=False):
                shutil.rmtree(child.path)
            else:
                os.unlink(child.path)
            removed += 1
        return removed

    def forget(self, run_root: str, name: str) -> None:
        """Remove one run's entry: its run was deleted."""
        self._path(run_root, name).unlink(missing_ok=True)


__all__ = ["DIRNAME", "MAX_TOKENS", "SCHEMA", "Facts", "RunScanCache", "ScanBudgetSpent", "cacheable", "min_token",
           "scanner_identity"]
