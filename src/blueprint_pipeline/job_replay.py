"""Replay a real website job's remaining stages offline, in one command, paying for nothing.

    . ~/.blueprint-cloud/env.sh
    PYTHONPATH=src "$BLUEPRINT_PIPELINE_PYTHON" -m blueprint_pipeline.job_replay <request id or host capture root>
    ... --past website_retained_static_masks_manifest_changed   # step past a known stop, flagged

Input is a website request id (``itest-...``), found under the host's handoff root, or a host capture root
(``.../scenes/site-<id>/captures/walkthrough-<id>``). The tool then:

1. pulls, through the operator door's two read routes only (``/fs/list``, ``/fs/read``), what the replay needs
   into ``<scratch>/cache/<host path>``: every record (JSON and text, and any file of 256 KiB or less) of the
   capture and its scene's withdrawal folder. Bulk files (frames, splats, meshes, zips) are pulled only when a
   stage names them: a missing input, or a path the code looked for and did not find, that exists on the host is
   pulled once (with its same-kind neighbours, for a frame set) and the replay runs again, at most ``--rounds``
   times. A cached file whose host size and mtime have not changed is reused, after its digest is checked.
2. replays, in a private mount and network namespace (``unshare``, as ``atlas_preview_replay`` uses it) where the
   pulled files stand at their own host paths (an overlay: the wanted cached files, hard-linked, as its read-only
   layer, and a per-run upper layer for every write), each remaining stage of the job's website path in order,
   each in its own process:

   - ``atlas_preview``: the Atlas preview controller, ``atlas_preview_replay.run_steps`` until it stops;
   - ``preparation``: once the preview is ready (or the job has none), the website preparation as the listener's
     recovery runs it once its claim and admission pass: ``run_e2e.run_end_to_end(resume_completed_stages=True)``;
   - ``scene_handoff``: the same run from its call of ``website_scene_handoff.prepare_website_scene_handoff``;
   - ``company_simulation``: ``website_company_simulation.publish_company_simulation``, the package compile.

   Each stage says whether it ran, what it read, where it stopped, what it would buy, how long it took and its peak
   resident memory. A stage that stops ends the chain; later stages are ``skipped`` and say why.
3. checks the pulled evidence's digest before and after: the replay process never sees the cache or its links (a
   tmpfs is mounted over both inside the namespace), so it cannot change them, and the check proves it did not.

``--past <code>`` (repeatable) steps past a named stop, explicitly and on the record. Where a step has a known
retained output (``SUBSTITUTIONS``: ``atlas_completion_gap_survey_changed``, the completion's own retained report;
``website_retained_static_masks_manifest_changed``, the masks the host completed), that output is used in place of
the recomputation and the stage goes on. Any other named stop marks its stage ``assumed`` and the next stage runs on
the files as the host left them. Every assumed step is listed in ``assumed`` and the report says ``clean: false``,
so nobody mistakes it for a clean run.

Boundaries; no flag lifts any of them: the namespace has no network (and every socket connection, name lookup and
network tool in a stage's process is refused and recorded; any attempt fails the run, exit 2); credential-like
variables are removed and the container's credential files are masked; WebApp reads are answered from the job's
retained records (``atlas_preview_replay``: task context, Atlas configuration; here also the scene sponsorship); a
settlement, which only releases what a bought call held beyond its cost, is recorded, never sent, and answered as
no answer (every caller then keeps the hold, as when the WebApp does not answer); every other WebApp,
artifact-store or provider effect stops as ``external_effect``; every new purchase stops as ``would_buy`` with its
provider, operation and maximum USD, never made. The door is only read: nothing here holds,
releases, deploys or starts anything. Without a private namespace the tool says so and replays nothing.

What it cannot show: provider behaviour, the host's own memory and disk (peak RSS is this container's), and any
record the door refuses to serve (listed under ``pull.door_refused``). ``docs/runbooks/job-replay.md``.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import importlib.util
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "job_replay_report.v1"
STAGES = ("atlas_preview", "preparation", "scene_handoff", "company_simulation")
#: Stage groups, each replayed in its own process (the preparation and its handoff are one ``run_e2e`` call).
GROUPS = (("atlas_preview",), ("preparation", "scene_handoff"), ("company_simulation",))
HANDOFF_ROOT = "/var/lib/blueprint/pubsub-handoffs"
CODE_ROOT = Path(__file__).resolve().parents[2]
DOOR_CLIENT = CODE_ROOT / "scripts" / "operator_door.py"
#: The door's only routes this tool calls; both are reads.
DOOR_ROUTES = frozenset({"/fs/list", "/fs/read"})
DOOR_CHUNK_BYTES = 16 * 1024 * 1024
RECORD_SUFFIXES = (".json", ".jsonl", ".txt", ".md", ".yaml", ".yml", ".csv", ".sha256", ".lock", ".log")
SMALL_BYTES = 256 * 1024
#: A record larger than this is bulk all the same (the door serves at most 16 MiB a read; records are smaller).
MAXIMUM_RECORD_BYTES = 64 * 1024 * 1024
DEFAULT_ROUNDS = 8
#: A listed folder's unpulled files are pulled when together they are at most this large.
LISTED_PULL_BYTES = 64 * 1024 * 1024
#: Paths outside the capture a stage may name (a scene intent, the team scene queue); each is looked up once.
MAXIMUM_OUTSIDE_LOOKUPS = 40
INDEX_FILE = ".job_replay_index.json"
SCENE_RETIREMENT_ROOT = "/var/lib/blueprint/scene-retirement"


def _own_retirement_records(relative: str, capture_root: str) -> bool:
    """The scene retirement store holds every capture's birth evidence; only this capture's (named by its path's
    digest, ``task_evaluation_scene_retirement_generations``) and the coordinator fence are pulled."""
    key = hashlib.sha256(capture_root.encode()).hexdigest()
    return relative.startswith("coordinator/") or relative.startswith(f"generations/{key}")


@dataclass(frozen=True)
class Selector:
    folders: tuple[str, ...]
    keep: Callable[[str, str], bool]


#: Host folders shared by every job: only the sub-folders listed are walked, and only this job's files pulled.
OUTSIDE_SELECTORS: dict[str, Selector] = {
    SCENE_RETIREMENT_ROOT: Selector(("coordinator", "generations"), _own_retirement_records)}
#: Folder modes the host's own fences check (a door pull carries no mode or owner); set on the work copy only,
#: and listed in the report. The copy's owner stands for the host's service account (``_SERVICE_IDENTITY``).
HOST_MODES = {SCENE_RETIREMENT_ROOT + "/generations": 0o700}
#: Inside the namespace every hidden entry of the home folder is masked (credentials live there: cloud, git and
#: tool logins) except these, which Python tooling reads; then these credential folders under them, and the host's.
HOME_KEPT = frozenset({".cache", ".local", ".config", ".bashrc", ".profile", ".cargo", ".rustup"})
CREDENTIAL_PATHS = ("~/.config/gcloud", "~/.config/gh", "~/.config/configstore", "/etc/blueprint/provider-secrets")
#: A WebApp settlement in a replay: recorded, never sent, answered as an unanswered write.
SETTLEMENT_NOT_SENT = "website_control_preparation-settlement_not_sent_in_replay"
STOP_KINDS = ("completed", "would_buy", "refusal", "missing_input", "external_effect", "error", "waiting")
_EXIT_BREACH, _EXIT_UNSUPPORTED = 2, 5


# ----------------------------------------------------------------------------------------------- the door (reads)

def _door_client() -> Any:
    spec = importlib.util.spec_from_file_location("blueprint_operator_door_client", DOOR_CLIENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class DoorReader:
    """The operator door's two read routes and nothing else: no request kind, hold, deploy or unit."""

    def __init__(self, client: Any = None) -> None:
        self._client = client or _door_client()
        self.calls = self.retries = 0

    def _get(self, route: str, query: Mapping[str, Any]) -> Any:
        if route not in DOOR_ROUTES:
            raise ValueError(f"job_replay_door_route_not_read:{route}")
        self.calls += 1
        return self._client._request("GET", route, dict(query))

    def _retried(self, call: Callable[[], Any]) -> Any:
        """A read again after a dropped connection or a door 5xx (``DoorError`` 4 and 5), at most four times."""
        for attempt in range(5):
            try:
                return call()
            except Exception as error:  # noqa: BLE001 - only the door's transient failures are retried
                transient = getattr(error, "exit_code", None) in (4, 5) or isinstance(error, (OSError, EOFError))
                if not transient or attempt == 4:
                    raise
                self.retries += 1
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def list(self, path: str) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            with self._get("/fs/list", {"path": path}) as response:
                return json.loads(response.read())
        return self._retried(call)

    def read(self, path: str, offset: int, length: int) -> tuple[bytes, int, bool]:
        def call() -> tuple[bytes, int, bool]:
            with self._get("/fs/read", {"path": path, "offset": offset, "length": length}) as response:
                data = response.read()
                return data, int(response.headers.get("X-Door-Size")), response.headers.get("X-Door-Eof") == "true"
        return self._retried(call)


def _door_error(error: BaseException) -> str | None:
    text = str(error)
    return text if type(error).__name__ == "DoorError" else None


def resolve_capture(door: Any, target: str) -> str:
    """A request id's walkthrough capture under the handoff root, or the capture root given."""
    if target.startswith("/"):
        return target.rstrip("/")
    found = []
    for bucket in door.list(HANDOFF_ROOT)["entries"]:
        if bucket.get("type") != "dir" or bucket["name"].startswith((".", "_")):
            continue
        site = f"{HANDOFF_ROOT}/{bucket['name']}/scenes/site-{target}/captures"
        try:
            listing = door.list(site)
        except Exception as error:  # noqa: BLE001 - a bucket without the site
            if _door_error(error) is None and "not_found" not in str(error):
                raise
            continue
        found += [f"{site}/{row['name']}" for row in listing["entries"]
                  if row.get("type") == "dir" and row["name"].startswith("walkthrough-")]
    if len(found) != 1:
        raise ValueError(f"job_replay_capture_not_unique:{target}:{len(found)}")
    return found[0]


def remote_tree(door: Any, root: str, *, workers: int = 8, only: Sequence[str] | None = None) -> dict[str, Any]:
    """Every file under ``root`` (``only``: under these sub-folders of it) as the door lists it: ``files`` (relative
    path -> size, mtime), the directories, symlinks, and what the door refused to name or could not list."""
    files: dict[str, dict[str, Any]] = {}
    dirs, links, refused, truncated = [""], {}, [], []

    def listing(relative: str) -> tuple[str, dict[str, Any]]:
        try:
            return relative, door.list(root + ("/" + relative if relative else ""))
        except Exception as error:  # noqa: BLE001 - one folder the door cannot list is named, not fatal
            if relative == "" and only is None:
                raise
            refused.append({"path": relative, "rule": f"list_failed:{str(error)[:120]}"})
            return relative, {"entries": []}

    level = [""] if only is None else list(only)
    dirs += level[1:] if only is None else level
    with ThreadPoolExecutor(max(1, workers)) as pool:
        while level:
            following = []
            for relative, value in pool.map(listing, level):
                if value.get("truncated"):
                    truncated.append(relative)
                for entry in value.get("entries") or []:
                    path = (relative + "/" if relative else "") + entry["name"]
                    if entry.get("refused"):
                        refused.append({"path": path, "rule": entry["refused"]})
                    elif entry.get("type") == "dir":
                        following.append(path)
                        dirs.append(path)
                    elif entry.get("type") == "file":
                        files[path] = {"size": int(entry["size"]), "mtime": int(entry["mtime"])}
                    elif entry.get("type") == "symlink":
                        links[path] = entry.get("target")
            level = following
    return {"root": root, "files": files, "dirs": sorted(dirs), "symlinks": links, "door_refused": refused,
            "truncated_dirs": truncated}


def _is_record(relative: str, size: int) -> bool:
    return size <= SMALL_BYTES or (relative.endswith(RECORD_SUFFIXES) and size <= MAXIMUM_RECORD_BYTES)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


class PullCache:
    """``<root>/<host path>`` for every pulled host file, and an index of its host size, mtime and digest."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.index: dict[str, list[Any]] = json.loads((self.root / INDEX_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.index = {}

    def local(self, host_path: str) -> Path:
        return self.root / host_path.lstrip("/")

    def save(self) -> None:
        temporary = self.root / (INDEX_FILE + ".partial")
        temporary.write_text(json.dumps(self.index, sort_keys=True), encoding="utf-8")
        temporary.replace(self.root / INDEX_FILE)

    def _listing_path(self, capture_root: str) -> Path:
        return self.root / ".job_replay_listings" / (hashlib.sha256(capture_root.encode()).hexdigest()[:32] + ".json")

    def save_listing(self, capture_root: str, trees: Mapping[str, Any]) -> None:
        path = self._listing_path(capture_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"capture_root": capture_root, "trees": trees}, sort_keys=True), encoding="utf-8")

    def load_listing(self, capture_root: str) -> dict[str, Any]:
        """The door's last listing of this capture (``--offline`` replays from it and the cached files)."""
        try:
            value = json.loads(self._listing_path(capture_root).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return value.get("trees") or {} if value.get("capture_root") == capture_root else {}

    def current(self, host_path: str, size: int, mtime: int) -> bool:
        """Pulled before, the host's size and mtime unchanged, and the local copy still its recorded bytes."""
        row, local = self.index.get(host_path), self.local(host_path)
        return (isinstance(row, list) and row[:2] == [size, mtime] and local.is_file() and not local.is_symlink()
                and local.stat().st_size == size and _sha256_file(local) == row[2])

    def pull(self, door: Any, host_path: str, size: int, mtime: int) -> dict[str, Any]:
        local = self.local(host_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + ".job-replay-partial")
        digest, offset, served = hashlib.sha256(), 0, size
        with open(partial, "wb") as stream:
            while True:
                data, served, eof = door.read(host_path, offset, DOOR_CHUNK_BYTES)
                stream.write(data)
                digest.update(data)
                offset += len(data)
                if eof or not data:
                    break
        os.utime(partial, (mtime, mtime))
        partial.replace(local)
        row = {"path": host_path, "bytes": offset, "sha256": "sha256:" + digest.hexdigest(),
               "changed_since_listing": served != size or offset != size}
        self.index[host_path] = [offset if row["changed_since_listing"] else size, mtime, row["sha256"]]
        return row


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def evidence_digest(cache: PullCache, host_paths: Iterable[str]) -> dict[str, Any]:
    """One digest over the pulled files' paths and bytes."""
    digest, count, size = hashlib.sha256(), 0, 0
    for path in sorted(set(host_paths)):
        local = cache.local(path)
        digest.update(path.encode() + b"\0")
        if local.is_file() and not local.is_symlink():
            digest.update(_sha256_file(local).encode() + b"\0")
            count, size = count + 1, size + local.stat().st_size
        else:
            digest.update(b"absent\0")
    return {"digest": "sha256:" + digest.hexdigest(), "files": count, "bytes": size}


class Pull:
    """What one replay pulled, kept and could not get, across its rounds."""

    def __init__(self, door: Any, cache: PullCache, capture_root: str, *, workers: int = 6) -> None:
        self.door, self.cache, self.capture_root, self.workers = door, cache, capture_root, workers
        self.scene_root = capture_root.rsplit("/captures/", 1)[0]
        self.trees: dict[str, dict[str, Any]] = {}
        self.wanted: dict[str, dict[str, Any]] = {}  # host path -> size, mtime
        self.pulled, self.reused, self.rounds, self.looked_up = [], [], [], {}
        self.listed_not_pulled: dict[str, dict[str, int]] = {}
        self.failed: dict[str, str] = {}

    def list_roots(self, roots: Sequence[str]) -> None:
        for root in roots:
            try:
                self.trees[root] = remote_tree(self.door, root)
            except Exception as error:  # noqa: BLE001 - a scene without a withdrawal folder, say
                if "not_found" not in str(error):
                    raise

    def empty_dirs(self) -> list[str]:
        """Host folders with no file anywhere beneath them: pulled whole by being made."""
        rows = []
        for root, tree in self.trees.items():
            filled = {str(Path(relative).parent) for relative in tree["files"]}
            filled |= {str(parent) for relative in tree["files"] for parent in Path(relative).parents}
            rows += [f"{root}/{relative}" for relative in tree["dirs"]
                     if relative and relative not in filled and relative not in tree["truncated_dirs"]]
        return rows

    def host_files(self) -> dict[str, dict[str, Any]]:
        return {f"{root}/{relative}": row for root, tree in self.trees.items()
                for relative, row in tree["files"].items()}

    def select_records(self) -> None:
        for path, row in self.host_files().items():
            if _is_record(path, row["size"]):
                self.wanted.setdefault(path, row)

    def select_cached(self, *, offline: bool = False) -> None:
        """Every file an earlier replay pulled whose host size and mtime have not changed: reused, not found again.
        ``offline``: every listed file the cache holds, as it holds it (one the host rewrote mid-pull is newer)."""
        for path, row in self.host_files().items():
            cached = self.cache.index.get(path)
            if isinstance(cached, list) and (offline or cached[:2] == [row["size"], row["mtime"]]):
                self.wanted.setdefault(path, {"size": cached[0], "mtime": cached[1]} if offline else row)

    def select_all(self) -> None:
        self.wanted.update(self.host_files())

    def select_globs(self, patterns: Sequence[str]) -> None:
        for path, row in self.host_files().items():
            relative = path[len(self.capture_root) + 1:] if path.startswith(self.capture_root + "/") else path
            if any(fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path, pattern) for pattern in patterns):
                self.wanted.setdefault(path, row)

    def expand(self, named: Iterable[str]) -> list[str]:
        """Host paths a stage named, as host files not yet wanted: a directory's files; a bulk file with its
        same-kind neighbours (a frame set is read whole); a path outside the listed roots looked up once."""
        files = self.host_files()
        added: list[str] = []
        for path in dict.fromkeys(named):
            path = path.rstrip("/")
            if not path.startswith("/") or path in self.wanted:
                continue
            if path in files:
                group = {path: files[path]}
                if not _is_record(path, files[path]["size"]):
                    parent, suffix = os.path.dirname(path), os.path.splitext(path)[1]
                    root = parent
                    while os.path.basename(root).isdigit():
                        root = os.path.dirname(root)  # ``outputs/000012/depth.bin``: the set is all of ``outputs``
                    group.update({other: row for other, row in files.items()
                                  if (other.startswith(root + "/") if root != parent
                                      else os.path.dirname(other) == parent and other.endswith(suffix))})
            elif any(other.startswith(path + "/") for other in files):
                group = {other: row for other, row in files.items() if other.startswith(path + "/")}
            elif any(path == root or path.startswith(root + "/") for root in self.trees):
                continue  # inside a listed root and absent on the host: not pullable
            else:
                group = self._outside(path)
            for other, row in group.items():
                if other not in self.wanted:
                    self.wanted[other] = row
                    added.append(other)
        return added

    def listed(self, folders: Iterable[str]) -> list[str]:
        """A folder a stage listed: its unpulled host files, when they are small together (a listing sees only what
        was pulled, so the upload in ``raw/`` would otherwise be silently absent); larger ones are only reported."""
        files, added = self.host_files(), []
        for folder in dict.fromkeys(folders):
            missing = {path: row for path, row in files.items()
                       if os.path.dirname(path) == folder and path not in self.wanted}
            size = sum(row["size"] for row in missing.values())
            if not missing:
                continue
            if size > LISTED_PULL_BYTES:
                self.listed_not_pulled[folder] = {"files": len(missing), "bytes": size}
                continue
            self.wanted.update(missing)
            added += list(missing)
        return added

    def referenced(self, paths: Iterable[str]) -> list[str]:
        """Records of a shared host folder that a pulled record of it names by absolute path (the scene retirement
        store's owner, delivery and membership records), so its chain is pulled in one round, not one a round."""
        files, added, todo = self.host_files(), [], [path for path in paths if path.endswith(".json")]
        shared = [root for root in self.trees if root != self.capture_root and root in OUTSIDE_SELECTORS]
        while todo:
            path = todo.pop()
            if not any(path.startswith(root + "/") for root in shared):
                continue
            try:
                value = json.loads(self.cache.local(path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for text in _strings(value):
                if text in files and text not in self.wanted and _is_record(text, files[text]["size"]):
                    self.wanted[text] = files[text]
                    added.append(text)
                    todo.append(text)
        return added

    def _outside(self, path: str) -> dict[str, dict[str, Any]]:
        if path in self.looked_up or len(self.looked_up) >= MAXIMUM_OUTSIDE_LOOKUPS:
            return {}
        try:
            value = self.door.list(path)
        except Exception as error:  # noqa: BLE001 - absent on the host too, or refused: recorded
            self.looked_up[path] = str(error)[:200]
            return {}
        self.looked_up[path] = value.get("type")
        if value.get("type") == "file":
            return {path: {"size": int(value["size"]), "mtime": int(value["mtime"])}}
        if value.get("type") == "dir":
            chosen = OUTSIDE_SELECTORS.get(path)
            tree = remote_tree(self.door, path, only=chosen.folders if chosen else None)
            self.trees[path] = tree
            return {f"{path}/{relative}": row for relative, row in tree["files"].items()
                    if _is_record(relative, row["size"])
                    and (chosen is None or chosen.keep(relative, self.capture_root))}
        return {}

    def fetch(self, reason: str, paths: Sequence[str] | None = None) -> dict[str, Any]:
        todo = [(path, self.wanted[path]) for path in (paths if paths is not None else list(self.wanted))]
        pulled, reused = [], []

        def one(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
            path, row = item
            if self.cache.current(path, row["size"], row["mtime"]):
                return path, None
            try:
                return path, self.cache.pull(self.door, path, row["size"], row["mtime"])
            except Exception as error:  # noqa: BLE001 - named and left out; the replay names it if it is read
                return path, {"failed": f"{type(error).__name__}: {str(error)[:200]}"}

        with ThreadPoolExecutor(max(1, self.workers)) as pool:
            for path, result in pool.map(one, todo):
                if result is None:
                    reused.append(path)
                elif "failed" in result:
                    self.failed[path] = result["failed"]
                    self.wanted.pop(path, None)
                else:
                    pulled.append(result)
        self.cache.save()
        self.pulled += pulled
        self.reused += reused
        summary = {"reason": reason, "pulled_files": len(pulled), "pulled_bytes": sum(r["bytes"] for r in pulled),
                   "reused_files": len(reused), "changed_since_listing": [r["path"] for r in pulled
                                                                          if r["changed_since_listing"]][:20],
                   "paths": [r["path"] for r in pulled][:50]}
        self.rounds.append(summary)
        return summary

    def report(self) -> dict[str, Any]:
        files = self.host_files()
        return {"capture_root": self.capture_root, "roots": sorted(self.trees),
                "host_files": len(files), "host_bytes": sum(row["size"] for row in files.values()),
                "wanted_files": len(self.wanted), "wanted_bytes": sum(row["size"] for row in self.wanted.values()),
                "pulled_files": len(self.pulled), "pulled_bytes": sum(row["bytes"] for row in self.pulled),
                "reused_files": len(set(self.reused) - {row["path"] for row in self.pulled}),
                "rounds": self.rounds, "door_calls": getattr(self.door, "calls", None),
                "door_refused": [{"root": root, **row} for root, tree in self.trees.items()
                                 for row in tree["door_refused"]][:200],
                "truncated_dirs": [f"{root}/{row}" for root, tree in self.trees.items()
                                   for row in tree["truncated_dirs"]],
                "outside_lookups": self.looked_up, "listed_not_pulled": self.listed_not_pulled,
                "failed": self.failed, "door_retries": getattr(self.door, "retries", None)}


# ------------------------------------------------------------------------------------- stepping past a known stop

@dataclass(frozen=True)
class Substitution:
    """Where a known stop's step has a retained output the host produced for it."""

    code: str
    stage: str
    module: str
    function: str
    callers: tuple[str, ...]
    retained: Callable[[Mapping[str, Any]], tuple[Any, Path] | None]
    note: str


def _retained_completion(options: Mapping[str, Any]) -> tuple[Any, Path] | None:
    """The completion's own retained report (``<root>/completion.json``) when it did not pass: nothing was generated
    to stand in for, so the controller goes on exactly as the host's run did (a passed one is adopted by the code)."""
    saved = Path(options["root"]) / "completion.json"
    if not saved.is_file():
        return None
    report = json.loads(saved.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") == "passed":
        return None
    return (None, report), saved


def _retained_static_masks(options: Mapping[str, Any]) -> tuple[Any, Path] | None:
    """The kept objects' masks the host completed for this binding (``<output_root>/<binding>/task_masks.json``)."""
    from .decision_evidence_contracts import canonical_digest
    from .website_track_mask_codec import load_track_json, plain_runs

    binding = (options.get("task_masks") or {}).get("binding")
    if not isinstance(binding, Mapping):
        return None
    path = Path(options["output_root"]).resolve() / canonical_digest(binding)[7:23] / "task_masks.json"
    if not path.is_file():
        return None
    value = load_track_json(path)
    return (value if options.get("compact_runs") else plain_runs(value)), path


SUBSTITUTIONS: dict[str, Substitution] = {row.code: row for row in (
    Substitution("atlas_completion_gap_survey_changed", "atlas_preview", "blueprint_pipeline.atlas_scene_completion",
                 "complete_scene", ("blueprint_pipeline.atlas_scene_automation",
                                    "blueprint_pipeline.atlas_preserved_completion",
                                    "blueprint_pipeline.atlas_scene_coverage"),
                 _retained_completion,
                 "the completion's retained report stands in for a gap survey this tree cannot recompute bit for bit"),
    Substitution("website_retained_static_masks_manifest_changed", "scene_handoff",
                 "blueprint_pipeline.website_task_masks", "complete_retained_static_task_masks", (),
                 _retained_static_masks,
                 "the kept objects' masks the host completed stand in for this tree's recomputation"),
)}


def _install_substitutions(stack: contextlib.ExitStack, past: Sequence[str], record: list[dict[str, Any]],
                           stage: Callable[[], str | None]) -> None:
    from .atlas_preview_replay import _swap

    for code in past:
        row = SUBSTITUTIONS.get(code)
        if row is None:
            continue
        for caller in row.callers:
            importlib.import_module(caller)  # loaded first, so every alias of the function is swapped
        module = importlib.import_module(row.module)
        original = getattr(module, row.function)

        def substituted(*args: Any, _row: Substitution = row, _original: Any = original, **options: Any) -> Any:
            try:
                return _original(*args, **options)
            except ValueError as error:
                if str(error) != _row.code:
                    raise
                found = _row.retained(options)
                if found is None:
                    record.append({"code": _row.code, "stage": stage(), "how": "no_retained_output"})
                    raise
                value, path = found
                record.append({"code": _row.code, "stage": stage(), "how": "retained_output", "path": str(path),
                               "digest": _sha256_file(path), "note": _row.note,
                               "replaced": f"{_row.module.removeprefix('blueprint_pipeline.')}.{_row.function}"})
                return value
        _swap(stack, module, row.function, substituted)


def _past_matches(stop: Mapping[str, Any] | None, past: Sequence[str]) -> bool:
    if not stop:
        return False
    return str(stop.get("code")) in past or f"{stop.get('kind')}:{stop.get('code')}" in past


# ------------------------------------------------------------------------------------------- one stage's process

def _install_socket_guard() -> list[dict[str, str]]:
    """``test_job_matrix``'s refusal of every socket connection and name lookup, for the whole process (its
    program allowlist is not: a stage may run ffmpeg; the harness refuses network tools and dead-ends proxies)."""
    from .test_job_matrix import install_network_guard

    return install_network_guard(subprocesses=False)


def _isolate_process() -> dict[str, Any]:
    from . import task_evaluation_scene_retirement_lifetime as retirement
    from .test_job_matrix import PROXY_NAMES, scrub_credentials

    removed = scrub_credentials()
    for name in PROXY_NAMES:
        os.environ[name] = "http://127.0.0.1:9"
    # The copy's owner stands for the host's service account in the scene retirement fence's ownership checks.
    retirement._SERVICE_IDENTITY = (os.getuid(), os.getgid())
    _stand_in_host_folders()
    return {"credentials_removed": removed, "attempts": _install_socket_guard()}


#: Every host property a stage met that a pulled copy cannot carry, and what stood in for it.
_STAND_INS: list[dict[str, Any]] = []


def _stand_in_host_folders() -> None:
    """A generation record of the scene retirement store pins the host folder it names by device, inode and mode,
    which no copy has. Each copied folder a record names stands for that host folder (``_identity`` answers the
    record's identity for it); everything else the fence checks (the record's digest, path and admissible state,
    the birth evidence and its digests) runs as on the host. Each folder stood in is listed."""
    import sys

    from . import task_evaluation_scene_retirement_generations  # noqa: F401 - loaded, so its alias is swapped
    from . import task_evaluation_scene_retirement_lifetime as retirement

    table: dict[tuple[int, int], tuple[Any, Any, Any]] = {}
    store = Path(SCENE_RETIREMENT_ROOT) / "generations"
    for record in sorted(store.glob("*.json")) if store.is_dir() else []:
        try:
            value = json.loads(record.read_text(encoding="utf-8"))
            info = os.stat(value["canonical_path"], follow_symlinks=False)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        if all(type(value.get(key)) is int for key in ("dev", "ino", "mode")):
            table[(info.st_dev, info.st_ino)] = (value["dev"], value["ino"], value["mode"])
            _STAND_INS.append({"stand_in": "scene_retirement_folder_identity", "path": value["canonical_path"],
                               "record": str(record), "state": value.get("state")})
    if not table:
        return
    original = retirement._identity

    def identity(info: os.stat_result) -> tuple[Any, Any, Any]:
        return table.get((info.st_dev, info.st_ino)) or original(info)
    for name, module in list(sys.modules.items()):
        if module is not None and name.startswith("blueprint_pipeline.") and vars(module).get("_identity") is original:
            setattr(module, "_identity", identity)


class _Clock:
    """Each stage's duration and peak resident memory inside one process (``/proc/self/clear_refs`` resets the
    peak; without it the peak is the process's own, said so)."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.current: str | None = None

    @staticmethod
    def _reset() -> bool:
        try:
            Path("/proc/self/clear_refs").write_text("5")
            return True
        except OSError:
            return False

    @staticmethod
    def _peak() -> int:
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            pass
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    def start(self, stage: str) -> None:
        self.stop()
        self.current = stage
        self.rows[stage] = {"started": time.monotonic(), "peak_scope": "stage" if self._reset() else "process",
                            "children_before": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}

    def stop(self) -> None:
        if self.current is None:
            return
        row = self.rows[self.current]
        children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        row.update(duration_seconds=round(time.monotonic() - row.pop("started"), 3), peak_rss_bytes=self._peak(),
                   peak_rss_children_bytes=children * 1024 if children > row.pop("children_before") else None)
        self.current = None


def _chain(error: BaseException, attribute: str) -> list[BaseException]:
    items = [error]
    while len(items) < 8 and getattr(items[-1], attribute) is not None:
        items.append(getattr(items[-1], attribute))
    return items


def _full_path(error: OSError) -> str:
    """The missing file's absolute path: a name opened relative to a directory descriptor (component by component,
    as the scene retirement fence opens its stores) is completed from the absolute path the raising frames held."""
    name = os.fsdecode(error.filename)
    if name.startswith("/"):
        return name
    trace, frames = error.__traceback__, []
    while trace is not None:
        frames.append(trace.tb_frame)
        trace = trace.tb_next
    for frame in reversed(frames):
        for value in frame.f_locals.values():
            if isinstance(value, (str, Path)) and str(value).startswith("/") and name in Path(value).parts:
                parts = Path(value).parts
                return str(Path(*parts[:parts.index(name) + 1]))
    return name


def _classify(error: BaseException, stage: str | None, host: Callable[[str], str] = str) -> dict[str, Any]:
    """A raised error as a stop: a missing file names its path, a typed refusal its code, anything else its type.

    The explicit causes (``raise ... from``) are read first: the pipeline wraps a stage's refusal that way. An error
    only handled on the way (``__context__``) never turns a refusal into a missing input."""
    import errno

    from .atlas_preview_replay import _raised_at
    from .core.common import StageError

    causes = _chain(error, "__cause__")
    for candidate in causes:
        if (isinstance(candidate, OSError) and candidate.errno == errno.ENOENT
                and isinstance(candidate.filename, (str, bytes, os.PathLike))):
            return {"kind": "missing_input", "stage": stage, "code": host(_full_path(candidate)),
                    "raised_at": _raised_at(candidate)}
    typed_kinds = (ValueError, StageError)
    typed = next((item for item in reversed(causes) if isinstance(item, typed_kinds)), None)
    if typed is None or (type(typed) is StageError and causes[-1] is typed):
        typed = next((item for item in _chain(causes[-1], "__context__")[1:] if isinstance(item, ValueError)),
                     typed)
    inner = typed if typed is not None else causes[-1]
    text = str(typed if typed is not None else error)
    if isinstance(typed, StageError):
        text = text.partition(": ")[2] or text
    head = text.split("; ", 1)[0].strip()
    while ": " in head and head.split(": ", 1)[0].replace("_", "").isalnum():
        head = head.split(": ", 1)[1]  # "website_scene_preparation: <code>" -> "<code>"
    code = head.partition(":")[0] if head.partition(":")[0].replace("_", "").isalnum() else head
    return {"kind": "refusal" if typed is not None else "error", "stage": stage, "code": code[:300],
            "exception": f"{type(error).__name__}: {error}"[:600], "raised_at": _raised_at(inner) or _raised_at(error)}


def _harness_class() -> type:
    """``atlas_preview_replay``'s harness, also answering the scene sponsorship from the job's retained record and
    watching every host path the stage probes (not only the scene's), so an unpulled one can be named."""
    from .atlas_preview_replay import _Harness, _set

    class JobHarness(_Harness):
        stub_modules = _Harness.stub_modules | {__name__}

        def __init__(self, *, watched: str, **options: Any) -> None:
            super().__init__(**options)
            self.watched = watched.rstrip("/")
            self.answers: list[dict[str, Any]] = []
            self.effects: list[dict[str, Any]] = []

        def webapp(self, *, capture_id: str, operation: str, payload: Mapping[str, Any],
                   source_kind: str = "website") -> dict[str, Any]:
            if operation == "scene-sponsorship":
                path = self.workspace / "pipeline" / "website_scene_sponsorship.json"
                if not path.is_file():
                    self.stop("missing_input", str(path))
                self.answers.append({"operation": operation, "answered_from": str(path),
                                     "create": payload.get("create", True)})
                return json.loads(path.read_text(encoding="utf-8"))
            if operation == "preparation-settlement":
                # It only releases what a bought call held beyond its cost. Recorded, never sent, and answered as no
                # answer: the code's own path when the WebApp does not answer (the reservation keeps its hold).
                settlement = payload.get("settlement") if isinstance(payload.get("settlement"), Mapping) else {}
                self.effects.append({"operation": operation, "stage": self.stage,
                                     "call_site": self.origin()["call_site"],
                                     **{key: settlement.get(key) for key in (
                                         "provider", "provider_charge_amount_usd", "completed_request_count",
                                         "allocation_binding_digest")}})
                raise ValueError(SETTLEMENT_NOT_SENT)
            if operation in ("task-context", "atlas-configuration"):
                self.answers.append({"operation": operation, "answered_from": "retained"})
            elif not (operation == "preparation-spend" and payload.get("create") is not False):
                # A write nothing local can answer: named with where it was made and what it carried (keys only).
                self.stop("external_effect", "webapp:" + operation, call_site=self.origin()["call_site"],
                          payload_keys=sorted(payload)[:20],
                          **{key: payload[key] for key in ("provider", "maximum_cost_usd", "settled_cost_usd",
                                                           "allocation_binding_digest") if key in payload})
            return super().webapp(capture_id=capture_id, operation=operation, payload=payload,
                                  source_kind=source_kind)

        def _guard(self, stack: contextlib.ExitStack) -> None:
            import pathlib

            super()._guard(stack)
            is_dir, listdir, scandir = pathlib.Path.is_dir, os.listdir, os.scandir
            self.listed: dict[str, None] = {}

            def probe(path: Any, *args: Any, **kwargs: Any) -> bool:
                found = is_dir(path, *args, **kwargs)
                if not found:
                    self.note(self.absent, path)
                return found

            def listed(original: Callable[..., Any]) -> Callable[..., Any]:
                def call(path: Any = ".", *args: Any, **kwargs: Any) -> Any:
                    if isinstance(path, (str, bytes, os.PathLike)):
                        self.note(self.listed, os.path.abspath(os.fsdecode(path)))
                    return original(path, *args, **kwargs)
                return call
            _set(stack, pathlib.Path, "is_dir", probe)
            _set(stack, os, "listdir", listed(listdir))
            _set(stack, os, "scandir", listed(scandir))

    return JobHarness


def _harness(capture_root: Path, *, watched: str, commit: str) -> Any:
    from .atlas_preview_replay import _snapshots

    configuration, context, _sources = _snapshots(capture_root, str, None, None)
    return _harness_class()(capture_root=capture_root, workspace=capture_root, prefix=None,
                            configuration=configuration, task_context=context, commit=commit, watched=watched)


def _row(stage: str, status: str, stop: Mapping[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
    return {"stage": stage, "status": status, "stop": dict(stop) if stop else None, "would_buy": [],
            "maximum_usd": None, "duration_seconds": None, "peak_rss_bytes": None, "reads": None,
            "substituted": [], **fields}


def _buys(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = ("boundary", "provider", "resource_class", "task", "maximum_cost_usd", "request_count",
            "already_reserved", "call_site")
    return [{key: row.get(key) for key in keys} for row in rows]


def _reads(paths: Sequence[str]) -> dict[str, Any]:
    return {"count": len(paths), "paths": list(paths[:400])}


def _atlas_group(capture_root: Path, *, past: Sequence[str], commit: str, steps: int | None,
                 watched: str) -> list[dict[str, Any]]:
    from .atlas_preview_replay import run_steps
    from .website_atlas_preview_store import JOB_FILE

    from . import atlas_preview_replay

    job = capture_root / "pipeline" / "atlas" / JOB_FILE
    if not job.is_file():
        return [_row("atlas_preview", "skipped", reason="no_atlas_preview_job")]
    clock, substituted, made = _Clock(), [], []
    base = _harness_class()

    class AtlasHarness(base):  # type: ignore[misc, valid-type]
        """The controller replay's own harness, with this tool's settlement, listing and probe records."""

        def __init__(self, **options: Any) -> None:
            super().__init__(watched=watched, **options)
            made.append(self)
    with contextlib.ExitStack() as stack:
        atlas_preview_replay._set(stack, atlas_preview_replay, "_Harness", AtlasHarness)
        _install_substitutions(stack, past, substituted, lambda: "atlas_preview")
        clock.start("atlas_preview")
        core = run_steps(job_path=job, capture_root=capture_root, workspace=capture_root, commit=commit, steps=steps)
        clock.stop()
    harness = made[0] if made else None
    final = core.get("final_job") or {}
    stop = core.get("first_stop")
    if stop is None:
        stop = ({"kind": "completed", "stage": final.get("stage"), "code": "ready"} if final.get("state") == "ready"
                else {"kind": "waiting", "stage": final.get("stage"), "code": f"{final.get('state')}:no_progress"})
    buys = _buys(core.get("would_buy") or [])
    used = any(row["how"] == "retained_output" for row in substituted)
    return [_row("atlas_preview", "assumed" if used else "ran", stop, would_buy=buys,
                 maximum_usd=_total(buys), substituted=substituted, reads=_reads(core.get("read_paths") or []),
                 atlas={key: core.get(key) for key in ("steps_run", "stages_reached", "final_job", "adopted",
                                                       "skipped_effects", "refused_spend", "retried_in_overlay")},
                 wanted=[*(core.get("absent_paths_probed") or []),
                         *([stop["code"]] if stop.get("kind") == "missing_input" else [])],
                 network_attempts=core.get("network_attempts") or [],
                 unsent_settlements=harness.effects if harness else [],
                 listed=list(harness.listed)[:2000] if harness else [],
                 **clock.rows["atlas_preview"])]


def _total(buys: Sequence[Mapping[str, Any]]) -> float | None:
    values = [row["maximum_cost_usd"] for row in buys if type(row.get("maximum_cost_usd")) in (int, float)]
    return round(sum(values), 6) if values else None


def _name_unpulled(stop: dict[str, Any] | None, unpulled: Sequence[str]) -> None:
    """A retained record that was not pulled reads as absent and can turn into a refusal (``atlas_preview_replay``):
    a refusal or error lists what the code looked for just before it and did not find."""
    if stop and stop.get("kind") in ("refusal", "error", "would_buy") and unpulled:
        stop["unpulled_before"] = list(unpulled[-10:])


def _stopped(harness: Any, error: BaseException, stage: str | None) -> dict[str, Any]:
    from .atlas_preview_replay import ReplayStop

    if isinstance(error, ReplayStop):
        return {**error.report(), "stage": stage}
    if harness.stops:  # a stop raised where something absorbed it and raised another error
        return {**harness.stops[0].report(), "stage": stage}
    return _classify(error, stage)


def _preparation_group(capture_root: Path, *, past: Sequence[str], commit: str, watched: str,
                       **_: Any) -> list[dict[str, Any]]:
    """``run_e2e`` exactly as the listener calls it once a recovery's claim and admission pass; the scene handoff
    is the same run from its call of ``prepare_website_scene_handoff``."""
    from . import run_e2e
    from . import website_scene_handoff as handoff
    from .atlas_preview_replay import ReplayStop, _swap

    clock, substituted, marks = _Clock(), [], {}
    harness = _harness(capture_root, watched=watched, commit=commit)
    stop: dict[str, Any] | None = None
    result: Any = None
    with contextlib.ExitStack() as stack:
        stack.enter_context(harness)
        harness.stage = "preparation"
        _install_substitutions(stack, past, substituted, lambda: harness.stage)
        original = handoff.prepare_website_scene_handoff

        def entered(**options: Any) -> Any:
            marks["reads"] = len(harness.reads)
            marks["buys"] = len(harness.would_buy)
            clock.start("scene_handoff")
            harness.stage = "scene_handoff"
            value = original(**options)
            marks["handoff"] = value
            return value
        _swap(stack, handoff, "prepare_website_scene_handoff", entered)
        clock.start("preparation")
        try:
            result = run_e2e.run_end_to_end(capture_root=str(capture_root), provider="openai",
                                            resume_completed_stages=True)
        except ReplayStop as error:
            stop = _stopped(harness, error, harness.stage)
        except Exception as error:  # noqa: BLE001 - the stage's own refusal or defect is the finding
            stop = _stopped(harness, error, harness.stage)
        clock.stop()
        reads = [harness.host(path) for path in harness.reads]
        unpulled = harness.unpulled()
    common = {"network_attempts": harness.network, "webapp_answers": harness.answers,
              "listed": list(harness.listed)[:2000]}

    def settlements(stage: str) -> list[dict[str, Any]]:
        return [row for row in harness.effects if row.get("stage") == stage]
    _name_unpulled(stop, unpulled)
    split, bought = marks.get("reads", len(reads)), marks.get("buys", len(harness.would_buy))
    reached = "scene_handoff" in clock.rows
    wanted = [*unpulled, *([stop["code"]] if stop and stop.get("kind") == "missing_input" else [])]
    prep_stop = (stop if not reached else {"kind": "completed", "stage": "preparation", "code": "reached_scene_handoff"})
    if not reached and stop is None:
        prep_stop = {"kind": "completed", "stage": "preparation",
                     "code": "ended_without_handoff:" + str((result or {}).get("pipeline_status"))}
    prep_buys, handoff_buys = _buys(harness.would_buy[:bought]), _buys(harness.would_buy[bought:])
    rows = [_row("preparation", "ran", prep_stop, would_buy=prep_buys, maximum_usd=_total(prep_buys),
                 reads=_reads(reads[:split]), substituted=[row for row in substituted if row["stage"] == "preparation"],
                 wanted=wanted if not reached else [], unsent_settlements=settlements("preparation"), **common,
                 **clock.rows["preparation"])]
    if reached:
        value = marks.get("handoff") if isinstance(marks.get("handoff"), Mapping) else {}
        handoff_stop = stop or {"kind": "completed", "stage": "scene_handoff",
                                "code": str(value.get("status") or "handoff_returned")}
        mine = [row for row in substituted if row["stage"] == "scene_handoff"]
        rows.append(_row("scene_handoff", "assumed" if any(r["how"] == "retained_output" for r in mine) else "ran",
                         handoff_stop, would_buy=handoff_buys, maximum_usd=_total(handoff_buys),
                         reads=_reads(reads[split:]), substituted=mine, wanted=wanted,
                         unsent_settlements=settlements("scene_handoff"), **common,
                         handoff=_handoff_summary(value), **clock.rows["scene_handoff"]))
    else:
        rows.append(_row("scene_handoff", "skipped", reason="not_reached_by_the_preparation"))
    return rows


def _handoff_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """What the handoff decided, and for a room package why (its record's flags and the pieces it leaves out)."""
    summary = {key: value[key] for key in ("status", "blockers", "digest", "job_disposition", "dispatch_blockers",
                                           "rule_withheld", "native_task_preparation_gap") if key in value}
    package = value.get("room_package")
    if isinstance(package, Mapping):
        summary["room_package"] = {key: item for key, item in package.items() if key != "path"}
        try:
            record = json.loads(Path(str(package.get("path"))).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
        summary["room_package"].update({key: record[key] for key in ("flags", "not_prepared", "task_kind")
                                        if key in record})
    return summary


def _company_group(capture_root: Path, *, commit: str, watched: str, **_: Any) -> list[dict[str, Any]]:
    """The company simulation pass for this capture: its source, its build, and the first effect after it."""
    import zipfile

    from . import website_company_simulation as simulation
    from .atlas_preview_replay import ReplayStop

    from .atlas_preview_replay import _raised_at, _swap

    clock, built, caught = _Clock(), {}, []
    harness = _harness(capture_root, watched=watched, commit=commit)
    package = capture_root.joinpath(*simulation.DIRECTORY) / simulation.PACKAGE_FILE
    before = _sha256_file(package) if package.is_file() else None
    stop: dict[str, Any] | None = None
    with harness, contextlib.ExitStack() as stack:
        harness.stage = "company_simulation"
        own_reason = simulation._own_reason

        def reason(error: BaseException | str) -> str:
            # The pass never raises: what it turned into a status is kept for the report.
            if isinstance(error, BaseException):
                caught.append({"exception": f"{type(error).__name__}: {error}"[:600], "raised_at": _raised_at(error)})
            return own_reason(error)
        _swap(stack, simulation, "_own_reason", reason)

        def archive(path: Path, *, identity: Mapping[str, str]) -> dict[str, Any]:
            with zipfile.ZipFile(path) as zipped:
                members = zipped.namelist()
            built.update(path=str(path), digest=_sha256_file(path), size_bytes=path.stat().st_size, members=members)
            harness.stop("external_effect", "artifact_store:" + simulation.ARTIFACT_KIND)

        def retract(uri: str) -> None:
            harness.stop("external_effect", "artifact_store:retract")

        def deliver(record: Mapping[str, Any], *, identity: Mapping[str, str]) -> None:
            harness.stop("external_effect", "webapp:company-simulation")

        clock.start("company_simulation")
        try:
            outcome = simulation.publish_company_simulation(capture_root, archive=archive, retract=retract,
                                                            deliver=deliver)
            status = str(outcome.get("status"))
            stop = {"kind": "completed" if status in ("published", "current") else
                    "waiting" if status in ("no_source", "busy") else "error" if status == "failed" else "refusal",
                    "stage": "company_simulation", "code": ":".join(
                        str(part) for part in (status, outcome.get("reason")) if part),
                    **(caught[-1] if status == "failed" and caught else {})}
        except ReplayStop as error:
            stop = _stopped(harness, error, "company_simulation")
        except Exception as error:  # noqa: BLE001
            stop = _stopped(harness, error, "company_simulation")
        clock.stop()
        reads = [harness.host(path) for path in harness.reads]
        unpulled = harness.unpulled()
        listed = list(harness.listed)[:2000]
    _name_unpulled(stop, unpulled)
    if not built and package.is_file() and _sha256_file(package) != before:
        built.update(path=str(package), digest=_sha256_file(package), size_bytes=package.stat().st_size)
    buys = _buys(harness.would_buy)
    return [_row("company_simulation", "ran", stop, would_buy=buys, maximum_usd=_total(buys), reads=_reads(reads),
                 package=built or None, network_attempts=harness.network, webapp_answers=harness.answers,
                 unsent_settlements=harness.effects, listed=listed,
                 wanted=[*unpulled, *([stop["code"]] if stop and stop.get("kind") == "missing_input" else [])],
                 **clock.rows["company_simulation"])]


_GROUP_RUNNERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "atlas_preview": _atlas_group, "preparation": _preparation_group, "company_simulation": _company_group}


def _run_group(group: str, capture_root: Path, *, run_dir: Path, past: Sequence[str], commit: str,
               steps: int | None, watched: str, inline: bool) -> list[dict[str, Any]]:
    """One stage group, in its own process (``inline``: in this one, for tests that stand stages in)."""
    options = dict(past=past, commit=commit, steps=steps, watched=watched)
    if inline:
        try:
            return _GROUP_RUNNERS[group](capture_root, **options)
        except Exception as error:  # noqa: BLE001 - a defect of the replay itself, said so
            return [_row(group, "ran", _classify(error, group))]
    output = run_dir / f"worker-{group}.json"
    argv = [sys.executable, "-m", "blueprint_pipeline.job_replay", "--worker", group, str(capture_root),
            "--run-dir", str(run_dir), "--source-commit", commit, "--watched", watched,
            *[item for code in past for item in ("--past", code)], *(["--steps", str(steps)] if steps else [])]
    log = run_dir / f"worker-{group}.log"
    started = time.monotonic()
    with open(log, "wb") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, cwd=str(CODE_ROOT),
                                   env={**os.environ, "PYTHONPATH": str(CODE_ROOT / "src")})
        _pid, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
    if output.is_file():
        value = json.loads(output.read_text(encoding="utf-8"))
        rows = value["rows"]
        for row in rows:
            row.setdefault("network_attempts", [])
            row["network_attempts"] = [*row["network_attempts"], *value.get("process_network_attempts", [])]
            row["credentials_removed"] = value.get("credentials_removed")
        if rows:
            rows[0]["stand_ins"] = value.get("stand_ins") or []
        return rows
    killed = -process.returncode if process.returncode < 0 else None
    code = (f"worker_killed:{signal.Signals(killed).name}" if killed else f"worker_exit:{process.returncode}")
    tail = log.read_bytes()[-1500:].decode("utf-8", "replace")
    stages = next(row for row in GROUPS if row[0] == group)
    return [_row(stages[0], "ran", {"kind": "error", "stage": stages[0], "code": code, "log_tail": tail},
                 duration_seconds=round(time.monotonic() - started, 3), peak_rss_bytes=usage.ru_maxrss * 1024),
            *[_row(stage, "skipped", reason=f"{stages[0]}_worker_ended:{code}") for stage in stages[1:]]]


def run_stages(capture_root: Path, *, run_dir: Path, past: Sequence[str] = (), commit: str = "",
               steps: int | None = None, watched: str | None = None, inline: bool = False) -> dict[str, Any]:
    """Every remaining stage of the job's website path, in order, on the capture root as it stands (a scratch copy at
    the capture's own path: the caller makes it). A stage that stops ends the chain unless ``past`` names its stop."""
    capture_root, run_dir = Path(capture_root), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    watched = watched or str(capture_root.parents[1])
    rows: list[dict[str, Any]] = []
    assumed: list[dict[str, Any]] = []
    halted: str | None = None
    for group in GROUPS:
        if halted is not None:
            rows += [_row(stage, "skipped", reason=halted) for stage in group]
            continue
        produced = _run_group(group[0], capture_root, run_dir=run_dir, past=past, commit=commit, steps=steps,
                              watched=watched, inline=inline)
        for row in produced:
            rows.append(row)
            assumed += [{"stage": row["stage"], **item} for item in row["substituted"]
                        if item.get("how") == "retained_output"]
            stop = row.get("stop") or {}
            if row["status"] == "skipped" or halted is not None:
                continue
            if stop.get("kind") == "completed":
                continue
            if row["stage"] == "atlas_preview" and stop.get("kind") == "waiting" and _past_matches(stop, past) is False:
                halted = f"waits_on_atlas_preview:{stop.get('code')}"
            if _past_matches(stop, past):
                row["status"] = "assumed"
                assumed.append({"stage": row["stage"], "code": stop.get("code"), "kind": stop.get("kind"),
                                "how": "assumed_passed",
                                "note": "this stage's stop was stepped past; the next stage reads the files as the "
                                        "host left them"})
                if row["stage"] == "preparation":
                    # Nothing stands in for the clean plate result the handoff takes: it is not re-run.
                    following = next(item for item in produced if item["stage"] == "scene_handoff")
                    following.update(status="skipped", reason="its input, the preparation's clean plate result, "
                                                              "was not made; nothing stands in for it")
                continue
            halted = halted or f"after_{row['stage']}:{stop.get('kind')}:{stop.get('code')}"
    wanted = sorted({path for row in rows for path in row.pop("wanted", []) or [] if isinstance(path, str)})
    listed = sorted({path for row in rows for path in row.pop("listed", []) or [] if isinstance(path, str)})
    attempts = [attempt for row in rows for attempt in row.get("network_attempts") or []]
    return {"stages": rows, "assumed": assumed, "wanted": wanted, "listed": listed, "network_attempts": attempts,
            "clean": not assumed and not any(row["status"] == "assumed" for row in rows),
            "breaches": ["network_attempted"] if attempts else []}


# ---------------------------------------------------------------------------------------- the private namespace

def _namespaces() -> str:
    return " ".join(os.readlink(f"/proc/self/ns/{kind}") for kind in ("mnt", "net"))


def namespace_prefix(probe_dir: Path) -> list[str] | None:
    """``unshare`` with a private mount and network namespace, if this host permits it (``atlas_preview_replay``)."""
    from .atlas_preview_replay import _namespace_prefix

    return _namespace_prefix(probe_dir)


def _mount_roots(paths: Iterable[str], *, avoid: Path) -> list[str]:
    """The host folders the work copy stands over: the first three components of each host path
    (``/var/lib/blueprint``), deeper where that folder would also cover ``avoid`` (the replay's own scratch)."""
    avoid = Path(os.path.abspath(avoid))
    roots: set[str] = set()
    for path in paths:
        parts = Path(path).parts
        depth = 4
        while depth < len(parts) and (avoid == Path(*parts[:depth]) or Path(*parts[:depth]) in avoid.parents):
            depth += 1
        roots.add(str(Path(*parts[:depth])))
    return sorted(root for root in roots if not any(other != root and root.startswith(other + "/")
                                                    for other in roots))


def _mask_credentials() -> list[str]:
    masked = []
    home = Path(os.path.expanduser("~"))
    hidden = sorted(str(path) for path in home.iterdir() if path.name.startswith(".")
                    and path.name not in HOME_KEPT) if home.is_dir() else []
    for item in (*hidden, *CREDENTIAL_PATHS):
        path = Path(os.path.expanduser(item))
        if path.is_symlink() or not path.exists():
            continue
        if path.is_dir():
            subprocess.run(["mount", "-t", "tmpfs", "-o", "ro,size=1k", "tmpfs", str(path)], check=True)
        else:
            subprocess.run(["mount", "--bind", "/dev/null", str(path)], check=True)
        masked.append(str(path))
    return masked


def _inner_main(args: argparse.Namespace) -> int:
    """Inside the namespace: stand the work copy over the host paths, hide the evidence, replay, write the report."""
    if not args.outer_namespace or set(_namespaces().split()) & set(args.outer_namespace.split()):
        raise SystemExit("job_replay --inner runs only inside the private namespace the replay creates")
    run_dir = Path(args.run_dir)
    plan = json.loads((run_dir / "inner_plan.json").read_text(encoding="utf-8"))
    view = _stand_work(plan, run_dir)
    masked = _mask_credentials()
    result = run_stages(Path(plan["capture_root"]), run_dir=run_dir, past=plan["past"], commit=plan["commit"],
                        steps=plan.get("steps"), watched=plan["watched"])
    result["masked_paths"] = masked
    result["work_view"] = view
    result["network_probe"] = _network_probe()
    (run_dir / "inner_report.json").write_text(json.dumps(result, sort_keys=True, default=str), encoding="utf-8")
    return 0


def _network_probe() -> dict[str, Any]:
    """What the namespace's network is: its own interfaces (only ``lo`` when private)."""
    import socket

    try:
        names = sorted(name for _index, name in socket.if_nameindex())
    except OSError:
        names = []
    return {"interfaces": names}


def _link_work(cache: PullCache, host_paths: Iterable[str], lower: Path) -> None:
    """The run's read-only layer: exactly the wanted cached files, hard-linked (a copy where a link cannot be made).
    It is only ever an overlay's lower layer, which overlayfs never writes, and is hidden once mounted."""
    for path in sorted(set(host_paths)):
        source = cache.local(path)
        if not source.is_file():
            continue
        target = lower / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def _stand_work(plan: Mapping[str, Any], run_dir: Path) -> str:
    """Stand the run's view over each host root: an overlay of the linked layer and a fresh upper layer (writes go
    there; nothing is copied), else a copy of the layer. Then the layer and the cache are hidden."""
    lower = Path(plan["lower"])
    mode = "overlay"
    for root in plan["mount_roots"]:
        relative = root.lstrip("/")
        upper, work = run_dir / "upper" / relative, run_dir / "overlay_work" / relative
        upper.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)
        options = f"lowerdir={lower / relative},upperdir={upper},workdir={work}"
        mounted = subprocess.run(["mount", "-t", "overlay", "overlay", "-o", options, root], capture_output=True,
                                 check=False)
        if mounted.returncode != 0:  # no overlayfs here: the stages get a copy of the layer instead
            mode = "copy"
            copy = run_dir / "copy" / relative
            shutil.copytree(lower / relative, copy, symlinks=True, dirs_exist_ok=True)
            subprocess.run(["mount", "--bind", str(copy), root], check=True)
    # The replay never sees the pulled evidence or its links, so nothing it runs can change them.
    for hidden in (str(lower), plan["cache"]):
        subprocess.run(["mount", "-t", "tmpfs", "-o", "ro,size=1k", "tmpfs", hidden], check=True)
    return mode


def _replay_once(pull: Pull, *, scratch: Path, past: Sequence[str], commit: str, steps: int | None,
                 namespace: list[str], keep: bool, index: int) -> dict[str, Any]:
    run_dir = Path(tempfile.mkdtemp(prefix=f"run{index}-", dir=scratch / "runs"))
    lower = run_dir / "lower"
    _link_work(pull.cache, pull.wanted, lower)
    for path in pull.empty_dirs():
        (lower / path.lstrip("/")).mkdir(parents=True, exist_ok=True)
    modes = {}
    for path, mode in HOST_MODES.items():
        local = lower / path.lstrip("/")
        if local.is_dir():  # a folder of the run's own layer, never a cached file
            local.chmod(mode)
            modes[path] = oct(mode)
    roots = _mount_roots([pull.capture_root, *pull.wanted], avoid=scratch)
    created = []
    for root in roots:
        (lower / root.lstrip("/")).mkdir(parents=True, exist_ok=True)
        if not Path(root).exists():
            Path(root).mkdir(parents=True)  # an empty mount point, only where the host folder is not here at all
            created.append(root)
    plan = {"lower": str(lower), "cache": str(pull.cache.root), "mount_roots": roots, "capture_root": pull.capture_root,
            "past": list(past), "commit": commit, "steps": steps,
            "watched": next(root for root in roots if pull.capture_root.startswith(root + "/"))}
    (run_dir / "inner_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    environment = {name: value for name, value in os.environ.items() if not _credential_name(name)}
    environment.update(PYTHONPATH=str(CODE_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([*namespace, sys.executable, "-m", "blueprint_pipeline.job_replay", "--inner",
                                "--run-dir", str(run_dir), "--outer-namespace", _namespaces()],
                               capture_output=True, text=True, check=False, env=environment, cwd=str(CODE_ROOT))
    inner = run_dir / "inner_report.json"
    result = (json.loads(inner.read_text(encoding="utf-8")) if inner.is_file() else
              {"stages": [_row("atlas_preview", "ran", {"kind": "error", "stage": None, "code": "replay_namespace_failed",
                                                        "log_tail": completed.stderr[-2000:]})],
               "assumed": [], "wanted": [], "network_attempts": []})
    result.update(run_dir=str(run_dir), mount_roots=roots, created_mount_points=created,
                  stand_ins={"host_folder_modes": modes,
                             "used": [item for row in result.get("stages", []) for item in row.get("stand_ins") or []],
                             "service_account": "the work copy's owner stands for the host's service account in "
                                                "the scene retirement fence (a door pull carries no owner)"})
    if not keep:
        for name in ("lower", "upper", "overlay_work", "copy"):
            shutil.rmtree(run_dir / name, ignore_errors=True)
    return result


def _credential_name(name: str) -> bool:
    from .test_job_matrix import CREDENTIAL_NAME

    return bool(CREDENTIAL_NAME.search(name))


def _source_state() -> tuple[str, bool]:
    from .atlas_preview_replay import _source_state as state

    return state(CODE_ROOT)


def _finish(report: dict[str, Any], result: Mapping[str, Any], *, runs: list[dict[str, Any]], pull: Pull,
            door: Any, namespace: list[str], converged: bool, pending: Sequence[str], started: float,
            scratch: Path, report_to: str | Path | None) -> dict[str, Any]:
    """The report as it stands after a round, written beside that round's run (and to ``report_to``), so a run that
    is stopped part way still leaves its last complete round."""
    attempts = result.get("network_attempts") or []
    breaches = [*(result.get("breaches") or []),
                *([] if (result.get("evidence") or {}).get("unchanged") else ["evidence_changed"])]
    report.update(stages=result.get("stages", []), assumed=result.get("assumed", []),
                  clean=bool(result.get("clean")), passed=not breaches, breaches=breaches,
                  evidence=result.get("evidence"), rounds=runs, pull=pull.report() if door is not None else None,
                  boundaries={"network_attempts": attempts, "masked_paths": result.get("masked_paths"),
                              "network_probe": result.get("network_probe"), "namespace": namespace,
                              "mount_roots": result.get("mount_roots"), "work_view": result.get("work_view"),
                              "created_mount_points": result.get("created_mount_points"),
                              "stand_ins": result.get("stand_ins"),
                              "credentials_removed": sorted({name for row in result.get("stages", [])
                                                             for name in row.get("credentials_removed") or []}),
                              "new_paid_calls": 0},
                  unpulled_named=list(result.get("wanted", []))[:100], converged=converged,
                  named_not_pulled={"files": len(pending), "paths": list(pending)[:50]},
                  elapsed_seconds=round(time.time() - started, 3))
    path = Path(result.get("run_dir") or scratch) / "job_replay_report.v1.json"
    report["report_path"] = str(path)
    text = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf-8")
    if report_to:
        Path(report_to).write_text(text, encoding="utf-8")
    return report


def replay_job(target: str, *, door: Any = None, scratch: str | Path | None = None, past: Sequence[str] = (),
               rounds: int = DEFAULT_ROUNDS, pull_all: bool = False, pull_globs: Sequence[str] = (),
               steps: int | None = None, keep: bool = False, offline: bool = False,
               report_to: str | Path | None = None) -> dict[str, Any]:
    """Pull what the job's replay needs, replay its remaining stages in a private namespace, and report."""
    started = time.time()
    scratch = Path(scratch or os.environ.get("BLUEPRINT_JOB_REPLAY_SCRATCH")
                   or Path(tempfile.gettempdir()) / "blueprint-job-replay")
    (scratch / "runs").mkdir(parents=True, exist_ok=True)
    commit, clean_tree = _source_state()
    report: dict[str, Any] = {"schema_version": SCHEMA, "target": target, "source_commit": commit,
                              "source_tree_clean": clean_tree, "past": list(past),
                              "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))}
    probe = Path(tempfile.mkdtemp(prefix="probe-", dir=scratch / "runs"))
    namespace = namespace_prefix(probe)
    probe.rmdir()
    if namespace is None:
        return {**report, "replay_unsupported": "private_namespace_unavailable", "stages": [], "passed": False,
                "clean": False, "elapsed_seconds": round(time.time() - started, 3)}
    door = door or (None if offline else DoorReader())
    cache = PullCache(scratch / "cache")
    with _locked(scratch / "cache" / ".lock"):
        capture_root = resolve_capture(door, target) if door is not None else target.rstrip("/")
        pull = Pull(door, cache, capture_root)
        if door is not None:
            pull.list_roots([capture_root, pull.scene_root + "/website_withdrawal"])
            pull.select_all() if pull_all else pull.select_records()
            pull.select_cached()
            pull.select_globs(pull_globs)
            pull.fetch("records" if not pull_all else "all")
        else:
            pull.trees = cache.load_listing(capture_root)
            if not pull.trees:
                return {**report, "replay_unsupported": "offline_without_an_earlier_pull", "stages": [],
                        "passed": False, "clean": False, "elapsed_seconds": round(time.time() - started, 3)}
            pull.select_cached(offline=True)
    report["capture_root"] = capture_root
    runs, result, converged, pending = [], {}, door is None, []
    for index in range(rounds + 1):
        before = evidence_digest(cache, pull.wanted)
        result = _replay_once(pull, scratch=scratch, past=past, commit=commit, steps=steps, namespace=namespace,
                              keep=keep, index=index)
        after = evidence_digest(cache, pull.wanted)
        runs.append({"round": index, "run_dir": result["run_dir"], "evidence_unchanged": before == after,
                     "stops": [f"{row['stage']}:{(row.get('stop') or {}).get('kind')}" for row in result["stages"]]})
        result["evidence"] = {"digest_before": before["digest"], "digest_after": after["digest"],
                              "files": after["files"], "bytes": after["bytes"], "unchanged": before == after}
        _finish(report, result, runs=runs, pull=pull, door=door, namespace=namespace, converged=False,
                pending=[], started=started, scratch=scratch, report_to=report_to)
        if door is None or not result["evidence"]["unchanged"]:
            break
        with _locked(scratch / "cache" / ".lock"):
            kept = dict(pull.wanted)
            added = [*pull.expand(result.get("wanted") or []), *pull.listed(result.get("listed") or [])]
            if not added:
                converged = True
                break
            if index == rounds:
                pull.wanted = kept  # out of rounds: what the last replay named is reported, not pulled
                pending = added
                break
            pull.fetch(f"named_by_round_{index}", added)
            followed = pull.referenced(added)
            if followed:
                pull.fetch(f"referenced_by_round_{index}", followed)
    if door is not None:
        cache.save_listing(capture_root, pull.trees)
    return _finish(report, result, runs=runs, pull=pull, door=door, namespace=namespace, converged=converged,
                   pending=pending, started=started, scratch=scratch, report_to=report_to)


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


# ---------------------------------------------------------------------------------------------------- the table

def _size(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return "-"


def table(report: Mapping[str, Any]) -> str:
    header = ("stage", "ran/skipped/assumed", "stop kind:code", "would buy (max USD)", "duration", "peak RSS")
    rows = [header]
    for row in report.get("stages") or []:
        stop = row.get("stop") or {}
        stopped = f"{stop.get('kind')}:{stop.get('code')}" if stop else f"-: {row.get('reason') or ''}"
        buys = ", ".join(f"{item.get('provider') or item.get('resource_class')} {item.get('task') or ''}".strip()
                         + f" <= ${item.get('maximum_cost_usd')}" for item in row.get("would_buy") or []) or "-"
        duration = row.get("duration_seconds")
        rows.append((row["stage"], row["status"], stopped[:110], buys[:70],
                     f"{duration:.1f}s" if isinstance(duration, (int, float)) else "-",
                     _size(row.get("peak_rss_bytes"))))
    widths = [max(len(str(item[index])) for item in rows) for index in range(len(header))]
    lines = ["  ".join(str(cell).ljust(width) for cell, width in zip(item, widths, strict=True)).rstrip()
             for item in rows]
    for item in report.get("assumed") or []:
        lines.append(f"  ASSUMED {item.get('stage')}: {item.get('code')} ({item.get('how')}"
                     + (f", {item['path']}" if item.get("path") else "") + ")")
    evidence = report.get("evidence") or {}
    pending = (report.get("named_not_pulled") or {}).get("files")
    if report.get("converged") is False and pending:
        lines.append(f"NOT CONVERGED: the last replay named {pending} host files not pulled yet, so its stops may be "
                     "the pull's, not the job's; run again (pulls are cached) or raise --rounds")
    lines.append(("NOT A CLEAN RUN: assumed steps above. " if not report.get("clean") else "clean run. ")
                 + f"evidence {evidence.get('files')} files unchanged={evidence.get('unchanged')}; "
                 + f"network attempts {len((report.get('boundaries') or {}).get('network_attempts') or [])}; "
                 + "$0 spent")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------ commands

def _worker_main(args: argparse.Namespace) -> int:
    isolation = _isolate_process()
    run_dir = Path(args.run_dir)
    rows = _GROUP_RUNNERS[args.worker](Path(args.target), past=args.past, commit=args.source_commit,
                                       steps=args.steps, watched=args.watched)
    value = {"rows": rows, "process_network_attempts": list(isolation["attempts"]),
             "credentials_removed": isolation["credentials_removed"], "stand_ins": _STAND_INS}
    (run_dir / f"worker-{args.worker}.json").write_text(json.dumps(value, sort_keys=True, default=str),
                                                        encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", help="a website request id, or the job's host capture root")
    parser.add_argument("--past", action="append", default=[], help="step past this stop code, flagged (repeatable)")
    parser.add_argument("--scratch", help="pull cache and runs (default: $BLUEPRINT_JOB_REPLAY_SCRATCH or the "
                                          "temp directory's blueprint-job-replay)")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help="replays after the first, each after pulling what the last one named")
    parser.add_argument("--pull", action="append", default=[], help="also pull host files matching this glob")
    parser.add_argument("--pull-all", action="store_true", help="pull every file of the capture (slow: GBs)")
    parser.add_argument("--offline", action="store_true", help="use only what is already pulled; no door")
    parser.add_argument("--steps", type=int, help="Atlas controller steps (default: until it stops)")
    parser.add_argument("--report", help="also write the JSON report here")
    parser.add_argument("--keep", action="store_true", help="keep the work copies")
    parser.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", choices=sorted(_GROUP_RUNNERS), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", help=argparse.SUPPRESS)
    parser.add_argument("--outer-namespace", default="", help=argparse.SUPPRESS)
    parser.add_argument("--source-commit", default="", help=argparse.SUPPRESS)
    parser.add_argument("--watched", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.inner:
        return _inner_main(args)
    if args.worker:
        return _worker_main(args)
    if not args.target:
        parser.error("a request id or capture root is required")
    if args.rounds < 0:
        parser.error("--rounds must be 0 or more")
    report = replay_job(args.target, scratch=args.scratch, past=args.past, rounds=args.rounds,
                        pull_all=args.pull_all, pull_globs=args.pull, steps=args.steps, keep=args.keep,
                        offline=args.offline, report_to=args.report)
    if args.report and report.get("replay_unsupported"):
        Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n",
                                     encoding="utf-8")
    print(table(report))
    print(f"report: {report.get('report_path')}")
    if report.get("replay_unsupported"):
        print(f"not replayed: {report['replay_unsupported']}", file=sys.stderr)
        return _EXIT_UNSUPPORTED
    return 0 if report.get("passed") else _EXIT_BREACH


__all__ = ["DoorReader", "Pull", "PullCache", "SCHEMA", "STAGES", "SUBSTITUTIONS", "main", "remote_tree",
           "replay_job", "resolve_capture", "run_stages", "table"]

if __name__ == "__main__":
    raise SystemExit(main())
