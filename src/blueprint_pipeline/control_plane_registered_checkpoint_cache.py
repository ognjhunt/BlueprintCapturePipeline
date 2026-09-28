"""ADP-009D/day28: authenticated needed checkpoint cache, always retained.

Root issuance is separate from current read/fill permission and paid execution.
New enrolled paths never use the legacy pathname downloader or SDK uploader.
"""
from __future__ import annotations

import fcntl
import grp
import hashlib
import math
import os
import pwd
import re
import secrets
import stat
import threading
import time
from pathlib import Path, PurePosixPath
from contextlib import contextmanager

from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch as scratch
from . import control_plane_lane_scratch_decisions as retained
from . import control_plane_lane_experiment_retirement as installed
from .control_plane_lane_experiment_publication import _BirthFiles
from .control_plane_lane_owner_target_publication import _publish_owned_metadata
from .control_plane_reference_budget import ReferenceCollectionBudget
from .decision_evidence_contracts import canonical_digest
from .control_plane_disk_budget import reserve_control_plane_disk
from .control_plane_disk_reservation_heartbeat import keep_reservation_live

SCHEMA = "control_plane_needed_cache_creation_intent.v1"
MARKER = ".needed-cache-birth.v1.json"
_STORE_LOCK = ".cache-store.lock"
_DEFAULT_CONFIG = "/etc/blueprint-operator-door/door.json"
_PUBLIC_REGISTRATION = Path("/var/lib/blueprint-operator-door/needed-checkpoint-cache-registration")
_PUBLIC_INVENTORY = Path("/opt/blueprint/control-plane-config-tools/operator-door-source/configs/g1_humanoidarena_checkpoint_inventory.v1.json")
_REGISTERED_ROOTS = (Path("/mnt/blueprint-work/lanes"),
                     Path("/var/lib/blueprint/task-evaluation-inputs/lanes"))
_QUANTUM = 1024 * 1024
_PART = 8 * _QUANTUM
_MAX_BYTES = 32 * 1024**3
_INTENT_FIELDS = frozenset({"intent_id", "schema_version", "issuer_kind", "issuer_uid", "principal", "owner",
    "root", "lane", "name", "reference_kind", "reference_value", "reason", "class_intent", "cleanup",
    "issued_at_epoch", "expires_at_epoch", "lease_ttl_seconds", "size_budget_bytes", "inventory_raw_sha256",
    "inventory_raw_size_bytes", "candidate_inventory_digests", "policy_sha256", "policy_size_bytes",
    "writer_scope", "generation", "intent_digest"})


class NeededCheckpointCacheError(ValueError):
    """Fixed typed refusal; no policy, URL, path or credentials in errors."""


def _require(condition, code="needed_cache_authority_invalid"):
    if not condition:
        raise NeededCheckpointCacheError(code)


def _blueprint_identity():
    try:
        return pwd.getpwnam("blueprint").pw_uid, grp.getgrnam("blueprint").gr_gid
    except KeyError:
        raise NeededCheckpointCacheError("needed_cache_account_missing") from None


def is_registered_checkpoint_path(path):
    """Lexical namespace gate before config, marker, stat or credential access."""
    value = Path(path)
    return any(value == root / "g1-checkpoint" or root / "g1-checkpoint" in value.parents
               for root in _REGISTERED_ROOTS)


def _inventory_rows(inventory):
    from .native_g1_development_pair import PAIR_ORDER
    _require(type(inventory) is dict and inventory.get("schema_version") ==
             "g1_humanoidarena_checkpoint_inventory.v1" and type(inventory.get("candidates")) is list,
             "needed_cache_inventory_invalid")
    candidates = inventory["candidates"]
    _require(len(candidates) == 4 and tuple(c.get("candidate_id") for c in candidates) == PAIR_ORDER,
             "needed_cache_inventory_invalid")
    result, seen = [], set()
    for candidate in candidates:
        folder = PurePosixPath(candidate.get("subdirectory", ""))
        files = candidate.get("files")
        _require(folder.parts and not folder.is_absolute() and ".." not in folder.parts
                 and type(files) is list and len(files) == 6, "needed_cache_inventory_invalid")
        _require(candidate.get("inventory_digest") == "sha256:" + hashlib.sha256(json_bytes(files)).hexdigest(),
                 "needed_cache_inventory_invalid")
        for row in files:
            _require(type(row) is dict and set(row) == {"path", "sha256", "size_bytes"},
                     "needed_cache_inventory_invalid")
            path = PurePosixPath(row["path"])
            relative = (folder / path).as_posix()
            _require(path.parts and not path.is_absolute() and ".." not in path.parts and len((folder / path).parts) <= 64
                     and len(relative.encode()) <= 4096 and relative not in seen
                     and type(row["size_bytes"]) is int and 0 < row["size_bytes"] <= _MAX_BYTES
                     and type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
                     "needed_cache_inventory_invalid")
            seen.add(relative)
            result.append(dict(relative_path=relative, sha256="sha256:" + row["sha256"],
                               size_bytes=row["size_bytes"], candidate_id=candidate["candidate_id"]))
    _require(sum(r["size_bytes"] for r in result) <= _MAX_BYTES, "needed_cache_inventory_invalid")
    return result


def json_bytes(value):
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def derive_checkpoint_flow_resources(inventory, flow, *, present_paths=()):
    """Conserve actual all-four passes; quotas do not imply transfer throughput."""
    rows = _inventory_rows(inventory)
    _require(flow in ("fill", "resume", "hit", "stage_hit", "stage_miss"), "needed_cache_flow_invalid")
    present = frozenset(present_paths)
    _require(present <= {row["relative_path"] for row in rows}, "needed_cache_flow_invalid")
    q = sum((row["size_bytes"] + _QUANTUM - 1) // _QUANTUM for row in rows)
    qp = sum((r["size_bytes"] + _QUANTUM - 1) // _QUANTUM for r in rows if r["relative_path"] in present)
    passes = {"fill": 3*q, "resume": 2*qp + 3*(q-qp), "hit": q, "stage_hit": 2*q, "stage_miss": 3*q}[flow]
    parts = sum((r["size_bytes"] + _PART - 1)//_PART for r in rows)
    ranges = sum((r["size_bytes"] + 128*_QUANTUM - 1)//(128*_QUANTUM)
                 for r in rows if r["size_bytes"] >= 512*_QUANTUM)
    # Preserve reviewed conservative finite maxima; actual smaller shapes consume less.
    overhead = 128*24 + 32*256 + 4*16*156 + 4*256 + 16
    return dict(file_count=len(rows), logical_bytes=sum(r["size_bytes"] for r in rows),
                quantum_count=q, payload_windows=passes, part_count=parts, range_count=ranges,
                maximum_checks=8*passes+overhead+(2*parts if flow == "stage_miss" else 0))


def require_cache_use(value):
    # Reject duck objects, native subclasses, reload classes and partial exact instances BEFORE any callback.
    _require(type(value) is NeededCheckpointCacheUse, "needed_cache_use_invalid")
    state = object.__getattribute__(value, "__dict__")
    required = {"_initialized", "_closed", "_files", "_root_fd", "_root", "_layout", "_entry", "_birth",
                "_sources", "_rows", "_inventory", "_gid", "_mode", "_authority_epoch", "_authority_version",
                "_writer", "_now", "_monotonic", "_origin", "_last", "_deadline", "_failure", "_lock",
                "_counts", "_windows", "_checks", "_health", "_reservation", "_resources",
                "_native_pending", "_native_unknown"}
    _require(required <= state.keys() and state.get("_initialized") is _USE_TOKEN,
             "needed_cache_use_invalid")
    _require(type(state["_closed"]) is bool and type(state["_files"]) is _PayloadFiles
             and type(state["_root"]) is type(Path("/")) and type(state["_root_fd"]) is int
             and state["_root_fd"] >= 0 and type(state["_sources"]) in (list, tuple)
             and type(state["_rows"]) is list and type(state["_mode"]) is str
             and state["_mode"] in ("read", "fill")
             and all(type(state[k]) is dict for k in ("_layout", "_entry", "_birth", "_inventory",
                                                    "_counts", "_windows", "_resources"))
             and all(type(state[k]) is int and state[k] >= 0 for k in
                     ("_gid", "_checks", "_authority_version", "_native_pending", "_native_unknown"))
             and all(type(state[k]) in (int, float) and math.isfinite(state[k]) for k in
                     ("_origin", "_last", "_deadline"))
             and callable(state["_now"]) and callable(state["_monotonic"])
             and (state["_failure"] is None or type(state["_failure"]) is str),
             "needed_cache_use_invalid")
    _require(not state["_closed"], "needed_cache_use_closed")
    return value


def _lock(files, parent, name, mode, operation):
    files.location(parent)
    fd = files.open(name, os.O_RDONLY | os.O_NONBLOCK, parent=parent)
    info = files.acquired[fd]
    _require(stat.S_ISREG(info.st_mode) and info.st_uid == 0 and info.st_nlink == 1
             and stat.S_IMODE(info.st_mode) == mode and info.st_size == 0, "needed_cache_lock_unsafe")
    files.location(fd)
    try:
        fcntl.flock(fd, operation | fcntl.LOCK_NB)
    except BlockingIOError:
        raise NeededCheckpointCacheError("needed_cache_busy") from None
    return fd


def _private_store(files, config):
    parent, _ = files.parent(Path(config.needed_checkpoint_cache_record_store) / "record.json", protected=True)
    owners._protected(os.fstat(parent), directory=True, mode=0o700)
    _lock(files, parent, _STORE_LOCK, 0o600, fcntl.LOCK_EX)
    return parent


def _authority_parent(files, config, operation):
    parent, _ = files.parent(Path(config.needed_checkpoint_cache_authority_root) / "HEAD.json")
    info = os.fstat(parent)
    _, gid = _blueprint_identity()
    _require(info.st_uid == 0 and info.st_gid == gid and stat.S_IMODE(info.st_mode) == 0o750,
             "needed_cache_authority_unsafe")
    _lock(files, parent, ".authority.lock", 0o640, operation)
    return parent


def _store_capacity(files, parent, *, new_intent=None):
    total, count = 0, 0
    files.slot()
    with os.scandir(parent) as entries:
        for entry in entries:
            files.budget.charge("entries")
            files.location(parent)
            info = os.stat(entry.name, dir_fd=parent, follow_symlinks=False)
            owners._protected(info, mode=0o600)
            if entry.name == _STORE_LOCK:
                _require(info.st_size == 0, "needed_cache_store_unsafe")
                continue
            _require(re.fullmatch(r"[0-9a-f]{32}(?:\.[a-z-]+)?\.json", entry.name)
                     and 0 < info.st_size <= 32768, "needed_cache_store_unsafe")
            total += info.st_size
            count += 1
            _require(count <= 256 and total <= 64*1024*1024, "needed_cache_store_full")
            if new_intent is not None and re.fullmatch(r"[0-9a-f]{32}\.json", entry.name):
                fd = files.open(entry.name, os.O_RDONLY, parent=parent)
                old = retained._document(files.read_bytes(fd, 32768), 32768, _work_budget=files.budget)
                _require(old.get("name") != new_intent["name"], "needed_cache_name_consumed")
                files.close(fd)
    return total, count


def issue_needed_checkpoint_cache_intent(*, principal, owner, name, reference_kind, reference_value,
        lease_ttl_seconds, size_budget_bytes, inventory_raw_sha256, inventory_raw_size_bytes,
        installed_config_path=_DEFAULT_CONFIG, now=time.time, monotonic=time.monotonic):
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000, monotonic=monotonic))
    try:
        _require(os.geteuid() == 0, "needed_cache_root_required")
        config = installed._configuration(files, installed_config_path)
        _require(Path(config.lane_scratch_work_root) == _REGISTERED_ROOTS[0],
                 "needed_cache_namespace_invalid")
        _require(config.needed_checkpoint_cache_creation_enabled is True, "needed_cache_creation_disabled")
        issued = now()
        _require(owners._number(issued) and owners._matches(principal, owners._PRINCIPAL)
                 and owners._matches(owner, owners._OWNER) and owners._matches(name, scratch._ID)
                 and reference_kind in ("run_ref", "scene_ref") and owners._matches(reference_value, owners._OWNER)
                 and type(lease_ttl_seconds) is int and 0 < lease_ttl_seconds <= 1209600
                 and type(size_budget_bytes) is int and 0 < size_budget_bytes <= _MAX_BYTES
                 and owners._matches(inventory_raw_sha256, owners._DIGEST)
                 and type(inventory_raw_size_bytes) is int and 0 < inventory_raw_size_bytes <= 65536,
                 "needed_cache_intent_invalid")
        raw, policy_record = files.read(config.lane_owner_policy_file, cap=65536, protected=True, mode=0o600)
        policy = owners._policy(raw, principal, files.budget)
        expiry = issued + lease_ttl_seconds
        owners._authorize(dict(action="register", owner=owner, ttl_seconds=lease_ttl_seconds), policy, expiry, issued)
        inventory_raw, inventory_record = files.read(config.needed_checkpoint_cache_inventory_file,
                                                      cap=65536, protected=True)
        owners._identity(inventory_raw, inventory_raw_sha256, inventory_raw_size_bytes, files.budget)
        inventory = retained._document(inventory_raw, 65536, _work_budget=files.budget)
        _inventory_rows(inventory)
        intent_id, generation = secrets.token_hex(16), secrets.token_hex(16)
        _require(intent_id != generation and owners._matches(intent_id, owners._CONSENT_ID)
                 and owners._matches(generation, owners._CONSENT_ID), "needed_cache_intent_invalid")
        intent = dict(intent_id=intent_id, schema_version=SCHEMA, issuer_kind="local_root", issuer_uid=0,
            principal=principal, owner=owner, root="work", lane="g1-checkpoint", name=name,
            reference_kind=reference_kind, reference_value=reference_value, reason="g1_checkpoint_retry_cache",
            class_intent="cache", cleanup="owner_review", issued_at_epoch=issued, expires_at_epoch=expiry,
            lease_ttl_seconds=lease_ttl_seconds, size_budget_bytes=size_budget_bytes,
            inventory_raw_sha256=inventory_raw_sha256, inventory_raw_size_bytes=inventory_raw_size_bytes,
            candidate_inventory_digests=[c["inventory_digest"] for c in inventory["candidates"]],
            policy_sha256=retained._digest(raw, _work_budget=files.budget), policy_size_bytes=len(raw),
            writer_scope="g1_checkpoint_fetcher.v1", generation=generation)
        intent["intent_digest"] = canonical_digest(intent, digest_field="intent_digest")
        payload = owners._encoded(intent, files.budget, cap=32768)
        _authority_parent(files, config, fcntl.LOCK_EX)
        store = _private_store(files, config)
        total, count = _store_capacity(files, store, new_intent=intent)
        _require(count < 256 and total + len(payload) <= 64*1024*1024, "needed_cache_store_full")
        files.verify_record(policy_record)
        files.verify_record(inventory_record)
        published = _publish_owned_metadata(files, store, intent_id + ".json", payload,
                                            mode=0o600, artifact_kind="attestation")
        return dict(intent_id=intent_id, intent={k: published[k] for k in ("sha256", "size_bytes")})
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()


_USE_TOKEN = object()


class NeededCheckpointCacheUse:
    """Exact private lifetime; the original four-hour controller is never reset."""
    def __init__(self):
        raise NeededCheckpointCacheError("needed_cache_use_invalid")

    @classmethod
    def open_registered(cls, root, *, mode="read", installed_config_path=_DEFAULT_CONFIG,
                        now=time.time, monotonic=time.monotonic):
        return _open_registered(cls, root, mode=mode, installed_config_path=installed_config_path,
                                now=now, monotonic=monotonic)

    def check(self):
        return _use_check(self)

    def row(self, path):
        return _use_row(self, path)

    def payload_open(self, row):
        return _payload_open(self, row)

    def chunks(self, path, *, role):
        return _use_chunks(self, path, role=role)

    def hash_file(self, path, *, role="wam_hash"):
        return _use_hash(self, path, role=role)

    def materialize_candidate(self, **kwargs):
        return _use_materialize(self, **kwargs)

    def verify_cache(self, root):
        return _use_verify(self, root)

    def download_file(self, row):
        return _use_download(self, row)

    def close(self):
        return _use_close(self)

    def __enter__(self):
        return require_cache_use(self)

    def __exit__(self, *args):
        self.close()

    @property
    def closed(self):
        return self._closed

    @property
    def failure(self):
        return self._failure

    @property
    def resource_counters(self):
        return dict(current_acquisitions=self._checks,
                    roles={k: dict(v) for k, v in self._counts.items()},
                    maximum_current_acquisitions=2*self._resources["maximum_checks"])


class _PayloadFiles:
    """Retain named originals; unknown first tokens and reused numbers stay foreign."""
    def __init__(self, initial_budget=None):
        self.initial_budget = initial_budget
        self.owned, self.bindings = {}, {}
        self.unresolved = 0

    def proof(self, fd):
        if self.initial_budget is not None:
            self.initial_budget.tick()
        expected = self.owned.get(fd)
        _require(expected is not None, "needed_cache_descriptor_unproven")
        info = os.fstat(fd)
        _require((info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)) == expected,
                 "needed_cache_descriptor_changed")
        return info

    def location(self, fd):
        for _ in range(64):
            info = self.proof(fd)
            parent, name, security = self.bindings[fd]
            if parent is not None:
                self.proof(parent)
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            _require(owners._security(info) == security == owners._security(named)
                     and not info.st_mode & 0o022, "needed_cache_location_changed")
            if parent is None:
                return info
            fd = parent
        raise NeededCheckpointCacheError("needed_cache_resource_exhausted")

    def open(self, parent, name, flags, *, mode=0o640):
        if self.initial_budget is not None:
            self.initial_budget.tick()
        _require(len(self.owned) < 104, "needed_cache_descriptor_limit")
        if parent is not None:
            self.location(parent)
        creating = bool(flags & os.O_CREAT)
        _require(not flags & (os.O_TRUNC | os.O_APPEND) and (not creating or flags & os.O_EXCL),
                 "needed_cache_open_invalid")
        if creating:
            try:
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise NeededCheckpointCacheError("needed_cache_destination_exists")
            named = None
        else:
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            _require(stat.S_ISDIR(named.st_mode) if flags & os.O_DIRECTORY else stat.S_ISREG(named.st_mode),
                     "needed_cache_open_invalid")
        fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, mode, dir_fd=parent)
        try:
            if parent is not None:
                self.location(parent)
            if creating:
                named = os.stat(name, dir_fd=parent, follow_symlinks=False)
                _require(stat.S_ISREG(named.st_mode) and named.st_uid == 0 and named.st_nlink == 1
                         and not named.st_mode & 0o022, "needed_cache_descriptor_unproven")
            actual = os.fstat(fd)
            expected = (named.st_dev, named.st_ino, stat.S_IFMT(named.st_mode))
            _require(type(fd) is int and fd not in self.owned and
                     (actual.st_dev, actual.st_ino, stat.S_IFMT(actual.st_mode)) == expected,
                     "needed_cache_descriptor_unproven")
        except (OSError, ValueError):
            self.unresolved += 1
            raise NeededCheckpointCacheError("needed_cache_descriptor_unproven") from None
        self.owned[fd] = expected
        self.bindings[fd] = (parent, name, owners._security(named))
        self.location(fd)
        return fd

    def directory(self, path):
        path = Path(path)
        _require(path.is_absolute() and ".." not in path.parts and len(path.parts) <= 64,
                 "needed_cache_path_invalid")
        current = self.open(None, "/", os.O_RDONLY | os.O_DIRECTORY)
        for part in path.parts[1:]:
            child = self.open(current, part, os.O_RDONLY | os.O_DIRECTORY)
            current = child
        return current

    def close(self, fd):
        expected = self.owned.get(fd)
        if expected is None:
            return
        for _ in range(2):
            try:
                info = os.fstat(fd)
            except OSError:
                continue
            if (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)) != expected:
                self.owned.pop(fd, None)
                self.unresolved += 1
                return
            try:
                os.close(fd)
            except OSError:
                continue
            self.owned.pop(fd, None)
            return

    def finish(self):
        for fd in reversed(tuple(self.owned)):
            self.close(fd)
        _require(not self.owned and not self.unresolved, "needed_cache_cleanup_incomplete")


def _public_read(files, path, cap, mode, gid):
    raw, acquired = files.read(path, cap=cap)
    info = acquired.info
    _require(info.st_uid == 0 and info.st_gid == gid and stat.S_IMODE(info.st_mode) == mode
             and info.st_nlink == 1 and 0 < info.st_size <= cap, "needed_cache_public_record_unsafe")
    files.verify_record(acquired)
    return retained._document(raw, cap, _work_budget=files.budget), raw


def _current_projection(files, root, gid, *, _lock_held=False):
    parent, _ = files.parent(Path(root) / "HEAD.json")
    info = os.fstat(parent)
    _require(info.st_uid == 0 and info.st_gid == gid and stat.S_IMODE(info.st_mode) == 0o750,
             "needed_cache_authority_unsafe")
    if not _lock_held:
        _lock(files, parent, ".authority.lock", 0o640, fcntl.LOCK_SH)
    head, raw = _public_read(files, Path(root) / "HEAD.json", 4096, 0o640, gid)
    _require(set(head) == {"schema_version", "authority_epoch_id", "version", "record_name", "record_sha256",
                          "record_size_bytes", "head_digest"}
             and head["schema_version"] == "control_plane_needed_cache_authority_head.v1"
             and owners._matches(head["authority_epoch_id"], owners._CONSENT_ID)
             and type(head["version"]) is int and 0 < head["version"] <= 1000000
             and head["record_name"] == f"v-{head['version']:012d}.json"
             and head["head_digest"] == canonical_digest(head, digest_field="head_digest"),
             "needed_cache_head_invalid")
    value, source = _public_read(files, Path(root) / head["record_name"], 32768, 0o640, gid)
    owners._identity(source, head["record_sha256"], head["record_size_bytes"], files.budget)
    _require(set(value) == {"schema_version", "authority_epoch_id", "version", "previous_record_sha256", "state",
        "issued_at_epoch", "expires_at_epoch", "policy_raw_sha256", "policy_raw_size_bytes", "enrollments", "authority_digest"}
        and value["schema_version"] == "control_plane_needed_cache_authority.v1"
        and value["authority_epoch_id"] == head["authority_epoch_id"] and value["version"] == head["version"]
        and value["authority_digest"] == canonical_digest(value, digest_field="authority_digest")
        and value["state"] in ("enabled", "disabled") and owners._number(value["issued_at_epoch"])
        and owners._number(value["expires_at_epoch"]) and value["issued_at_epoch"] < value["expires_at_epoch"]
        and (value["previous_record_sha256"] is None if head["version"] == 1 else
             owners._matches(value["previous_record_sha256"], owners._DIGEST))
        and owners._matches(value["policy_raw_sha256"], owners._DIGEST)
        and type(value["policy_raw_size_bytes"]) is int and 0 < value["policy_raw_size_bytes"] <= 65536
        and type(value["enrollments"]) is list and len(value["enrollments"]) <= 100,
        "needed_cache_authority_invalid")
    seen = set()
    fields = {"intent_id", "generation", "target", "birth_raw_sha256", "birth_raw_size_bytes",
              "lease_raw_sha256", "lease_raw_size_bytes", "owner", "inventory_raw_sha256",
              "inventory_raw_size_bytes", "size_budget_bytes", "admission_state", "expires_at_epoch"}
    for row in value["enrollments"]:
        files.budget.charge("facts")
        _require(type(row) is dict and set(row) == fields and owners._matches(row["intent_id"], owners._CONSENT_ID)
                 and row["intent_id"] not in seen and owners._matches(row["generation"], owners._CONSENT_ID)
                 and type(row["target"]) is dict and set(row["target"]) == {"root", "lane", "name"}
                 and row["target"]["root"] == "work" and row["target"]["lane"] == "g1-checkpoint"
                 and owners._matches(row["target"]["name"], scratch._ID)
                 and row["admission_state"] in ("fill_only", "ready", "revoked")
                 and owners._matches(row["owner"], owners._OWNER) and owners._number(row["expires_at_epoch"])
                 and type(row["size_budget_bytes"]) is int and 0 < row["size_budget_bytes"] <= _MAX_BYTES,
                 "needed_cache_authority_invalid")
        for prefix, cap in (("birth", 32768), ("lease", 8192), ("inventory", 65536)):
            _require(owners._matches(row[prefix+"_raw_sha256"], owners._DIGEST)
                     and type(row[prefix+"_raw_size_bytes"]) is int
                     and 0 < row[prefix+"_raw_size_bytes"] <= cap, "needed_cache_authority_invalid")
        seen.add(row["intent_id"])
    files.verify()
    return head, value, raw


def _read_layout(files, config_path):
    config = installed._configuration(files, config_path)
    _require(Path(config.lane_scratch_work_root) == _REGISTERED_ROOTS[0],
             "needed_cache_namespace_invalid")
    return dict(root=Path(config.lane_scratch_work_root),
                public=Path(config.needed_checkpoint_cache_registration_root),
                authority=Path(config.needed_checkpoint_cache_authority_root),
                inventory=Path(config.needed_checkpoint_cache_inventory_file),
                private=Path(config.needed_checkpoint_cache_record_store), config=config)


def _public_reader_layout(files, target):
    """Fixed accessible installation, not private config or environment capability."""
    target = Path(target)
    root = _REGISTERED_ROOTS[0]
    _require(target.parent == root / "g1-checkpoint", "needed_cache_target_invalid")
    public, inventory = Path(_PUBLIC_REGISTRATION), Path(_PUBLIC_INVENTORY)
    parent, _ = files.parent(public / "record.json", protected=True)
    info = os.fstat(parent)
    _require(info.st_uid == info.st_gid == 0 and stat.S_IMODE(info.st_mode) == 0o755,
             "needed_cache_registration_unsafe")
    files.parent(inventory, protected=True)
    return dict(root=root, public=public, authority=public / "authority", inventory=inventory,
                private=None, config=None)


def _validate_public_birth(birth, entry):
    fields = {"schema_version", "intent_id", "generation", "target", "claim_raw_sha256", "claim_raw_size_bytes",
              "publication_raw_sha256", "publication_raw_size_bytes", "target_identity", "lease_raw_sha256",
              "lease_raw_size_bytes", "marker_raw_sha256", "marker_raw_size_bytes", "owner", "reference_kind",
              "reference_value", "class_intent", "cleanup", "size_budget_bytes", "expires_at_epoch",
              "inventory_raw_sha256", "inventory_raw_size_bytes", "writer_scope", "birth_digest"}
    _require(type(birth) is dict and set(birth) == fields
             and birth["schema_version"] == "control_plane_needed_cache_birth.v1"
             and birth["birth_digest"] == canonical_digest(birth, digest_field="birth_digest")
             and birth["target"] == entry["target"] and birth["generation"] == entry["generation"]
             and birth["intent_id"] == entry["intent_id"] and birth["owner"] == entry["owner"]
             and birth["class_intent"] == "cache" and birth["cleanup"] == "owner_review"
             and birth["size_budget_bytes"] == entry["size_budget_bytes"]
             and birth["inventory_raw_sha256"] == entry["inventory_raw_sha256"]
             and birth["inventory_raw_size_bytes"] == entry["inventory_raw_size_bytes"]
             and birth["writer_scope"] == "g1_checkpoint_fetcher.v1"
             and birth["reference_kind"] in ("run_ref", "scene_ref")
             and owners._matches(birth["reference_value"], owners._OWNER)
             and owners._number(birth["expires_at_epoch"]) and birth["expires_at_epoch"] > 0,
             "needed_cache_birth_invalid")
    identity = birth["target_identity"]
    _require(type(identity) is dict and set(identity) == {"dev", "ino", "type"}
             and identity["type"] == "directory"
             and all(type(identity[k]) is int and identity[k] >= 0 for k in ("dev", "ino")),
             "needed_cache_birth_invalid")
    for prefix, cap in (("claim", 32768), ("publication", 32768), ("lease", 8192), ("marker", 4096)):
        _require(owners._matches(birth[prefix+"_raw_sha256"], owners._DIGEST)
                 and type(birth[prefix+"_raw_size_bytes"]) is int
                 and 0 < birth[prefix+"_raw_size_bytes"] <= cap, "needed_cache_birth_invalid")


def _retain_source_records(files, payload, records):
    selected = set()
    for record in records:
        files.verify_record(record)
        current = record.fd
        for _ in range(64):
            selected.add(current)
            parent, _, _ = files.bindings[current]
            if parent is None:
                break
            current = parent
        else:
            raise NeededCheckpointCacheError("needed_cache_resource_exhausted")
    _require(len(payload.owned) + len(selected) < 104, "needed_cache_descriptor_limit")
    for fd in selected:
        files.location(fd)
        _require(fd not in payload.owned, "needed_cache_descriptor_collision")
        payload.owned[fd] = files.proof(fd)
        payload.bindings[fd] = files.bindings[fd]
    for fd in selected:
        files.owned.pop(fd)
    return tuple(records)


def _open_registered(cls, root, *, mode="read", installed_config_path=_DEFAULT_CONFIG,
                     now=time.time, monotonic=time.monotonic):
    _require(cls is NeededCheckpointCacheUse and mode in ("read", "fill"), "needed_cache_use_invalid")
    origin = monotonic()
    _require(type(origin) in (int, float) and math.isfinite(origin), "needed_cache_clock_invalid")
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000, monotonic=monotonic))
    payload = _PayloadFiles(files.budget)
    result = None
    try:
        # Preflight current authority completes before an existing target SH acquisition.
        if mode == "read" and os.fspath(installed_config_path) == _DEFAULT_CONFIG:
            layout = _public_reader_layout(files, root)
        else:
            _require(os.geteuid() == 0, "needed_cache_root_required")
            layout = _read_layout(files, installed_config_path)
        _, gid = _blueprint_identity()
        head, authority, _ = _current_projection(files, layout["authority"], gid)
        target = Path(root)
        _require(target.parent == layout["root"] / "g1-checkpoint"
                 and owners._matches(target.name, scratch._ID), "needed_cache_target_invalid")
        entries = [r for r in authority["enrollments"] if r["target"]["name"] == target.name]
        _require(len(entries) == 1, "needed_cache_not_registered")
        entry = entries[0]
        current = now()
        _require(owners._number(current) and authority["state"] == "enabled"
                 and authority["issued_at_epoch"] <= current < authority["expires_at_epoch"]
                 and current < entry["expires_at_epoch"]
                 and entry["admission_state"] == ("ready" if mode == "read" else "fill_only"),
                 "needed_cache_admission_refused")
        birth, birth_raw = _public_read(files, layout["public"] / (entry["intent_id"] + ".birth.json"),
                                        32768, 0o644, 0)
        owners._identity(birth_raw, entry["birth_raw_sha256"], entry["birth_raw_size_bytes"], files.budget)
        _validate_public_birth(birth, entry)
        birth_record = files.records[-1]
        inventory_raw, inventory_record = files.read(layout["inventory"], cap=65536)
        owners._identity(inventory_raw, entry["inventory_raw_sha256"], entry["inventory_raw_size_bytes"], files.budget)
        inventory = retained._document(inventory_raw, 65536, _work_budget=files.budget)
        rows = _inventory_rows(inventory)
        # Preserve named originals while releasing the administrative SH.
        sources = _retain_source_records(files, payload, (birth_record, inventory_record))
        files.finish()
        files.budget.tick()
        root_fd = payload.directory(target)
        info = payload.proof(root_fd)
        _require(info.st_uid == 0 and info.st_gid == gid and stat.S_IMODE(info.st_mode) == 0o750
                 and birth["target_identity"] == dict(dev=info.st_dev, ino=info.st_ino, type="directory"),
                 "needed_cache_target_changed")
        fcntl.flock(root_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        lifetime = payload.open(root_fd, ".needed-cache-lifetime.lock", os.O_RDONLY)
        _require(payload.proof(lifetime).st_size == 0, "needed_cache_lock_unsafe")
        fcntl.flock(lifetime, fcntl.LOCK_SH | fcntl.LOCK_NB)
        writer = None
        if mode == "fill":
            _require(os.geteuid() == 0, "needed_cache_root_required")
            writer = payload.open(root_fd, ".needed-cache-writer.lock", os.O_RDWR)
            fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = object.__new__(NeededCheckpointCacheUse)
        result._initialized, result._closed = _USE_TOKEN, False
        result._files, result._root_fd, result._root = payload, root_fd, target
        result._layout, result._entry, result._birth = layout, entry, birth
        result._sources = sources
        result._rows, result._inventory, result._gid, result._mode = rows, inventory, gid, mode
        result._authority_epoch, result._authority_version = head["authority_epoch_id"], head["version"]
        result._writer, result._now, result._monotonic = writer, now, monotonic
        result._origin = result._last = origin
        _require(type(result._origin) in (int, float) and math.isfinite(result._origin), "needed_cache_clock_invalid")
        result._deadline, result._failure = result._origin + 4*3600, None
        result._lock, result._counts, result._windows = threading.RLock(), {}, {}
        result._checks, result._health, result._reservation = 0, None, None
        result._native_pending, result._native_unknown = 0, 0
        result._resources = derive_checkpoint_flow_resources(inventory, "stage_miss" if mode == "read" else "fill")
        result._initial_budget = files.budget
        result.check()
        files.budget.tick()
        result._initial_budget = None
        payload.initial_budget = None
        return result
    except (OSError, ValueError) as exc:
        if result is not None:
            result._closed = True
        try:
            payload.finish()
        except ValueError:
            raise NeededCheckpointCacheError("needed_cache_cleanup_incomplete") from None
        if isinstance(exc, NeededCheckpointCacheError):
            raise
        raise NeededCheckpointCacheError("needed_cache_acquisition_failed") from None
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()




def _use_check(self):
    require_cache_use(self)
    with self._lock:
        if self._failure:
            raise NeededCheckpointCacheError(self._failure)
        try:
            value = self._monotonic()
            _require(type(value) in (int, float) and math.isfinite(value) and value >= self._last,
                     "needed_cache_clock_invalid")
            self._last = value
            _require(value < self._deadline, "needed_cache_deadline_exceeded")
            self._checks += 1
            _require(self._checks <= 2*self._resources["maximum_checks"], "needed_cache_check_limit")
            self._files.location(self._root_fd)
            for record in self._sources:
                self._files.location(record.fd)
                _require(owners._metadata(self._files.proof(record.fd)) == owners._metadata(record.info)
                         == owners._metadata(os.stat(record.name, dir_fd=record.parent, follow_symlinks=False)),
                         "needed_cache_source_changed")
            if self._health is not None:
                self._health.check()
            with _metadata(self._monotonic, _budget=getattr(self, "_initial_budget", None)) as files:
                head, current, _ = _current_projection(files, self._layout["authority"], self._gid)
                _require(head["authority_epoch_id"] == self._authority_epoch
                         and head["version"] >= self._authority_version, "needed_cache_authority_rollback")
                moment = self._now()
                entries = [r for r in current["enrollments"] if r["intent_id"] == self._entry["intent_id"]]
                _require(len(entries) == 1 and entries[0] == self._entry
                         and current["state"] == "enabled" and owners._number(moment)
                         and current["issued_at_epoch"] <= moment < current["expires_at_epoch"]
                         and moment < self._entry["expires_at_epoch"], "needed_cache_current_refused")
                self._authority_version = head["version"]
                lease, lease_raw = _public_read(files, self._root / scratch.LEASE_FILE, 8192, 0o640, self._gid)
                owners._identity(lease_raw, self._entry["lease_raw_sha256"], self._entry["lease_raw_size_bytes"], files.budget)
                marker, marker_raw = _public_read(files, self._root / MARKER, 4096, 0o640, self._gid)
                owners._identity(marker_raw, self._birth["marker_raw_sha256"], self._birth["marker_raw_size_bytes"], files.budget)
                _require(scratch._lease_fields_valid(lease) and lease.get("lease_digest") ==
                         canonical_digest(lease, digest_field="lease_digest")
                         and lease["lane"] == "g1-checkpoint" and lease["name"] == self._root.name
                         and lease["owner"] == self._entry["owner"] and lease["class_intent"] == "cache"
                         and lease["cleanup"] == "owner_review" and lease["released_at_epoch"] is None
                         and lease["size_budget_bytes"] == self._entry["size_budget_bytes"]
                         and lease["expires_at_epoch"] == self._entry["expires_at_epoch"]
                         and marker.get("marker_digest") == canonical_digest(marker, digest_field="marker_digest")
                         and marker.get("intent_id") == self._entry["intent_id"]
                         and marker.get("generation") == self._entry["generation"], "needed_cache_metadata_changed")
            return self
        except (OSError, ValueError) as exc:
            self._failure = str(exc) if isinstance(exc, NeededCheckpointCacheError) else "needed_cache_checkpoint_failed"
            raise NeededCheckpointCacheError(self._failure) from None




@contextmanager
def _metadata(monotonic=time.monotonic, *, _budget=None):
    budget = _budget if _budget is not None else ReferenceCollectionBudget(values_limit=10000, monotonic=monotonic)
    files = _BirthFiles(budget)
    try:
        files.budget.tick()
        yield files
    finally:
        try:
            files.finish()
        finally:
            if _budget is None:
                files.budget.close()


def _use_row(self, path):
    require_cache_use(self)
    path = Path(path)
    try:
        relative = path.relative_to(self._root).as_posix()
    except ValueError:
        raise NeededCheckpointCacheError("needed_cache_payload_outside") from None
    rows = [r for r in self._rows if r["relative_path"] == relative]
    _require(len(rows) == 1, "needed_cache_payload_unpinned")
    return rows[0]


@contextmanager
def _payload_open(self, row):
    self.check()
    parent = self._root_fd
    directories = []
    fd = None
    parts = PurePosixPath(row["relative_path"]).parts
    try:
        for part in parts[:-1]:
            self.check()
            child = self._files.open(parent, part, os.O_RDONLY | os.O_DIRECTORY)
            _require(self._files.proof(child).st_dev == self._files.proof(self._root_fd).st_dev,
                     "needed_cache_cross_device")
            directories.append(child)
            parent = child
        self.check()
        fd = self._files.open(parent, parts[-1], os.O_RDONLY)
        before = self._files.proof(fd)
        _require(before.st_uid == 0 and before.st_gid == self._gid and stat.S_IMODE(before.st_mode) == 0o640
                 and before.st_nlink == 1 and before.st_size == row["size_bytes"]
                 and before.st_dev == self._files.proof(self._root_fd).st_dev, "needed_cache_payload_unsafe")
        yield fd, parent, parts[-1], before
        self.check()
        _require(owners._metadata(self._files.proof(fd)) == owners._metadata(before)
                 == owners._metadata(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)),
                 "needed_cache_payload_changed")
    finally:
        if fd is not None:
            self._files.close(fd)
        for item in reversed(directories):
            self._files.close(item)
        if self._files.unresolved:
            self._failure = "needed_cache_cleanup_incomplete"
            raise NeededCheckpointCacheError(self._failure)


def _use_chunks(self, path, *, role):
    try:
        yield from _use_chunks_checked(self, path, role=role)
    except (OSError, ValueError) as exc:
        self._failure = str(exc) if isinstance(exc, NeededCheckpointCacheError) else "needed_cache_payload_failed"
        raise NeededCheckpointCacheError(self._failure) from None


def _use_chunks_checked(self, path, *, role):
    _require(role in ("verify", "wam_hash", "upload", "preverify", "existing_hash", "fill_hash"), "needed_cache_role_invalid")
    row = self.row(path)
    with self.payload_open(row) as (fd, parent, name, before):
        offset = 0
        while offset < row["size_bytes"]:
            self.check()
            window = offset // _QUANTUM
            requested = min(_QUANTUM-offset % _QUANTUM, row["size_bytes"]-offset)
            key = (role, row["relative_path"], window)
            with self._lock:
                self._windows[key] = self._windows.get(key, 0) + 1
                _require(self._windows[key] <= 8, "needed_cache_fragment_limit")
                counts = self._counts.setdefault(role, dict(bytes=0, calls=0))
                counts["calls"] += 1
            self._files.location(fd)
            _require(owners._metadata(self._files.proof(fd)) == owners._metadata(before)
                     == owners._metadata(os.stat(name, dir_fd=parent, follow_symlinks=False)),
                     "needed_cache_payload_changed")
            block = os.pread(fd, requested, offset)
            self.check()
            _require(type(block) is bytes and 0 < len(block) <= requested, "needed_cache_payload_short")
            offset += len(block)
            counts["bytes"] += len(block)
            _require(counts["bytes"] <= self._resources["logical_bytes"], "needed_cache_role_byte_limit")
            yield block
        self.check()
        _require(os.pread(fd, 1, offset) == b"", "needed_cache_payload_extra")
        self.check()


def _use_hash(self, path, *, role="wam_hash"):
    try:
        return _use_hash_checked(self, path, role=role)
    except (OSError, ValueError) as exc:
        self._failure = str(exc) if isinstance(exc, NeededCheckpointCacheError) else "needed_cache_hash_failed"
        raise NeededCheckpointCacheError(self._failure) from None


def _use_hash_checked(self, path, *, role="wam_hash"):
    digest, size = hashlib.sha256(), 0
    for chunk in self.chunks(path, role=role):
        digest.update(chunk)
        size += len(chunk)
    row = self.row(path)
    _require("sha256:" + digest.hexdigest() == row["sha256"] and size == row["size_bytes"],
             "needed_cache_payload_hash_changed")
    return digest.hexdigest(), size


def _use_materialize(self, *, inventory_path, candidate_id, output_dir, verify_only, cancel_event=None):
    require_cache_use(self)
    _require(Path(output_dir) == self._root and Path(inventory_path) == self._layout["inventory"],
             "needed_cache_inventory_unbound")
    selected = [r for r in self._rows if r["candidate_id"] == candidate_id]
    _require(len(selected) == 6, "needed_cache_candidate_unpinned")
    result = []
    for row in selected:
        if cancel_event is not None and cancel_event.is_set():
            raise NeededCheckpointCacheError("needed_cache_cancelled")
        self.check()
        try:
            os.stat(self._root / row["relative_path"], follow_symlinks=False)
        except FileNotFoundError:
            _require(not verify_only and self._mode == "fill" and self._reservation is not None,
                     "needed_cache_payload_missing")
            self.download_file(row)
        else:
            self.hash_file(self._root / row["relative_path"], role="verify" if verify_only else "existing_hash")
        result.append({key: row[key] for key in ("relative_path", "sha256", "size_bytes")})
    candidate = next(c for c in self._inventory["candidates"] if c["candidate_id"] == candidate_id)
    return dict(status="checkpoint_bytes_verified", candidate_id=candidate_id,
        candidate_inventory_digest=candidate["inventory_digest"], policy_role=candidate.get("policy_role"),
        inventory_file_sha256=self._entry["inventory_raw_sha256"], files=result)


def _use_verify(self, root):
    _require(Path(root) == self._root and self._mode == "read", "needed_cache_reader_invalid")
    from .native_g1_development_pair import PAIR_ORDER
    from .native_g1_checkpoint_cache import _fetcher
    fetcher = _fetcher()
    result = []
    for candidate in PAIR_ORDER:
        receipt = fetcher.materialize_candidate(inventory_path=self._layout["inventory"], candidate_id=candidate,
            output_dir=self._root, verify_only=True, _cache_use=self)
        _require(receipt.get("status") == "checkpoint_bytes_verified", "needed_cache_verification_failed")
        result.extend(receipt["files"])
    return result


def _use_close(self):
    require_cache_use(self)
    self._closed = True
    self._files.finish()
    _require(self._native_pending == self._native_unknown == 0, "needed_cache_native_cleanup_incomplete")




def _raw(payload):
    return dict(sha256="sha256:" + hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))


def _encode(files, value, digest_field, cap=32768):
    files.budget.measure(value, cap=cap-100)
    record = value | {digest_field: canonical_digest(value, digest_field=digest_field)}
    return record, owners._encoded(record, files.budget, cap=cap)


def _publish_metadata(files, parent, name, payload, *, mode, gid=0, cap=32768):
    """Own immutable cache publication; no replacement or shared schema widening."""
    _require(type(payload) is bytes and len(payload) <= cap and mode in (0o600, 0o640, 0o644)
             and type(name) is str and (re.fullmatch(r"[0-9a-f]{32}(?:\.[a-z-]+)?\.json", name)
             or re.fullmatch(r"v-[0-9]{12}\.json", name)
             or name in (scratch.LEASE_FILE, MARKER, ".needed-cache-lifetime.lock", ".needed-cache-writer.lock", "HEAD.json")),
             "needed_cache_publication_invalid")
    files.budget.charge("output_bytes", len(payload))
    files.location(parent)
    before_parent = files.proof(parent)
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise NeededCheckpointCacheError("needed_cache_destination_exists")
    temporary = ".target-version-" + secrets.token_hex(16) + ".tmp"
    fd = files.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL, parent=parent)
    expected, size, links, current_name, published = files.proof(fd), 0, 1, temporary, False
    current_mode, current_gid = 0o600, 0
    def guard(*, cleanup=False):
        if not cleanup:
            files.budget.tick()
        files.location(parent, cleanup=cleanup)
        _require(files.proof(parent) == before_parent and files.proof(fd) == expected,
                 "needed_cache_publication_changed")
        info = os.fstat(fd)
        _require(info.st_uid == 0 and info.st_gid == current_gid and stat.S_IMODE(info.st_mode) == current_mode
                 and info.st_size == size and info.st_nlink == links and stat.S_ISREG(info.st_mode)
                 and owners._metadata(info) == owners._metadata(os.stat(current_name, dir_fd=parent, follow_symlinks=False)),
                 "needed_cache_publication_changed")
        if published:
            _require(owners._metadata(info) == owners._metadata(os.stat(name, dir_fd=parent, follow_symlinks=False)),
                     "needed_cache_publication_changed")
        else:
            try:
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise NeededCheckpointCacheError("needed_cache_destination_exists")
    try:
        while size < len(payload):
            guard()
            count = os.write(fd, memoryview(payload)[size:])
            _require(type(count) is int and 0 < count <= len(payload)-size, "needed_cache_publication_short")
            size += count
        guard()
        if gid != current_gid:
            os.fchown(fd, 0, gid)
            current_gid = gid
        guard()
        os.fchmod(fd, mode)
        current_mode = mode
        guard()
        os.fsync(fd)
        guard()
        os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        links, published = 2, True
        guard()
        os.unlink(temporary, dir_fd=parent)
        links, current_name = 1, name
        guard()
        os.fsync(parent)
        guard()
        check = files.open(name, os.O_RDONLY, parent=parent)
        _require(files.proof(check) == expected and files.read_bytes(check, max(1, cap)) == payload,
                 "needed_cache_publication_readback")
        guard()
        files.close(check)
        return _raw(payload)
    finally:
        if current_name == temporary:
            try:
                guard(cleanup=True)
                os.unlink(temporary, dir_fd=parent)
            except (OSError, ValueError):
                pass
        files.close(fd)


def _head_cas(files, parent, payload, previous):
    return _mutable_record_cas(files, parent, payload, previous, name="HEAD.json", cap=4096)


def _mutable_record_cas(files, parent, payload, previous, *, name, cap):
    """Only the two fixed cache current records; original inode, no replacement."""
    _require(name in ("HEAD.json", scratch.LEASE_FILE) and len(payload) <= cap, "needed_cache_cas_invalid")
    files.location(parent)
    if previous is None:
        return _publish_metadata(files, parent, name, payload, mode=0o640,
                                 gid=_blueprint_identity()[1], cap=cap)
    fd = files.open(name, os.O_RDWR, parent=parent)
    old = files.acquired[fd]
    _require(files.read_bytes(fd, cap) == previous and old.st_uid == 0 and old.st_nlink == 1
             and old.st_gid == _blueprint_identity()[1] and stat.S_IMODE(old.st_mode) == 0o640,
             "needed_cache_head_changed")
    # Remove only exact retained historical HEAD observations after its CAS proof.
    for record in tuple(files.records):
        if record.parent == parent and record.name == name:
            files.verify_record(record)
            files.records.remove(record)
    expected, size = files.proof(fd), old.st_size
    def guard():
        files.budget.tick()
        files.location(parent)
        _require(files.proof(fd) == expected, "needed_cache_head_changed")
        info = os.fstat(fd)
        _require(info.st_uid == old.st_uid and info.st_gid == old.st_gid and info.st_nlink == 1
                 and stat.S_IMODE(info.st_mode) == 0o640 and info.st_size == size
                 and owners._metadata(info) == owners._metadata(os.stat(name, dir_fd=parent, follow_symlinks=False)),
                 "needed_cache_head_changed")
    guard()
    os.ftruncate(fd, 0)
    size = 0
    while size < len(payload):
        guard()
        count = os.pwrite(fd, payload[size:], size)
        _require(type(count) is int and 0 < count <= len(payload)-size, "needed_cache_head_short")
        size += count
    guard()
    os.fsync(fd)
    guard()
    os.lseek(fd, 0, os.SEEK_SET)
    _require(files.read_bytes(fd, cap) == payload, "needed_cache_head_readback")
    guard()
    os.fsync(parent)
    guard()
    files.close(fd)
    return _raw(payload)


def _process_identity():
    # Linux kernel-owned fixed records, finite input; never identifies consumers.
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text()[:37].strip()
        raw = Path("/proc/self/stat").read_text()[:4096]
        ticks = int(raw[raw.rfind(")")+2:].split()[19])
        namespace = os.stat("/proc/self/ns/pid").st_ino
        _require(re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot)
                 and 0 <= ticks < 2**64 and 0 <= namespace < 2**64, "needed_cache_process_unproven")
        return dict(boot_id=boot, pid=os.getpid(), start_ticks=ticks, pid_namespace_inode=namespace)
    except (OSError, ValueError, IndexError):
        raise NeededCheckpointCacheError("needed_cache_process_unproven") from None


def _event(handles, intent, intent_ref, operation_id, kind, previous, moment, **fields):
    common = dict(schema_version="control_plane_needed_cache_"+kind+".v1", event_id=secrets.token_hex(16),
        previous_event_sha256=previous["sha256"] if previous else None,
        intent_raw_sha256=intent_ref["sha256"], intent_raw_size_bytes=intent_ref["size_bytes"],
        intent_id=intent["intent_id"], generation=intent["generation"],
        target=dict(root="work", lane="g1-checkpoint", name=intent["name"]), operation_id=operation_id,
        recorded_at_epoch=moment)
    return _encode(handles, common | fields, "event_digest")


def _record_event(handles, store, intent, intent_ref, operation, kind, previous, moment, **fields):
    record, payload = _event(handles, intent, intent_ref, operation, kind, previous, moment, **fields)
    return _publish_metadata(handles, store, record["event_id"]+".json", payload, mode=0o600)


def _read_intent(files, layout, intent_id, expected, moment):
    _require(owners._matches(intent_id, owners._CONSENT_ID), "needed_cache_intent_invalid")
    raw, acquired = files.read(layout["private"] / (intent_id+".json"), cap=32768, protected=True, mode=0o600)
    owners._identity(raw, expected["sha256"], expected["size_bytes"], files.budget)
    value = retained._document(raw, 32768, _work_budget=files.budget)
    _require(set(value) == _INTENT_FIELDS and value["schema_version"] == SCHEMA
             and value["intent_id"] == intent_id and value["issuer_kind"] == "local_root"
             and type(value["issuer_uid"]) is int and value["issuer_uid"] == 0
             and value["intent_digest"] == canonical_digest(value, digest_field="intent_digest")
             and value["root"] == "work" and value["lane"] == "g1-checkpoint"
             and value["class_intent"] == "cache" and value["cleanup"] == "owner_review"
             and owners._number(moment) and value["issued_at_epoch"] <= moment < value["expires_at_epoch"],
             "needed_cache_intent_invalid")
    policy, policy_record = files.read(layout["config"].lane_owner_policy_file, cap=65536, protected=True, mode=0o600)
    owners._identity(policy, value["policy_sha256"], value["policy_size_bytes"], files.budget)
    selected = owners._policy(policy, value["principal"], files.budget)
    owners._authorize(dict(action="register", owner=value["owner"], ttl_seconds=value["lease_ttl_seconds"]),
                      selected, value["expires_at_epoch"], value["issued_at_epoch"])
    inventory_raw, inventory_record = files.read(layout["inventory"], cap=65536, protected=True)
    owners._identity(inventory_raw, value["inventory_raw_sha256"], value["inventory_raw_size_bytes"], files.budget)
    inventory = retained._document(inventory_raw, 65536, _work_budget=files.budget)
    rows = _inventory_rows(inventory)
    _require(value["candidate_inventory_digests"] == [c["inventory_digest"] for c in inventory["candidates"]],
             "needed_cache_inventory_changed")
    files.verify_record(acquired)
    files.verify_record(policy_record)
    files.verify_record(inventory_record)
    return value, inventory, rows, moment + selected["max_consent_seconds"]


def _existing_projection(files, layout, gid):
    parent, _ = files.parent(layout["authority"] / "HEAD.json")
    try:
        os.stat("HEAD.json", dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        files.slot()
        with os.scandir(parent) as entries:
            for item in entries:
                files.budget.charge("entries")
                _require(item.name == ".authority.lock", "needed_cache_head_missing")
        return None
    return _current_projection(files, layout["authority"], gid, _lock_held=True)


def _publish_projection(files, layout, authority_parent, store, intent, intent_ref, operation,
                        previous, enrollments, state, moment, expiry):
    old_head, old_authority, old_raw = previous if previous is not None else (None, None, None)
    version = old_head["version"] + 1 if old_head else 1
    _require(version <= 1000000 and len(enrollments) <= 100, "needed_cache_authority_full")
    epoch = old_head["authority_epoch_id"] if old_head else secrets.token_hex(16)
    authority, payload = _encode(files, dict(schema_version="control_plane_needed_cache_authority.v1",
        authority_epoch_id=epoch, version=version,
        previous_record_sha256=old_head["record_sha256"] if old_head else None,
        state=state, issued_at_epoch=moment, expires_at_epoch=expiry,
        policy_raw_sha256=intent["policy_sha256"], policy_raw_size_bytes=intent["policy_size_bytes"],
        enrollments=sorted(enrollments, key=lambda r: r["intent_id"])), "authority_digest")
    named = f"v-{version:012d}.json"
    selector = _publish_metadata(files, authority_parent, named, payload, mode=0o640, gid=_blueprint_identity()[1])
    head, head_bytes = _encode(files, dict(schema_version="control_plane_needed_cache_authority_head.v1",
        authority_epoch_id=epoch, version=version, record_name=named,
        record_sha256=selector["sha256"], record_size_bytes=selector["size_bytes"]), "head_digest", 4096)
    pending = _record_event(files, store, intent, intent_ref, operation, "authority_update_pending", None, moment,
        next_version_raw_sha256=selector["sha256"], next_version_raw_size_bytes=selector["size_bytes"],
        prior_head_sha256=_raw(old_raw)["sha256"] if old_raw else None,
        next_head_sha256=_raw(head_bytes)["sha256"], next_head_size_bytes=len(head_bytes))
    installed_head = _head_cas(files, authority_parent, head_bytes, old_raw)
    _record_event(files, store, intent, intent_ref, operation, "authority_update_complete", pending, moment,
        head_raw_sha256=installed_head["sha256"], head_raw_size_bytes=installed_head["size_bytes"])
    return head, authority


def _transfer_target(files, root_fd, extra):
    payload, selected = _PayloadFiles(), set(extra)
    current = root_fd
    for _ in range(64):
        selected.add(current)
        parent, _, _ = files.bindings[current]
        if parent is None:
            break
        current = parent
    else:
        raise NeededCheckpointCacheError("needed_cache_resource_exhausted")
    for fd in selected:
        files.location(fd)
        expected = files.proof(fd)
        payload.owned[fd] = expected
        payload.bindings[fd] = files.bindings[fd]
    for fd in selected:
        files.owned.pop(fd)
    return payload


def _create_cache(files, layout, intent, intent_ref, inventory, rows, reservation, moment, monotonic):
    _, gid = _blueprint_identity()
    authority_parent = _authority_parent(files, layout["config"], fcntl.LOCK_EX)
    store = _private_store(files, layout["config"])
    previous = _existing_projection(files, layout, gid)
    if previous is not None:
        _require(previous[1]["state"] == "enabled" and all(r["intent_id"] != intent["intent_id"]
                 and r["target"]["name"] != intent["name"] for r in previous[1]["enrollments"]),
                 "needed_cache_name_consumed")
    total, count = _store_capacity(files, store)
    _require(count + 24 < 256 and total + 24*32768 <= 64*1024*1024, "needed_cache_store_full")
    root_parent, _ = files.parent(layout["root"] / "g1-checkpoint")
    _lock(files, root_parent, ".lane-scratch.lock", 0o600, fcntl.LOCK_EX)
    lane = files.open("g1-checkpoint", os.O_RDONLY | os.O_DIRECTORY, parent=root_parent)
    files.location(lane)
    try:
        os.stat(intent["name"], dir_fd=lane, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise NeededCheckpointCacheError("needed_cache_target_exists")
    operation, process = secrets.token_hex(16), _process_identity()
    origin = monotonic()
    _require(type(origin) in (int, float) and math.isfinite(origin), "needed_cache_clock_invalid")
    deadline = origin + 4*3600
    claim = _record_event(files, store, intent, intent_ref, operation, "claim", None, moment,
        target_was_absent=True, lane_identity=dict(dev=os.fstat(lane).st_dev, ino=os.fstat(lane).st_ino, type="directory"),
        intent_deadline=intent["expires_at_epoch"], inventory_raw_sha256=intent["inventory_raw_sha256"],
        inventory_raw_size_bytes=intent["inventory_raw_size_bytes"], size_budget_bytes=intent["size_budget_bytes"])
    stage_name = ".needed-"+operation
    create = _record_event(files, store, intent, intent_ref, operation, "operation", claim, moment,
        operation_kind="create", process_identity=process, stage_name=stage_name,
        stage_parent_identity=dict(dev=os.fstat(lane).st_dev, ino=os.fstat(lane).st_ino, type="directory"),
        prior_birth_raw_sha256=None, prior_birth_raw_size_bytes=None, reservation_id=reservation.token,
        planned_miss_bytes=reservation.expected_bytes, metadata_allowance_bytes=_QUANTUM,
        transfer_deadline_epoch=deadline, verification_raw_sha256=None, verification_raw_size_bytes=None,
        verified_present_files_digest=None, verification_phase_id=None, long_controller_deadline_epoch=None)
    files.location(lane)
    os.mkdir(stage_name, 0o700, dir_fd=lane)
    files.location(lane)
    stage = files.open(stage_name, os.O_RDONLY | os.O_DIRECTORY, parent=lane)
    stage_info = files.acquired[stage]
    _require(stage_info.st_uid == stage_info.st_gid == 0 and stat.S_IMODE(stage_info.st_mode) == 0o700,
             "needed_cache_stage_unsafe")
    fcntl.flock(stage, fcntl.LOCK_SH | fcntl.LOCK_NB)
    lease = scratch._creation_lease("g1-checkpoint", intent["name"], owner=intent["owner"],
        reason="g1_checkpoint_retry_cache", class_intent="cache", cleanup="owner_review",
        ttl_seconds=intent["lease_ttl_seconds"], size_budget_bytes=intent["size_budget_bytes"],
        **{intent["reference_kind"]: intent["reference_value"]}, now=lambda: intent["issued_at_epoch"],
        consumer_lifetime_contract=scratch.CONSUMER_LIFETIME_PROTOCOL)
    lease_bytes = owners._encoded(lease, files.budget, cap=8192)
    lease_ref = _publish_metadata(files, stage, scratch.LEASE_FILE, lease_bytes, mode=0o640, gid=gid, cap=8192)
    target = dict(root="work", lane="g1-checkpoint", name=intent["name"])
    marker, marker_bytes = _encode(files, dict(schema_version="control_plane_needed_cache_marker.v1",
        intent_id=intent["intent_id"], generation=intent["generation"], target=target, create_operation_id=operation,
        intent_raw_sha256=intent_ref["sha256"], intent_raw_size_bytes=intent_ref["size_bytes"],
        claim_raw_sha256=claim["sha256"], claim_raw_size_bytes=claim["size_bytes"],
        create_operation_raw_sha256=create["sha256"], create_operation_raw_size_bytes=create["size_bytes"],
        inventory_raw_sha256=intent["inventory_raw_sha256"], inventory_raw_size_bytes=intent["inventory_raw_size_bytes"],
        writer_scope=intent["writer_scope"]), "marker_digest", 4096)
    marker_ref = _publish_metadata(files, stage, MARKER, marker_bytes, mode=0o640, gid=gid, cap=4096)
    locks = []
    for name in (".needed-cache-lifetime.lock", ".needed-cache-writer.lock"):
        _publish_metadata(files, stage, name, b"", mode=0o640, gid=gid, cap=1)
        locks.append(files.open(name, os.O_RDONLY, parent=stage))
    fcntl.flock(locks[0], fcntl.LOCK_SH | fcntl.LOCK_NB)
    files.location(stage)
    if gid:
        os.fchown(stage, 0, gid)
        named = os.stat(stage_name, dir_fd=lane, follow_symlinks=False)
        _require(files.proof(stage) == (named.st_dev, named.st_ino, stat.S_IFMT(named.st_mode)), "needed_cache_stage_changed")
        files.bindings[stage] = (lane, stage_name, owners._security(named))
    files.location(stage)
    os.fchmod(stage, 0o750)
    named = os.stat(stage_name, dir_fd=lane, follow_symlinks=False)
    _require(files.proof(stage) == (named.st_dev, named.st_ino, stat.S_IFMT(named.st_mode)), "needed_cache_stage_changed")
    files.bindings[stage] = (lane, stage_name, owners._security(named))
    files.location(stage)
    os.fsync(stage)
    identity = dict(dev=stage_info.st_dev, ino=stage_info.st_ino, type="directory")
    prepared = _record_event(files, store, intent, intent_ref, operation, "stage", create, moment,
        operation_raw_sha256=create["sha256"], operation_raw_size_bytes=create["size_bytes"], stage_identity=identity,
        lease_raw_sha256=lease_ref["sha256"], lease_raw_size_bytes=lease_ref["size_bytes"],
        marker_raw_sha256=marker_ref["sha256"], marker_raw_size_bytes=marker_ref["size_bytes"],
        lifetime_lock_identity=dict(dev=os.fstat(locks[0]).st_dev, ino=os.fstat(locks[0]).st_ino, type="regular"),
        fill_lock_identity=dict(dev=os.fstat(locks[1]).st_dev, ino=os.fstat(locks[1]).st_ino, type="regular"), stage_ready=True)
    files.location(stage)
    files.location(lane)
    scratch._publish_no_replace(lane, stage_name, intent["name"])
    named = os.stat(intent["name"], dir_fd=lane, follow_symlinks=False)
    _require((named.st_dev, named.st_ino) == (identity["dev"], identity["ino"]), "needed_cache_publication_changed")
    files.bindings[stage] = (lane, intent["name"], owners._security(named))
    files.location(stage)
    files.location(lane)
    os.fsync(lane)
    published = _record_event(files, store, intent, intent_ref, operation, "publication", prepared, moment,
        stage_raw_sha256=prepared["sha256"], stage_raw_size_bytes=prepared["size_bytes"], stage_identity=identity,
        target_identity=identity, no_replace_result="published",
        lane_identity=dict(dev=os.fstat(lane).st_dev, ino=os.fstat(lane).st_ino, type="directory"),
        directory_publication_fsync=True)
    birth, birth_bytes = _encode(files, dict(schema_version="control_plane_needed_cache_birth.v1",
        intent_id=intent["intent_id"], generation=intent["generation"], target=target,
        claim_raw_sha256=claim["sha256"], claim_raw_size_bytes=claim["size_bytes"],
        publication_raw_sha256=published["sha256"], publication_raw_size_bytes=published["size_bytes"],
        target_identity=identity, lease_raw_sha256=lease_ref["sha256"], lease_raw_size_bytes=lease_ref["size_bytes"],
        marker_raw_sha256=marker_ref["sha256"], marker_raw_size_bytes=marker_ref["size_bytes"], owner=intent["owner"],
        reference_kind=intent["reference_kind"], reference_value=intent["reference_value"], class_intent="cache",
        cleanup="owner_review", size_budget_bytes=intent["size_budget_bytes"], expires_at_epoch=intent["expires_at_epoch"],
        inventory_raw_sha256=intent["inventory_raw_sha256"], inventory_raw_size_bytes=intent["inventory_raw_size_bytes"],
        writer_scope=intent["writer_scope"]), "birth_digest")
    public, _ = files.parent(layout["public"] / "record.json")
    public_info = os.fstat(public)
    _require(public_info.st_uid == public_info.st_gid == 0 and stat.S_IMODE(public_info.st_mode) == 0o755,
             "needed_cache_registration_unsafe")
    birth_ref = _publish_metadata(files, public, intent["intent_id"]+".birth.json", birth_bytes, mode=0o644)
    correspondence = _record_event(files, store, intent, intent_ref, operation, "registration_correspondence", published, moment,
        claim_raw_sha256=claim["sha256"], claim_raw_size_bytes=claim["size_bytes"],
        publication_raw_sha256=published["sha256"], publication_raw_size_bytes=published["size_bytes"],
        public_birth_raw_sha256=birth_ref["sha256"], public_birth_raw_size_bytes=birth_ref["size_bytes"],
        marker_raw_sha256=marker_ref["sha256"], marker_raw_size_bytes=marker_ref["size_bytes"], target_identity=identity,
        lease_raw_sha256=lease_ref["sha256"], lease_raw_size_bytes=lease_ref["size_bytes"],
        policy_raw_sha256=intent["policy_sha256"], policy_raw_size_bytes=intent["policy_size_bytes"],
        principal=intent["principal"], owner=intent["owner"])
    entry = dict(intent_id=intent["intent_id"], generation=intent["generation"], target=target,
        birth_raw_sha256=birth_ref["sha256"], birth_raw_size_bytes=birth_ref["size_bytes"],
        lease_raw_sha256=lease_ref["sha256"], lease_raw_size_bytes=lease_ref["size_bytes"], owner=intent["owner"],
        inventory_raw_sha256=intent["inventory_raw_sha256"], inventory_raw_size_bytes=intent["inventory_raw_size_bytes"],
        size_budget_bytes=intent["size_budget_bytes"], admission_state="fill_only", expires_at_epoch=intent["expires_at_epoch"])
    entries = (previous[1]["enrollments"] if previous else [])+[entry]
    head, authority = _publish_projection(files, layout, authority_parent, store, intent, intent_ref, operation,
        previous, entries, "enabled", moment, min(moment+3600, intent["expires_at_epoch"]))
    _record_event(files, store, intent, intent_ref, operation, "operation_terminal", correspondence, moment,
        operation_raw_sha256=create["sha256"], operation_raw_size_bytes=create["size_bytes"], outcome="completed",
        fixed_code="birth_completed", process_identity=process, threads_joined=True,
        owned_fd_closed=False, unresolved_fd_count=0, reservation_release_observed=False)
    payload = _transfer_target(files, stage, [locks[0]])
    use = object.__new__(NeededCheckpointCacheUse)
    use._initialized, use._closed = _USE_TOKEN, False
    use._files, use._root_fd, use._root = payload, stage, layout["root"] / "g1-checkpoint" / intent["name"]
    use._layout, use._entry, use._birth, use._rows, use._inventory, use._gid, use._mode = layout, entry, birth, rows, inventory, gid, "fill"
    use._sources = []
    use._writer, use._now, use._monotonic = None, None, monotonic
    use._origin = use._last = origin
    use._deadline, use._failure = deadline, None
    use._lock, use._counts, use._windows = threading.RLock(), {}, {}
    use._checks, use._health, use._reservation = 0, None, reservation
    use._native_pending, use._native_unknown = 0, 0
    use._resources = derive_checkpoint_flow_resources(inventory, "fill")
    use._authority_epoch, use._authority_version = head["authority_epoch_id"], head["version"]
    return use, operation, create


def _use_download(self, row):
    from .control_plane_registered_checkpoint_io import download_file
    return download_file(self, row)


def _finish_fill(use, intent, intent_ref, operation, fill_ref, moment):
    use.check()
    rows = []
    for row in use._rows:
        info = os.stat(use._root / row["relative_path"], follow_symlinks=False)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size == row["size_bytes"],
                 "needed_cache_final_payload_changed")
        rows.append(dict(relative_path=row["relative_path"], pinned_sha256=row["sha256"], pinned_size_bytes=row["size_bytes"],
            observed_identity=dict(dev=info.st_dev, ino=info.st_ino, type="regular", uid=info.st_uid, gid=info.st_gid,
                                   mode=stat.S_IMODE(info.st_mode), size_bytes=info.st_size, mtime_ns=info.st_mtime_ns)))
    allocated = sum(os.stat(use._root / r["relative_path"], follow_symlinks=False).st_blocks*512 for r in use._rows)
    _require(allocated + _QUANTUM <= intent["size_budget_bytes"], "needed_cache_owner_budget_exceeded")
    with _metadata(use._monotonic) as files:
        _authority_parent(files, use._layout["config"], fcntl.LOCK_SH)
        store = _private_store(files, use._layout["config"])
        prepared = _record_event(files, store, intent, intent_ref, operation, "prepared_completion", fill_ref, moment,
            operation_raw_sha256=fill_ref["sha256"], operation_raw_size_bytes=fill_ref["size_bytes"],
            birth_raw_sha256=use._entry["birth_raw_sha256"], birth_raw_size_bytes=use._entry["birth_raw_size_bytes"],
            inventory_raw_sha256=intent["inventory_raw_sha256"], inventory_raw_size_bytes=intent["inventory_raw_size_bytes"],
            files=rows, allocated_bytes=allocated, reservation_id=use._reservation.token,
            transfer_completed=True, responses_closed=True, threads_joined=True)
    use._files.close(use._writer)
    use._writer = None
    with _metadata(use._monotonic) as files:
        parent = _authority_parent(files, use._layout["config"], fcntl.LOCK_EX)
        store = _private_store(files, use._layout["config"])
        previous = _existing_projection(files, use._layout, use._gid)
        _require(previous is not None and previous[1]["state"] == "enabled"
                 and next(r for r in previous[1]["enrollments"] if r["intent_id"] == intent["intent_id"]) == use._entry,
                 "needed_cache_ready_revoked")
        _read_intent(files, use._layout, intent["intent_id"], intent_ref, use._now())
        for row in rows:
            actual = os.stat(use._root / row["relative_path"], follow_symlinks=False)
            expected = row["observed_identity"]
            _require(dict(dev=actual.st_dev, ino=actual.st_ino, type="regular", uid=actual.st_uid, gid=actual.st_gid,
                          mode=stat.S_IMODE(actual.st_mode), size_bytes=actual.st_size, mtime_ns=actual.st_mtime_ns) == expected,
                     "needed_cache_ready_payload_changed")
        ready = _record_event(files, store, intent, intent_ref, operation, "ready", prepared, moment,
            prepared_completion_raw_sha256=prepared["sha256"], prepared_completion_raw_size_bytes=prepared["size_bytes"],
            birth_raw_sha256=use._entry["birth_raw_sha256"], birth_raw_size_bytes=use._entry["birth_raw_size_bytes"],
            inventory_raw_sha256=intent["inventory_raw_sha256"], inventory_raw_size_bytes=intent["inventory_raw_size_bytes"],
            files=rows, allocated_bytes=allocated, reservation_id=use._reservation.token, transfer_completed=True)
        changed = use._entry | dict(admission_state="ready")
        entries = [changed if r["intent_id"] == intent["intent_id"] else r for r in previous[1]["enrollments"]]
        _publish_projection(files, use._layout, parent, store, intent, intent_ref, operation, previous,
                            entries, "enabled", moment, previous[1]["expires_at_epoch"])
        return ready


def fill_needed_checkpoint_cache(intent_id, *, expected_sha256, expected_size_bytes,
        installed_config_path=_DEFAULT_CONFIG, now=time.time, monotonic=time.monotonic):
    _require(os.geteuid() == 0, "needed_cache_root_required")
    reservation, use, status, fill_ref = None, None, "failed", None
    expected = dict(sha256=expected_sha256, size_bytes=expected_size_bytes)
    try:
        with _metadata(monotonic) as files:
            layout = _read_layout(files, installed_config_path)
            _require(layout["config"].needed_checkpoint_cache_creation_enabled is True, "needed_cache_creation_disabled")
            intent, inventory, rows, _ = _read_intent(files, layout, intent_id, expected, now())
            unit = os.statvfs(layout["root"]).f_frsize
            _require(type(unit) is int and 0 < unit <= 4096, "needed_cache_allocation_unit_unsupported")
            need = sum(((r["size_bytes"]+unit-1)//unit)*unit for r in rows) + _QUANTUM
            _require(need <= intent["size_budget_bytes"] and need <= _MAX_BYTES, "needed_cache_owner_budget_exceeded")
            target = layout["root"] / "g1-checkpoint" / intent["name"]
            try:
                os.stat(layout["authority"] / "HEAD.json", follow_symlinks=False)
            except FileNotFoundError:
                authority = None
            else:
                head, authority, _ = _current_projection(files, layout["authority"], _blueprint_identity()[1])
            entry = next((r for r in authority["enrollments"] if r["intent_id"] == intent_id), None) if authority else None
            if entry is not None and entry["admission_state"] == "ready":
                files.finish()
                with NeededCheckpointCacheUse.open_registered(target, installed_config_path=installed_config_path,
                                                              now=now, monotonic=monotonic) as hit:
                    hit._resources = derive_checkpoint_flow_resources(inventory, "hit")
                    for row in hit._rows:
                        hit.hash_file(target / row["relative_path"], role="preverify")
                    return dict(status="cache_ready", path=str(target), reservation_required=False,
                                target_ready_observed=True, generation=entry["generation"],
                                resource_counters=hit.resource_counters)
            files.finish()
            files.parents.clear()
            files.records.clear()
            files.budget.tick()
            reservation = reserve_control_plane_disk("g1_checkpoint_cache", target_root=layout["root"],
                expected_bytes=need, minimum_bytes=need, workspace=target, fresh=True, evictor=None)
            use, operation, create_ref = _create_cache(files, layout, intent, expected, inventory, rows,
                                                       reservation, now(), monotonic)
            sources = _BirthFiles(files.budget)
            try:
                _public_read(sources, layout["public"] / (intent_id+".birth.json"), 32768, 0o644, 0)
                birth_record = sources.records[-1]
                _, inventory_record = sources.read(layout["inventory"], cap=65536)
                use._sources = _retain_source_records(sources, use._files, (birth_record, inventory_record))
            finally:
                sources.finish()
        use._now = now
        with keep_reservation_live(reservation) as health:
            use._health = health
            use.check()
            use._writer = use._files.open(use._root_fd, ".needed-cache-writer.lock", os.O_RDWR)
            fcntl.flock(use._writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            use.check()
            with _metadata(monotonic) as files:
                _authority_parent(files, layout["config"], fcntl.LOCK_SH)
                store = _private_store(files, layout["config"])
                fill_ref = _record_event(files, store, intent, expected, operation, "operation", create_ref, now(),
                    operation_kind="fill", process_identity=_process_identity(), stage_name=None, stage_parent_identity=None,
                    prior_birth_raw_sha256=use._entry["birth_raw_sha256"], prior_birth_raw_size_bytes=use._entry["birth_raw_size_bytes"],
                    reservation_id=reservation.token, planned_miss_bytes=need, metadata_allowance_bytes=_QUANTUM,
                    transfer_deadline_epoch=use._deadline, verification_raw_sha256=create_ref["sha256"],
                    verification_raw_size_bytes=create_ref["size_bytes"], verified_present_files_digest=canonical_digest({"files": []}),
                    verification_phase_id=operation, long_controller_deadline_epoch=use._deadline)
            from .native_g1_checkpoint_cache import _fetcher
            fetcher = _fetcher()
            from .native_g1_development_pair import PAIR_ORDER
            for candidate in PAIR_ORDER:
                fetcher.materialize_candidate(inventory_path=layout["inventory"], candidate_id=candidate,
                                             output_dir=use._root, _cache_use=use)
            ready = _finish_fill(use, intent, expected, operation, fill_ref, now())
            status = "completed"
            result = dict(status="cache_ready", path=str(use._root), target_ready_observed=True,
                          generation=intent["generation"], ready=ready, resource_counters=use.resource_counters)
        return result
    finally:
        try:
            if use is not None:
                use.close()
        finally:
            if reservation is not None:
                reservation.release(outcome=status)
            if status == "failed" and fill_ref is not None and use is not None and use._closed:
                _failed_fill_terminal(use, intent, expected, operation, fill_ref, reservation, now(), monotonic)


def update_needed_checkpoint_cache_authority(*, operation, intent_id=None,
        installed_config_path=_DEFAULT_CONFIG, now=time.time, monotonic=time.monotonic):
    _require(os.geteuid() == 0 and operation in ("refresh", "revoke", "disable"), "needed_cache_update_invalid")
    with _metadata(monotonic) as files:
        layout = _read_layout(files, installed_config_path)
        parent = _authority_parent(files, layout["config"], fcntl.LOCK_EX)
        store = _private_store(files, layout["config"])
        previous = _existing_projection(files, layout, _blueprint_identity()[1])
        _require(previous is not None, "needed_cache_head_missing")
        entries = previous[1]["enrollments"]
        selected = next((r for r in entries if r["intent_id"] == intent_id), entries[0] if operation == "disable" and entries else None)
        _require(selected is not None, "needed_cache_not_registered")
        raw, _ = files.read(layout["private"] / (selected["intent_id"]+".json"), cap=32768, protected=True, mode=0o600)
        intent = retained._document(raw, 32768, _work_budget=files.budget)
        _require(set(intent) == _INTENT_FIELDS and intent["intent_digest"] == canonical_digest(intent, digest_field="intent_digest"),
                 "needed_cache_intent_invalid")
        # Revoke/disable remain available after expiry, policy changes or creation switch OFF.
        if operation == "refresh":
            _read_intent(files, layout, selected["intent_id"], _raw(raw), now())
            _require(selected["admission_state"] != "revoked", "needed_cache_revoke_permanent")
        next_entries = [r | {"admission_state": "revoked"} if operation == "revoke" and r["intent_id"] == intent_id else r
                        for r in entries]
        head, _ = _publish_projection(files, layout, parent, store, intent, _raw(raw), secrets.token_hex(16), previous,
            next_entries, "disabled" if operation == "disable" else previous[1]["state"], now(),
            max(now()+1, previous[1]["expires_at_epoch"]))
        return dict(status="current_authority_updated", authority_epoch_id=head["authority_epoch_id"], version=head["version"])

def renew_needed_checkpoint_cache(intent_id, *, principal, owner, lease_ttl_seconds,
        size_budget_bytes, installed_config_path=_DEFAULT_CONFIG, now=time.time, monotonic=time.monotonic):
    """Explicit root owner renewal; expiry never deletes, transfers or renews bytes itself."""
    _require(os.geteuid() == 0 and owners._matches(intent_id, owners._CONSENT_ID)
             and owners._matches(principal, owners._PRINCIPAL) and owners._matches(owner, owners._OWNER)
             and type(lease_ttl_seconds) is int and 0 < lease_ttl_seconds <= 1209600
             and type(size_budget_bytes) is int and 0 < size_budget_bytes <= _MAX_BYTES,
             'needed_cache_renewal_invalid')
    with _metadata(monotonic) as files:
        layout = _read_layout(files, installed_config_path)
        moment = now()
        _require(owners._number(moment), 'needed_cache_renewal_invalid')
        raw, acquired = files.read(layout['private'] / (intent_id+'.json'), cap=32768,
                                   protected=True, mode=0o600)
        intent = retained._document(raw, 32768, _work_budget=files.budget)
        _require(set(intent) == _INTENT_FIELDS and intent['schema_version'] == SCHEMA
                 and intent['intent_id'] == intent_id and intent['issuer_uid'] == 0
                 and intent['issuer_kind'] == 'local_root'
                 and intent['intent_digest'] == canonical_digest(intent, digest_field='intent_digest')
                 and intent['owner'] == owner and intent['class_intent'] == 'cache'
                 and intent['cleanup'] == 'owner_review' and intent['size_budget_bytes'] == size_budget_bytes,
                 'needed_cache_renewal_invalid')
        policy_raw, policy_record = files.read(layout['config'].lane_owner_policy_file, cap=65536,
                                               protected=True, mode=0o600)
        policy = owners._policy(policy_raw, principal, files.budget)
        expiry = moment+lease_ttl_seconds
        owners._authorize(dict(action='register', owner=owner, ttl_seconds=lease_ttl_seconds),
                          policy, expiry, moment)
        target = layout['root'] / 'g1-checkpoint' / intent['name']
        target_parent, _ = files.parent(target / scratch.LEASE_FILE)
        files.location(target_parent)
        _require(os.fstat(target_parent).st_uid == 0 and os.fstat(target_parent).st_gid == _blueprint_identity()[1]
                 and stat.S_IMODE(os.fstat(target_parent).st_mode) == 0o750, 'needed_cache_target_changed')
        try:
            fcntl.flock(target_parent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise NeededCheckpointCacheError('needed_cache_busy') from None
        authority_parent = _authority_parent(files, layout['config'], fcntl.LOCK_EX)
        store = _private_store(files, layout['config'])
        previous = _existing_projection(files, layout, _blueprint_identity()[1])
        _require(previous is not None and previous[1]['state'] == 'enabled', 'needed_cache_renewal_refused')
        selected = next((r for r in previous[1]['enrollments'] if r['intent_id'] == intent_id), None)
        _require(selected is not None and selected['admission_state'] == 'ready'
                 and selected['generation'] == intent['generation'] and selected['owner'] == owner
                 and selected['size_budget_bytes'] == size_budget_bytes, 'needed_cache_renewal_refused')
        birth, birth_raw = _public_read(files, layout['public'] / (intent_id+'.birth.json'), 32768, 0o644, 0)
        owners._identity(birth_raw, selected['birth_raw_sha256'], selected['birth_raw_size_bytes'], files.budget)
        info = os.fstat(target_parent)
        _require(birth['target_identity'] == dict(dev=info.st_dev, ino=info.st_ino, type='directory')
                 and birth['generation'] == selected['generation']
                 and birth['birth_digest'] == canonical_digest(birth, digest_field='birth_digest'),
                 'needed_cache_target_changed')
        lease, lease_raw = _public_read(files, target / scratch.LEASE_FILE, 8192, 0o640, _blueprint_identity()[1])
        owners._identity(lease_raw, selected['lease_raw_sha256'], selected['lease_raw_size_bytes'], files.budget)
        _require(scratch._lease_fields_valid(lease) and lease['lease_digest'] ==
                 canonical_digest(lease, digest_field='lease_digest') and lease['released_at_epoch'] is None
                 and lease['class_intent'] == 'cache' and lease['cleanup'] == 'owner_review'
                 and lease['lane'] == 'g1-checkpoint' and lease['name'] == intent['name']
                 and lease['owner'] == owner and lease['size_budget_bytes'] == size_budget_bytes,
                 'needed_cache_renewal_refused')
        inventory_raw, inventory_record = files.read(layout['inventory'], cap=65536, protected=True)
        owners._identity(inventory_raw, selected['inventory_raw_sha256'], selected['inventory_raw_size_bytes'], files.budget)
        inventory = retained._document(inventory_raw, 65536, _work_budget=files.budget)
        rows = _inventory_rows(inventory)
        _require(_renew_allocation(files, target_parent, rows) + _QUANTUM <= size_budget_bytes,
                 'needed_cache_owner_budget_exceeded')
        _, count = _store_capacity(files, store)
        _require(count+5 <= 256, 'needed_cache_store_full')
        files.verify_record(acquired)
        files.verify_record(policy_record)
        files.verify_record(inventory_record)
        operation = secrets.token_hex(16)
        updated, payload = _encode(files, lease | dict(renewed_at_epoch=moment, expires_at_epoch=expiry),
                                   'lease_digest', 8192)
        _require(scratch._lease_fields_valid(updated), 'needed_cache_renewal_invalid')
        pending = _record_event(files, store, intent, _raw(raw), operation, 'renewal_pending', None, moment,
            principal=principal, owner=owner, policy_raw_sha256=_raw(policy_raw)['sha256'],
            policy_raw_size_bytes=len(policy_raw), prior_lease_raw_sha256=selected['lease_raw_sha256'],
            prior_lease_raw_size_bytes=selected['lease_raw_size_bytes'], next_lease_raw_sha256=_raw(payload)['sha256'],
            next_lease_raw_size_bytes=len(payload), expires_at_epoch=expiry, size_budget_bytes=size_budget_bytes,
            target_identity=birth['target_identity'])
        lease_ref = _mutable_record_cas(files, target_parent, payload, lease_raw, name=scratch.LEASE_FILE, cap=8192)
        changed = selected | dict(lease_raw_sha256=lease_ref['sha256'], lease_raw_size_bytes=lease_ref['size_bytes'],
                                 expires_at_epoch=expiry)
        next_entries = [changed if r['intent_id'] == intent_id else r for r in previous[1]['enrollments']]
        projection_intent = intent | dict(policy_sha256=_raw(policy_raw)['sha256'], policy_size_bytes=len(policy_raw))
        head, _ = _publish_projection(files, layout, authority_parent, store, projection_intent, _raw(raw), operation,
                                     previous, next_entries, 'enabled', moment, expiry)
        _record_event(files, store, intent, _raw(raw), operation, 'renewal_complete', pending, moment,
            lease_raw_sha256=lease_ref['sha256'], lease_raw_size_bytes=lease_ref['size_bytes'],
            authority_version=head['version'], expires_at_epoch=expiry, generation=selected['generation'])
        return dict(status='needed_cache_renewed', generation=selected['generation'], expires_at_epoch=expiry,
                    class_intent='cache', cleanup='owner_review', size_budget_bytes=size_budget_bytes)


def _renew_allocation(files, root, rows, *, require_complete=True):
    """Finite original-parent inventory; unlisted bytes refuse renewed budget admission."""
    expected = {r['relative_path']: r for r in rows}
    metadata = {scratch.LEASE_FILE, MARKER, '.needed-cache-lifetime.lock', '.needed-cache-writer.lock'}
    stack, total, observed, directories = [(root, '')], 0, set(), 0
    while stack:
        parent, prefix = stack.pop()
        files.location(parent)
        with os.scandir(parent) as entries:
            for item in entries:
                files.budget.charge('entries')
                files.location(parent)
                path = prefix+item.name
                info = os.stat(item.name, dir_fd=parent, follow_symlinks=False)
                _require(info.st_uid == 0 and info.st_gid == _blueprint_identity()[1]
                         and not info.st_mode & 0o022, 'needed_cache_renewal_payload_unsafe')
                if stat.S_ISDIR(info.st_mode):
                    directories += 1
                    _require(directories <= 256 and len(PurePosixPath(path).parts) <= 64
                             and any(p.startswith(path+'/') for p in expected), 'needed_cache_unknown_payload')
                    child = files.open(item.name, os.O_RDONLY | os.O_DIRECTORY, parent=parent)
                    _require(os.fstat(child).st_dev == os.fstat(root).st_dev, 'needed_cache_cross_device')
                    stack.append((child, path+'/'))
                else:
                    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                             and info.st_dev == os.fstat(root).st_dev, 'needed_cache_renewal_payload_unsafe')
                    if not prefix and item.name in metadata:
                        continue
                    _require(path in expected and info.st_size == expected[path]['size_bytes'],
                             'needed_cache_unknown_payload')
                    observed.add(path)
                    total += info.st_blocks*512
    _require(not require_complete or observed == set(expected), 'needed_cache_payload_missing')
    return total

def _failed_fill_terminal(use, intent, intent_ref, operation, fill_ref, reservation, moment, monotonic):
    """Post-cleanup observation only; cannot promote payload/current authority."""
    _require(use._closed and not use._files.owned and not use._files.unresolved
             and reservation.released and use._native_pending == use._native_unknown == 0,
             'needed_cache_cleanup_incomplete')
    with _metadata(monotonic) as files:
        target, _ = files.parent(use._root / scratch.LEASE_FILE)
        info = os.fstat(target)
        _require(use._birth['target_identity'] == dict(dev=info.st_dev, ino=info.st_ino, type='directory'),
                 'needed_cache_target_changed')
        try:
            fcntl.flock(target, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise NeededCheckpointCacheError('needed_cache_busy') from None
        _lock(files, target, '.needed-cache-writer.lock', 0o640, fcntl.LOCK_EX)
        _authority_parent(files, use._layout['config'], fcntl.LOCK_SH)
        store = _private_store(files, use._layout['config'])
        # No response/worker remains: transfer's joined executor/context escaped before close.
        _record_event(files, store, intent, intent_ref, operation, 'operation_terminal', fill_ref, moment,
            operation_raw_sha256=fill_ref['sha256'], operation_raw_size_bytes=fill_ref['size_bytes'],
            outcome='failed', fixed_code=use._failure or 'needed_cache_fill_failed', process_identity=_process_identity(),
            threads_joined=True, owned_fd_closed=True, unresolved_fd_count=0, reservation_release_observed=True)


def _prior_closed_fill(files, layout, intent, entry):
    """Authenticated finite operation history, never infer closure from missing metadata."""
    store = _private_store(files, layout['config'])
    _, count = _store_capacity(files, store)
    _require(count+12 <= 256, 'needed_cache_store_full')
    operations, terminals = [], []
    with os.scandir(store) as records:
        for item in records:
            files.budget.charge('entries')
            if item.name == _STORE_LOCK:
                continue
            raw, acquired = files.read(layout['private'] / item.name, cap=32768, protected=True, mode=0o600)
            value = retained._document(raw, 32768, _work_budget=files.budget)
            files.verify_record(acquired)
            files.close(acquired.fd)
            files.records.remove(acquired)
            if value.get('intent_id') != intent['intent_id']:
                continue
            schema = value.get('schema_version')
            if schema not in ('control_plane_needed_cache_operation.v1',
                              'control_plane_needed_cache_operation_terminal.v1'):
                continue
            _require(value.get('event_digest') == canonical_digest(value, digest_field='event_digest')
                     and value.get('generation') == entry['generation']
                     and value.get('target') == entry['target'], 'needed_cache_operation_invalid')
            if schema.endswith('_operation.v1') and value.get('operation_kind') in ('fill', 'resume'):
                _require(value.get('prior_birth_raw_sha256') == entry['birth_raw_sha256']
                         and value.get('prior_birth_raw_size_bytes') == entry['birth_raw_size_bytes']
                         and owners._number(value.get('recorded_at_epoch')), 'needed_cache_operation_invalid')
                operations.append((value, _raw(raw)))
            elif schema.endswith('_terminal.v1'):
                terminals.append(value)
    _require(bool(operations), 'needed_cache_prior_operation_unproven')
    operations.sort(key=lambda row: row[0]['recorded_at_epoch'])
    last, selector = operations[-1]
    _require(len(operations) == 1 or operations[-2][0]['recorded_at_epoch'] < last['recorded_at_epoch'],
             'needed_cache_prior_operation_ambiguous')
    matches = [t for t in terminals if t.get('operation_raw_sha256') == selector['sha256']
               and t.get('operation_raw_size_bytes') == selector['size_bytes']
               and t.get('operation_id') == last['operation_id']]
    _require(len(matches) == 1, 'needed_cache_prior_operation_unproven')
    terminal = matches[0]
    _require(terminal.get('outcome') in ('completed', 'failed') and terminal.get('threads_joined') is True
             and terminal.get('owned_fd_closed') is True and type(terminal.get('unresolved_fd_count')) is int
             and terminal['unresolved_fd_count'] == 0 and terminal.get('reservation_release_observed') is True
             and terminal.get('process_identity') == last.get('process_identity'),
             'needed_cache_prior_operation_unproven')
    return selector


def resume_needed_checkpoint_cache(intent_id, *, expected_sha256, expected_size_bytes,
        installed_config_path=_DEFAULT_CONFIG, now=time.time, monotonic=time.monotonic):
    """Same authentic birth/absolute expiry; read-only proof precedes miss reservation."""
    _require(os.geteuid() == 0, 'needed_cache_root_required')
    expected = dict(sha256=expected_sha256, size_bytes=expected_size_bytes)
    reservation, use, status, fill_ref = None, None, 'failed', None
    with _metadata(monotonic) as files:
        layout = _read_layout(files, installed_config_path)
        _require(layout['config'].needed_checkpoint_cache_creation_enabled is True, 'needed_cache_creation_disabled')
        intent, inventory, rows, _ = _read_intent(files, layout, intent_id, expected, now())
        target = layout['root'] / 'g1-checkpoint' / intent['name']
    try:
        use = NeededCheckpointCacheUse.open_registered(target, mode='fill', installed_config_path=installed_config_path,
                                                       now=now, monotonic=monotonic)
        with _metadata(monotonic) as files:
            _authority_parent(files, layout['config'], fcntl.LOCK_SH)
            prior = _prior_closed_fill(files, layout, intent, use._entry)
            target_parent, _ = files.parent(target / scratch.LEASE_FILE)
            _renew_allocation(files, target_parent, rows, require_complete=False)
        present = []
        for row in rows:
            use.check()
            try:
                os.stat(target / row['relative_path'], follow_symlinks=False)
            except FileNotFoundError:
                continue
            use.hash_file(target / row['relative_path'], role='preverify')
            present.append(row['relative_path'])
        use._resources = derive_checkpoint_flow_resources(inventory, 'resume', present_paths=present)
        unit = os.statvfs(target).f_frsize
        _require(type(unit) is int and 0 < unit <= 4096, 'needed_cache_allocation_unit_unsupported')
        need = sum(((r['size_bytes']+unit-1)//unit)*unit for r in rows if r['relative_path'] not in present)+_QUANTUM
        allocated = sum(os.stat(target / r, follow_symlinks=False).st_blocks*512 for r in present)
        _require(need+allocated <= intent['size_budget_bytes'], 'needed_cache_owner_budget_exceeded')
        use.check()
        reservation = reserve_control_plane_disk('g1_checkpoint_cache', target_root=layout['root'],
            expected_bytes=need, minimum_bytes=need, workspace=target, fresh=False, evictor=None)
        use._reservation = reservation
        operation = secrets.token_hex(16)
        with keep_reservation_live(reservation) as health:
            use._health = health
            use.check()
            with _metadata(monotonic) as files:
                _authority_parent(files, layout['config'], fcntl.LOCK_SH)
                store = _private_store(files, layout['config'])
                _read_intent(files, layout, intent_id, expected, now())
                fill_ref = _record_event(files, store, intent, expected, operation, 'operation', prior, now(),
                    operation_kind='resume', process_identity=_process_identity(), stage_name=None, stage_parent_identity=None,
                    prior_birth_raw_sha256=use._entry['birth_raw_sha256'], prior_birth_raw_size_bytes=use._entry['birth_raw_size_bytes'],
                    reservation_id=reservation.token, planned_miss_bytes=need, metadata_allowance_bytes=_QUANTUM,
                    transfer_deadline_epoch=use._deadline, verification_raw_sha256=prior['sha256'],
                    verification_raw_size_bytes=prior['size_bytes'], verified_present_files_digest=canonical_digest({'files': present}),
                    verification_phase_id=operation, long_controller_deadline_epoch=use._deadline)
            from .native_g1_checkpoint_cache import _fetcher
            from .native_g1_development_pair import PAIR_ORDER
            for candidate in PAIR_ORDER:
                _fetcher().materialize_candidate(inventory_path=layout['inventory'], candidate_id=candidate,
                                                output_dir=target, _cache_use=use)
            ready = _finish_fill(use, intent, expected, operation, fill_ref, now())
            status = 'completed'
            return dict(status='cache_ready', path=str(target), generation=intent['generation'],
                        target_ready_observed=True, ready=ready, resource_counters=use.resource_counters)
    finally:
        try:
            if use is not None:
                use.close()
        finally:
            if reservation is not None:
                reservation.release(outcome=status)
            if status == 'failed' and fill_ref is not None and use is not None and use._closed:
                _failed_fill_terminal(use, intent, expected, operation, fill_ref, reservation, now(), monotonic)
