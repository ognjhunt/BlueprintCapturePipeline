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

from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch as scratch
from . import control_plane_lane_scratch_decisions as retained
from . import control_plane_lane_experiment_retirement as installed
from .control_plane_lane_experiment_publication import _BirthFiles
from .control_plane_lane_owner_target_publication import _publish_owned_metadata
from .control_plane_reference_budget import ReferenceCollectionBudget
from .decision_evidence_contracts import canonical_digest

SCHEMA = "control_plane_needed_cache_creation_intent.v1"
MARKER = ".needed-cache-birth.v1.json"
_STORE_LOCK = ".cache-store.lock"
_DEFAULT_CONFIG = "/etc/blueprint-operator-door/door.json"
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
    _require(type(value) is NeededCheckpointCacheUse and getattr(value, "_initialized", None) is _USE_TOKEN,
             "needed_cache_use_invalid")
    _require(not value._closed, "needed_cache_use_closed")
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
    """Canonical private lifetime; construction is confined to authenticated acquisition."""
    def __init__(self):
        raise NeededCheckpointCacheError("needed_cache_use_invalid")


class _PayloadFiles:
    """Retain named originals; unknown first tokens and reused numbers stay foreign."""
    def __init__(self):
        self.owned, self.bindings = {}, {}
        self.unresolved = 0

    def proof(self, fd):
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


def _current_projection(files, root, gid):
    parent, _ = files.parent(Path(root) / "HEAD.json")
    info = os.fstat(parent)
    _require(info.st_uid == 0 and info.st_gid == gid and stat.S_IMODE(info.st_mode) == 0o750,
             "needed_cache_authority_unsafe")
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
    return dict(root=Path(config.lane_scratch_work_root),
                public=Path(config.needed_checkpoint_cache_registration_root),
                authority=Path(config.needed_checkpoint_cache_authority_root),
                inventory=Path(config.needed_checkpoint_cache_inventory_file),
                private=Path(config.needed_checkpoint_cache_record_store), config=config)


@classmethod
def _open_registered(cls, root, *, mode="read", installed_config_path=_DEFAULT_CONFIG,
                     now=time.time, monotonic=time.monotonic):
    _require(cls is NeededCheckpointCacheUse and mode in ("read", "fill"), "needed_cache_use_invalid")
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000, monotonic=monotonic))
    payload = _PayloadFiles()
    result = None
    try:
        # Preflight current authority completes before an existing target SH acquisition.
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
        _require(birth.get("schema_version") == "control_plane_needed_cache_birth.v1"
                 and birth.get("birth_digest") == canonical_digest(birth, digest_field="birth_digest")
                 and birth.get("target") == entry["target"] and birth.get("generation") == entry["generation"]
                 and birth.get("intent_id") == entry["intent_id"], "needed_cache_birth_invalid")
        inventory_raw, _ = files.read(layout["inventory"], cap=65536)
        owners._identity(inventory_raw, entry["inventory_raw_sha256"], entry["inventory_raw_size_bytes"], files.budget)
        inventory = retained._document(inventory_raw, 65536, _work_budget=files.budget)
        rows = _inventory_rows(inventory)
        # Release administrative SH before opening/locking an existing target.
        files.finish()
        files.budget.close()
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
        result._rows, result._inventory, result._gid, result._mode = rows, inventory, gid, mode
        result._writer, result._now, result._monotonic = writer, now, monotonic
        result._origin = result._last = monotonic()
        _require(type(result._origin) in (int, float) and math.isfinite(result._origin), "needed_cache_clock_invalid")
        result._deadline, result._failure = result._origin + 4*3600, None
        result._lock, result._counts, result._windows = threading.RLock(), {}, {}
        result._checks, result._health, result._reservation = 0, None, None
        result._resources = derive_checkpoint_flow_resources(inventory, "stage_miss" if mode == "read" else "fill")
        result.check()
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


NeededCheckpointCacheUse.open_registered = _open_registered
