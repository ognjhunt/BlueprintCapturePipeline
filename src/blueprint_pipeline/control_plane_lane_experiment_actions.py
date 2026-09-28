"""Root-authenticated finite experiment actions; historical expiry is not a grant."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import stat
import time
from pathlib import Path

from . import control_plane_lane_experiment_birth as birth_code
from . import control_plane_lane_experiment_retirement as issuance
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch as scratch
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_authority import _current, _read
from .control_plane_lane_experiment_publication import _BirthFiles, _publish
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _epoch, _require
from .control_plane_reference_budget import ReferenceCollectionBudget
from .decision_evidence_contracts import canonical_digest

ACTION_SCHEMA = "control_plane_lane_experiment_action_intent.v1"
MANIFEST_SCHEMA = "control_plane_lane_experiment_manifest.v1"
_ACTION_FIELDS = frozenset({"schema_version", "intent_id", "action_id", "issuer_uid", "principal", "owner",
    "generation", "birth", "target_identity", "lease", "completion", "manifest", "action",
    "issued_at_epoch", "expires_at_epoch", "policy", "action_digest"})
_METADATA = frozenset({scratch.LEASE_FILE, ".registered-experiment.v1.json"})


def _encoded(value, field, cap):
    value = value | {field: canonical_digest(value, digest_field=field)}
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    _require(len(raw) <= cap, "experiment_record_limit")
    return raw


def _lease(files, target, entry):
    raw, record = files.read(target / scratch.LEASE_FILE, cap=scratch.MAX_LEASE_BYTES)
    _require(issuance._selector(raw, files.budget) == entry["lease"], "experiment_lease_changed")
    lease = retained._document(raw, scratch.MAX_LEASE_BYTES, _work_budget=files.budget)
    _require(lease["schema_version"] == scratch.SCHEMA_VERSION and scratch._lease_fields_valid(lease)
             and lease["lease_digest"] == canonical_digest(lease, digest_field="lease_digest")
             and all(lease[key] == entry[key] for key in ("owner", "lane", "name"))
             and lease.get("consumer_lifetime_contract") == scratch.CONSUMER_LIFETIME_PROTOCOL,
             "experiment_lease_changed")
    return lease, record


def _target(files, config, entry, *, lock=True):
    root = config.lane_scratch_work_root if entry["root"] == "work" else config.lane_scratch_inputs_root
    target = Path(root) / "g1" / entry["name"]
    parent, _ = files.parent(target / scratch.LEASE_FILE)
    info = os.fstat(parent)
    _require(stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino)
             == (entry["target_identity"]["dev"], entry["target_identity"]["ino"]), "experiment_target_changed")
    files.location(parent)
    files.proof(parent)
    if lock:
        try:
            fcntl.flock(parent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise OwnerTargetVersionError("experiment_target_busy") from None
    return target, parent


def _selected(files, config, intent_id, issued, gid):
    public, _ = files.parent(Path(config.experiment_authority_root) / "HEAD.json")
    lock = files.open(".authority.lock", os.O_RDONLY | os.O_NONBLOCK, parent=public)
    info = files.acquired[lock]
    _require(info.st_uid == 0 and info.st_gid == gid and stat.S_IMODE(info.st_mode) == 0o640
             and info.st_nlink == 1 and info.st_size == 0, "experiment_authority_unsafe")
    files.location(public)
    files.proof(lock)
    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
    try:
        current = _current(files, public, gid)
        _require(current is not None and current[1]["state"] == "enabled"
                 and current[1]["issued_at_epoch"] <= issued < current[1]["expires_at_epoch"],
                 "experiment_authority_inactive")
        rows = [row for row in current[1]["enrollments"] if row["intent_id"] == intent_id]
        _require(len(rows) == 1, "experiment_not_registered")
        return public, current, rows[0]
    finally:
        files.proof(lock)
        fcntl.flock(lock, fcntl.LOCK_UN)


def _birth(files, public, entry, gid):
    raw, _ = _read(files, public, entry["intent_id"] + ".birth.json", 32768, gid)
    _require(issuance._selector(raw, files.budget) == entry["birth"], "experiment_birth_changed")
    result = retained._document(raw, 32768, _work_budget=files.budget)
    _require(result["schema_version"] == "control_plane_lane_experiment_birth.v1"
             and result["birth_digest"] == canonical_digest(result, digest_field="birth_digest")
             and all(result[key] == entry[key] for key in ("intent_id", "generation", "root", "lane", "name", "owner", "target_identity")),
             "experiment_birth_changed")
    return result


def _version(files, public, current, replacement, gid, policy, issued):
    head, authority, head_record = current
    _require(authority["policy"] == policy and head["version"] < 99999999, "experiment_policy_changed")
    version = head["version"] + 1
    value = authority | {"version": version, "previous_record": head["record"], "issued_at_epoch": issued,
        "enrollments": [replacement if row["intent_id"] == replacement["intent_id"] else row
                        for row in authority["enrollments"]]}
    value.pop("authority_digest")
    raw = _encoded(value, "authority_digest", 32768)
    name = f"authority-{version:08d}-{head['authority_epoch_id']}.json"
    selected = _publish(files, public, name, raw, kind="authority", blueprint_gid=gid)
    next_head = {"schema_version": head["schema_version"], "authority_epoch_id": head["authority_epoch_id"],
                 "version": version, "record_name": name, "record": selected}
    head_raw = _encoded(next_head, "head_digest", 4096)
    return head_raw, head_record



def _install_head(files, public, payload, gid, previous):
    aliases = [record for record in files.records if record.parent == public and record.name == "HEAD.json"
               and owners._metadata(record.info) == owners._metadata(previous.info)]
    for record in aliases:
        files.verify_record(record)
    selected = _publish(files, public, "HEAD.json", payload, kind="head", blueprint_gid=gid, _expected_head=previous)
    for record in aliases:
        if record in files.records:
            files.records.remove(record)
        files.close(record.fd)
    return selected

def _manifest(files, target, target_fd, *, hash_payload=True):
    """Five-column compact rows, sequential owned FDs, bounded native metadata."""
    rows, seen, logical, allocated = [], set(), 0, 0
    started = time.monotonic()
    def walk(parent, prefix, depth):
        nonlocal logical, allocated
        _require(depth <= 16, "experiment_manifest_depth")
        files.slot()
        with os.scandir(parent) as stream:
            names = []
            for item in stream:
                files.budget.charge("entries")
                _require(len(names) + len(rows) < 4096 and item.name not in (".", "..") and len(item.name.encode()) <= 255,
                         "experiment_manifest_limit")
                if not prefix and item.name in _METADATA:
                    continue
                names.append(item.name)
        for name in sorted(names):
            _require(time.monotonic() - started <= 4 * 3600, "experiment_action_deadline")
            relative = prefix + name
            _require(len(relative.encode()) <= 1024 and len(rows) < 4096, "experiment_manifest_limit")
            files.location(parent)
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            kind = "directory" if stat.S_ISDIR(named.st_mode) else "file" if stat.S_ISREG(named.st_mode) else None
            _require(kind is not None and (kind == "directory" or named.st_nlink == 1), "experiment_member_unsupported")
            fd = files.open(name, os.O_RDONLY | os.O_NONBLOCK | (os.O_DIRECTORY if kind == "directory" else 0), parent=parent)
            info = os.fstat(fd)
            _require(owners._metadata(named) == owners._metadata(info), "experiment_member_changed")
            digest = None
            try:
                if kind == "file":
                    hashed = hashlib.sha256()
                    total = 0
                    while hash_payload:
                        files.location(fd)
                        block = os.read(fd, 1024 * 1024)
                        if not block:
                            break
                        hashed.update(block)
                        total += len(block)
                        _require(total <= info.st_size and time.monotonic() - started <= 4 * 3600,
                                 "experiment_action_deadline")
                    _require((not hash_payload or total == info.st_size) and owners._metadata(os.fstat(fd)) == owners._metadata(info),
                             "experiment_member_changed")
                    digest = "sha256:" + hashed.hexdigest() if hash_payload else None
                    logical += info.st_size
                    _require(logical <= 128 * 1024**3, "experiment_payload_limit")
                identity = (info.st_dev, info.st_ino)
                if identity not in seen:
                    allocated += info.st_blocks * 512
                    seen.add(identity)
                rows.append([relative, kind, f"{info.st_dev}:{info.st_ino}:{kind}",
                    ":".join(str(getattr(info, key)) for key in ("st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")), digest])
                if kind == "directory":
                    walk(fd, relative + "/", depth + 1)
                _require(owners._metadata(os.stat(name, dir_fd=parent, follow_symlinks=False)) == owners._metadata(info)
                         == owners._metadata(os.fstat(fd)), "experiment_member_changed")
            finally:
                files.close(fd)
    walk(target_fd, "", 0)
    rows.sort(key=lambda row: row[0])
    return dict(schema_version=MANIFEST_SCHEMA, rows=rows, logical_bytes=logical, allocated_bytes=allocated)


def _context(files, config_path, issued):
    _require(os.geteuid() == 0 and _epoch(issued), "experiment_issuer_required")
    config = issuance._configuration(files, config_path)
    _, gid = birth_code._blueprint_identity()
    return config, gid


def issue_action(intent_id, *, principal, owner, action, expires_at_epoch, installed_config_path, now):
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    manifest_files = None
    try:
        issued = now()
        config, gid = _context(files, installed_config_path, issued)
        _require(config.experiment_retirement_enabled is True, "experiment_retirement_disabled")
        _require(owners._matches(intent_id, owners._CONSENT_ID) and action in ("delete", "offload", "owner_review")
                 and _epoch(expires_at_epoch) and issued < expires_at_epoch, "experiment_action_invalid")
        public, current, entry = _selected(files, config, intent_id, issued, gid)
        target, target_fd = _target(files, config, entry)
        public = birth_code._authority_lock(files, config.experiment_authority_root, gid)
        refreshed = _current(files, public, gid)
        _require(refreshed[0] == current[0] and entry["state"] == "active" and entry["owner"] == owner,
                 "experiment_action_current_changed")
        lease, _ = _lease(files, target, entry)
        _require(lease["released_at_epoch"] is None and issued >= lease["expires_at_epoch"], "experiment_not_expired")
        origin = _birth(files, public, entry, gid)
        if action == "delete":
            _require(origin["participant_profile"] == "local_root_disposable.v1" and lease["class_intent"] == "scratch"
                     and lease["cleanup"] == "delete", "experiment_delete_ineligible")
        if action == "offload":
            _require(lease["class_intent"] == "evidence" and entry["completion"] is not None,
                     "experiment_completion_required")
        policy_raw, policy_record = files.read(config.lane_owner_policy_file, cap=owners.MAX_POLICY_BYTES,
                                             protected=True, mode=0o600)
        policy = owners._policy(policy_raw, principal, files.budget)
        decision = {"owner": owner, "action": "keep" if action == "owner_review" else action,
                    "expires_at_epoch": expires_at_epoch}
        owners._authorize(decision, policy, expires_at_epoch, issued)
        policy_selector = issuance._selector(policy_raw, files.budget)
        _require(current[1]["policy"] == policy_selector, "experiment_policy_changed")
        manifest_files = _BirthFiles(ReferenceCollectionBudget(values_limit=100000))
        _, manifest_target = _target(manifest_files, config, entry, lock=False)
        _require(len(files.owned) + len(manifest_files.owned) < 104, "experiment_descriptor_limit")
        manifest = _manifest(manifest_files, target, manifest_target, hash_payload=action != "owner_review")
        manifest_files.budget.measure(manifest, cap=1048576 - 100)
        manifest_raw = _encoded(manifest, "manifest_digest", 1048576)
        manifest_files.finish()
        manifest_files.budget.close()
        manifest_files = None
        action_id = secrets.token_hex(16)
        _require(owners._matches(action_id, owners._CONSENT_ID) and action_id != intent_id,
                 "experiment_action_invalid")
        store = issuance._store(files, config.experiment_record_store)
        occupied = issuance._capacity(files, store, adding_registration=False)
        _require(occupied + len(manifest_raw) + 8 * 32768 <= issuance.MAX_EXPERIMENT_STORE_BYTES,
                 "experiment_store_full")
        manifest_selector = _publish(files, store, action_id + ".manifest.json", manifest_raw, kind="manifest")
        value = dict(schema_version=ACTION_SCHEMA, intent_id=intent_id, action_id=action_id, issuer_uid=0,
            principal=principal, owner=owner, generation=entry["generation"], birth=entry["birth"],
            target_identity=entry["target_identity"], lease=entry["lease"], completion=entry["completion"],
            manifest=manifest_selector, action=action, issued_at_epoch=issued,
            expires_at_epoch=expires_at_epoch, policy=policy_selector)
        payload = _encoded(value, "action_digest", 32768)
        files.verify_record(policy_record)
        files.verify()
        selected = _publish(files, store, action_id + ".action.json", payload, kind="private")
        prepared, old_head = _version(files, public, refreshed, entry | {"operation_id": action_id}, gid, policy_selector, issued)
        _publish(files, store, action_id + ".head-prepared.json", prepared, kind="private")
        files.verify()
        _install_head(files, public, prepared, gid, old_head)
        return {"action_id": action_id, "action_intent": selected}
    finally:
        try:
            if manifest_files is not None:
                manifest_files.finish()
                manifest_files.budget.close()
            files.finish()
        finally:
            files.budget.close()


def _outcome(action, decision, reason, *, receipt=None, logical=0, allocated=0):
    return dict(action_id=action["action_id"], intent_id=action["intent_id"], decision=decision,
                reason=reason, receipt=receipt, removed_logical_bytes=logical, removed_allocated_bytes=allocated)


def _read_action(files, config, action_id, expected, issued):
    _require(owners._matches(action_id, owners._CONSENT_ID) and isinstance(expected, dict)
             and set(expected) == {"sha256", "size_bytes"}, "experiment_action_invalid")
    raw, record = files.read(Path(config.experiment_record_store) / (action_id + ".action.json"),
                             cap=32768, protected=True, mode=0o600)
    owners._identity(raw, expected["sha256"], expected["size_bytes"], files.budget)
    action = retained._document(raw, 32768, _work_budget=files.budget)
    _require(set(action) == _ACTION_FIELDS and action["schema_version"] == ACTION_SCHEMA
             and action["action_id"] == action_id and type(action["issuer_uid"]) is int and action["issuer_uid"] == 0
             and action["action_digest"] == canonical_digest(action, digest_field="action_digest")
             and _epoch(action["issued_at_epoch"]) and _epoch(action["expires_at_epoch"])
             and action["issued_at_epoch"] <= issued < action["expires_at_epoch"], "experiment_action_invalid")
    policy_raw, policy_record = files.read(config.lane_owner_policy_file, cap=owners.MAX_POLICY_BYTES,
                                          protected=True, mode=0o600)
    _require(issuance._selector(policy_raw, files.budget) == action["policy"], "experiment_policy_changed")
    policy = owners._policy(policy_raw, action["principal"], files.budget)
    owners._authorize({"owner": action["owner"], "action": "keep" if action["action"] == "owner_review" else action["action"],
                      "expires_at_epoch": action["expires_at_epoch"]}, policy, action["expires_at_epoch"], action["issued_at_epoch"])
    files.verify_record(record)
    files.verify_record(policy_record)
    return action


def _directory(files, parent, name, *, create=False):
    files.location(parent)
    if create:
        try:
            os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            os.mkdir(name, mode=0o700, dir_fd=parent)
            files.location(parent)
            os.fsync(parent)
        else:
            raise OwnerTargetVersionError("experiment_operation_exists")
    fd = files.open(name, os.O_RDONLY | os.O_DIRECTORY, parent=parent)
    info = files.acquired[fd]
    _require(info.st_uid == info.st_gid == 0 and stat.S_IMODE(info.st_mode) == 0o700,
             "experiment_operation_unsafe")
    files.location(fd)
    return fd



def _reference_configuration(files, config, selected_root):
    from .control_plane_storage_pins import PINS_ROOT_ENV
    raw, record = files.read(config.experiment_gc_environment_file, cap=65536, protected=True)
    _require(stat.S_IMODE(record.info.st_mode) in (0o600, 0o640), "experiment_reference_configuration_unsafe")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError:
        raise OwnerTargetVersionError("experiment_reference_configuration_invalid") from None
    _require(len(lines) <= 1024, "experiment_reference_configuration_limit")
    files.budget.charge("values", len(lines))
    selections = []
    for line in lines:
        key, separator, value = line.strip().partition("=")
        if key.strip() != PINS_ROOT_ENV:
            continue
        _require(len(line.encode()) <= 8192, "experiment_reference_configuration_limit")
        _require(separator and not selections, "experiment_reference_configuration_invalid")
        value = value.strip()
        if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
            value = value[1:-1]
        _require(value and not any(character in value for character in ("\\", "$", "'", '"', "\n", "\r")),
                 "experiment_reference_configuration_invalid")
        try:
            path = retained._path(value, _work_budget=files.budget)
        except retained.CensusDecisionError:
            raise OwnerTargetVersionError("experiment_reference_configuration_invalid") from None
        selections.append(path)
    _require(len(selections) == 1 and isinstance(selected_root, (str, Path))
             and os.fspath(selected_root) == str(selections[0]), "experiment_reference_configuration_changed")
    files.verify_record(record)
    return issuance._selector(raw, files.budget)

def _pin_fence(files, config, root, target, issued):
    """Exact same publisher authority directory, never a file-name lock."""
    from .control_plane_storage_pin_observation import observe_storage_pins
    configuration = _reference_configuration(files, config, root)
    _require(isinstance(root, (str, Path)) and Path(root).is_absolute(), "experiment_reference_authority_missing")
    parent, _ = files.parent(Path(root) / ".reference-probe")
    initial = os.fstat(parent)
    files.location(parent)
    observed = observe_storage_pins(str(root), observed_at_epoch=issued, budget=files.budget)
    _require(observed.complete and observed.root_identity == (initial.st_dev, initial.st_ino),
             "experiment_pin_observation_incomplete")
    files.proof(parent)
    fcntl.flock(parent, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files.location(parent)
    _require(owners._metadata(os.fstat(parent)) == owners._metadata(initial), "experiment_pin_inventory_changed")
    for row in observed.rows:
        row_path = Path(row.row_path)
        named = os.stat(row_path, follow_symlinks=False)
        _require((named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns, named.st_ctime_ns) == row.row_identity,
                 "experiment_pin_inventory_changed")
        _require(not any(Path(path) == target or target in Path(path).parents or Path(path) in target.parents
                         for path in row.paths), "experiment_pin_reference_present")
    return dict(configuration=configuration, root=str(root), identity=dict(dev=initial.st_dev, ino=initial.st_ino, type="directory")), parent


def _event(files, directory, action, kind, body, index, previous, issued):
    _require(0 <= index < 12288, "experiment_event_limit")
    value = dict(schema_version="control_plane_lane_experiment_event.v1", event_id=secrets.token_hex(16),
        operation_id=action["action_id"], event_kind=kind, intent_id=action["intent_id"],
        generation=action["generation"], sequence=index, previous_event=previous,
        issued_at_epoch=issued, body=body)
    payload = _encoded(value, "event_digest", 4096 if kind == "member_removed" else 32768)
    return _publish(files, directory, f"e-{index:05d}.json", payload, kind="event")


def _member(files, target, row, current_directory_metadata=None):
    relative, kind, identity, token, digest = row
    path = Path(relative)
    _require(not path.is_absolute() and path.parts and all(part not in (".", "..") for part in path.parts),
             "experiment_manifest_invalid")
    parent, _ = files.parent(target / path)
    files.location(parent)
    named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
    before = tuple(int(value) for value in token.split(":"))
    metadata = tuple(getattr(named, key) for key in ("st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))
    expected = current_directory_metadata if kind == "directory" and current_directory_metadata is not None else before
    _require(len(before) == 7 and identity == f"{named.st_dev}:{named.st_ino}:{kind}"
             and (stat.S_ISDIR(named.st_mode) if kind == "directory" else stat.S_ISREG(named.st_mode))
             and metadata == expected, "experiment_member_changed")
    fd = files.open(path.name, os.O_RDONLY | os.O_NONBLOCK | (os.O_DIRECTORY if kind == "directory" else 0), parent=parent)
    initial = os.fstat(fd)
    try:
        _require(owners._metadata(initial) == owners._metadata(named), "experiment_member_changed")
        if kind == "file":
            actual = hashlib.sha256()
            amount = 0
            while True:
                files.location(fd)
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                amount += len(block)
                _require(amount <= named.st_size, "experiment_member_changed")
                actual.update(block)
            _require(amount == named.st_size and "sha256:" + actual.hexdigest() == digest,
                     "experiment_member_changed")
        files.location(parent)
        files.proof(fd)
        _require(owners._metadata(os.fstat(fd)) == owners._metadata(initial)
                 == owners._metadata(os.stat(path.name, dir_fd=parent, follow_symlinks=False)), "experiment_member_changed")
        return parent, path.name, fd, initial
    except BaseException:
        files.close(fd)
        raise


def run_action(action_id, *, expected_action_intent, installed_config_path, now, _pins_root):
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        issued = now()
        config, gid = _context(files, installed_config_path, issued)
        action = _read_action(files, config, action_id, expected_action_intent, issued)
        if config.experiment_retirement_enabled is not True:
            return _outcome(action, "kept", "experiment_retirement_disabled")
        public, current, entry = _selected(files, config, action["intent_id"], issued, gid)
        _require(all(action[key] == entry[key] for key in ("generation", "birth", "target_identity", "lease", "completion", "owner"))
                 and current[1]["policy"] == action["policy"] and entry["operation_id"] == action_id,
                 "experiment_action_current_changed")
        if action["action"] == "owner_review":
            return _outcome(action, "kept", "owner_review")
        if entry["state"] == "retired":
            manifest_raw, _ = files.read(Path(config.experiment_record_store) / (action_id + ".manifest.json"),
                                        cap=1048576, protected=True, mode=0o600)
            _require(issuance._selector(manifest_raw, files.budget) == action["manifest"], "experiment_manifest_changed")
            saved_manifest = retained._document(manifest_raw, 1048576, _work_budget=files.budget)
            index = len(saved_manifest["rows"]) + 1
            completed, _ = files.read(Path(config.experiment_record_store) / "operations" / action_id / f"e-{index:05d}.json",
                                       cap=32768, protected=True, mode=0o600)
            summary = retained._document(completed, 32768, _work_budget=files.budget)
            _require(summary["event_kind"] == "retired" and summary["body"]["partial"] is False,
                     "experiment_retired_receipt_missing")
            return _outcome(action, "retired", "already_retired", receipt=issuance._selector(completed, files.budget),
                logical=summary["body"]["removed_logical_bytes"], allocated=summary["body"]["eligible_allocated_bytes"])
        target, target_fd = _target(files, config, entry)
        lease, lease_record = _lease(files, target, entry)
        _require(issued >= lease["expires_at_epoch"] and lease["released_at_epoch"] is None, "experiment_not_expired")
        original = _birth(files, public, entry, gid)
        _require(action["action"] == "delete" and original["participant_profile"] == "local_root_disposable.v1"
                 and lease["class_intent"] == "scratch" and lease["cleanup"] == "delete", "experiment_action_profile_unsupported")
        if _pins_root is None:
            return _outcome(action, "kept", "experiment_reference_authority_missing")
        reference, reference_fd = _pin_fence(files, config, _pins_root, target, issued)
        public = birth_code._authority_lock(files, config.experiment_authority_root, gid)
        refreshed = _current(files, public, gid)
        _require(refreshed[0] == current[0] and entry["state"] in ("active", "retiring"), "experiment_action_current_changed")
        store = issuance._store(files, config.experiment_record_store)
        manifest_raw, _ = files.read(Path(config.experiment_record_store) / (action_id + ".manifest.json"),
                                    cap=1048576, protected=True, mode=0o600)
        _require(issuance._selector(manifest_raw, files.budget) == action["manifest"], "experiment_manifest_changed")
        manifest = retained._document(manifest_raw, 1048576, _work_budget=files.budget)
        _require(manifest["schema_version"] == MANIFEST_SCHEMA and manifest["manifest_digest"]
                 == canonical_digest(manifest, digest_field="manifest_digest"), "experiment_manifest_invalid")
        _require(len(manifest["rows"]) <= 4096, "experiment_manifest_limit")
        from . import control_plane_lane_experiment_recovery as recovery
        rows = sorted(manifest["rows"], key=lambda row: (len(Path(row[0]).parts), row[0]), reverse=True)
        files._store_path = config.experiment_record_store
        reservation_size = len(rows) * 2 * 4096 + 8 * 32768
        reserve_raw = _encoded(dict(schema_version="control_plane_lane_experiment_reservation.v1",
            operation_id=action_id, reserved_bytes=reservation_size), "reservation_digest", 4096)
        if entry["state"] == "active":
            occupied = issuance._capacity(files, store, adding_registration=False)
            _require(occupied + reservation_size + 32768 <= issuance.MAX_EXPERIMENT_STORE_BYTES, "experiment_store_full")
        recovery._once(files, store, action_id + ".reservation.json", reserve_raw, kind="private")
        operation, retiring, previous, logical, allocated, changed_directories, removed_count, receipt = recovery.begin(
            files, config, action, expected_action_intent, entry, current, refreshed, public, store,
            target, rows, reference, issued, gid)
        for index, row in enumerate(rows, 1):
            if index <= removed_count:
                continue
            _require(now() < action["expires_at_epoch"], "experiment_action_expired")
            files.verify_record(lease_record)
            files.verify()
            files.location(target_fd)
            files.location(reference_fd)
            files.proof(reference_fd)
            files.verify_record(retiring[2])
            parent, name, fd, info = _member(files, target, row, changed_directories.get(row[0]))
            try:
                files.location(parent)
                files.proof(fd)
                _require(owners._metadata(os.fstat(fd)) == owners._metadata(info)
                         == owners._metadata(os.stat(name, dir_fd=parent, follow_symlinks=False)), "experiment_member_changed")
                if row[1] == "directory":
                    os.rmdir(name, dir_fd=parent)
                else:
                    os.unlink(name, dir_fd=parent)
                    logical += info.st_size
                allocated += info.st_blocks * 512
                files.location(parent)
                os.fsync(parent)
                updated = os.fstat(parent)
                changed_directories[str(Path(row[0]).parent)] = tuple(getattr(updated, key) for key in
                    ("st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))
                previous = _event(files, operation, action, "member_removed", dict(preservation=None,
                    action=expected_action_intent, manifest=action["manifest"], index=index - 1,
                    path=row[0], original_identity=dict(dev=info.st_dev, ino=info.st_ino, type=row[1]),
                    logical_bytes=info.st_size if row[1] == "file" else 0, eligible_allocated_bytes=info.st_blocks * 512,
                    parent_after=dict(path=str(Path(row[0]).parent), identity=dict(dev=updated.st_dev, ino=updated.st_ino, type="directory"),
                        stat_token=":".join(str(value) for value in changed_directories[str(Path(row[0]).parent)]))),
                    index, previous, issued)
            finally:
                files.close(fd)
        files.location(target_fd)
        os.fsync(target_fd)
        if receipt is None:
            receipt = _event(files, operation, action, "retired", dict(preservation=None, manifest=action["manifest"],
            removed_event_count=len(rows), removed_logical_bytes=logical, eligible_allocated_bytes=allocated,
            remaining_metadata=[entry["lease"], original["marker"]], partial=False), len(rows) + 1, previous, issued)
        prepared, old_head = _version(files, public, retiring, entry | {"state": "retired"}, gid, action["policy"], issued)
        recovery._once(files, store, action_id + ".retired-head.json", prepared, kind="private")
        _install_head(files, public, prepared, gid, old_head)
        return _outcome(action, "retired", "disposable_expired", receipt=receipt, logical=logical, allocated=allocated)
    except OSError:
        raise OwnerTargetVersionError("experiment_action_io_failed") from None
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()


def gc_actions(*, installed_config_path, enabled, apply, pins_root, now):
    empty = dict(enabled=False, outcomes=[])
    if enabled is not True or apply is not True:
        return empty
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        issued = now()
        config, gid = _context(files, installed_config_path, issued)
        if config.experiment_retirement_enabled is not True:
            return empty
        public = birth_code._authority_lock(files, config.experiment_authority_root, gid)
        current = _current(files, public, gid)
        actions = []
        if current is not None:
            for entry in current[1]["enrollments"]:
                if entry["state"] in ("active", "retiring") and entry["operation_id"] is not None:
                    if len(actions) == 2:
                        break
                    raw, _ = files.read(Path(config.experiment_record_store) / (entry["operation_id"] + ".action.json"),
                                        cap=32768, protected=True, mode=0o600)
                    actions.append((entry["operation_id"], issuance._selector(raw, files.budget), entry["intent_id"]))
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()
    outcomes = []
    for action_id, expected, intent_id in actions:
        try:
            outcomes.append(run_action(action_id, expected_action_intent=expected,
                installed_config_path=installed_config_path, now=now, _pins_root=pins_root))
        except ValueError as error:
            outcomes.append(dict(action_id=action_id, intent_id=intent_id, decision="kept", reason=getattr(error, "code", "experiment_action_refused"),
                                 receipt=None, removed_logical_bytes=0, removed_allocated_bytes=0))
    return dict(enabled=True, outcomes=outcomes)
