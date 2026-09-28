"""Root-authenticated finite experiment actions; historical expiry is not a grant."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
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
from .control_plane_lane_experiment_publication import _BirthFiles, _publish as _native_publish
from .control_plane_lane_experiment_work import _ActionFiles
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _epoch, _require, _valid_digest
from .control_plane_reference_budget import ReferenceCollectionBudget
from .decision_evidence_contracts import canonical_digest

ACTION_SCHEMA = "control_plane_lane_experiment_action_intent.v1"
MANIFEST_SCHEMA = "control_plane_lane_experiment_manifest.v1"
_ACTION_FIELDS = frozenset({"schema_version", "intent_id", "action_id", "issuer_uid", "principal", "owner",
    "generation", "birth", "target_identity", "lease", "completion", "manifest", "action",
    "issued_at_epoch", "expires_at_epoch", "policy", "action_digest"})
_METADATA = frozenset({scratch.LEASE_FILE, ".registered-experiment.v1.json"})
_ISSUE_SELECTION_SCHEMA = 'control_plane_lane_experiment_issue_selection.v1'


def _issue_selection(files, config, entry, authority, store, *, principal, owner, action, expiry):
    """ONE protected current-authority operation, selected before any new UUID."""
    intent_raw, intent_record = files.read(Path(config.experiment_record_store) / (entry['intent_id'] + '.json'),
                                          cap=32768, protected=True, mode=0o600)
    intent = retained._document(intent_raw, 32768, _work_budget=files.budget)
    _require(set(intent) == birth_code._INTENT_FIELDS
             and intent['intent_digest'] == canonical_digest(intent, digest_field='intent_digest')
             and all(intent[key] == entry[key] for key in ('intent_id', 'generation', 'owner', 'root', 'lane', 'name')),
             'experiment_issue_selection_invalid')
    expected = dict(schema_version=_ISSUE_SELECTION_SCHEMA, intent_id=entry['intent_id'],
                    intent=issuance._selector(intent_raw, files.budget), generation=entry['generation'],
                    birth=entry['birth'], target_identity=entry['target_identity'], lease=entry['lease'],
                    current_authority=authority, policy=files._issue_policy, principal=principal,
                    owner=owner, action=action, expires_at_epoch=expiry)
    _require(_valid_digest(authority['sha256']), 'experiment_issue_selection_invalid')
    name = entry['intent_id'] + '.issue-selection-' + authority['sha256'][7:] + '.json'
    files.location(store)
    try:
        os.stat(name, dir_fd=store, follow_symlinks=False)
    except FileNotFoundError:
        operation_id = secrets.token_hex(16)
        _require(owners._matches(operation_id, owners._CONSENT_ID) and operation_id != entry['intent_id'],
                 'experiment_issue_selection_invalid')
        controller = files.controller()
        value = expected | dict(operation_id=operation_id, controller=controller)
        raw = _encoded(value, 'selection_digest', 4096)
        _publish(files, store, name, raw, kind='issue_selection')
    else:
        raw, record = files.read(Path(config.experiment_record_store) / name, cap=4096, protected=True, mode=0o600)
        value = retained._document(raw, 4096, _work_budget=files.budget)
        _require(set(value) == set(expected) | {'operation_id', 'controller', 'selection_digest'}
                 and all(value[key] == part for key, part in expected.items())
                 and owners._matches(value['operation_id'], owners._CONSENT_ID)
                 and value['operation_id'] != entry['intent_id']
                 and value['selection_digest'] == canonical_digest(value, digest_field='selection_digest'),
                 'experiment_issue_selection_invalid')
        files.verify_record(record)
    files.verify_record(intent_record)
    files.verify()
    files.bind_controller(value['controller'])
    return value['operation_id']


def _publish(files, *args, **kwargs):
    if isinstance(files, _ActionFiles):
        files.reserve_output(len(args[2]))
    before = set(files.owned)
    selected = _native_publish(files, *args, **kwargs)
    if isinstance(files, _ActionFiles):
        files.publication_complete(args[0], args[1], before)
    return selected


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

def _manifest(files, target, target_fd, *, binding, hash_payload=True):
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
                rows.append([relative, kind, f"{info.st_dev}:{info.st_ino}:" + ("r" if kind == "file" else "d"),
                    ":".join(map(str, (stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns))), digest])
                if kind == "directory":
                    walk(fd, relative + "/", depth + 1)
                _require(owners._metadata(os.stat(name, dir_fd=parent, follow_symlinks=False)) == owners._metadata(info)
                         == owners._metadata(os.fstat(fd)), "experiment_member_changed")
            finally:
                files.close(fd)
    walk(target_fd, "", 0)
    rows.sort(key=lambda row: row[0])
    return dict(schema_version=MANIFEST_SCHEMA, members=rows, logical_bytes=logical, allocated_bytes=allocated,
                **{key: binding[key] for key in ("generation", "birth", "target_identity", "lease", "completion")})


def _manifest_record(files, raw, binding):
    """Finite compact decoder; the supplied binding is independently selected."""
    value = retained._document(raw, 1048576, _work_budget=files.budget)
    fields = {'schema_version', 'generation', 'birth', 'target_identity', 'lease', 'completion',
              'members', 'logical_bytes', 'allocated_bytes', 'manifest_digest'}
    _require(type(value) is dict and set(value) == fields and value['schema_version'] == MANIFEST_SCHEMA
             and value['manifest_digest'] == canonical_digest(value, digest_field='manifest_digest')
             and all(value[key] == binding[key] for key in ('generation', 'birth', 'target_identity', 'lease', 'completion'))
             and type(value['members']) is list and len(value['members']) <= 4096
             and all(type(value[key]) is int and 0 <= value[key] <= 128 * 1024**3
                     for key in ('logical_bytes', 'allocated_bytes')), 'experiment_manifest_invalid')
    seen, directories, logical = set(), set(), 0
    for row in value['members']:
        files.budget.charge('values', 11)
        _require(type(row) is list and len(row) == 5, 'experiment_manifest_invalid')
        path, kind, identity, token, digest = row
        _require(type(path) is str and 0 < len(path.encode('utf-8')) <= 1024 and '\x00' not in path
                 and kind in ('file', 'directory') and type(identity) is str and len(identity) <= 64
                 and type(token) is str and len(token) <= 160, 'experiment_manifest_invalid')
        parts = Path(path).parts
        _require(parts and len(parts) <= 16 and not Path(path).is_absolute() and str(Path(path)) == path
                 and all(part not in ('.', '..') and len(part.encode('utf-8')) <= 255 for part in parts)
                 and parts[0] not in _METADATA and path not in seen, 'experiment_manifest_invalid')
        _require(all(str(Path(*parts[:index])) in directories for index in range(1, len(parts))),
                 'experiment_manifest_invalid')
        ids, metadata = identity.split(':'), token.split(':')
        _require(len(ids) == 3 and ids[2] == ('r' if kind == 'file' else 'd') and len(metadata) == 7
                 and all(re.fullmatch(r'(?:0|[1-9][0-9]{0,19})', item)
                         and int(item) <= (1 << 64) - 1 for item in ids[:2] + metadata),
                 'experiment_manifest_invalid')
        numbers = tuple(map(int, metadata))
        _require(int(ids[1]) > 0 and numbers[0] <= 0o7777 and numbers[1] <= (1 << 32)-1
                 and numbers[2] <= (1 << 32)-1 and 0 < numbers[3] <= (1 << 32)-1
                 and numbers[4] <= 128 * 1024**3
                 and (digest is None if kind == 'directory' else numbers[3] == 1 and _valid_digest(digest)),
                 'experiment_manifest_invalid')
        if kind == 'file':
            logical += numbers[4]
            _require(logical <= 128 * 1024**3, 'experiment_manifest_invalid')
        else:
            directories.add(path)
        seen.add(path)
    _require(logical == value['logical_bytes'], 'experiment_manifest_invalid')
    return value


def _context(files, config_path, issued):
    _require(os.geteuid() == 0 and _epoch(issued), "experiment_issuer_required")
    config = issuance._configuration(files, config_path)
    _, gid = birth_code._blueprint_identity()
    return config, gid


def _hash_manifest(files, target, target_fd, manifest, *, role):
    """One declared full payload pass, retaining one original member at a time."""
    _require(type(files) is _ActionFiles and len(manifest['members']) <= 4096,
             'experiment_work_payload_invalid')
    # Account the fixed retained digest slots before long IO closes native B.
    files.budget.charge('values', sum(row[1] == 'file' for row in manifest['members']))
    files.budget.available('output_bytes', len(manifest['members']) * 71)
    files.payload(target, target_fd, expected_payload_bytes=files.payload_size if files.payload_size is not None else manifest['logical_bytes'])
    for row in manifest['members']:
        if row[1] != 'file':
            continue
        files.verify()
        parent, name, fd, initial = _member(files, target, row, hash_payload=False)
        try:
            digest, amount = hashlib.sha256(), 0
            while True:
                files.verify()
                block = files.payload_read(fd, 1024 * 1024, role=role)
                if not block:
                    break
                amount += len(block)
                _require(amount <= initial.st_size, 'experiment_member_changed')
                digest.update(block)
            files.location(parent)
            files.proof(fd)
            _require(amount == initial.st_size and owners._metadata(os.fstat(fd)) == owners._metadata(initial)
                     == owners._metadata(os.stat(name, dir_fd=parent, follow_symlinks=False)),
                     'experiment_member_changed')
            row[4] = 'sha256:' + digest.hexdigest()
        finally:
            files.close(fd)
            files.trim_payload()


def issue_action(intent_id, *, principal, owner, action, expires_at_epoch, installed_config_path, now):
    files = _ActionFiles(now=now)
    try:
        issued = now()
        files.bind_deadline(expires_at_epoch)
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
            from .control_plane_lane_experiment_completion import selected_completion
            selected_completion(files, config, entry)
        policy_raw, policy_record = files.read(config.lane_owner_policy_file, cap=owners.MAX_POLICY_BYTES,
                                             protected=True, mode=0o600)
        policy = owners._policy(policy_raw, principal, files.budget)
        decision = {"owner": owner, "action": "keep" if action == "owner_review" else action,
                    "expires_at_epoch": expires_at_epoch}
        owners._authorize(decision, policy, expires_at_epoch, issued)
        policy_selector = issuance._selector(policy_raw, files.budget)
        _require(current[1]["policy"] == policy_selector, "experiment_policy_changed")
        store = issuance._store(files, config.experiment_record_store)
        issuance._capacity(files, store, adding_registration=False)
        files._issue_policy = policy_selector
        action_id = _issue_selection(files, config, entry, current[0]['record'], store,
            principal=principal, owner=owner, action=action, expiry=expires_at_epoch)
        files.phase('manifest')
        manifest = _manifest(files, target, target_fd, binding=entry, hash_payload=False)
        files.budget.measure(manifest, cap=1048576 - 100)
        _hash_manifest(files, target, target_fd, manifest, role='issue_hash')
        files.phase('finalize')
        manifest_raw = _encoded(manifest, "manifest_digest", 1048576)
        _require(owners._matches(action_id, owners._CONSENT_ID) and action_id != intent_id,
                 "experiment_action_invalid")
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



def _installed_reference_selection(files, config):
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
    _require(len(selections) == 1, "experiment_reference_configuration_changed")
    files.verify_record(record)
    return selections[0], issuance._selector(raw, files.budget)


def _reference_configuration(files, config, selected_root):
    path, selector = _installed_reference_selection(files, config)
    _require(isinstance(selected_root, (str, Path)) and os.fspath(selected_root) == str(path),
             "experiment_reference_configuration_changed")
    return selector

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


def _member(files, target, row, current_directory_metadata=None, *, hash_payload=True, hash_role="archive_prehash"):
    relative, kind, identity, token, digest = row
    path = Path(relative)
    _require(not path.is_absolute() and path.parts and all(part not in (".", "..") for part in path.parts),
             "experiment_manifest_invalid")
    parent, _ = files.parent(target / path)
    files.location(parent)
    named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
    before = tuple(int(value) for value in token.split(":"))
    metadata = (stat.S_IMODE(named.st_mode), named.st_uid, named.st_gid, named.st_nlink, named.st_size, named.st_mtime_ns, named.st_ctime_ns)
    expected = ((stat.S_IMODE(current_directory_metadata[0]), *current_directory_metadata[1:])
                if kind == "directory" and current_directory_metadata is not None else before)
    _require(len(before) == 7 and identity == f"{named.st_dev}:{named.st_ino}:" + ("d" if kind == "directory" else "r")
             and (stat.S_ISDIR(named.st_mode) if kind == "directory" else stat.S_ISREG(named.st_mode))
             and metadata == expected, "experiment_member_changed")
    fd = files.open(path.name, os.O_RDONLY | os.O_NONBLOCK | (os.O_DIRECTORY if kind == "directory" else 0), parent=parent)
    initial = os.fstat(fd)
    try:
        _require(owners._metadata(initial) == owners._metadata(named), "experiment_member_changed")
        if kind == "file" and hash_payload:
            actual = hashlib.sha256()
            amount = 0
            while True:
                files.location(fd)
                block = (files.payload_read(fd, 1024 * 1024, role=hash_role)
                         if isinstance(files, _ActionFiles) and files.payload_mode else os.read(fd, 1024 * 1024))
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
    files = _ActionFiles(now=now)
    try:
        issued = now()
        config, gid = _context(files, installed_config_path, issued)
        action = _read_action(files, config, action_id, expected_action_intent, issued)
        files.bind_deadline(action["expires_at_epoch"])
        if config.experiment_retirement_enabled is not True:
            return _outcome(action, "kept", "experiment_retirement_disabled")
        public, current, entry = _selected(files, config, action["intent_id"], issued, gid)
        _require(all(action[key] == entry[key] for key in ("generation", "birth", "target_identity", "lease", "completion", "owner"))
                 and current[1]["policy"] == action["policy"] and entry["operation_id"] == action_id,
                 "experiment_action_current_changed")
        if action["action"] == "owner_review":
            return _outcome(action, "kept", "owner_review")
        if entry["state"] == "retired":
            files.phase('manifest')
            manifest_raw, _ = files.read(Path(config.experiment_record_store) / (action_id + ".manifest.json"),
                                        cap=1048576, protected=True, mode=0o600)
            _require(issuance._selector(manifest_raw, files.budget) == action["manifest"], "experiment_manifest_changed")
            saved_manifest = _manifest_record(files, manifest_raw, entry)
            from . import control_plane_lane_experiment_recovery as recovery
            target, target_fd = _target(files, config, entry)
            _lease(files, target, entry)
            original = _birth(files, public, entry, gid)
            rows = sorted(saved_manifest["members"], key=lambda row: (len(Path(row[0]).parts), row[0]), reverse=True)
            receipt, logical, allocated = recovery.retired(files, config, action, expected_action_intent,
                                                          target, rows, entry, original["marker"])
            return _outcome(action, "retired", "already_retired", receipt=receipt, logical=logical, allocated=allocated)
        target, target_fd = _target(files, config, entry)
        lease, lease_record = _lease(files, target, entry)
        _require(issued >= lease["expires_at_epoch"] and lease["released_at_epoch"] is None, "experiment_not_expired")
        original = _birth(files, public, entry, gid)
        if action["action"] == "delete":
            _require(original["participant_profile"] == "local_root_disposable.v1" and lease["class_intent"] == "scratch"
                     and lease["cleanup"] == "delete", "experiment_action_profile_unsupported")
        else:
            _require(action["action"] == "offload" and lease["class_intent"] == "evidence", "experiment_action_profile_unsupported")
            from .control_plane_lane_experiment_completion import selected_completion
            selected_completion(files, config, entry)
        if _pins_root is None:
            return _outcome(action, "kept", "experiment_reference_authority_missing")
        reference, reference_fd = _pin_fence(files, config, _pins_root, target, issued)
        public = birth_code._authority_lock(files, config.experiment_authority_root, gid)
        refreshed = _current(files, public, gid)
        _require(refreshed[0] == current[0] and entry["state"] in ("active", "retiring"), "experiment_action_current_changed")
        store = issuance._store(files, config.experiment_record_store)
        files.phase('manifest')
        manifest_raw, _ = files.read(Path(config.experiment_record_store) / (action_id + ".manifest.json"),
                                    cap=1048576, protected=True, mode=0o600)
        _require(issuance._selector(manifest_raw, files.budget) == action["manifest"], "experiment_manifest_changed")
        manifest = _manifest_record(files, manifest_raw, entry)
        _require(manifest["schema_version"] == MANIFEST_SCHEMA and manifest["manifest_digest"]
                 == canonical_digest(manifest, digest_field="manifest_digest"), "experiment_manifest_invalid")
        _require(len(manifest["members"]) <= 4096, "experiment_manifest_limit")
        from . import control_plane_lane_experiment_recovery as recovery
        rows = sorted(manifest["members"], key=lambda row: (len(Path(row[0]).parts), row[0]), reverse=True)
        files._store_path = config.experiment_record_store
        reservation_size = len(rows) * 2 * 4096 + 8 * 32768
        reserve_raw = _encoded(dict(schema_version="control_plane_lane_experiment_reservation.v1",
            operation_id=action_id, reserved_bytes=reservation_size), "reservation_digest", 4096)
        if entry["state"] == "active":
            occupied = issuance._capacity(files, store, adding_registration=False)
            _require(occupied + reservation_size + 32768 <= issuance.MAX_EXPERIMENT_STORE_BYTES, "experiment_store_full")
        recovery._once(files, store, action_id + ".reservation.json", reserve_raw, kind="private")
        operation, retiring, previous, logical, allocated, changed_directories, removed_count, receipt, preservation = recovery.begin(
            files, config, action, expected_action_intent, entry, current, refreshed, public, store,
            target, rows, reference, issued, gid)
        offset = int(action["action"] == "offload")
        if offset:
            from . import control_plane_lane_experiment_archive as archive
            def archive_guard():
                _require(now() < action["expires_at_epoch"], "experiment_action_expired")
                files.verify()
                files.location(target_fd)
                files.location(reference_fd)
                files.verify_record(retiring[2])
            if preservation is None:
                archived = archive.preserve(files, config, target, rows, manifest_raw, archive_guard)
                files.phase("ready")
                ready = _event(files, operation, action, "preservation_ready", dict(started=previous,
                    action=expected_action_intent, birth=entry["birth"], manifest=action["manifest"], archive=archived,
                    target_identity=entry["target_identity"], lease=entry["lease"]), 1, previous, issued)
                preservation, previous = (ready, archived), ready
            else:
                archive.verify_preservation(files, config, preservation[1], archive_guard)
                files.phase("ready")
        # Exactly 16 source members per declared native mutation phase. Full
        # hashing runs under the original action clock with original FDs held;
        # metadata clocks are never reset inside a batch or payload loop.
        for start in range(removed_count, len(rows), 16):
            files.payload(target, target_fd, expected_payload_bytes=files.payload_size if files.payload_size is not None else manifest["logical_bytes"])
            held = {}
            try:
                for index in range(start + 1, min(start + 16, len(rows)) + 1):
                    row = rows[index - 1]
                    if row[1] == "file":
                        archive_guard() if offset else files.verify()
                        held[index] = _member(files, target, row, hash_role="remove_hash")
                files.phase("removal_batch")
                for index in range(start + 1, min(start + 16, len(rows)) + 1):
                    row = rows[index - 1]
                    _require(now() < action["expires_at_epoch"], "experiment_action_expired")
                    files.verify_record(lease_record)
                    files.verify()
                    files.location(target_fd)
                    files.location(reference_fd)
                    files.proof(reference_fd)
                    files.verify_record(retiring[2])
                    parent, name, fd, info = held.pop(index) if row[1] == "file" else _member(
                        files, target, row, changed_directories.get(row[0]), hash_payload=False)
                    try:
                        files.location(parent)
                        files.proof(fd)
                        _require(owners._metadata(os.fstat(fd)) == owners._metadata(info)
                                 == owners._metadata(os.stat(name, dir_fd=parent, follow_symlinks=False)), "experiment_member_changed")
                        if row[1] == "directory":
                            os.rmdir(name, dir_fd=parent)
                            files.removed_directory(target / row[0], fd)
                        else:
                            os.unlink(name, dir_fd=parent)
                            logical += info.st_size
                        allocated += info.st_blocks * 512
                        files.location(parent)
                        os.fsync(parent)
                        updated = os.fstat(parent)
                        changed_directories[str(Path(row[0]).parent)] = tuple(getattr(updated, key) for key in
                            ("st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))
                        previous = _event(files, operation, action, "member_removed", dict(preservation=preservation[0] if preservation else None,
                            action=expected_action_intent, manifest=action["manifest"], index=index - 1,
                            path=row[0], original_identity=dict(dev=info.st_dev, ino=info.st_ino, type=row[1]),
                            logical_bytes=info.st_size if row[1] == "file" else 0, eligible_allocated_bytes=info.st_blocks * 512,
                            parent_after=dict(path=str(Path(row[0]).parent), identity=dict(dev=updated.st_dev, ino=updated.st_ino, type="directory"),
                                stat_token=":".join(str(value) for value in changed_directories[str(Path(row[0]).parent)]))),
                            index + offset, previous, issued)
                    finally:
                        files.close(fd)
            finally:
                for _, _, fd, _ in held.values():
                    files.close(fd)
                files.trim_payload()
        files.location(target_fd)
        os.fsync(target_fd)
        if receipt is None:
            receipt = _event(files, operation, action, "retired", dict(preservation=preservation[0] if preservation else None, manifest=action["manifest"],
            removed_event_count=len(rows), removed_logical_bytes=logical, eligible_allocated_bytes=allocated,
            remaining_metadata=[entry["lease"], original["marker"]], partial=False), len(rows) + 1 + offset, previous, issued)
        files.phase("finalize")
        prepared, old_head = _version(files, public, retiring, entry | {"state": "retired"}, gid, action["policy"], issued)
        recovery._once(files, store, action_id + ".retired-head.json", prepared, kind="private")
        _install_head(files, public, prepared, gid, old_head)
        return _outcome(action, "retired", "evidence_preserved" if offset else "disposable_expired", receipt=receipt, logical=logical, allocated=allocated)
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
