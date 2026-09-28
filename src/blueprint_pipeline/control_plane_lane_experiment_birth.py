"""Authenticated new experiment birth, current authority and owned stage creation."""
from __future__ import annotations

import fcntl
import grp
import os
import pwd
import secrets
import stat
import time
from pathlib import Path

from . import control_plane_lane_experiment_retirement as issuance
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch as scratch
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_publication import _BirthFiles, _publish
from .control_plane_lane_experiment_authority import _current
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _epoch, _require
from .control_plane_reference_budget import ReferenceCollectionBudget
from .decision_evidence_contracts import canonical_digest

_INTENT_FIELDS = frozenset({"schema_version", "intent_id", "generation", "issuer_uid", "principal", "owner",
    "root", "lane", "name", "reference_kind", "reference_value", "reason", "class_intent", "cleanup",
    "lease_ttl_seconds", "issued_at_epoch", "expires_at_epoch", "policy", "request_records", "writer_scope",
    "participant_profile", "intent_digest"})
_MARKER = ".registered-experiment.v1.json"


def _blueprint_identity():
    try:
        return pwd.getpwnam("blueprint").pw_uid, grp.getgrnam("blueprint").gr_gid
    except KeyError:
        raise OwnerTargetVersionError("experiment_account_missing") from None


def _encode(files, value, field, cap=32768):
    files.budget.measure(value, cap=cap - 100)
    files.budget.tick()
    record = value | {field: canonical_digest(value, digest_field=field)}
    files.budget.tick()
    return record, owners._encoded(record, files.budget, cap=cap)


def _event(files, intent, operation_id, kind, body, sequence, previous, issued):
    return _encode(files, dict(schema_version="control_plane_lane_experiment_event.v1",
        event_id=secrets.token_hex(16), operation_id=operation_id, event_kind=kind,
        intent_id=intent["intent_id"], generation=intent["generation"], sequence=sequence,
        previous_event=previous, issued_at_epoch=issued, body=body), "event_digest")


def _authority_lock(files, path, gid):
    parent, _ = files.parent(Path(path) / "HEAD.json")
    info = os.fstat(parent)
    _require(info.st_uid == 0 and info.st_gid == gid and stat.S_IMODE(info.st_mode) == 0o750,
             "experiment_authority_unsafe")
    fd = files.open(".authority.lock", os.O_RDONLY | os.O_NONBLOCK, parent=parent)
    info = files.acquired[fd]
    _require(info.st_uid == 0 and info.st_gid == gid and stat.S_IMODE(info.st_mode) == 0o640
             and info.st_nlink == 1 and info.st_size == 0, "experiment_authority_unsafe")
    files.location(parent)
    files.proof(fd)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise OwnerTargetVersionError("experiment_authority_busy") from None
    return parent


def _locked_lane(files, root):
    root_fd, _ = files.parent(Path(root) / "g1")
    lock = files.open(".lane-scratch.lock", os.O_RDWR | os.O_NONBLOCK, parent=root_fd)
    info = files.acquired[lock]
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size == 0
             and not info.st_mode & 0o022, "experiment_lane_lock_unsafe")
    files.location(root_fd)
    files.proof(lock)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise OwnerTargetVersionError("experiment_lane_busy") from None
    return files.open("g1", os.O_RDONLY | os.O_DIRECTORY, parent=root_fd)


class _RegisteredBirth:
    """Exact private native-constructor composition, never a caller callback."""
    def __init__(self, files, lane_fd, root, stage_name, intent, operation, claim, uid, gid):
        self.files, self.lane_fd, self.root = files, lane_fd, root
        self.stage_name, self.intent = stage_name, intent
        self.operation, self.claim, self.uid, self.gid = operation, claim, uid, gid
        self.fd = None
        self.lease = None

    def publish_creation(self, lease):
        _require(self.fd is None and lease["lane"] == "g1" and lease["name"] == self.stage_name
                 and lease["owner"] == self.intent["owner"], "experiment_creation_invalid")
        files = self.files
        self.fd = files.new_directory(self.lane_fd, self.stage_name)
        self.lease = scratch._seal(lease | {"name": self.intent["name"],
            "expires_at_epoch": self.intent["expires_at_epoch"]})
        payload = owners._encoded(self.lease, files.budget, cap=scratch.MAX_LEASE_BYTES)
        self.lease_selector = _publish(files, self.fd, scratch.LEASE_FILE, payload, kind="lease")
        files.location(self.fd)
        files.proof(self.fd)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise OwnerTargetVersionError("experiment_stage_busy") from None
        return Path(self.root) / "g1" / self.stage_name


def _read_intent(files, config, intent_id, expected, issued):
    _require(owners._matches(intent_id, owners._CONSENT_ID) and isinstance(expected, dict)
             and set(expected) == {"sha256", "size_bytes"}, "experiment_creation_invalid")
    raw, record = files.read(Path(config.experiment_record_store) / (intent_id + ".json"),
                              cap=32768, protected=True, mode=0o600)
    owners._identity(raw, expected["sha256"], expected["size_bytes"], files.budget)
    intent = retained._document(raw, 32768, _work_budget=files.budget)
    _require(set(intent) == _INTENT_FIELDS and intent["schema_version"] == issuance.CREATION_SCHEMA
             and intent["intent_id"] == intent_id and intent["name"] == "registered-" + intent_id
             and intent["lane"] == "g1" and intent["root"] in ("work", "inputs")
             and intent["issuer_uid"] == 0 and type(intent["issuer_uid"]) is int
             and intent["intent_digest"] == canonical_digest(intent, digest_field="intent_digest")
             and _epoch(issued) and _epoch(intent["expires_at_epoch"])
             and intent["issued_at_epoch"] <= issued < intent["expires_at_epoch"], "experiment_creation_invalid")
    policy_raw, policy_record = files.read(config.lane_owner_policy_file, cap=owners.MAX_POLICY_BYTES,
                                          protected=True, mode=0o600)
    _require(issuance._selector(policy_raw, files.budget) == intent["policy"], "experiment_policy_changed")
    policy = owners._policy(policy_raw, intent["principal"], files.budget)
    owners._authorize(dict(action="register", owner=intent["owner"], ttl_seconds=intent["lease_ttl_seconds"]),
                      policy, intent["expires_at_epoch"], intent["issued_at_epoch"])
    files.verify_record(record)
    files.verify_record(policy_record)
    return intent, issued + policy["max_consent_seconds"]


def _create(files, intent_id, expected, config_path, issued):
    _require(os.geteuid() == 0, "experiment_issuer_required")
    config = issuance._configuration(files, config_path)
    _require(config.experiment_creation_enabled is True, "experiment_creation_disabled")
    intent, projection_expiry = _read_intent(files, config, intent_id, expected, issued)
    uid, gid = _blueprint_identity()
    public = _authority_lock(files, config.experiment_authority_root, gid)
    store = issuance._store(files, config.experiment_record_store)
    try:
        os.stat(intent_id + ".claim.json", dir_fd=store, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise OwnerTargetVersionError("experiment_creation_already_claimed")
    previous = _current(files, public, gid)
    if previous is not None:
        _require(previous[1]["state"] == "enabled" and previous[1]["policy"] == intent["policy"]
                 and len(previous[1]["enrollments"]) < 100
                 and all(row["intent_id"] != intent_id for row in previous[1]["enrollments"]),
                 "experiment_authority_update_refused")
    occupied = issuance._capacity(files, store, adding_registration=False)
    # Reserve the complete finite birth before its first durable claim. Actual
    # encoded records remain charged; this does not reserve payload or archive.
    _require(occupied + 8 * 32768 <= issuance.MAX_EXPERIMENT_STORE_BYTES, "experiment_store_full")
    operation_id = secrets.token_hex(16)
    _, claim_bytes = _event(files, intent, operation_id, "creation_claim",
        dict(intent=expected, root=intent["root"], lane="g1", name=intent["name"]), 0, None, issued)
    claim = _publish(files, store, intent_id + ".claim.json", claim_bytes, kind="private")
    root = config.lane_scratch_work_root if intent["root"] == "work" else config.lane_scratch_inputs_root
    lane = _locked_lane(files, root)
    try:
        os.stat(intent["name"], dir_fd=lane, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise OwnerTargetVersionError("experiment_creation_target_exists")
    stage_name = "create-" + operation_id
    composer = _RegisteredBirth(files, lane, root, stage_name, intent, operation_id, claim, uid, gid)
    scratch.create_lane_scratch("g1", stage_name, owner=intent["owner"], reason=intent["reason"],
        class_intent=intent["class_intent"], cleanup=intent["cleanup"], ttl_seconds=intent["lease_ttl_seconds"],
        run_ref=intent["reference_value"], root=root, now=lambda: intent["issued_at_epoch"],
        consumer_lifetime_contract=scratch.CONSUMER_LIFETIME_PROTOCOL, _registered_birth=composer)
    stage = composer.fd
    stage_identity = dict(dev=os.fstat(stage).st_dev, ino=os.fstat(stage).st_ino, type="directory")
    _, creation_bytes = _event(files, intent, operation_id, "creation_prepared",
        dict(stage_name=stage_name, stage_identity=stage_identity, lease=composer.lease_selector), 1, claim, issued)
    creation = _publish(files, store, intent_id + ".creation.json", creation_bytes, kind="private")
    _, marker_bytes = _encode(files, dict(schema_version="control_plane_lane_experiment_marker.v1",
        intent=expected, claim=claim, create_operation=creation, generation=intent["generation"],
        root=intent["root"], lane="g1", name=intent["name"], writer_scope=intent["writer_scope"]),
        "marker_digest", 4096)
    marker = _publish(files, stage, _MARKER, marker_bytes, kind="marker")
    for name in (scratch.LEASE_FILE, _MARKER):
        fd = files.open(name, os.O_RDONLY, parent=stage)
        files.transition_owner(fd, uid, gid, 0o600)
        files.close(fd)
    files.transition_owner(stage, uid, gid, 0o700)
    files.location(lane)
    files.location(stage)
    scratch._publish_no_replace(lane, stage_name, intent["name"])
    named = os.stat(intent["name"], dir_fd=lane, follow_symlinks=False)
    _require((named.st_dev, named.st_ino) == (stage_identity["dev"], stage_identity["ino"]),
             "experiment_location_changed")
    files.bindings[stage] = (lane, intent["name"], owners._security(named))
    files.location(stage)
    files.location(lane)
    os.fsync(lane)
    _, publication_bytes = _event(files, intent, operation_id, "creation_published",
        dict(creation=creation, target_identity=stage_identity, marker=marker, lease=composer.lease_selector),
        2, creation, issued)
    publication = _publish(files, store, intent_id + ".publication.json", publication_bytes, kind="private")
    _, birth_bytes = _encode(files, dict(schema_version="control_plane_lane_experiment_birth.v1",
        intent_id=intent_id, generation=intent["generation"], root=intent["root"], lane="g1", name=intent["name"],
        publication=publication, marker=marker, target_identity=stage_identity, lease=composer.lease_selector,
        owner=intent["owner"], reference_kind=intent["reference_kind"], reference_value=intent["reference_value"],
        class_intent=intent["class_intent"], cleanup=intent["cleanup"], expires_at_epoch=intent["expires_at_epoch"],
        writer_scope=intent["writer_scope"], participant_profile=intent["participant_profile"]), "birth_digest")
    birth = _publish(files, public, intent_id + ".birth.json", birth_bytes, kind="birth", blueprint_gid=gid)
    _, correspondence_bytes = _event(files, intent, operation_id, "birth_correspondence",
        dict(birth=birth, intent=expected, claim=claim, publication=publication, marker=marker,
             target_identity=stage_identity, lease=composer.lease_selector, principal=intent["principal"], policy=intent["policy"]),
        3, publication, issued)
    correspondence = _publish(files, store, intent_id + ".correspondence.json", correspondence_bytes, kind="private")
    epoch = previous[0]["authority_epoch_id"] if previous is not None else secrets.token_hex(16)
    version = previous[0]["version"] + 1 if previous is not None else 0
    entry = dict(intent_id=intent_id, generation=intent["generation"], birth=birth, target_identity=stage_identity,
        lease=composer.lease_selector, owner=intent["owner"], root=intent["root"], lane="g1", name=intent["name"],
        state="active", completion=None, restoration=None, operation_id=None, expires_at_epoch=intent["expires_at_epoch"])
    _, authority_bytes = _encode(files, dict(schema_version="control_plane_lane_experiment_authority.v1",
        authority_epoch_id=epoch, version=version, previous_record=previous[0]["record"] if previous is not None else None,
        state="enabled", issued_at_epoch=issued, expires_at_epoch=projection_expiry, policy=intent["policy"],
        enrollments=sorted((previous[1]["enrollments"] if previous is not None else []) + [entry],
                           key=lambda row: row["intent_id"])), "authority_digest")
    authority_name = f"authority-{version:08d}-{epoch}.json"
    authority = _publish(files, public, authority_name, authority_bytes, kind="authority", blueprint_gid=gid)
    _, head_bytes = _encode(files, dict(schema_version="control_plane_lane_experiment_head.v1",
        authority_epoch_id=epoch, version=version, record_name=authority_name, record=authority), "head_digest", 4096)
    files.location(stage)
    for name, selector, cap in ((scratch.LEASE_FILE, composer.lease_selector, scratch.MAX_LEASE_BYTES),
                                (_MARKER, marker, 4096)):
        raw, record = files.read(Path(root) / "g1" / intent["name"] / name, cap=cap)
        _require(issuance._selector(raw, files.budget) == selector, "experiment_stage_changed")
        files.verify_record(record)
    files.verify()
    prepared_head = _publish(files, store, intent_id + ".head-prepared.json", head_bytes, kind="private")
    previous_head = None
    if previous is not None:
        files.verify_record(previous[2])
        files.proof(previous[2].fd)
        os.lseek(previous[2].fd, 0, os.SEEK_SET)
        previous_head = issuance._selector(files.read_bytes(previous[2].fd, 4096), files.budget)
        files.verify_record(previous[2])
    _, pending_bytes = _event(files, intent, operation_id, "authority_pending",
        dict(authority=authority, prepared_head=prepared_head, previous_head=previous_head), 4, correspondence, issued)
    pending = _publish(files, store, intent_id + ".authority-pending.json", pending_bytes, kind="private")
    files.verify()
    head = _publish(files, public, "HEAD.json", head_bytes, kind="head", blueprint_gid=gid,
                    _expected_head=previous[2] if previous is not None else None)
    _, completed_bytes = _event(files, intent, operation_id, "birth_completed",
        dict(birth=birth, authority=authority, head=head), 5, pending, issued)
    _publish(files, store, intent_id + ".completed.json", completed_bytes, kind="private")
    return dict(path=str(Path(root) / "g1" / intent["name"]), generation=intent["generation"], birth=birth)


def create_registered_experiment(intent_id, *, expected_intent,
        installed_config_path="/etc/blueprint-operator-door/door.json", now=time.time):
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        return _create(files, intent_id, expected_intent, installed_config_path, now())
    except OSError:
        raise OwnerTargetVersionError("experiment_birth_io_failed") from None
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()
