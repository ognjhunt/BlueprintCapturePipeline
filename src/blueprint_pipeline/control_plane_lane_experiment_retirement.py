"""Fixed root experiment registration and retirement entrypoints.

Creation intent is permission for a new birth, never a payload deletion grant.
Legacy experiment directories are not adopted by this interface.
"""
from __future__ import annotations

import fcntl
import os
import re
import secrets
import sys
import time
import types
from pathlib import Path

from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_owner_target_io import _TargetFiles
from .control_plane_lane_owner_target_publication import _publish_owned_metadata
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _epoch, _require
from .control_plane_reference_budget import ReferenceCollectionBudget
from .decision_evidence_contracts import canonical_digest

CREATION_SCHEMA = "control_plane_lane_experiment_creation_intent.v1"
_LOCK = ".experiment-authority.lock"
_MAX_INTENT = 32768
MAX_EXPERIMENT_REGISTRATIONS = 256
MAX_EXPERIMENT_STORE_BYTES = 64 * 1024 * 1024
_STORE_NAME = re.compile(r"([0-9a-f]{32})(?:\.(claim|creation|publication|correspondence|completed|producer-completion|completion-head|head-prepared|authority-pending|action|manifest|reservation|retiring-head|retired-head))?\.json\Z")
_PROFILES = {
    "local_root_disposable.v1": ("owner_disposable_scratch", "scratch", "delete", "fixed_root_scratch_issuer.v1", 0),
    "g1_local_prelaunch_block.v1": ("g1_development_pair", "evidence", "owner_review", "native_g1_development_pair.v1", 2),
    "g1_local_contained_completed.v1": ("g1_development_pair", "evidence", "owner_review", "native_g1_development_pair.v1", 2),
}


def _selector(raw, budget):
    return {"sha256": retained._digest(raw, _work_budget=budget), "size_bytes": len(raw)}


def _configuration(files, path):
    """Only acquired protected installed bridge bytes may select fixed paths."""
    raw, _ = files.read(path, cap=owners.MAX_POLICY_BYTES, protected=True)
    package = owners.INSTALLED_PACKAGE_ROOT / "operator_door"
    files.read(package / "__init__.py", cap=owners.MAX_POLICY_BYTES, protected=True)
    source, acquired = files.read(package / "config.py", cap=owners.MAX_POLICY_BYTES, protected=True)
    files.verify()
    name = "_blueprint_experiment_config_" + secrets.token_hex(8)
    module = types.ModuleType(name)
    module.__file__ = str(package / "config.py")
    sys.modules[name] = module
    try:
        files.budget.tick()
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        files.budget.tick()
        mapping = retained._document(raw, owners.MAX_POLICY_BYTES, _work_budget=files.budget)
        config = module.config_from_mapping(mapping, _work_budget=files.budget)
    except (SyntaxError, UnicodeError, ImportError, AttributeError, retained.CensusDecisionError):
        raise OwnerTargetVersionError("experiment_configuration_invalid") from None
    except ValueError as error:
        if type(error).__name__ == "DoorConfigError":
            raise OwnerTargetVersionError("experiment_configuration_invalid") from None
        raise
    finally:
        sys.modules.pop(name, None)
    files.verify_record(acquired)
    roots = owners._roots(config, files.budget)
    private, public = Path(config.experiment_record_store), Path(config.experiment_authority_root)
    _require(private != public and private not in public.parents and public not in private.parents
             and all(root != location and root not in location.parents and location not in root.parents
                     for root in roots for location in (private, public)), "experiment_configuration_invalid")
    return config


def _store(files, path):
    parent, _ = files.parent(Path(path) / "record.json", protected=True)
    owners._protected(os.fstat(parent), directory=True, mode=0o700)
    fd = files.open(_LOCK, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, parent=parent)
    info = files.acquired[fd]
    owners._protected(info, mode=0o600)
    _require(info.st_size == 0, "experiment_store_unsafe")
    files.proof(parent)
    files.proof(fd)
    _require(owners._metadata(os.stat(_LOCK, dir_fd=parent, follow_symlinks=False)) == owners._metadata(info),
             "experiment_store_unsafe")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise OwnerTargetVersionError("experiment_store_busy") from None
    files.records.append(owners._Acquired(fd, parent, _LOCK, info))
    return parent


def _capacity(files, parent, *, adding_registration=True):
    count, total, records = 0, 0, 0
    files.slot()
    with os.scandir(parent) as entries:
        for item in entries:
            files.budget.charge("entries")
            files.proof(parent)
            info = os.stat(item.name, dir_fd=parent, follow_symlinks=False)
            if item.name == "operations":
                owners._protected(info, directory=True, mode=0o700)
                operations = files.open(item.name, os.O_RDONLY | os.O_DIRECTORY, parent=parent)
                try:
                    files.slot()
                    operation_count = 0
                    with os.scandir(operations) as named_operations:
                        for operation in named_operations:
                            files.budget.charge("entries")
                            operation_count += 1
                            _require(operation_count <= 256 and owners._matches(operation.name, owners._CONSENT_ID),
                                     "experiment_store_full")
                            files.proof(operations)
                            owners._protected(os.stat(operation.name, dir_fd=operations, follow_symlinks=False),
                                              directory=True, mode=0o700)
                            reservation_fd = files.open(operation.name + ".reservation.json", os.O_RDONLY, parent=parent)
                            try:
                                reservation_info = os.fstat(reservation_fd)
                                owners._protected(reservation_info, mode=0o600)
                                reserved = retained._document(files.read_bytes(reservation_fd, 4096), 4096,
                                                              _work_budget=files.budget)
                                _require(set(reserved) == {"schema_version", "operation_id", "reserved_bytes", "reservation_digest"}
                                         and reserved["schema_version"] == "control_plane_lane_experiment_reservation.v1"
                                         and reserved["operation_id"] == operation.name
                                         and type(reserved["reserved_bytes"]) is int and 0 < reserved["reserved_bytes"] <= MAX_EXPERIMENT_STORE_BYTES
                                         and reserved["reservation_digest"] == canonical_digest(reserved, digest_field="reservation_digest"),
                                         "experiment_store_unsafe")
                                total += reserved["reserved_bytes"]
                                _require(total <= MAX_EXPERIMENT_STORE_BYTES, "experiment_store_full")
                            finally:
                                files.close(reservation_fd)
                finally:
                    files.close(operations)
                continue
            owners._protected(info, mode=0o600)
            if item.name == _LOCK:
                _require(info.st_size == 0, "experiment_store_unsafe")
                continue
            match = _STORE_NAME.fullmatch(item.name)
            _require(match is not None, "experiment_store_unsafe")
            records += 1
            if match.group(2) is None:
                count += 1
            total += info.st_size
            _require(records <= MAX_EXPERIMENT_REGISTRATIONS * 12
                     and count <= MAX_EXPERIMENT_REGISTRATIONS - int(adding_registration)
                     and 0 < info.st_size <= (1048576 if match.group(2) == "manifest" else _MAX_INTENT)
                     and total <= MAX_EXPERIMENT_STORE_BYTES,
                     "experiment_store_full")
    files.budget.tick()
    return total


def _issue(files, *, installed_config_path, principal, owner, root, reference_value,
           lease_ttl_seconds, participant_profile, request_records, issued, expiry):
    _require(os.geteuid() == 0, "experiment_issuer_required")
    config = _configuration(files, installed_config_path)
    _require(config.experiment_creation_enabled is True, "experiment_creation_disabled")
    _require(_epoch(issued) and type(lease_ttl_seconds) is int and 0 < lease_ttl_seconds <= 1209600
             and _epoch(expiry) and issued < expiry <= issued + lease_ttl_seconds
             and owners._matches(principal, owners._PRINCIPAL) and owners._matches(owner, owners._OWNER)
             and root in ("work", "inputs") and owners._matches(reference_value, owners._OWNER)
             and participant_profile in _PROFILES and isinstance(request_records, (tuple, list)),
             "experiment_creation_invalid")
    reason, class_intent, cleanup, writer, number = _PROFILES[participant_profile]
    _require(len(request_records) == number, "experiment_creation_invalid")
    policy_raw, policy_record = files.read(config.lane_owner_policy_file, cap=owners.MAX_POLICY_BYTES,
                                          protected=True, mode=0o600)
    policy = owners._policy(policy_raw, principal, files.budget)
    owners._authorize(dict(action="register", owner=owner, ttl_seconds=lease_ttl_seconds), policy, expiry, issued)
    selectors = []
    for request in request_records:
        _require(isinstance(request, tuple) and len(request) == 2 and isinstance(request[1], dict)
                 and set(request[1]) == {"sha256", "size_bytes"}, "experiment_creation_invalid")
        raw, _ = files.read(request[0], cap=owners.MAX_POLICY_BYTES, protected=True)
        owners._identity(raw, request[1]["sha256"], request[1]["size_bytes"], files.budget)
        retained._document(raw, owners.MAX_POLICY_BYTES, _work_budget=files.budget)
        selectors.append(_selector(raw, files.budget))
    intent_id, generation = secrets.token_hex(16), secrets.token_hex(16)
    _require(owners._matches(intent_id, owners._CONSENT_ID) and owners._matches(generation, owners._CONSENT_ID)
             and intent_id != generation, "experiment_creation_invalid")
    record = dict(schema_version=CREATION_SCHEMA, intent_id=intent_id, generation=generation, issuer_uid=0,
        principal=principal, owner=owner, root=root, lane="g1", name="registered-" + intent_id,
        reference_kind="run_ref", reference_value=reference_value, reason=reason, class_intent=class_intent,
        cleanup=cleanup, lease_ttl_seconds=lease_ttl_seconds, issued_at_epoch=issued, expires_at_epoch=expiry,
        policy=_selector(policy_raw, files.budget), request_records=selectors,
        writer_scope=writer, participant_profile=participant_profile)
    record["intent_digest"] = canonical_digest(record, digest_field="intent_digest")
    payload = owners._encoded(record, files.budget, cap=_MAX_INTENT)
    parent = _store(files, config.experiment_record_store)
    occupied = _capacity(files, parent)
    _require(occupied + len(payload) <= MAX_EXPERIMENT_STORE_BYTES, "experiment_store_full")
    files.verify_record(policy_record)
    files.verify()
    published = _publish_owned_metadata(files, parent, intent_id + ".json", payload,
                                        mode=0o600, artifact_kind="attestation")
    files.verify()
    return {"intent_id": intent_id, "intent": {key: published[key] for key in ("sha256", "size_bytes")}}


def issue_experiment_creation_intent(*, installed_config_path="/etc/blueprint-operator-door/door.json",
        principal, owner, root, reference_value, lease_ttl_seconds, participant_profile,
        request_records, expires_at_epoch=None, now=time.time):
    """Root-only authentic immutable intent; no target or child is created here."""
    budget = ReferenceCollectionBudget(values_limit=10000)
    files = _TargetFiles(budget)
    try:
        issued = now()
        expiry = (issued + lease_ttl_seconds if expires_at_epoch is None
                  and _epoch(issued) and type(lease_ttl_seconds) is int else expires_at_epoch)
        return _issue(files, installed_config_path=installed_config_path, principal=principal, owner=owner,
            root=root, reference_value=reference_value, lease_ttl_seconds=lease_ttl_seconds,
            participant_profile=participant_profile, request_records=request_records, issued=issued, expiry=expiry)
    except OSError:
        raise OwnerTargetVersionError("experiment_io_failed") from None
    finally:
        try:
            files.finish()
        finally:
            budget.close()


def issue_experiment_action_intent(intent_id, *, principal, owner, action, expires_at_epoch,
        installed_config_path="/etc/blueprint-operator-door/door.json", now=time.time):
    from .control_plane_lane_experiment_actions import issue_action
    return issue_action(intent_id, principal=principal, owner=owner, action=action,
        expires_at_epoch=expires_at_epoch, installed_config_path=installed_config_path, now=now)


def run_registered_experiment_action(action_id, *, expected_action_intent,
        installed_config_path="/etc/blueprint-operator-door/door.json", now=time.time, _pins_root=None):
    from .control_plane_lane_experiment_actions import run_action
    return run_action(action_id, expected_action_intent=expected_action_intent,
        installed_config_path=installed_config_path, now=now, _pins_root=_pins_root)
