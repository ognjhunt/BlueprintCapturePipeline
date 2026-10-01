"""Fixed root experiment registration and retirement entrypoints.

Creation intent is permission for a new birth, never a payload deletion grant.
Legacy experiment directories are not adopted by this interface.
"""
from __future__ import annotations

import argparse
import base64
import json
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
from .control_plane_lane_experiment_publication import _BirthFiles, _publish
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
_STORE_NAME = re.compile(r"([0-9a-f]{32})(?:\.(claim|creation|publication|correspondence|completed|producer-invocation|producer-completion|completion-head|restore-intent|restore-selection|restore-pending-head|restore-head|restored-head|head-prepared|authority-pending|action|manifest|stage-manifest|payload-manifest|lease-transition|reservation|scan-reservation|retiring-head|retired-head))?\.json\Z")
_ARENA_TAG = re.compile(r"arena-launch-(r[1-9][0-9]{0,5})\Z")
_ARENA_CLAIM_NAME = re.compile(r"arena-launch-r[1-9][0-9]{0,5}\.arena-claim\.json\Z")
_ISSUE_SELECTION_NAME = re.compile(r'[0-9a-f]{32}\.issue-selection-[0-9a-f]{64}\.json\Z')
_PROFILES = {
    "arena_owner_review.v1": ("arena_construction_launch", "evidence", "owner_review", "arena_construction_launch_chain.v1", 0),
    "local_root_disposable.v1": ("owner_disposable_scratch", "scratch", "delete", "fixed_root_scratch_issuer.v1", 0),
    "g1_local_prelaunch_block.v1": ("g1_development_pair", "evidence", "owner_review", "native_g1_development_pair.v1", 2),
    "g1_local_contained_completed.v1": ("g1_development_pair", "evidence", "owner_review", "native_g1_development_pair.v1", 2),
    "root_disk_diagnostic_disposable.v1": ("disk_capacity_diagnostic", "scratch", "delete", "fixed_root_disk_diagnostic.v1", 1),
    "root_disk_diagnostic_evidence.v1": ("disk_capacity_diagnostic", "evidence", "offload", "fixed_root_disk_diagnostic.v1", 1),
}


def _profile_lane(profile):
    if profile == "arena_owner_review.v1":
        return "arena"
    if profile in ("root_disk_diagnostic_disposable.v1", "root_disk_diagnostic_evidence.v1"):
        return "diagnostics"
    return "g1"


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
    from .control_plane_lane_experiment_consumer import LANE_ROOTS
    _require((Path(config.lane_scratch_work_root), Path(config.lane_scratch_inputs_root)) == tuple(LANE_ROOTS),
             'experiment_installed_namespace_changed')
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
                            found_reservations = 0
                            for reservation_suffix in ("reservation", "scan-reservation"):
                                try:
                                    os.stat(operation.name + "." + reservation_suffix + ".json", dir_fd=parent, follow_symlinks=False)
                                except FileNotFoundError:
                                    continue
                                found_reservations += 1
                                reservation_fd = files.open(operation.name + "." + reservation_suffix + ".json", os.O_RDONLY, parent=parent)
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
                            _require(found_reservations > 0, "experiment_store_unsafe")
                finally:
                    files.close(operations)
                continue
            owners._protected(info, mode=0o600)
            if item.name == _LOCK:
                _require(info.st_size == 0, "experiment_store_unsafe")
                continue
            match = _STORE_NAME.fullmatch(item.name)
            selection = _ISSUE_SELECTION_NAME.fullmatch(item.name) is not None
            arena_claim = _ARENA_CLAIM_NAME.fullmatch(item.name) is not None
            _require(match is not None or selection or arena_claim, "experiment_store_unsafe")
            records += 1
            if match is not None and match.group(2) is None:
                count += 1
            total += info.st_size
            if match is not None and match.group(2) == "producer-invocation":
                from .control_plane_lane_disk_diagnostic import _reserved_bytes
                fd = files.open(item.name, os.O_RDONLY | os.O_NONBLOCK, parent=parent)
                try:
                    value = retained._document(files.read_bytes(fd, 32768), 32768, _work_budget=files.budget)
                    _require(value['intent_id'] == match.group(1), "experiment_store_unsafe")
                    total += _reserved_bytes(value)
                finally:
                    files.close(fd)
            _require(records <= MAX_EXPERIMENT_REGISTRATIONS * 12
                     and count <= MAX_EXPERIMENT_REGISTRATIONS - int(adding_registration)
                     and 0 < info.st_size <= (4096 if selection or arena_claim else 1048576 if match.group(2) in ("manifest", "stage-manifest", "payload-manifest") else _MAX_INTENT)
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
    arena = participant_profile == "arena_owner_review.v1"
    _require(not arena or root == "inputs" and _ARENA_TAG.fullmatch(reference_value),
             "experiment_arena_tag_invalid")
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
        if _profile_lane(participant_profile) == "diagnostics":
            from .control_plane_lane_disk_diagnostic import validate_request
            validate_request(files, raw, config=config, installed_config_path=installed_config_path,
                             run_ref=reference_value)
        selectors.append(_selector(raw, files.budget))
    parent = _store(files, config.experiment_record_store)
    occupied = _capacity(files, parent)
    claim_name = reference_value + ".arena-claim.json" if arena else None
    if arena:
        files.location(parent)
        try:
            os.stat(claim_name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise OwnerTargetVersionError("experiment_arena_tag_claimed")
    intent_id, generation = secrets.token_hex(16), secrets.token_hex(16)
    _require(owners._matches(intent_id, owners._CONSENT_ID) and owners._matches(generation, owners._CONSENT_ID)
             and intent_id != generation, "experiment_creation_invalid")
    record = dict(schema_version=CREATION_SCHEMA, intent_id=intent_id, generation=generation, issuer_uid=0,
        principal=principal, owner=owner, root=root, lane=_profile_lane(participant_profile), name="registered-" + intent_id,
        reference_kind="run_ref", reference_value=reference_value, reason=reason, class_intent=class_intent,
        cleanup=cleanup, lease_ttl_seconds=lease_ttl_seconds, issued_at_epoch=issued, expires_at_epoch=expiry,
        policy=_selector(policy_raw, files.budget), request_records=selectors,
        writer_scope=writer, participant_profile=participant_profile)
    record["intent_digest"] = canonical_digest(record, digest_field="intent_digest")
    payload = owners._encoded(record, files.budget, cap=_MAX_INTENT)
    claim_payload = None
    if arena:
        _require(len(payload) <= 2304, "experiment_arena_claim_invalid")
        claim = dict(schema_version="control_plane_lane_arena_issue_claim.v1",
            tag=_ARENA_TAG.fullmatch(reference_value).group(1), intent_id=intent_id, generation=generation,
            principal=principal, owner=owner, policy=record["policy"], issued_at_epoch=issued,
            expires_at_epoch=expiry, intent=_selector(payload, files.budget),
            intent_payload_base64=base64.b64encode(payload).decode("ascii"))
        claim["claim_digest"] = canonical_digest(claim, digest_field="claim_digest")
        claim_payload = owners._encoded(claim, files.budget, cap=4096)
    _require(occupied + len(payload) + (len(claim_payload) if claim_payload else 0)
             <= MAX_EXPERIMENT_STORE_BYTES, "experiment_store_full")
    files.verify_record(policy_record)
    files.verify()
    if claim_payload is not None:
        _publish(files, parent, claim_name, claim_payload, kind="arena_claim")
    published = _publish_owned_metadata(files, parent, intent_id + ".json", payload,
                                        mode=0o600, artifact_kind="attestation")
    files.verify()
    return {"intent_id": intent_id, "intent": {key: published[key] for key in ("sha256", "size_bytes")}}


def issue_experiment_creation_intent(*, installed_config_path="/etc/blueprint-operator-door/door.json",
        principal, owner, root, reference_value, lease_ttl_seconds, participant_profile,
        request_records, expires_at_epoch=None, now=time.time):
    """Root-only authentic immutable intent; no target or child is created here."""
    budget = ReferenceCollectionBudget(values_limit=10000)
    files = _BirthFiles(budget)
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



def recover_arena_issue(tag, *, principal, owner,
        installed_config_path="/etc/blueprint-operator-door/door.json", now=time.time):
    """Recover one durably consumed tag's exact original grant, never reissue."""
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        _require(os.geteuid() == 0, "experiment_issuer_required")
        _require(isinstance(tag, str) and _ARENA_TAG.fullmatch("arena-launch-" + tag),
                 "experiment_arena_tag_invalid")
        issued = now()
        config = _configuration(files, installed_config_path)
        _require(config.experiment_creation_enabled is True, "experiment_creation_disabled")
        parent = _store(files, config.experiment_record_store)
        occupied = _capacity(files, parent, adding_registration=False)
        raw, original = files.read(Path(config.experiment_record_store) / ("arena-launch-" + tag + ".arena-claim.json"),
                                   cap=4096, protected=True, mode=0o600)
        claim = retained._document(raw, 4096, _work_budget=files.budget)
        fields = {"schema_version", "tag", "intent_id", "generation", "principal", "owner", "policy",
                  "issued_at_epoch", "expires_at_epoch", "intent", "intent_payload_base64", "claim_digest"}
        _require(set(claim) == fields and claim["schema_version"] == "control_plane_lane_arena_issue_claim.v1"
                 and claim["tag"] == tag and claim["principal"] == principal and claim["owner"] == owner
                 and owners._matches(claim["intent_id"], owners._CONSENT_ID)
                 and owners._matches(claim["generation"], owners._CONSENT_ID)
                 and claim["intent_id"] != claim["generation"]
                 and _epoch(issued) and _epoch(claim["issued_at_epoch"]) and _epoch(claim["expires_at_epoch"])
                 and claim["issued_at_epoch"] <= issued < claim["expires_at_epoch"]
                 and claim["claim_digest"] == canonical_digest(claim, digest_field="claim_digest")
                 and isinstance(claim["intent_payload_base64"], str)
                 and 0 < len(claim["intent_payload_base64"]) <= 3072,
                 "experiment_arena_claim_invalid")
        files.budget.charge("output_bytes", 2304)
        try:
            payload = base64.b64decode(claim["intent_payload_base64"], validate=True)
        except ValueError:
            raise OwnerTargetVersionError("experiment_arena_claim_invalid") from None
        _require(0 < len(payload) <= 2304 and _selector(payload, files.budget) == claim["intent"],
                 "experiment_arena_claim_invalid")
        intent = retained._document(payload, 2304, _work_budget=files.budget)
        from .control_plane_lane_experiment_birth import _INTENT_FIELDS
        _require(set(intent) == _INTENT_FIELDS and intent["schema_version"] == CREATION_SCHEMA
                 and intent["intent_digest"] == canonical_digest(intent, digest_field="intent_digest")
                 and all(intent[key] == claim[key] for key in ("intent_id", "generation", "principal", "owner", "policy",
                                                              "issued_at_epoch", "expires_at_epoch"))
                 and intent["issuer_uid"] == 0 and type(intent["issuer_uid"]) is int
                 and intent["root"] == "inputs" and intent["lane"] == "arena"
                 and intent["name"] == "registered-" + claim["intent_id"]
                 and intent["participant_profile"] == "arena_owner_review.v1"
                 and intent["reference_kind"] == "run_ref" and intent["reference_value"] == "arena-launch-" + tag
                 and intent["request_records"] == []
                 and (intent["reason"], intent["class_intent"], intent["cleanup"], intent["writer_scope"])
                     == _PROFILES["arena_owner_review.v1"][:4]
                 and type(intent["lease_ttl_seconds"]) is int and 0 < intent["lease_ttl_seconds"] <= 1209600
                 and intent["expires_at_epoch"] <= intent["issued_at_epoch"] + intent["lease_ttl_seconds"],
                 "experiment_arena_claim_invalid")
        policy_raw, policy_record = files.read(config.lane_owner_policy_file, cap=owners.MAX_POLICY_BYTES,
                                              protected=True, mode=0o600)
        _require(_selector(policy_raw, files.budget) == claim["policy"], "experiment_policy_changed")
        policy = owners._policy(policy_raw, principal, files.budget)
        owners._authorize(dict(action="register", owner=owner, ttl_seconds=intent["lease_ttl_seconds"]),
                          policy, claim["expires_at_epoch"], claim["issued_at_epoch"])
        name = claim["intent_id"] + ".json"
        files.location(parent)
        try:
            os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            _require(occupied + len(payload) <= MAX_EXPERIMENT_STORE_BYTES, "experiment_store_full")
            files.verify_record(original)
            files.verify_record(policy_record)
            _publish(files, parent, name, payload, kind="private")
        else:
            existing, record = files.read(Path(config.experiment_record_store) / name, cap=2304, protected=True, mode=0o600)
            _require(existing == payload, "experiment_arena_claim_invalid")
            files.verify_record(record)
        files.verify()
        return {"intent_id": claim["intent_id"], "intent": claim["intent"]}
    except OSError:
        raise OwnerTargetVersionError("experiment_arena_claim_invalid") from None
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()


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


def issue_experiment_restore_intent(intent_id, *, principal, owner, lease_ttl_seconds, expires_at_epoch,
        installed_config_path="/etc/blueprint-operator-door/door.json", now=time.time):
    from .control_plane_lane_experiment_restore import issue_restore
    return issue_restore(intent_id, principal=principal, owner=owner, lease_ttl_seconds=lease_ttl_seconds,
        expires_at_epoch=expires_at_epoch, installed_config_path=installed_config_path, now=now)


def restore_registered_experiment(action_id, *, expected_restore_intent,
        installed_config_path="/etc/blueprint-operator-door/door.json", now=time.time, _pins_root=None):
    from .control_plane_lane_experiment_restore import restore
    return restore(action_id, expected_restore_intent=expected_restore_intent,
        installed_config_path=installed_config_path, now=now, pins_root=_pins_root)


def issue_experiment_producer_bootstrap(intent_id, *, expected_intent_sha256, expected_intent_size_bytes,
        request_paths, installed_config_path="/etc/blueprint-operator-door/door.json", now=time.time):
    from .control_plane_lane_experiment_bootstrap import issue_bootstrap
    return issue_bootstrap(intent_id, expected_intent_sha256=expected_intent_sha256,
        expected_intent_size_bytes=expected_intent_size_bytes, request_paths=request_paths,
        installed_config_path=installed_config_path, now=now)


INSTALLED_CONFIG_PATH = "/etc/blueprint-operator-door/door.json"


def prepare_registered_experiment_state(*, installed_config_path=INSTALLED_CONFIG_PATH):
    from .control_plane_lane_experiment_installation import prepare
    return prepare(installed_config_path=installed_config_path)


class _FixedCommandParser(argparse.ArgumentParser):
    def error(self, message):
        # Argument text can contain paths/private values. Emit only the finite
        # refusal, never argparse's copy of the rejected body.
        raise OwnerTargetVersionError("experiment_cli_arguments_invalid")


def _command_parser():
    parser = _FixedCommandParser(prog="registered-experiment", allow_abbrev=False)
    commands = parser.add_subparsers(dest="operation", required=True, parser_class=_FixedCommandParser)
    commands.add_parser("prepare", allow_abbrev=False)
    recover = commands.add_parser("recover-arena-issue", allow_abbrev=False)
    recover.add_argument("tag")
    recover.add_argument("--principal", required=True)
    recover.add_argument("--owner", required=True)
    for operation in ("issue-create", "create", "issue-action", "apply", "issue-restore", "restore", "bootstrap", "run"):
        command = commands.add_parser(operation, allow_abbrev=False)
        if operation != "issue-create":
            command.add_argument("intent_id")
        if operation in ("issue-create", "issue-action", "issue-restore"):
            command.add_argument("--principal", required=True)
            command.add_argument("--owner", required=True)
        if operation in ("issue-action", "issue-restore"):
            command.add_argument("--expires-at", required=True, type=float)
        if operation in ("issue-create", "issue-restore"):
            command.add_argument("--ttl", required=True, type=int)
        if operation in ("create", "apply", "restore", "bootstrap", "run"):
            command.add_argument("--sha256", required=True)
            command.add_argument("--size-bytes", required=True, type=int)
        if operation == "issue-create":
            command.add_argument("--root", required=True, choices=("work", "inputs"))
            command.add_argument("--reference", required=True)
            command.add_argument("--profile", required=True, choices=tuple(_PROFILES))
            command.add_argument("--request", nargs=3, action="append", default=[], metavar=("SEALED_PATH", "SHA256", "SIZE"))
        if operation == "issue-action":
            command.add_argument("--action", required=True, choices=("delete", "offload", "owner_review"))
        if operation == "bootstrap":
            command.add_argument("--request-path", action="append", required=True)
    return parser


def _installed_pin_root():
    from .control_plane_lane_experiment_actions import _installed_reference_selection
    files = _TargetFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        config = _configuration(files, INSTALLED_CONFIG_PATH)
        path, _ = _installed_reference_selection(files, config)
        files.verify()
        return Path(path)
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()


def _dispatch_fixed_command(arguments):
    operation = arguments.operation
    if operation == "prepare":
        return prepare_registered_experiment_state(installed_config_path=INSTALLED_CONFIG_PATH)
    fixed = {"installed_config_path": INSTALLED_CONFIG_PATH, "now": lambda: time.time()}
    if operation == "recover-arena-issue":
        return recover_arena_issue(arguments.tag, principal=arguments.principal, owner=arguments.owner, **fixed)
    if operation == "issue-create":
        requests = []
        for path, digest, size in arguments.request:
            try:
                count = int(size)
            except ValueError:
                raise OwnerTargetVersionError("experiment_cli_arguments_invalid") from None
            requests.append((path, {"sha256": digest, "size_bytes": count}))
        return issue_experiment_creation_intent(principal=arguments.principal, owner=arguments.owner,
            root=arguments.root, reference_value=arguments.reference, lease_ttl_seconds=arguments.ttl,
            participant_profile=arguments.profile, request_records=requests, **fixed)
    if operation == "issue-action":
        return issue_experiment_action_intent(arguments.intent_id, principal=arguments.principal,
            owner=arguments.owner, action=arguments.action, expires_at_epoch=arguments.expires_at, **fixed)
    if operation == "issue-restore":
        return issue_experiment_restore_intent(arguments.intent_id, principal=arguments.principal,
            owner=arguments.owner, lease_ttl_seconds=arguments.ttl, expires_at_epoch=arguments.expires_at, **fixed)
    selector = {"sha256": arguments.sha256, "size_bytes": arguments.size_bytes}
    _require(owners._matches(arguments.intent_id, owners._CONSENT_ID)
             and re.fullmatch(r"sha256:[0-9a-f]{64}", arguments.sha256) is not None
             and type(arguments.size_bytes) is int and 0 < arguments.size_bytes <= _MAX_INTENT,
             "experiment_cli_arguments_invalid")
    if operation == "create":
        from .control_plane_lane_experiment_birth import create_registered_experiment
        return create_registered_experiment(arguments.intent_id, expected_intent=selector, **fixed)
    if operation == "apply":
        return run_registered_experiment_action(arguments.intent_id, expected_action_intent=selector,
            _pins_root=_installed_pin_root(), **fixed)
    if operation == "restore":
        return restore_registered_experiment(arguments.intent_id, expected_restore_intent=selector,
            _pins_root=_installed_pin_root(), **fixed)
    if operation == "bootstrap":
        return issue_experiment_producer_bootstrap(arguments.intent_id, expected_intent_sha256=arguments.sha256,
            expected_intent_size_bytes=arguments.size_bytes, request_paths=tuple(arguments.request_path), **fixed)
    from .native_g1_registered_containment import run_registered_experiment
    return run_registered_experiment(arguments.intent_id, expected_intent=selector, **fixed)


def main(argv=None):
    """Fixed root local command; private config/reference authority have no flags."""
    try:
        _require(os.geteuid() == 0, "experiment_issuer_required")
        argv = sys.argv[1:] if argv is None else argv
        _require(isinstance(argv, (list, tuple)) and len(argv) <= 64
                 and all(isinstance(value, str) and len(value.encode()) <= 8192 for value in argv)
                 and sum(len(value.encode()) for value in argv) <= 32768,
                 "experiment_cli_arguments_invalid")
        arguments = _command_parser().parse_args(argv)
        result = _dispatch_fixed_command(arguments)
        outcome = {"decision": "completed", "result": result}
        status = 0
        if isinstance(result, dict) and result.get("decision") in ("kept", "partial", "refused"):
            outcome["decision"] = result["decision"]
            status = 2
        payload = json.dumps(outcome, sort_keys=True, separators=(",", ":")) + "\n"
        _require(len(payload.encode()) <= 8192, "experiment_cli_output_limit")
    except (OwnerTargetVersionError, OSError) as error:
        reason = str(error) if isinstance(error, OwnerTargetVersionError) else "experiment_cli_io_failed"
        if re.fullmatch(r"[a-z][a-z0-9_]{0,127}", reason) is None:
            reason = "experiment_cli_refused"
        payload = json.dumps({"decision": "refused", "reason": reason}, separators=(",", ":")) + "\n"
        status = 2
    sys.stdout.write(payload)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
