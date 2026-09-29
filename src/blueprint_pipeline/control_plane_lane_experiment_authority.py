"""Finite acquired public current authority; never chooses an older fallback."""
from __future__ import annotations

import os
import stat

from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_owner_target_versions import _counter, _epoch, _require, _valid_digest
from .decision_evidence_contracts import canonical_digest

ENTRY_FIELDS = frozenset({"intent_id", "generation", "birth", "target_identity", "lease", "owner", "root",
    "lane", "name", "state", "completion", "restoration", "operation_id", "expires_at_epoch"})
_HEAD_FIELDS = frozenset({"schema_version", "authority_epoch_id", "version", "record_name", "record", "head_digest"})
_AUTHORITY_FIELDS = frozenset({"schema_version", "authority_epoch_id", "version", "previous_record", "state",
    "issued_at_epoch", "expires_at_epoch", "policy", "enrollments", "authority_digest"})


def _raw_selector(value):
    return (isinstance(value, dict) and set(value) == {"sha256", "size_bytes"}
            and _valid_digest(value["sha256"]) and _counter(value["size_bytes"], positive=True)
            and value["size_bytes"] <= 32768)


def _read(files, parent, name, cap, gid):
    files.location(parent)
    fd = files.open(name, os.O_RDONLY | os.O_NONBLOCK, parent=parent)
    info = files.acquired[fd]
    _require(info.st_uid == 0 and info.st_gid == gid and stat.S_IMODE(info.st_mode) == 0o640
             and info.st_nlink == 1 and 0 < info.st_size <= cap, "experiment_authority_unsafe")
    record = owners._Acquired(fd, parent, name, info)
    raw = files.read_bytes(fd, cap)
    _require(len(raw) == info.st_size, "experiment_authority_changed")
    files.verify_record(record)
    files.records.append(record)
    return raw, record


def _current(files, parent, gid):
    files.location(parent)
    try:
        os.stat("HEAD.json", dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        files.slot()
        with os.scandir(parent) as entries:
            for entry in entries:
                files.budget.charge("entries")
                files.location(parent)
                _require(entry.name == ".authority.lock", "experiment_authority_missing")
        files.location(parent)
        return None
    raw, record = _read(files, parent, "HEAD.json", 4096, gid)
    head = retained._document(raw, 4096, _work_budget=files.budget)
    _require(set(head) == _HEAD_FIELDS and head["schema_version"] == "control_plane_lane_experiment_head.v1"
             and owners._matches(head["authority_epoch_id"], owners._CONSENT_ID)
             and _counter(head["version"], maximum=99999999) and _raw_selector(head["record"])
             and isinstance(head["record_name"], str)
             and head["record_name"] == f"authority-{head['version']:08d}-{head['authority_epoch_id']}.json"
             and head["head_digest"] == canonical_digest(head, digest_field="head_digest"),
             "experiment_authority_invalid")
    source, _ = _read(files, parent, head["record_name"], 32768, gid)
    _require(len(source) == head["record"]["size_bytes"]
             and retained._digest(source, _work_budget=files.budget) == head["record"]["sha256"],
             "experiment_authority_changed")
    value = retained._document(source, 32768, _work_budget=files.budget)
    _require(set(value) == _AUTHORITY_FIELDS and value["schema_version"] == "control_plane_lane_experiment_authority.v1"
             and value["authority_epoch_id"] == head["authority_epoch_id"] and value["version"] == head["version"]
             and value["authority_digest"] == canonical_digest(value, digest_field="authority_digest")
             and value["state"] in ("enabled", "disabled") and _epoch(value["issued_at_epoch"])
             and _epoch(value["expires_at_epoch"]) and value["issued_at_epoch"] < value["expires_at_epoch"]
             and _raw_selector(value["policy"])
             and (value["previous_record"] is None if head["version"] == 0 else _raw_selector(value["previous_record"]))
             and isinstance(value["enrollments"], list) and len(value["enrollments"]) <= 100,
             "experiment_authority_invalid")
    seen = set()
    for entry in value["enrollments"]:
        files.budget.charge("facts")
        _require(isinstance(entry, dict) and set(entry) == ENTRY_FIELDS
                 and owners._matches(entry["intent_id"], owners._CONSENT_ID) and entry["intent_id"] not in seen
                 and owners._matches(entry["generation"], owners._CONSENT_ID)
                 and entry["name"] == "registered-" + entry["intent_id"] and entry["lane"] in ("g1", "arena")
                 and (entry["lane"] != "arena" or entry["root"] == "inputs")
                 and entry["root"] in ("work", "inputs") and owners._matches(entry["owner"], owners._OWNER)
                 and entry["state"] in ("creating", "active", "retiring", "retired", "restoring", "revoked")
                 and _epoch(entry["expires_at_epoch"]) and _raw_selector(entry["birth"])
                 and _raw_selector(entry["lease"]) and all(entry[key] is None or _raw_selector(entry[key])
                                                          for key in ("completion", "restoration"))
                 and (entry["operation_id"] is None or owners._matches(entry["operation_id"], owners._CONSENT_ID)),
                 "experiment_authority_invalid")
        identity = entry["target_identity"]
        _require(isinstance(identity, dict) and set(identity) == {"dev", "ino", "type"}
                 and identity["type"] == "directory"
                 and all(_counter(identity[k], maximum=2**64 - 1) for k in ("dev", "ino")), "experiment_authority_invalid")
        seen.add(entry["intent_id"])
    files.verify_record(record)
    return head, value, record
