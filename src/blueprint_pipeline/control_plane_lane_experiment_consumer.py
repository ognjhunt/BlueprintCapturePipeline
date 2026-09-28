"""Current authenticated experiment target lifetime for finite enrolled consumers.

Admission decodes under one native bounded budget. Retained proof checks allocate
no new records; an authority change refuses this admission instead of adopting it.
"""
from __future__ import annotations

import fcntl
import grp
import os
import re
import stat
import time
from contextlib import contextmanager
from pathlib import Path

from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch as scratch
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_authority import _current, _read
from .control_plane_lane_experiment_publication import _BirthFiles
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _epoch, _require
from .control_plane_reference_budget import ReferenceCollectionBudget
from .control_plane_scratch_lifetime import LANE_ROOTS, LeasedScratchUse
from .decision_evidence_contracts import canonical_digest

AUTHORITY_ROOT = Path("/var/lib/blueprint-operator-door/experiment-authority")


def _blueprint_gid():
    try:
        return grp.getgrnam("blueprint").gr_gid
    except KeyError:
        raise OwnerTargetVersionError("experiment_account_missing") from None


def registered_target(output, roots):
    """Lexical reservation is checked before any payload or marker access."""
    if not isinstance(output, Path) or not output.is_absolute():
        return None
    for root in roots:
        if output.is_relative_to(root):
            parts = output.relative_to(root).parts
            if len(parts) >= 2 and parts[0] == "g1" and re.fullmatch(r"registered-[0-9a-f]{32}", parts[1]):
                return root / "g1" / parts[1]
    return None


def require_registered_use(output, use, roots):
    target = registered_target(output, roots)
    if target is not None:
        _require(type(use) is RegisteredExperimentUse and use.path == target and not use._closed,
                 "experiment_consumer_authority_required")
        use.check()
    elif use is not None:
        raise OwnerTargetVersionError("experiment_consumer_authority_required")
    return target


class RegisteredExperimentUse(LeasedScratchUse):
    """Borrowed scopes retain the same target SH; they never close the parent."""
    @classmethod
    def admit(cls, target, *, expected_birth=None, expected_generation=None, now=time.time,
              _producer_request_paths=None):
        _require(isinstance(target, Path) and registered_target(target, LANE_ROOTS) == target,
                 "experiment_consumer_authority_required")
        files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
        result = None
        accepted = False
        try:
            gid = _blueprint_gid()
            public, _ = files.parent(AUTHORITY_ROOT / "HEAD.json")
            info = os.fstat(public)
            _require(info.st_uid == 0 and info.st_gid == gid and stat.S_IMODE(info.st_mode) == 0o750,
                     "experiment_authority_unsafe")
            lock = files.open(".authority.lock", os.O_RDONLY | os.O_NONBLOCK, parent=public)
            lock_info = files.acquired[lock]
            _require(lock_info.st_uid == 0 and lock_info.st_gid == gid and stat.S_IMODE(lock_info.st_mode) == 0o640
                     and lock_info.st_nlink == 1 and lock_info.st_size == 0, "experiment_authority_unsafe")
            files.location(public)
            files.proof(lock)
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            current = _current(files, public, gid)
            _require(current is not None, "experiment_authority_missing")
            head, authority, head_record = current
            issued = now()
            _require(_epoch(issued) and authority["state"] == "enabled"
                     and authority["issued_at_epoch"] <= issued < authority["expires_at_epoch"],
                     "experiment_consumer_inactive")
            root = next(root for root in LANE_ROOTS if target.is_relative_to(root))
            root_selector = "work" if root == LANE_ROOTS[0] else "inputs"
            rows = [entry for entry in authority["enrollments"] if entry["root"] == root_selector
                    and entry["name"] == target.name and entry["lane"] == "g1"]
            _require(len(rows) == 1, "experiment_consumer_authority_required")
            entry = rows[0]
            _require(entry["state"] == "active" and issued < entry["expires_at_epoch"]
                     and (expected_birth is None or expected_birth == entry["birth"])
                     and (expected_generation is None or expected_generation == entry["generation"]),
                     "experiment_consumer_inactive")
            birth_raw, birth_record = _read(files, public, entry["intent_id"] + ".birth.json", 32768, gid)
            _require(len(birth_raw) == entry["birth"]["size_bytes"]
                     and retained._digest(birth_raw, _work_budget=files.budget) == entry["birth"]["sha256"],
                     "experiment_birth_changed")
            birth = retained._document(birth_raw, 32768, _work_budget=files.budget)
            _require(birth["schema_version"] == "control_plane_lane_experiment_birth.v1"
                     and birth["birth_digest"] == canonical_digest(birth, digest_field="birth_digest")
                     and all(birth[key] == entry[key] for key in ("intent_id", "generation", "root", "lane", "name",
                                                                "target_identity", "owner")), "experiment_birth_changed")
            files.proof(lock)
            fcntl.flock(lock, fcntl.LOCK_UN)
            parent, _ = files.parent(target / scratch.LEASE_FILE)
            files.location(parent)
            original = os.fstat(parent)
            _require((original.st_dev, original.st_ino) == (entry["target_identity"]["dev"], entry["target_identity"]["ino"]),
                     "experiment_target_changed")
            files.proof(parent)
            fcntl.flock(parent, fcntl.LOCK_SH | fcntl.LOCK_NB)
            lease_raw, lease_record = files.read(target / scratch.LEASE_FILE, cap=scratch.MAX_LEASE_BYTES)
            _require(len(lease_raw) == entry["lease"]["size_bytes"]
                     and retained._digest(lease_raw, _work_budget=files.budget) == entry["lease"]["sha256"],
                     "experiment_lease_changed")
            lease = retained._document(lease_raw, scratch.MAX_LEASE_BYTES, _work_budget=files.budget)
            _require(lease["schema_version"] == scratch.SCHEMA_VERSION and scratch._lease_fields_valid(lease)
                     and lease["lease_digest"] == canonical_digest(lease, digest_field="lease_digest")
                     and all(lease[key] == entry[key] for key in ("lane", "name", "owner"))
                     and lease.get("consumer_lifetime_contract") == scratch.CONSUMER_LIFETIME_PROTOCOL
                     and lease["released_at_epoch"] is None and _epoch(lease["expires_at_epoch"])
                     and issued < lease["expires_at_epoch"], "experiment_consumer_inactive")
            result = cls()
            result.files, result.fd, result._authority_lock = files, parent, lock
            result.root, result.lane, result.name = root, "g1", target.name
            result.exclusive, result.now = False, now
            result.entry, result.birth, result._head_record, result._lease_record = entry, birth, head_record, lease_record
            result._birth_record, result._projection_expiry = birth_record, authority["expires_at_epoch"]
            result._started, result._checks = time.monotonic(), 0
            result.identity = dict(root=str(root), lane="g1", name=target.name, owner=lease["owner"],
                run_ref=lease["run_ref"], lease_digest=lease["lease_digest"], consumer_lifetime_contract=scratch.CONSUMER_LIFETIME_PROTOCOL)
            result._producer_requests = []
            if _producer_request_paths is not None:
                _require(os.geteuid() == 0 and isinstance(_producer_request_paths, (list, tuple))
                         and len(_producer_request_paths) == 2 and birth["class_intent"] == "evidence"
                         and birth["cleanup"] == "owner_review"
                         and birth["writer_scope"] == "native_g1_development_pair.v1"
                         and birth["participant_profile"] in ("g1_local_prelaunch_block.v1", "g1_local_contained_completed.v1"),
                         "experiment_producer_authority_required")
                marker_raw, _ = files.read(target / ".registered-experiment.v1.json", cap=4096)
                _require(len(marker_raw) == birth["marker"]["size_bytes"]
                         and retained._digest(marker_raw, _work_budget=files.budget) == birth["marker"]["sha256"],
                         "experiment_birth_changed")
                marker = retained._document(marker_raw, 4096, _work_budget=files.budget)
                _require(marker["marker_digest"] == canonical_digest(marker, digest_field="marker_digest")
                         and marker["generation"] == entry["generation"], "experiment_birth_changed")
                private = AUTHORITY_ROOT.parent / "requests/experiment-records" / (entry["intent_id"] + ".json")
                intent_raw, _ = files.read(private, cap=32768, protected=True, mode=0o600)
                _require(len(intent_raw) == marker["intent"]["size_bytes"]
                         and retained._digest(intent_raw, _work_budget=files.budget) == marker["intent"]["sha256"],
                         "experiment_producer_authority_required")
                intent = retained._document(intent_raw, 32768, _work_budget=files.budget)
                _require(intent["intent_digest"] == canonical_digest(intent, digest_field="intent_digest")
                         and intent["generation"] == entry["generation"] and intent["intent_id"] == entry["intent_id"]
                         and isinstance(intent["request_records"], list) and len(intent["request_records"]) == 2,
                         "experiment_producer_authority_required")
                for path, selector in zip(_producer_request_paths, intent["request_records"], strict=True):
                    _require(isinstance(path, Path) and path.is_absolute(), "experiment_producer_request_changed")
                    request_raw, request_record = files.read(path, cap=65536, protected=True)
                    owners._identity(request_raw, selector["sha256"], selector["size_bytes"], files.budget)
                    request = retained._document(request_raw, 65536, _work_budget=files.budget)
                    _require(request["request_digest"] == canonical_digest(request, digest_field="request_digest"),
                             "experiment_producer_request_changed")
                    result._producer_requests.append((path, request_record, request["candidate_id"], request["request_digest"]))
            files.budget.close()
            result.check()
            accepted = True
            return result
        except (OSError, StopIteration, KeyError, TypeError):
            raise OwnerTargetVersionError("experiment_consumer_admission_failed") from None
        finally:
            if not accepted:
                if result is not None:
                    result._closed = True
                try:
                    files.finish()
                finally:
                    files.budget.close()

    def authorize_g1_pair(self, paths):
        _require(isinstance(paths, (list, tuple)) and len(paths) == len(self._producer_requests) == 2,
                 "experiment_producer_authority_required")
        self.check()
        for path, supplied in zip(paths, self._producer_requests, strict=True):
            expected, record, _, _ = supplied
            _require(path == expected, "experiment_producer_request_changed")
            self.files.location(record.parent, cleanup=True)
            self.files.proof(record.fd)
            _require(owners._metadata(os.fstat(record.fd)) == owners._metadata(record.info)
                     == owners._metadata(os.stat(record.name, dir_fd=record.parent, follow_symlinks=False)),
                     "experiment_producer_request_changed")

    def authorize_worker(self, request, output):
        _require(any(output.name == candidate and request.get("request_digest") == digest
                     and canonical_digest(request, digest_field="request_digest") == digest
                     for _, _, candidate, digest in self._producer_requests), "experiment_producer_request_changed")
        self.check()

    def check(self):
        _require(not self._closed and self._checks < 20000 and time.monotonic() - self._started <= 4 * 3600,
                 "experiment_consumer_resource_exhausted")
        self._checks += 1
        issued = self.now()
        _require(_epoch(issued) and issued < min(self.entry["expires_at_epoch"], self._projection_expiry),
                 "experiment_consumer_inactive")
        files = self.files
        try:
            files.location(self.fd, cleanup=True)
            files.proof(self._authority_lock)
            fcntl.flock(self._authority_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            try:
                for record in (self._head_record, self._lease_record, self._birth_record):
                    files.location(record.parent, cleanup=True)
                    files.proof(record.fd)
                    _require(owners._metadata(os.fstat(record.fd)) == owners._metadata(record.info)
                             == owners._metadata(os.stat(record.name, dir_fd=record.parent, follow_symlinks=False)),
                             "experiment_consumer_version_changed")
            finally:
                files.proof(self._authority_lock)
                fcntl.flock(self._authority_lock, fcntl.LOCK_UN)
        except OSError:
            raise OwnerTargetVersionError("experiment_consumer_check_failed") from None

    @contextmanager
    def borrow(self, output):
        _require(isinstance(output, Path) and output.parent == self.path, "experiment_worker_descendant_invalid")
        self.check()
        try:
            yield self
        finally:
            self.check()

    def refresh(self):
        raise OwnerTargetVersionError("experiment_explicit_new_admission_required")

    def mkdir(self, relative):
        raise OwnerTargetVersionError("experiment_consumer_operation_unsupported")

    def close(self):
        self._closed = True
        self.files.finish()

    def _take(self, fd):
        raise OwnerTargetVersionError("experiment_descriptor_ownership_unproven")

    def _take_all(self, descriptors):
        raise OwnerTargetVersionError("experiment_descriptor_ownership_unproven")

    def _open(self, *args, **kwargs):
        raise OwnerTargetVersionError("experiment_consumer_operation_unsupported")

    def _dup(self, original):
        raise OwnerTargetVersionError("experiment_consumer_operation_unsupported")
